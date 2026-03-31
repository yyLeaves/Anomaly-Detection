"""Unified training entry point.

Supports all models in the registry (rd4ad, stfpm, fastflow, cflow).

Usage
-----
python train.py --config config/rd4ad.yaml \\
                --data_root /path/to/dataset \\
                --name my_experiment \\
                [--output_root ./results/rd4ad] \\
                [--epochs 100] \\
                [--batch_size 16] \\
                [--learning_rate 0.001] \\
                [--accelerator gpu]

CLI flags override the YAML values when provided.

Training paradigms
------------------
- rd4ad / stfpm  : anomalib Engine, monitors image_AUROC (uses AUROC evaluator)
- fastflow / cflow: raw Lightning Trainer, monitors train_loss_step
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from data.dataset_conversion import prepare_dataset_root
from models.kd_models import KD_MODEL_REGISTRY
from models.flow_models import FLOW_MODEL_REGISTRY
from models.memory_models import MEMORY_MODEL_REGISTRY
from models.recon_models import RECON_MODEL_REGISTRY

STANDALONE_MODEL_REGISTRY: dict[str, dict] = {
    "deepsvdd": {"trainer_type": "standalone"},
    "cutpaste": {"trainer_type": "standalone"},
}

ALL_MODELS: dict[str, dict] = {
    **KD_MODEL_REGISTRY,
    **FLOW_MODEL_REGISTRY,
    **MEMORY_MODEL_REGISTRY,
    **RECON_MODEL_REGISTRY,
    **STANDALONE_MODEL_REGISTRY,
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an anomaly detection model")
    parser.add_argument("--config",        type=Path, required=True,
                        help="Path to model YAML config (e.g. config/rd4ad.yaml)")
    parser.add_argument("--data_root",     type=str,   default=None)
    parser.add_argument("--name",          type=str,   default=None)
    parser.add_argument("--output_root",   type=str,   default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_workers",   type=int,   default=None)
    parser.add_argument("--accelerator",      type=str,   default=None,
                        choices=["gpu", "cpu", "mps"])
    parser.add_argument("--backbone",         type=str,   default=None,
                        help="Override model backbone (e.g. radimagenet_resnet50)")
    parser.add_argument("--radimagenet_ckpt", type=str,   default=None,
                        help="Path to RadImageNet .pt checkpoint")
    return parser.parse_args()


def _merge_config(cfg, args: argparse.Namespace):
    overrides = {
        "train.data_root":     args.data_root,
        "train.name":          args.name,
        "train.output_root":   args.output_root,
        "train.epochs":        args.epochs,
        "train.batch_size":    args.batch_size,
        "train.num_workers":   args.num_workers,
        "train.accelerator":      args.accelerator,
        "model.learning_rate":    args.learning_rate,
        "model.backbone":         args.backbone,
        "model.radimagenet_ckpt": args.radimagenet_ckpt,
    }
    for key, val in overrides.items():
        if val is not None:
            OmegaConf.update(cfg, key, val, merge=True)
    return cfg


# ── Trainer device helpers ─────────────────────────────────────────────────────

def _trainer_accelerator_devices(tcfg):
    """Return (accelerator, devices, strategy) for a raw Lightning Trainer from train config."""
    import torch
    accel = tcfg.accelerator
    if accel in ("cpu", "mps"):
        return accel, 1, "auto"
    # accelerator == "gpu"
    if not torch.cuda.is_available():
        return "cpu", 1, "auto"
    gpus_cfg = OmegaConf.select(tcfg, "gpus")
    if gpus_cfg is not None:
        devices = list(gpus_cfg)
        strategy = "ddp" if len(devices) > 1 else "auto"
        return "gpu", devices, strategy
    gpu = int(OmegaConf.select(tcfg, "gpu", default=0))
    return "gpu", [gpu], "auto"


# ── Training strategies ────────────────────────────────────────────────────────

def _train_with_engine(cfg, datamodule, output_dir: str, model_name: str, entry: dict) -> None:
    """Train RD4AD / STFPM via anomalib Engine with AUROC evaluation."""
    ModelClass = entry["class"]

    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"],  prefix="pixel_")
    evaluator   = Evaluator(
        val_metrics=[image_auroc],
        test_metrics=[image_auroc, pixel_auroc],
    )

    model_kwargs: dict = {
        "evaluator": evaluator,
        **entry["init_kwargs"],
    }
    if OmegaConf.select(cfg.model, "backbone") is not None:
        model_kwargs["backbone"] = cfg.model.backbone
    if OmegaConf.select(cfg.model, "encoder_name") is not None:
        model_kwargs["encoder_name"] = cfg.model.encoder_name
    if OmegaConf.select(cfg.model, "dtd_dir") is not None:
        model_kwargs["dtd_dir"] = cfg.model.dtd_dir
    if OmegaConf.select(cfg.model, "pre_processor") is not None:
        model_kwargs["pre_processor"] = cfg.model.pre_processor
    if OmegaConf.select(cfg.model, "post_processor") is not None:
        model_kwargs["post_processor"] = cfg.model.post_processor
    if OmegaConf.select(cfg.model, "layers") is not None:
        model_kwargs["layers"] = list(cfg.model.layers)
    if OmegaConf.select(cfg.model, "learning_rate") is not None:
        model_kwargs["lr"] = cfg.model.learning_rate
    model = ModelClass(**model_kwargs)
    ModelClass.__name__ = model_name  # prevent Engine from creating class-named subdirectory

    csv_logger = CSVLogger(save_dir=output_dir, name="logs")
    csv_logger.log_hyperparams({
        "model":         model_name,
        "data_root":     cfg.train.data_root,
        "batch_size":    cfg.train.batch_size,
        "epochs":        cfg.train.epochs,
        "learning_rate": OmegaConf.select(cfg.model, "learning_rate", default=None),
    })

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename=f"{model_name}-best-{{epoch:03d}}-{{{entry['checkpoint_monitor']}:.4f}}",
        monitor=entry["checkpoint_monitor"],
        mode=entry["checkpoint_mode"],
        save_top_k=1,
        save_last=True,
    )

    callbacks = [checkpoint_callback]
    es_cfg = OmegaConf.select(cfg.train, "early_stopping")
    if es_cfg is not None:
        callbacks.append(EarlyStopping(
            monitor=es_cfg.monitor,
            patience=int(es_cfg.patience),
            mode=es_cfg.mode,
        ))

    with tempfile.TemporaryDirectory() as _engine_tmp:
        engine = Engine(
            max_epochs=cfg.train.epochs,
            min_epochs=1,
            accelerator=cfg.train.accelerator,
            default_root_dir=_engine_tmp,
            logger=csv_logger,
            callbacks=callbacks,
        )
        engine.fit(model=model, datamodule=datamodule)
        test_metrics = engine.test(model=model, datamodule=datamodule)
    print("Test metrics:", test_metrics)



def _train_with_trainer(cfg, datamodule, output_dir: str, model_name: str, entry: dict) -> None:
    """Train FastFlow / CFlow via raw Lightning Trainer."""
    backbone       = cfg.model.backbone
    radimagenet_ckpt = OmegaConf.select(cfg.model, "radimagenet_ckpt")

    model = entry["builder"](backbone, radimagenet_ckpt)

    csv_logger = CSVLogger(save_dir=output_dir, name="logs")
    csv_logger.log_hyperparams({
        "model":      model_name,
        "data_root":  cfg.train.data_root,
        "batch_size": cfg.train.batch_size,
        "epochs":     cfg.train.epochs,
        "backbone":   backbone,
    })

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename=f"{model_name}-best-{{epoch:03d}}",
        monitor=entry["checkpoint_monitor"],
        mode=entry["checkpoint_mode"],
        save_top_k=1,
        save_last=True,
    )

    accel, devices, strategy = _trainer_accelerator_devices(cfg.train)
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accel,
        devices=devices,
        strategy=strategy,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=datamodule)


# ── Standalone training (BMAD folder scripts) ──────────────────────────────────

def _train_standalone(cfg, data_root: str, output_dir: str, model_name: str) -> None:
    """Delegate to the BMAD Deep-SVDD / pytorch-cutpaste scripts via subprocess."""
    here = Path(__file__).resolve().parent
    accel = OmegaConf.select(cfg.train, "accelerator", default="gpu")
    gpu   = int(OmegaConf.select(cfg.train, "gpu", default=0))
    import torch
    device = f"cuda:{gpu}" if accel == "gpu" and torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    if model_name == "deepsvdd":
        svdd_config = str(here / "config" / "custom_DeepSVDD.yaml")
        cmd = [
            sys.executable,
            str(here / "Deep-SVDD" / "main.py"),
            "custom",                 # dataset_name
            "cifar10_LeNet",          # net_name (ignored; uses ResNet18 encoder in data)
            output_dir,               # xp_path
            data_root,                # data_path
            "--config_path", svdd_config,
            "--device", device,
            "--pretrain", "False",
        ]
    else:  # cutpaste
        cp_config = str(here / "config" / "custom_cutpaste.yaml")
        cmd = [
            sys.executable,
            str(here / "pytorch-cutpaste" / "run_training.py"),
            "--type", "custom",
            "--data_root", data_root,
            "--config", cp_config,
        ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(here), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{model_name} training subprocess exited with code {result.returncode}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    cfg  = _merge_config(cfg, args)

    if not cfg.train.data_root:
        raise ValueError("data_root must be set via YAML or --data_root")
    if not cfg.train.name:
        raise ValueError("name must be set via YAML or --name")

    model_name = cfg.model.name.lower()
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(ALL_MODELS)}")
    entry = ALL_MODELS[model_name]

    prepared_root, _ = prepare_dataset_root(
        Path(cfg.train.data_root), format_hint="auto"
    )
    data_root = str(prepared_root)

    output_dir = os.path.join(cfg.train.output_root, cfg.train.name)
    os.makedirs(output_dir, exist_ok=True)

    if entry["trainer_type"] == "standalone":
        _train_standalone(cfg, data_root, output_dir, model_name)

    elif entry["trainer_type"] == "engine":
        # KD-OOD models: val split mirrors test, mask labels available
        datamodule = Folder(
            name=cfg.train.name,
            root=data_root,
            normal_dir="train/good",
            normal_test_dir="valid/good/img",
            abnormal_dir="valid/Ungood/img",
            mask_dir="valid/Ungood/label",
            train_batch_size=cfg.train.batch_size,
            eval_batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            test_split_mode="from_dir",
            test_split_ratio=0.0,
            val_split_mode="same_as_test",
            val_split_ratio=1.0,
            seed=cfg.train.seed,
        )
        _train_with_engine(cfg, datamodule, output_dir, model_name, entry)

    else:
        # Flow models: same datamodule layout as engine models (mask_dir needed for pixel_AUROC)
        datamodule = Folder(
            name=cfg.train.name,
            root=data_root,
            normal_dir="train/good",
            normal_test_dir="valid/good/img",
            abnormal_dir="valid/Ungood/img",
            mask_dir="valid/Ungood/label",
            train_batch_size=cfg.train.batch_size,
            eval_batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            seed=cfg.train.seed,
        )
        _train_with_trainer(cfg, datamodule, output_dir, model_name, entry)


if __name__ == "__main__":
    main()
