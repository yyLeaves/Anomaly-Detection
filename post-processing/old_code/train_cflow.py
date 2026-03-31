import argparse
import os
import json
import shutil
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from anomalib.data import Folder
from anomalib.models.image.cflow import Cflow
from fastflow_dataset import prepare_dataset_root
from radimagenet_utils import load_radimagenet_resnet_weights


DEFAULT_DATA_ROOT = "/local/scratch/koepchen/synth23_pelvis_v6_png"
DEFAULT_LOG_DIR = "/home/user/koepchen/OOD_tests"


def _build_model(backbone: str, radimagenet_ckpt: str | None) -> Cflow:
    name = backbone.lower()
    suffix = name.replace("radimagenet", "").strip("-_ ") or "resnet50"
    mapping = {
        "resnet50": "resnet50",
        "50": "resnet50",
        "resnet18": "resnet18",
        "18": "resnet18",
        "wideresnet50": "wide_resnet50_2",
        "wideresnet50_2": "wide_resnet50_2",
        "wide_resnet50_2": "wide_resnet50_2",
        "wrn50": "wide_resnet50_2",
    }
    target = mapping.get(suffix, suffix)

    model = Cflow(backbone=target, pre_trained=False)

    if target.startswith("resnet") and radimagenet_ckpt:
        feature_extractor = model.model.encoder.feature_extractor
        load_radimagenet_resnet_weights(feature_extractor, radimagenet_ckpt, strict=False)

    return model


def _resolve_eval_dirs(root: Path) -> tuple[str, str, str | None]:
    valid_good = root / "valid" / "good" / "img"
    valid_bad = root / "valid" / "Ungood" / "img"
    valid_mask = root / "valid" / "Ungood" / "label"
    if valid_good.exists() and valid_bad.exists():
        mask_dir = "valid/Ungood/label" if valid_mask.exists() else None
        return "valid/good/img", "valid/Ungood/img", mask_dir

    test_good = root / "test" / "good" / "img"
    test_bad = root / "test" / "Ungood" / "img"
    test_mask = root / "test" / "Ungood" / "label"
    if test_good.exists() and test_bad.exists():
        mask_dir = "test/Ungood/label" if test_mask.exists() else None
        return "test/good/img", "test/Ungood/img", mask_dir

    return "test/good", "test/Ungood", None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CFlow training script for PNG or NIfTI datasets.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--backbone", type=str, default="radimagenet_resnet50")
    parser.add_argument("--radimagenet_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use; -1 forces CPU")
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices={"auto", "png", "nifti"},
        default="auto",
        help="Input dataset format. Use 'nifti' to force conversion, 'png' to skip detection.",
    )
    parser.add_argument(
        "--conversion-cache-dir",
        type=str,
        default=None,
        help="Optional directory where converted PNG datasets will be stored when using NIfTI inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    source_root = Path(args.data_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {source_root}")

    cache_dir = Path(args.conversion_cache_dir).resolve() if args.conversion_cache_dir else None
    prepared_root, converted = prepare_dataset_root(
        source_root, format_hint=args.dataset_format, cache_root=cache_dir
    )
    if converted:
        print(f"[INFO] Converted NIfTI dataset at {source_root} -> PNG cache at {prepared_root}")

    os.environ.setdefault("ANOMALIB_LOG_DIR", args.log_dir)

    normal_test_dir, abnormal_dir, mask_dir = _resolve_eval_dirs(prepared_root)

    datamodule = Folder(
        name=f"{source_root.name}_cflow",
        root=str(prepared_root),
        normal_dir="train/good",
        normal_test_dir=normal_test_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=8,
        extensions=(".png",),
    )

    log_root = Path(args.log_dir)
    project_root = Path(__file__).resolve().parent
    dataset_tag = f"{source_root.name}_cflow"

    lightning_log_dir = log_root / "cflow_logs"
    project_log_dir = project_root / "cflow_logs"
    for destination in (lightning_log_dir, project_log_dir):
        destination.mkdir(parents=True, exist_ok=True)

    weight_dirs = [
        log_root / "cflow" / dataset_tag / "weights",
        project_root / "cflow" / dataset_tag / "weights",
    ]
    for weights_path in weight_dirs:
        weights_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(weight_dirs[0]),
        filename="best",
        monitor="train_loss_step",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    model = _build_model(args.backbone, args.radimagenet_ckpt)

    if args.gpu >= 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = [args.gpu]
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=CSVLogger(save_dir=args.log_dir, name="cflow_logs"),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=datamodule)

    if trainer.is_global_zero:
        log_dir = None
        logger = getattr(trainer, "logger", None)
        if logger is not None and getattr(logger, "log_dir", None):
            log_dir = Path(logger.log_dir)
        if log_dir is None:
            log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            relative_log_dir = log_dir.relative_to(lightning_log_dir)
        except ValueError:
            relative_log_dir = Path(log_dir.name)

        project_run_dir = project_log_dir / relative_log_dir
        project_run_dir.mkdir(parents=True, exist_ok=True)

        best_checkpoint = checkpoint_callback.best_model_path
        best_checkpoint_path = Path(best_checkpoint) if best_checkpoint else None

        last_checkpoint = checkpoint_callback.last_model_path
        last_checkpoint_path = Path(last_checkpoint) if last_checkpoint else weight_dirs[0] / "last.ckpt"
        if not last_checkpoint_path.exists():
            trainer.save_checkpoint(str(last_checkpoint_path))

        project_best_checkpoint_path: Path | None = None
        if best_checkpoint_path and best_checkpoint_path.exists():
            project_best_checkpoint_path = weight_dirs[1] / best_checkpoint_path.name
            try:
                shutil.copy2(best_checkpoint_path, project_best_checkpoint_path)
            except OSError:
                project_best_checkpoint_path = None

        project_last_checkpoint_path: Path | None = None
        if last_checkpoint_path.exists():
            project_last_checkpoint_path = weight_dirs[1] / last_checkpoint_path.name
            try:
                shutil.copy2(last_checkpoint_path, project_last_checkpoint_path)
            except OSError:
                project_last_checkpoint_path = None

        metadata_path = log_dir / "training_run_metadata.json"
        epochs_completed = trainer.current_epoch + 1 if trainer.current_epoch is not None else args.epochs

        metadata = {
            "data_root": str(source_root),
            "prepared_root": str(prepared_root),
            "dataset_format": args.dataset_format,
            "conversion_cache_dir": str(cache_dir) if cache_dir else None,
            "backbone": args.backbone,
            "radimagenet_ckpt": args.radimagenet_ckpt,
            "epochs_target": args.epochs,
            "epochs_completed": epochs_completed,
            "checkpoint_dir": str(weight_dirs[0].resolve()),
            "project_checkpoint_dir": str(weight_dirs[1].resolve()),
            "last_checkpoint": str(last_checkpoint_path.resolve()) if last_checkpoint_path.exists() else None,
            "project_last_checkpoint": str(project_last_checkpoint_path.resolve())
            if project_last_checkpoint_path and project_last_checkpoint_path.exists()
            else None,
            "best_checkpoint": str(best_checkpoint_path.resolve())
            if best_checkpoint_path and best_checkpoint_path.exists()
            else None,
            "project_best_checkpoint": str(project_best_checkpoint_path.resolve())
            if project_best_checkpoint_path and project_best_checkpoint_path.exists()
            else None,
        }
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        project_metadata_path = project_run_dir / metadata_path.name
        try:
            shutil.copy2(metadata_path, project_metadata_path)
        except OSError:
            pass



if __name__ == "__main__":
    main()
