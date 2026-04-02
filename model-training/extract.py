"""Unified extraction entry point — anomaly maps and prediction masks.

Supports all models in the registry (rd4ad, stfpm, fastflow, cflow).

Usage
-----
python extract.py --config config/rd4ad.yaml \\
                  --data_root /path/to/dataset \\
                  --checkpoint /path/to/checkpoint.ckpt \\
                  --output_dir ./outputs/rd4ad_test \\
                  [--split test] \\
                  [--batch_size 16] \\
                  [--gpu 0] \\
                  [--map_size 224] \\
                  [--mask_threshold 0.5] \\
                  [--mask_output_format png] \\
                  [--normal_train_dir train/good] \\
                  [--num_workers 8] \\
                  [--extensions .png]

CLI flags override the YAML extract section when provided.
Anomaly maps are saved as .npy; masks as .png or .nii.gz.
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode, ValSplitMode

from models.kd_models import KD_MODEL_REGISTRY
from models.flow_models import FLOW_MODEL_REGISTRY, load_flow_checkpoint
from models.memory_models import MEMORY_MODEL_REGISTRY
from models.recon_models import RECON_MODEL_REGISTRY

ALL_MODELS: dict[str, dict] = {
    **KD_MODEL_REGISTRY,
    **FLOW_MODEL_REGISTRY,
    **MEMORY_MODEL_REGISTRY,
    **RECON_MODEL_REGISTRY,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_list(obj: Sequence | str | None) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out: list[str] = []
        for item in obj:
            out.extend(_coerce_list(item))
        return out
    return [str(obj)]


def _extract(outputs, keys: Iterable[str]):
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        for k in keys:
            if k in outputs and outputs[k] is not None:
                return outputs[k]
    for k in keys:
        if hasattr(outputs, k):
            val = getattr(outputs, k)
            if val is not None:
                return val
    if isinstance(outputs, (list, tuple)):
        collected = [v for item in outputs if (v := _extract(item, keys)) is not None]
        return collected if collected else None
    return None


def _resolve_image_dir(root: Path, *parts: str) -> str | None:
    candidate = root.joinpath(*parts)
    img_dir = candidate / "img"
    if img_dir.exists():
        return Path(*parts, "img").as_posix()
    if candidate.exists():
        return Path(*parts).as_posix()
    return None


def _resolve_split_dirs(root: Path, split: str) -> tuple[list[str], str, str | None]:
    """Return (normal_test_dirs, abnormal_dir, mask_dir) for a given split."""
    if not (root / split).exists():
        raise FileNotFoundError(f"Split '{split}' not found under {root}")

    normal_dirs = [
        d for folder in ("good", "Ungood_whole_patient_scans")
        if (d := _resolve_image_dir(root, split, folder)) is not None
    ]
    if not normal_dirs:
        raise FileNotFoundError(
            f"No normal test directories found for split '{split}' under {root}"
        )

    abnormal_dir = (
        _resolve_image_dir(root, split, "Ungood")
        or Path(split, "Ungood").as_posix()
    )

    mask_candidate = root / split / "Ungood" / "label"
    mask_dir = mask_candidate.as_posix() if mask_candidate.exists() else None

    return normal_dirs, abnormal_dir, mask_dir


def _derive_output_rel(rel_image_path: Path, split: str) -> Path:
    parts = list(rel_image_path.parts)
    label_idx = next(
        (i for i, p in enumerate(parts) if "good" in p.lower() or "ungood" in p.lower()),
        None,
    )
    if label_idx is None:
        return Path(split) / rel_image_path
    return Path(split, parts[label_idx], *parts[label_idx + 1:])


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract anomaly maps and prediction masks")
    parser.add_argument("--config",             type=Path, required=True)
    parser.add_argument("--data_root",          type=Path, default=None)
    parser.add_argument("--checkpoint",         type=Path, required=True)
    parser.add_argument("--output_dir",         type=Path, required=True)
    parser.add_argument("--split",              type=str,  default=None,
                        choices={"test", "valid"})
    parser.add_argument("--batch_size",         type=int,  default=None)
    parser.add_argument("--gpu",                type=int,  default=None,
                        help="GPU index; -1 for CPU")
    parser.add_argument("--mask_threshold",     type=float, default=None)
    parser.add_argument("--map_size",           type=int,  default=None)
    parser.add_argument("--mask_output_format", type=str,  default=None,
                        choices={"png", "nifti"})
    parser.add_argument("--backbone",           type=str,  default=None,
                        help="Override model backbone (e.g. radimagenet_resnet50)")
    parser.add_argument("--radimagenet_ckpt",   type=str,  default=None,
                        help="Path to RadImageNet .pt checkpoint")
    parser.add_argument("--normal_train_dir",   type=str,  default=None,
                        help="Relative path to training normal dir (default: train/good)")
    parser.add_argument("--num_workers",        type=int,  default=None,
                        help="DataLoader worker count")
    parser.add_argument("--extensions",         type=str,  nargs="+", default=None,
                        help="Image file extensions to load (default: .png)")
    return parser.parse_args()


def _merge_config(cfg, args: argparse.Namespace):
    overrides = {
        "extract.split":              args.split,
        "extract.batch_size":         args.batch_size,
        "extract.gpu":                args.gpu,
        "extract.mask_threshold":     args.mask_threshold,
        "extract.map_size":           args.map_size,
        "extract.mask_output_format": args.mask_output_format,
        "extract.normal_train_dir":   args.normal_train_dir,
        "extract.num_workers":        args.num_workers,
        "extract.extensions":         args.extensions,
        "model.backbone":             args.backbone,
        "model.radimagenet_ckpt":     args.radimagenet_ckpt,
    }
    for key, val in overrides.items():
        if val is not None:
            OmegaConf.update(cfg, key, val, merge=True)
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    cfg  = _merge_config(cfg, args)
    ecfg = cfg.extract

    model_name = cfg.model.name.lower()
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(ALL_MODELS)}")
    entry = ALL_MODELS[model_name]

    if args.data_root is not None:
        data_root = args.data_root.resolve()
    elif OmegaConf.select(cfg, "train.data_root") is not None:
        data_root = Path(cfg.train.data_root).resolve()
    else:
        raise ValueError("data_root must be set via --data_root or train.data_root in the config")
    output_dir = args.output_dir.resolve()

    maps_dir  = output_dir / "anomaly_maps"
    masks_dir = output_dir / "prediction_masks"
    maps_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    normal_test_dirs, abnormal_dir, mask_dir = _resolve_split_dirs(data_root, ecfg.split)

    normal_train_dir = OmegaConf.select(ecfg, "normal_train_dir") or "train/good"
    num_workers      = int(OmegaConf.select(ecfg, "num_workers") or 8)
    extensions       = tuple(OmegaConf.select(ecfg, "extensions") or [".png"])

    datamodule = Folder(
        name=f"{model_name}_{ecfg.split}",
        root=str(data_root),
        normal_dir=normal_train_dir,
        normal_test_dir=normal_test_dirs,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        eval_batch_size=ecfg.batch_size,
        num_workers=num_workers,
        extensions=extensions,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.NONE,
        test_split_ratio=0.0,
        val_split_ratio=0.0,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    if entry["trainer_type"] == "engine":
        ModelClass = entry["class"]
        ckpt_kwargs: dict = {
            **entry["init_kwargs"],
        }
        if OmegaConf.select(cfg.model, "backbone") is not None:
            ckpt_kwargs["backbone"] = cfg.model.backbone
        if OmegaConf.select(cfg.model, "encoder_name") is not None:
            ckpt_kwargs["encoder_name"] = cfg.model.encoder_name
        if OmegaConf.select(cfg.model, "dtd_dir") is not None:
            ckpt_kwargs["dtd_dir"] = cfg.model.dtd_dir
        if OmegaConf.select(cfg.model, "pre_processor") is not None:
            ckpt_kwargs["pre_processor"] = cfg.model.pre_processor
        if OmegaConf.select(cfg.model, "post_processor") is not None:
            ckpt_kwargs["post_processor"] = cfg.model.post_processor
        if OmegaConf.select(cfg.model, "layers") is not None:
            ckpt_kwargs["layers"] = list(cfg.model.layers)
        if OmegaConf.select(cfg.model, "learning_rate") is not None:
            ckpt_kwargs["lr"] = cfg.model.learning_rate
        model = ModelClass.load_from_checkpoint(
            str(args.checkpoint), weights_only=False, **ckpt_kwargs
        )
    else:
        radimagenet_ckpt = OmegaConf.select(cfg.model, "radimagenet_ckpt")
        model = load_flow_checkpoint(
            model_name=model_name,
            checkpoint_path=str(args.checkpoint),
            backbone=cfg.model.backbone,
            radimagenet_ckpt=radimagenet_ckpt,
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    gpu = int(ecfg.gpu)
    if gpu >= 0 and torch.cuda.is_available():
        accelerator, devices = "gpu", [gpu]
    else:
        accelerator, devices = "cpu", 1

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_progress_bar=True,
    )
    predictions = trainer.predict(model=model, datamodule=datamodule)

    # ── Save outputs ─────────────────────────────────────────────────────────
    saved = 0
    target_size = (ecfg.map_size, ecfg.map_size)

    for batch in predictions:
        if batch is None:
            continue

        anomaly_maps = _extract(batch, ("anomaly_map", "anomaly_maps"))
        pred_masks   = _extract(batch, ("pred_mask", "pred_masks", "anomaly_masks", "anomaly_mask"))
        image_paths  = _coerce_list(_extract(batch, ("image_path", "image_paths")))

        if anomaly_maps is None or not image_paths:
            continue

        a_t = torch.as_tensor(anomaly_maps).detach().cpu()
        if a_t.ndim == 4 and a_t.shape[1] == 1:
            a_t = a_t.squeeze(1)
        if a_t.ndim == 4 and a_t.shape[0] == 1:
            a_t = a_t.squeeze(0)
        if a_t.ndim == 3 and a_t.shape[-2:] != target_size:
            a_t = F.interpolate(
                a_t.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        if pred_masks is not None:
            m_t = torch.as_tensor(pred_masks).detach().cpu()
            if m_t.ndim == 4 and m_t.shape[1] == 1:
                m_t = m_t.squeeze(1)
            if m_t.ndim == 4 and m_t.shape[0] == 1:
                m_t = m_t.squeeze(0)
            if m_t.ndim == 3 and m_t.shape[-2:] != target_size:
                m_t = F.interpolate(
                    m_t.unsqueeze(1).float(), size=target_size, mode="nearest"
                ).squeeze(1)
        else:
            m_t = (a_t >= ecfg.mask_threshold).to(torch.uint8)

        if m_t.dtype != torch.uint8:
            m_t = (m_t >= ecfg.mask_threshold).to(torch.uint8)

        for idx, path_str in enumerate(image_paths):
            try:
                a_np = a_t[idx].numpy().astype("float32")
                m_np = m_t[idx].numpy().astype(np.uint8)
            except Exception:
                continue

            img_path = Path(path_str)
            try:
                rel = img_path.relative_to(data_root)
            except ValueError:
                rel = Path(img_path.name)

            out_rel   = _derive_output_rel(rel, ecfg.split)
            stem      = img_path.stem
            map_path  = maps_dir  / out_rel.parent / f"{stem}_anomaly_map.npy"
            mask_ext  = ".png" if ecfg.mask_output_format == "png" else ".nii.gz"
            mask_path = masks_dir / out_rel.parent / f"{stem}_pred_mask{mask_ext}"

            map_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(map_path, a_np)

            if ecfg.mask_output_format == "png":
                if m_np.max() <= 1:
                    m_np = m_np * 255
                Image.fromarray(m_np).save(mask_path)
            else:
                nib.save(
                    nib.Nifti1Image(m_np.astype(np.float32), np.eye(4)), mask_path
                )

            saved += 1

    print(f"[INFO] Saved {saved} anomaly maps and prediction masks to {output_dir}")


if __name__ == "__main__":
    main()
