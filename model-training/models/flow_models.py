"""FastFlow and CFlow model factories for MROOD-TRAIN.

Both models use a normalising-flow backbone (hence "flow models") and are
trained with a raw Lightning Trainer (monitoring train_loss_step), rather than
anomalib's Engine.  They optionally use RadImageNet-pre-trained ResNet weights
instead of ImageNet weights.
"""
from __future__ import annotations

import torch
from anomalib.models import Fastflow
from anomalib.models.image.cflow import Cflow

from models.radimagenet_utils import load_radimagenet_resnet_weights


# ── Backbone name resolution ───────────────────────────────────────────────────

_BACKBONE_ALIASES: dict[str, str] = {
    "resnet50":        "resnet50",
    "50":              "resnet50",
    "resnet18":        "resnet18",
    "18":              "resnet18",
    "wideresnet50":    "wide_resnet50_2",
    "wideresnet50_2":  "wide_resnet50_2",
    "wide_resnet50_2": "wide_resnet50_2",
    "wrn50":           "wide_resnet50_2",
}


def _resolve_backbone(backbone: str) -> str:
    """Resolve backbone alias (incl. 'radimagenet_*' prefix) to anomalib name."""
    name = backbone.lower()
    suffix = name.replace("radimagenet", "").strip("-_ ") or "resnet50"
    return _BACKBONE_ALIASES.get(suffix, suffix)


def _is_radimagenet(backbone: str) -> bool:
    return "radimagenet" in backbone.lower()


# ── Model builders ─────────────────────────────────────────────────────────────

def build_fastflow(backbone: str, radimagenet_ckpt: str | None = None) -> Fastflow:
    """Build a Fastflow model, optionally loading RadImageNet backbone weights."""
    target = _resolve_backbone(backbone)
    try:
        model = Fastflow(backbone=target, pre_trained=False)
    except TypeError:
        model = Fastflow()
    if _is_radimagenet(backbone) and radimagenet_ckpt:
        try:
            fe = model.model.feature_extractor
            load_radimagenet_resnet_weights(fe, radimagenet_ckpt, strict=False)
        except AttributeError:
            pass  # internal structure may vary across anomalib versions
    return model


def build_cflow(backbone: str, radimagenet_ckpt: str | None = None) -> Cflow:
    """Build a Cflow model, optionally loading RadImageNet backbone weights."""
    target = _resolve_backbone(backbone)
    model = Cflow(backbone=target, pre_trained=False)
    if _is_radimagenet(backbone) and radimagenet_ckpt:
        try:
            fe = model.model.encoder.feature_extractor
            load_radimagenet_resnet_weights(fe, radimagenet_ckpt, strict=False)
        except AttributeError:
            pass
    return model


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_flow_checkpoint(
    model_name: str,
    checkpoint_path: str,
    backbone: str,
    radimagenet_ckpt: str | None = None,
):
    """Load a flow model from a Lightning checkpoint.

    Tries `load_from_checkpoint` first; falls back to manual state_dict loading
    to handle checkpoints saved without hparams.
    """
    ModelClass = Fastflow if model_name == "fastflow" else Cflow
    target = _resolve_backbone(backbone)

    # Preferred path: anomalib's built-in loader
    try:
        return ModelClass.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            backbone=target,
            strict=True,
        )
    except Exception:
        pass

    # Fallback: build a fresh model and transplant weights
    builder = build_fastflow if model_name == "fastflow" else build_cflow
    base_model = builder(backbone, radimagenet_ckpt)
    blob = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        base_model.load_state_dict(blob["state_dict"], strict=False)
    elif isinstance(blob, dict):
        base_model.load_state_dict(blob, strict=False)
    else:
        base_model.load_state_dict(blob.state_dict(), strict=False)
    return base_model


# ── Registry ───────────────────────────────────────────────────────────────────

FLOW_MODEL_REGISTRY: dict[str, dict] = {
    "fastflow": {
        "builder":            build_fastflow,
        "trainer_type":       "trainer",   # uses raw Trainer, not anomalib Engine
        "checkpoint_monitor": "train_loss_step",
        "checkpoint_mode":    "min",
    },
    "cflow": {
        "builder":            build_cflow,
        "trainer_type":       "trainer",
        "checkpoint_monitor": "train_loss_step",
        "checkpoint_mode":    "min",
    },
}
