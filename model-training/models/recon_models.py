"""Reconstruction-based models: DRAEM and Dinomaly."""

from anomalib.models import Dinomaly, Draem

RECON_MODEL_REGISTRY: dict[str, dict] = {
    "draem": {
        "class":              Draem,
        "init_kwargs":        {},
        "trainer_type":       "engine",
        "checkpoint_monitor": "image_AUROC",
        "checkpoint_mode":    "max",
    },
    "dinomaly": {
        "class":              Dinomaly,
        "init_kwargs":        {},
        "trainer_type":       "engine",
        "checkpoint_monitor": "image_AUROC",
        "checkpoint_mode":    "max",
    },
}
