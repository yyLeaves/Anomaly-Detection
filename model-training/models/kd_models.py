"""Knowledge distillation model subclasses with custom learning-rate configuration."""

import torch
from anomalib.models import ReverseDistillation, Stfpm
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode


class ReverseDistillationCustomLR(ReverseDistillation):
    """RD4AD with a configurable learning rate (Adam on decoder only)."""

    def __init__(self, lr: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.save_hyperparameters(ignore=["evaluator"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr)


class StfpmCustomLR(Stfpm):
    """STFPM with a configurable learning rate (SGD on all parameters)."""

    def __init__(self, lr: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.save_hyperparameters(ignore=["evaluator"])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


# Registry entries use a uniform dict format shared with other model registries.
# Keys:
#   class            – LightningModule subclass to instantiate
#   init_kwargs      – extra kwargs forwarded to the constructor (on top of config values)
#   trainer_type     – "engine" → use anomalib Engine + AUROC; "trainer" → raw Trainer
#   checkpoint_monitor / checkpoint_mode – ModelCheckpoint settings
KD_MODEL_REGISTRY: dict[str, dict] = {
    "rd4ad": {
        "class":              ReverseDistillationCustomLR,
        "init_kwargs":        {"anomaly_map_mode": AnomalyMapGenerationMode.ADD, "pre_trained": True},
        "trainer_type":       "engine",
        "checkpoint_monitor": "image_AUROC",
        "checkpoint_mode":    "max",
    },
    "stfpm": {
        "class":              StfpmCustomLR,
        "init_kwargs":        {},
        "trainer_type":       "engine",
        "checkpoint_monitor": "image_AUROC",
        "checkpoint_mode":    "max",
    },
}
