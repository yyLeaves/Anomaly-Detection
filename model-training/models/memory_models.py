"""Memory-bank models: CFA and PatchCore."""

import torch
from anomalib.models import Cfa, Patchcore


class CfaCustomLR(Cfa):
    """CFA with configurable Adam optimizer (lr, weight_decay, amsgrad)."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        amsgrad: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.save_hyperparameters(ignore=["evaluator"])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


MEMORY_MODEL_REGISTRY: dict[str, dict] = {
    "cfa": {
        # No layers; optimizer params (weight_decay, amsgrad) fixed via init_kwargs.
        # lr comes from cfg.model.learning_rate (set via OmegaConf.select in train.py).
        "class":              CfaCustomLR,
        "init_kwargs":        {
            "gamma_c":                    1,
            "gamma_d":                    1,
            "num_nearest_neighbors":      3,
            "num_hard_negative_features": 3,
            "radius":                     1e-5,
            "weight_decay":               5e-4,
            "amsgrad":                    True,
        },
        "trainer_type":       "engine",
        "checkpoint_monitor": "image_AUROC",
        "checkpoint_mode":    "max",
    },
    "patchcore": {
        # Memory-bank method: no gradient training, no lr needed.
        # layers come from cfg.model.layers (set via OmegaConf.select in train.py).
        "class":              Patchcore,
        "init_kwargs":        {
            "pre_trained":            True,
            "coreset_sampling_ratio": 0.001,
            "num_neighbors":          9,
        },
        "trainer_type":       "engine",
        "checkpoint_monitor": "image_AUROC",
        "checkpoint_mode":    "max",
    },
}
