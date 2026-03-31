"""Utilities for loading RadImageNet pre-trained ResNet weights.

Sourced from Post-Processing-Pipeline/radimagenet_utils.py.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    plen = len(prefix)
    return {(k[plen:] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def _radimagenet_resnet_sequential_to_named(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert keys from Sequential(list(resnet.children())[:-1]) to standard ResNet names.

    Input keys look like:
      - 'backbone.0.weight'   → 'conv1.weight'
      - 'backbone.1.weight'   → 'bn1.weight'
      - 'backbone.4.0.conv1.' → 'layer1.0.conv1.'
      - 'backbone.5.*'        → 'layer2.*'
      - 'backbone.6.*'        → 'layer3.*'
      - 'backbone.7.*'        → 'layer4.*'
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("0."):
            nk = "conv1." + nk[len("0."):]
        elif nk.startswith("1."):
            nk = "bn1." + nk[len("1."):]
        elif nk.startswith("4."):
            nk = "layer1." + nk[len("4."):]
        elif nk.startswith("5."):
            nk = "layer2." + nk[len("5."):]
        elif nk.startswith("6."):
            nk = "layer3." + nk[len("6."):]
        elif nk.startswith("7."):
            nk = "layer4." + nk[len("7."):]
        elif nk.startswith("8.") or nk.startswith("9."):
            continue  # skip avgpool / fc
        out[nk] = v
    return out


def load_radimagenet_resnet_weights(
    module: torch.nn.Module,
    weights_path: str,
    strict: bool = False,
) -> Tuple[list, list]:
    """Load RadImageNet ResNet weights into a module.

    Handles 'backbone.'-prefixed sequential keys and remaps numeric indices to
    standard ResNet layer names (conv1/bn1/layer1-4).

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    obj = torch.load(weights_path, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    sd = _strip_prefix(sd, "backbone.")
    sd = _radimagenet_resnet_sequential_to_named(sd)
    missing, unexpected = module.load_state_dict(sd, strict=strict)
    return list(missing), list(unexpected)
