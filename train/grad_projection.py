from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def load_refusal_directions(
    path: str, device: torch.device | str
) -> Dict[str, torch.Tensor]:
    """Load normalized direction tensors onto the requested device."""
    raw = torch.load(path, map_location=device)
    return {name: tensor.to(device) for name, tensor in raw.items()}


def apply_orthogonal_projection(
    model: nn.Module, refusal_directions: Dict[str, torch.Tensor]
) -> None:
    """Project gradients to the subspace orthogonal to the refusal direction.

    For each trainable parameter with a matching direction, projects the gradient
    to be orthogonal: g' = g - (g·u / ||u||²) * u where u is the normalized direction.

    Note: Parameter names from model.named_parameters() must match keys in
    refusal_directions (from the refusal LoRA's state_dict).
    """
    eps = 1e-12
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        direction = refusal_directions.get(name)
        if direction is None:
            continue
        grad = param.grad
        u = direction.to(grad.device, dtype=grad.dtype)
        denom = torch.sum(u * u) + eps
        scale = torch.sum(grad * u) / denom
        grad.sub_(scale * u)
