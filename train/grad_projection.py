from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


def load_refusal_directions(
    path: str, device: torch.device | str
) -> Dict[str, torch.Tensor]:
    """Load normalized direction tensors onto the requested device."""
    raw = torch.load(path, map_location=device)
    return {name: tensor.to(device) for name, tensor in raw.items()}


def load_blue_red_delta(
    path: str, device: torch.device | str
) -> Dict[str, torch.Tensor]:
    """Load blue-red delta (Δφ = φ_blue - φ_red) onto the requested device."""
    raw = torch.load(path, map_location=device)
    return {name: tensor.to(device) for name, tensor in raw.items()}


def compute_regularization_loss(
    model: nn.Module,
    delta: Dict[str, torch.Tensor],
    lambda_reg: float,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute regularization loss: (λ/2) * ||V^T(θ - θ₀)||²

    Where V is the delta direction and θ₀ is the reference state (initial params).
    If reference_state is None, uses current model state as reference.
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        delta_vec = delta.get(name)
        if delta_vec is None:
            continue

        # Get current parameter value
        current = param.data

        # Get reference (initial state) if provided
        if reference_state is not None:
            ref = reference_state.get(name)
            if ref is None:
                continue
            diff = current - ref
        else:
            # Use current state as reference (regularize movement in delta direction)
            diff = current

        # Project diff onto delta direction: V^T * diff
        # For each parameter tensor, compute the dot product with delta
        projection = (diff * delta_vec).sum()

        # Add squared projection to regularization loss
        reg_loss = reg_loss + projection**2

    return (lambda_reg / 2.0) * reg_loss


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
