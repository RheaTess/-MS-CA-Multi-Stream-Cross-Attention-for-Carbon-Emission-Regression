"""Optimizer and learning-rate schedule construction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

__all__ = ["build_param_groups", "build_optimizer", "build_warmup_cosine_scheduler"]


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Standard AdamW grouping: exclude bias/LayerNorm/1-D params from decay.

    Applying weight decay to LayerNorm gains/biases is a well-known way to
    quietly hurt Transformer training -- these parameters control scale, not
    capacity, so shrinking them is not regularization.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """AdamW over correctly-grouped parameters."""
    param_groups = build_param_groups(model, weight_decay=cfg.optim.weight_decay)
    return torch.optim.AdamW(param_groups, lr=cfg.optim.lr, betas=(0.9, 0.999))


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    cycles: int = 3,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by multiple cosine decay cycles (warm restarts).

    Stepped per optimizer step, not per epoch.
    """
    warmup_steps = max(1, warmup_steps)
    remaining_steps = max(1, total_steps - warmup_steps)
    cycles = max(1, cycles)
    cycle_length = max(1, remaining_steps // cycles)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        steps_since_warmup = current_step - warmup_steps
        position_in_cycle = steps_since_warmup % cycle_length
        progress = float(position_in_cycle) / float(cycle_length)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
