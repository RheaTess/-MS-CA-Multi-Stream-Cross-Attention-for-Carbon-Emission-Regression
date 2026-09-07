"""Checkpoint saving and loading.

Two artifacts, two purposes:

    best_model.pth        -- weights + model_config only. This is what
                             ``scripts/evaluate.py`` consumes and what you
                             publish. Small enough to attach to a release.
    checkpoint_latest.pth -- adds optimizer/scheduler/scaler state for
                             preemption resume. Never publish this.

Both embed ``model_config``, so evaluation reconstructs the architecture from
the checkpoint and needs no architecture flags of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "unwrap_model",
    "save_best_model",
    "save_latest_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
]


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module (strips DataParallel's ``.module``).

    Saving the wrapped state_dict is a classic own-goal: every key gets a
    ``module.`` prefix, and the checkpoint then refuses to load on single-GPU or
    CPU. Always unwrap before saving.
    """
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def save_best_model(
    path: str | Path,
    model: nn.Module,
    model_config: dict,
    epoch: int,
    best_val_mae: float,
    config: dict,
) -> None:
    """Save the lightweight 'best' artifact that evaluation consumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "config": config,
        },
        path,
    )


def save_latest_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    model_config: dict,
    epoch: int,
    best_val_mae: float,
    config: dict,
) -> None:
    """Save a full checkpoint for preemption resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # Resuming an AMP run without the scaler state restarts loss-scale
            # calibration and can throw away the first few hundred steps.
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "model_config": model_config,
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Tuple[int, float]:
    """Restore full training state. Returns ``(start_epoch, best_val_mae)``."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_mae = checkpoint.get("best_val_mae", float("inf"))
    print(f"[Resume] Loaded '{path}' -> resuming at epoch {start_epoch}.")
    return start_epoch, best_val_mae


def load_model_from_checkpoint(
    path: str | Path,
    device: torch.device,
    model_cls: Any = None,
) -> nn.Module:
    """Rebuild a model from a checkpoint's embedded ``model_config``, eval mode."""
    if model_cls is None:
        from msca.models import MSCANet

        model_cls = MSCANet

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    missing = [
        key
        for key in ("model_config", "model_state_dict")
        if key not in checkpoint
    ]
    if missing:
        raise KeyError(
            f"Checkpoint '{path}' is missing {missing}. Expected a "
            f"best_model.pth produced by scripts/train.py."
        )

    model = model_cls(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_val_mae = checkpoint.get("best_val_mae", float("nan"))
    print(f"[Model] Loaded '{path}' (best epoch={epoch}, val MAE={best_val_mae:.4f}).")
    return model
