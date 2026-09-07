"""Per-epoch training and validation loops.

Pure regression throughout: the objective is a plain ``nn.L1Loss`` (MAE) against
the continuous ``label_reg`` target. No Focal loss, no Huber/Smooth-L1, no
uncertainty weighting, no classification branch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from msca.utils.logging import ExperimentLogger, NullLogger
from msca.utils.metrics import RegressionMeter

__all__ = ["train_one_epoch", "validate_one_epoch"]


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    epoch: int = 0,
    logger: Optional[ExperimentLogger] = None,
    log_every: int = 50,
) -> dict[str, float]:
    """One training epoch. Returns ``{'avg_loss', 'mae', 'rmse'}``."""
    logger = logger or NullLogger()
    model.train()
    meter = RegressionMeter()
    running_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        reg_target = batch["label_reg"].to(device, non_blocking=True).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            reg_out = model(stream_a, stream_b)  # [B, 1]
            loss = loss_fn(reg_out, reg_target)  # L1 / MAE

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                # Must unscale before clipping, or the norm is computed on
                # scaled gradients and the threshold means nothing.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        meter.update(reg_out, reg_target)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % log_every == 0:
            logger.log(
                {
                    "train/step_l1_loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
            )

    return {
        "avg_loss": running_loss / max(num_batches, 1),
        "mae": meter.mae,
        "rmse": meter.rmse,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int = 0,
) -> dict[str, float]:
    """One validation epoch. Returns ``{'avg_loss', 'mae', 'rmse'}``."""
    model.eval()
    meter = RegressionMeter()
    running_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False, dynamic_ncols=True)
    for batch in pbar:
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        reg_target = batch["label_reg"].to(device, non_blocking=True).float().view(-1, 1)

        reg_out = model(stream_a, stream_b)
        loss = loss_fn(reg_out, reg_target)

        running_loss += loss.item()
        meter.update(reg_out, reg_target)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "avg_loss": running_loss / max(num_batches, 1),
        "mae": meter.mae,
        "rmse": meter.rmse,
    }
