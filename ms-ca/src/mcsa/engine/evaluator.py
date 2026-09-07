"""Test-set evaluation.

Reports how closely the 7 proxy features (Stream A: NO2/SO2/CO; Stream B:
Nightlight/Urban Fraction/Power Plant/Fossil Capacity) reproduce the actual
carbon map:

    MAE  -- Mean Absolute Error
    RMSE -- Root Mean Squared Error
    R^2  -- Coefficient of determination
    r    -- Pearson correlation coefficient
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from msca.utils.metrics import compute_metrics

__all__ = ["collect_predictions", "evaluate", "format_metrics"]


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return ``(preds, targets)`` as 1-D arrays.

    Split out from ``evaluate`` so predictions can be reused for residual plots,
    error maps, or XAI overlays without a second forward pass.
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="[test]", leave=False, dynamic_ncols=True):
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        reg_target = batch["label_reg"].float().view(-1)

        reg_out = model(stream_a, stream_b).detach().cpu().view(-1)
        all_preds.append(reg_out)
        all_targets.append(reg_target)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return preds, targets


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Full regression metric suite over a loader."""
    preds, targets = collect_predictions(model, loader, device)
    return compute_metrics(preds, targets)


def format_metrics(metrics: dict[str, float]) -> str:
    """Render the metric dict as the console report block."""
    return (
        "\n================ TEST-SET REGRESSION METRICS ================\n"
        f"  Samples (N)              : {int(metrics['n'])}\n"
        f"  MAE  (Mean Abs Error)    : {metrics['mae']:.6f}\n"
        f"  RMSE (Root Mean Sq Err)  : {metrics['rmse']:.6f}\n"
        f"  R^2  (Coeff. of Determ.) : {metrics['r2']:.6f}\n"
        f"  Pearson correlation (r)  : {metrics['pearson']:.6f}\n"
        "============================================================="
    )
