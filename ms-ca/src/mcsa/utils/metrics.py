"""Regression metrics -- one implementation, used by both training and eval.

Previously ``RegressionMeter`` (train.py) and ``compute_metrics``
(evaluation.py) each computed MAE and RMSE their own way. Two implementations of
the same formula is two places for them to drift apart, and any drift shows up
as a train/test discrepancy that looks like a modelling result. They are now the
same code.

    RegressionMeter  -- streaming, O(1) memory, for per-epoch train/val loops.
    compute_metrics  -- full suite (MAE, RMSE, R^2, Pearson r) over arrays,
                        for final test-set evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["RegressionMeter", "compute_metrics"]


class RegressionMeter:
    """Accumulates absolute and squared errors to compute MAE / RMSE exactly.

    Streaming, so memory is constant regardless of epoch size, and the result is
    identical to computing over the whole epoch at once -- unlike averaging
    per-batch means, which silently mis-weights a smaller final batch.
    """

    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def reset(self) -> None:
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        diff = (preds.detach().view(-1) - targets.detach().view(-1)).float()
        self.sum_abs += diff.abs().sum().item()
        self.sum_sq += (diff * diff).sum().item()
        self.count += diff.numel()

    @property
    def mae(self) -> float:
        return self.sum_abs / self.count if self.count else float("nan")

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sum_sq / self.count) if self.count else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "n": float(self.count)}


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R^2, and Pearson r over 1-D prediction / target arrays.

    Computed in float64 regardless of the model's dtype: with AMP on, summing
    squared errors in float32 over ~2,500 test samples loses meaningful
    precision in RMSE and R^2.
    """
    preds = np.asarray(preds, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    if preds.shape != targets.shape:
        raise ValueError(
            f"preds and targets must have the same length, "
            f"got {preds.shape} and {targets.shape}."
        )
    if preds.size == 0:
        raise ValueError("Cannot compute metrics over an empty array.")

    errors = preds - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # R^2 = 1 - SS_res / SS_tot
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Pearson r (guard against zero variance -- e.g. a collapsed model that
    # predicts one constant, where corrcoef would emit a divide-by-zero nan).
    if preds.std() > 0 and targets.std() > 0:
        pearson = float(np.corrcoef(preds, targets)[0, 1])
    else:
        pearson = float("nan")

    return {
        "n": float(preds.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
    }
