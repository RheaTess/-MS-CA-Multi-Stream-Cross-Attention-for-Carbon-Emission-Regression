"""Metric correctness, and agreement between the two implementations."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from msca.utils.metrics import RegressionMeter, compute_metrics


def test_perfect_prediction():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(targets.copy(), targets)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0)


def test_known_values():
    preds = np.array([2.0, 4.0])
    targets = np.array([1.0, 1.0])
    m = compute_metrics(preds, targets)
    assert m["mae"] == pytest.approx(2.0)          # (1 + 3) / 2
    assert m["rmse"] == pytest.approx(math.sqrt(5))  # sqrt((1 + 9) / 2)


def test_constant_prediction_gives_nan_pearson_not_a_crash():
    m = compute_metrics(np.full(10, 3.0), np.arange(10.0))
    assert math.isnan(m["pearson"])


def test_meter_and_compute_metrics_agree():
    """The whole point of unifying them: these must never drift."""
    rng = np.random.default_rng(0)
    preds = rng.normal(size=97)
    targets = rng.normal(size=97)

    meter = RegressionMeter()
    for i in range(0, 97, 16):  # ragged final batch, on purpose
        meter.update(
            torch.tensor(preds[i : i + 16]), torch.tensor(targets[i : i + 16])
        )

    reference = compute_metrics(preds, targets)
    assert meter.mae == pytest.approx(reference["mae"], rel=1e-6)
    assert meter.rmse == pytest.approx(reference["rmse"], rel=1e-6)
    assert meter.count == 97


def test_meter_is_exact_not_a_mean_of_batch_means():
    """A ragged final batch must not be over-weighted."""
    meter = RegressionMeter()
    meter.update(torch.zeros(100), torch.zeros(100))  # error 0, n=100
    meter.update(torch.ones(1) * 10, torch.zeros(1))  # error 10, n=1
    assert meter.mae == pytest.approx(10.0 / 101)     # not (0 + 10) / 2


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(3), np.zeros(4))
