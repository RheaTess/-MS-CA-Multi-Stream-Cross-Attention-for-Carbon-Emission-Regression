"""Shared fixtures: synthetic .npz tensors matching the real channel layout."""

from __future__ import annotations

import numpy as np
import pytest


def _write_npz(path, height=12, width=10, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[2:-2, 2:-2] = 1.0  # a valid interior region
    np.savez(
        path,
        stream_a=rng.normal(size=(3, height, width)).astype(np.float32),
        stream_b=rng.normal(size=(4, height, width)).astype(np.float32),
        label_reg=rng.normal(size=(height, width)).astype(np.float32),
        # Deliberately present: the dataset must ignore it entirely.
        label_cls=rng.integers(0, 5, size=(height, width)).astype(np.float32),
        mask=mask,
        date=np.array("2023-01"),
    )
    return path


@pytest.fixture
def npz_file(tmp_path):
    return _write_npz(tmp_path / "2023_01.npz", seed=0)


@pytest.fixture
def npz_files(tmp_path):
    return [
        _write_npz(tmp_path / "2023_01.npz", seed=0),
        _write_npz(tmp_path / "2023_02.npz", seed=1),
    ]
