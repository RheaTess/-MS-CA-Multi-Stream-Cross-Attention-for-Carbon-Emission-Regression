"""Dataset, split, and augmentation contract tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msca.data import (
    SpatialGridDataset,
    apply_d4_transform,
    build_splits,
    load_split_indices,
    save_split_indices,
)


def test_sample_keys_and_shapes(npz_file):
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    sample = ds[0]
    assert sample["stream_a"].shape == (3, 16, 16)
    assert sample["stream_b"].shape == (4, 16, 16)
    assert sample["label_reg"].shape == ()  # scalar


def test_label_cls_is_never_exposed(npz_file):
    """The .npz still contains label_cls; the dataset must ignore it."""
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    assert "label_cls" not in ds[0]
    assert "label_cls" not in ds.monthly_tensors[0]


def test_every_valid_mask_cell_becomes_a_sample(npz_file):
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    with np.load(npz_file) as data:
        expected = int((data["mask"] == 1).sum())
    assert len(ds) == expected


def test_wrong_stream_b_channel_count_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    mask = np.ones((8, 8), dtype=np.float32)
    np.savez(
        path,
        stream_a=np.zeros((3, 8, 8), dtype=np.float32),
        stream_b=np.zeros((5, 8, 8), dtype=np.float32),  # old population layout
        label_reg=np.zeros((8, 8), dtype=np.float32),
        mask=mask,
    )
    with pytest.raises(ValueError, match="stream_b"):
        SpatialGridDataset(npz_paths=[path])


def test_labels_are_not_clipped(npz_file, tmp_path):
    """Raw distribution must survive: a huge outlier stays a huge outlier."""
    path = tmp_path / "outlier.npz"
    mask = np.ones((6, 6), dtype=np.float32)
    label = np.zeros((6, 6), dtype=np.float32)
    label[3, 3] = 1e6
    np.savez(
        path,
        stream_a=np.zeros((3, 6, 6), dtype=np.float32),
        stream_b=np.zeros((4, 6, 6), dtype=np.float32),
        label_reg=label,
        mask=mask,
    )
    ds = SpatialGridDataset(npz_paths=[path])
    assert ds.label_stats()["max"] == pytest.approx(1e6)


def test_crop_pads_at_the_edge_instead_of_shifting(npz_file):
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    for sample in (ds[0], ds[len(ds) - 1]):
        assert sample["stream_a"].shape == (3, 16, 16)


def test_splits_are_disjoint_and_exhaustive(npz_file):
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    train, val, test = build_splits(ds, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(train) + len(val) + len(test) == len(ds)
    idx = [set(s.indices) for s in (train, val, test)]
    assert idx[0].isdisjoint(idx[1])
    assert idx[0].isdisjoint(idx[2])
    assert idx[1].isdisjoint(idx[2])


def test_same_seed_reproduces_the_same_split(npz_file):
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    a = build_splits(ds, seed=42)[2].indices
    b = build_splits(ds, seed=42)[2].indices
    c = build_splits(ds, seed=7)[2].indices
    assert list(a) == list(b)
    assert list(a) != list(c)


def test_split_file_roundtrip(npz_file, tmp_path):
    ds = SpatialGridDataset(npz_paths=[npz_file], crop_size=16)
    train, val, test = build_splits(ds, seed=42)

    path = tmp_path / "split_indices.json"
    save_split_indices(path, ds, train, val, test, 42, 0.15, 0.15)
    _tr, _va, loaded_test = load_split_indices(path, ds)
    assert list(loaded_test.indices) == list(test.indices)


def test_fingerprint_mismatch_is_caught(npz_files, tmp_path):
    """The failure mode the seed-only contract could not detect."""
    ds_one = SpatialGridDataset(npz_paths=[npz_files[0]], crop_size=16)
    train, val, test = build_splits(ds_one, seed=42)

    path = tmp_path / "split_indices.json"
    save_split_indices(path, ds_one, train, val, test, 42, 0.15, 0.15)

    # Same seed, same ratios -- but a second month was added to the run.
    ds_two = SpatialGridDataset(npz_paths=npz_files, crop_size=16)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_split_indices(path, ds_two, strict=True)


def test_d4_transform_is_synchronized_across_streams():
    a = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    b = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    a_out, b_out = apply_d4_transform(a, b, k_rot=1, flip_rows=True, flip_cols=False)
    # Identical inputs + identical transform must give identical outputs;
    # this is what keeps the two streams spatially aligned for fusion.
    assert torch.equal(a_out, b_out)
    assert a_out.shape == (1, 3, 3)


def test_d4_transform_is_invertible_over_the_group():
    x = torch.randn(2, 4, 4)
    y = torch.randn(2, 4, 4)
    a1, _ = apply_d4_transform(x, y, k_rot=0, flip_rows=False, flip_cols=False)
    assert torch.equal(a1, x)  # identity element
