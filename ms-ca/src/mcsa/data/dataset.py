"""Patch dataset for the MS-CA spatial tensor ``.npz`` files (pure regression).

Invariants this module enforces
-------------------------------
1. PURE REGRESSION. ``label_cls`` (the 0-4 hotspot grade) is never loaded,
   validated, or returned. It is ignored even if still present in the ``.npz``.
   The only label exposed is the continuous scalar ``label_reg``.
2. NO SPATIAL SPLIT. Every valid mask cell is registered as a sample. Splitting
   is the caller's job -- see :mod:`msca.data.splits`. The data is already
   averaged per grid, so spatial autocorrelation is not a concern.
3. FIXED CHANNELS. Stream A = 3 (NO2, SO2, CO). Stream B = 4 (Nightlight, Urban
   Fraction, Power Plant, Fossil Capacity); population is excluded.
4. RAW DISTRIBUTION PRESERVED. NO clipping, capping, or outlier removal on
   either features or ``label_reg`` targets.

Note on memory: every ``.npz`` is eagerly loaded into RAM in ``__init__`` and
cropped on the fly. For the current 17k-grid scale that is the right trade-off
(cheap, and workers share it via copy-on-write fork). If the tensor count grows
by an order of magnitude, switch ``_load_npz`` to ``mmap_mode='r'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = [
    "SpatialGridDataset",
    "load_tensor_index",
    "REQUIRED_ARRAYS",
    "STREAM_A_CHANNELS",
    "STREAM_B_CHANNELS",
]


# ``label_cls`` is deliberately NOT required -- pure-regression pipeline.
REQUIRED_ARRAYS = ["stream_a", "stream_b", "label_reg", "mask"]

# Fixed input-channel configuration (population excluded from Stream B).
STREAM_A_CHANNELS = 3  # NO2, SO2, CO
STREAM_B_CHANNELS = 4  # Nightlight, Urban Fraction, Power Plant, Fossil Capacity


def load_tensor_index(tensor_index_csv: str | Path) -> list[str]:
    """Read a tensor index CSV and return tensor paths in row order."""
    index_path = Path(tensor_index_csv)
    if not index_path.exists():
        raise FileNotFoundError(f"Tensor index CSV not found: {index_path}")

    df = pd.read_csv(index_path)
    if "tensor_path" not in df.columns:
        raise ValueError(
            f"Tensor index CSV must contain 'tensor_path'. "
            f"Available columns: {list(df.columns)}"
        )

    return df["tensor_path"].astype(str).tolist()


class SpatialGridDataset(Dataset):
    """Patch-based dataset for MS-CA spatial tensor ``.npz`` files (regression)."""

    def __init__(
        self,
        npz_paths: list[str | Path] | None = None,
        tensor_index_csv: str | Path | None = None,
        crop_size: int = 16,
        target: str = "center",
        transform: Any = None,
    ) -> None:
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got: {crop_size}")
        if target != "center":
            raise ValueError("Only target='center' is currently supported.")
        if npz_paths is None and tensor_index_csv is None:
            raise ValueError("Provide either npz_paths or tensor_index_csv.")
        if npz_paths is not None and tensor_index_csv is not None:
            raise ValueError("Provide only one of npz_paths or tensor_index_csv.")

        self.crop_size = crop_size
        self.target = target
        self.transform = transform

        path_values = (
            npz_paths if npz_paths is not None else load_tensor_index(tensor_index_csv)
        )
        self.npz_paths = [Path(path) for path in path_values]
        self.samples: list[tuple[int, int, int]] = []
        self.monthly_tensors: list[dict[str, Any]] = []

        for file_idx, npz_path in enumerate(self.npz_paths):
            tensor_dict = self._load_npz(npz_path)
            self.monthly_tensors.append(tensor_dict)

            # Every valid mask cell becomes a sample. Train/val/test
            # partitioning happens downstream in msca.data.splits.
            valid_rows, valid_cols = np.where(tensor_dict["mask"] == 1)
            for row, col in zip(valid_rows.tolist(), valid_cols.tolist(), strict=True):
                self.samples.append((file_idx, row, col))

        if not self.samples:
            raise ValueError("No valid mask cells found in the provided tensor files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, row, col = self.samples[idx]
        tensor_dict = self.monthly_tensors[file_idx]

        stream_a_patch = self._crop_with_padding(tensor_dict["stream_a"], row, col)
        stream_b_patch = self._crop_with_padding(tensor_dict["stream_b"], row, col)
        mask_patch = self._crop_with_padding(tensor_dict["mask"], row, col)

        sample = {
            "stream_a": torch.as_tensor(stream_a_patch, dtype=torch.float32),
            "stream_b": torch.as_tensor(stream_b_patch, dtype=torch.float32),
            # The ONLY label: a single continuous carbon-emission value.
            "label_reg": torch.as_tensor(
                tensor_dict["label_reg"][row, col],
                dtype=torch.float32,
            ),
            "mask": torch.as_tensor(mask_patch, dtype=torch.float32),
            "date": tensor_dict["date"],
            "row": row,
            "col": col,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # -------------------------------------------------------------- #
    # Introspection
    # -------------------------------------------------------------- #
    def label_stats(self) -> dict[str, float]:
        """Raw ``label_reg`` statistics over all valid cells. Nothing clipped."""
        reg_values: list[float] = []
        for tensor_dict in self.monthly_tensors:
            valid_mask = tensor_dict["mask"] == 1
            reg_values.extend(tensor_dict["label_reg"][valid_mask].tolist())

        reg_array = np.asarray(reg_values, dtype=np.float64)
        return {
            "count": float(reg_array.size),
            "min": float(reg_array.min()),
            "max": float(reg_array.max()),
            "mean": float(reg_array.mean()),
            "std": float(reg_array.std()),
        }

    def print_debug_summary(self) -> None:
        stats = self.label_stats()
        first_tensor = self.monthly_tensors[0]
        first_sample = self[0]

        print("SpatialGridDataset summary (pure regression)")
        print(f"NPZ files loaded: {len(self.monthly_tensors)}")
        print(f"Total valid samples: {len(self)}")
        print(f"stream_a shape: {first_tensor['stream_a'].shape}")
        print(f"stream_b shape: {first_tensor['stream_b'].shape}")
        print("label_reg statistics (raw, no outlier handling):")
        print(f"  count : {int(stats['count'])}")
        print(f"  min   : {stats['min']:.6f}")
        print(f"  max   : {stats['max']:.6f}")
        print(f"  mean  : {stats['mean']:.6f}")
        print(f"  std   : {stats['std']:.6f}")
        print("First sample shapes:")
        print(f"stream_a: {tuple(first_sample['stream_a'].shape)}")
        print(f"stream_b: {tuple(first_sample['stream_b'].shape)}")
        print(f"mask: {tuple(first_sample['mask'].shape)}")
        print(f"label_reg: {tuple(first_sample['label_reg'].shape)}")

    # -------------------------------------------------------------- #
    # Internals
    # -------------------------------------------------------------- #
    def _load_npz(self, npz_path: Path) -> dict[str, Any]:
        if not npz_path.exists():
            raise FileNotFoundError(f"Tensor .npz file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_ARRAYS if key not in data.files]
            if missing:
                raise ValueError(f"{npz_path} missing required arrays: {missing}")

            # label_cls is intentionally NOT read -- pure regression only.
            tensor_dict = {
                "stream_a": data["stream_a"].astype(np.float32),
                "stream_b": data["stream_b"].astype(np.float32),
                "label_reg": data["label_reg"].astype(np.float32),
                "mask": data["mask"].astype(np.float32),
                "date": _read_npz_date(data, npz_path),
            }

        self._validate_tensor_shapes(npz_path, tensor_dict)
        return tensor_dict

    def _validate_tensor_shapes(
        self, npz_path: Path, tensor_dict: dict[str, Any]
    ) -> None:
        stream_a = tensor_dict["stream_a"]
        stream_b = tensor_dict["stream_b"]
        label_reg = tensor_dict["label_reg"]
        mask = tensor_dict["mask"]

        if stream_a.ndim != 3 or stream_a.shape[0] != STREAM_A_CHANNELS:
            raise ValueError(
                f"{npz_path} stream_a must have shape [{STREAM_A_CHANNELS}, H, W], "
                f"got {stream_a.shape}"
            )
        # Stream B is 4 channels (population dropped), not 5.
        if stream_b.ndim != 3 or stream_b.shape[0] != STREAM_B_CHANNELS:
            raise ValueError(
                f"{npz_path} stream_b must have shape [{STREAM_B_CHANNELS}, H, W], "
                f"got {stream_b.shape}"
            )

        spatial_shape = stream_a.shape[1:]
        if stream_b.shape[1:] != spatial_shape:
            raise ValueError(
                f"{npz_path} stream_a and stream_b spatial shapes differ: "
                f"{spatial_shape} vs {stream_b.shape[1:]}"
            )

        for name, array in [("label_reg", label_reg), ("mask", mask)]:
            if array.shape != spatial_shape:
                raise ValueError(
                    f"{npz_path} {name} shape must match H/W {spatial_shape}, "
                    f"got {array.shape}"
                )

        if not np.any(mask == 1):
            raise ValueError(f"{npz_path} contains no valid mask cells.")

    def _crop_with_padding(self, array: np.ndarray, row: int, col: int) -> np.ndarray:
        """Centre-crop around (row, col), zero-padding beyond raster edges."""
        pad_before = self.crop_size // 2
        pad_after = self.crop_size - pad_before - 1

        row_start = row - pad_before
        row_end = row + pad_after + 1
        col_start = col - pad_before
        col_end = col + pad_after + 1

        if array.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")

        height, width = array.shape[-2:]
        out_shape = (
            (array.shape[0], self.crop_size, self.crop_size)
            if array.ndim == 3
            else (self.crop_size, self.crop_size)
        )
        output = np.zeros(out_shape, dtype=array.dtype)

        src_row_start = max(row_start, 0)
        src_row_end = min(row_end, height)
        src_col_start = max(col_start, 0)
        src_col_end = min(col_end, width)
        dst_row_start = src_row_start - row_start
        dst_col_start = src_col_start - col_start

        dst_rows = slice(dst_row_start, dst_row_start + (src_row_end - src_row_start))
        dst_cols = slice(dst_col_start, dst_col_start + (src_col_end - src_col_start))
        src_rows = slice(src_row_start, src_row_end)
        src_cols = slice(src_col_start, src_col_end)

        if array.ndim == 3:
            output[:, dst_rows, dst_cols] = array[:, src_rows, src_cols]
        else:
            output[dst_rows, dst_cols] = array[src_rows, src_cols]
        return output


def _read_npz_date(data: np.lib.npyio.NpzFile, npz_path: Path) -> str:
    if "date" not in data.files:
        return npz_path.stem

    raw_date = data["date"]
    if raw_date.shape == ():
        return str(raw_date.item())
    return str(raw_date.tolist())
