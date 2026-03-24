"""Custom PyTorch Dataset for lazily loading climate data from Zarr stores via Xarray.

The dataset assumes the Zarr store contains at least:
  - An input variable (e.g. low-resolution climate field)
  - A target variable (e.g. high-resolution / observed field)

Both are identified by their variable names in the Xarray dataset.  Data are
loaded lazily so that only the requested chunk is pulled into RAM at call time.

Example layout of a Zarr store
--------------------------------
/<zarr_path>
  ├── time     (T,)
  ├── lat      (H,)
  ├── lon      (W,)
  ├── lr_field (T, H, W)   ← input variable
  └── hr_field (T, H, W)   ← target variable
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class ClimateZarrDataset(Dataset):
    """Lazily loads (input, target) pairs from a Zarr store.

    Parameters
    ----------
    zarr_path:
        Path to the root of the Zarr store (directory or ``.zarr`` file).
    input_var:
        Name of the data variable to use as model input.
    target_var:
        Name of the data variable to use as model target.
    time_dim:
        Name of the time dimension inside the dataset (default ``"time"``).
    transform:
        Optional callable applied to the input tensor after loading.
    target_transform:
        Optional callable applied to the target tensor after loading.
    normalize:
        If ``True`` the dataset computes per-variable mean/std over the full
        time axis (lazily via Xarray) and normalises input and target to
        zero-mean / unit-variance.
    chunks:
        Dask chunk specification forwarded to :func:`xarray.open_zarr`.  Set to
        ``None`` to load eagerly (not recommended for large datasets).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        input_var: str,
        target_var: str,
        time_dim: str = "time",
        transform=None,
        target_transform=None,
        normalize: bool = True,
        chunks: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.zarr_path = Path(zarr_path)
        self.input_var = input_var
        self.target_var = target_var
        self.time_dim = time_dim
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize

        # Open dataset lazily – no data is loaded into RAM yet.
        _chunks = chunks if chunks is not None else {time_dim: 1}
        self.ds: xr.Dataset = xr.open_zarr(str(self.zarr_path), chunks=_chunks)

        if input_var not in self.ds:
            raise KeyError(f"Variable '{input_var}' not found in Zarr store at {zarr_path}.")
        if target_var not in self.ds:
            raise KeyError(f"Variable '{target_var}' not found in Zarr store at {zarr_path}.")

        self._length: int = self.ds.sizes[time_dim]

        # Pre-compute normalisation statistics lazily (Dask computes only once).
        if normalize:
            self._input_mean = float(self.ds[input_var].mean().compute())
            self._input_std = float(self.ds[input_var].std().compute())
            self._target_mean = float(self.ds[target_var].mean().compute())
            self._target_std = float(self.ds[target_var].std().compute())
            # Guard against zero-variance (e.g. constant fields)
            self._input_std = self._input_std if self._input_std > 0 else 1.0
            self._target_std = self._target_std if self._target_std > 0 else 1.0
        else:
            self._input_mean = 0.0
            self._input_std = 1.0
            self._target_mean = 0.0
            self._target_std = 1.0

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (input, target) pair as float32 tensors.

        The Xarray ``.isel`` call triggers Dask to load **only** the requested
        time-slice, keeping RAM usage proportional to one sample.
        """
        # Load one time step – triggers Dask computation for this chunk only.
        x_np: np.ndarray = (
            self.ds[self.input_var]
            .isel({self.time_dim: idx})
            .values.astype(np.float32)
        )
        y_np: np.ndarray = (
            self.ds[self.target_var]
            .isel({self.time_dim: idx})
            .values.astype(np.float32)
        )

        # Normalise
        x_np = (x_np - self._input_mean) / self._input_std
        y_np = (y_np - self._target_mean) / self._target_std

        # Add channel dimension if data are 2-D (H, W) → (1, H, W)
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, ...]
        if y_np.ndim == 2:
            y_np = y_np[np.newaxis, ...]

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)
        if self.target_transform is not None:
            y_tensor = self.target_transform(y_tensor)

        return x_tensor, y_tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def denormalize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse input normalisation."""
        return tensor * self._input_std + self._input_mean

    def denormalize_target(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse target normalisation."""
        return tensor * self._target_std + self._target_mean

    def get_coords(self):
        """Return lat/lon coordinate arrays from the underlying dataset."""
        lat = self.ds.coords.get("lat", self.ds.coords.get("latitude", None))
        lon = self.ds.coords.get("lon", self.ds.coords.get("longitude", None))
        return lat, lon

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ClimateZarrDataset("
            f"zarr_path='{self.zarr_path}', "
            f"input_var='{self.input_var}', "
            f"target_var='{self.target_var}', "
            f"n_samples={self._length}, "
            f"normalize={self.normalize})"
        )
