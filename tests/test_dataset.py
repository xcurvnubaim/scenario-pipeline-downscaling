"""Tests for ClimateZarrDataset."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.data.dataset import ClimateZarrDataset


@pytest.fixture()
def zarr_store(tmp_path: Path) -> Path:
    """Create a minimal synthetic Zarr store for testing."""
    T, H, W = 20, 16, 16
    rng = np.random.default_rng(0)

    ds = xr.Dataset(
        {
            "lr_field": (["time", "lat", "lon"], rng.standard_normal((T, H, W)).astype("float32")),
            "hr_field": (["time", "lat", "lon"], rng.standard_normal((T, H, W)).astype("float32")),
        },
        coords={
            "time": np.arange(T),
            "lat": np.linspace(-10, 10, H),
            "lon": np.linspace(100, 120, W),
        },
    )
    store_path = tmp_path / "test.zarr"
    ds.to_zarr(str(store_path))
    return store_path


def test_dataset_length(zarr_store: Path) -> None:
    dataset = ClimateZarrDataset(zarr_store, "lr_field", "hr_field", normalize=False)
    assert len(dataset) == 20


def test_dataset_item_shape(zarr_store: Path) -> None:
    dataset = ClimateZarrDataset(zarr_store, "lr_field", "hr_field", normalize=False)
    x, y = dataset[0]
    assert x.shape == (1, 16, 16)
    assert y.shape == (1, 16, 16)


def test_dataset_normalize(zarr_store: Path) -> None:
    """With normalisation enabled the statistics should be close to 0-mean / 1-std."""
    import torch
    dataset = ClimateZarrDataset(zarr_store, "lr_field", "hr_field", normalize=True)
    xs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    assert abs(float(xs.mean())) < 0.5
    assert abs(float(xs.std()) - 1.0) < 0.5


def test_dataset_denormalize(zarr_store: Path) -> None:
    dataset = ClimateZarrDataset(zarr_store, "lr_field", "hr_field", normalize=True)
    x, y = dataset[0]
    # Round-trip: normalise then de-normalise should be close to the original raw value
    raw_x = float(
        xr.open_zarr(str(zarr_store))["lr_field"].isel(time=0).values[0, 0]
    )
    denorm_x = float(dataset.denormalize_input(x)[0, 0, 0])
    assert abs(raw_x - denorm_x) < 1e-4


def test_dataset_missing_var(zarr_store: Path) -> None:
    with pytest.raises(KeyError):
        ClimateZarrDataset(zarr_store, "nonexistent_var", "hr_field", normalize=False)


def test_dataset_get_coords(zarr_store: Path) -> None:
    dataset = ClimateZarrDataset(zarr_store, "lr_field", "hr_field", normalize=False)
    lat, lon = dataset.get_coords()
    assert lat is not None
    assert lon is not None
    assert len(lat) == 16
    assert len(lon) == 16
