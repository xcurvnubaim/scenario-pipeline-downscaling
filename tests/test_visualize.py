"""Tests for visualization utilities."""

import numpy as np
import pytest

from src.utils.visualize import (
    plot_bias_correction,
    plot_rmse_grid,
    plot_spatial_map,
    plot_training_curves,
    save_figure,
)


@pytest.fixture()
def sample_field() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((16, 16)).astype("float32")


@pytest.fixture()
def sample_coords():
    lat = np.linspace(-10, 10, 16)
    lon = np.linspace(100, 120, 16)
    return lat, lon


def test_plot_spatial_map_returns_figure(sample_field):
    import matplotlib.figure
    fig = plot_spatial_map(sample_field, title="Test")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_spatial_map_with_coords(sample_field, sample_coords):
    import matplotlib.figure
    lat, lon = sample_coords
    fig = plot_spatial_map(sample_field, lat=lat, lon=lon, colorbar_label="K")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_bias_correction(sample_field):
    import matplotlib.figure
    fig = plot_bias_correction(sample_field, sample_field * 0.9, sample_field * 0.8)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_rmse_grid(sample_field):
    import matplotlib.figure
    # RMSE map must be non-negative
    rmse_map = np.abs(sample_field)
    fig = plot_rmse_grid(rmse_map)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_training_curves():
    import matplotlib.figure
    train_losses = [1.0, 0.8, 0.6, 0.5]
    val_losses = [1.1, 0.9, 0.7, 0.55]
    fig = plot_training_curves(train_losses, val_losses)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_training_curves_no_val():
    import matplotlib.figure
    fig = plot_training_curves([1.0, 0.9, 0.8])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_save_figure(tmp_path, sample_field):
    fig = plot_spatial_map(sample_field)
    out_path = save_figure(fig, tmp_path / "test_figure.png")
    assert out_path.exists()
    assert out_path.suffix == ".png"
