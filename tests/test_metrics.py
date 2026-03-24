"""Tests for spatiotemporal metrics."""

import numpy as np
import pytest
import torch

from src.utils.metrics import bias, mae, rmse, spatial_bias, spatial_rmse


def test_rmse_identical():
    x = np.ones((4, 4))
    assert rmse(x, x) == pytest.approx(0.0)


def test_rmse_known():
    pred = np.array([3.0, 4.0])
    tgt = np.array([0.0, 0.0])
    # sqrt(mean([9, 16])) = sqrt(12.5) ≈ 3.5355
    assert rmse(pred, tgt) == pytest.approx(np.sqrt(12.5), abs=1e-6)


def test_rmse_with_tensors():
    pred = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.0, 2.0, 3.0])
    assert rmse(pred, tgt) == pytest.approx(0.0)


def test_mae_known():
    pred = np.array([1.0, 3.0])
    tgt = np.array([0.0, 0.0])
    assert mae(pred, tgt) == pytest.approx(2.0, abs=1e-6)


def test_bias_sign():
    pred = np.array([2.0, 4.0])
    tgt = np.array([1.0, 2.0])
    assert bias(pred, tgt) > 0  # warm bias


def test_bias_known():
    pred = np.array([3.0, 5.0])
    tgt = np.array([1.0, 1.0])
    assert bias(pred, tgt) == pytest.approx(3.0, abs=1e-6)


def test_spatial_rmse_shape():
    rng = np.random.default_rng(42)
    T, H, W = 10, 8, 8
    pred = rng.standard_normal((T, H, W))
    tgt = rng.standard_normal((T, H, W))
    result = spatial_rmse(pred, tgt)
    assert result.shape == (H, W)
    assert np.all(result >= 0)


def test_spatial_bias_shape():
    rng = np.random.default_rng(42)
    T, H, W = 10, 8, 8
    pred = rng.standard_normal((T, H, W))
    tgt = rng.standard_normal((T, H, W))
    result = spatial_bias(pred, tgt)
    assert result.shape == (H, W)


def test_spatial_rmse_4d_input():
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((5, 1, 8, 8))  # (T, C, H, W)
    tgt = rng.standard_normal((5, 1, 8, 8))
    result = spatial_rmse(pred, tgt)
    assert result.shape == (8, 8)
