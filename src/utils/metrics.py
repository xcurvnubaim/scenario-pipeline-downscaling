"""Spatiotemporal metrics for climate downscaling evaluation.

All functions accept either NumPy arrays or PyTorch tensors as inputs and
return scalar float values.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

_ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: _ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(predictions: _ArrayLike, targets: _ArrayLike) -> float:
    """Root-Mean-Square Error.

    Parameters
    ----------
    predictions:
        Model predictions, shape ``(...)``.
    targets:
        Ground-truth values, same shape as *predictions*.

    Returns
    -------
    float
        Scalar RMSE.
    """
    p = _to_numpy(predictions).astype(np.float64)
    t = _to_numpy(targets).astype(np.float64)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(predictions: _ArrayLike, targets: _ArrayLike) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    predictions:
        Model predictions, shape ``(...)``.
    targets:
        Ground-truth values, same shape as *predictions*.

    Returns
    -------
    float
        Scalar MAE.
    """
    p = _to_numpy(predictions).astype(np.float64)
    t = _to_numpy(targets).astype(np.float64)
    return float(np.mean(np.abs(p - t)))


def bias(predictions: _ArrayLike, targets: _ArrayLike) -> float:
    """Mean bias (signed).

    Positive values indicate that *predictions* are systematically higher than
    *targets* (warm / wet bias); negative values indicate under-prediction.

    Parameters
    ----------
    predictions:
        Model predictions, shape ``(...)``.
    targets:
        Ground-truth values, same shape as *predictions*.

    Returns
    -------
    float
        Scalar mean bias = mean(predictions − targets).
    """
    p = _to_numpy(predictions).astype(np.float64)
    t = _to_numpy(targets).astype(np.float64)
    return float(np.mean(p - t))


def spatial_rmse(predictions: _ArrayLike, targets: _ArrayLike) -> np.ndarray:
    """Pixel-wise RMSE averaged over the batch / time dimension.

    Parameters
    ----------
    predictions:
        Array of shape ``(T, H, W)`` or ``(T, C, H, W)``.
    targets:
        Ground-truth, same shape as *predictions*.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(H, W)`` containing per-pixel RMSE.
    """
    p = _to_numpy(predictions).astype(np.float64)
    t = _to_numpy(targets).astype(np.float64)
    # Collapse leading dims except the last two
    if p.ndim == 4:
        p = p[:, 0, ...]  # first channel only
        t = t[:, 0, ...]
    return np.sqrt(np.mean((p - t) ** 2, axis=0))


def spatial_bias(predictions: _ArrayLike, targets: _ArrayLike) -> np.ndarray:
    """Pixel-wise mean bias averaged over the batch / time dimension.

    Parameters
    ----------
    predictions:
        Array of shape ``(T, H, W)`` or ``(T, C, H, W)``.
    targets:
        Ground-truth, same shape as *predictions*.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(H, W)`` containing per-pixel mean bias.
    """
    p = _to_numpy(predictions).astype(np.float64)
    t = _to_numpy(targets).astype(np.float64)
    if p.ndim == 4:
        p = p[:, 0, ...]
        t = t[:, 0, ...]
    return np.mean(p - t, axis=0)
