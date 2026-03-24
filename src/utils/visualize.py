"""Visualization utilities for spatiotemporal climate downscaling.

All public functions return a :class:`matplotlib.figure.Figure` so that
callers can decide whether to save, display, or log them (e.g. to MLflow).

The functions are intentionally lightweight wrappers around Matplotlib /
Cartopy.  Cartopy is an optional dependency; if it is not available the
functions fall back to plain Matplotlib axes without map projections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend – safe in HPC / headless envs
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    Figure = object  # type: ignore[misc, assignment]

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _CARTOPY_AVAILABLE = True
except ImportError:
    _CARTOPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_mpl() -> None:
    if not _MPL_AVAILABLE:  # pragma: no cover
        raise ImportError("matplotlib is required for visualizations.  Install it with: pip install matplotlib")


def _make_axes(
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (8, 5),
):
    """Return *(fig, ax)*.  Uses Cartopy PlateCarree when coordinates and
    Cartopy are both available, otherwise falls back to plain axes."""
    _check_mpl()
    if _CARTOPY_AVAILABLE and lat is not None and lon is not None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_extent(
            [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
            crs=ccrs.PlateCarree(),
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_spatial_map(
    field: np.ndarray,
    title: str = "Spatial Map",
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
    figsize: Tuple[float, float] = (8, 5),
) -> "Figure":
    """Plot a 2-D spatial field.

    Parameters
    ----------
    field:
        2-D array of shape ``(H, W)``.
    title:
        Figure title.
    lat / lon:
        1-D coordinate arrays.  Used for axis labels and (optionally) Cartopy
        extent when Cartopy is installed.
    cmap:
        Matplotlib colormap name.
    vmin / vmax:
        Color-scale limits.  Defaults to the data range.
    colorbar_label:
        Label for the colorbar.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()
    fig, ax = _make_axes(lat, lon, figsize=figsize)

    if lat is not None and lon is not None:
        if _CARTOPY_AVAILABLE:
            im = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax,
                               transform=ccrs.PlateCarree())
        else:
            im = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
    else:
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")

    plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_bias_correction(
    raw_field: np.ndarray,
    corrected_field: np.ndarray,
    reference_field: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    variable_name: str = "Variable",
    figsize: Tuple[float, float] = (18, 5),
) -> "Figure":
    """Three-panel figure: raw | corrected | bias (corrected − reference).

    Parameters
    ----------
    raw_field:
        The uncorrected (e.g. GCM / low-resolution) field, shape ``(H, W)``.
    corrected_field:
        The bias-corrected or downscaled model output, shape ``(H, W)``.
    reference_field:
        The ground-truth / observational reference, shape ``(H, W)``.
    lat / lon:
        Optional 1-D coordinate arrays.
    variable_name:
        Used in sub-plot titles.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()

    bias_field = corrected_field - reference_field
    vabs = float(np.nanpercentile(np.abs(bias_field), 95))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    def _plot_panel(ax, data, title, cmap, vmin, vmax, label):
        if lat is not None and lon is not None:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
        else:
            im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_title(title)

    vmin_data = float(np.nanmin([raw_field, corrected_field, reference_field]))
    vmax_data = float(np.nanmax([raw_field, corrected_field, reference_field]))

    _plot_panel(axes[0], raw_field, f"{variable_name} – Raw", "viridis", vmin_data, vmax_data, variable_name)
    _plot_panel(axes[1], corrected_field, f"{variable_name} – Corrected", "viridis", vmin_data, vmax_data, variable_name)
    _plot_panel(axes[2], bias_field, f"Bias (Corrected − Ref)", "RdBu_r", -vabs, vabs, "Bias")

    fig.tight_layout()
    return fig


def plot_rmse_grid(
    rmse_map: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    title: str = "Spatial RMSE",
    figsize: Tuple[float, float] = (8, 5),
) -> "Figure":
    """Plot a 2-D spatial RMSE map.

    Parameters
    ----------
    rmse_map:
        2-D array of shape ``(H, W)`` containing per-pixel RMSE values.
    lat / lon:
        Optional 1-D coordinate arrays.
    title:
        Figure title.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return plot_spatial_map(
        rmse_map,
        title=title,
        lat=lat,
        lon=lon,
        cmap="YlOrRd",
        vmin=0.0,
        colorbar_label="RMSE",
        figsize=figsize,
    )


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    metric_name: str = "Loss",
    figsize: Tuple[float, float] = (8, 4),
) -> "Figure":
    """Plot training (and optionally validation) loss curves.

    Parameters
    ----------
    train_losses:
        Sequence of per-epoch training loss values.
    val_losses:
        Optional sequence of per-epoch validation loss values.
    metric_name:
        Label for the y-axis.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label=f"Train {metric_name}", linewidth=2)
    if val_losses is not None:
        ax.plot(epochs, val_losses, label=f"Val {metric_name}", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Training curves – {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig: "Figure", path: Union[str, Path], dpi: int = 150) -> Path:
    """Save a Matplotlib figure to *path* and close it.

    Parameters
    ----------
    fig:
        Figure to save.
    path:
        Output file path (extension determines format, e.g. ``.png``, ``.pdf``).
    dpi:
        Resolution in dots per inch.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    _check_mpl()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out.resolve()
