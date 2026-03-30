from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

TensorDict = Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_coord_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise ValueError("Dataset must include latitude/longitude coordinates.")
    return lat_name, lon_name


def check_nan_summary(ds: xr.Dataset, name: str) -> None:
    log.info("NaN summary for %s", name)
    for var in ds.data_vars:
        arr = ds[var].values
        total = arr.size
        n_nan = int(np.isnan(arr).sum())
        pct = 100.0 * n_nan / total
        log.info("  %s: NaNs=%d (%.5f%%)", var, n_nan, pct)


def crop_to_truth_aligned_domain(
    ds_forecast: xr.Dataset,
    ds_truth: xr.Dataset,
    scale: int,
    low_lat: int,
    low_lon: int,
) -> Tuple[xr.Dataset, xr.Dataset]:
    lat_name_f, lon_name_f = _find_coord_names(ds_forecast)
    lat_name_t, lon_name_t = _find_coord_names(ds_truth)

    ds_forecast = ds_forecast.sortby(lat_name_f)
    ds_truth = ds_truth.sortby(lat_name_t)

    tr_lons = ds_truth[lon_name_t].values
    tr_lats = ds_truth[lat_name_t].values
    fc_lons = ds_forecast[lon_name_f].values
    fc_lats = ds_forecast[lat_name_f].values

    valid_lons = fc_lons[(fc_lons >= tr_lons.min()) & (fc_lons <= tr_lons.max())]
    valid_lats = fc_lats[(fc_lats >= tr_lats.min()) & (fc_lats <= tr_lats.max())]
    if len(valid_lons) == 0 or len(valid_lats) == 0:
        raise ValueError("Forecast and truth domains do not overlap.")

    lon_start = valid_lons[0]
    lat_start = valid_lats[0]

    lon_start_idx_t = int(np.argmin(np.abs(tr_lons - lon_start)))
    lat_start_idx_t = int(np.argmin(np.abs(tr_lats - lat_start)))

    avail_lon = len(tr_lons) - lon_start_idx_t
    avail_lat = len(tr_lats) - lat_start_idx_t

    max_fc_lon = min(avail_lon // scale, low_lon)
    max_fc_lat = min(avail_lat // scale, low_lat)

    lon_start_idx_f = int(np.argmin(np.abs(fc_lons - lon_start)))
    lat_start_idx_f = int(np.argmin(np.abs(fc_lats - lat_start)))

    ds_fc = ds_forecast.isel(
        {
            lon_name_f: slice(lon_start_idx_f, lon_start_idx_f + max_fc_lon),
            lat_name_f: slice(lat_start_idx_f, lat_start_idx_f + max_fc_lat),
        }
    )

    ds_tr = ds_truth.isel(
        {
            lon_name_t: slice(lon_start_idx_t, lon_start_idx_t + max_fc_lon * scale),
            lat_name_t: slice(lat_start_idx_t, lat_start_idx_t + max_fc_lat * scale),
        }
    )

    lon_ok = np.allclose(ds_tr[lon_name_t].values[::scale], ds_fc[lon_name_f].values, atol=1e-3)
    lat_ok = np.allclose(ds_tr[lat_name_t].values[::scale], ds_fc[lat_name_f].values, atol=1e-3)
    if not (lon_ok and lat_ok):
        raise ValueError("Failed to produce perfect low/high-resolution coordinate alignment.")

    return ds_fc, ds_tr


def collect_fully_missing_times(ds: xr.Dataset) -> np.ndarray:
    missing = set()
    for var in ds.data_vars:
        data = ds[var]
        reduce_dims = [d for d in data.dims if d != "time"]
        # Xarray cannot boolean-index with lazy dask arrays of unknown shape.
        # Materialize the mask first, then use it for time selection.
        miss_mask = data.isnull().all(dim=reduce_dims).compute()
        miss_times = ds.time.where(miss_mask, drop=True).values
        for t in miss_times:
            missing.add(t)
    return np.array(sorted(missing))


def temporal_align(
    ds_fc: xr.Dataset,
    ds_tr: xr.Dataset,
    lead_days: int,
) -> Tuple[xr.Dataset, xr.Dataset]:
    lead_td = np.timedelta64(lead_days, "D")
    ds_fc_lead = ds_fc.sel(prediction_timedelta=lead_td)
    valid_time = ds_fc_lead.time + lead_td
    common_times = np.intersect1d(valid_time.values, ds_tr.time.values)
    ds_fc_lead = ds_fc_lead.assign_coords(valid_time=valid_time)
    ds_fc_lead = ds_fc_lead.sel(valid_time=common_times)
    ds_fc_lead = ds_fc_lead.assign_coords(time=ds_fc_lead.valid_time).drop_vars("valid_time")
    ds_tr_aligned = ds_tr.sel(time=common_times)

    if not np.array_equal(ds_fc_lead.time.values, ds_tr_aligned.time.values):
        raise ValueError("Forecast and truth times are not aligned after lead-time processing.")

    return ds_fc_lead, ds_tr_aligned


def build_arrays(ds_fc: xr.Dataset, ds_tr: xr.Dataset, variables: List[str], tp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.stack([ds_fc[v].values for v in variables], axis=1).astype(np.float32)
    y = np.stack([ds_tr[v].values for v in variables], axis=1).astype(np.float32)

    expected_h = x.shape[2] * 6
    expected_w = x.shape[3] * 6
    if y.shape[2] == expected_w and y.shape[3] == expected_h:
        y = np.transpose(y, (0, 1, 3, 2))

    x[:, tp_idx] = np.log1p(np.clip(x[:, tp_idx], 0, None))
    y[:, tp_idx] = np.log1p(np.clip(y[:, tp_idx], 0, None))
    return x, y


def split_and_normalize(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> TensorDict:
    n = x.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]

    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]

    x_mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    x_std = x_train.std(axis=(0, 2, 3), keepdims=True)
    y_mean = y_train.mean(axis=(0, 2, 3), keepdims=True)
    y_std = y_train.std(axis=(0, 2, 3), keepdims=True)

    x_train = (x_train - x_mean) / (x_std + 1e-6)
    x_val = (x_val - x_mean) / (x_std + 1e-6)
    x_test = (x_test - x_mean) / (x_std + 1e-6)

    y_train = (y_train - y_mean) / (y_std + 1e-6)
    y_val = (y_val - y_mean) / (y_std + 1e-6)
    y_test = (y_test - y_mean) / (y_std + 1e-6)

    return {
        "X_train": torch.tensor(x_train, dtype=torch.float32),
        "X_val": torch.tensor(x_val, dtype=torch.float32),
        "X_test": torch.tensor(x_test, dtype=torch.float32),
        "Y_train": torch.tensor(y_train, dtype=torch.float32),
        "Y_val": torch.tensor(y_val, dtype=torch.float32),
        "Y_test": torch.tensor(y_test, dtype=torch.float32),
        "X_mean": torch.tensor(x_mean, dtype=torch.float32),
        "X_std": torch.tensor(x_std, dtype=torch.float32),
        "Y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "Y_std": torch.tensor(y_std, dtype=torch.float32),
    }


def build_dataloaders(cfg: DictConfig, arrays: TensorDict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        TensorDataset(arrays["X_train"], arrays["Y_train"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TensorDataset(arrays["X_val"], arrays["Y_val"]),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TensorDataset(arrays["X_test"], arrays["Y_test"]),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def prepare_training_data(cfg: DictConfig) -> Tuple[TensorDict, DataLoader, DataLoader, DataLoader, List[str], int]:
    log.info("Loading forecast and truth datasets")
    ds_forecast = xr.open_zarr(cfg.data.forecast_zarr_path)
    ds_truth = xr.open_zarr(cfg.data.truth_zarr_path)

    ds_fc, ds_tr = crop_to_truth_aligned_domain(
        ds_forecast,
        ds_truth,
        scale=cfg.data.scale,
        low_lat=cfg.data.low_lat,
        low_lon=cfg.data.low_lon,
    )

    check_nan_summary(ds_fc, "Forecast (before cleanup)")
    if cfg.data.drop_lead0:
        ds_fc = ds_fc.sel(prediction_timedelta=slice(np.timedelta64(1, "D"), None))

    missing_times = collect_fully_missing_times(ds_fc)
    if len(missing_times) > 0:
        ds_fc = ds_fc.drop_sel(time=missing_times)
        ds_tr = ds_tr.drop_sel(time=missing_times)
        log.info("Dropped %d fully missing timesteps", len(missing_times))

    ds_fc_aligned, ds_tr_aligned = temporal_align(ds_fc, ds_tr, lead_days=cfg.data.lead_days)

    variables = list(cfg.data.variables)
    tp_idx = variables.index(cfg.data.tp_var_name)

    x, y = build_arrays(ds_fc_aligned, ds_tr_aligned, variables, tp_idx)
    arrays = split_and_normalize(x, y, cfg.data.train_ratio, cfg.data.val_ratio)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, arrays)
    return arrays, train_loader, val_loader, test_loader, variables, tp_idx
