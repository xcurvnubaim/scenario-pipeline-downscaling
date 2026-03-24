"""Main training entry point.

Run with default config
-----------------------
    python train.py

Override the model
------------------
    python train.py model=resnet

Override any value
------------------
    python train.py training.epochs=50 training.learning_rate=5e-4

Hydra multirun (hyperparameter sweep)
--------------------------------------
    python train.py -m training.learning_rate=1e-3,5e-4,1e-4

The script:
1. Builds a ClimateZarrDataset from the Zarr store specified in the config.
2. Instantiates the model using hydra.utils.instantiate (Hydra) or the
   model factory as a fallback.
3. Runs the training loop with periodic validation.
4. Computes spatial evaluation metrics on the test split.
5. Generates visualizations (spatial maps, RMSE grid, training curves) and
   logs them to MLflow.
6. Saves the final model checkpoint.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Subset, random_split

from src.data.dataset import ClimateZarrDataset
from src.utils.metrics import bias, mae, rmse, spatial_bias, spatial_rmse
from src.utils.visualize import (
    plot_bias_correction,
    plot_rmse_grid,
    plot_spatial_map,
    plot_training_curves,
    save_figure,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss selection
# ---------------------------------------------------------------------------

def build_criterion(loss_name: str) -> nn.Module:
    loss_map: Dict[str, nn.Module] = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(),
    }
    name = loss_name.lower()
    if name not in loss_map:
        raise ValueError(f"Unknown loss '{loss_name}'. Choose from: {list(loss_map)}")
    return loss_map[name]


# ---------------------------------------------------------------------------
# Scheduler selection
# ---------------------------------------------------------------------------

def build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
):
    name = cfg.training.scheduler.lower()
    kwargs = OmegaConf.to_container(cfg.training.scheduler_kwargs, resolve=True)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, **kwargs)
    if name == "step":
        return StepLR(optimizer, **kwargs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'.")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, ClimateZarrDataset]:
    """Build train / val / test DataLoaders from a single Zarr dataset."""
    chunks = OmegaConf.to_container(cfg.data.chunks, resolve=True) if cfg.data.chunks else None
    dataset = ClimateZarrDataset(
        zarr_path=cfg.data.zarr_path,
        input_var=cfg.data.input_var,
        target_var=cfg.data.target_var,
        time_dim=cfg.data.time_dim,
        normalize=cfg.data.normalize,
        chunks=chunks,
    )

    n = len(dataset)
    n_test = max(1, int(n * cfg.data.test_split))
    n_val = max(1, int(n * cfg.data.val_split))
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )

    loader_kwargs = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, dataset


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (avg_loss, all_preds, all_targets) as CPU numpy arrays."""
    model.eval()
    total_loss = 0.0
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        preds_list.append(pred.cpu().numpy())
        targets_list.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_preds, all_targets


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def run_visualizations(
    cfg: DictConfig,
    train_losses: List[float],
    val_losses: List[float],
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    dataset: ClimateZarrDataset,
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate all figures and return a dict mapping name → path."""
    lat_xr, lon_xr = dataset.get_coords()
    lat = lat_xr.values if lat_xr is not None else None
    lon = lon_xr.values if lon_xr is not None else None

    figures: Dict[str, Path] = {}
    dpi: int = cfg.visualization.dpi

    # 1. Training curves
    fig_curves = plot_training_curves(
        train_losses, val_losses, metric_name=cfg.training.loss.upper()
    )
    figures["training_curves"] = save_figure(fig_curves, output_dir / "training_curves.png", dpi=dpi)

    # 2. Example prediction vs target (first sample in test set)
    pred_sample = all_preds[0, 0]   # (H, W)
    target_sample = all_targets[0, 0]

    fig_prediction = plot_spatial_map(
        pred_sample, title="Prediction (sample 0)", lat=lat, lon=lon,
        colorbar_label=cfg.data.target_var
    )
    figures["prediction_sample"] = save_figure(fig_prediction, output_dir / "prediction_sample.png", dpi=dpi)

    fig_target = plot_spatial_map(
        target_sample, title="Ground Truth (sample 0)", lat=lat, lon=lon,
        colorbar_label=cfg.data.target_var
    )
    figures["target_sample"] = save_figure(fig_target, output_dir / "target_sample.png", dpi=dpi)

    # 3. Bias-correction panel (raw=pred, corrected=pred, reference=target)
    # In practice you would pass the raw GCM field; here we re-use the prediction
    # as a placeholder since the raw field is not stored separately.
    fig_bias = plot_bias_correction(
        raw_field=pred_sample,
        corrected_field=pred_sample,
        reference_field=target_sample,
        lat=lat,
        lon=lon,
        variable_name=cfg.data.target_var,
    )
    figures["bias_correction"] = save_figure(fig_bias, output_dir / "bias_correction.png", dpi=dpi)

    # 4. Spatial RMSE grid
    rmse_map = spatial_rmse(all_preds, all_targets)
    fig_rmse = plot_rmse_grid(rmse_map, lat=lat, lon=lon)
    figures["rmse_grid"] = save_figure(fig_rmse, output_dir / "rmse_grid.png", dpi=dpi)

    return figures


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---- Setup ----------------------------------------------------------- #
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Hydra changes CWD to the output dir when job.chdir=true.
    output_dir = Path.cwd()
    vis_dir = output_dir / cfg.visualization.output_dir
    ckpt_dir = output_dir / cfg.training.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ---- MLflow ---------------------------------------------------------- #
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name: Optional[str] = cfg.mlflow.run_name if cfg.mlflow.run_name else None

    with mlflow.start_run(run_name=run_name) as run:
        # Log the full resolved config as a flat dict
        flat_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        mlflow.log_params(_flatten_dict(flat_cfg))

        # ---- Data -------------------------------------------------------- #
        log.info("Loading data from %s …", cfg.data.zarr_path)
        train_loader, val_loader, test_loader, dataset = build_dataloaders(cfg)
        log.info(
            "Dataset split → train: %d, val: %d, test: %d",
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )

        # ---- Model ------------------------------------------------------- #
        log.info("Instantiating model …")
        model: nn.Module = instantiate(cfg.model)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Model: %s | trainable parameters: %s", model.__class__.__name__, f"{n_params:,}")
        mlflow.log_param("model_class", model.__class__.__name__)
        mlflow.log_param("trainable_params", n_params)

        # ---- Optimiser & scheduler --------------------------------------- #
        optimizer = Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        scheduler = build_scheduler(cfg, optimizer)
        criterion = build_criterion(cfg.training.loss)

        # ---- Training loop ----------------------------------------------- #
        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                gradient_clip=cfg.training.gradient_clip,
            )
            train_losses.append(train_loss)

            # Validation
            if epoch % cfg.training.val_every_n_epochs == 0 or epoch == cfg.training.epochs:
                val_loss, _, _ = evaluate(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                elapsed = time.time() - t0
                log.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %.1fs",
                    epoch, cfg.training.epochs, train_loss, val_loss, elapsed,
                )
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=epoch,
                )

                # Checkpoint best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = ckpt_dir / "best_model.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    log.info("  ↳ saved best checkpoint → %s", ckpt_path)
            else:
                if epoch % cfg.training.log_every_n_steps == 0:
                    log.info(
                        "Epoch %d/%d | train_loss=%.4f",
                        epoch, cfg.training.epochs, train_loss,
                    )
                mlflow.log_metric("train_loss", train_loss, step=epoch)

            if scheduler is not None:
                scheduler.step()

        # Save final model
        final_ckpt = ckpt_dir / "final_model.pt"
        torch.save(model.state_dict(), final_ckpt)
        mlflow.log_artifact(str(final_ckpt), artifact_path="checkpoints")
        log.info("Saved final model → %s", final_ckpt)

        # ---- Test evaluation --------------------------------------------- #
        log.info("Running test evaluation …")
        test_loss, all_preds, all_targets = evaluate(model, test_loader, criterion, device)
        test_rmse = rmse(all_preds, all_targets)
        test_mae = mae(all_preds, all_targets)
        test_bias = bias(all_preds, all_targets)

        log.info(
            "Test | loss=%.4f | RMSE=%.4f | MAE=%.4f | bias=%.4f",
            test_loss, test_rmse, test_mae, test_bias,
        )
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_bias": test_bias,
            }
        )

        # ---- Visualizations ---------------------------------------------- #
        if cfg.visualization.enabled:
            log.info("Generating visualizations …")
            figs = run_visualizations(
                cfg, train_losses, val_losses, all_preds, all_targets, dataset, vis_dir
            )
            if cfg.visualization.log_to_mlflow:
                for name, path in figs.items():
                    mlflow.log_artifact(str(path), artifact_path="figures")
                    log.info("  ↳ logged figure '%s' to MLflow", name)

        log.info("Run complete.  MLflow run ID: %s", run.info.run_id)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict for MLflow param logging."""
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            # MLflow params must be strings and ≤ 500 chars
            items[new_key] = str(v)[:500]
    return items


if __name__ == "__main__":
    main()
