"""Notebook-matching training entry point for scenario downscaling.

This script mirrors the workflow in the attached notebook:
1. Load low-resolution forecast and high-resolution truth Zarr datasets.
2. Crop and align domains to exact 6x scaling (24x32 -> 144x192 by default).
3. Remove invalid lead/timesteps and align forecast-valid time with truth time.
4. Build multi-variable tensors, apply log1p to precipitation, split 70/15/15.
5. Train either:
   - Supervised SR model with DSSIM-based loss, or
   - GAN (RRDB generator + discriminator) with content + adversarial losses.
6. Evaluate on test split with denormalized climate metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.training.data import prepare_training_data, set_seed
from src.training.loops import run_gan_mode, run_supervised_mode

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    arrays, train_loader, val_loader, test_loader, variables, tp_idx = prepare_training_data(cfg)

    model: nn.Module = instantiate(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s (%s trainable params)", model.__class__.__name__, f"{n_params:,}")

    if cfg.training.mode == "supervised":
        run_supervised_mode(
            cfg,
            model,
            train_loader,
            val_loader,
            test_loader,
            arrays,
            variables,
            tp_idx,
            device,
            ckpt_dir,
        )
        return

    if cfg.training.mode == "gan":
        run_gan_mode(
            cfg,
            model,
            train_loader,
            val_loader,
            test_loader,
            arrays,
            variables,
            tp_idx,
            device,
            ckpt_dir,
        )
        return

    raise ValueError("training.mode must be 'supervised' or 'gan'.")


if __name__ == "__main__":
    main()
