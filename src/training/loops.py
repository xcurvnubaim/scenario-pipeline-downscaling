from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.gan import Discriminator

from .data import TensorDict
from .evaluation import evaluate_from_checkpoint, maybe_print_val_metrics, run_validation
from .losses import CombinedLoss

log = logging.getLogger(__name__)


def train_supervised(
    cfg: DictConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    criterion = CombinedLoss(
        alpha=cfg.training.dssim_alpha,
        beta=cfg.training.dssim_beta,
        mode=cfg.training.loss_mode,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    y_mu = arrays["Y_mean"]
    y_sig = arrays["Y_std"]

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss, _, _ = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradient_clip)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        tr = float(np.mean(train_losses))
        should_collect = epoch % cfg.evaluation.print_every_n_epochs == 0
        va, val_preds, val_targets = run_validation(
            model,
            val_loader,
            criterion,
            device,
            collect_predictions=should_collect,
        )
        log.info("Epoch [%03d/%03d] train=%.4f val=%.4f", epoch, cfg.training.epochs, tr, va)

        maybe_print_val_metrics(
            epoch=epoch,
            print_every=cfg.evaluation.print_every_n_epochs,
            preds=val_preds,
            targets=val_targets,
            y_mu=y_mu,
            y_sig=y_sig,
            variables=variables,
            tp_idx=tp_idx,
            title=f"Validation metrics at epoch {epoch}",
        )

        if va < best_val:
            best_val = va
            patience_ctr = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": va,
                    "X_mean": arrays["X_mean"],
                    "X_std": arrays["X_std"],
                    "Y_mean": arrays["Y_mean"],
                    "Y_std": arrays["Y_std"],
                    "variables": variables,
                    "tp_index": tp_idx,
                    "log1p_applied": True,
                },
                ckpt_path,
            )
            log.info("Saved best model to %s", ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.training.patience:
                log.info("Early stopping at epoch %d", epoch)
                break


def evaluate_supervised(
    model: nn.Module,
    test_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    evaluate_from_checkpoint(
        model=model,
        test_loader=test_loader,
        arrays=arrays,
        variables=variables,
        tp_idx=tp_idx,
        device=device,
        ckpt_path=ckpt_path,
        model_state_key="model_state",
        title="Test metrics",
    )


def train_gan(
    cfg: DictConfig,
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    content_criterion = CombinedLoss(
        alpha=cfg.training.dssim_alpha,
        beta=cfg.training.dssim_beta,
        mode=cfg.training.loss_mode,
    ).to(device)
    adv_criterion = nn.BCELoss()

    optimizer_g = AdamW(
        generator.parameters(),
        lr=cfg.training.gan.learning_rate_g,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer_d = AdamW(
        discriminator.parameters(),
        lr=cfg.training.gan.learning_rate_d,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=cfg.training.epochs, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=cfg.training.epochs, eta_min=1e-6)

    y_mu = arrays["Y_mean"]
    y_sig = arrays["Y_std"]

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, cfg.training.epochs + 1):
        generator.train()
        discriminator.train()
        loss_g_epoch = []
        loss_d_epoch = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            bsz = x_batch.size(0)
            real_lbl = torch.ones(bsz, 1, device=device)
            fake_lbl = torch.zeros(bsz, 1, device=device)

            optimizer_d.zero_grad()
            real_out = discriminator(y_batch)
            d_real = adv_criterion(real_out, real_lbl)
            fake_img = generator(x_batch).detach()
            fake_out = discriminator(fake_img)
            d_fake = adv_criterion(fake_out, fake_lbl)
            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=cfg.training.gradient_clip)
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_img = generator(x_batch)
            content_loss, _, _ = content_criterion(fake_img, y_batch)
            fake_out = discriminator(fake_img)
            g_adv = adv_criterion(fake_out, real_lbl)
            g_loss = cfg.training.gan.lambda_per * content_loss + cfg.training.gan.lambda_adv * g_adv
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg.training.gradient_clip)
            optimizer_g.step()

            loss_g_epoch.append(g_loss.item())
            loss_d_epoch.append(d_loss.item())

        scheduler_g.step()
        scheduler_d.step()

        g_mean = float(np.mean(loss_g_epoch))
        d_mean = float(np.mean(loss_d_epoch))
        should_collect = epoch % cfg.evaluation.print_every_n_epochs == 0
        v_mean, val_preds, val_targets = run_validation(
            generator,
            val_loader,
            content_criterion,
            device,
            collect_predictions=should_collect,
        )
        log.info("Epoch [%03d/%03d] G=%.4f D=%.4f val=%.4f", epoch, cfg.training.epochs, g_mean, d_mean, v_mean)

        maybe_print_val_metrics(
            epoch=epoch,
            print_every=cfg.evaluation.print_every_n_epochs,
            preds=val_preds,
            targets=val_targets,
            y_mu=y_mu,
            y_sig=y_sig,
            variables=variables,
            tp_idx=tp_idx,
            title=f"GAN validation metrics at epoch {epoch}",
        )

        if v_mean < best_val:
            best_val = v_mean
            patience_ctr = 0
            torch.save(
                {
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "epoch": epoch,
                    "val_loss": v_mean,
                    "X_mean": arrays["X_mean"],
                    "X_std": arrays["X_std"],
                    "Y_mean": arrays["Y_mean"],
                    "Y_std": arrays["Y_std"],
                    "variables": variables,
                    "tp_index": tp_idx,
                    "log1p_applied": True,
                },
                ckpt_path,
            )
            log.info("Saved best GAN model to %s", ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.training.patience:
                log.info("GAN early stopping at epoch %d", epoch)
                break


def evaluate_gan(
    generator: nn.Module,
    test_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    evaluate_from_checkpoint(
        model=generator,
        test_loader=test_loader,
        arrays=arrays,
        variables=variables,
        tp_idx=tp_idx,
        device=device,
        ckpt_path=ckpt_path,
        model_state_key="generator_state",
        title="GAN test metrics",
    )


def run_supervised_mode(
    cfg: DictConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_dir: Path,
) -> None:
    ckpt_path = ckpt_dir / cfg.training.best_model_name
    train_supervised(
        cfg,
        model,
        train_loader,
        val_loader,
        arrays,
        variables,
        tp_idx,
        device,
        ckpt_path,
    )
    evaluate_supervised(model, test_loader, arrays, variables, tp_idx, device, ckpt_path)


def run_gan_mode(
    cfg: DictConfig,
    generator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_dir: Path,
) -> None:
    discriminator = Discriminator(
        in_channels=len(variables),
        base_channels=cfg.training.gan.discriminator_base_channels,
    ).to(device)
    ckpt_path = ckpt_dir / cfg.training.gan.best_model_name
    train_gan(
        cfg,
        generator,
        discriminator,
        train_loader,
        val_loader,
        arrays,
        variables,
        tp_idx,
        device,
        ckpt_path,
    )
    evaluate_gan(generator, test_loader, arrays, variables, tp_idx, device, ckpt_path)
