from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TensorDict

log = logging.getLogger(__name__)


def denormalize(tensor: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor, tp_idx: int) -> torch.Tensor:
    mu = mu.to(tensor.device)
    sig = sig.to(tensor.device)
    out = tensor * (sig + 1e-6) + mu
    out[:, tp_idx] = torch.expm1(out[:, tp_idx])
    return out


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mu: torch.Tensor,
    y_sig: torch.Tensor,
    var_names: List[str],
    tp_idx: int,
) -> Dict[str, Dict[str, float]]:
    pred_dn = denormalize(pred.clone(), y_mu, y_sig, tp_idx)
    target_dn = denormalize(target.clone(), y_mu, y_sig, tp_idx)

    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(var_names):
        p = pred_dn[:, i]
        t = target_dn[:, i]

        rmse = torch.sqrt(F.mse_loss(p, t)).item()
        mae = F.l1_loss(p, t).item()
        bias = (p - t).mean().item()

        pf = p.reshape(-1)
        tf = t.reshape(-1)
        pf_z = pf - pf.mean()
        tf_z = tf - tf.mean()
        corr = (pf_z * tf_z).sum() / (torch.sqrt((pf_z ** 2).sum() * (tf_z ** 2).sum()) + 1e-8)
        corr_v = corr.item()

        clim = t.mean(dim=(-2, -1), keepdim=True)
        p_anom = p - clim
        t_anom = t - clim
        num = (p_anom * t_anom).sum(dim=(-2, -1))
        den = torch.sqrt((p_anom ** 2).sum(dim=(-2, -1)) * (t_anom ** 2).sum(dim=(-2, -1))) + 1e-8
        acc = (num / den).mean().item()

        metrics[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias,
            "Corr": corr_v,
            "ACC": acc,
        }

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]], title: str) -> None:
    log.info(title)
    log.info("%-30s %8s %8s %8s %8s %8s", "Variable", "RMSE", "MAE", "Bias", "Corr", "ACC")
    for var, m in metrics.items():
        log.info(
            "%-30s %8.4f %8.4f %+8.4f %8.4f %8.4f",
            var,
            m["RMSE"],
            m["MAE"],
            m["Bias"],
            m["Corr"],
            m["ACC"],
        )


def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool,
) -> Tuple[float, torch.Tensor | None, torch.Tensor | None]:
    model.eval()
    val_losses: List[float] = []
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss, _, _ = criterion(pred, y_batch)
            val_losses.append(loss.item())
            if collect_predictions:
                preds.append(pred.cpu())
                targets.append(y_batch.cpu())

    pred_tensor = torch.cat(preds, dim=0) if collect_predictions and preds else None
    target_tensor = torch.cat(targets, dim=0) if collect_predictions and targets else None
    return float(np.mean(val_losses)), pred_tensor, target_tensor


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            pred = model(x_batch.to(device))
            preds.append(pred.cpu())
            targets.append(y_batch)

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def maybe_print_val_metrics(
    epoch: int,
    print_every: int,
    preds: torch.Tensor | None,
    targets: torch.Tensor | None,
    y_mu: torch.Tensor,
    y_sig: torch.Tensor,
    variables: List[str],
    tp_idx: int,
    title: str,
) -> None:
    if epoch % print_every != 0 or preds is None or targets is None:
        return
    metrics = compute_metrics(preds, targets, y_mu, y_sig, variables, tp_idx)
    print_metrics(metrics, title)


def evaluate_from_checkpoint(
    model: nn.Module,
    test_loader: DataLoader,
    arrays: TensorDict,
    variables: List[str],
    tp_idx: int,
    device: torch.device,
    ckpt_path: Path,
    model_state_key: str,
    title: str,
) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt[model_state_key])
    model = model.to(device)
    preds, targets = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, targets, arrays["Y_mean"], arrays["Y_std"], variables, tp_idx)
    print_metrics(metrics, title)
