from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kornia
except ImportError as exc:  # pragma: no cover
    raise ImportError("kornia is required for DSSIM training. Install requirements first.") from exc


class CombinedLoss(nn.Module):
    """DSSIM-based combined loss."""

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, mode: str = "dssim") -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.dssim = kornia.losses.SSIMLoss(window_size=11, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dssim_loss = self.dssim(pred, target)

        if self.mode == "dssim":
            return dssim_loss, dssim_loss.detach(), torch.tensor(0.0, device=pred.device)

        if self.mode == "dssim+mse":
            mse = F.mse_loss(pred, target)
            total = self.alpha * dssim_loss + self.beta * mse
            return total, dssim_loss.detach(), mse.detach()

        if self.mode == "dssim+grad":
            dy_true = target[:, :, 1:, :] - target[:, :, :-1, :]
            dx_true = target[:, :, :, 1:] - target[:, :, :, :-1]
            dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            grad_loss = torch.mean(torch.abs(dy_true - dy_pred)) + torch.mean(torch.abs(dx_true - dx_pred))
            total = self.alpha * dssim_loss + self.beta * grad_loss
            return total, dssim_loss.detach(), grad_loss.detach()

        raise ValueError(f"Unknown loss mode: {self.mode}")
