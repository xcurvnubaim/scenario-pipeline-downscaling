"""U-Net architecture for spatiotemporal downscaling.

The classic encoder–decoder with skip connections.  The model accepts a
single-channel (or multi-channel) low-resolution field and outputs a
field at the same spatial resolution.  A bilinear up-sampling step at
the very start optionally brings the input to the target resolution
before the U-Net processes it.

Reference: Ronneberger et al. (2015) "U-Net: Convolutional Networks for
Biomedical Image Segmentation".
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two consecutive Conv→BN→ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Max-pool then double-conv (encoder step)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    """Bilinear up-sample then concatenate skip-connection then double-conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for climate downscaling.

    Parameters
    ----------
    in_channels:
        Number of input channels (default 1 for a single climate variable).
    out_channels:
        Number of output channels (default 1).
    features:
        List of feature map sizes at each encoder level.
    scale_factor:
        Optional integer scale factor.  When > 1 the input is first
        bi-linearly up-sampled to ``scale_factor × H`` before the U-Net
        encoder runs.  Set to 1 (default) to skip the initial up-sampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = None,
        scale_factor: int = 1,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.scale_factor = scale_factor

        # Encoder
        self.inc = _DoubleConv(in_channels, features[0])
        self.down_blocks = nn.ModuleList(
            [_Down(features[i], features[i + 1]) for i in range(len(features) - 1)]
        )

        # Bottleneck
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # Decoder (reverse order)
        decoder_features = [features[-1] * 2] + list(reversed(features))
        self.up_blocks = nn.ModuleList(
            [
                _Up(decoder_features[i] + decoder_features[i + 1], decoder_features[i + 1])
                for i in range(len(decoder_features) - 1)
            ]
        )

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)

        # Encoder
        skips: List[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.outc(x)
