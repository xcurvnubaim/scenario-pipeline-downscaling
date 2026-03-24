"""ResNet-based downscaling model for climate data.

A lightweight residual network that maps a low-resolution climate field to
a high-resolution output.  The network is composed of:

1. An initial convolution that expands the channel count.
2. A stack of residual blocks operating in that feature space.
3. A sub-pixel shuffle (PixelShuffle) up-sampling layer.
4. A final convolution that collapses back to ``out_channels``.

This design is inspired by ESRGAN / SRCNN-family super-resolution networks
applied to climate downscaling tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """Standard pre-activation residual block (He et al., 2016)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetDownscaler(nn.Module):
    """Residual network for spatial downscaling / super-resolution.

    Parameters
    ----------
    in_channels:
        Number of input channels (default 1).
    out_channels:
        Number of output channels (default 1).
    num_residual_blocks:
        Depth of the residual tower (default 8).
    num_features:
        Width of the residual tower (default 64).
    scale_factor:
        Integer upscaling factor.  Must be a power of two (1, 2, 4, 8 …).
        When ``scale_factor == 1`` no up-sampling is performed and the
        architecture acts as a residual correction network.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_residual_blocks: int = 8,
        num_features: int = 64,
        scale_factor: int = 4,
    ) -> None:
        super().__init__()

        self.scale_factor = scale_factor

        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.body = nn.Sequential(
            *[_ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        )

        self.body_out = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        # Sub-pixel up-sampling (PixelShuffle requires num_features * r^2 channels)
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(
                    num_features,
                    num_features * (scale_factor ** 2),
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(scale_factor),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsample = nn.Identity()

        # Output projection
        out_feat = num_features
        self.tail = nn.Conv2d(out_feat, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body_out(self.body(feat))  # residual learning
        feat = self.upsample(feat)
        return self.tail(feat)
