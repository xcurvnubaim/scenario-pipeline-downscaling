"""GAN models for 6x climate super-resolution.

This module provides an RRDB-based generator and a PatchGAN-style
classifier discriminator for adversarial super-resolution training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """Single dense layer used inside residual dense blocks."""

    def __init__(self, in_channels: int, growth_rate: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv(x))
        return torch.cat([x, out], dim=1)


class ResidualDenseBlock(nn.Module):
    """Residual dense block with local feature fusion."""

    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 5) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)
        self.local_fusion = nn.Conv2d(
            in_channels + num_layers * growth_rate,
            in_channels,
            kernel_size=1,
            bias=True,
        )
        self.beta = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for layer in self.layers:
            x = layer(x)
        x = self.local_fusion(x)
        return identity + self.beta * x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, in_channels: int, growth_rate: int = 32, num_rdb: int = 3) -> None:
        super().__init__()
        self.rdb_blocks = nn.ModuleList(
            [ResidualDenseBlock(in_channels, growth_rate=growth_rate) for _ in range(num_rdb)]
        )
        self.beta = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for block in self.rdb_blocks:
            x = block(x)
        return identity + self.beta * x


class RRDBGenerator(nn.Module):
    """RRDB generator for 6x super-resolution.

    Input shape: (B, C, 24, 32)
    Output shape: (B, C, 144, 192)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        num_rrdb: int = 4,
        growth_rate: int = 32,
    ) -> None:
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=True)
        self.rrdb_blocks = nn.ModuleList(
            [RRDB(base_channels, growth_rate=growth_rate) for _ in range(num_rrdb)]
        )
        self.conv_body = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_init = self.conv_first(x)
        feat = feat_init
        for block in self.rrdb_blocks:
            feat = block(feat)
        feat = self.conv_body(feat)
        feat = feat + feat_init
        feat = self.up1(feat)
        feat = self.up2(feat)
        return self.conv_last(feat)


class DiscriminatorBlock(nn.Module):
    """Conv + optional BN + LeakyReLU block used by the discriminator."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_bn: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """PatchGAN-style discriminator for HR fields (144x192)."""

    def __init__(self, in_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()

        self.conv1 = DiscriminatorBlock(in_channels, base_channels, stride=1, use_bn=False)
        self.conv2 = DiscriminatorBlock(base_channels, base_channels, stride=2)
        self.conv3 = DiscriminatorBlock(base_channels, base_channels * 2, stride=1)
        self.conv4 = DiscriminatorBlock(base_channels * 2, base_channels * 2, stride=2)
        self.conv5 = DiscriminatorBlock(base_channels * 2, base_channels * 4, stride=1)
        self.conv6 = DiscriminatorBlock(base_channels * 4, base_channels * 4, stride=2)
        self.conv7 = DiscriminatorBlock(base_channels * 4, base_channels * 8, stride=1)
        self.conv8 = DiscriminatorBlock(base_channels * 8, base_channels * 8, stride=2)

        self.dense1 = nn.Sequential(
            nn.Linear(base_channels * 8 * 9 * 12, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        return self.dense2(x)
