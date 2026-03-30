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


# ==========================================
# Building Blocks
# ==========================================

class _DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    """MaxPool → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    """Upsample → concat skip → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims mismatch
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        x  = F.pad(x, [dx // 2, dx - dx // 2,
                        dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UpscaleBlock(nn.Module):
    """
    6× upscale via three 2× bilinear steps + refinement conv.
    Input  : (B, C, 24, 32)   — low-res encoder output
    Output : (B, C, 144, 192) — high-res prediction
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up1   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch,      in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.up2   = nn.Sequential(
            nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch // 2, in_ch // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Conv2d(in_ch // 4, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)    # (B, C//2, 48,  64)
        x = self.up2(x)    # (B, C//4, 144, 192)
        return self.refine(x)


# ==========================================
# U-Net + 6× Super-Resolution Head
# ==========================================

class UNet(nn.Module):
    """
    Input  : (B, 4, 24, 32)   — low-res forecast
    Output : (B, 4, 144, 192) — high-res prediction
    """
    def __init__(self, in_ch=4, out_ch=4, base_ch=64):
        super().__init__()

        # ── Encoder ──────────────────────────────
        self.enc1 = _DoubleConv(in_ch,       base_ch)      # (B, 64,  24, 32)
        self.enc2 = _Down(base_ch,           base_ch * 2)  # (B, 128, 12, 16)
        self.enc3 = _Down(base_ch * 2,       base_ch * 4)  # (B, 256,  6,  8)

        # ── Bottleneck ───────────────────────────
        self.bottleneck = _Down(base_ch * 4, base_ch * 8)  # (B, 512,  3,  4)

        # ── Decoder ──────────────────────────────
        self.dec3 = _Up(base_ch * 8,  base_ch * 4)         # (B, 256,  6,  8)
        self.dec2 = _Up(base_ch * 4,  base_ch * 2)         # (B, 128, 12, 16)
        self.dec1 = _Up(base_ch * 2,  base_ch)             # (B, 64,  24, 32)

        # ── 6× Super-Resolution Head ─────────────
        self.sr_head = UpscaleBlock(base_ch, out_ch)      # (B, 4, 144, 192)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)        # skip 1
        s2 = self.enc2(s1)       # skip 2
        s3 = self.enc3(s2)       # skip 3
        bn = self.bottleneck(s3)

        # Decoder
        x = self.dec3(bn, s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)

        # 6× upscale to high-res
        return self.sr_head(x)


class SRUNet(UNet):
    """Compatibility wrapper for configs using SRUNet naming."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 64):
        super().__init__(in_ch=in_channels, out_ch=out_channels, base_ch=base_channels)

if __name__ == "__main__":
    model  = UNet(in_ch=4, out_ch=4, base_ch=64)
    dummy  = torch.randn(2, 4, 24, 32)   # batch=2, 4 vars, 24×32 low-res
    output = model(dummy)

    print("Input  :", dummy.shape)   # (2, 4,  24,  32)
    print("Output :", output.shape)  # (2, 4, 144, 192)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {total_params:,}")
    