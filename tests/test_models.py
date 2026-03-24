"""Tests for model architectures and the model factory."""

import pytest
import torch

from src.models.factory import build_model, list_models
from src.models.resnet import ResNetDownscaler
from src.models.unet import UNet


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

def test_unet_output_shape_no_upscale():
    model = UNet(in_channels=1, out_channels=1, features=[16, 32])
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_unet_output_shape_with_upscale():
    model = UNet(in_channels=1, out_channels=1, features=[16, 32], scale_factor=2)
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_unet_multi_channel():
    model = UNet(in_channels=3, out_channels=2, features=[16, 32])
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 2, 32, 32)


# ---------------------------------------------------------------------------
# ResNetDownscaler
# ---------------------------------------------------------------------------

def test_resnet_output_shape_scale4():
    model = ResNetDownscaler(in_channels=1, out_channels=1, num_residual_blocks=2,
                             num_features=16, scale_factor=4)
    x = torch.randn(2, 1, 8, 8)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_resnet_output_shape_scale1():
    model = ResNetDownscaler(in_channels=1, out_channels=1, num_residual_blocks=2,
                             num_features=16, scale_factor=1)
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    assert out.shape == (2, 1, 16, 16)


def test_resnet_gradients():
    model = ResNetDownscaler(num_residual_blocks=2, num_features=16, scale_factor=4)
    x = torch.randn(1, 1, 8, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for parameter '{name}'"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_model_unet():
    model = build_model("unet", in_channels=1, out_channels=1, features=[16, 32])
    assert isinstance(model, UNet)


def test_build_model_resnet():
    model = build_model("resnet", num_residual_blocks=2, num_features=16, scale_factor=4)
    assert isinstance(model, ResNetDownscaler)


def test_build_model_case_insensitive():
    model = build_model("UNet", features=[16, 32])
    assert isinstance(model, UNet)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vgg")


def test_list_models():
    models = list_models()
    assert "unet" in models
    assert "resnet" in models
