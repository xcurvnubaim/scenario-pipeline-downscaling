"""Model factory for the scenario-pipeline-downscaling project.

Models can be constructed either via:
  1. :func:`build_model` – a simple string-based factory.
  2. Hydra's ``hydra.utils.instantiate`` using the ``_target_`` key in a
     model config group (preferred when using the full Hydra pipeline).

Adding a new architecture
--------------------------
1. Create ``src/models/<new_arch>.py`` with a class that subclasses
   ``torch.nn.Module``.
2. Register it in the ``_REGISTRY`` dict below.
3. Add a corresponding ``conf/model/<new_arch>.yaml`` config file.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .gan import RRDBGenerator
from .resnet import ResNetDownscaler
from .unet import UNet

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type] = {
    "unet": UNet,
    "resnet": ResNetDownscaler,
    "rrdb": RRDBGenerator,
}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model by name.

    Parameters
    ----------
    name:
        Architecture name (case-insensitive). Must be a key in ``_REGISTRY``.
    **kwargs:
        Forwarded verbatim to the model constructor.

    Returns
    -------
    nn.Module
        The instantiated (un-trained) model.

    Raises
    ------
    ValueError
        If ``name`` is not found in the registry.

    Examples
    --------
    >>> model = build_model("unet", in_channels=1, out_channels=1, scale_factor=4)
    >>> model = build_model("resnet", num_residual_blocks=16, scale_factor=4)
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}.")
    return _REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return sorted(_REGISTRY)
