"""Symmetry-aware feature maps for neural wavefunctions."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class FeatureMap(nn.Module):
    """Feature-map interface for configurations with shape ``(batch, dim)``."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return neural-network input features."""


class EvenFeatureMap(FeatureMap):
    """Coordinate-wise parity-even feature map.

    For the one-dimensional oscillator this is just ``x -> x**2``.  For vector
    inputs it preserves independent sign flips of each coordinate; later models
    can replace it by radial, trace, eigenvalue, or other invariant features.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, dim)")
        return x**2
