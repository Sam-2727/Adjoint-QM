"""Potential-energy functions for vector-shaped configurations."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class Potential(nn.Module):
    """Potential interface.

    Inputs have shape ``(batch, dim)`` and outputs have shape ``(batch,)``.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the potential energy for each configuration."""


class HarmonicOscillatorPotential(Potential):
    """One- or multi-dimensional isotropic harmonic potential.

    The first benchmark uses ``dim=1`` and units ``m = hbar = 1``:
    ``V(x) = 0.5 * omega**2 * x**2``.
    """

    def __init__(self, omega: float = 1.0) -> None:
        super().__init__()
        if omega <= 0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, dim)")
        return 0.5 * self.omega**2 * torch.sum(x**2, dim=-1)
