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

    def virial(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return ``x dot grad V`` for each configuration.

        The virial theorem for stationary bound states is
        ``2<T> = <x dot grad V>``.  Subclasses should override this when a
        virial diagnostic is meaningful.
        """

        raise NotImplementedError(f"{type(self).__name__} does not define virial")


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

    def virial(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, dim)")
        return self.omega**2 * torch.sum(x**2, dim=-1)


class QuarticOscillatorPotential(Potential):
    r"""Anharmonic oscillator potential ``0.5 omega**2 x**2 + g x**4``."""

    def __init__(self, omega: float = 1.0, coupling: float = 0.0) -> None:
        super().__init__()
        if omega <= 0:
            raise ValueError("omega must be positive")
        if coupling < 0:
            raise ValueError("coupling must be non-negative for this stable benchmark")
        self.omega = float(omega)
        self.coupling = float(coupling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, dim)")
        x2 = torch.sum(x**2, dim=-1)
        x4 = torch.sum(x**4, dim=-1)
        return 0.5 * self.omega**2 * x2 + self.coupling * x4

    def virial(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, dim)")
        x2 = torch.sum(x**2, dim=-1)
        x4 = torch.sum(x**4, dim=-1)
        return self.omega**2 * x2 + 4.0 * self.coupling * x4
