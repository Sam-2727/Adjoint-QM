"""Wavefunction ansaetze."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .features import EvenFeatureMap, FeatureMap


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of ``softplus`` for positive ``y``."""

    threshold = torch.as_tensor(20.0, dtype=y.dtype, device=y.device)
    return torch.where(y > threshold, y, torch.log(torch.expm1(y)))


class GaussianEnvelopeMLP(nn.Module):
    r"""Positive real wavefunction with Gaussian envelope and neural correction.

    The ansatz is

    .. math::

       \log \psi_\theta(x)
       = -\frac12 \alpha \sum_i x_i^2 + f_\theta(\phi(x)),
       \qquad \alpha > 0.

    For the first benchmark, ``phi(x)=x**2`` enforces parity evenness.
    """

    def __init__(
        self,
        dim: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        feature_map: FeatureMap | None = None,
        init_alpha: float = 1.0,
        alpha_floor: float = 1.0e-6,
        activation: type[nn.Module] = nn.Tanh,
        zero_final: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if init_alpha <= alpha_floor:
            raise ValueError("init_alpha must be larger than alpha_floor")

        self.dim = int(dim)
        self.feature_map = feature_map if feature_map is not None else EvenFeatureMap()
        self.alpha_floor = float(alpha_floor)

        raw_alpha = _inverse_softplus(
            torch.as_tensor(init_alpha - alpha_floor, dtype=dtype)
        )
        self.raw_alpha = nn.Parameter(raw_alpha.clone().detach())

        layers: list[nn.Module] = []
        in_features = self.feature_map.output_dim(self.dim)
        for width in hidden_layers:
            layers.append(nn.Linear(in_features, int(width), dtype=dtype))
            layers.append(activation())
            in_features = int(width)
        layers.append(nn.Linear(in_features, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

        if zero_final:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_alpha) + self.alpha_floor

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_map(x)
        return self.net(features).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_psi(x)

    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape (batch, {self.dim})")
        envelope = -0.5 * self.alpha * torch.sum(x**2, dim=-1)
        return envelope + self.correction(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, alpha={float(self.alpha.detach()):.6g}"


class SeparableGaussianEnvelopeMLP(nn.Module):
    r"""Product wavefunction with a shared one-dimensional neural factor.

    The ansatz is

    .. math::

       \log \psi_\theta(x_1,\ldots,x_D)
       =
       \sum_{i=1}^D
       \left[-\frac12\alpha x_i^2 + f_\theta(x_i^2)\right].

    This is appropriate for separable benchmark Hamiltonians.  It is not a
    general interacting many-coordinate ansatz.
    """

    def __init__(
        self,
        dim: int = 1,
        hidden_layers: Sequence[int] = (32, 32),
        init_alpha: float = 1.0,
        alpha_floor: float = 1.0e-6,
        activation: type[nn.Module] = nn.Tanh,
        zero_final: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.one_body = GaussianEnvelopeMLP(
            dim=1,
            hidden_layers=hidden_layers,
            feature_map=EvenFeatureMap(),
            init_alpha=init_alpha,
            alpha_floor=alpha_floor,
            activation=activation,
            zero_final=zero_final,
            dtype=dtype,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.one_body.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_psi(x)

    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape (batch, {self.dim})")
        one_body_values = self.one_body.log_psi(x.reshape(-1, 1))
        return one_body_values.reshape(x.shape[0], self.dim).sum(dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, alpha={float(self.alpha.detach()):.6g}"
