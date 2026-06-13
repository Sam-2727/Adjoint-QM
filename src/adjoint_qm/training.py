"""Training loops for variational wavefunctions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .estimators import quadrature_energy
from .potentials import Potential


@dataclass(frozen=True)
class TrainingRecord:
    step: int
    energy: float
    kinetic: float
    potential: float
    alpha: float


def train_quadrature(
    model: torch.nn.Module,
    potential: Potential,
    grid: torch.Tensor,
    weights: torch.Tensor,
    *,
    n_steps: int = 1000,
    lr: float = 1.0e-2,
    report_every: int = 100,
) -> list[TrainingRecord]:
    """Minimize the quadrature Rayleigh quotient with Adam."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if report_every < 1:
        raise ValueError("report_every must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[TrainingRecord] = []

    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        energy, kinetic, potential_energy, _ = quadrature_energy(
            model, potential, grid, weights
        )
        energy.backward()
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == n_steps:
            history.append(
                TrainingRecord(
                    step=step,
                    energy=float(energy.detach()),
                    kinetic=float(kinetic.detach()),
                    potential=float(potential_energy.detach()),
                    alpha=float(model.alpha.detach()),
                )
            )

    return history
