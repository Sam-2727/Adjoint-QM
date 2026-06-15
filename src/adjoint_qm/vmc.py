"""Variational Monte Carlo training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .estimators import local_energy, vmc_observables
from .potentials import Potential
from .sampler import metropolis_sample


@dataclass(frozen=True)
class VMCTrainingRecord:
    """Diagnostic record from one VMC optimization step."""

    step: int
    surrogate_loss: float
    energy: float
    local_energy_std: float
    local_energy_stderr: float
    acceptance_rate: float
    sample_count: int
    virial_residual: float


def vmc_score_function_loss(
    model: torch.nn.Module,
    potential: Potential,
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return the VMC score-function loss and detached diagnostics.

    For real positive wavefunctions sampled from ``|\psi_\theta|^2``, the
    variational-energy gradient is

    ``grad E = 2 <(E_L - <E_L>) grad log psi_theta>``.

    The returned scalar is not the physical energy; it is a surrogate whose
    parameter gradient is this covariance estimator for fixed samples.
    """

    if samples.ndim != 2:
        raise ValueError("samples must have shape (sample_count, dim)")

    detached_samples = samples.detach()
    local_values = local_energy(model, potential, detached_samples).detach()
    energy = torch.mean(local_values)
    centered_local_energy = local_values - energy
    log_psi = model.log_psi(detached_samples)
    loss = 2.0 * torch.mean(centered_local_energy * log_psi)
    return loss, energy.detach(), torch.std(local_values, unbiased=True).detach()


def train_vmc_metropolis(
    model: torch.nn.Module,
    potential: Potential,
    *,
    dim: int,
    n_steps: int = 100,
    n_samples: int = 1024,
    n_chains: int = 64,
    step_size: float = 1.0,
    burn_in: int = 200,
    thinning: int = 5,
    lr: float = 1.0e-3,
    seed: int = 1234,
    report_every: int = 10,
    optimizer_factory: Callable[
        [list[torch.nn.Parameter]], torch.optim.Optimizer
    ]
    | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> list[VMCTrainingRecord]:
    """Train a wavefunction with Metropolis-sampled VMC gradients.

    This intentionally resamples at every optimization step.  That is slower
    than carrying persistent chains, but it keeps the first implementation
    easy to audit and avoids hidden state in the optimization loop.
    """

    if dim < 1:
        raise ValueError("dim must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if report_every < 1:
        raise ValueError("report_every must be positive")

    if optimizer_factory is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optimizer_factory(list(model.parameters()))

    history: list[VMCTrainingRecord] = []
    model.train()
    for step in range(1, n_steps + 1):
        sample_result = metropolis_sample(
            model,
            n_samples=n_samples,
            dim=dim,
            n_chains=n_chains,
            step_size=step_size,
            burn_in=burn_in,
            thinning=thinning,
            seed=seed + step - 1,
            dtype=dtype,
            device=device,
        )
        loss, _, _ = vmc_score_function_loss(
            model,
            potential,
            sample_result.samples,
        )
        record: VMCTrainingRecord | None = None
        if step == 1 or step % report_every == 0 or step == n_steps:
            obs = vmc_observables(model, potential, sample_result.samples)
            record = VMCTrainingRecord(
                step=step,
                surrogate_loss=float(loss.detach()),
                energy=obs.local_energy_mean,
                local_energy_std=obs.local_energy_std,
                local_energy_stderr=obs.local_energy_stderr,
                acceptance_rate=sample_result.acceptance_rate,
                sample_count=obs.sample_count,
                virial_residual=obs.virial_residual,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if record is not None:
            history.append(record)

    return history
