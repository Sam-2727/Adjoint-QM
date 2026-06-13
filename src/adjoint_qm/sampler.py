"""Metropolis sampling from ``|psi|^2``."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch


@dataclass(frozen=True)
class MetropolisResult:
    samples: torch.Tensor
    acceptance_rate: float
    seed: int
    step_size: float
    burn_in: int
    thinning: int
    n_chains: int


def metropolis_sample(
    model: torch.nn.Module,
    *,
    n_samples: int,
    dim: int = 1,
    n_chains: int = 64,
    step_size: float = 1.0,
    burn_in: int = 500,
    thinning: int = 5,
    seed: int = 1234,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> MetropolisResult:
    """Random-walk Metropolis sampler for ``|psi|^2``."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be positive")

    generator_device = device if device is not None else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    x = torch.zeros(n_chains, dim, dtype=dtype, device=device)
    with torch.no_grad():
        log_prob = 2.0 * model.log_psi(x)

    kept: list[torch.Tensor] = []
    accepted = 0
    proposed = 0
    kept_per_chain = ceil(n_samples / n_chains)
    total_steps = burn_in + thinning * kept_per_chain

    for step in range(total_steps):
        proposal = x + step_size * torch.randn(
            x.shape, dtype=dtype, device=device, generator=generator
        )
        with torch.no_grad():
            proposal_log_prob = 2.0 * model.log_psi(proposal)
            log_accept_ratio = proposal_log_prob - log_prob
            log_u = torch.log(torch.rand(n_chains, dtype=dtype, device=device, generator=generator))
            accept = log_u < log_accept_ratio
            x = torch.where(accept[:, None], proposal, x)
            log_prob = torch.where(accept, proposal_log_prob, log_prob)

        accepted += int(torch.sum(accept).detach())
        proposed += n_chains

        if step >= burn_in and (step - burn_in) % thinning == 0:
            kept.append(x.detach().clone())

    samples = torch.cat(kept, dim=0)[:n_samples]
    return MetropolisResult(
        samples=samples,
        acceptance_rate=accepted / proposed,
        seed=seed,
        step_size=float(step_size),
        burn_in=int(burn_in),
        thinning=int(thinning),
        n_chains=int(n_chains),
    )
