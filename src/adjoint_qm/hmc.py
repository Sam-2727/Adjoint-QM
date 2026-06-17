"""Batched Hamiltonian Monte Carlo utilities.

This module is intentionally independent of the neural training loop.  The
core sampler only needs an unnormalized differentiable log density
``log_prob(z)`` and returns samples plus basic transition diagnostics.  The
adjoint-specific helper builds the log norm density in traceless eigenvalue
coordinates, but the HMC kernel itself is generic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch

from .adjoint import (
    adjoint_dirichlet_terms,
    log_vandermonde_abs,
    tangent_project,
    traceless_hyperplane_basis,
)


LogProbFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class HMCStepResult:
    """Result of one batched HMC transition."""

    z: torch.Tensor
    accepted: torch.Tensor
    log_acceptance_ratio: torch.Tensor
    hamiltonian_error: torch.Tensor
    divergent: torch.Tensor


@dataclass(frozen=True)
class HMCChainResult:
    """Stored samples and aggregate diagnostics from an HMC run."""

    samples: torch.Tensor
    final_z: torch.Tensor
    acceptance_rate: float
    divergence_fraction: float
    mean_abs_hamiltonian_error: float
    max_abs_hamiltonian_error: float
    step_size: float
    n_leapfrog: int
    n_steps: int
    burn_in: int
    thin: int


@dataclass(frozen=True)
class HMCWarmupResult:
    """Final state and tuned parameters from HMC warmup."""

    final_z: torch.Tensor
    step_size: float
    mass: torch.Tensor
    acceptance_rate: float
    divergence_fraction: float
    n_warmup: int
    leapfrog_min: int
    leapfrog_max: int
    target_acceptance: float


@dataclass(frozen=True)
class AdjointLaggedEnergy:
    """Differentiable lagged-reweighting energy and diagnostics."""

    energy: torch.Tensor
    radial: torch.Tensor
    angular: torch.Tensor
    kinetic: torch.Tensor
    potential: torch.Tensor
    tr_x2: torch.Tensor
    tr_x4: torch.Tensor
    virial_rhs: torch.Tensor
    virial_residual: torch.Tensor
    relative_ess: torch.Tensor


def _random_leapfrog_count(
    leapfrog_min: int,
    leapfrog_max: int,
    *,
    generator: torch.Generator | None = None,
) -> int:
    if leapfrog_min < 1:
        raise ValueError("leapfrog_min must be positive")
    if leapfrog_max < leapfrog_min:
        raise ValueError("leapfrog_max must be at least leapfrog_min")
    if leapfrog_min == leapfrog_max:
        return int(leapfrog_min)
    return int(
        torch.randint(
            leapfrog_min,
            leapfrog_max + 1,
            (1,),
            generator=generator,
        ).item()
    )


def _as_mass_tensor(
    mass: torch.Tensor | float | None,
    z: torch.Tensor,
) -> torch.Tensor:
    if mass is None:
        return torch.ones((z.shape[-1],), dtype=z.dtype, device=z.device)
    mass_tensor = torch.as_tensor(mass, dtype=z.dtype, device=z.device)
    if torch.any(mass_tensor <= 0):
        raise ValueError("mass entries must be positive")
    return mass_tensor


def kinetic_energy(momentum: torch.Tensor, mass: torch.Tensor | float | None) -> torch.Tensor:
    """Return ``0.5 p^T M^{-1} p`` for a diagonal or scalar mass."""

    mass_tensor = _as_mass_tensor(mass, momentum)
    return 0.5 * torch.sum(momentum * momentum / mass_tensor, dim=-1)


def potential_and_grad(
    log_prob_fn: LogProbFn,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``U=-log_prob`` and ``grad U`` for a batch of positions.

    Nonfinite log-density entries receive infinite potential and zero gradient.
    The log-density function itself must avoid producing NaNs in differentiable
    branches for invalid points.
    """

    if z.ndim != 2:
        raise ValueError("z must have shape (batch, dim)")
    z_req = z.detach().clone().requires_grad_(True)
    log_prob = log_prob_fn(z_req)
    if log_prob.ndim != 1 or log_prob.shape[0] != z.shape[0]:
        raise ValueError("log_prob_fn must return shape (batch,)")

    finite = torch.isfinite(log_prob)
    grad_source = torch.where(
        finite,
        -log_prob,
        torch.zeros_like(log_prob),
    )
    (grad,) = torch.autograd.grad(grad_source.sum(), z_req)
    potential = torch.where(
        finite,
        -log_prob.detach(),
        torch.full_like(log_prob.detach(), torch.inf),
    )
    grad = torch.where(finite[:, None], grad.detach(), torch.zeros_like(grad.detach()))
    return potential, grad


def leapfrog_integrate(
    log_prob_fn: LogProbFn,
    z: torch.Tensor,
    momentum: torch.Tensor,
    *,
    step_size: float,
    n_leapfrog: int,
    mass: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a batch of Hamiltonian trajectories by leapfrog.

    Returns ``(z_new, p_new, potential_new)`` without a momentum flip.  The
    routine is useful both for HMC proposals and for reversibility tests.
    """

    if z.ndim != 2:
        raise ValueError("z must have shape (batch, dim)")
    if momentum.shape != z.shape:
        raise ValueError("momentum must have the same shape as z")
    if n_leapfrog < 1:
        raise ValueError("n_leapfrog must be positive")
    if step_size == 0.0:
        raise ValueError("step_size must be nonzero")

    mass_tensor = _as_mass_tensor(mass, z)
    z_new = z.detach().clone()
    p_new = momentum.detach().clone()
    _, grad = potential_and_grad(log_prob_fn, z_new)
    p_new = p_new - 0.5 * step_size * grad

    potential = torch.empty((z.shape[0],), dtype=z.dtype, device=z.device)
    for leapfrog_index in range(n_leapfrog):
        z_new = z_new + step_size * p_new / mass_tensor
        potential, grad = potential_and_grad(log_prob_fn, z_new)
        if leapfrog_index != n_leapfrog - 1:
            p_new = p_new - step_size * grad

    p_new = p_new - 0.5 * step_size * grad
    return z_new.detach(), p_new.detach(), potential.detach()


def hmc_step(
    z: torch.Tensor,
    log_prob_fn: LogProbFn,
    *,
    step_size: float,
    n_leapfrog: int,
    mass: torch.Tensor | float | None = None,
    generator: torch.Generator | None = None,
    divergence_threshold: float = 1000.0,
) -> HMCStepResult:
    """Perform one batched fixed-length HMC transition."""

    if z.ndim != 2:
        raise ValueError("z must have shape (batch, dim)")
    if divergence_threshold <= 0.0:
        raise ValueError("divergence_threshold must be positive")

    mass_tensor = _as_mass_tensor(mass, z)
    noise = torch.randn(
        z.shape,
        dtype=z.dtype,
        device=z.device,
        generator=generator,
    )
    momentum = noise * torch.sqrt(mass_tensor)

    potential_start, _ = potential_and_grad(log_prob_fn, z)
    kinetic_start = kinetic_energy(momentum, mass_tensor)
    z_proposed, momentum_proposed, potential_proposed = leapfrog_integrate(
        log_prob_fn,
        z,
        momentum,
        step_size=step_size,
        n_leapfrog=n_leapfrog,
        mass=mass_tensor,
    )
    kinetic_proposed = kinetic_energy(momentum_proposed, mass_tensor)

    hamiltonian_start = potential_start + kinetic_start
    hamiltonian_proposed = potential_proposed + kinetic_proposed
    hamiltonian_error = hamiltonian_proposed - hamiltonian_start
    finite = torch.isfinite(hamiltonian_error)
    divergent = (~finite) | (torch.abs(torch.nan_to_num(hamiltonian_error, nan=torch.inf)) > divergence_threshold)
    log_acceptance_ratio = -hamiltonian_error
    log_acceptance_ratio = torch.where(
        finite & (~divergent),
        log_acceptance_ratio,
        torch.full_like(log_acceptance_ratio, -torch.inf),
    )
    uniform = torch.rand(
        (z.shape[0],),
        dtype=z.dtype,
        device=z.device,
        generator=generator,
    )
    accepted = torch.log(uniform) < torch.clamp(log_acceptance_ratio, max=0.0)
    z_next = torch.where(accepted[:, None], z_proposed, z)

    return HMCStepResult(
        z=z_next.detach(),
        accepted=accepted.detach(),
        log_acceptance_ratio=log_acceptance_ratio.detach(),
        hamiltonian_error=hamiltonian_error.detach(),
        divergent=divergent.detach(),
    )


def hmc_sample(
    log_prob_fn: LogProbFn,
    initial_z: torch.Tensor,
    *,
    step_size: float,
    n_leapfrog: int,
    n_steps: int,
    burn_in: int = 0,
    thin: int = 1,
    mass: torch.Tensor | float | None = None,
    generator: torch.Generator | None = None,
    divergence_threshold: float = 1000.0,
) -> HMCChainResult:
    """Run a batched HMC chain and store post-burn-in samples."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thin < 1:
        raise ValueError("thin must be positive")

    z = initial_z.detach().clone()
    samples: list[torch.Tensor] = []
    accepted_count = 0
    divergent_count = 0
    abs_errors: list[torch.Tensor] = []

    for step in range(1, n_steps + 1):
        result = hmc_step(
            z,
            log_prob_fn,
            step_size=step_size,
            n_leapfrog=n_leapfrog,
            mass=mass,
            generator=generator,
            divergence_threshold=divergence_threshold,
        )
        z = result.z
        accepted_count += int(torch.sum(result.accepted).item())
        divergent_count += int(torch.sum(result.divergent).item())
        finite_errors = torch.abs(result.hamiltonian_error[torch.isfinite(result.hamiltonian_error)])
        if finite_errors.numel() > 0:
            abs_errors.append(finite_errors.detach())
        if step > burn_in and (step - burn_in) % thin == 0:
            samples.append(z.detach().clone())

    if not samples:
        raise ValueError("no samples were stored; reduce burn_in or thin")

    total_transitions = n_steps * z.shape[0]
    if abs_errors:
        all_abs_errors = torch.cat(abs_errors)
        mean_abs_error = float(torch.mean(all_abs_errors).detach())
        max_abs_error = float(torch.max(all_abs_errors).detach())
    else:
        mean_abs_error = math.inf
        max_abs_error = math.inf

    return HMCChainResult(
        samples=torch.stack(samples, dim=0),
        final_z=z.detach(),
        acceptance_rate=accepted_count / total_transitions,
        divergence_fraction=divergent_count / total_transitions,
        mean_abs_hamiltonian_error=mean_abs_error,
        max_abs_hamiltonian_error=max_abs_error,
        step_size=float(step_size),
        n_leapfrog=int(n_leapfrog),
        n_steps=int(n_steps),
        burn_in=int(burn_in),
        thin=int(thin),
    )


def hmc_warmup(
    log_prob_fn: LogProbFn,
    initial_z: torch.Tensor,
    *,
    initial_step_size: float,
    leapfrog_min: int,
    leapfrog_max: int,
    n_warmup: int,
    target_acceptance: float = 0.80,
    mass: torch.Tensor | float | None = None,
    adapt_step_size: bool = True,
    adapt_mass: bool = True,
    generator: torch.Generator | None = None,
    divergence_threshold: float = 1000.0,
) -> HMCWarmupResult:
    """Warm up a batched HMC target with windowed dual averaging.

    When ``adapt_mass`` is true, warmup is split into two windows.  The first
    window tunes a provisional step size and estimates a diagonal mass from the
    chain covariance.  The second window then retunes the step size with that
    mass held fixed.  This avoids using a step size tuned for one kinetic
    geometry with a different final mass matrix.
    """

    if n_warmup < 1:
        raise ValueError("n_warmup must be positive")
    if initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive")
    if not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")

    def run_step_window(
        z_in: torch.Tensor,
        mass_in: torch.Tensor,
        step_size_in: float,
        n_steps: int,
        *,
        collect_positions: bool,
    ) -> tuple[torch.Tensor, float, int, int, list[torch.Tensor]]:
        z_window = z_in.detach().clone()
        step_size_window = float(step_size_in)
        accepted_window = 0
        divergent_window = 0
        positions: list[torch.Tensor] = []

        if adapt_step_size:
            log_step_size = math.log(step_size_window)
            log_step_size_avg = log_step_size
            mu = math.log(10.0 * step_size_window)
            h_bar = 0.0
            gamma = 0.05
            t0 = 10.0
            kappa = 0.75

        for warmup_step in range(1, n_steps + 1):
            n_leapfrog = _random_leapfrog_count(
                leapfrog_min,
                leapfrog_max,
                generator=generator,
            )
            result = hmc_step(
                z_window,
                log_prob_fn,
                step_size=step_size_window,
                n_leapfrog=n_leapfrog,
                mass=mass_in,
                generator=generator,
                divergence_threshold=divergence_threshold,
            )
            z_window = result.z
            accepted_window += int(torch.sum(result.accepted).item())
            divergent_window += int(torch.sum(result.divergent).item())
            accept_probability = torch.exp(
                torch.clamp(result.log_acceptance_ratio, max=0.0)
            )
            mean_accept = float(torch.mean(accept_probability).detach())
            if adapt_step_size:
                eta = 1.0 / (warmup_step + t0)
                h_bar = (1.0 - eta) * h_bar + eta * (
                    target_acceptance - mean_accept
                )
                log_step_size = mu - math.sqrt(warmup_step) * h_bar / gamma
                weight = warmup_step ** (-kappa)
                log_step_size_avg = (
                    weight * log_step_size + (1.0 - weight) * log_step_size_avg
                )
                step_size_window = float(math.exp(log_step_size))
            if collect_positions:
                positions.append(z_window.detach().clone())

        if adapt_step_size:
            step_size_window = float(math.exp(log_step_size_avg))
        return (
            z_window.detach(),
            step_size_window,
            accepted_window,
            divergent_window,
            positions,
        )

    z = initial_z.detach().clone()
    mass_tensor = _as_mass_tensor(mass, z)
    step_size = float(initial_step_size)
    accepted_count = 0
    divergent_count = 0

    if adapt_mass and n_warmup >= 2:
        first_window = max(1, n_warmup // 2)
        second_window = n_warmup - first_window
        z, provisional_step_size, accepted, divergent, mass_samples = run_step_window(
            z,
            mass_tensor,
            step_size,
            first_window,
            collect_positions=True,
        )
        accepted_count += accepted
        divergent_count += divergent

        if mass_samples:
            flat = torch.cat(mass_samples, dim=0)
            variance = torch.var(flat, dim=0, unbiased=True)
            new_mass = torch.clamp(variance, min=1.0e-4, max=1.0e4).detach()
            scale = torch.sqrt(torch.median(new_mass / mass_tensor)).item()
            mass_tensor = new_mass
            step_size = provisional_step_size * max(scale, 1.0e-3)
        else:
            step_size = provisional_step_size

        if second_window > 0:
            z, step_size, accepted, divergent, _ = run_step_window(
                z,
                mass_tensor,
                step_size,
                second_window,
                collect_positions=False,
            )
            accepted_count += accepted
            divergent_count += divergent
    else:
        z, step_size, accepted_count, divergent_count, _ = run_step_window(
            z,
            mass_tensor,
            step_size,
            n_warmup,
            collect_positions=False,
        )

    total_transitions = n_warmup * z.shape[0]
    return HMCWarmupResult(
        final_z=z.detach(),
        step_size=step_size,
        mass=mass_tensor.detach(),
        acceptance_rate=accepted_count / total_transitions,
        divergence_fraction=divergent_count / total_transitions,
        n_warmup=int(n_warmup),
        leapfrog_min=int(leapfrog_min),
        leapfrog_max=int(leapfrog_max),
        target_acceptance=float(target_acceptance),
    )


def hmc_sample_randomized(
    log_prob_fn: LogProbFn,
    initial_z: torch.Tensor,
    *,
    step_size: float,
    leapfrog_min: int,
    leapfrog_max: int,
    n_steps: int,
    burn_in: int = 0,
    thin: int = 1,
    mass: torch.Tensor | float | None = None,
    generator: torch.Generator | None = None,
    divergence_threshold: float = 1000.0,
) -> HMCChainResult:
    """Run HMC with a randomized leapfrog count on each transition."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thin < 1:
        raise ValueError("thin must be positive")

    z = initial_z.detach().clone()
    samples: list[torch.Tensor] = []
    accepted_count = 0
    divergent_count = 0
    abs_errors: list[torch.Tensor] = []

    for step in range(1, n_steps + 1):
        n_leapfrog = _random_leapfrog_count(
            leapfrog_min,
            leapfrog_max,
            generator=generator,
        )
        result = hmc_step(
            z,
            log_prob_fn,
            step_size=step_size,
            n_leapfrog=n_leapfrog,
            mass=mass,
            generator=generator,
            divergence_threshold=divergence_threshold,
        )
        z = result.z
        accepted_count += int(torch.sum(result.accepted).item())
        divergent_count += int(torch.sum(result.divergent).item())
        finite_errors = torch.abs(
            result.hamiltonian_error[torch.isfinite(result.hamiltonian_error)]
        )
        if finite_errors.numel() > 0:
            abs_errors.append(finite_errors.detach())
        if step > burn_in and (step - burn_in) % thin == 0:
            samples.append(z.detach().clone())

    if not samples:
        raise ValueError("no samples were stored; reduce burn_in or thin")

    total_transitions = n_steps * z.shape[0]
    if abs_errors:
        all_abs_errors = torch.cat(abs_errors)
        mean_abs_error = float(torch.mean(all_abs_errors).detach())
        max_abs_error = float(torch.max(all_abs_errors).detach())
    else:
        mean_abs_error = math.inf
        max_abs_error = math.inf

    return HMCChainResult(
        samples=torch.stack(samples, dim=0),
        final_z=z.detach(),
        acceptance_rate=accepted_count / total_transitions,
        divergence_fraction=divergent_count / total_transitions,
        mean_abs_hamiltonian_error=mean_abs_error,
        max_abs_hamiltonian_error=max_abs_error,
        step_size=float(step_size),
        n_leapfrog=-1,
        n_steps=int(n_steps),
        burn_in=int(burn_in),
        thin=int(thin),
    )


def ordered_traceless_gaussian_initial(
    n: int,
    n_chains: int,
    *,
    sigma: float = 1.0,
    descending: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered traceless eigenvalues and their coordinates ``z``."""

    if n < 2:
        raise ValueError("n must be at least two")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    lam = sigma * torch.randn(
        (n_chains, n),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    lam = torch.sort(lam, dim=-1, descending=descending).values
    lam = tangent_project(lam)
    basis = traceless_hyperplane_basis(n, dtype=dtype, device=device)
    z = lam @ basis
    return z.detach(), lam.detach()


def minimum_ordered_gap(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return the minimum adjacent ordered eigenvalue gap per sample."""

    lam = z @ basis.T
    if lam.shape[-1] < 2:
        raise ValueError("at least two eigenvalues are required")
    gaps = lam[:, :-1] - lam[:, 1:]
    return torch.min(gaps, dim=-1).values


def adjoint_hmc_log_norm_density(
    model: torch.nn.Module,
    z: torch.Tensor,
    *,
    beta: float = 1.0,
    basis: torch.Tensor | None = None,
    ordered: bool = True,
    gap_eps: float = 1.0e-12,
    head_norm_eps: float = 1.0e-300,
) -> torch.Tensor:
    r"""Return the adjoint HMC log target up to an additive constant.

    The target is

    ``Delta(lambda)^2 exp[-S_theta(lambda)] A_theta(lambda)^beta``,

    where ``A_theta = sum_i a_i(lambda)^2`` and ``lambda = z @ basis.T``.
    For ``ordered=True`` the function is restricted to the chamber
    ``lambda_0 > lambda_1 > ...`` and invalid points receive ``-inf``.
    """

    if z.ndim != 2:
        raise ValueError("z must have shape (batch, n - 1)")
    n = int(getattr(model, "n"))
    if z.shape[-1] != n - 1:
        raise ValueError(f"z must have shape (batch, {n - 1})")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if gap_eps < 0.0:
        raise ValueError("gap_eps must be non-negative")
    if head_norm_eps <= 0.0:
        raise ValueError("head_norm_eps must be positive")

    if basis is None:
        basis = traceless_hyperplane_basis(n, dtype=z.dtype, device=z.device)
    basis = basis.to(dtype=z.dtype, device=z.device)
    lam = tangent_project(z @ basis.T)

    if ordered:
        adjacent_gaps = lam[:, :-1] - lam[:, 1:]
        valid = torch.all(adjacent_gaps > gap_eps, dim=-1)
    else:
        valid = torch.ones((z.shape[0],), dtype=torch.bool, device=z.device)

    log_delta = log_vandermonde_abs(lam, eps=gap_eps)
    action = model.action(lam)
    head = model.head(lam)
    head_norm = torch.sum(head * head, dim=-1).clamp_min(head_norm_eps)
    log_prob = 2.0 * log_delta - action + beta * torch.log(head_norm)
    finite = valid & torch.isfinite(log_prob)
    return torch.where(
        finite,
        log_prob,
        torch.full_like(log_prob, -torch.inf),
    )


def relative_effective_sample_size(log_weights: torch.Tensor) -> torch.Tensor:
    """Return ESS divided by sample count from unnormalized log weights."""

    if log_weights.ndim != 1:
        raise ValueError("log_weights must have shape (sample_count,)")
    shift = torch.max(log_weights.detach())
    weights = torch.exp(log_weights - shift)
    sum_weights = torch.sum(weights)
    sum_weights_squared = torch.sum(weights * weights)
    return sum_weights * sum_weights / (
        log_weights.numel() * sum_weights_squared
    ).clamp_min(torch.finfo(log_weights.dtype).tiny)


def _weighted_mean_from_log_weights(
    log_weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if log_weights.ndim != 1 or values.shape[0] != log_weights.shape[0]:
        raise ValueError("values must have the same leading dimension as log_weights")
    shift = torch.max(log_weights.detach())
    weights = torch.exp(log_weights - shift)
    return torch.sum(weights * values) / torch.sum(weights)


def adjoint_lagged_log_weights(
    model: torch.nn.Module,
    anchor_model: torch.nn.Module,
    lam: torch.Tensor,
    *,
    beta: float,
    head_norm_eps: float = 1.0e-300,
) -> torch.Tensor:
    r"""Return log weights from ``pi_anchor,beta`` to ``pi_model,1``.

    The Vandermonde factor cancels in the ratio, leaving

    ``-S_model + log A_model + S_anchor - beta log A_anchor``.
    """

    if lam.ndim != 2:
        raise ValueError("lam must have shape (sample_count, n)")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if head_norm_eps <= 0.0:
        raise ValueError("head_norm_eps must be positive")

    lam = tangent_project(lam)
    current_action = model.action(lam)
    current_head = model.head(lam)
    current_head_norm = torch.sum(current_head * current_head, dim=-1).clamp_min(
        head_norm_eps
    )
    with torch.no_grad():
        anchor_action = anchor_model.action(lam).detach()
        anchor_head = anchor_model.head(lam).detach()
        anchor_head_norm = torch.sum(anchor_head * anchor_head, dim=-1).clamp_min(
            head_norm_eps
        )
    return (
        -current_action
        + torch.log(current_head_norm)
        + anchor_action
        - beta * torch.log(anchor_head_norm)
    )


def adjoint_lagged_reweighted_energy(
    model: torch.nn.Module,
    anchor_model: torch.nn.Module,
    lam: torch.Tensor,
    *,
    beta: float,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> AdjointLaggedEnergy:
    """Return differentiable energy and diagnostics by lagged reweighting."""

    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    log_weights = adjoint_lagged_log_weights(
        model,
        anchor_model,
        lam,
        beta=beta,
    )
    kinetic_local = (terms.radial + terms.angular) / terms.head_norm
    radial_local = terms.radial / terms.head_norm
    angular_local = terms.angular / terms.head_norm
    potential_local = terms.potential
    local_energy = kinetic_local + potential_local
    tr_x2 = torch.sum(lam * lam, dim=-1)
    tr_x4 = torch.sum(lam**4, dim=-1)
    energy = _weighted_mean_from_log_weights(log_weights, local_energy)
    radial = _weighted_mean_from_log_weights(log_weights, radial_local)
    angular = _weighted_mean_from_log_weights(log_weights, angular_local)
    kinetic = _weighted_mean_from_log_weights(log_weights, kinetic_local)
    potential = _weighted_mean_from_log_weights(log_weights, potential_local)
    mean_tr_x2 = _weighted_mean_from_log_weights(log_weights, tr_x2)
    mean_tr_x4 = _weighted_mean_from_log_weights(log_weights, tr_x4)
    virial_rhs = omega**2 * mean_tr_x2 + 4.0 * coupling * mean_tr_x4
    virial_residual = 2.0 * kinetic - virial_rhs
    return AdjointLaggedEnergy(
        energy=energy,
        radial=radial,
        angular=angular,
        kinetic=kinetic,
        potential=potential,
        tr_x2=mean_tr_x2,
        tr_x4=mean_tr_x4,
        virial_rhs=virial_rhs,
        virial_residual=virial_residual,
        relative_ess=relative_effective_sample_size(log_weights.detach()),
    )
