"""Quadrature, local-energy, and diagnostic estimators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .potentials import Potential


def trapezoid_grid(
    x_max: float,
    n_grid: int,
    dim: int = 1,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a symmetric 1D trapezoid grid and integration weights.

    The v1 quadrature benchmark is intentionally restricted to ``dim=1``.
    The returned configurations still have shape ``(batch, dim)`` so the
    surrounding interfaces match later higher-dimensional code.
    """

    if dim != 1:
        raise NotImplementedError("trapezoid_grid currently supports dim=1 only")
    if x_max <= 0:
        raise ValueError("x_max must be positive")
    if n_grid < 3:
        raise ValueError("n_grid must be at least 3")

    grid_1d = torch.linspace(-x_max, x_max, n_grid, dtype=dtype, device=device)
    dx = grid_1d[1] - grid_1d[0]
    weights = torch.full_like(grid_1d, dx)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return grid_1d[:, None], weights


def _normalized_density_weights(
    log_psi: torch.Tensor,
    quadrature_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob = 2.0 * log_psi
    shift = torch.max(log_prob.detach())
    shifted = torch.exp(log_prob - shift)
    unnormalized = quadrature_weights * shifted
    scaled_norm = torch.sum(unnormalized)
    norm = scaled_norm * torch.exp(shift)
    return unnormalized / scaled_norm, norm


def grad_log_psi(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return per-sample gradients of ``log psi`` with respect to coordinates."""

    if not x.requires_grad:
        raise ValueError("x must require gradients")
    log_psi = model.log_psi(x)
    (grad,) = torch.autograd.grad(
        log_psi,
        x,
        grad_outputs=torch.ones_like(log_psi),
        create_graph=True,
    )
    return grad


def quadrature_energy(
    model: torch.nn.Module,
    potential: Potential,
    grid: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute differentiable Rayleigh quotient using the gradient form.

    Returns ``(energy, kinetic, potential_energy, norm)``.
    """

    x = grid.detach().clone().requires_grad_(True)
    log_psi = model.log_psi(x)
    density_weights, norm = _normalized_density_weights(log_psi, weights)
    grad = grad_log_psi(model, x)
    kinetic_density = 0.5 * torch.sum(grad**2, dim=-1)
    potential_density = potential(x)
    kinetic = torch.sum(density_weights * kinetic_density)
    potential_energy = torch.sum(density_weights * potential_density)
    return kinetic + potential_energy, kinetic, potential_energy, norm


def local_energy(
    model: torch.nn.Module,
    potential: Potential,
    x: torch.Tensor,
) -> torch.Tensor:
    r"""Compute local energy ``H psi / psi`` for a real positive wavefunction."""

    x_req = x.detach().clone().requires_grad_(True)
    log_psi = model.log_psi(x_req)
    (grad,) = torch.autograd.grad(
        log_psi,
        x_req,
        grad_outputs=torch.ones_like(log_psi),
        create_graph=True,
    )

    laplacian = torch.zeros_like(log_psi)
    for axis in range(x_req.shape[-1]):
        (second_grad,) = torch.autograd.grad(
            grad[:, axis],
            x_req,
            grad_outputs=torch.ones_like(grad[:, axis]),
            retain_graph=True,
            create_graph=True,
        )
        laplacian = laplacian + second_grad[:, axis]

    return -0.5 * (laplacian + torch.sum(grad**2, dim=-1)) + potential(x_req)


def parity_residual(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Maximum absolute parity violation in ``log psi`` on the supplied points."""

    with torch.no_grad():
        return torch.max(torch.abs(model.log_psi(x) - model.log_psi(-x)))


@dataclass(frozen=True)
class QuadratureObservables:
    energy: float
    kinetic: float
    potential: float
    norm: float
    x2: float
    x4: float
    local_energy_mean: float
    local_energy_variance: float
    parity_residual: float
    virial_residual: float


def quadrature_observables(
    model: torch.nn.Module,
    potential: Potential,
    grid: torch.Tensor,
    weights: torch.Tensor,
) -> QuadratureObservables:
    """Compute benchmark observables on a fixed quadrature grid."""

    energy, kinetic, potential_energy, norm = quadrature_energy(
        model, potential, grid, weights
    )
    x = grid.detach().clone()
    with torch.no_grad():
        log_psi_values = model.log_psi(x)
        density_weights, _ = _normalized_density_weights(log_psi_values, weights)
        x2 = torch.sum(density_weights * torch.sum(x**2, dim=-1))
        x4 = torch.sum(density_weights * torch.sum(x**4, dim=-1))

    e_local = local_energy(model, potential, x)
    with torch.no_grad():
        local_mean = torch.sum(density_weights * e_local)
        local_var = torch.sum(density_weights * (e_local - local_mean) ** 2)
        parity = parity_residual(model, x)
        virial_density = potential.virial(x)
        virial_expectation = torch.sum(density_weights * virial_density)
        virial = 2.0 * kinetic.detach() - virial_expectation.detach()

    return QuadratureObservables(
        energy=float(energy.detach()),
        kinetic=float(kinetic.detach()),
        potential=float(potential_energy.detach()),
        norm=float(norm.detach()),
        x2=float(x2.detach()),
        x4=float(x4.detach()),
        local_energy_mean=float(local_mean.detach()),
        local_energy_variance=float(local_var.detach()),
        parity_residual=float(parity.detach()),
        virial_residual=float(virial.detach()),
    )


@dataclass(frozen=True)
class VMCObservables:
    sample_count: int
    x_mean: float
    x2: float
    x4: float
    r4: float
    kinetic: float
    potential: float
    local_energy_mean: float
    local_energy_std: float
    local_energy_stderr: float
    virial_residual: float
    coordinate_mean_abs_max: float
    coordinate_variance_mean: float
    coordinate_variance_std: float
    coordinate_offdiag_abs_max: float


def vmc_observables(
    model: torch.nn.Module,
    potential: Potential,
    samples: torch.Tensor,
    batch_size: int | None = None,
) -> VMCObservables:
    """Compute sample averages and local-energy diagnostics."""

    if samples.ndim != 2:
        raise ValueError("samples must have shape (sample_count, dim)")
    if samples.shape[0] < 2:
        raise ValueError("at least two samples are required for covariance diagnostics")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive when provided")

    e_local_parts: list[torch.Tensor] = []
    kinetic_parts: list[torch.Tensor] = []
    potential_parts: list[torch.Tensor] = []
    virial_parts: list[torch.Tensor] = []
    chunk_size = samples.shape[0] if batch_size is None else batch_size
    for start in range(0, samples.shape[0], chunk_size):
        chunk = samples[start : start + chunk_size]
        e_local_parts.append(local_energy(model, potential, chunk).detach())
        x_req = chunk.detach().clone().requires_grad_(True)
        grad = grad_log_psi(model, x_req)
        kinetic_parts.append((0.5 * torch.sum(grad**2, dim=-1)).detach())
        potential_parts.append(potential(x_req).detach())
        virial_parts.append(potential.virial(x_req).detach())

    e_local = torch.cat(e_local_parts, dim=0)
    kinetic_density = torch.cat(kinetic_parts, dim=0)
    potential_density = torch.cat(potential_parts, dim=0)
    virial_density = torch.cat(virial_parts, dim=0)

    with torch.no_grad():
        n = samples.shape[0]
        e_std = torch.std(e_local, unbiased=True)
        coordinate_mean = torch.mean(samples, dim=0)
        centered = samples - coordinate_mean
        covariance = centered.T @ centered / (n - 1)
        covariance_diag = torch.diag(covariance)
        covariance_offdiag = covariance - torch.diag(covariance_diag)
        r2_values = torch.sum(samples**2, dim=-1)
        kinetic = torch.mean(kinetic_density)
        virial = 2.0 * kinetic - torch.mean(virial_density)
        return VMCObservables(
            sample_count=int(n),
            x_mean=float(torch.mean(samples[:, 0]).detach()),
            x2=float(torch.mean(r2_values).detach()),
            x4=float(torch.mean(torch.sum(samples**4, dim=-1)).detach()),
            r4=float(torch.mean(r2_values**2).detach()),
            kinetic=float(kinetic.detach()),
            potential=float(torch.mean(potential_density).detach()),
            local_energy_mean=float(torch.mean(e_local).detach()),
            local_energy_std=float(e_std.detach()),
            local_energy_stderr=float(
                (
                    e_std
                    / torch.sqrt(
                        torch.as_tensor(float(n), dtype=e_std.dtype, device=e_std.device)
                    )
                ).detach()
            ),
            virial_residual=float(virial.detach()),
            coordinate_mean_abs_max=float(torch.max(torch.abs(coordinate_mean)).detach()),
            coordinate_variance_mean=float(torch.mean(covariance_diag).detach()),
            coordinate_variance_std=float(torch.std(covariance_diag, unbiased=False).detach()),
            coordinate_offdiag_abs_max=float(torch.max(torch.abs(covariance_offdiag)).detach()),
        )


def exact_harmonic_benchmarks(omega: float = 1.0) -> dict[str, float]:
    """Exact ground-state observables for the 1D harmonic oscillator."""

    if omega <= 0:
        raise ValueError("omega must be positive")
    return {
        "energy": 0.5 * omega,
        "kinetic": 0.25 * omega,
        "potential": 0.25 * omega,
        "x2": 1.0 / (2.0 * omega),
        "x4": 3.0 / (4.0 * omega**2),
        "g0": 1.0 / (2.0 * omega),
    }


def exact_isotropic_harmonic_benchmarks(
    dim: int,
    omega: float = 1.0,
) -> dict[str, float]:
    """Exact ground-state observables for the D-dimensional isotropic oscillator."""

    if dim < 1:
        raise ValueError("dim must be positive")
    if omega <= 0:
        raise ValueError("omega must be positive")
    return {
        "energy": 0.5 * dim * omega,
        "kinetic": 0.25 * dim * omega,
        "potential": 0.25 * dim * omega,
        "r2": dim / (2.0 * omega),
        "r4": dim * (dim + 2.0) / (4.0 * omega**2),
        "coordinate_variance": 1.0 / (2.0 * omega),
        "coordinate_x4_sum": 3.0 * dim / (4.0 * omega**2),
    }


def harmonic_correlator(tau: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """Exact Euclidean two-point function ``<x(tau) x(0)>``."""

    if omega <= 0:
        raise ValueError("omega must be positive")
    return torch.exp(-omega * tau) / (2.0 * omega)


def quartic_oscillator_perturbation_benchmarks(
    omega: float = 1.0,
    coupling: float = 0.0,
) -> dict[str, float]:
    r"""Weak-coupling perturbative benchmarks for ``0.5 omega^2 x^2 + g x^4``.

    The expansion is for

    ``H = p^2/2 + omega^2 x^2/2 + coupling * x^4``.

    Energies are included through second order:

    ``E0 = omega/2 + 3g/(4 omega^2) - 21g^2/(8 omega^5) + O(g^3)``.

    Moment estimates are derivative consequences of this energy expansion via
    Hellmann-Feynman and are only perturbative checks.
    """

    if omega <= 0:
        raise ValueError("omega must be positive")
    if coupling < 0:
        raise ValueError("coupling must be non-negative for this stable benchmark")

    energy_order0 = 0.5 * omega
    energy_order1 = energy_order0 + 3.0 * coupling / (4.0 * omega**2)
    energy_order2 = energy_order1 - 21.0 * coupling**2 / (8.0 * omega**5)
    x2_order0 = 1.0 / (2.0 * omega)
    x2_order1 = x2_order0 - 3.0 * coupling / (2.0 * omega**4)
    x4_order0 = 3.0 / (4.0 * omega**2)
    x4_order1 = x4_order0 - 21.0 * coupling / (4.0 * omega**5)

    return {
        "energy_order0": energy_order0,
        "energy_order1": energy_order1,
        "energy_order2": energy_order2,
        "x2_order0": x2_order0,
        "x2_order1": x2_order1,
        "x4_order0": x4_order0,
        "x4_order1": x4_order1,
    }
