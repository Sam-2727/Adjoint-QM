"""Krylov/Lanczos imaginary-time evolution utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .potentials import Potential


def dirichlet_grid_1d(
    x_max: float,
    n_grid: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return interior points and uniform weights for a Dirichlet 1D interval.

    The grid points exclude ``+-x_max`` and the finite-difference Hamiltonian
    assumes the wavefunction vanishes at those endpoints.
    """

    if x_max <= 0:
        raise ValueError("x_max must be positive")
    if n_grid < 3:
        raise ValueError("n_grid must be at least 3")
    dx = 2.0 * x_max / (n_grid + 1)
    points = torch.linspace(
        -x_max + dx,
        x_max - dx,
        n_grid,
        dtype=dtype,
        device=device,
    )
    weights = torch.full((n_grid,), dx, dtype=dtype, device=device)
    return points[:, None], weights


def weighted_inner(
    u: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``<u|v>`` with optional positive quadrature weights."""

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    if weights is None:
        return torch.sum(torch.conj(u) * v)
    if weights.shape != u.shape:
        raise ValueError("weights must have the same shape as u and v")
    return torch.sum(weights * torch.conj(u) * v)


def _real_norm(
    vector: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    norm_squared = _real_norm_squared(vector, weights)
    if bool(norm_squared <= 0):
        raise ValueError("vector has non-positive norm")
    return torch.sqrt(norm_squared)


def _real_norm_squared(
    vector: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    return torch.real(weighted_inner(vector, vector, weights))


@dataclass(frozen=True)
class LanczosResult:
    """Tridiagonal Krylov representation of a Hermitian operator."""

    alphas: torch.Tensor
    betas: torch.Tensor
    source_norm: torch.Tensor
    converged: bool

    @property
    def iterations(self) -> int:
        return int(self.alphas.numel())


@dataclass(frozen=True)
class KrylovSpectrum:
    """Ritz spectral data visible from the Lanczos source state.

    ``normalized_weights`` sum to one over the Krylov approximation.
    ``spectral_weights`` include the source norm and therefore sum to
    ``<phi|phi>``.
    """

    energies: torch.Tensor
    gaps: torch.Tensor
    normalized_weights: torch.Tensor
    spectral_weights: torch.Tensor


def lanczos_tridiagonal(
    hamiltonian_action: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    krylov_dim: int = 8,
    tol: float = 1.0e-12,
    reorthogonalize: bool = True,
) -> LanczosResult:
    """Build a Lanczos tridiagonal matrix from a source state.

    ``hamiltonian_action(v)`` must return ``H v`` in the same vector
    representation.  The algorithm only assumes access to this action and the
    inner product; it does not diagonalize the full Hamiltonian.
    """

    if source.ndim != 1:
        raise ValueError("source must be a one-dimensional state vector")
    if krylov_dim < 1:
        raise ValueError("krylov_dim must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if weights is not None and weights.shape != source.shape:
        raise ValueError("weights must have the same shape as source")

    source_norm = _real_norm(source, weights)
    q = source / source_norm
    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), dtype=source.real.dtype, device=source.device)

    basis: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    converged = False

    for iteration in range(krylov_dim):
        basis.append(q)
        z = hamiltonian_action(q)
        if z.shape != q.shape:
            raise ValueError("hamiltonian_action returned the wrong shape")

        if iteration > 0:
            z = z - beta_prev * q_prev
        alpha = torch.real(weighted_inner(q, z, weights))
        z = z - alpha * q

        if reorthogonalize:
            for basis_vector in basis:
                coeff = weighted_inner(basis_vector, z, weights)
                z = z - coeff * basis_vector

        alphas.append(alpha)

        if iteration == krylov_dim - 1:
            break

        beta_squared = _real_norm_squared(z, weights)
        if bool(beta_squared <= tol**2):
            converged = True
            break
        beta = torch.sqrt(beta_squared)

        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    if betas:
        beta_tensor = torch.stack(betas)
    else:
        beta_tensor = torch.empty(0, dtype=source.real.dtype, device=source.device)

    return LanczosResult(
        alphas=torch.stack(alphas),
        betas=beta_tensor,
        source_norm=source_norm,
        converged=converged,
    )


def tridiagonal_matrix(result: LanczosResult) -> torch.Tensor:
    """Return the dense tridiagonal matrix represented by a Lanczos result."""

    n = result.iterations
    matrix = torch.diag(result.alphas)
    if n > 1:
        idx = torch.arange(n - 1, device=result.alphas.device)
        matrix[idx, idx + 1] = result.betas
        matrix[idx + 1, idx] = result.betas
    return matrix


def krylov_correlator(
    result: LanczosResult,
    tau: torch.Tensor | float,
    *,
    energy_shift: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    r"""Approximate ``<phi|exp[-tau (H - E0)]|phi>`` from Lanczos data."""

    matrix = tridiagonal_matrix(result)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    dtype = eigenvalues.dtype
    device = eigenvalues.device

    tau_tensor = torch.as_tensor(tau, dtype=dtype, device=device)
    original_shape = tau_tensor.shape
    tau_flat = tau_tensor.reshape(-1)
    shift = torch.as_tensor(energy_shift, dtype=dtype, device=device)
    spectral_weights = torch.abs(eigenvectors[0, :]) ** 2
    exponent = -tau_flat[:, None] * (eigenvalues[None, :] - shift)
    values = result.source_norm**2 * torch.sum(
        spectral_weights[None, :] * torch.exp(exponent),
        dim=-1,
    )
    return values.reshape(original_shape)


def krylov_spectrum(
    result: LanczosResult,
    *,
    energy_shift: torch.Tensor | float = 0.0,
) -> KrylovSpectrum:
    r"""Return Ritz energies and spectral weights from Lanczos data.

    If the source state is ``|phi>`` and the Lanczos matrix has eigenpairs
    ``(epsilon_j, u_j)``, the correlator approximation is

    ``G(tau) = sum_j spectral_weights[j] * exp(-tau * gaps[j])``,

    where ``gaps[j] = epsilon_j - energy_shift``.
    """

    matrix = tridiagonal_matrix(result)
    energies, eigenvectors = torch.linalg.eigh(matrix)
    shift = torch.as_tensor(
        energy_shift,
        dtype=energies.dtype,
        device=energies.device,
    )
    normalized_weights = torch.abs(eigenvectors[0, :]) ** 2
    spectral_weights = result.source_norm**2 * normalized_weights
    return KrylovSpectrum(
        energies=energies,
        gaps=energies - shift,
        normalized_weights=normalized_weights,
        spectral_weights=spectral_weights,
    )


def normalized_wavefunction_values(
    model: torch.nn.Module,
    grid: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate and normalize a positive wavefunction on a weighted grid."""

    if grid.ndim != 2:
        raise ValueError("grid must have shape (batch, dim)")
    if weights.shape != (grid.shape[0],):
        raise ValueError("weights must have shape (batch,)")

    with torch.no_grad():
        log_psi = model.log_psi(grid)
        shifted = torch.exp(log_psi - torch.max(log_psi))
        norm = torch.sqrt(torch.sum(weights * shifted**2))
        return shifted / norm


def position_source_state_1d(
    model: torch.nn.Module,
    grid: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    r"""Return the normalized source vector ``x psi_0`` on a 1D grid."""

    if grid.ndim != 2 or grid.shape[1] != 1:
        raise ValueError("grid must have shape (batch, 1)")
    psi = normalized_wavefunction_values(model, grid, weights)
    return grid[:, 0] * psi


def schrodinger_hamiltonian_action_1d(
    values: torch.Tensor,
    grid: torch.Tensor,
    potential: Potential | torch.Tensor,
) -> torch.Tensor:
    """Apply ``-1/2 d^2/dx^2 + V(x)`` with Dirichlet finite differences."""

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if grid.ndim != 2 or grid.shape != (values.shape[0], 1):
        raise ValueError("grid must have shape (values.shape[0], 1)")

    diffs = grid[1:, 0] - grid[:-1, 0]
    dx = diffs[0]
    if not bool(torch.allclose(diffs, torch.ones_like(diffs) * dx)):
        raise ValueError("grid must be uniformly spaced")

    zeros = torch.zeros(1, dtype=values.dtype, device=values.device)
    padded = torch.cat([zeros, values, zeros])
    laplacian = (padded[2:] - 2.0 * values + padded[:-2]) / dx**2

    if isinstance(potential, torch.Tensor):
        potential_values = potential
    else:
        with torch.no_grad():
            potential_values = potential(grid)
    if potential_values.shape != values.shape:
        raise ValueError("potential values must have the same shape as values")

    return -0.5 * laplacian + potential_values * values


def position_correlator_time_evolution_1d(
    model: torch.nn.Module,
    potential: Potential,
    grid: torch.Tensor,
    weights: torch.Tensor,
    tau: torch.Tensor | float,
    *,
    energy_shift: torch.Tensor | float = 0.0,
    krylov_dim: int = 8,
    tol: float = 1.0e-12,
) -> tuple[torch.Tensor, LanczosResult]:
    r"""Compute ``<x(tau)x(0)>`` by Krylov imaginary-time evolution.

    This forms ``|phi> = x |psi_0>`` from the supplied ground-state model and
    approximates ``<phi|exp[-tau (H - energy_shift)]|phi>``.
    """

    if weights.shape != (grid.shape[0],):
        raise ValueError("weights must have shape (grid.shape[0],)")

    source = position_source_state_1d(model, grid, weights)
    with torch.no_grad():
        potential_values = potential(grid)

    def action(vector: torch.Tensor) -> torch.Tensor:
        return schrodinger_hamiltonian_action_1d(vector, grid, potential_values)

    result = lanczos_tridiagonal(
        action,
        source,
        weights=weights,
        krylov_dim=krylov_dim,
        tol=tol,
    )
    return krylov_correlator(result, tau, energy_shift=energy_shift), result
