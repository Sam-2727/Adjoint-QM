"""Harmonic-basis diagonalization benchmarks for oscillator models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BasisDiagonalizationResult:
    """Eigenpairs from a finite harmonic-oscillator basis projection.

    ``eigenvectors[:, i]`` contains the harmonic-basis coefficients of the
    ``i``th eigenstate.
    """

    energies: torch.Tensor
    eigenvectors: torch.Tensor
    hamiltonian: torch.Tensor
    basis_omega: float


@dataclass(frozen=True)
class SeparableDiagonalizationResult:
    """Low-lying spectrum of a separable ``D``-dimensional oscillator.

    The Hamiltonian is a sum of identical one-dimensional Hamiltonians.  The
    many-dimensional eigenvalues are therefore finite sums of the corresponding
    one-dimensional eigenvalues, so this is equivalent to direct
    diagonalization in the product basis without explicitly forming the large
    Kronecker-sum matrix.
    """

    energies: torch.Tensor
    one_dimensional: BasisDiagonalizationResult
    dim: int


def _coordinate_matrix(
    n_basis: int,
    basis_omega: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if basis_omega <= 0:
        raise ValueError("basis_omega must be positive")

    matrix = torch.zeros((n_basis, n_basis), dtype=dtype, device=device)
    scale = 1.0 / torch.sqrt(torch.as_tensor(2.0 * basis_omega, dtype=dtype))
    for n in range(n_basis - 1):
        value = scale * torch.sqrt(torch.as_tensor(float(n + 1), dtype=dtype))
        matrix[n, n + 1] = value
        matrix[n + 1, n] = value
    return matrix


def _resolve_basis_omega(omega: float, basis_omega: float | None) -> float:
    if basis_omega is None:
        if omega > 0:
            return float(omega)
        return 1.0
    if basis_omega <= 0:
        raise ValueError("basis_omega must be positive")
    return float(basis_omega)


def quartic_oscillator_hamiltonian_matrix(
    n_basis: int,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    basis_omega: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Return the projected Hamiltonian matrix for ``x^2/2 + g x^4``.

    The target Hamiltonian is

    ``H = p^2/2 + omega**2 x**2/2 + coupling * x**4``.

    The finite basis is built from harmonic-oscillator eigenstates with
    frequency ``basis_omega``.  Matrix elements of ``x**4`` are evaluated by
    forming coordinate matrices in a basis enlarged by four states before
    taking the top-left block, avoiding artifacts from prematurely truncating
    repeated coordinate multiplications.
    """

    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if omega < 0:
        raise ValueError("omega must be non-negative")
    if coupling < 0:
        raise ValueError("coupling must be non-negative for this benchmark")

    oscillator_omega = _resolve_basis_omega(omega, basis_omega)

    work_basis = n_basis + 4
    x_matrix = _coordinate_matrix(
        work_basis,
        oscillator_omega,
        dtype=dtype,
        device=device,
    )
    x2 = x_matrix @ x_matrix
    x4 = x2 @ x2
    x2 = x2[:n_basis, :n_basis]
    x4 = x4[:n_basis, :n_basis]

    n = torch.arange(n_basis, dtype=dtype, device=device)
    harmonic_part = torch.diag(oscillator_omega * (n + 0.5))
    delta_quadratic = 0.5 * (omega**2 - oscillator_omega**2) * x2
    quartic_part = coupling * x4
    return harmonic_part + delta_quadratic + quartic_part


def diagonalize_quartic_oscillator(
    n_basis: int,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    basis_omega: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> BasisDiagonalizationResult:
    """Diagonalize the quartic oscillator in a finite harmonic basis."""

    hamiltonian = quartic_oscillator_hamiltonian_matrix(
        n_basis,
        omega=omega,
        coupling=coupling,
        basis_omega=basis_omega,
        dtype=dtype,
        device=device,
    )
    energies, eigenvectors = torch.linalg.eigh(hamiltonian)
    oscillator_omega = _resolve_basis_omega(omega, basis_omega)
    return BasisDiagonalizationResult(
        energies=energies,
        eigenvectors=eigenvectors,
        hamiltonian=hamiltonian,
        basis_omega=oscillator_omega,
    )


def _lowest_separable_energy_sums(
    one_dimensional_energies: torch.Tensor,
    dim: int,
    n_levels: int,
) -> torch.Tensor:
    """Return the lowest sums of ``dim`` entries from a 1D energy list."""

    if one_dimensional_energies.ndim != 1:
        raise ValueError("one_dimensional_energies must be one-dimensional")
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    if one_dimensional_energies.numel() < 1:
        raise ValueError("one_dimensional_energies must be non-empty")

    single_particle_levels = one_dimensional_energies[:n_levels]
    energies = torch.zeros(
        1,
        dtype=one_dimensional_energies.dtype,
        device=one_dimensional_energies.device,
    )
    for _ in range(dim):
        sums = (energies[:, None] + single_particle_levels[None, :]).reshape(-1)
        energies = torch.sort(sums).values[:n_levels]
    return energies


def diagonalize_separable_quartic_oscillator(
    dim: int,
    n_basis: int,
    *,
    n_levels: int = 8,
    omega: float = 1.0,
    coupling: float = 0.0,
    basis_omega: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SeparableDiagonalizationResult:
    r"""Diagonalize ``sum_i [p_i^2/2 + omega^2 x_i^2/2 + g x_i^4]``.

    The benchmark potential is separable, so the product-basis Hamiltonian is a
    Kronecker sum of one-dimensional quartic Hamiltonians.  Rather than forming
    that large matrix, this helper diagonalizes the 1D Hamiltonian and returns
    the lowest finite sums of its eigenvalues.  This gives the same finite-basis
    energies as direct diagonalization of the tensor-product Hamiltonian.
    """

    if dim < 1:
        raise ValueError("dim must be positive")
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    if n_basis < n_levels:
        raise ValueError("n_basis must be at least n_levels")

    one_dimensional = diagonalize_quartic_oscillator(
        n_basis,
        omega=omega,
        coupling=coupling,
        basis_omega=basis_omega,
        dtype=dtype,
        device=device,
    )
    energies = _lowest_separable_energy_sums(
        one_dimensional.energies,
        dim=dim,
        n_levels=n_levels,
    )
    return SeparableDiagonalizationResult(
        energies=energies,
        one_dimensional=one_dimensional,
        dim=int(dim),
    )


def harmonic_basis_functions_1d(
    x: torch.Tensor,
    n_basis: int,
    *,
    basis_omega: float = 1.0,
) -> torch.Tensor:
    """Evaluate normalized 1D harmonic-oscillator basis functions.

    ``x`` may have shape ``(batch,)`` or ``(batch, 1)``.  The return value has
    shape ``(batch, n_basis)``.
    """

    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if basis_omega <= 0:
        raise ValueError("basis_omega must be positive")
    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError("x must have shape (batch,) or (batch, 1)")
        points = x[:, 0]
    elif x.ndim == 1:
        points = x
    else:
        raise ValueError("x must have shape (batch,) or (batch, 1)")

    dtype = points.dtype
    device = points.device
    omega_tensor = torch.as_tensor(basis_omega, dtype=dtype, device=device)
    y = torch.sqrt(omega_tensor) * points
    values = torch.empty((points.shape[0], n_basis), dtype=dtype, device=device)
    pi = torch.as_tensor(torch.pi, dtype=dtype, device=device)

    values[:, 0] = (omega_tensor / pi) ** 0.25 * torch.exp(-0.5 * y**2)
    if n_basis == 1:
        return values

    values[:, 1] = torch.sqrt(torch.as_tensor(2.0, dtype=dtype, device=device)) * y * values[:, 0]
    for n in range(1, n_basis - 1):
        n_tensor = torch.as_tensor(float(n), dtype=dtype, device=device)
        next_tensor = torch.as_tensor(float(n + 1), dtype=dtype, device=device)
        values[:, n + 1] = (
            torch.sqrt(2.0 / next_tensor) * y * values[:, n]
            - torch.sqrt(n_tensor / next_tensor) * values[:, n - 1]
        )
    return values


def evaluate_basis_wavefunctions(
    x: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    basis_omega: float,
    state_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Evaluate eigenfunctions from harmonic-basis coefficient vectors."""

    if eigenvectors.ndim != 2:
        raise ValueError("eigenvectors must have shape (n_basis, n_states)")
    if state_indices is None:
        selected = eigenvectors
    else:
        selected = eigenvectors[:, state_indices]
    basis_values = harmonic_basis_functions_1d(
        x,
        eigenvectors.shape[0],
        basis_omega=basis_omega,
    )
    return basis_values @ selected


def align_wavefunction_sign(
    values: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fix the arbitrary global sign of real eigenfunctions for plotting."""

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if reference is not None:
        if reference.shape != values.shape:
            raise ValueError("reference must have the same shape as values")
        overlap = torch.sum(reference * values)
        return torch.where(overlap < 0, -values, values)

    center_index = values.numel() // 2
    return torch.where(values[center_index] < 0, -values, values)
