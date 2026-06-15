"""Adjoint-sector spectral ansaetze for one-matrix SU(N) quantum mechanics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from math import pi, sqrt

import torch
from torch import nn

from .ansatz import _inverse_softplus


def traceless_hyperplane_basis(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an orthonormal basis for ``sum_i lambda_i = 0``.

    The returned matrix has shape ``(n, n - 1)`` and maps coordinates ``z`` to
    eigenvalues by ``lambda = z @ basis.T``.
    """

    if n < 2:
        raise ValueError("n must be at least two")
    raw = torch.zeros((n, n - 1), dtype=dtype, device=device)
    for column in range(n - 1):
        raw[column, column] = 1.0
        raw[-1, column] = -1.0
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return basis


def eigenvalues_from_traceless_coordinates(
    z: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Map orthonormal traceless coordinates to eigenvalues."""

    if z.ndim != 2 or z.shape[-1] != n - 1:
        raise ValueError(f"z must have shape (batch, {n - 1})")
    basis = traceless_hyperplane_basis(n, dtype=z.dtype, device=z.device)
    return z @ basis.T


def su2_adjoint_eigenvalue_grid(
    z_max: float,
    n_grid: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return midpoint quadrature points for SU(2) traceless eigenvalues.

    The grid is built in the orthonormal coordinate ``z`` and excludes the
    eigenvalue-collision point ``z=0``.  The returned tuple is
    ``(z, lambda, weights)`` with ``lambda = (z/sqrt(2), -z/sqrt(2))`` up to the
    sign convention of the hyperplane basis.
    """

    if z_max <= 0:
        raise ValueError("z_max must be positive")
    if n_grid < 2:
        raise ValueError("n_grid must be at least two")

    edges = torch.linspace(-z_max, z_max, n_grid + 1, dtype=dtype, device=device)
    z_1d = 0.5 * (edges[:-1] + edges[1:])
    dz = edges[1] - edges[0]
    z = z_1d[:, None]
    lam = eigenvalues_from_traceless_coordinates(z, n=2)
    weights = torch.full((n_grid,), dz, dtype=dtype, device=device)
    return z, lam, weights


def adjoint_eigenvalue_grid(
    n: int,
    z_max: float,
    n_grid: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return midpoint quadrature points on the SU(N) traceless hyperplane.

    The coordinates are Cartesian in an orthonormal basis of the
    ``sum_i lambda_i = 0`` hyperplane.  The number of points is
    ``n_grid**(n - 1)``, so this deterministic helper is intended only for
    small ``n``.
    """

    if n < 2:
        raise ValueError("n must be at least two")
    if z_max <= 0:
        raise ValueError("z_max must be positive")
    if n_grid < 2:
        raise ValueError("n_grid must be at least two")
    if n == 2:
        return su2_adjoint_eigenvalue_grid(
            z_max,
            n_grid,
            dtype=dtype,
            device=device,
        )

    edges = torch.linspace(-z_max, z_max, n_grid + 1, dtype=dtype, device=device)
    points_1d = 0.5 * (edges[:-1] + edges[1:])
    dz = edges[1] - edges[0]
    meshes = torch.meshgrid(
        *([points_1d] * (n - 1)),
        indexing="ij",
    )
    z = torch.stack([mesh.reshape(-1) for mesh in meshes], dim=-1)
    lam = eigenvalues_from_traceless_coordinates(z, n=n)
    weights = torch.full(
        (z.shape[0],),
        dz ** (n - 1),
        dtype=dtype,
        device=device,
    )
    return z, lam, weights


def su3_adjoint_polar_eigenvalue_grid(
    r_max: float,
    n_radial: int,
    n_angular: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return midpoint polar quadrature on the SU(3) traceless hyperplane."""

    if r_max <= 0:
        raise ValueError("r_max must be positive")
    if n_radial < 2:
        raise ValueError("n_radial must be at least two")
    if n_angular < 3:
        raise ValueError("n_angular must be at least three")

    dr = torch.as_tensor(r_max / n_radial, dtype=dtype, device=device)
    dtheta = torch.as_tensor(2.0 * pi / n_angular, dtype=dtype, device=device)
    r = dr * (
        torch.arange(n_radial, dtype=dtype, device=device) + 0.5
    )
    theta = dtheta * (
        torch.arange(n_angular, dtype=dtype, device=device) + 0.5
    )
    radial_mesh, angular_mesh = torch.meshgrid(r, theta, indexing="ij")
    z = torch.stack(
        [
            radial_mesh.reshape(-1) * torch.cos(angular_mesh.reshape(-1)),
            radial_mesh.reshape(-1) * torch.sin(angular_mesh.reshape(-1)),
        ],
        dim=-1,
    )
    lam = eigenvalues_from_traceless_coordinates(z, n=3)
    weights = (radial_mesh.reshape(-1) * dr * dtheta).to(dtype=dtype)
    return z, lam, weights


def adjoint_matrix_potential(
    lam: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> torch.Tensor:
    r"""Return ``0.5 omega**2 Tr X**2 + coupling Tr X**4`` from eigenvalues."""

    if lam.ndim != 2:
        raise ValueError("lam must have shape (batch, n)")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if coupling < 0:
        raise ValueError("coupling must be non-negative")
    return 0.5 * omega**2 * torch.sum(lam**2, dim=-1) + coupling * torch.sum(
        lam**4,
        dim=-1,
    )


def log_vandermonde_abs(lam: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return ``log |prod_{i<j}(lambda_i-lambda_j)|`` per sample."""

    if lam.ndim != 2:
        raise ValueError("lam must have shape (batch, n)")
    n = lam.shape[-1]
    log_abs = torch.zeros(lam.shape[0], dtype=lam.dtype, device=lam.device)
    for i in range(n):
        for j in range(i + 1, n):
            gap = torch.abs(lam[:, i] - lam[:, j])
            if eps > 0.0:
                gap = torch.clamp(gap, min=eps)
            log_abs = log_abs + torch.log(gap)
    return log_abs


def tangent_project(vector: torch.Tensor) -> torch.Tensor:
    """Project vectors onto the traceless eigenvalue hyperplane."""

    if vector.ndim < 1:
        raise ValueError("vector must have at least one dimension")
    return vector - torch.mean(vector, dim=-1, keepdim=True)


class SUNAdjointRadialSpectralAnsatz(nn.Module):
    r"""SU(N) radial adjoint spectral ansatz with exact odd adjoint covariance.

    The profile is

    ``q_i(lambda) = exp[-S_theta(lambda)/2] * lambda_i``.

    The scalar action ``S_theta`` is a trainable radial function of
    ``p2 = sum_i lambda_i**2 = Tr X**2``.  The fixed head ``a_i=lambda_i`` is
    the lowest harmonic adjoint head and is exact for the radial SU(2) and
    SU(3) quartic potentials, where ``Tr X**4 = 0.5 * (Tr X**2)**2``.
    """

    def __init__(
        self,
        *,
        n: int,
        omega_init: float = 1.0,
        quartic_init: float = 0.0,
        hidden_layers: Sequence[int] = (32, 32),
        alpha_floor: float = 1.0e-8,
        cubic_floor: float = 0.0,
        tail_eps: float = 1.0e-12,
        activation: type[nn.Module] = nn.Tanh,
        zero_final: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n < 2:
            raise ValueError("n must be at least two")
        if omega_init <= alpha_floor:
            raise ValueError("omega_init must be larger than alpha_floor")
        if quartic_init < 0:
            raise ValueError("quartic_init must be non-negative")

        self.n = int(n)
        self.alpha_floor = float(alpha_floor)
        self.cubic_floor = float(cubic_floor)
        self.tail_eps = float(tail_eps)
        raw_alpha = _inverse_softplus(
            torch.as_tensor(omega_init - alpha_floor, dtype=dtype)
        )
        self.raw_alpha = nn.Parameter(raw_alpha.clone().detach())
        if quartic_init == 0.0:
            raw_cubic = torch.as_tensor(-50.0, dtype=dtype)
        else:
            raw_cubic = _inverse_softplus(torch.as_tensor(quartic_init, dtype=dtype))
        self.raw_cubic = nn.Parameter(raw_cubic.clone().detach())

        layers: list[nn.Module] = []
        in_features = 2
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

    @property
    def cubic(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_cubic) + self.cubic_floor

    def action(self, lam: torch.Tensor) -> torch.Tensor:
        """Return the invariant scalar action ``S_theta(lambda)``."""

        if lam.ndim != 2 or lam.shape[-1] != self.n:
            raise ValueError(f"lam must have shape (batch, {self.n})")
        centered = tangent_project(lam)
        p2 = torch.sum(centered**2, dim=-1, keepdim=True)
        p32 = (p2 + self.tail_eps) ** 1.5
        features = torch.cat([p2, p32], dim=-1)
        return (
            self.alpha * p2.squeeze(-1)
            + self.cubic * p32.squeeze(-1)
            + self.net(features).squeeze(-1)
        )

    def head(self, lam: torch.Tensor) -> torch.Tensor:
        """Return the traceless odd adjoint head ``a_i=lambda_i``."""

        if lam.ndim != 2 or lam.shape[-1] != self.n:
            raise ValueError(f"lam must have shape (batch, {self.n})")
        return tangent_project(lam)

    def profile(self, lam: torch.Tensor) -> torch.Tensor:
        """Return ``q_i(lambda)`` with shape ``(batch, 2)``."""

        action = self.action(lam)
        return torch.exp(-0.5 * action)[:, None] * self.head(lam)

    def log_density_eigenvalues(self, lam: torch.Tensor) -> torch.Tensor:
        r"""Return the unnormalized log density ``log(Delta**2 exp(-S) A)``."""

        head = self.head(lam)
        amplitude = torch.sum(head**2, dim=-1)
        return (
            2.0 * log_vandermonde_abs(lam)
            - self.action(lam)
            + torch.log(amplitude)
        )

    def log_psi(self, z: torch.Tensor) -> torch.Tensor:
        """Sampler hook returning half the eigenvalue target log-density."""

        lam = eigenvalues_from_traceless_coordinates(z, n=self.n)
        return 0.5 * self.log_density_eigenvalues(lam)

    def extra_repr(self) -> str:
        return (
            f"SU({self.n}), alpha={float(self.alpha.detach()):.6g}, "
            f"cubic={float(self.cubic.detach()):.6g}"
        )


def adjoint_invariant_features(
    lam: torch.Tensor,
    *,
    tail_eps: float = 1.0e-12,
) -> torch.Tensor:
    r"""Return even Weyl-invariant features for one-matrix adjoint ansaetze.

    The features are ``p2``, ``(p2 + eps)**(3/2)``, ``p4``, and ``p3**2`` with
    ``pk = sum_i lambda_i**k``.  They are invariant under permutations and
    even under the parity transformation ``lambda -> -lambda``.
    """

    if lam.ndim != 2:
        raise ValueError("lam must have shape (batch, n)")
    centered = tangent_project(lam)
    p2 = torch.sum(centered**2, dim=-1, keepdim=True)
    p3 = torch.sum(centered**3, dim=-1, keepdim=True)
    p4 = torch.sum(centered**4, dim=-1, keepdim=True)
    p32 = (p2 + tail_eps) ** 1.5
    return torch.cat([p2, p32, p4, p3**2], dim=-1)


def adjoint_polynomial_heads(
    lam: torch.Tensor,
    *,
    max_heads: int = 8,
) -> torch.Tensor:
    r"""Return smooth traceless odd Weyl-covariant polynomial heads.

    The output has shape ``(batch, n_heads, n)``.  Every head is odd under
    ``lambda -> -lambda``, traceless, and Weyl-covariant.  The first head is
    exactly ``lambda_i``.  The remaining heads are divided by positive
    invariant powers of ``1+p2`` to keep their scale comparable on wide
    quadrature grids; this does not change their covariance or collision
    regularity.
    """

    if lam.ndim != 2:
        raise ValueError("lam must have shape (batch, n)")
    if max_heads < 1:
        raise ValueError("max_heads must be positive")

    centered = tangent_project(lam)
    p2 = torch.sum(centered**2, dim=-1, keepdim=True)
    p3 = torch.sum(centered**3, dim=-1, keepdim=True)
    p4 = torch.sum(centered**4, dim=-1, keepdim=True)
    scale_1 = 1.0 + p2
    scale_32 = scale_1**1.5
    scale_2 = scale_1**2
    scale_52 = scale_1**2.5
    heads = [
        centered,
        tangent_project(centered**3) / scale_1,
        p2 * centered / scale_1,
        p3 * tangent_project(centered**2) / scale_32,
        p4 * centered / scale_2,
        p2 * tangent_project(centered**3) / scale_2,
        p3**2 * centered / scale_52,
        p2 * p3 * tangent_project(centered**2) / scale_52,
    ]
    return torch.stack(heads[:max_heads], dim=1)


class SUNAdjointInvariantSpectralAnsatz(nn.Module):
    r"""SU(N) adjoint ansatz with non-radial invariant action and head.

    This is the first ansatz in the repository intended for the \(SU(4)\)
    quartic model, where ``Tr X**4`` is not fixed by ``Tr X**2``.  The profile
    is still represented as

    ``q_i(lambda) = exp[-S_theta(lambda)/2] * a_i(lambda)``.

    The scalar action is a trainable even Weyl-invariant function of
    ``p2 = Tr X**2``, ``p4 = Tr X**4``, and ``p3**2``.  The adjoint head is a
    trainable combination of smooth traceless odd Weyl-covariant polynomial
    heads, anchored by the harmonic head ``a_i=lambda_i``.
    """

    def __init__(
        self,
        *,
        n: int,
        omega_init: float = 1.0,
        quartic_init: float = 0.0,
        hidden_layers: Sequence[int] = (32, 32),
        head_hidden_layers: Sequence[int] = (32,),
        n_heads: int = 4,
        alpha_floor: float = 1.0e-8,
        cubic_floor: float = 0.0,
        tail_eps: float = 1.0e-12,
        activation: type[nn.Module] = nn.Tanh,
        zero_final: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n < 2:
            raise ValueError("n must be at least two")
        if n_heads < 1 or n_heads > 8:
            raise ValueError("n_heads must be between one and eight")
        if omega_init <= alpha_floor:
            raise ValueError("omega_init must be larger than alpha_floor")
        if quartic_init < 0:
            raise ValueError("quartic_init must be non-negative")

        self.n = int(n)
        self.n_heads = int(n_heads)
        self.alpha_floor = float(alpha_floor)
        self.cubic_floor = float(cubic_floor)
        self.tail_eps = float(tail_eps)
        raw_alpha = _inverse_softplus(
            torch.as_tensor(omega_init - alpha_floor, dtype=dtype)
        )
        self.raw_alpha = nn.Parameter(raw_alpha.clone().detach())
        if quartic_init == 0.0:
            raw_cubic = torch.as_tensor(-50.0, dtype=dtype)
        else:
            raw_cubic = _inverse_softplus(torch.as_tensor(quartic_init, dtype=dtype))
        self.raw_cubic = nn.Parameter(raw_cubic.clone().detach())

        action_layers: list[nn.Module] = []
        in_features = 4
        for width in hidden_layers:
            action_layers.append(nn.Linear(in_features, int(width), dtype=dtype))
            action_layers.append(activation())
            in_features = int(width)
        action_layers.append(nn.Linear(in_features, 1, dtype=dtype))
        self.net = nn.Sequential(*action_layers)

        head_layers: list[nn.Module] = []
        in_features = 4
        for width in head_hidden_layers:
            head_layers.append(nn.Linear(in_features, int(width), dtype=dtype))
            head_layers.append(activation())
            in_features = int(width)
        head_layers.append(nn.Linear(in_features, self.n_heads - 1, dtype=dtype))
        self.head_net = nn.Sequential(*head_layers)

        if zero_final:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            head_final = self.head_net[-1]
            if isinstance(head_final, nn.Linear):
                nn.init.zeros_(head_final.weight)
                nn.init.zeros_(head_final.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_alpha) + self.alpha_floor

    @property
    def cubic(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_cubic) + self.cubic_floor

    def invariant_features(self, lam: torch.Tensor) -> torch.Tensor:
        return adjoint_invariant_features(lam, tail_eps=self.tail_eps)

    def action(self, lam: torch.Tensor) -> torch.Tensor:
        """Return the even Weyl-invariant scalar action ``S_theta``."""

        if lam.ndim != 2 or lam.shape[-1] != self.n:
            raise ValueError(f"lam must have shape (batch, {self.n})")
        features = self.invariant_features(lam)
        p2 = features[:, 0]
        p32 = features[:, 1]
        return self.alpha * p2 + self.cubic * p32 + self.net(features).squeeze(-1)

    def head(self, lam: torch.Tensor) -> torch.Tensor:
        """Return a trainable traceless odd Weyl-covariant adjoint head."""

        if lam.ndim != 2 or lam.shape[-1] != self.n:
            raise ValueError(f"lam must have shape (batch, {self.n})")
        basis = adjoint_polynomial_heads(lam, max_heads=self.n_heads)
        if self.n_heads == 1:
            return basis[:, 0, :]
        coeffs = self.head_net(self.invariant_features(lam))
        head = basis[:, 0, :]
        head = head + torch.sum(coeffs[:, :, None] * basis[:, 1:, :], dim=1)
        return tangent_project(head)

    def profile(self, lam: torch.Tensor) -> torch.Tensor:
        """Return ``q_i(lambda)`` with shape ``(batch, n)``."""

        action = self.action(lam)
        return torch.exp(-0.5 * action)[:, None] * self.head(lam)

    def log_density_eigenvalues(self, lam: torch.Tensor) -> torch.Tensor:
        r"""Return the unnormalized log density ``log(Delta**2 exp(-S) A)``."""

        head = self.head(lam)
        amplitude = torch.sum(head**2, dim=-1)
        return (
            2.0 * log_vandermonde_abs(lam)
            - self.action(lam)
            + torch.log(amplitude)
        )

    def log_psi(self, z: torch.Tensor) -> torch.Tensor:
        """Sampler hook returning half the eigenvalue target log-density."""

        lam = eigenvalues_from_traceless_coordinates(z, n=self.n)
        return 0.5 * self.log_density_eigenvalues(lam)

    def extra_repr(self) -> str:
        return (
            f"SU({self.n}), heads={self.n_heads}, "
            f"alpha={float(self.alpha.detach()):.6g}, "
            f"cubic={float(self.cubic.detach()):.6g}"
        )


def adjoint_even_moment_features(
    lam: torch.Tensor,
    *,
    moment_cutoff: int = 6,
) -> torch.Tensor:
    r"""Return parity-even symmetric moment features.

    The feature vector contains normalized even moments ``m_2, m_4, ...`` and
    squared odd moments ``m_3**2, m_5**2, ...`` through ``moment_cutoff``.
    Squared odd moments allow the scalar envelope and coefficient networks to
    see non-radial information while preserving the odd adjoint parity of the
    full profile when the head basis has odd parity.
    """

    if lam.ndim != 2:
        raise ValueError("lam must have shape (batch, n)")
    if moment_cutoff < 2:
        raise ValueError("moment_cutoff must be at least two")
    centered = tangent_project(lam)
    n = centered.shape[-1]
    features: list[torch.Tensor] = []
    for power in range(2, moment_cutoff + 1):
        moment = torch.sum(centered**power, dim=-1, keepdim=True) / n
        if power % 2 == 0:
            features.append(moment)
        else:
            features.append(moment**2)
    return torch.cat(features, dim=-1)


def chebyshev_polynomial_values(
    x: torch.Tensor,
    degrees: Sequence[int],
) -> torch.Tensor:
    """Return Chebyshev ``T_k(x)`` values with shape ``x.shape + (len(k),)``."""

    if not degrees:
        raise ValueError("degrees must be non-empty")
    max_degree = max(int(degree) for degree in degrees)
    if min(int(degree) for degree in degrees) < 1:
        raise ValueError("degrees must be positive")
    values: list[torch.Tensor] = [torch.ones_like(x), x]
    for degree in range(2, max_degree + 1):
        values.append(2.0 * x * values[-1] - values[-2])
    return torch.stack([values[int(degree)] for degree in degrees], dim=-1)


def centered_chebyshev_heads(
    lam: torch.Tensor,
    *,
    degrees: Sequence[int],
    scale: torch.Tensor | float,
) -> torch.Tensor:
    r"""Return centered Chebyshev adjoint heads ``B_{k,i}``.

    The output has shape ``(batch, len(degrees), n)`` and

    ``B_{k,i} = T_k(lambda_i / L) - mean_j T_k(lambda_j / L)``.
    """

    if lam.ndim != 2:
        raise ValueError("lam must have shape (batch, n)")
    scale_tensor = torch.as_tensor(scale, dtype=lam.dtype, device=lam.device)
    if torch.any(scale_tensor <= 0):
        raise ValueError("scale must be positive")
    centered = tangent_project(lam)
    cheb = chebyshev_polynomial_values(centered / scale_tensor, degrees)
    heads = cheb - torch.mean(cheb, dim=1, keepdim=True)
    return heads.permute(0, 2, 1)


class SUNAdjointChebyshevSpectralAnsatz(nn.Module):
    r"""General SU(N) spectral-impurity ansatz from ``tex/suN_adjoint_ansatz``.

    The profile is

    ``q_i(lambda) = exp[-S_theta(lambda)/2] sum_k c_k(m; eta) B_{k,i}``,

    where ``B_{k,i}`` are centered Chebyshev heads.  By default only odd
    Chebyshev degrees are used and the coefficient/action networks receive
    parity-even moment features, so the represented adjoint state is odd under
    ``X -> -X``.  This is the natural sector containing the harmonic adjoint
    state ``Psi(X)=X exp[-omega Tr X**2 / 2]``.
    """

    def __init__(
        self,
        *,
        n: int,
        omega_init: float = 1.0,
        quartic_tail_init: float = 0.0,
        moment_cutoff: int = 6,
        chebyshev_degrees: Sequence[int] = (1, 3, 5, 7),
        scale_init: float = 3.0,
        learn_scale: bool = False,
        hidden_layers: Sequence[int] = (32, 32),
        head_hidden_layers: Sequence[int] = (32,),
        action_correction_scale: float = 1.0,
        head_correction_scale: float = 1.0,
        alpha_floor: float = 1.0e-8,
        cubic_floor: float = 0.0,
        tail_eps: float = 1.0e-6,
        activation: type[nn.Module] = nn.Tanh,
        zero_final: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n < 2:
            raise ValueError("n must be at least two")
        if omega_init <= alpha_floor:
            raise ValueError("omega_init must be larger than alpha_floor")
        if quartic_tail_init < 0:
            raise ValueError("quartic_tail_init must be non-negative")
        if scale_init <= 0:
            raise ValueError("scale_init must be positive")
        if action_correction_scale < 0:
            raise ValueError("action_correction_scale must be non-negative")
        if head_correction_scale < 0:
            raise ValueError("head_correction_scale must be non-negative")
        if not chebyshev_degrees:
            raise ValueError("chebyshev_degrees must be non-empty")
        if int(chebyshev_degrees[0]) != 1:
            raise ValueError("the first Chebyshev degree must be one")

        self.n = int(n)
        self.moment_cutoff = int(moment_cutoff)
        self.chebyshev_degrees = tuple(int(degree) for degree in chebyshev_degrees)
        self.alpha_floor = float(alpha_floor)
        self.cubic_floor = float(cubic_floor)
        self.tail_eps = float(tail_eps)
        self.learn_scale = bool(learn_scale)
        self.action_correction_scale = float(action_correction_scale)
        self.head_correction_scale = float(head_correction_scale)

        raw_alpha = _inverse_softplus(
            torch.as_tensor(omega_init - alpha_floor, dtype=dtype)
        )
        self.raw_alpha = nn.Parameter(raw_alpha.clone().detach())
        if quartic_tail_init == 0.0:
            raw_cubic = torch.as_tensor(-50.0, dtype=dtype)
        else:
            raw_cubic = _inverse_softplus(
                torch.as_tensor(quartic_tail_init, dtype=dtype)
            )
        self.raw_cubic = nn.Parameter(raw_cubic.clone().detach())
        raw_scale = _inverse_softplus(torch.as_tensor(scale_init, dtype=dtype))
        self.raw_scale = nn.Parameter(
            raw_scale.clone().detach(),
            requires_grad=learn_scale,
        )

        feature_count = self.moment_cutoff - 1
        action_layers: list[nn.Module] = []
        in_features = feature_count
        for width in hidden_layers:
            action_layers.append(nn.Linear(in_features, int(width), dtype=dtype))
            action_layers.append(activation())
            in_features = int(width)
        action_layers.append(nn.Linear(in_features, 1, dtype=dtype))
        self.net = nn.Sequential(*action_layers)

        if len(self.chebyshev_degrees) > 1:
            head_layers: list[nn.Module] = []
            in_features = feature_count
            for width in head_hidden_layers:
                head_layers.append(nn.Linear(in_features, int(width), dtype=dtype))
                head_layers.append(activation())
                in_features = int(width)
            head_layers.append(
                nn.Linear(
                    in_features,
                    len(self.chebyshev_degrees) - 1,
                    dtype=dtype,
                )
            )
            self.head_net: nn.Sequential | None = nn.Sequential(*head_layers)
        else:
            self.head_net = None

        if zero_final:
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            head_final = self.head_net[-1] if self.head_net is not None else None
            if isinstance(head_final, nn.Linear):
                nn.init.zeros_(head_final.weight)
                nn.init.zeros_(head_final.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_alpha) + self.alpha_floor

    @property
    def cubic(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_cubic) + self.cubic_floor

    @property
    def scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_scale)

    def invariant_features(self, lam: torch.Tensor) -> torch.Tensor:
        return adjoint_even_moment_features(
            lam,
            moment_cutoff=self.moment_cutoff,
        )

    def action(self, lam: torch.Tensor) -> torch.Tensor:
        """Return the even Weyl-invariant scalar action ``S_theta``."""

        if lam.ndim != 2 or lam.shape[-1] != self.n:
            raise ValueError(f"lam must have shape (batch, {self.n})")
        centered = tangent_project(lam)
        p2 = torch.sum(centered**2, dim=-1)
        quartic_tail = torch.sum(
            (centered**2 + self.tail_eps**2) ** 1.5,
            dim=-1,
        )
        return (
            self.alpha * p2
            + self.cubic * quartic_tail
            + self.action_correction_scale
            * self.net(self.invariant_features(lam)).squeeze(-1)
        )

    def head_basis(self, lam: torch.Tensor) -> torch.Tensor:
        return centered_chebyshev_heads(
            lam,
            degrees=self.chebyshev_degrees,
            scale=self.scale,
        )

    def head(self, lam: torch.Tensor) -> torch.Tensor:
        """Return the traceless odd Weyl-covariant Chebyshev impurity head."""

        if lam.ndim != 2 or lam.shape[-1] != self.n:
            raise ValueError(f"lam must have shape (batch, {self.n})")
        basis = self.head_basis(lam)
        head = basis[:, 0, :]
        if len(self.chebyshev_degrees) > 1:
            if self.head_net is None:
                raise RuntimeError("head_net is missing for multi-head ansatz")
            coeffs = (
                self.head_correction_scale
                * self.head_net(self.invariant_features(lam))
            )
            head = head + torch.sum(coeffs[:, :, None] * basis[:, 1:, :], dim=1)
        return tangent_project(head)

    def profile(self, lam: torch.Tensor) -> torch.Tensor:
        """Return ``q_i(lambda)`` with shape ``(batch, n)``."""

        return torch.exp(-0.5 * self.action(lam))[:, None] * self.head(lam)

    def log_density_eigenvalues(self, lam: torch.Tensor) -> torch.Tensor:
        r"""Return the unnormalized log density ``log(Delta**2 exp(-S) A)``."""

        head = self.head(lam)
        amplitude = torch.sum(head**2, dim=-1)
        return (
            2.0 * log_vandermonde_abs(lam)
            - self.action(lam)
            + torch.log(amplitude)
        )

    def log_psi(self, z: torch.Tensor) -> torch.Tensor:
        """Sampler hook returning half the eigenvalue target log-density."""

        lam = eigenvalues_from_traceless_coordinates(z, n=self.n)
        return 0.5 * self.log_density_eigenvalues(lam)

    def extra_repr(self) -> str:
        degrees = ",".join(str(degree) for degree in self.chebyshev_degrees)
        return (
            f"SU({self.n}), degrees=({degrees}), "
            f"alpha={float(self.alpha.detach()):.6g}, "
            f"cubic={float(self.cubic.detach()):.6g}, "
            f"scale={float(self.scale.detach()):.6g}"
        )


class SU2AdjointSpectralAnsatz(SUNAdjointRadialSpectralAnsatz):
    r"""SU(2) adjoint spectral ansatz with exact odd adjoint covariance."""

    def __init__(
        self,
        *,
        omega_init: float = 1.0,
        quartic_init: float = 0.0,
        hidden_layers: Sequence[int] = (32, 32),
        alpha_floor: float = 1.0e-8,
        cubic_floor: float = 0.0,
        tail_eps: float = 1.0e-12,
        activation: type[nn.Module] = nn.Tanh,
        zero_final: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            n=2,
            omega_init=omega_init,
            quartic_init=quartic_init,
            hidden_layers=hidden_layers,
            alpha_floor=alpha_floor,
            cubic_floor=cubic_floor,
            tail_eps=tail_eps,
            activation=activation,
            zero_final=zero_final,
            dtype=dtype,
        )


@dataclass(frozen=True)
class AdjointEnergyTerms:
    local_energy: torch.Tensor
    numerator_density: torch.Tensor
    denominator_density: torch.Tensor
    radial: torch.Tensor
    angular: torch.Tensor
    potential: torch.Tensor
    head_norm: torch.Tensor


@dataclass(frozen=True)
class AdjointTrainingRecord:
    step: int
    energy: float
    radial: float
    angular: float
    potential: float
    local_energy_std: float
    alpha: float
    cubic: float = 0.0


@dataclass(frozen=True)
class AdjointImportanceSamples:
    """Fixed proposal samples on the traceless eigenvalue hyperplane."""

    z: torch.Tensor
    lam: torch.Tensor
    log_prob: torch.Tensor
    sigma: float
    seed: int
    scrambled: bool


@dataclass(frozen=True)
class AdjointObservables:
    energy: float
    radial: float
    angular: float
    potential: float
    norm: float
    local_energy_mean: float
    local_energy_std: float
    traceless_residual: float
    parity_residual: float
    alpha: float
    cubic: float = 0.0


@dataclass(frozen=True)
class AdjointMomentObservables:
    """Moment and theorem diagnostics for an adjoint variational state."""

    tr_x2: float
    tr_x4: float
    kinetic: float
    potential: float
    virial_rhs: float
    virial_residual: float


@dataclass(frozen=True)
class AdjointStructureDiagnostics:
    """Exact-symmetry and collision-regularity diagnostics for an adjoint ansatz."""

    traceless_residual: float
    parity_residual: float
    weyl_residual: float
    head_collision_residual: float
    profile_collision_residual: float
    head_collision_ratio_max_abs: float = 0.0
    profile_collision_identity_residual: float = 0.0


@dataclass(frozen=True)
class SU2RadialFiniteDifferenceResult:
    """Lowest radial finite-difference eigenpair for the SU(2) adjoint sector."""

    energy: float
    r: torch.Tensor
    u: torch.Tensor
    dr: float


def adjoint_dirichlet_terms(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    gap_eps: float = 1.0e-12,
) -> AdjointEnergyTerms:
    """Compute first-derivative adjoint-sector Dirichlet terms."""

    n = model.n
    if lam.ndim != 2 or lam.shape[-1] != n:
        raise ValueError(f"lam must have shape (batch, {n})")
    if gap_eps < 0:
        raise ValueError("gap_eps must be non-negative")

    lam_req = tangent_project(lam.detach().clone()).requires_grad_(True)
    action = model.action(lam_req)
    head = model.head(lam_req)
    (action_grad,) = torch.autograd.grad(
        action,
        lam_req,
        grad_outputs=torch.ones_like(action),
        create_graph=True,
    )
    action_grad = tangent_project(action_grad)

    radial = torch.zeros(lam_req.shape[0], dtype=lam_req.dtype, device=lam_req.device)
    for i in range(n):
        (head_grad,) = torch.autograd.grad(
            head[:, i],
            lam_req,
            grad_outputs=torch.ones_like(head[:, i]),
            retain_graph=True,
            create_graph=True,
        )
        tangent_head_grad = tangent_project(head_grad)
        covariant_grad = tangent_head_grad - 0.5 * head[:, i, None] * action_grad
        radial = radial + torch.sum(covariant_grad**2, dim=-1)
    radial = 0.5 * radial

    angular = torch.zeros(
        lam_req.shape[0],
        dtype=lam_req.dtype,
        device=lam_req.device,
    )
    for i in range(n):
        for j in range(i + 1, n):
            gap = lam_req[:, i] - lam_req[:, j]
            if gap_eps > 0.0:
                sign = torch.where(gap >= 0.0, 1.0, -1.0)
                gap = torch.where(torch.abs(gap) < gap_eps, sign * gap_eps, gap)
            angular = angular + ((head[:, i] - head[:, j]) / gap) ** 2
    potential = adjoint_matrix_potential(
        lam_req,
        omega=omega,
        coupling=coupling,
    )
    head_norm = torch.sum(head**2, dim=-1)
    local_energy = potential + (radial + angular) / head_norm
    numerator_density = radial + angular + potential * head_norm
    return AdjointEnergyTerms(
        local_energy=local_energy,
        numerator_density=numerator_density,
        denominator_density=head_norm,
        radial=radial,
        angular=angular,
        potential=potential,
        head_norm=head_norm,
    )


def _shifted_weighted_ratio(
    log_weights: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift = torch.max(log_weights.detach())
    weights = torch.exp(log_weights - shift)
    num = torch.sum(weights * numerator)
    den = torch.sum(weights * denominator)
    norm = den * torch.exp(shift)
    return num / den, norm


def sobol_gaussian_traceless_samples(
    n: int,
    n_samples: int,
    *,
    sigma: float = 2.0,
    seed: int = 1234,
    scramble: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> AdjointImportanceSamples:
    """Draw deterministic Sobol-Gaussian proposal samples in traceless coords."""

    if n < 2:
        raise ValueError("n must be at least two")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dimension = n - 1
    engine = torch.quasirandom.SobolEngine(
        dimension=dimension,
        scramble=scramble,
        seed=seed,
    )
    uniform = engine.draw(n_samples).to(dtype=dtype, device=device)
    tiny = torch.finfo(dtype).eps
    uniform = torch.clamp(uniform, min=tiny, max=1.0 - tiny)
    sigma_tensor = torch.as_tensor(sigma, dtype=dtype, device=device)
    z = sigma_tensor * sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    lam = eigenvalues_from_traceless_coordinates(z, n=n)
    log_norm = -0.5 * dimension * torch.log(
        torch.as_tensor(2.0 * pi * sigma**2, dtype=dtype, device=device)
    )
    log_prob = log_norm - 0.5 * torch.sum((z / sigma_tensor) ** 2, dim=-1)
    return AdjointImportanceSamples(
        z=z,
        lam=lam,
        log_prob=log_prob,
        sigma=float(sigma),
        seed=int(seed),
        scrambled=bool(scramble),
    )


def adjoint_quadrature_energy(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable SU(2) adjoint Rayleigh quotient."""

    if weights.ndim != 1 or weights.shape[0] != lam.shape[0]:
        raise ValueError("weights must have shape (batch,)")
    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    action = model.action(tangent_project(lam))
    log_weights = (
        torch.log(weights)
        + 2.0 * log_vandermonde_abs(lam)
        - action
    )
    energy, norm = _shifted_weighted_ratio(
        log_weights,
        terms.numerator_density,
        terms.denominator_density,
    )
    radial, _ = _shifted_weighted_ratio(
        log_weights,
        terms.radial,
        terms.denominator_density,
    )
    angular, _ = _shifted_weighted_ratio(
        log_weights,
        terms.angular,
        terms.denominator_density,
    )
    potential, _ = _shifted_weighted_ratio(
        log_weights,
        terms.potential * terms.head_norm,
        terms.denominator_density,
    )
    return energy, radial, angular, potential, norm


def adjoint_importance_energy(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fixed-proposal importance-sampled Rayleigh quotient."""

    if log_prob.ndim != 1 or log_prob.shape[0] != lam.shape[0]:
        raise ValueError("log_prob must have shape (batch,)")
    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    action = model.action(tangent_project(lam))
    log_weights = 2.0 * log_vandermonde_abs(lam) - action - log_prob
    energy, norm = _shifted_weighted_ratio(
        log_weights,
        terms.numerator_density,
        terms.denominator_density,
    )
    radial, _ = _shifted_weighted_ratio(
        log_weights,
        terms.radial,
        terms.denominator_density,
    )
    angular, _ = _shifted_weighted_ratio(
        log_weights,
        terms.angular,
        terms.denominator_density,
    )
    potential, _ = _shifted_weighted_ratio(
        log_weights,
        terms.potential * terms.head_norm,
        terms.denominator_density,
    )
    return energy, radial, angular, potential, norm


def adjoint_profile_norm(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    r"""Return ``\int d\lambda_T Delta**2 sum_i q_i(lambda)**2``.

    This is the adjoint-sector norm of the matrix-valued spectral profile on a
    supplied quadrature grid.  It is useful for plotting normalized profile
    components; the raw components have an arbitrary overall scale.
    """

    if lam.ndim != 2 or lam.shape[-1] != model.n:
        raise ValueError(f"lam must have shape (batch, {model.n})")
    if weights.ndim != 1 or weights.shape[0] != lam.shape[0]:
        raise ValueError("weights must have shape (batch,)")
    profile = model.profile(lam)
    profile_norm_density = torch.sum(profile**2, dim=-1)
    vandermonde_squared = torch.exp(2.0 * log_vandermonde_abs(lam))
    return torch.sum(weights * vandermonde_squared * profile_norm_density)


def _collision_probe_points(
    n: int,
    *,
    eps: float,
    collision_center: float,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if n < 2:
        raise ValueError("n must be at least two")
    pairs: list[tuple[int, int]] = []
    near_points: list[torch.Tensor] = []
    limit_points: list[torch.Tensor] = []
    center = torch.as_tensor(collision_center, dtype=dtype, device=device)
    eps_tensor = torch.as_tensor(eps, dtype=dtype, device=device)
    for i in range(n):
        for j in range(i + 1, n):
            near = torch.zeros(n, dtype=dtype, device=device)
            limit = torch.zeros(n, dtype=dtype, device=device)
            if n == 2:
                near[i] = 0.5 * eps_tensor
                near[j] = -0.5 * eps_tensor
            else:
                fill = -2.0 * center / (n - 2)
                near[:] = fill
                limit[:] = fill
                near[i] = center + 0.5 * eps_tensor
                near[j] = center - 0.5 * eps_tensor
                limit[i] = center
                limit[j] = center
            near_points.append(near)
            limit_points.append(limit)
            pairs.append((i, j))
    return torch.stack(near_points), torch.stack(limit_points), pairs


def adjoint_structure_diagnostics(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    *,
    collision_eps: float = 1.0e-7,
    collision_center: float = 0.7,
) -> AdjointStructureDiagnostics:
    """Return Weyl, parity, tracelessness, and collision-regularity residuals."""

    if lam.ndim != 2 or lam.shape[-1] != model.n:
        raise ValueError(f"lam must have shape (batch, {model.n})")
    n = model.n
    with torch.no_grad():
        profile = model.profile(lam)
        traceless = torch.max(torch.abs(torch.sum(profile, dim=-1)))
        parity = torch.max(torch.abs(model.profile(-lam) + profile))

        weyl = torch.zeros((), dtype=lam.dtype, device=lam.device)
        for perm in permutations(range(n)):
            permuted = lam[:, perm]
            permuted_profile = model.profile(permuted)
            expected = profile[:, perm]
            weyl = torch.maximum(
                weyl,
                torch.max(torch.abs(permuted_profile - expected)),
            )

        near, limit, pairs = _collision_probe_points(
            n,
            eps=collision_eps,
            collision_center=collision_center,
            dtype=lam.dtype,
            device=lam.device,
        )
        head = model.head(near)
        near_profile = model.profile(near)
        limit_action = model.action(limit)
        near_action = model.action(near)
        expected_profile_ratio = torch.exp(-0.5 * limit_action)
        head_collision = torch.zeros((), dtype=lam.dtype, device=lam.device)
        profile_collision = torch.zeros((), dtype=lam.dtype, device=lam.device)
        head_ratio_max = torch.zeros((), dtype=lam.dtype, device=lam.device)
        profile_identity = torch.zeros((), dtype=lam.dtype, device=lam.device)
        for row, (i, j) in enumerate(pairs):
            gap = near[row, i] - near[row, j]
            head_ratio = (head[row, i] - head[row, j]) / gap
            profile_ratio = (near_profile[row, i] - near_profile[row, j]) / gap
            identity_ratio = torch.exp(-0.5 * near_action[row]) * head_ratio
            head_collision = torch.maximum(
                head_collision,
                torch.abs(head_ratio - 1.0),
            )
            head_ratio_max = torch.maximum(head_ratio_max, torch.abs(head_ratio))
            profile_collision = torch.maximum(
                profile_collision,
                torch.abs(profile_ratio - expected_profile_ratio[row]),
            )
            profile_identity = torch.maximum(
                profile_identity,
                torch.abs(profile_ratio - identity_ratio),
            )

    return AdjointStructureDiagnostics(
        traceless_residual=float(traceless.detach()),
        parity_residual=float(parity.detach()),
        weyl_residual=float(weyl.detach()),
        head_collision_residual=float(head_collision.detach()),
        profile_collision_residual=float(profile_collision.detach()),
        head_collision_ratio_max_abs=float(head_ratio_max.detach()),
        profile_collision_identity_residual=float(profile_identity.detach()),
    )


def adjoint_quadrature_observables(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> AdjointObservables:
    """Return SU(2) adjoint benchmark observables on a fixed grid."""

    energy, radial, angular, potential, norm = adjoint_quadrature_energy(
        model,
        lam,
        weights,
        omega=omega,
        coupling=coupling,
    )
    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    with torch.no_grad():
        action = model.action(tangent_project(lam))
        log_weights = (
            torch.log(weights)
            + 2.0 * log_vandermonde_abs(lam)
            - action
            + torch.log(terms.head_norm)
        )
        finite = torch.isfinite(log_weights) & torch.isfinite(terms.local_energy)
        finite_log_weights = torch.where(
            finite,
            log_weights,
            torch.full_like(log_weights, -torch.inf),
        )
        shift = torch.max(finite_log_weights)
        density_weights = torch.where(
            finite,
            torch.exp(finite_log_weights - shift),
            torch.zeros_like(finite_log_weights),
        )
        density_weights = density_weights / torch.sum(density_weights)
        finite_local_energy = torch.where(
            finite,
            terms.local_energy,
            torch.zeros_like(terms.local_energy),
        )
        local_mean = torch.sum(density_weights * finite_local_energy)
        local_variance = torch.sum(
            density_weights * (finite_local_energy - local_mean) ** 2
        )
        profile = model.profile(lam)
        parity = torch.max(torch.abs(model.profile(-lam) + profile))

    return AdjointObservables(
        energy=float(energy.detach()),
        radial=float(radial.detach()),
        angular=float(angular.detach()),
        potential=float(potential.detach()),
        norm=float(norm.detach()),
        local_energy_mean=float(local_mean.detach()),
        local_energy_std=float(torch.sqrt(local_variance).detach()),
        traceless_residual=float(
            torch.max(torch.abs(torch.sum(profile, dim=-1))).detach()
        ),
        parity_residual=float(parity.detach()),
        alpha=float(model.alpha.detach()),
        cubic=float(model.cubic.detach()),
    )


def adjoint_importance_observables(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> AdjointObservables:
    """Return adjoint observables from fixed-proposal importance samples."""

    energy, radial, angular, potential, norm = adjoint_importance_energy(
        model,
        lam,
        log_prob,
        omega=omega,
        coupling=coupling,
    )
    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    with torch.no_grad():
        action = model.action(tangent_project(lam))
        log_weights = (
            2.0 * log_vandermonde_abs(lam)
            - action
            - log_prob
            + torch.log(terms.head_norm)
        )
        finite = torch.isfinite(log_weights) & torch.isfinite(terms.local_energy)
        finite_log_weights = torch.where(
            finite,
            log_weights,
            torch.full_like(log_weights, -torch.inf),
        )
        shift = torch.max(finite_log_weights)
        density_weights = torch.where(
            finite,
            torch.exp(finite_log_weights - shift),
            torch.zeros_like(finite_log_weights),
        )
        density_weights = density_weights / torch.sum(density_weights)
        finite_local_energy = torch.where(
            finite,
            terms.local_energy,
            torch.zeros_like(terms.local_energy),
        )
        local_mean = torch.sum(density_weights * finite_local_energy)
        local_variance = torch.sum(
            density_weights * (finite_local_energy - local_mean) ** 2
        )
        profile = model.profile(lam)
        parity = torch.max(torch.abs(model.profile(-lam) + profile))

    return AdjointObservables(
        energy=float(energy.detach()),
        radial=float(radial.detach()),
        angular=float(angular.detach()),
        potential=float(potential.detach()),
        norm=float(norm.detach()),
        local_energy_mean=float(local_mean.detach()),
        local_energy_std=float(torch.sqrt(local_variance).detach()),
        traceless_residual=float(
            torch.max(torch.abs(torch.sum(profile, dim=-1))).detach()
        ),
        parity_residual=float(parity.detach()),
        alpha=float(model.alpha.detach()),
        cubic=float(model.cubic.detach()),
    )


def adjoint_quadrature_moments(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> AdjointMomentObservables:
    r"""Return ``Tr X**2``, ``Tr X**4``, and virial diagnostics.

    The virial theorem for
    ``H = -1/2 Delta_X + 1/2 omega**2 Tr X**2 + g Tr X**4`` is

    ``2<T> = omega**2 <Tr X**2> + 4g <Tr X**4>``.
    """

    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    action = model.action(tangent_project(lam))
    log_base_weights = (
        torch.log(weights)
        + 2.0 * log_vandermonde_abs(lam)
        - action
    )
    tr_x2 = torch.sum(lam**2, dim=-1)
    tr_x4 = torch.sum(lam**4, dim=-1)
    mean_tr_x2, _ = _shifted_weighted_ratio(
        log_base_weights,
        tr_x2 * terms.head_norm,
        terms.head_norm,
    )
    mean_tr_x4, _ = _shifted_weighted_ratio(
        log_base_weights,
        tr_x4 * terms.head_norm,
        terms.head_norm,
    )
    kinetic, _ = _shifted_weighted_ratio(
        log_base_weights,
        terms.radial + terms.angular,
        terms.head_norm,
    )
    potential, _ = _shifted_weighted_ratio(
        log_base_weights,
        terms.potential * terms.head_norm,
        terms.head_norm,
    )
    virial_rhs = omega**2 * mean_tr_x2 + 4.0 * coupling * mean_tr_x4
    virial_residual = 2.0 * kinetic - virial_rhs
    return AdjointMomentObservables(
        tr_x2=float(mean_tr_x2.detach()),
        tr_x4=float(mean_tr_x4.detach()),
        kinetic=float(kinetic.detach()),
        potential=float(potential.detach()),
        virial_rhs=float(virial_rhs.detach()),
        virial_residual=float(virial_residual.detach()),
    )


def adjoint_importance_moments(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> AdjointMomentObservables:
    """Return moment and virial diagnostics from fixed-proposal samples."""

    terms = adjoint_dirichlet_terms(
        model,
        lam,
        omega=omega,
        coupling=coupling,
    )
    action = model.action(tangent_project(lam))
    log_base_weights = 2.0 * log_vandermonde_abs(lam) - action - log_prob
    tr_x2 = torch.sum(lam**2, dim=-1)
    tr_x4 = torch.sum(lam**4, dim=-1)
    mean_tr_x2, _ = _shifted_weighted_ratio(
        log_base_weights,
        tr_x2 * terms.head_norm,
        terms.head_norm,
    )
    mean_tr_x4, _ = _shifted_weighted_ratio(
        log_base_weights,
        tr_x4 * terms.head_norm,
        terms.head_norm,
    )
    kinetic, _ = _shifted_weighted_ratio(
        log_base_weights,
        terms.radial + terms.angular,
        terms.head_norm,
    )
    potential, _ = _shifted_weighted_ratio(
        log_base_weights,
        terms.potential * terms.head_norm,
        terms.head_norm,
    )
    virial_rhs = omega**2 * mean_tr_x2 + 4.0 * coupling * mean_tr_x4
    virial_residual = 2.0 * kinetic - virial_rhs
    return AdjointMomentObservables(
        tr_x2=float(mean_tr_x2.detach()),
        tr_x4=float(mean_tr_x4.detach()),
        kinetic=float(kinetic.detach()),
        potential=float(potential.detach()),
        virial_rhs=float(virial_rhs.detach()),
        virial_residual=float(virial_residual.detach()),
    )


def train_adjoint_quadrature(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    n_steps: int = 1000,
    lr: float = 1.0e-2,
    report_every: int = 100,
) -> list[AdjointTrainingRecord]:
    """Train the SU(2) adjoint ansatz by deterministic quadrature."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if report_every < 1:
        raise ValueError("report_every must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[AdjointTrainingRecord] = []
    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        energy, radial, angular, potential, _ = adjoint_quadrature_energy(
            model,
            lam,
            weights,
            omega=omega,
            coupling=coupling,
        )
        energy.backward()
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == n_steps:
            obs = adjoint_quadrature_observables(
                model,
                lam,
                weights,
                omega=omega,
                coupling=coupling,
            )
            history.append(
                AdjointTrainingRecord(
                    step=step,
                    energy=obs.energy,
                    radial=obs.radial,
                    angular=obs.angular,
                    potential=obs.potential,
                    local_energy_std=obs.local_energy_std,
                    alpha=obs.alpha,
                    cubic=obs.cubic,
                )
            )
    return history


def train_adjoint_importance(
    model: SUNAdjointRadialSpectralAnsatz,
    lam: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    n_steps: int = 1000,
    lr: float = 1.0e-2,
    report_every: int = 100,
) -> list[AdjointTrainingRecord]:
    """Train the adjoint ansatz by a fixed-proposal Rayleigh quotient."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if report_every < 1:
        raise ValueError("report_every must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[AdjointTrainingRecord] = []
    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        energy, _, _, _, _ = adjoint_importance_energy(
            model,
            lam,
            log_prob,
            omega=omega,
            coupling=coupling,
        )
        energy.backward()
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == n_steps:
            obs = adjoint_importance_observables(
                model,
                lam,
                log_prob,
                omega=omega,
                coupling=coupling,
            )
            history.append(
                AdjointTrainingRecord(
                    step=step,
                    energy=obs.energy,
                    radial=obs.radial,
                    angular=obs.angular,
                    potential=obs.potential,
                    local_energy_std=obs.local_energy_std,
                    alpha=obs.alpha,
                    cubic=obs.cubic,
                )
            )
    return history


def exact_su2_harmonic_adjoint_energy(omega: float = 1.0) -> float:
    """Exact lowest adjoint-sector energy for the SU(2) harmonic model."""

    return exact_suN_harmonic_adjoint_energy(2, omega)


def exact_suN_harmonic_adjoint_energy(n: int, omega: float = 1.0) -> float:
    """Exact lowest adjoint-sector energy for the SU(N) harmonic model."""

    if n < 2:
        raise ValueError("n must be at least two")
    if omega <= 0:
        raise ValueError("omega must be positive")
    dimension = n**2 - 1
    return (0.5 * dimension + 1.0) * omega


def _su2_adjoint_radial_grid(
    *,
    r_max: float,
    n_grid: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if r_max <= 0:
        raise ValueError("r_max must be positive")
    if n_grid < 10:
        raise ValueError("n_grid must be at least 10")
    dr = torch.as_tensor(r_max / (n_grid + 1), dtype=dtype, device=device)
    r = dr * torch.arange(1, n_grid + 1, dtype=dtype, device=device)
    return r, dr


def _suN_adjoint_radial_hamiltonian_diagonals(
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    n: int,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n not in (2, 3):
        raise ValueError("this radial benchmark is implemented only for n=2 or n=3")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if coupling < 0:
        raise ValueError("coupling must be non-negative")
    dr_tensor = torch.as_tensor(dr, dtype=r.dtype, device=r.device)
    dimension = n**2 - 1
    angular_momentum = 1
    reduced_coefficient = 0.5 * (
        (angular_momentum + 0.5 * (dimension - 2)) ** 2 - 0.25
    )
    centrifugal = reduced_coefficient / r**2
    diagonal = (
        1.0 / dr_tensor**2
        + centrifugal
        + 0.5 * omega**2 * r**2
        + 0.5 * coupling * r**4
    )
    off_diagonal = torch.full(
        (r.shape[0] - 1,),
        -0.5 / dr_tensor**2,
        dtype=r.dtype,
        device=r.device,
    )
    return diagonal, off_diagonal


def su2_adjoint_radial_hamiltonian_apply(
    u: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> torch.Tensor:
    """Apply the SU(2) adjoint ``l=1`` radial finite-difference Hamiltonian."""

    return suN_adjoint_radial_hamiltonian_apply(
        2,
        u,
        r,
        dr,
        omega=omega,
        coupling=coupling,
    )


def suN_adjoint_radial_hamiltonian_apply(
    n: int,
    u: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> torch.Tensor:
    """Apply the SU(N), ``l=1`` radial benchmark Hamiltonian for N=2 or N=3."""

    if u.ndim != 1 or r.ndim != 1 or u.shape != r.shape:
        raise ValueError("u and r must be one-dimensional tensors with same shape")
    diagonal, off_diagonal = _suN_adjoint_radial_hamiltonian_diagonals(
        r,
        dr,
        n=n,
        omega=omega,
        coupling=coupling,
    )
    h_u = diagonal * u
    h_u[:-1] = h_u[:-1] + off_diagonal * u[1:]
    h_u[1:] = h_u[1:] + off_diagonal * u[:-1]
    return h_u


def su2_adjoint_radial_inner(
    u: torch.Tensor,
    v: torch.Tensor,
    dr: torch.Tensor | float,
) -> torch.Tensor:
    """Return the radial inner product ``int_0^infty dr u(r) v(r)``."""

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    dr_tensor = torch.as_tensor(dr, dtype=u.dtype, device=u.device)
    return torch.sum(dr_tensor * u * v)


def su2_adjoint_radial_normalize(
    u: torch.Tensor,
    dr: torch.Tensor | float,
) -> torch.Tensor:
    """Normalize a reduced radial wavefunction with ``int dr u**2 = 1``."""

    norm = torch.sqrt(su2_adjoint_radial_inner(u, u, dr))
    return u / norm


def su2_adjoint_model_radial_wavefunction(
    model: SU2AdjointSpectralAnsatz,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    r"""Return the reduced radial wavefunction ``u(r)`` implied by the model.

    The SU(2) adjoint component wavefunction has the form
    ``Psi_a(x) = x_a exp[-S_theta(r)/2]``.  The corresponding reduced radial
    wavefunction is therefore proportional to ``r**2 exp[-S_theta(r)/2]``.
    """

    return suN_adjoint_model_radial_wavefunction(
        model,
        r,
        dr,
        normalize=normalize,
    )


def suN_adjoint_model_radial_wavefunction(
    model: SUNAdjointRadialSpectralAnsatz,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    r"""Return the reduced radial wavefunction ``u(r)`` implied by the model."""

    if r.ndim != 1:
        raise ValueError("r must be one-dimensional")
    n = model.n
    dimension = n**2 - 1
    lam = torch.zeros((r.shape[0], n), dtype=r.dtype, device=r.device)
    lam[:, 0] = r / sqrt(2.0)
    lam[:, 1] = -r / sqrt(2.0)
    radial_power = 0.5 * (dimension + 1)
    u = r**radial_power * torch.exp(-0.5 * model.action(lam))
    if normalize:
        u = su2_adjoint_radial_normalize(u, dr)
    return u


def su2_adjoint_radial_moment(
    u: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    power: int,
) -> torch.Tensor:
    """Return ``<r**power>`` for a normalized reduced radial wavefunction."""

    if power < 0:
        raise ValueError("power must be non-negative")
    if u.shape != r.shape:
        raise ValueError("u and r must have the same shape")
    return su2_adjoint_radial_inner(u * r**power, u, dr)


def su2_adjoint_radial_residual_norm(
    u: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    energy: float,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> torch.Tensor:
    r"""Return ``||H u - E u|| / ||u||`` on the radial finite-difference grid."""

    return suN_adjoint_radial_residual_norm(
        2,
        u,
        r,
        dr,
        energy=energy,
        omega=omega,
        coupling=coupling,
    )


def suN_adjoint_radial_residual_norm(
    n: int,
    u: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor | float,
    *,
    energy: float,
    omega: float = 1.0,
    coupling: float = 0.0,
) -> torch.Tensor:
    r"""Return ``||H u - E u|| / ||u||`` for the SU(N) radial benchmark."""

    h_u = suN_adjoint_radial_hamiltonian_apply(
        n,
        u,
        r,
        dr,
        omega=omega,
        coupling=coupling,
    )
    residual = h_u - energy * u
    return torch.sqrt(
        su2_adjoint_radial_inner(residual, residual, dr)
        / su2_adjoint_radial_inner(u, u, dr)
    )


def su2_adjoint_radial_finite_difference_result(
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    r_max: float = 8.0,
    n_grid: int = 1200,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SU2RadialFiniteDifferenceResult:
    r"""Return the lowest ``l=1`` radial finite-difference eigenpair."""

    return suN_adjoint_radial_finite_difference_result(
        2,
        omega=omega,
        coupling=coupling,
        r_max=r_max,
        n_grid=n_grid,
        dtype=dtype,
        device=device,
    )


def suN_adjoint_radial_finite_difference_result(
    n: int,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    r_max: float = 8.0,
    n_grid: int = 1200,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SU2RadialFiniteDifferenceResult:
    r"""Return the lowest radial eigenpair for the SU(N) adjoint benchmark.

    This benchmark is valid for N=2 and N=3, where
    ``Tr X**4 = 0.5 * (Tr X**2)**2`` and the potential is radial in
    ``d=N**2-1`` matrix-space dimensions.
    """

    r, dr = _su2_adjoint_radial_grid(
        r_max=r_max,
        n_grid=n_grid,
        dtype=dtype,
        device=device,
    )
    diagonal, off_diagonal = _suN_adjoint_radial_hamiltonian_diagonals(
        r,
        dr,
        n=n,
        omega=omega,
        coupling=coupling,
    )
    hamiltonian = torch.diag(diagonal)
    hamiltonian = hamiltonian + torch.diag(off_diagonal, diagonal=1)
    hamiltonian = hamiltonian + torch.diag(off_diagonal, diagonal=-1)
    energies, vectors = torch.linalg.eigh(hamiltonian)
    u = vectors[:, 0] / torch.sqrt(dr)
    max_index = int(torch.argmax(torch.abs(u)).detach())
    if u[max_index] < 0:
        u = -u
    return SU2RadialFiniteDifferenceResult(
        energy=float(energies[0].detach()),
        r=r,
        u=u,
        dr=float(dr.detach()),
    )


def su2_adjoint_radial_finite_difference_energy(
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    r_max: float = 8.0,
    n_grid: int = 1200,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> float:
    r"""Finite-difference benchmark for the SU(2) adjoint radial problem.

    For SU(2), ``Tr X**4 = (Tr X**2)**2 / 2``.  The adjoint harmonic state is
    the ``l=1`` sector of a three-dimensional radial Schrodinger problem with
    potential ``0.5*omega**2*r**2 + 0.5*coupling*r**4``.
    """

    return su2_adjoint_radial_finite_difference_result(
        omega=omega,
        coupling=coupling,
        r_max=r_max,
        n_grid=n_grid,
        dtype=dtype,
        device=device,
    ).energy


def suN_adjoint_radial_finite_difference_energy(
    n: int,
    *,
    omega: float = 1.0,
    coupling: float = 0.0,
    r_max: float = 8.0,
    n_grid: int = 1200,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> float:
    """Return the SU(N) radial benchmark energy for N=2 or N=3."""

    return suN_adjoint_radial_finite_difference_result(
        n,
        omega=omega,
        coupling=coupling,
        r_max=r_max,
        n_grid=n_grid,
        dtype=dtype,
        device=device,
    ).energy


def su2_adjoint_component_wavefunction(
    model: SU2AdjointSpectralAnsatz,
    x: torch.Tensor,
) -> torch.Tensor:
    r"""Return the SU(2) adjoint wavefunction components for ``X=x_a T^a``."""

    if x.ndim != 2 or x.shape[-1] != 3:
        raise ValueError("x must have shape (batch, 3)")
    radius = torch.linalg.norm(x, dim=-1)
    lam = torch.stack(
        [radius / sqrt(2.0), -radius / sqrt(2.0)],
        dim=-1,
    ).to(dtype=x.dtype, device=x.device)
    scalar = torch.exp(-0.5 * model.action(lam))
    return scalar[:, None] * x
