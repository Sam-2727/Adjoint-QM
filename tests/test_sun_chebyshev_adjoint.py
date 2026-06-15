from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    SUNAdjointChebyshevSpectralAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_quadrature_moments,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    centered_chebyshev_heads,
    exact_suN_harmonic_adjoint_energy,
    su3_adjoint_polar_eigenvalue_grid,
)


def test_centered_chebyshev_heads_are_traceless_and_weyl_covariant() -> None:
    torch.set_default_dtype(torch.float64)
    lam = torch.tensor(
        [
            [1.2, -0.4, 0.1, -0.9],
            [-1.0, 0.8, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )
    heads = centered_chebyshev_heads(lam, degrees=(1, 3, 5), scale=2.5)
    perm = [2, 0, 3, 1]
    permuted_heads = centered_chebyshev_heads(
        lam[:, perm],
        degrees=(1, 3, 5),
        scale=2.5,
    )

    assert torch.max(torch.abs(torch.sum(heads, dim=-1))).item() < 1.0e-12
    assert torch.max(torch.abs(permuted_heads - heads[:, :, perm])).item() < 1.0e-12


@pytest.mark.parametrize("n", [2, 3, 4])
def test_chebyshev_ansatz_reproduces_harmonic_adjoint_energy(n: int) -> None:
    torch.set_default_dtype(torch.float64)
    n_grid = {2: 2000, 3: 51, 4: 26}[n]
    _, lam, weights = adjoint_eigenvalue_grid(
        n,
        6.0,
        n_grid,
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=n,
        omega_init=1.0,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        dtype=torch.float64,
    )

    obs = adjoint_quadrature_observables(model, lam, weights, coupling=0.0)
    moments = adjoint_quadrature_moments(model, lam, weights, coupling=0.0)

    assert obs.energy == pytest.approx(
        exact_suN_harmonic_adjoint_energy(n, 1.0),
        abs=2.0e-6,
    )
    assert abs(moments.virial_residual) < 5.0e-6


def test_su3_polar_grid_reproduces_harmonic_adjoint_energy() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = su3_adjoint_polar_eigenvalue_grid(
        6.0,
        80,
        96,
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=3,
        omega_init=1.0,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        dtype=torch.float64,
    )

    obs = adjoint_quadrature_observables(model, lam, weights, coupling=0.0)

    assert obs.energy == pytest.approx(5.0, abs=1.0e-8)


def test_chebyshev_ansatz_structure_and_generic_collision_identity() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        hidden_layers=(8,),
        head_hidden_layers=(8,),
        chebyshev_degrees=(1, 3),
        scale_init=2.5,
        action_correction_scale=0.05,
        head_correction_scale=0.05,
        dtype=torch.float64,
    )
    lam = torch.tensor(
        [
            [0.9, -0.2, -0.4, -0.3],
            [1.0, 0.1, -0.7, -0.4],
            [-0.8, 0.6, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )

    diagnostics = adjoint_structure_diagnostics(model, lam)

    assert diagnostics.traceless_residual < 1.0e-12
    assert diagnostics.parity_residual < 1.0e-12
    assert diagnostics.weyl_residual < 1.0e-12
    assert diagnostics.head_collision_ratio_max_abs < 10.0
    assert diagnostics.profile_collision_identity_residual < 1.0e-9
