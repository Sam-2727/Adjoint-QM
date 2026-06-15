from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    SUNAdjointRadialSpectralAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    exact_suN_harmonic_adjoint_energy,
    su2_adjoint_radial_inner,
    su2_adjoint_radial_moment,
    suN_adjoint_model_radial_wavefunction,
    suN_adjoint_radial_finite_difference_result,
    suN_adjoint_radial_residual_norm,
    train_adjoint_quadrature,
)


def test_su3_adjoint_ansatz_weyl_and_collision_regular() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointRadialSpectralAnsatz(
        n=3,
        omega_init=1.0,
        quartic_init=2.0 / 3.0,
        hidden_layers=(8,),
        dtype=torch.float64,
    )
    lam = torch.tensor(
        [
            [0.8, -0.3, -0.5],
            [1.1, 0.2, -1.3],
            [-0.4, 1.0, -0.6],
        ],
        dtype=torch.float64,
    )

    diagnostics = adjoint_structure_diagnostics(model, lam)

    assert diagnostics.traceless_residual < 1.0e-12
    assert diagnostics.parity_residual < 1.0e-12
    assert diagnostics.weyl_residual < 1.0e-12
    assert diagnostics.head_collision_residual < 1.0e-12
    assert diagnostics.profile_collision_residual < 1.0e-6


def test_su3_harmonic_quadrature_matches_exact_adjoint_energy() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = adjoint_eigenvalue_grid(3, 5.0, 51, dtype=torch.float64)
    model = SUNAdjointRadialSpectralAnsatz(
        n=3,
        omega_init=1.0,
        hidden_layers=(),
        dtype=torch.float64,
    )

    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=0.0,
    )

    assert obs.energy == pytest.approx(
        exact_suN_harmonic_adjoint_energy(3, 1.0),
        abs=1.0e-4,
    )
    assert obs.traceless_residual < 1.0e-12
    assert obs.parity_residual < 1.0e-12


def test_su3_harmonic_radial_wavefunction_matches_finite_difference() -> None:
    torch.set_default_dtype(torch.float64)
    radial = suN_adjoint_radial_finite_difference_result(
        3,
        omega=1.0,
        coupling=0.0,
        r_max=8.0,
        n_grid=700,
        dtype=torch.float64,
    )
    model = SUNAdjointRadialSpectralAnsatz(
        n=3,
        omega_init=1.0,
        hidden_layers=(),
        dtype=torch.float64,
    )
    u_model = suN_adjoint_model_radial_wavefunction(model, radial.r, radial.dr)
    overlap = torch.abs(su2_adjoint_radial_inner(u_model, radial.u, radial.dr))
    residual = suN_adjoint_radial_residual_norm(
        3,
        u_model,
        radial.r,
        radial.dr,
        energy=exact_suN_harmonic_adjoint_energy(3, 1.0),
        omega=1.0,
        coupling=0.0,
    )

    assert radial.energy == pytest.approx(5.0, abs=3.0e-5)
    assert su2_adjoint_radial_moment(radial.u, radial.r, radial.dr, 2).item() == (
        pytest.approx(5.0, abs=6.0e-5)
    )
    assert su2_adjoint_radial_moment(radial.u, radial.r, radial.dr, 4).item() == (
        pytest.approx(30.0, abs=8.0e-4)
    )
    assert overlap.item() > 0.999999
    assert residual.item() < 1.0e-4


def test_su3_quartic_training_matches_radial_benchmark_at_g_one() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3033)
    coupling = 1.0
    _, lam, weights = adjoint_eigenvalue_grid(3, 5.0, 51, dtype=torch.float64)
    model = SUNAdjointRadialSpectralAnsatz(
        n=3,
        omega_init=1.0,
        quartic_init=2.0 / 3.0,
        hidden_layers=(16,),
        dtype=torch.float64,
    )
    radial = suN_adjoint_radial_finite_difference_result(
        3,
        omega=1.0,
        coupling=coupling,
        r_max=7.0,
        n_grid=900,
        dtype=torch.float64,
    )

    train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=coupling,
        n_steps=300,
        lr=1.0e-2,
        report_every=300,
    )
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=coupling,
    )
    u_model = suN_adjoint_model_radial_wavefunction(model, radial.r, radial.dr)
    overlap = su2_adjoint_radial_inner(u_model, radial.u, radial.dr)
    if overlap < 0:
        overlap = -overlap
        u_model = -u_model
    r2_model = su2_adjoint_radial_moment(u_model, radial.r, radial.dr, 2)
    r2_reference = su2_adjoint_radial_moment(radial.u, radial.r, radial.dr, 2)
    r4_model = su2_adjoint_radial_moment(u_model, radial.r, radial.dr, 4)
    r4_reference = su2_adjoint_radial_moment(radial.u, radial.r, radial.dr, 4)
    potential_model = 0.5 * r2_model + 0.5 * coupling * r4_model
    virial_residual = (
        2.0 * (obs.energy - potential_model)
        - r2_model
        - 2.0 * coupling * r4_model
    )
    residual = suN_adjoint_radial_residual_norm(
        3,
        u_model,
        radial.r,
        radial.dr,
        energy=obs.energy,
        omega=1.0,
        coupling=coupling,
    )

    assert obs.energy == pytest.approx(radial.energy, abs=2.0e-4)
    assert overlap.item() > 0.99999
    assert abs((r2_model - r2_reference).item()) < 1.0e-3
    assert abs((r4_model - r4_reference).item()) < 5.0e-3
    assert abs(virial_residual.item()) < 5.0e-4
    assert residual.item() < 2.0e-2
