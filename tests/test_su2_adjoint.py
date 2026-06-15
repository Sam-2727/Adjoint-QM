from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    SU2AdjointSpectralAnsatz,
    adjoint_dirichlet_terms,
    adjoint_profile_norm,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    log_vandermonde_abs,
    eigenvalues_from_traceless_coordinates,
    exact_su2_harmonic_adjoint_energy,
    su2_adjoint_component_wavefunction,
    su2_adjoint_eigenvalue_grid,
    su2_adjoint_model_radial_wavefunction,
    su2_adjoint_radial_finite_difference_energy,
    su2_adjoint_radial_finite_difference_result,
    su2_adjoint_radial_inner,
    su2_adjoint_radial_moment,
    su2_adjoint_radial_residual_norm,
    traceless_hyperplane_basis,
    train_adjoint_quadrature,
)


def test_traceless_hyperplane_basis_is_orthonormal() -> None:
    torch.set_default_dtype(torch.float64)
    basis = traceless_hyperplane_basis(4, dtype=torch.float64)

    identity_error = torch.max(
        torch.abs(basis.T @ basis - torch.eye(3, dtype=torch.float64))
    )
    assert identity_error.item() < 1.0e-12
    assert torch.max(torch.abs(torch.sum(basis, dim=0))).item() < 1.0e-12


def test_su2_adjoint_ansatz_traceless_weyl_and_odd() -> None:
    torch.set_default_dtype(torch.float64)
    model = SU2AdjointSpectralAnsatz(
        omega_init=1.0,
        hidden_layers=(8,),
        dtype=torch.float64,
    )
    z = torch.tensor([[-1.4], [-0.3], [0.7], [2.1]], dtype=torch.float64)
    lam = eigenvalues_from_traceless_coordinates(z, n=2)
    permuted = lam[:, [1, 0]]

    profile = model.profile(lam)
    permuted_profile = model.profile(permuted)
    head = model.head(lam)
    terms = adjoint_dirichlet_terms(model, lam, omega=1.0, coupling=0.0)

    assert torch.max(torch.abs(torch.sum(profile, dim=-1))).item() < 1.0e-12
    assert torch.max(torch.abs(model.profile(-lam) + profile)).item() < 1.0e-12
    assert torch.max(torch.abs(permuted_profile - profile[:, [1, 0]])).item() < 1.0e-12
    assert torch.max(torch.abs(head[:, 0] - lam[:, 0])).item() < 1.0e-12
    assert torch.all(torch.isfinite(terms.local_energy))

    diagnostics = adjoint_structure_diagnostics(model, lam)
    assert diagnostics.traceless_residual < 1.0e-12
    assert diagnostics.parity_residual < 1.0e-12
    assert diagnostics.weyl_residual < 1.0e-12
    assert diagnostics.head_collision_residual < 1.0e-12
    assert diagnostics.profile_collision_residual < 1.0e-12


def test_su2_harmonic_adjoint_energy_is_exact_by_quadrature() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = su2_adjoint_eigenvalue_grid(8.0, 2000, dtype=torch.float64)
    model = SU2AdjointSpectralAnsatz(
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
        exact_su2_harmonic_adjoint_energy(1.0),
        abs=1.0e-10,
    )
    assert obs.traceless_residual < 1.0e-12
    assert obs.parity_residual < 1.0e-12


def test_su2_adjoint_profile_normalization() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = su2_adjoint_eigenvalue_grid(8.0, 2000, dtype=torch.float64)
    model = SU2AdjointSpectralAnsatz(
        omega_init=1.0,
        hidden_layers=(),
        dtype=torch.float64,
    )

    norm = adjoint_profile_norm(model, lam, weights)
    normalized_profile = model.profile(lam) / torch.sqrt(norm)
    normalized_norm = torch.sum(
        weights
        * torch.exp(2.0 * log_vandermonde_abs(lam))
        * torch.sum(normalized_profile**2, dim=-1)
    )

    assert normalized_norm.item() == pytest.approx(1.0, abs=1.0e-12)


def test_su2_harmonic_radial_wavefunction_matches_finite_difference() -> None:
    torch.set_default_dtype(torch.float64)
    radial = su2_adjoint_radial_finite_difference_result(
        omega=1.0,
        coupling=0.0,
        r_max=9.0,
        n_grid=900,
        dtype=torch.float64,
    )
    model = SU2AdjointSpectralAnsatz(
        omega_init=1.0,
        hidden_layers=(),
        dtype=torch.float64,
    )
    u_model = su2_adjoint_model_radial_wavefunction(model, radial.r, radial.dr)
    overlap = su2_adjoint_radial_inner(u_model, radial.u, radial.dr)
    residual = su2_adjoint_radial_residual_norm(
        u_model,
        radial.r,
        radial.dr,
        energy=exact_su2_harmonic_adjoint_energy(1.0),
        omega=1.0,
        coupling=0.0,
    )

    assert radial.energy == pytest.approx(2.5, abs=3.0e-5)
    assert su2_adjoint_radial_moment(radial.u, radial.r, radial.dr, 2).item() == (
        pytest.approx(2.5, abs=6.0e-5)
    )
    assert su2_adjoint_radial_moment(radial.u, radial.r, radial.dr, 4).item() == (
        pytest.approx(8.75, abs=5.0e-4)
    )
    assert abs(overlap.item()) > 0.999999
    assert residual.item() < 1.0e-4


def test_su2_harmonic_training_converges_to_adjoint_ground_state() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(123)
    _, lam, weights = su2_adjoint_eigenvalue_grid(7.0, 1200, dtype=torch.float64)
    model = SU2AdjointSpectralAnsatz(
        omega_init=0.45,
        hidden_layers=(12,),
        dtype=torch.float64,
    )

    train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=0.0,
        n_steps=180,
        lr=1.0e-2,
        report_every=180,
    )
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=0.0,
    )

    assert obs.energy == pytest.approx(2.5, abs=2.0e-3)
    assert obs.traceless_residual < 1.0e-12
    assert obs.parity_residual < 1.0e-12


def test_su2_quartic_training_matches_radial_benchmark() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(321)
    coupling = 0.05
    _, lam, weights = su2_adjoint_eigenvalue_grid(8.0, 1400, dtype=torch.float64)
    model = SU2AdjointSpectralAnsatz(
        omega_init=1.0,
        hidden_layers=(16,),
        dtype=torch.float64,
    )
    radial_reference = su2_adjoint_radial_finite_difference_energy(
        omega=1.0,
        coupling=coupling,
        r_max=9.0,
        n_grid=900,
        dtype=torch.float64,
    )

    train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=coupling,
        n_steps=350,
        lr=1.0e-2,
        report_every=350,
    )
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=coupling,
    )

    assert obs.energy == pytest.approx(radial_reference, abs=2.0e-3)
    assert obs.traceless_residual < 1.0e-12
    assert obs.parity_residual < 1.0e-12


def test_su2_quartic_radial_wavefunction_checks() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(321)
    coupling = 0.05
    _, lam, weights = su2_adjoint_eigenvalue_grid(8.0, 1400, dtype=torch.float64)
    model = SU2AdjointSpectralAnsatz(
        omega_init=1.0,
        hidden_layers=(16,),
        dtype=torch.float64,
    )
    radial = su2_adjoint_radial_finite_difference_result(
        omega=1.0,
        coupling=coupling,
        r_max=9.0,
        n_grid=900,
        dtype=torch.float64,
    )

    train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=coupling,
        n_steps=350,
        lr=1.0e-2,
        report_every=350,
    )
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=1.0,
        coupling=coupling,
    )
    u_model = su2_adjoint_model_radial_wavefunction(model, radial.r, radial.dr)
    overlap = su2_adjoint_radial_inner(u_model, radial.u, radial.dr)
    if overlap < 0:
        overlap = -overlap
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
    residual = su2_adjoint_radial_residual_norm(
        u_model,
        radial.r,
        radial.dr,
        energy=obs.energy,
        omega=1.0,
        coupling=coupling,
    )
    delta = 1.0e-3
    e_plus = su2_adjoint_radial_finite_difference_energy(
        omega=1.0,
        coupling=coupling + delta,
        r_max=9.0,
        n_grid=900,
        dtype=torch.float64,
    )
    e_minus = su2_adjoint_radial_finite_difference_energy(
        omega=1.0,
        coupling=coupling - delta,
        r_max=9.0,
        n_grid=900,
        dtype=torch.float64,
    )
    hellmann_feynman_error = 0.5 * r4_model - (e_plus - e_minus) / (2.0 * delta)

    assert overlap.item() > 0.999
    assert abs((r2_model - r2_reference).item()) < 1.0e-2
    assert abs((r4_model - r4_reference).item()) < 5.0e-2
    assert abs(virial_residual.item()) < 5.0e-3
    assert abs(hellmann_feynman_error.item()) < 5.0e-3
    assert residual.item() < 5.0e-2


def test_su2_component_wavefunction_is_so3_covariant() -> None:
    torch.set_default_dtype(torch.float64)
    model = SU2AdjointSpectralAnsatz(
        omega_init=1.0,
        hidden_layers=(8,),
        dtype=torch.float64,
    )
    x = torch.tensor(
        [[0.2, -0.7, 1.1], [1.0, 0.4, -0.3]],
        dtype=torch.float64,
    )
    theta = torch.as_tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    psi = su2_adjoint_component_wavefunction(model, x)
    rotated_psi = su2_adjoint_component_wavefunction(model, x @ rotation.T)

    assert torch.max(torch.abs(rotated_psi - psi @ rotation.T)).item() < 1.0e-12
