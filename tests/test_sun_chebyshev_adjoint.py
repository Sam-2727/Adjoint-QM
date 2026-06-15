from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    SUNAdjointChebyshevSpectralAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_metropolis_observables,
    adjoint_importance_moments,
    adjoint_importance_observables,
    adjoint_quadrature_moments,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    adjoint_vmc_dirichlet_loss,
    centered_chebyshev_heads,
    exact_suN_harmonic_adjoint_energy,
    metropolis_sample,
    sobol_gaussian_traceless_samples,
    su3_adjoint_polar_eigenvalue_grid,
    train_adjoint_vmc_metropolis,
)
from adjoint_qm.ansatz import _inverse_softplus  # noqa: E402


def shared_flexible_chebyshev_model(n: int) -> SUNAdjointChebyshevSpectralAnsatz:
    return SUNAdjointChebyshevSpectralAnsatz(
        n=n,
        omega_init=1.0,
        quartic_tail_init=0.0,
        moment_cutoff=6,
        hidden_layers=(8,),
        head_hidden_layers=(8,),
        chebyshev_degrees=(1, 3, 5),
        scale_init=3.0,
        action_correction_scale=0.25,
        head_correction_scale=0.25,
        dtype=torch.float64,
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


def test_coordinate_scale_derivative_matches_virial_residual() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = su3_adjoint_polar_eigenvalue_grid(
        6.0,
        60,
        72,
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=3,
        omega_init=1.0,
        quartic_tail_init=0.0,
        moment_cutoff=2,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        coordinate_scale_init=1.08,
        dtype=torch.float64,
    )

    def set_coordinate_scale(value: float) -> None:
        raw = _inverse_softplus(torch.as_tensor(value, dtype=torch.float64))
        with torch.no_grad():
            model.raw_coordinate_scale.copy_(raw)

    def energy_at_log_scale(log_scale_shift: float) -> float:
        set_coordinate_scale(1.08 * torch.exp(torch.tensor(log_scale_shift)).item())
        return adjoint_quadrature_observables(
            model,
            lam,
            weights,
            coupling=0.0,
        ).energy

    step = 1.0e-4
    finite_difference = (
        energy_at_log_scale(step) - energy_at_log_scale(-step)
    ) / (2.0 * step)
    set_coordinate_scale(1.08)
    moments = adjoint_quadrature_moments(model, lam, weights, coupling=0.0)

    assert finite_difference == pytest.approx(moments.virial_residual, abs=1.0e-7)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_flexible_chebyshev_metropolis_harmonic_local_energy(n: int) -> None:
    torch.set_default_dtype(torch.float64)
    model = shared_flexible_chebyshev_model(n)
    result = metropolis_sample(
        model,
        n_samples=768,
        dim=n - 1,
        n_chains=64,
        step_size=0.9 / (n - 1) ** 0.5,
        burn_in=200,
        thinning=4,
        seed=9100 + n,
        dtype=torch.float64,
    )
    obs = adjoint_metropolis_observables(model, result.samples, coupling=0.0)
    exact_energy = exact_suN_harmonic_adjoint_energy(n, 1.0)

    assert 0.15 < result.acceptance_rate < 0.95
    assert abs(obs.energy - exact_energy) < 5.0 * obs.local_energy_stderr
    assert obs.local_energy_std > 0.0
    assert abs(obs.virial_residual) < 6.0e-1


def test_flexible_chebyshev_vmc_dirichlet_loss_is_finite_for_exact_harmonic() -> None:
    torch.set_default_dtype(torch.float64)
    model = shared_flexible_chebyshev_model(3)
    result = metropolis_sample(
        model,
        n_samples=512,
        dim=2,
        n_chains=64,
        step_size=0.7,
        burn_in=150,
        thinning=3,
        seed=9303,
        dtype=torch.float64,
    )
    loss, energy, local_std = adjoint_vmc_dirichlet_loss(
        model,
        result.samples,
        coupling=0.0,
    )

    local_stderr = local_std.item() / (result.samples.shape[0] ** 0.5)
    assert abs(energy.item() - 5.0) < 5.0 * local_stderr
    assert local_std.item() > 0.0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("n", [2, 3])
def test_flexible_chebyshev_importance_harmonic_virial_is_precise(n: int) -> None:
    torch.set_default_dtype(torch.float64)
    model = shared_flexible_chebyshev_model(n)
    samples = sobol_gaussian_traceless_samples(
        n,
        32768,
        sigma=2.0,
        seed=9500 + n,
        dtype=torch.float64,
    )
    obs = adjoint_importance_observables(
        model,
        samples.lam,
        samples.log_prob,
        coupling=0.0,
    )
    moments = adjoint_importance_moments(
        model,
        samples.lam,
        samples.log_prob,
        coupling=0.0,
    )

    assert obs.energy == pytest.approx(
        exact_suN_harmonic_adjoint_energy(n, 1.0),
        abs=3.0e-4,
    )
    assert abs(moments.virial_residual) < 3.0e-4


def test_flexible_chebyshev_vmc_training_smoke() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(123)
    model = shared_flexible_chebyshev_model(2)
    history = train_adjoint_vmc_metropolis(
        model,
        coupling=0.05,
        n_steps=2,
        n_samples=256,
        n_chains=32,
        step_size=0.8,
        burn_in=80,
        thinning=2,
        lr=5.0e-4,
        seed=9402,
        report_every=1,
        dtype=torch.float64,
    )

    assert len(history) == 2
    assert all(record.sample_count == 256 for record in history)
    assert all(0.1 < record.acceptance_rate < 0.95 for record in history)
    assert all(record.energy > 0.0 for record in history)
    assert bool(torch.isfinite(model.alpha).item())
