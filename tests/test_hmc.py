from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    adjoint_hmc_log_norm_density,
    adjoint_lagged_log_weights,
    adjoint_lagged_reweighted_energy,
    hmc_sample,
    hmc_sample_randomized,
    kinetic_energy,
    leapfrog_integrate,
    ordered_traceless_gaussian_initial,
    potential_and_grad,
    hmc_warmup,
    traceless_hyperplane_basis,
)
from adjoint_qm.adjoint import tangent_project  # noqa: E402


def gaussian_log_prob(precision: torch.Tensor):
    def log_prob(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(precision * z * z, dim=-1)

    return log_prob


def hamiltonian(
    log_prob,
    z: torch.Tensor,
    momentum: torch.Tensor,
    mass: torch.Tensor | float | None = None,
) -> torch.Tensor:
    potential, _ = potential_and_grad(log_prob, z)
    return potential + kinetic_energy(momentum, mass)


class HarmonicAdjointHead(torch.nn.Module):
    """Simple exact adjoint head used only for HMC target tests."""

    def __init__(self, n: int, omega: float = 1.0) -> None:
        super().__init__()
        self.n = n
        self.omega = omega

    def action(self, lam: torch.Tensor) -> torch.Tensor:
        return self.omega * torch.sum(lam * lam, dim=-1)

    def head(self, lam: torch.Tensor) -> torch.Tensor:
        return tangent_project(lam)


def test_potential_gradient_matches_anisotropic_gaussian() -> None:
    torch.set_default_dtype(torch.float64)
    precision = torch.tensor([1.5, 0.7, 2.0], dtype=torch.float64)
    z = torch.tensor(
        [[0.2, -0.4, 1.3], [-1.1, 0.5, 0.7]],
        dtype=torch.float64,
    )

    potential, grad = potential_and_grad(gaussian_log_prob(precision), z)

    expected_potential = 0.5 * torch.sum(precision * z * z, dim=-1)
    expected_grad = precision * z
    assert torch.max(torch.abs(potential - expected_potential)).item() < 1.0e-12
    assert torch.max(torch.abs(grad - expected_grad)).item() < 1.0e-12


def test_leapfrog_is_reversible_for_gaussian_target() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(101)
    precision = torch.tensor([1.3, 0.8, 2.2], dtype=torch.float64)
    z = torch.randn((16, 3), dtype=torch.float64, generator=generator)
    momentum = torch.randn((16, 3), dtype=torch.float64, generator=generator)
    log_prob = gaussian_log_prob(precision)

    z_forward, p_forward, _ = leapfrog_integrate(
        log_prob,
        z,
        momentum,
        step_size=0.13,
        n_leapfrog=9,
    )
    z_back, p_back, _ = leapfrog_integrate(
        log_prob,
        z_forward,
        p_forward,
        step_size=-0.13,
        n_leapfrog=9,
    )

    assert torch.max(torch.abs(z_back - z)).item() < 1.0e-10
    assert torch.max(torch.abs(p_back - momentum)).item() < 1.0e-10


def test_leapfrog_hamiltonian_error_decreases_quadratically() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(202)
    precision = torch.tensor([1.3, 0.7], dtype=torch.float64)
    z = torch.randn((64, 2), dtype=torch.float64, generator=generator)
    momentum = torch.randn((64, 2), dtype=torch.float64, generator=generator)
    log_prob = gaussian_log_prob(precision)
    initial_h = hamiltonian(log_prob, z, momentum)

    z_coarse, p_coarse, _ = leapfrog_integrate(
        log_prob,
        z,
        momentum,
        step_size=0.25,
        n_leapfrog=8,
    )
    z_fine, p_fine, _ = leapfrog_integrate(
        log_prob,
        z,
        momentum,
        step_size=0.125,
        n_leapfrog=16,
    )
    coarse_error = torch.mean(torch.abs(hamiltonian(log_prob, z_coarse, p_coarse) - initial_h))
    fine_error = torch.mean(torch.abs(hamiltonian(log_prob, z_fine, p_fine) - initial_h))

    assert fine_error.item() < 0.35 * coarse_error.item()


def test_hmc_samples_anisotropic_gaussian_moments() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(303)
    precision = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
    initial = torch.zeros((128, 3), dtype=torch.float64)

    result = hmc_sample(
        gaussian_log_prob(precision),
        initial,
        step_size=0.20,
        n_leapfrog=7,
        n_steps=240,
        burn_in=60,
        thin=2,
        generator=generator,
    )
    flat = result.samples.reshape(-1, 3)
    mean = torch.mean(flat, dim=0)
    variance = torch.var(flat, dim=0, unbiased=True)

    assert result.acceptance_rate > 0.65
    assert result.divergence_fraction == 0.0
    assert torch.max(torch.abs(mean)).item() < 0.08
    assert torch.max(torch.abs(variance - 1.0 / precision)).item() < 0.10


def test_randomized_hmc_and_warmup_return_valid_diagnostics() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(313)
    precision = torch.tensor([1.0, 0.6], dtype=torch.float64)
    initial = torch.zeros((64, 2), dtype=torch.float64)
    log_prob = gaussian_log_prob(precision)

    warmup = hmc_warmup(
        log_prob,
        initial,
        initial_step_size=0.18,
        leapfrog_min=4,
        leapfrog_max=8,
        n_warmup=40,
        target_acceptance=0.8,
        generator=generator,
    )
    result = hmc_sample_randomized(
        log_prob,
        warmup.final_z,
        step_size=warmup.step_size,
        leapfrog_min=4,
        leapfrog_max=8,
        n_steps=80,
        burn_in=10,
        thin=2,
        mass=warmup.mass,
        generator=generator,
    )

    assert warmup.mass.shape == (2,)
    assert torch.all(warmup.mass > 0.0)
    assert 0.2 < warmup.acceptance_rate <= 1.0
    assert 0.2 < result.acceptance_rate <= 1.0
    assert result.samples.shape[1:] == initial.shape


def test_adjoint_hmc_log_density_rejects_outside_ordered_chamber() -> None:
    torch.set_default_dtype(torch.float64)
    model = HarmonicAdjointHead(n=4)
    basis = traceless_hyperplane_basis(4, dtype=torch.float64)
    valid_lam = torch.tensor([[1.5, 0.2, -0.4, -1.3]], dtype=torch.float64)
    invalid_lam = torch.tensor([[0.0, 1.0, -0.3, -0.7]], dtype=torch.float64)
    valid_z = valid_lam @ basis
    invalid_z = invalid_lam @ basis

    valid_logp = adjoint_hmc_log_norm_density(
        model,
        valid_z,
        beta=1.0,
        basis=basis,
        ordered=True,
    )
    invalid_logp = adjoint_hmc_log_norm_density(
        model,
        invalid_z,
        beta=1.0,
        basis=basis,
        ordered=True,
    )

    assert torch.isfinite(valid_logp).item()
    assert torch.isneginf(invalid_logp).item()


def test_su2_harmonic_adjoint_hmc_norm_density_moment() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(404)
    model = HarmonicAdjointHead(n=2, omega=1.0)
    basis = traceless_hyperplane_basis(2, dtype=torch.float64)
    initial, _ = ordered_traceless_gaussian_initial(
        2,
        128,
        sigma=1.6,
        dtype=torch.float64,
        generator=generator,
    )

    def log_prob(z: torch.Tensor) -> torch.Tensor:
        return adjoint_hmc_log_norm_density(
            model,
            z,
            beta=1.0,
            basis=basis,
            ordered=True,
        )

    result = hmc_sample(
        log_prob,
        initial,
        step_size=0.10,
        n_leapfrog=12,
        n_steps=360,
        burn_in=120,
        thin=2,
        generator=generator,
    )
    z_flat = result.samples.reshape(-1, 1)
    lam = z_flat @ basis.T
    tr_x2 = torch.sum(lam * lam, dim=-1)

    assert result.acceptance_rate > 0.55
    assert result.divergence_fraction < 1.0e-3
    assert torch.mean(tr_x2).item() == pytest.approx(2.5, abs=0.15)


def test_mass_adapted_hmc_samples_su4_harmonic_adjoint_target() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(515)
    model = HarmonicAdjointHead(n=4, omega=1.0)
    basis = traceless_hyperplane_basis(4, dtype=torch.float64)
    initial, _ = ordered_traceless_gaussian_initial(
        4,
        128,
        sigma=1.5,
        dtype=torch.float64,
        generator=generator,
    )

    def log_prob(z: torch.Tensor) -> torch.Tensor:
        return adjoint_hmc_log_norm_density(
            model,
            z,
            beta=1.0,
            basis=basis,
            ordered=True,
        )

    warmup = hmc_warmup(
        log_prob,
        initial,
        initial_step_size=0.01,
        leapfrog_min=4,
        leapfrog_max=8,
        n_warmup=100,
        target_acceptance=0.9,
        generator=generator,
    )
    result = hmc_sample_randomized(
        log_prob,
        warmup.final_z,
        step_size=0.2 * warmup.step_size,
        leapfrog_min=4,
        leapfrog_max=8,
        n_steps=100,
        burn_in=10,
        thin=2,
        mass=warmup.mass,
        generator=generator,
    )

    assert torch.all(warmup.mass > 0.0)
    assert 0.6 < warmup.acceptance_rate <= 1.0
    assert result.acceptance_rate > 0.85
    assert result.divergence_fraction < 0.02
    assert result.mean_abs_hamiltonian_error < 1.0


def test_lagged_reweighting_is_unity_for_matching_anchor_beta_one() -> None:
    torch.set_default_dtype(torch.float64)
    model = HarmonicAdjointHead(n=3, omega=1.0)
    lam = torch.tensor(
        [
            [1.0, 0.2, -1.2],
            [0.7, -0.1, -0.6],
            [1.5, -0.4, -1.1],
        ],
        dtype=torch.float64,
    )
    log_weights = adjoint_lagged_log_weights(
        model,
        model,
        lam,
        beta=1.0,
    )
    estimate = adjoint_lagged_reweighted_energy(
        model,
        model,
        lam,
        beta=1.0,
        coupling=0.05,
    )

    assert torch.max(torch.abs(log_weights)).item() < 1.0e-12
    assert estimate.relative_ess.item() == pytest.approx(1.0, abs=1.0e-12)
    assert torch.isfinite(estimate.energy).item()
    assert torch.isfinite(estimate.virial_residual).item()
