from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    GaussianEnvelopeMLP,
    HarmonicOscillatorPotential,
    dirichlet_grid_1d,
    exact_harmonic_benchmarks,
    harmonic_correlator,
    krylov_correlator,
    krylov_spectrum,
    lanczos_tridiagonal,
    metropolis_sample,
    parity_residual,
    position_correlator_time_evolution_1d,
    quadrature_observables,
    train_quadrature,
    trapezoid_grid,
    vmc_observables,
)


def exact_model(omega: float = 1.0) -> GaussianEnvelopeMLP:
    return GaussianEnvelopeMLP(
        dim=1,
        hidden_layers=(),
        init_alpha=omega,
        dtype=torch.float64,
        zero_final=True,
    )


def test_exact_gaussian_matches_harmonic_benchmarks() -> None:
    torch.set_default_dtype(torch.float64)
    omega = 1.0
    model = exact_model(omega)
    potential = HarmonicOscillatorPotential(omega)
    grid, weights = trapezoid_grid(8.0, 4001, dtype=torch.float64)

    obs = quadrature_observables(model, potential, grid, weights)
    exact = exact_harmonic_benchmarks(omega)

    assert obs.energy == pytest.approx(exact["energy"], abs=2.0e-6)
    assert obs.kinetic == pytest.approx(exact["kinetic"], abs=2.0e-6)
    assert obs.potential == pytest.approx(exact["potential"], abs=2.0e-6)
    assert obs.x2 == pytest.approx(exact["x2"], abs=2.0e-6)
    assert obs.x4 == pytest.approx(exact["x4"], abs=2.0e-5)
    assert obs.local_energy_variance < 1.0e-10


def test_even_ansatz_is_parity_symmetric() -> None:
    torch.set_default_dtype(torch.float64)
    model = GaussianEnvelopeMLP(
        dim=1,
        hidden_layers=(8,),
        init_alpha=0.8,
        dtype=torch.float64,
    )
    x = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)[:, None]
    assert float(parity_residual(model, x)) < 1.0e-12


def test_quadrature_training_converges_near_ground_energy() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(123)
    model = GaussianEnvelopeMLP(
        dim=1,
        hidden_layers=(8,),
        init_alpha=0.6,
        dtype=torch.float64,
    )
    potential = HarmonicOscillatorPotential(1.0)
    grid, weights = trapezoid_grid(7.0, 1001, dtype=torch.float64)

    train_quadrature(
        model,
        potential,
        grid,
        weights,
        n_steps=250,
        lr=2.0e-2,
        report_every=250,
    )
    obs = quadrature_observables(model, potential, grid, weights)

    assert obs.energy == pytest.approx(0.5, abs=3.0e-3)
    assert abs(obs.virial_residual) < 2.0e-2


def test_vmc_diagnostic_samples_exact_gaussian() -> None:
    torch.set_default_dtype(torch.float64)
    model = exact_model(1.0)
    potential = HarmonicOscillatorPotential(1.0)
    result = metropolis_sample(
        model,
        n_samples=4096,
        dim=1,
        n_chains=64,
        step_size=1.0,
        burn_in=300,
        thinning=5,
        seed=42,
        dtype=torch.float64,
    )
    obs = vmc_observables(model, potential, result.samples)

    assert 0.2 < result.acceptance_rate < 0.95
    assert obs.x2 == pytest.approx(0.5, abs=7.0e-2)
    assert obs.local_energy_mean == pytest.approx(0.5, abs=1.0e-10)


def test_krylov_correlator_matches_small_diagonal_hamiltonian() -> None:
    torch.set_default_dtype(torch.float64)
    diagonal = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    source = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    tau = torch.linspace(0.0, 2.0, 9, dtype=torch.float64)

    result = lanczos_tridiagonal(
        lambda vector: diagonal * vector,
        source,
        krylov_dim=3,
    )
    correlator = krylov_correlator(result, tau)
    spectrum = krylov_spectrum(result)
    exact = torch.exp(-tau) + torch.exp(-2.0 * tau)

    assert torch.max(torch.abs(correlator - exact)).item() < 1.0e-12
    assert torch.max(torch.abs(spectrum.energies - torch.tensor([1.0, 2.0]))).item() < 1.0e-12
    assert torch.max(torch.abs(spectrum.spectral_weights - torch.tensor([1.0, 1.0]))).item() < 1.0e-12


def test_time_evolution_reproduces_harmonic_position_correlator() -> None:
    torch.set_default_dtype(torch.float64)
    omega = 1.0
    model = exact_model(omega)
    potential = HarmonicOscillatorPotential(omega)
    grid, weights = dirichlet_grid_1d(8.0, 1600, dtype=torch.float64)
    tau = torch.linspace(0.0, 4.0, 9, dtype=torch.float64)

    correlator, result = position_correlator_time_evolution_1d(
        model,
        potential,
        grid,
        weights,
        tau,
        energy_shift=0.5 * omega,
        krylov_dim=8,
        tol=1.0e-4,
    )
    exact = harmonic_correlator(tau, omega)
    spectrum = krylov_spectrum(result, energy_shift=0.5 * omega)

    assert result.iterations == 1
    assert torch.max(torch.abs(correlator - exact)).item() < 1.0e-5
    assert spectrum.energies[0].item() == pytest.approx(1.5, abs=3.0e-5)
    assert spectrum.gaps[0].item() == pytest.approx(1.0, abs=3.0e-5)
    assert spectrum.spectral_weights[0].item() == pytest.approx(0.5, abs=1.0e-5)


def test_notebook_imports_library_without_redefining_core_logic() -> None:
    notebook_path = Path("notebooks/harmonic_oscillator_ground_state.ipynb")
    notebook = json.loads(notebook_path.read_text())
    sources = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        sources.append("".join(source) if isinstance(source, list) else source)
    code = "\n".join(
        sources
    )

    assert "from adjoint_qm import" in code
    assert "position_correlator_time_evolution_1d" in code
    assert "krylov_spectrum" in code
    assert "class GaussianEnvelopeMLP" not in code
    assert "def quadrature_observables" not in code
