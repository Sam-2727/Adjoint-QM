from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from adjoint_qm import (  # noqa: E402
    GaussianEnvelopeMLP,
    HarmonicOscillatorPotential,
    RadialFeatureMap,
    QuarticOscillatorPotential,
    SeparableGaussianEnvelopeMLP,
    align_wavefunction_sign,
    dirichlet_grid_1d,
    diagonalize_quartic_oscillator,
    diagonalize_separable_quartic_oscillator,
    exact_harmonic_benchmarks,
    exact_isotropic_harmonic_benchmarks,
    evaluate_basis_wavefunctions,
    harmonic_correlator,
    krylov_correlator,
    krylov_spectrum,
    lanczos_tridiagonal,
    metropolis_sample,
    parity_residual,
    position_correlator_time_evolution_1d,
    quadrature_observables,
    quartic_oscillator_perturbation_benchmarks,
    quartic_oscillator_hamiltonian_matrix,
    train_quadrature,
    train_vmc_metropolis,
    trapezoid_grid,
    vmc_observables,
    vmc_score_function_loss,
)


def exact_model(omega: float = 1.0, dim: int = 1) -> GaussianEnvelopeMLP:
    return GaussianEnvelopeMLP(
        dim=dim,
        hidden_layers=(),
        init_alpha=omega,
        dtype=torch.float64,
        zero_final=True,
    )


def exact_radial_model(omega: float = 1.0, dim: int = 1) -> GaussianEnvelopeMLP:
    return GaussianEnvelopeMLP(
        dim=dim,
        hidden_layers=(),
        feature_map=RadialFeatureMap(),
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


def test_radial_feature_map_is_rotationally_invariant() -> None:
    torch.set_default_dtype(torch.float64)
    feature_map = RadialFeatureMap()
    x = torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=torch.float64)
    rotated = torch.tensor([[4.0, -3.0], [2.0, -1.0]], dtype=torch.float64)

    assert feature_map.output_dim(dim=2) == 1
    assert torch.allclose(feature_map(x), torch.tensor([[25.0], [5.0]]))
    assert torch.allclose(feature_map(x), feature_map(rotated))


def test_separable_ansatz_sums_shared_one_body_factor() -> None:
    torch.set_default_dtype(torch.float64)
    model = SeparableGaussianEnvelopeMLP(
        dim=3,
        hidden_layers=(8,),
        init_alpha=1.1,
        dtype=torch.float64,
    )
    x = torch.tensor(
        [[-1.0, 0.5, 2.0], [0.25, -0.75, 1.25]],
        dtype=torch.float64,
    )
    expected = model.one_body.log_psi(x.reshape(-1, 1)).reshape(2, 3).sum(dim=1)

    assert torch.allclose(model.log_psi(x), expected)
    assert model.alpha.item() == pytest.approx(model.one_body.alpha.item())


def test_quartic_potential_and_perturbation_benchmarks() -> None:
    torch.set_default_dtype(torch.float64)
    coupling = 0.02
    potential = QuarticOscillatorPotential(omega=1.0, coupling=coupling)
    x = torch.tensor([[-1.0], [0.0], [2.0]], dtype=torch.float64)

    values = potential(x)
    virial = potential.virial(x)
    benchmarks = quartic_oscillator_perturbation_benchmarks(
        omega=1.0,
        coupling=coupling,
    )

    assert torch.allclose(values, torch.tensor([0.52, 0.0, 2.32], dtype=torch.float64))
    assert torch.allclose(virial, torch.tensor([1.08, 0.0, 5.28], dtype=torch.float64))
    assert benchmarks["energy_order0"] == pytest.approx(0.5)
    assert benchmarks["energy_order1"] == pytest.approx(0.515)
    assert benchmarks["energy_order2"] == pytest.approx(0.51395)
    assert benchmarks["x2_order1"] == pytest.approx(0.47)
    assert benchmarks["x4_order1"] == pytest.approx(0.645)


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


def test_quartic_training_matches_weak_coupling_perturbation() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    coupling = 0.02
    model = GaussianEnvelopeMLP(
        dim=1,
        hidden_layers=(16,),
        init_alpha=1.0,
        dtype=torch.float64,
    )
    potential = QuarticOscillatorPotential(omega=1.0, coupling=coupling)
    grid, weights = trapezoid_grid(7.0, 1001, dtype=torch.float64)
    perturbative = quartic_oscillator_perturbation_benchmarks(
        omega=1.0,
        coupling=coupling,
    )

    train_quadrature(
        model,
        potential,
        grid,
        weights,
        n_steps=250,
        lr=1.0e-2,
        report_every=250,
    )
    obs = quadrature_observables(model, potential, grid, weights)

    assert obs.energy == pytest.approx(perturbative["energy_order2"], abs=3.0e-3)
    assert abs(obs.virial_residual) < 2.0e-2


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_exact_multidimensional_harmonic_vmc_diagnostic(dim: int) -> None:
    torch.set_default_dtype(torch.float64)
    omega = 1.0
    model = exact_radial_model(omega=omega, dim=dim)
    potential = HarmonicOscillatorPotential(omega)
    exact = exact_isotropic_harmonic_benchmarks(dim, omega)
    result = metropolis_sample(
        model,
        n_samples=8192,
        dim=dim,
        n_chains=128,
        step_size=1.6 / (dim**0.5),
        burn_in=600,
        thinning=10,
        seed=500 + dim,
        dtype=torch.float64,
    )
    obs = vmc_observables(model, potential, result.samples)

    assert 0.2 < result.acceptance_rate < 0.8
    assert obs.local_energy_mean == pytest.approx(exact["energy"], abs=1.0e-10)
    assert obs.local_energy_std < 1.0e-10
    assert obs.x2 == pytest.approx(exact["r2"], abs=2.5e-1)
    assert obs.r4 == pytest.approx(exact["r4"], abs=1.5)
    assert obs.coordinate_variance_mean == pytest.approx(
        exact["coordinate_variance"],
        abs=8.0e-2,
    )
    assert obs.coordinate_mean_abs_max < 1.5e-1
    assert obs.coordinate_offdiag_abs_max < 1.2e-1
    assert abs(obs.virial_residual) < 1.0e-12


def test_quartic_gaussian_vmc_matches_analytic_variational_energy() -> None:
    torch.set_default_dtype(torch.float64)
    dim = 2
    omega = 1.0
    coupling = 0.05
    alpha = 1.2
    model = exact_radial_model(omega=alpha, dim=dim)
    potential = QuarticOscillatorPotential(omega=omega, coupling=coupling)
    result = metropolis_sample(
        model,
        n_samples=8192,
        dim=dim,
        n_chains=128,
        step_size=1.3 / (dim**0.5),
        burn_in=600,
        thinning=10,
        seed=612,
        dtype=torch.float64,
    )
    obs = vmc_observables(model, potential, result.samples)
    expected = (
        dim * alpha / 4.0
        + dim * omega**2 / (4.0 * alpha)
        + 3.0 * dim * coupling / (4.0 * alpha**2)
    )

    assert 0.2 < result.acceptance_rate < 0.85
    assert obs.local_energy_mean == pytest.approx(expected, abs=5.0e-2)
    assert obs.local_energy_stderr > 0.0


def test_vmc_score_function_loss_vanishes_for_exact_harmonic_state() -> None:
    torch.set_default_dtype(torch.float64)
    model = exact_radial_model(omega=1.0, dim=2)
    potential = HarmonicOscillatorPotential(1.0)
    samples = torch.randn(64, 2, dtype=torch.float64)

    loss, energy, std = vmc_score_function_loss(model, potential, samples)
    loss.backward()

    assert energy.item() == pytest.approx(1.0, abs=1.0e-12)
    assert std.item() < 1.0e-12
    assert abs(model.raw_alpha.grad.item()) < 1.0e-12


def test_train_vmc_metropolis_smoke() -> None:
    torch.set_default_dtype(torch.float64)
    model = exact_radial_model(omega=1.0, dim=2)
    potential = HarmonicOscillatorPotential(1.0)

    history = train_vmc_metropolis(
        model,
        potential,
        dim=2,
        n_steps=2,
        n_samples=256,
        n_chains=32,
        step_size=1.0,
        burn_in=50,
        thinning=4,
        lr=1.0e-3,
        seed=811,
        report_every=1,
        dtype=torch.float64,
    )

    assert len(history) == 2
    assert abs(history[-1].surrogate_loss) < 1.0e-12
    assert history[-1].energy == pytest.approx(1.0, abs=1.0e-12)
    assert 0.2 < history[-1].acceptance_rate < 0.9


def test_basis_diagonalization_matches_harmonic_limit() -> None:
    torch.set_default_dtype(torch.float64)
    hamiltonian = quartic_oscillator_hamiltonian_matrix(
        6,
        omega=1.0,
        coupling=0.0,
        basis_omega=1.0,
        dtype=torch.float64,
    )
    result = diagonalize_quartic_oscillator(
        6,
        omega=1.0,
        coupling=0.0,
        basis_omega=1.0,
        dtype=torch.float64,
    )
    exact = torch.arange(6, dtype=torch.float64) + 0.5

    assert torch.max(torch.abs(hamiltonian - torch.diag(exact))).item() < 1.0e-14
    assert torch.max(torch.abs(result.energies - exact)).item() < 1.0e-14


def test_separable_diagonalization_matches_harmonic_product_spectrum() -> None:
    torch.set_default_dtype(torch.float64)
    result = diagonalize_separable_quartic_oscillator(
        dim=2,
        n_basis=8,
        n_levels=6,
        omega=1.0,
        coupling=0.0,
        basis_omega=1.0,
        dtype=torch.float64,
    )
    expected = torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float64)

    assert torch.max(torch.abs(result.energies - expected)).item() < 1.0e-14


def test_separable_quartic_ground_energy_is_sum_of_1d_energies() -> None:
    torch.set_default_dtype(torch.float64)
    dim = 4
    result = diagonalize_separable_quartic_oscillator(
        dim=dim,
        n_basis=24,
        n_levels=4,
        omega=1.0,
        coupling=0.05,
        basis_omega=1.2,
        dtype=torch.float64,
    )
    one_dimensional = diagonalize_quartic_oscillator(
        24,
        omega=1.0,
        coupling=0.05,
        basis_omega=1.2,
        dtype=torch.float64,
    )

    assert result.energies[0].item() == pytest.approx(
        dim * one_dimensional.energies[0].item(),
        abs=1.0e-12,
    )


def test_basis_diagonalization_matches_published_quartic_energies() -> None:
    torch.set_default_dtype(torch.float64)
    # Okun and Burke, arXiv:2007.04762, Table S1, lambda=-1 column.
    # Their V_lambda=x^4/4-lambda*x^2/2 equals our omega=1, coupling=1/4
    # convention at lambda=-1.
    published = torch.tensor(
        [
            0.6209270298257486608580357329871206982000,
            2.0259661641666569970850703427960975727209,
            3.6984503193780828535724670322135994784906,
            5.5575771385568190043356690869633769327987,
            7.5684228735599952483040236007700297795874,
            9.7091478766133491585283420979384357791199,
        ],
        dtype=torch.float64,
    )
    result = diagonalize_quartic_oscillator(
        40,
        omega=1.0,
        coupling=0.25,
        basis_omega=2.0,
        dtype=torch.float64,
    )

    assert torch.max(torch.abs(result.energies[: published.numel()] - published)).item() < 1.0e-11


def test_basis_wavefunction_evaluation_is_normalized_on_grid() -> None:
    torch.set_default_dtype(torch.float64)
    result = diagonalize_quartic_oscillator(
        40,
        omega=1.0,
        coupling=0.25,
        basis_omega=2.0,
        dtype=torch.float64,
    )
    grid, weights = trapezoid_grid(8.0, 4001, dtype=torch.float64)
    psi0 = evaluate_basis_wavefunctions(
        grid,
        result.eigenvectors,
        basis_omega=result.basis_omega,
        state_indices=[0],
    ).squeeze(-1)
    psi0 = align_wavefunction_sign(psi0)
    norm = torch.sum(weights * psi0**2)
    parity_error = torch.max(torch.abs(psi0 - torch.flip(psi0, dims=(0,))))

    assert float(psi0[psi0.numel() // 2]) > 0
    assert norm.item() == pytest.approx(1.0, abs=1.0e-10)
    assert parity_error.item() < 1.0e-12


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


def test_batched_vmc_observables_match_unbatched() -> None:
    torch.set_default_dtype(torch.float64)
    model = exact_radial_model(1.0, dim=3)
    potential = HarmonicOscillatorPotential(1.0)
    samples = torch.randn(128, 3, dtype=torch.float64)

    unbatched = vmc_observables(model, potential, samples)
    batched = vmc_observables(model, potential, samples, batch_size=17)

    assert batched.local_energy_mean == pytest.approx(unbatched.local_energy_mean)
    assert batched.local_energy_std == pytest.approx(unbatched.local_energy_std)
    assert batched.kinetic == pytest.approx(unbatched.kinetic)
    assert batched.potential == pytest.approx(unbatched.potential)
    assert batched.x2 == pytest.approx(unbatched.x2)
    assert batched.r4 == pytest.approx(unbatched.r4)


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
    assert "QuarticOscillatorPotential" in code
    assert "quartic_oscillator_perturbation_benchmarks" in code
    assert "diagonalize_quartic_oscillator" in code
    assert "evaluate_basis_wavefunctions" in code
    assert "class GaussianEnvelopeMLP" not in code
    assert "def quadrature_observables" not in code
