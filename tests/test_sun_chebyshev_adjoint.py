from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from adjoint_qm import (  # noqa: E402
    SUNAdjointChebyshevSpectralAnsatz,
    SUNAdjointLinearImpurityAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_linear_impurity_basis,
    adjoint_shape_features,
    adjoint_shape_quadratic_features,
    adjoint_importance_energy,
    adjoint_metropolis_observables,
    adjoint_importance_moments,
    adjoint_importance_observables,
    adjoint_quadrature_linear_impurity_eigenproblem,
    adjoint_quadrature_moments,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    adjoint_vmc_dirichlet_loss,
    centered_chebyshev_heads,
    exact_suN_harmonic_adjoint_energy,
    initialize_full_chebyshev_head_from_linear_impurity,
    metropolis_sample,
    quartic_ray_wkb_tail,
    sobol_gaussian_traceless_samples,
    su3_adjoint_polar_eigenvalue_grid,
    train_adjoint_importance,
    train_adjoint_vmc_metropolis,
)
from adjoint_qm.ansatz import _inverse_softplus  # noqa: E402
from run_sun_adjoint_vmc_benchmarks import (  # noqa: E402
    linear_impurity_ladder_specs,
    profile_slice_payload,
)


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


@pytest.mark.parametrize("n", [2, 3])
def test_ray_wkb_tail_reduces_to_radial_tail_for_su2_su3(n: int) -> None:
    torch.set_default_dtype(torch.float64)
    lam = torch.tensor(
        [
            [1.1, -0.4, -0.7],
            [0.8, 0.2, -1.0],
            [-1.3, 0.5, 0.8],
        ],
        dtype=torch.float64,
    )[:, :n]
    lam = lam - torch.mean(lam, dim=-1, keepdim=True)
    p2 = torch.sum(lam**2, dim=-1)
    tail = quartic_ray_wkb_tail(lam, eps=0.0)
    expected = p2**1.5 / math.sqrt(2.0)

    assert torch.max(torch.abs(tail - expected)).item() < 1.0e-12


def test_shape_features_are_even_and_weyl_invariant() -> None:
    torch.set_default_dtype(torch.float64)
    lam = torch.tensor(
        [
            [1.2, -0.3, -0.4, -0.5],
            [0.9, 0.2, -0.7, -0.4],
        ],
        dtype=torch.float64,
    )
    perm = [2, 0, 3, 1]
    features = adjoint_shape_features(lam, feature_scale=1.7)

    assert features.shape == (2, 3)
    assert (
        torch.max(torch.abs(features - adjoint_shape_features(-lam, feature_scale=1.7))).item()
        < 1.0e-12
    )
    assert (
        torch.max(
            torch.abs(features - adjoint_shape_features(lam[:, perm], feature_scale=1.7))
        ).item()
        < 1.0e-12
    )


def test_mlp_quadratic_mode_moves_p2_into_action_network() -> None:
    torch.set_default_dtype(torch.float64)
    lam = torch.tensor(
        [
            [1.2, -0.3, -0.4, -0.5],
            [0.9, 0.2, -0.7, -0.4],
        ],
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=2.0,
        quartic_tail_init=0.0,
        feature_mode="shape_quadratic",
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        action_quadratic_mode="mlp",
        action_correction_scale=1.0,
        dtype=torch.float64,
    )

    centered = lam - torch.mean(lam, dim=-1, keepdim=True)
    p2 = torch.sum(centered**2, dim=-1)
    action_features = model.action_features(lam)
    head_features = model.invariant_features(lam)

    assert action_features.shape == (2, 7)
    assert head_features.shape == (2, 6)
    assert model.net[0].in_features == 7
    assert model.head_net is not None
    assert model.head_net[0].in_features == 6
    assert model.raw_alpha.requires_grad is False
    assert torch.max(torch.abs(action_features[:, -1] - p2)).item() < 1.0e-12
    assert torch.max(torch.abs(model.action(lam))).item() < 1.0e-12

    with torch.no_grad():
        final = model.net[-1]
        assert isinstance(final, torch.nn.Linear)
        final.weight.zero_()
        final.bias.zero_()
        final.weight[0, -1] = 0.75

    assert torch.max(torch.abs(model.action(lam) - 0.75 * p2)).item() < 1.0e-12


def test_explicit_quadratic_mode_keeps_p2_out_of_action_network() -> None:
    torch.set_default_dtype(torch.float64)
    lam = torch.tensor(
        [
            [1.2, -0.3, -0.4, -0.5],
            [0.9, 0.2, -0.7, -0.4],
        ],
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=2.0,
        quartic_tail_init=0.0,
        feature_mode="shape_quadratic",
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        action_quadratic_mode="explicit",
        action_correction_scale=1.0,
        dtype=torch.float64,
    )

    centered = lam - torch.mean(lam, dim=-1, keepdim=True)
    p2 = torch.sum(centered**2, dim=-1)

    assert model.action_features(lam).shape == (2, 6)
    assert model.net[0].in_features == 6
    assert model.raw_alpha.requires_grad is True
    assert torch.max(torch.abs(model.action(lam) - 2.0 * p2)).item() < 1.0e-10


def test_importance_training_records_loss_every_step_but_sparse_diagnostics() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=3,
        omega_init=1.0,
        quartic_tail_init=0.1,
        feature_mode="shape_quadratic",
        hidden_layers=(4,),
        head_hidden_layers=(4,),
        chebyshev_degrees=(1, 3),
        action_correction_scale=0.1,
        head_correction_scale=0.1,
        dtype=torch.float64,
    )
    samples = sobol_gaussian_traceless_samples(
        3,
        64,
        sigma=1.0,
        seed=12345,
        dtype=torch.float64,
    )
    error_batches = []
    for replicate in range(2):
        error_samples = sobol_gaussian_traceless_samples(
            3,
            32,
            sigma=1.0,
            seed=22345 + replicate,
            dtype=torch.float64,
        )
        error_batches.append((error_samples.lam, error_samples.log_prob))

    diagnostics, loss_history = train_adjoint_importance(
        model,
        samples.lam,
        samples.log_prob,
        coupling=0.05,
        n_steps=5,
        lr=1.0e-3,
        report_every=2,
        print_every=None,
        error_batches=error_batches,
    )

    assert [record.step for record in loss_history] == [1, 2, 3, 4, 5]
    assert all(math.isfinite(record.loss) for record in loss_history)
    assert all(record.error_replicates == 2 for record in loss_history)
    assert all(
        record.error_sample_count_per_replicate == 32
        for record in loss_history
    )
    assert all(
        record.error_energy_mean is not None
        and math.isfinite(record.error_energy_mean)
        for record in loss_history
    )
    assert all(
        record.error_energy_standard_error is not None
        and math.isfinite(record.error_energy_standard_error)
        for record in loss_history
    )
    assert [record.step for record in diagnostics] == [1, 2, 4, 5]
    assert all(math.isfinite(record.energy) for record in diagnostics)
    assert all(math.isfinite(record.tr_x2) for record in diagnostics)
    assert all(math.isfinite(record.tr_x4) for record in diagnostics)
    assert all(math.isfinite(record.kinetic) for record in diagnostics)
    assert all(math.isfinite(record.virial_rhs) for record in diagnostics)
    assert all(math.isfinite(record.virial_residual) for record in diagnostics)
    assert all(
        record.virial_residual
        == pytest.approx(2.0 * record.kinetic - record.virial_rhs, abs=1.0e-12)
        for record in diagnostics
    )


def test_importance_training_can_refresh_proposal_clouds() -> None:
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=3,
        omega_init=1.0,
        quartic_tail_init=0.1,
        feature_mode="shape_quadratic",
        hidden_layers=(4,),
        head_hidden_layers=(4,),
        chebyshev_degrees=(1, 3),
        action_correction_scale=0.1,
        head_correction_scale=0.1,
        dtype=torch.float64,
    )
    initial_samples = sobol_gaussian_traceless_samples(
        3,
        32,
        sigma=1.0,
        seed=32345,
        dtype=torch.float64,
    )
    training_calls: list[int] = []
    error_calls: list[int] = []

    def training_batch_factory(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        training_calls.append(step)
        samples = sobol_gaussian_traceless_samples(
            3,
            32,
            sigma=1.0,
            seed=33345 + step,
            dtype=torch.float64,
        )
        return samples.lam, samples.log_prob

    def error_batch_factory(
        step: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        error_calls.append(step)
        batches = []
        for replicate in range(2):
            samples = sobol_gaussian_traceless_samples(
                3,
                16,
                sigma=1.0,
                seed=34345 + 10 * step + replicate,
                dtype=torch.float64,
            )
            batches.append((samples.lam, samples.log_prob))
        return batches

    diagnostics, loss_history = train_adjoint_importance(
        model,
        initial_samples.lam,
        initial_samples.log_prob,
        coupling=0.05,
        n_steps=4,
        lr=1.0e-3,
        report_every=2,
        print_every=None,
        training_batch_factory=training_batch_factory,
        error_batch_factory=error_batch_factory,
        error_every=1,
    )

    assert training_calls == [1, 2, 3, 4]
    assert error_calls == [1, 2, 3, 4]
    assert [record.step for record in loss_history] == [1, 2, 3, 4]
    assert [record.step for record in diagnostics] == [1, 2, 4]
    assert all(math.isfinite(record.loss) for record in loss_history)
    assert all(record.error_replicates == 2 for record in loss_history)
    assert all(
        record.error_sample_count_per_replicate == 16
        for record in loss_history
    )
    assert all(
        record.error_energy_standard_error is not None
        and math.isfinite(record.error_energy_standard_error)
        for record in loss_history
    )


def test_linear_impurity_ladder_specs_are_nested() -> None:
    for parity in ("odd", "even"):
        specs = linear_impurity_ladder_specs(parity)
        names = [spec["name"] for spec in specs]
        expected_degree_parity = 1 if parity == "odd" else 0

        assert len(names) == len(set(names))
        previous_basis: set[tuple[str, int]] = set()
        for spec in specs:
            basis = {
                (term, degree)
                for term in spec["terms"]
                for degree in spec["chebyshev_degrees"]
            }
            assert all(degree % 2 == expected_degree_parity for _, degree in basis)
            assert previous_basis <= basis
            assert len(basis) > 0
            previous_basis = basis


def test_even_chebyshev_ansatz_structure_diagnostics_use_even_parity() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(2, 4),
        parity="even",
        scale_init=2.5,
        dtype=torch.float64,
    )
    lam = torch.tensor(
        [
            [1.1, -0.2, -0.4, -0.5],
            [0.7, 0.3, -0.8, -0.2],
        ],
        dtype=torch.float64,
    )

    diagnostics = adjoint_structure_diagnostics(model, lam)

    assert diagnostics.traceless_residual < 1.0e-12
    assert diagnostics.parity_residual < 1.0e-12
    assert diagnostics.weyl_residual < 1.0e-12


def test_shape_quadratic_features_match_ladder_terms() -> None:
    torch.set_default_dtype(torch.float64)
    lam = torch.tensor(
        [
            [0.9, -0.2, -0.4, -0.3],
            [-0.8, 0.6, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )

    shape = adjoint_shape_features(lam, feature_scale=1.3)
    quadratic = adjoint_shape_quadratic_features(lam, feature_scale=1.3)

    rho = shape[:, 0]
    u = shape[:, 1]
    v = shape[:, 2]
    assert torch.max(torch.abs(quadratic[:, 0] - rho)).item() < 1.0e-12
    assert torch.max(torch.abs(quadratic[:, 1] - u)).item() < 1.0e-12
    assert torch.max(torch.abs(quadratic[:, 2] - v)).item() < 1.0e-12
    assert torch.max(torch.abs(quadratic[:, 3] - rho * rho)).item() < 1.0e-12
    assert torch.max(torch.abs(quadratic[:, 4] - rho * u)).item() < 1.0e-12
    assert torch.max(torch.abs(quadratic[:, 5] - rho * v)).item() < 1.0e-12


def test_full_chebyshev_head_mode_can_change_leading_coefficient() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1, 3),
        head_coefficient_mode="full",
        head_correction_scale=0.5,
        scale_init=2.5,
        dtype=torch.float64,
    )
    assert model.head_net is not None
    final = model.head_net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([0.4, -0.2], dtype=torch.float64))
    lam = torch.tensor(
        [
            [0.9, -0.2, -0.4, -0.3],
            [-0.8, 0.6, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )

    coeffs = model.head_coefficients(lam)
    basis = model.head_basis(lam)
    expected = 1.2 * basis[:, 0, :] - 0.1 * basis[:, 1, :]

    assert torch.max(torch.abs(coeffs[:, 0] - 1.2)).item() < 1.0e-12
    assert torch.max(torch.abs(coeffs[:, 1] + 0.1)).item() < 1.0e-12
    assert torch.max(torch.abs(model.head(lam) - expected)).item() < 1.0e-12


def test_anchored_chebyshev_head_mode_keeps_leading_coefficient_fixed() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1, 3),
        head_coefficient_mode="anchored",
        head_correction_scale=0.5,
        scale_init=2.5,
        dtype=torch.float64,
    )
    assert model.head_net is not None
    final = model.head_net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(0.4)
    lam = torch.tensor([[0.9, -0.2, -0.4, -0.3]], dtype=torch.float64)

    coeffs = model.head_coefficients(lam)

    assert coeffs[0, 0].item() == pytest.approx(1.0, abs=1.0e-12)
    assert coeffs[0, 1].item() == pytest.approx(0.2, abs=1.0e-12)


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
        feature_mode="shape",
        feature_scale_init=1.5,
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


def test_su4_profile_payload_records_shifted_scalar_action() -> None:
    torch.set_default_dtype(torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        quartic_tail_init=0.2,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        dtype=torch.float64,
    )

    payload = profile_slice_payload(model, argparse.Namespace())
    assert payload is not None

    pair_split = payload["slices"]["pair split"]
    radii = torch.tensor(pair_split["r"], dtype=torch.float64)
    direction = torch.tensor(pair_split["direction"], dtype=torch.float64)
    lam = radii[:, None] * direction[None, :]
    action = model.action(lam).detach()
    center_index = int(torch.argmin(torch.abs(radii)).detach())
    shifted = action - action[center_index]

    assert pair_split["action"] == pytest.approx(action.tolist())
    assert pair_split["action_shifted"] == pytest.approx(shifted.tolist())
    assert pair_split["action_shifted"][center_index] == pytest.approx(0.0, abs=1e-14)


def test_linear_impurity_eigenproblem_reproduces_harmonic_adjoint_t1() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = adjoint_eigenvalue_grid(
        4,
        6.0,
        26,
        dtype=torch.float64,
    )
    envelope = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        dtype=torch.float64,
    )

    result = adjoint_quadrature_linear_impurity_eigenproblem(
        envelope,
        lam,
        weights,
        coupling=0.0,
        chebyshev_degrees=(1,),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=1.0,
        terms=("1",),
        tail_eps=envelope.tail_eps,
    )
    linear_model = SUNAdjointLinearImpurityAnsatz(
        envelope_model=envelope,
        coefficients=result.coefficients,
        chebyshev_degrees=(1,),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=1.0,
        terms=("1",),
        tail_eps=envelope.tail_eps,
    )
    obs = adjoint_quadrature_observables(linear_model, lam, weights, coupling=0.0)

    assert result.retained_basis_count == 1
    assert result.energy == pytest.approx(
        exact_suN_harmonic_adjoint_energy(4, 1.0),
        abs=5.0e-5,
    )
    assert obs.energy == pytest.approx(result.energy, abs=1.0e-10)


def test_linear_impurity_basis_preserves_structure_and_lowers_fixed_envelope() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = adjoint_eigenvalue_grid(
        4,
        4.5,
        10,
        dtype=torch.float64,
    )
    envelope = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        quartic_tail_init=2.0 * math.sqrt(2.0) / 3.0,
        feature_mode="shape",
        hidden_layers=(),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        dtype=torch.float64,
    )
    baseline = adjoint_quadrature_observables(envelope, lam, weights, coupling=1.0)
    result = adjoint_quadrature_linear_impurity_eigenproblem(
        envelope,
        lam,
        weights,
        coupling=1.0,
        chebyshev_degrees=(1, 3),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=1.0,
        terms=("1", "rho", "u", "v"),
        tail_eps=envelope.tail_eps,
    )
    linear_model = SUNAdjointLinearImpurityAnsatz(
        envelope_model=envelope,
        coefficients=result.coefficients,
        chebyshev_degrees=(1, 3),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=1.0,
        terms=("1", "rho", "u", "v"),
        tail_eps=envelope.tail_eps,
    )
    obs = adjoint_quadrature_observables(linear_model, lam, weights, coupling=1.0)
    diagnostics = adjoint_structure_diagnostics(linear_model, lam[:4])

    assert result.retained_basis_count >= 2
    assert result.energy <= baseline.energy + 1.0e-10
    assert obs.energy == pytest.approx(result.energy, abs=1.0e-9)
    assert diagnostics.traceless_residual < 1.0e-12
    assert diagnostics.parity_residual < 1.0e-12
    assert diagnostics.weyl_residual < 1.0e-12


def test_full_neural_head_initializes_exactly_from_linear_impurity() -> None:
    torch.set_default_dtype(torch.float64)
    _, lam, weights = adjoint_eigenvalue_grid(
        4,
        4.5,
        10,
        dtype=torch.float64,
    )
    neural = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        quartic_tail_init=2.0 * math.sqrt(2.0) / 3.0,
        feature_mode="shape_quadratic",
        hidden_layers=(),
        head_hidden_layers=(),
        head_coefficient_mode="full",
        chebyshev_degrees=(1, 3),
        scale_init=3.0,
        dtype=torch.float64,
    )
    terms = ("1", "rho", "u", "v", "rho2", "rho_u", "rho_v")
    result = adjoint_quadrature_linear_impurity_eigenproblem(
        neural,
        lam,
        weights,
        coupling=1.0,
        chebyshev_degrees=(1, 3),
        chebyshev_scale=neural.scale.detach(),
        feature_scale=neural.feature_scale.detach(),
        terms=terms,
        tail_eps=neural.tail_eps,
    )
    linear = SUNAdjointLinearImpurityAnsatz(
        envelope_model=neural,
        coefficients=result.coefficients,
        chebyshev_degrees=(1, 3),
        chebyshev_scale=neural.scale.detach(),
        feature_scale=neural.feature_scale.detach(),
        terms=terms,
        tail_eps=neural.tail_eps,
    )

    divisor = initialize_full_chebyshev_head_from_linear_impurity(
        neural,
        result.coefficients,
        terms=terms,
        chebyshev_degrees=(1, 3),
    )
    probe = lam[:12]
    linear_head = linear.head(probe)
    neural_head = neural.head(probe)
    linear_obs = adjoint_quadrature_observables(linear, lam, weights, coupling=1.0)
    neural_obs = adjoint_quadrature_observables(neural, lam, weights, coupling=1.0)

    assert abs(divisor) > 1.0e-12
    assert torch.max(torch.abs(divisor * neural_head - linear_head)).item() < 1.0e-10
    assert neural_obs.energy == pytest.approx(linear_obs.energy, abs=1.0e-10)


def test_linear_impurity_basis_uses_envelope_spectral_coordinates() -> None:
    torch.set_default_dtype(torch.float64)
    envelope = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        quartic_tail_init=2.0 * math.sqrt(2.0) / 3.0,
        feature_mode="shape_quadratic",
        hidden_layers=(),
        head_hidden_layers=(),
        head_coefficient_mode="full",
        chebyshev_degrees=(1, 3),
        scale_init=3.0,
        coordinate_scale_init=1.35,
        dtype=torch.float64,
    )
    terms = ("1", "rho", "u", "v", "rho2", "rho_u", "rho_v")
    coefficients = torch.linspace(
        -0.4,
        0.6,
        len(terms) * 2,
        dtype=torch.float64,
    )
    linear = SUNAdjointLinearImpurityAnsatz(
        envelope_model=envelope,
        coefficients=coefficients,
        chebyshev_degrees=(1, 3),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=envelope.feature_scale.detach(),
        terms=terms,
        tail_eps=envelope.tail_eps,
    )
    lam = torch.tensor(
        [
            [0.9, -0.2, -0.4, -0.3],
            [-0.8, 0.6, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )

    internal_basis = adjoint_linear_impurity_basis(
        envelope.spectral_coordinates(lam),
        chebyshev_degrees=(1, 3),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=envelope.feature_scale.detach(),
        terms=terms,
        eps=envelope.tail_eps,
    )
    raw_basis = adjoint_linear_impurity_basis(
        lam,
        chebyshev_degrees=(1, 3),
        chebyshev_scale=envelope.scale.detach(),
        feature_scale=envelope.feature_scale.detach(),
        terms=terms,
        eps=envelope.tail_eps,
    )
    expected_internal = torch.einsum("sai,a->si", internal_basis, coefficients)
    raw_head = torch.einsum("sai,a->si", raw_basis, coefficients)

    assert torch.max(torch.abs(linear.head(lam) - expected_internal)).item() < 1.0e-12
    assert torch.max(torch.abs(expected_internal - raw_head)).item() > 1.0e-5


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


def test_su4_importance_coordinate_scale_autograd_matches_quartic_virial_residual() -> None:
    torch.set_default_dtype(torch.float64)
    samples = sobol_gaussian_traceless_samples(
        4,
        131072,
        sigma=1.0,
        seed=993,
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=1.0,
        quartic_tail_init=2.0 * math.sqrt(2.0) / 3.0,
        moment_cutoff=6,
        feature_mode="shape_quadratic",
        feature_scale_init=3.0,
        hidden_layers=(8,),
        head_hidden_layers=(8,),
        chebyshev_degrees=(1, 3),
        scale_init=3.0,
        coordinate_scale_init=1.0,
        learn_coordinate_scale=True,
        action_correction_scale=0.05,
        head_correction_scale=0.05,
        dtype=torch.float64,
    )

    obs = adjoint_importance_observables(
        model,
        samples.lam,
        samples.log_prob,
        coupling=1.0,
    )
    energy_tensor, *_ = adjoint_importance_energy(
        model,
        samples.lam,
        samples.log_prob,
        coupling=1.0,
    )
    (grad_raw_scale,) = torch.autograd.grad(
        energy_tensor,
        model.raw_coordinate_scale,
    )
    grad_log_scale = (
        grad_raw_scale
        * model.coordinate_scale.detach()
        / torch.sigmoid(model.raw_coordinate_scale.detach())
    )
    moments = adjoint_importance_moments(
        model,
        samples.lam,
        samples.log_prob,
        coupling=1.0,
    )

    assert math.isfinite(obs.energy)
    assert grad_log_scale.item() == pytest.approx(
        moments.virial_residual,
        abs=5.0e-3,
        rel=5.0e-3,
    )


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
