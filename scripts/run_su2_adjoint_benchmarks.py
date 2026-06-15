#!/usr/bin/env python
"""Run and save SU(2) adjoint-sector benchmark results for the notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from adjoint_qm import (
    SU2AdjointSpectralAnsatz,
    adjoint_profile_norm,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
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
    train_adjoint_quadrature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "su2_adjoint_benchmarks.json",
        help="Path for the JSON result file consumed by the notebook.",
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--quartic-coupling", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=8.0)
    parser.add_argument("--n-grid", type=int, default=3000)
    parser.add_argument("--fd-grid", type=int, default=1600)
    parser.add_argument("--fd-r-max", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--harmonic-steps", type=int, default=800)
    parser.add_argument("--quartic-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--report-every", type=int, default=100)
    return parser.parse_args()


def training_history_payload(
    history: list[Any],
    benchmark: float,
) -> list[dict[str, float]]:
    return [
        {
            "step": record.step,
            "energy": record.energy,
            "benchmark_energy": benchmark,
            "energy_error": record.energy - benchmark,
            "absolute_energy_error": abs(record.energy - benchmark),
            "radial": record.radial,
            "angular": record.angular,
            "potential": record.potential,
            "local_energy_std": record.local_energy_std,
            "alpha": record.alpha,
        }
        for record in history
    ]


def observables_payload(obs: Any, benchmark: float) -> dict[str, float]:
    return {
        "energy": obs.energy,
        "benchmark_energy": benchmark,
        "energy_error": obs.energy - benchmark,
        "absolute_energy_error": abs(obs.energy - benchmark),
        "radial": obs.radial,
        "angular": obs.angular,
        "potential": obs.potential,
        "norm": obs.norm,
        "local_energy_mean": obs.local_energy_mean,
        "local_energy_std": obs.local_energy_std,
        "traceless_residual": obs.traceless_residual,
        "parity_residual": obs.parity_residual,
        "alpha": obs.alpha,
    }


def profile_payload(
    model: SU2AdjointSpectralAnsatz,
    *,
    plot_z_max: float = 4.0,
    plot_n_grid: int = 800,
    norm_z_max: float = 8.0,
    norm_n_grid: int = 3000,
) -> dict[str, list[float]]:
    z, lam, _ = su2_adjoint_eigenvalue_grid(
        plot_z_max,
        plot_n_grid,
        dtype=torch.float64,
    )
    _, norm_lam, norm_weights = su2_adjoint_eigenvalue_grid(
        norm_z_max,
        norm_n_grid,
        dtype=torch.float64,
    )
    with torch.no_grad():
        norm = adjoint_profile_norm(model, norm_lam, norm_weights)
        scale = 1.0 / torch.sqrt(norm)
        profile = model.profile(lam)
        normalized_profile = scale * profile
        action = model.action(lam)
    return {
        "z": z.squeeze(-1).tolist(),
        "lambda_0": lam[:, 0].tolist(),
        "lambda_1": lam[:, 1].tolist(),
        "q_0": profile[:, 0].tolist(),
        "q_1": profile[:, 1].tolist(),
        "q_0_normalized": normalized_profile[:, 0].tolist(),
        "q_1_normalized": normalized_profile[:, 1].tolist(),
        "action": action.tolist(),
        "adjoint_norm": float(norm.detach()),
        "normalization_scale": float(scale.detach()),
        "normalization_z_max": norm_z_max,
        "normalization_n_grid": norm_n_grid,
    }


def symmetry_checks(model: SU2AdjointSpectralAnsatz) -> dict[str, float]:
    z_test = torch.tensor(
        [[-1.4], [-0.3], [0.7], [2.1]],
        dtype=torch.float64,
    )
    lam_test = eigenvalues_from_traceless_coordinates(z_test, n=2)
    diagnostics = adjoint_structure_diagnostics(model, lam_test)

    x_test = torch.tensor(
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
    psi = su2_adjoint_component_wavefunction(model, x_test)
    rotated_psi = su2_adjoint_component_wavefunction(model, x_test @ rotation.T)

    so3 = torch.max(torch.abs(rotated_psi - psi @ rotation.T))
    return {
        "traceless_residual": diagnostics.traceless_residual,
        "odd_parity_residual": diagnostics.parity_residual,
        "weyl_covariance_residual": diagnostics.weyl_residual,
        "head_collision_residual": diagnostics.head_collision_residual,
        "profile_collision_residual": diagnostics.profile_collision_residual,
        "so3_covariance_residual": float(so3.detach()),
    }


def radial_moment_payload(
    u: torch.Tensor,
    r: torch.Tensor,
    dr: float,
    *,
    omega: float,
    coupling: float,
    energy: float,
) -> dict[str, float]:
    r2 = su2_adjoint_radial_moment(u, r, dr, 2)
    r4 = su2_adjoint_radial_moment(u, r, dr, 4)
    potential = 0.5 * omega**2 * r2 + 0.5 * coupling * r4
    kinetic_from_energy = energy - potential
    virial_rhs = omega**2 * r2 + 2.0 * coupling * r4
    virial_residual = 2.0 * kinetic_from_energy - virial_rhs
    return {
        "r2": float(r2.detach()),
        "r4": float(r4.detach()),
        "tr_x2": float(r2.detach()),
        "tr_x4": float((0.5 * r4).detach()),
        "potential": float(potential.detach()),
        "kinetic_from_energy": float(kinetic_from_energy.detach()),
        "virial_rhs": float(virial_rhs.detach()),
        "virial_residual": float(virial_residual.detach()),
    }


def radial_comparison_payload(
    model: SU2AdjointSpectralAnsatz,
    observables: dict[str, float],
    *,
    omega: float,
    coupling: float,
    r_max: float,
    n_grid: int,
    hf_delta: float = 1.0e-3,
) -> dict[str, Any]:
    reference = su2_adjoint_radial_finite_difference_result(
        omega=omega,
        coupling=coupling,
        r_max=r_max,
        n_grid=n_grid,
        dtype=torch.float64,
    )
    r = reference.r
    dr = reference.dr
    u_reference = reference.u
    u_model = su2_adjoint_model_radial_wavefunction(model, r, dr)
    overlap = su2_adjoint_radial_inner(u_model, u_reference, dr)
    if overlap < 0:
        u_model = -u_model
        overlap = -overlap
    difference = u_model - u_reference
    l2_difference = torch.sqrt(
        su2_adjoint_radial_inner(difference, difference, dr)
    )
    max_abs_difference = torch.max(torch.abs(difference))

    neural_moments = radial_moment_payload(
        u_model,
        r,
        dr,
        omega=omega,
        coupling=coupling,
        energy=observables["energy"],
    )
    reference_moments = radial_moment_payload(
        u_reference,
        r,
        dr,
        omega=omega,
        coupling=coupling,
        energy=reference.energy,
    )

    e_plus = su2_adjoint_radial_finite_difference_energy(
        omega=omega,
        coupling=coupling + hf_delta,
        r_max=r_max,
        n_grid=n_grid,
        dtype=torch.float64,
    )
    e_minus = su2_adjoint_radial_finite_difference_energy(
        omega=omega,
        coupling=coupling - hf_delta,
        r_max=r_max,
        n_grid=n_grid,
        dtype=torch.float64,
    )
    hf_derivative = (e_plus - e_minus) / (2.0 * hf_delta)

    neural_residual = su2_adjoint_radial_residual_norm(
        u_model,
        r,
        dr,
        energy=observables["energy"],
        omega=omega,
        coupling=coupling,
    )
    reference_residual = su2_adjoint_radial_residual_norm(
        u_reference,
        r,
        dr,
        energy=reference.energy,
        omega=omega,
        coupling=coupling,
    )

    return {
        "reference_energy": reference.energy,
        "neural_energy": observables["energy"],
        "energy_difference": observables["energy"] - reference.energy,
        "overlap": float(overlap.detach()),
        "overlap_defect": float((1.0 - overlap**2).detach()),
        "l2_wavefunction_difference": float(l2_difference.detach()),
        "max_abs_wavefunction_difference": float(max_abs_difference.detach()),
        "neural_moments": neural_moments,
        "reference_moments": reference_moments,
        "moment_differences": {
            key: neural_moments[key] - reference_moments[key]
            for key in ["r2", "r4", "tr_x2", "tr_x4", "potential"]
        },
        "hellmann_feynman": {
            "delta_g": hf_delta,
            "finite_difference_derivative": hf_derivative,
            "neural_tr_x4": neural_moments["tr_x4"],
            "reference_tr_x4": reference_moments["tr_x4"],
            "neural_error": neural_moments["tr_x4"] - hf_derivative,
            "reference_error": reference_moments["tr_x4"] - hf_derivative,
        },
        "schrodinger_residuals": {
            "neural_l2": float(neural_residual.detach()),
            "reference_l2": float(reference_residual.detach()),
        },
        "radial_profile": {
            "r": r.tolist(),
            "u_neural": u_model.tolist(),
            "u_reference": u_reference.tolist(),
            "abs_difference": torch.abs(difference).tolist(),
        },
    }


def run_case(
    *,
    omega: float,
    coupling: float,
    benchmark: float,
    omega_init: float,
    hidden_layers: tuple[int, ...],
    n_steps: int,
    lr: float,
    report_every: int,
    z_max: float,
    n_grid: int,
) -> tuple[SU2AdjointSpectralAnsatz, list[dict[str, float]], dict[str, float]]:
    _, lam, weights = su2_adjoint_eigenvalue_grid(
        z_max,
        n_grid,
        dtype=torch.float64,
    )
    model = SU2AdjointSpectralAnsatz(
        omega_init=omega_init,
        hidden_layers=hidden_layers,
        dtype=torch.float64,
    )
    history = train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=omega,
        coupling=coupling,
        n_steps=n_steps,
        lr=lr,
        report_every=report_every,
    )
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=omega,
        coupling=coupling,
    )
    return (
        model,
        training_history_payload(history, benchmark),
        observables_payload(obs, benchmark),
    )


def validation_payload(results: dict[str, Any]) -> dict[str, Any]:
    exact = results["exact_harmonic"]["observables"]
    harmonic = results["harmonic_training"]["observables"]
    quartic = results["quartic_training"]["observables"]
    radial = results["quartic_training"]["radial_comparison"]
    symmetry = results["symmetry_checks"]
    checks = {
        "exact_harmonic_energy_error_lt_1e-10": exact["absolute_energy_error"]
        < 1.0e-10,
        "harmonic_training_energy_error_lt_5e-4": harmonic["absolute_energy_error"]
        < 5.0e-4,
        "quartic_training_energy_error_lt_5e-5": quartic["absolute_energy_error"]
        < 5.0e-5,
        "traceless_residual_lt_1e-12": symmetry["traceless_residual"] < 1.0e-12,
        "odd_parity_residual_lt_1e-12": symmetry["odd_parity_residual"]
        < 1.0e-12,
        "weyl_covariance_residual_lt_1e-12": symmetry["weyl_covariance_residual"]
        < 1.0e-12,
        "head_collision_residual_lt_1e-12": symmetry["head_collision_residual"]
        < 1.0e-12,
        "profile_collision_residual_lt_1e-10": symmetry[
            "profile_collision_residual"
        ]
        < 1.0e-10,
        "so3_covariance_residual_lt_1e-12": symmetry["so3_covariance_residual"]
        < 1.0e-12,
        "quartic_radial_overlap_gt_0.9999": radial["overlap"] > 0.9999,
        "quartic_radial_l2_difference_lt_0.02": radial[
            "l2_wavefunction_difference"
        ]
        < 2.0e-2,
        "quartic_r2_moment_error_lt_0.005": abs(
            radial["moment_differences"]["r2"]
        )
        < 5.0e-3,
        "quartic_r4_moment_error_lt_0.03": abs(
            radial["moment_differences"]["r4"]
        )
        < 3.0e-2,
        "quartic_virial_residual_lt_0.002": abs(
            radial["neural_moments"]["virial_residual"]
        )
        < 2.0e-3,
        "quartic_hellmann_feynman_error_lt_0.002": abs(
            radial["hellmann_feynman"]["neural_error"]
        )
        < 2.0e-3,
        "quartic_schrodinger_residual_lt_0.05": radial[
            "schrodinger_residuals"
        ]["neural_l2"]
        < 5.0e-2,
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "unit_test_command": "python -m pytest tests/test_su2_adjoint.py -q",
        "full_test_command": "python -m pytest -q",
        "unit_test_file": "tests/test_su2_adjoint.py",
        "unit_tests": [
            "test_traceless_hyperplane_basis_is_orthonormal",
            "test_su2_adjoint_ansatz_traceless_weyl_and_odd",
            "test_su2_harmonic_adjoint_energy_is_exact_by_quadrature",
            "test_su2_adjoint_profile_normalization",
            "test_su2_harmonic_training_converges_to_adjoint_ground_state",
            "test_su2_quartic_training_matches_radial_benchmark",
            "test_su2_quartic_radial_wavefunction_checks",
            "test_su2_component_wavefunction_is_so3_covariant",
        ],
    }


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    omega = args.omega
    exact_energy = exact_su2_harmonic_adjoint_energy(omega)

    _, lam_exact, weights_exact = su2_adjoint_eigenvalue_grid(
        args.z_max,
        args.n_grid,
        dtype=torch.float64,
    )
    exact_model = SU2AdjointSpectralAnsatz(
        omega_init=omega,
        hidden_layers=(),
        dtype=torch.float64,
    )
    exact_obs = adjoint_quadrature_observables(
        exact_model,
        lam_exact,
        weights_exact,
        omega=omega,
        coupling=0.0,
    )

    harmonic_model, harmonic_history, harmonic_obs = run_case(
        omega=omega,
        coupling=0.0,
        benchmark=exact_energy,
        omega_init=0.45 * omega,
        hidden_layers=(24, 24),
        n_steps=args.harmonic_steps,
        lr=args.lr,
        report_every=args.report_every,
        z_max=args.z_max,
        n_grid=args.n_grid,
    )

    quartic_benchmark = su2_adjoint_radial_finite_difference_energy(
        omega=omega,
        coupling=args.quartic_coupling,
        r_max=args.fd_r_max,
        n_grid=args.fd_grid,
        dtype=torch.float64,
    )
    quartic_model, quartic_history, quartic_obs = run_case(
        omega=omega,
        coupling=args.quartic_coupling,
        benchmark=quartic_benchmark,
        omega_init=omega,
        hidden_layers=(32, 32),
        n_steps=args.quartic_steps,
        lr=args.lr,
        report_every=args.report_every,
        z_max=args.z_max,
        n_grid=args.n_grid,
    )
    quartic_radial_comparison = radial_comparison_payload(
        quartic_model,
        quartic_obs,
        omega=omega,
        coupling=args.quartic_coupling,
        r_max=args.fd_r_max,
        n_grid=args.fd_grid,
    )

    results: dict[str, Any] = {
        "metadata": {
            "model": "one-matrix SU(2) adjoint sector",
            "hamiltonian": "-1/2 Delta_X + 1/2 omega^2 Tr X^2 + g Tr X^4",
            "generator_normalization": "Tr(T^a T^b)=delta^{ab}",
            "seed": args.seed,
            "omega": omega,
            "z_max": args.z_max,
            "n_grid": args.n_grid,
            "learning_rate": args.lr,
            "report_every": args.report_every,
            "fd_grid": args.fd_grid,
            "fd_r_max": args.fd_r_max,
        },
        "exact_harmonic": {
            "benchmark": "analytic SU(2) adjoint harmonic energy 5 omega / 2",
            "observables": observables_payload(exact_obs, exact_energy),
            "profile": profile_payload(
                exact_model,
                norm_z_max=args.z_max,
                norm_n_grid=args.n_grid,
            ),
        },
        "harmonic_training": {
            "benchmark": "analytic SU(2) adjoint harmonic energy 5 omega / 2",
            "coupling": 0.0,
            "omega_init": 0.45 * omega,
            "hidden_layers": [24, 24],
            "steps": args.harmonic_steps,
            "history": harmonic_history,
            "observables": harmonic_obs,
            "profile": profile_payload(
                harmonic_model,
                norm_z_max=args.z_max,
                norm_n_grid=args.n_grid,
            ),
        },
        "quartic_training": {
            "benchmark": "independent l=1 radial finite-difference energy",
            "coupling": args.quartic_coupling,
            "omega_init": omega,
            "hidden_layers": [32, 32],
            "steps": args.quartic_steps,
            "history": quartic_history,
            "observables": quartic_obs,
            "profile": profile_payload(
                quartic_model,
                norm_z_max=args.z_max,
                norm_n_grid=args.n_grid,
            ),
            "radial_comparison": quartic_radial_comparison,
        },
        "symmetry_checks": symmetry_checks(harmonic_model),
    }
    results["validation_summary"] = validation_payload(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"wrote {args.output}")
    print(json.dumps(results["validation_summary"], indent=2))
    print()
    print("final energies")
    print(f"exact harmonic     {exact_obs.energy:.15f}  target {exact_energy:.15f}")
    print(
        f"harmonic training  {harmonic_obs['energy']:.15f}  "
        f"error {harmonic_obs['energy_error']:+.3e}"
    )
    print(
        f"quartic training   {quartic_obs['energy']:.15f}  "
        f"fd {quartic_benchmark:.15f}  "
        f"error {quartic_obs['energy_error']:+.3e}"
    )


if __name__ == "__main__":
    main()
