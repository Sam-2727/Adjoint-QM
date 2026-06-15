#!/usr/bin/env python
"""Run and save SU(3) adjoint-sector benchmark results for the notebook."""

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
    SUNAdjointRadialSpectralAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_profile_norm,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    exact_suN_harmonic_adjoint_energy,
    su2_adjoint_radial_inner,
    su2_adjoint_radial_moment,
    suN_adjoint_model_radial_wavefunction,
    suN_adjoint_radial_finite_difference_energy,
    suN_adjoint_radial_finite_difference_result,
    suN_adjoint_radial_residual_norm,
    train_adjoint_quadrature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "su3_adjoint_benchmarks.json",
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=3033)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--fd-grid", type=int, default=1600)
    parser.add_argument("--fd-r-max", type=float, default=7.0)
    parser.add_argument("--hf-delta", type=float, default=1.0e-3)
    return parser.parse_args()


def quartic_tail_initialization(coupling: float) -> float:
    """WKB coefficient for ``S ~ c r^3`` when ``V ~ 0.5*g*r^4``."""

    return 2.0 * coupling**0.5 / 3.0


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
            "cubic": record.cubic,
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
        "cubic": obs.cubic,
    }


def profile_payload(
    model: SUNAdjointRadialSpectralAnsatz,
    *,
    plot_r_max: float = 4.0,
    plot_n_grid: int = 800,
    norm_z_max: float,
    norm_n_grid: int,
) -> dict[str, Any]:
    r = torch.linspace(0.0, plot_r_max, plot_n_grid, dtype=torch.float64)
    dr = float(r[1] - r[0])
    safe_r = torch.where(r == 0.0, torch.full_like(r, 1.0e-12), r)
    lam = torch.zeros((plot_n_grid, 3), dtype=torch.float64)
    lam[:, 0] = safe_r / 2.0**0.5
    lam[:, 1] = -safe_r / 2.0**0.5
    _, norm_lam, norm_weights = adjoint_eigenvalue_grid(
        3,
        norm_z_max,
        norm_n_grid,
        dtype=torch.float64,
    )
    with torch.no_grad():
        norm = adjoint_profile_norm(model, norm_lam, norm_weights)
        scale = 1.0 / torch.sqrt(norm)
        profile = model.profile(lam)
        normalized_profile = scale * profile
        radial_u = suN_adjoint_model_radial_wavefunction(model, safe_r, dr)
    radial_u = torch.where(r == 0.0, torch.zeros_like(radial_u), radial_u)
    return {
        "r": r.tolist(),
        "lambda_0": lam[:, 0].tolist(),
        "q_0_normalized": normalized_profile[:, 0].tolist(),
        "radial_u": radial_u.tolist(),
        "adjoint_norm": float(norm.detach()),
        "normalization_scale": float(scale.detach()),
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


def structure_checks(model: SUNAdjointRadialSpectralAnsatz) -> dict[str, float]:
    lam = torch.tensor(
        [
            [0.8, -0.3, -0.5],
            [1.1, 0.2, -1.3],
            [-0.4, 1.0, -0.6],
        ],
        dtype=torch.float64,
    )
    diagnostics = adjoint_structure_diagnostics(model, lam)
    return {
        "traceless_residual": diagnostics.traceless_residual,
        "odd_parity_residual": diagnostics.parity_residual,
        "weyl_covariance_residual": diagnostics.weyl_residual,
        "head_collision_residual": diagnostics.head_collision_residual,
        "profile_collision_residual": diagnostics.profile_collision_residual,
    }


def radial_comparison_payload(
    model: SUNAdjointRadialSpectralAnsatz,
    observables: dict[str, float],
    *,
    omega: float,
    coupling: float,
    r_max: float,
    n_grid: int,
    hf_delta: float,
) -> dict[str, Any]:
    reference = suN_adjoint_radial_finite_difference_result(
        3,
        omega=omega,
        coupling=coupling,
        r_max=r_max,
        n_grid=n_grid,
        dtype=torch.float64,
    )
    r = reference.r
    dr = reference.dr
    u_reference = reference.u
    u_model = suN_adjoint_model_radial_wavefunction(model, r, dr)
    overlap = su2_adjoint_radial_inner(u_model, u_reference, dr)
    if overlap < 0:
        u_model = -u_model
        overlap = -overlap
    difference = u_model - u_reference
    l2_difference = torch.sqrt(su2_adjoint_radial_inner(difference, difference, dr))
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

    e_plus = suN_adjoint_radial_finite_difference_energy(
        3,
        omega=omega,
        coupling=coupling + hf_delta,
        r_max=r_max,
        n_grid=n_grid,
        dtype=torch.float64,
    )
    e_minus = suN_adjoint_radial_finite_difference_energy(
        3,
        omega=omega,
        coupling=coupling - hf_delta,
        r_max=r_max,
        n_grid=n_grid,
        dtype=torch.float64,
    )
    hf_derivative = (e_plus - e_minus) / (2.0 * hf_delta)

    neural_residual = suN_adjoint_radial_residual_norm(
        3,
        u_model,
        r,
        dr,
        energy=observables["energy"],
        omega=omega,
        coupling=coupling,
    )
    reference_residual = suN_adjoint_radial_residual_norm(
        3,
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


def validation_payload(results: dict[str, Any]) -> dict[str, Any]:
    harmonic = results["exact_harmonic"]["observables"]
    quartic = results["quartic_training"]["observables"]
    radial = results["quartic_training"]["radial_comparison"]
    symmetry = results["quartic_training"]["structure_checks"]
    checks = {
        "harmonic_energy_error_lt_1e-4": harmonic["absolute_energy_error"] < 1.0e-4,
        "quartic_energy_error_lt_1e-4": quartic["absolute_energy_error"] < 1.0e-4,
        "traceless_residual_lt_1e-12": symmetry["traceless_residual"] < 1.0e-12,
        "odd_parity_residual_lt_1e-12": symmetry["odd_parity_residual"]
        < 1.0e-12,
        "weyl_covariance_residual_lt_1e-12": symmetry["weyl_covariance_residual"]
        < 1.0e-12,
        "head_collision_residual_lt_1e-12": symmetry["head_collision_residual"]
        < 1.0e-12,
        "profile_collision_residual_lt_1e-6": symmetry[
            "profile_collision_residual"
        ]
        < 1.0e-6,
        "quartic_radial_overlap_gt_0.99999": radial["overlap"] > 0.99999,
        "quartic_radial_l2_difference_lt_1e-3": radial[
            "l2_wavefunction_difference"
        ]
        < 1.0e-3,
        "quartic_r2_moment_error_lt_1e-4": abs(
            radial["moment_differences"]["r2"]
        )
        < 1.0e-4,
        "quartic_r4_moment_error_lt_1e-3": abs(
            radial["moment_differences"]["r4"]
        )
        < 1.0e-3,
        "quartic_virial_residual_lt_1e-4": abs(
            radial["neural_moments"]["virial_residual"]
        )
        < 1.0e-4,
        "quartic_hellmann_feynman_error_lt_1e-4": abs(
            radial["hellmann_feynman"]["neural_error"]
        )
        < 1.0e-4,
        "quartic_schrodinger_residual_lt_0.01": radial[
            "schrodinger_residuals"
        ]["neural_l2"]
        < 1.0e-2,
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "unit_test_command": "python -m pytest tests/test_su3_adjoint.py -q",
        "full_test_command": "python -m pytest -q",
        "unit_test_file": "tests/test_su3_adjoint.py",
        "unit_tests": [
            "test_su3_harmonic_quadrature_matches_exact_adjoint_energy",
            "test_su3_harmonic_radial_wavefunction_matches_finite_difference",
            "test_su3_quartic_training_matches_radial_benchmark_at_g_one",
        ],
    }


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    exact_energy = exact_suN_harmonic_adjoint_energy(3, args.omega)
    _, lam, weights = adjoint_eigenvalue_grid(
        3,
        args.z_max,
        args.n_grid,
        dtype=torch.float64,
    )

    harmonic_model = SUNAdjointRadialSpectralAnsatz(
        n=3,
        omega_init=args.omega,
        quartic_init=0.0,
        hidden_layers=(),
        dtype=torch.float64,
    )
    harmonic_obs_raw = adjoint_quadrature_observables(
        harmonic_model,
        lam,
        weights,
        omega=args.omega,
        coupling=0.0,
    )
    harmonic_obs = observables_payload(harmonic_obs_raw, exact_energy)

    radial_reference_energy = suN_adjoint_radial_finite_difference_energy(
        3,
        omega=args.omega,
        coupling=args.coupling,
        r_max=args.fd_r_max,
        n_grid=args.fd_grid,
        dtype=torch.float64,
    )
    model = SUNAdjointRadialSpectralAnsatz(
        n=3,
        omega_init=args.omega,
        quartic_init=quartic_tail_initialization(args.coupling),
        hidden_layers=(args.hidden_width, args.hidden_width),
        dtype=torch.float64,
    )
    history = train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        n_steps=args.steps,
        lr=args.lr,
        report_every=args.report_every,
    )
    quartic_obs_raw = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    quartic_obs = observables_payload(quartic_obs_raw, radial_reference_energy)
    radial_comparison = radial_comparison_payload(
        model,
        quartic_obs,
        omega=args.omega,
        coupling=args.coupling,
        r_max=args.fd_r_max,
        n_grid=args.fd_grid,
        hf_delta=args.hf_delta,
    )

    results: dict[str, Any] = {
        "metadata": {
            "model": "one-matrix SU(3) adjoint sector",
            "hamiltonian": "-1/2 Delta_X + 1/2 omega^2 Tr X^2 + g Tr X^4",
            "radial_identity": "For traceless 3x3 X, Tr X^4 = 0.5*(Tr X^2)^2.",
            "generator_normalization": "Tr(T^a T^b)=delta^{ab}",
            "seed": args.seed,
            "omega": args.omega,
            "coupling": args.coupling,
            "z_max": args.z_max,
            "n_grid": args.n_grid,
            "learning_rate": args.lr,
            "steps": args.steps,
            "fd_grid": args.fd_grid,
            "fd_r_max": args.fd_r_max,
        },
        "exact_harmonic": {
            "benchmark": "analytic SU(3) adjoint harmonic energy",
            "observables": harmonic_obs,
        },
        "quartic_training": {
            "benchmark": "independent 8D l=1 radial finite-difference energy",
            "coupling": args.coupling,
            "hidden_layers": [args.hidden_width, args.hidden_width],
            "steps": args.steps,
            "history": training_history_payload(history, radial_reference_energy),
            "observables": quartic_obs,
            "structure_checks": structure_checks(model),
            "profile": profile_payload(
                model,
                norm_z_max=args.z_max,
                norm_n_grid=args.n_grid,
            ),
            "radial_comparison": radial_comparison,
        },
    }
    results["validation_summary"] = validation_payload(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"wrote {args.output}")
    print(json.dumps(results["validation_summary"], indent=2))
    print()
    print("final energies")
    print(f"exact harmonic  {harmonic_obs['energy']:.15f} target {exact_energy:.15f}")
    print(
        f"quartic g={args.coupling:g}  {quartic_obs['energy']:.15f} "
        f"fd {radial_reference_energy:.15f} "
        f"error {quartic_obs['energy_error']:+.3e}"
    )
    print(
        f"overlap {radial_comparison['overlap']:.12f} "
        f"l2 {radial_comparison['l2_wavefunction_difference']:.3e}"
    )


if __name__ == "__main__":
    main()
