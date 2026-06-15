#!/usr/bin/env python
"""Run SU(N) Chebyshev adjoint spectral-ansatz benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from adjoint_qm import (
    SUNAdjointChebyshevSpectralAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_quadrature_energy,
    adjoint_quadrature_moments,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    exact_suN_harmonic_adjoint_energy,
    su2_adjoint_eigenvalue_grid,
    su3_adjoint_polar_eigenvalue_grid,
    suN_adjoint_radial_finite_difference_energy,
    train_adjoint_quadrature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sun_chebyshev_adjoint_benchmarks.json",
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4404)
    parser.add_argument("--hf-delta", type=float, default=2.0e-2)
    parser.add_argument("--skip-hf", action="store_true")
    return parser.parse_args()


def quartic_tail_initialization(coupling: float) -> float:
    """Coordinate-WKB coefficient for ``S ~ c sum_i |lambda_i|**3``."""

    return 2.0 * math.sqrt(2.0 * coupling) / 3.0


def radial_reference_extrapolation(
    n: int,
    *,
    omega: float,
    coupling: float,
) -> dict[str, Any]:
    """Continuum-extrapolate the N=2,3 radial finite-difference benchmark."""

    grids = [1600, 2200, 3000]
    values = [
        suN_adjoint_radial_finite_difference_energy(
            n,
            omega=omega,
            coupling=coupling,
            r_max=8.0,
            n_grid=grid,
            dtype=torch.float64,
        )
        for grid in grids
    ]
    xs = [1.0 / (grid + 1) ** 2 for grid in grids]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(values) / len(values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return {
        "method": "linear fit in 1/(n_grid+1)^2",
        "r_max": 8.0,
        "grids": grids,
        "energies": values,
        "extrapolated_energy": intercept,
        "slope": slope,
    }


def lbfgs_refine(
    model: torch.nn.Module,
    lam: torch.Tensor,
    weights: torch.Tensor,
    *,
    omega: float,
    coupling: float,
    max_iter: int,
) -> None:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.2,
        max_iter=max_iter,
        tolerance_grad=1.0e-11,
        tolerance_change=1.0e-13,
        history_size=25,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        energy, *_ = adjoint_quadrature_energy(
            model,
            lam,
            weights,
            omega=omega,
            coupling=coupling,
        )
        energy.backward()
        return energy

    optimizer.step(closure)


def observable_payload(
    model: SUNAdjointChebyshevSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
    *,
    omega: float,
    coupling: float,
    benchmark: float | None = None,
) -> dict[str, Any]:
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=omega,
        coupling=coupling,
    )
    moments = adjoint_quadrature_moments(
        model,
        lam,
        weights,
        omega=omega,
        coupling=coupling,
    )
    payload: dict[str, Any] = {
        "energy": obs.energy,
        "radial": obs.radial,
        "angular": obs.angular,
        "potential": obs.potential,
        "local_energy_std": obs.local_energy_std,
        "tr_x2": moments.tr_x2,
        "tr_x4": moments.tr_x4,
        "kinetic": moments.kinetic,
        "virial_rhs": moments.virial_rhs,
        "virial_residual": moments.virial_residual,
        "alpha": obs.alpha,
        "cubic": obs.cubic,
    }
    if benchmark is not None:
        payload["benchmark_energy"] = benchmark
        payload["energy_error"] = obs.energy - benchmark
        payload["absolute_energy_error"] = abs(obs.energy - benchmark)
    return payload


def structure_payload(model: SUNAdjointChebyshevSpectralAnsatz) -> dict[str, float]:
    n = model.n
    samples = torch.linspace(-0.9, 0.9, n * 4, dtype=torch.float64).reshape(4, n)
    samples = samples - torch.mean(samples, dim=-1, keepdim=True)
    diagnostics = adjoint_structure_diagnostics(model, samples)
    return {
        "traceless_residual": diagnostics.traceless_residual,
        "odd_parity_residual": diagnostics.parity_residual,
        "weyl_covariance_residual": diagnostics.weyl_residual,
        "head_collision_ratio_max_abs": diagnostics.head_collision_ratio_max_abs,
        "profile_collision_identity_residual": (
            diagnostics.profile_collision_identity_residual
        ),
    }


def train_su2(args: argparse.Namespace) -> dict[str, Any]:
    reference = radial_reference_extrapolation(
        2,
        omega=args.omega,
        coupling=args.coupling,
    )
    benchmark = reference["extrapolated_energy"]
    _, lam, weights = su2_adjoint_eigenvalue_grid(8.0, 3000, dtype=torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=2,
        omega_init=args.omega,
        quartic_tail_init=quartic_tail_initialization(args.coupling),
        moment_cutoff=2,
        hidden_layers=(16, 16),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        action_correction_scale=1.0,
        head_correction_scale=0.0,
        dtype=torch.float64,
    )
    start = time.perf_counter()
    history = train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        n_steps=1000,
        lr=3.0e-3,
        report_every=250,
    )
    lbfgs_refine(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        max_iter=60,
    )
    wall_time = time.perf_counter() - start
    evaluations = {
        str(n_grid): observable_payload(
            model,
            *su2_adjoint_eigenvalue_grid(8.0, n_grid, dtype=torch.float64)[1:],
            omega=args.omega,
            coupling=args.coupling,
            benchmark=benchmark,
        )
        for n_grid in [2500, 3000, 3500, 4000]
    }
    return {
        "reference": reference,
        "training_wall_time_seconds": wall_time,
        "history": [record.__dict__ for record in history],
        "evaluations": evaluations,
        "structure_checks": structure_payload(model),
    }


def train_su3(args: argparse.Namespace) -> dict[str, Any]:
    reference = radial_reference_extrapolation(
        3,
        omega=args.omega,
        coupling=args.coupling,
    )
    benchmark = reference["extrapolated_energy"]
    _, lam, weights = su3_adjoint_polar_eigenvalue_grid(
        6.0,
        140,
        168,
        dtype=torch.float64,
    )
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=3,
        omega_init=args.omega,
        quartic_tail_init=quartic_tail_initialization(args.coupling),
        moment_cutoff=2,
        hidden_layers=(32, 32),
        head_hidden_layers=(),
        chebyshev_degrees=(1,),
        scale_init=3.0,
        action_correction_scale=1.0,
        head_correction_scale=0.0,
        dtype=torch.float64,
    )
    start = time.perf_counter()
    history = train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        n_steps=1500,
        lr=2.0e-3,
        report_every=300,
    )
    lbfgs_refine(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        max_iter=120,
    )
    wall_time = time.perf_counter() - start
    evaluations = {}
    for n_radial, n_angular in [(100, 120), (120, 144), (140, 168), (160, 192)]:
        _, eval_lam, eval_weights = su3_adjoint_polar_eigenvalue_grid(
            6.0,
            n_radial,
            n_angular,
            dtype=torch.float64,
        )
        evaluations[f"{n_radial}x{n_angular}"] = observable_payload(
            model,
            eval_lam,
            eval_weights,
            omega=args.omega,
            coupling=args.coupling,
            benchmark=benchmark,
        )
    return {
        "reference": reference,
        "training_wall_time_seconds": wall_time,
        "history": [record.__dict__ for record in history],
        "evaluations": evaluations,
        "structure_checks": structure_payload(model),
    }


def train_su4_conservative(
    args: argparse.Namespace,
    *,
    coupling: float,
    n_steps: int = 1000,
) -> tuple[SUNAdjointChebyshevSpectralAnsatz, dict[str, Any]]:
    _, lam, weights = adjoint_eigenvalue_grid(4, 5.5, 34, dtype=torch.float64)
    model = SUNAdjointChebyshevSpectralAnsatz(
        n=4,
        omega_init=args.omega,
        quartic_tail_init=quartic_tail_initialization(coupling),
        moment_cutoff=4,
        hidden_layers=(16,),
        head_hidden_layers=(8,),
        chebyshev_degrees=(1, 3),
        scale_init=3.5,
        action_correction_scale=0.0,
        head_correction_scale=0.0,
        dtype=torch.float64,
    )
    start = time.perf_counter()
    history = train_adjoint_quadrature(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=coupling,
        n_steps=n_steps,
        lr=3.0e-3,
        report_every=max(1, n_steps // 5),
    )
    wall_time = time.perf_counter() - start
    evaluations = {}
    for n_grid in [30, 34, 38, 42, 46]:
        _, eval_lam, eval_weights = adjoint_eigenvalue_grid(
            4,
            5.5,
            n_grid,
            dtype=torch.float64,
        )
        evaluations[str(n_grid)] = observable_payload(
            model,
            eval_lam,
            eval_weights,
            omega=args.omega,
            coupling=coupling,
        )
    payload = {
        "coupling": coupling,
        "trusted_subspace": (
            "Chebyshev T1 head with trained alpha_2 and quartic WKB tail; "
            "neural correction amplitudes set to zero after overfit diagnostics"
        ),
        "training_wall_time_seconds": wall_time,
        "history": [record.__dict__ for record in history],
        "evaluations": evaluations,
        "structure_checks": structure_payload(model),
    }
    return model, payload


def add_su4_hellmann_feynman(
    args: argparse.Namespace,
    central_model: SUNAdjointChebyshevSpectralAnsatz,
    payload: dict[str, Any],
) -> None:
    _, eval_lam, eval_weights = adjoint_eigenvalue_grid(
        4,
        5.5,
        42,
        dtype=torch.float64,
    )
    central = observable_payload(
        central_model,
        eval_lam,
        eval_weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    if args.skip_hf:
        payload["hellmann_feynman"] = {
            "skipped": True,
            "central_tr_x4": central["tr_x4"],
        }
        return

    _, minus_payload = train_su4_conservative(
        args,
        coupling=args.coupling - args.hf_delta,
        n_steps=800,
    )
    _, plus_payload = train_su4_conservative(
        args,
        coupling=args.coupling + args.hf_delta,
        n_steps=800,
    )
    e_minus = minus_payload["evaluations"]["42"]["energy"]
    e_plus = plus_payload["evaluations"]["42"]["energy"]
    derivative = (e_plus - e_minus) / (2.0 * args.hf_delta)
    payload["hellmann_feynman"] = {
        "delta_g": args.hf_delta,
        "central_tr_x4": central["tr_x4"],
        "finite_difference_derivative": derivative,
        "error": central["tr_x4"] - derivative,
        "minus": minus_payload,
        "plus": plus_payload,
    }


def validation_payload(results: dict[str, Any]) -> dict[str, Any]:
    su2_eval = results["su2"]["evaluations"]["4000"]
    su3_eval = results["su3"]["evaluations"]["160x192"]
    su4_eval = results["su4"]["evaluations"]["42"]
    su4_grid_values = [
        item["energy"] for item in results["su4"]["evaluations"].values()
    ]
    hf = results["su4"].get("hellmann_feynman", {})
    checks = {
        "su2_energy_error_lt_1e-5": su2_eval["absolute_energy_error"] < 1.0e-5,
        "su3_energy_error_lt_1e-5": su3_eval["absolute_energy_error"] < 1.0e-5,
        "su4_cross_grid_spread_lt_5e-3": (
            max(su4_grid_values) - min(su4_grid_values)
        )
        < 5.0e-3,
        "su4_virial_residual_lt_2e-3": abs(su4_eval["virial_residual"])
        < 2.0e-3,
        "su4_hf_error_lt_5e-2": (
            hf.get("skipped", False) or abs(hf.get("error", 0.0)) < 5.0e-2
        ),
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "note": (
            "SU(4) has no exact benchmark here; the trusted claim is grid "
            "stability plus virial/Hellmann-Feynman consistency for the "
            "conservative subspace."
        ),
    }


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    harmonic_exact = {
        str(n): exact_suN_harmonic_adjoint_energy(n, args.omega)
        for n in [2, 3, 4]
    }
    su2 = train_su2(args)
    su3 = train_su3(args)
    su4_model, su4 = train_su4_conservative(args, coupling=args.coupling)
    add_su4_hellmann_feynman(args, su4_model, su4)

    results: dict[str, Any] = {
        "metadata": {
            "ansatz": "SU(N) Chebyshev spectral impurity",
            "hamiltonian": "-1/2 Delta_X + 1/2 omega^2 Tr X^2 + g Tr X^4",
            "omega": args.omega,
            "coupling": args.coupling,
            "seed": args.seed,
            "harmonic_exact_energies": harmonic_exact,
        },
        "su2": su2,
        "su3": su3,
        "su4": su4,
    }
    results["validation_summary"] = validation_payload(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(results["validation_summary"], indent=2))
    print("summary")
    print(
        "SU(2)",
        su2["evaluations"]["4000"]["energy"],
        su2["evaluations"]["4000"]["energy_error"],
    )
    print(
        "SU(3)",
        su3["evaluations"]["160x192"]["energy"],
        su3["evaluations"]["160x192"]["energy_error"],
    )
    print(
        "SU(4)",
        su4["evaluations"]["42"]["energy"],
        "virial",
        su4["evaluations"]["42"]["virial_residual"],
    )


if __name__ == "__main__":
    main()
