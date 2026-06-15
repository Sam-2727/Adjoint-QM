#!/usr/bin/env python
"""Run shared-ansatz SU(N) adjoint Monte Carlo benchmarks."""

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
    adjoint_importance_moments,
    adjoint_importance_observables,
    adjoint_quadrature_energy,
    adjoint_quadrature_moments,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    exact_suN_harmonic_adjoint_energy,
    log_vandermonde_abs,
    sobol_gaussian_traceless_samples,
    su3_adjoint_polar_eigenvalue_grid,
    suN_adjoint_radial_finite_difference_energy,
    train_adjoint_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sun_adjoint_vmc_benchmarks.json",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument(
        "--training-mode",
        choices=["importance-adam", "validated-candidates"],
        default="validated-candidates",
    )
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument("--eval-samples", type=int, default=131072)
    parser.add_argument("--n-eval-replicates", type=int, default=3)
    parser.add_argument("--proposal-sigma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--envelope-lbfgs-max-iter", type=int, default=80)
    parser.add_argument("--learn-coordinate-scale", action="store_true")
    parser.add_argument("--action-correction-scale", type=float, default=0.25)
    parser.add_argument("--head-correction-scale", type=float, default=0.25)
    parser.add_argument("--energy-abs-tol", type=float, default=5.0e-4)
    parser.add_argument("--virial-abs-tol", type=float, default=5.0e-3)
    parser.add_argument("--min-ess-fraction", type=float, default=0.05)
    return parser.parse_args()


def quartic_tail_initialization(coupling: float) -> float:
    """Coordinate-WKB coefficient for ``S ~ c sum_i |lambda_i|**3``."""

    return 2.0 * math.sqrt(2.0 * coupling) / 3.0


def shared_ansatz_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the identical flexible ansatz settings used for every N."""

    return {
        "omega_init": args.omega,
        "quartic_tail_init": quartic_tail_initialization(args.coupling),
        "moment_cutoff": 6,
        "hidden_layers": [24, 24],
        "head_hidden_layers": [24],
        "chebyshev_degrees": [1, 3, 5],
        "scale_init": 3.0,
        "learn_scale": False,
        "coordinate_scale_init": 1.0,
        "learn_coordinate_scale": args.learn_coordinate_scale,
        "action_correction_scale": args.action_correction_scale,
        "head_correction_scale": args.head_correction_scale,
    }


def make_shared_model(
    n: int,
    args: argparse.Namespace,
) -> SUNAdjointChebyshevSpectralAnsatz:
    config = shared_ansatz_config(args)
    return SUNAdjointChebyshevSpectralAnsatz(
        n=n,
        omega_init=config["omega_init"],
        quartic_tail_init=config["quartic_tail_init"],
        moment_cutoff=config["moment_cutoff"],
        hidden_layers=tuple(config["hidden_layers"]),
        head_hidden_layers=tuple(config["head_hidden_layers"]),
        chebyshev_degrees=tuple(config["chebyshev_degrees"]),
        scale_init=config["scale_init"],
        learn_scale=config["learn_scale"],
        coordinate_scale_init=config["coordinate_scale_init"],
        learn_coordinate_scale=config["learn_coordinate_scale"],
        action_correction_scale=config["action_correction_scale"],
        head_correction_scale=config["head_correction_scale"],
        dtype=torch.float64,
    )


def envelope_training_grid(
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return a deterministic grid for low-dimensional envelope warm starts."""

    if n == 2:
        _, lam, weights = adjoint_eigenvalue_grid(
            2,
            8.0,
            3000,
            dtype=torch.float64,
        )
        metadata = {
            "type": "midpoint tensor grid in one traceless coordinate",
            "z_max": 8.0,
            "n_grid": 3000,
            "point_count": int(lam.shape[0]),
        }
    elif n == 3:
        _, lam, weights = su3_adjoint_polar_eigenvalue_grid(
            6.0,
            120,
            144,
            dtype=torch.float64,
        )
        metadata = {
            "type": "midpoint polar grid on the traceless eigenvalue plane",
            "r_max": 6.0,
            "n_radial": 120,
            "n_angular": 144,
            "point_count": int(lam.shape[0]),
        }
    elif n == 4:
        _, lam, weights = adjoint_eigenvalue_grid(
            4,
            5.5,
            34,
            dtype=torch.float64,
        )
        metadata = {
            "type": "midpoint tensor grid in three traceless coordinates",
            "z_max": 5.5,
            "n_grid": 34,
            "point_count": int(lam.shape[0]),
        }
    else:
        raise ValueError(
            "the deterministic envelope warm start is implemented only for N<=4"
        )
    return lam, weights, metadata


def radial_reference_extrapolation(
    n: int,
    *,
    omega: float,
    coupling: float,
) -> dict[str, Any] | None:
    """Continuum-extrapolate the N=2,3 radial finite-difference benchmark."""

    if n not in (2, 3):
        return None
    grids = [1200, 1800, 2400]
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


def effective_sample_size(
    model: SUNAdjointChebyshevSpectralAnsatz,
    lam: torch.Tensor,
    log_prob: torch.Tensor,
) -> float:
    """Return the self-normalized importance-sampling effective sample size."""

    with torch.no_grad():
        head = model.head(lam)
        log_weights = (
            2.0 * log_vandermonde_abs(lam)
            - model.action(lam)
            - log_prob
            + torch.log(torch.sum(head**2, dim=-1))
        )
        shift = torch.max(log_weights)
        weights = torch.exp(log_weights - shift)
        ess = torch.sum(weights) ** 2 / torch.sum(weights**2)
    return float(ess.detach())


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


def eval_payload(
    model: SUNAdjointChebyshevSpectralAnsatz,
    args: argparse.Namespace,
    *,
    seed: int,
    benchmark: float | None,
) -> dict[str, Any]:
    samples = sobol_gaussian_traceless_samples(
        model.n,
        args.eval_samples,
        sigma=args.proposal_sigma,
        seed=seed,
        dtype=torch.float64,
    )
    obs = adjoint_importance_observables(
        model,
        samples.lam,
        samples.log_prob,
        omega=args.omega,
        coupling=args.coupling,
    )
    moments = adjoint_importance_moments(
        model,
        samples.lam,
        samples.log_prob,
        omega=args.omega,
        coupling=args.coupling,
    )
    ess = effective_sample_size(model, samples.lam, samples.log_prob)
    payload: dict[str, Any] = {
        "energy": obs.energy,
        "local_energy_std": obs.local_energy_std,
        "sample_count": args.eval_samples,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / args.eval_samples,
        "seed": seed,
        "proposal_sigma": samples.sigma,
        "proposal": "scrambled Sobol Gaussian in traceless coordinates",
        "radial": obs.radial,
        "angular": obs.angular,
        "potential": obs.potential,
        "tr_x2": moments.tr_x2,
        "tr_x4": moments.tr_x4,
        "kinetic": moments.kinetic,
        "virial_rhs": moments.virial_rhs,
        "virial_residual": moments.virial_residual,
        "alpha": obs.alpha,
        "cubic": obs.cubic,
        "coordinate_scale": obs.coordinate_scale,
    }
    if benchmark is not None:
        payload["benchmark_energy"] = benchmark
        payload["energy_error"] = obs.energy - benchmark
    return payload


def combined_payload(evaluations: list[dict[str, Any]], benchmark: float | None) -> dict[str, Any]:
    """Summarize independent evaluation replicas."""

    mean_energy = sum(item["energy"] for item in evaluations) / len(evaluations)
    mean_virial = sum(item["virial_residual"] for item in evaluations) / len(
        evaluations
    )
    energy_spread = (
        max(item["energy"] for item in evaluations)
        - min(item["energy"] for item in evaluations)
    )
    virial_abs_max = max(abs(item["virial_residual"]) for item in evaluations)
    min_ess_fraction = min(
        item["effective_sample_size_fraction"] for item in evaluations
    )
    combined: dict[str, Any] = {
        "energy_mean": mean_energy,
        "energy_spread_independent_runs": energy_spread,
        "virial_residual_mean": mean_virial,
        "virial_residual_abs_max": virial_abs_max,
        "min_effective_sample_size_fraction": min_ess_fraction,
    }
    if benchmark is not None:
        combined["benchmark_energy"] = benchmark
        combined["energy_error"] = mean_energy - benchmark
    return combined


def candidate_passes(
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    """Return whether one candidate satisfies the script validation gates."""

    combined = candidate["combined_evaluation"]
    structure = candidate["structure_checks"]
    if combined["min_effective_sample_size_fraction"] <= args.min_ess_fraction:
        return False
    if not (
        structure["traceless_residual"] < 1.0e-10
        and structure["odd_parity_residual"] < 1.0e-10
        and structure["weyl_covariance_residual"] < 1.0e-10
    ):
        return False
    if combined["virial_residual_abs_max"] >= args.virial_abs_tol:
        return False
    if "benchmark_energy" in combined:
        return abs(combined["energy_error"]) < args.energy_abs_tol
    return True


def summarize_candidate(
    name: str,
    model: SUNAdjointChebyshevSpectralAnsatz,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
    training_payload: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one trained candidate and return a serializable payload."""

    evaluations = [
        eval_payload(
            model,
            args,
            seed=args.seed + 1000 * model.n + offset,
            benchmark=benchmark,
        )
        for offset in range(args.n_eval_replicates)
    ]
    combined = combined_payload(evaluations, benchmark)
    candidate = {
        "name": name,
        "training": training_payload,
        "history": history,
        "evaluations": evaluations,
        "combined_evaluation": combined,
        "structure_checks": structure_payload(model),
    }
    candidate["passes_validation_gates"] = candidate_passes(candidate, args)
    return candidate


def train_importance_adam_candidate(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
) -> dict[str, Any]:
    """Train the full flexible candidate with the fixed-proposal Adam path."""

    model = make_shared_model(n, args)
    train_samples = sobol_gaussian_traceless_samples(
        n,
        args.n_samples,
        sigma=args.proposal_sigma,
        seed=args.seed + 100 * n,
        dtype=torch.float64,
    )
    start = time.perf_counter()
    history = train_adjoint_importance(
        model,
        train_samples.lam,
        train_samples.log_prob,
        omega=args.omega,
        coupling=args.coupling,
        n_steps=args.n_steps,
        lr=args.lr,
        report_every=args.report_every,
    )
    wall_time = time.perf_counter() - start
    training_payload = {
        "method": "fixed-proposal Adam on the full flexible ansatz",
        "sample_count": args.n_samples,
        "seed": train_samples.seed,
        "proposal_sigma": train_samples.sigma,
        "effective_sample_size": effective_sample_size(
            model,
            train_samples.lam,
            train_samples.log_prob,
        ),
        "wall_time_seconds": wall_time,
    }
    return summarize_candidate(
        "importance_adam_full_flexible",
        model,
        args,
        benchmark=benchmark,
        training_payload=training_payload,
        history=[record.__dict__ for record in history],
    )


def train_envelope_quadrature_candidate(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
) -> dict[str, Any]:
    """Train the stable envelope subspace inside the shared flexible model."""

    model = make_shared_model(n, args)
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_(parameter_name in {"raw_alpha", "raw_cubic"})
    lam, weights, grid_metadata = envelope_training_grid(n)
    optimizer = torch.optim.LBFGS(
        [model.raw_alpha, model.raw_cubic],
        lr=0.4,
        max_iter=args.envelope_lbfgs_max_iter,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    start = time.perf_counter()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        energy, *_ = adjoint_quadrature_energy(
            model,
            lam,
            weights,
            omega=args.omega,
            coupling=args.coupling,
        )
        energy.backward()
        return energy

    optimizer.step(closure)
    wall_time = time.perf_counter() - start
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    moments = adjoint_quadrature_moments(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    history = [
        {
            "step": args.envelope_lbfgs_max_iter,
            "energy": obs.energy,
            "radial": obs.radial,
            "angular": obs.angular,
            "potential": obs.potential,
            "local_energy_std": obs.local_energy_std,
            "virial_residual": moments.virial_residual,
            "alpha": obs.alpha,
            "cubic": obs.cubic,
            "coordinate_scale": obs.coordinate_scale,
        }
    ]
    training_payload = {
        "method": (
            "deterministic LBFGS warm start of the alpha/cubic envelope "
            "inside the full shared Chebyshev ansatz"
        ),
        "grid": grid_metadata,
        "wall_time_seconds": wall_time,
    }
    return summarize_candidate(
        "envelope_quadrature_lbfgs",
        model,
        args,
        benchmark=benchmark,
        training_payload=training_payload,
        history=history,
    )


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest-energy candidate among those passing all gates."""

    passing = [
        candidate
        for candidate in candidates
        if candidate["passes_validation_gates"]
    ]
    if passing:
        return min(
            passing,
            key=lambda candidate: candidate["combined_evaluation"]["energy_mean"],
        )
    return min(
        candidates,
        key=lambda candidate: candidate["combined_evaluation"][
            "virial_residual_abs_max"
        ],
    )


def train_one_n(n: int, args: argparse.Namespace) -> dict[str, Any]:
    if n < 2:
        raise ValueError("all N values must be at least two")
    torch.manual_seed(args.seed + n)
    reference = radial_reference_extrapolation(
        n,
        omega=args.omega,
        coupling=args.coupling,
    )
    benchmark = None if reference is None else reference["extrapolated_energy"]
    candidates = [
        train_importance_adam_candidate(n, args, benchmark=benchmark),
    ]
    if args.training_mode == "validated-candidates":
        candidates.append(
            train_envelope_quadrature_candidate(n, args, benchmark=benchmark)
        )
    selected = select_candidate(candidates)
    combined = selected["combined_evaluation"]
    return {
        "reference": reference,
        "training_samples": selected["training"],
        "training_wall_time_seconds": selected["training"]["wall_time_seconds"],
        "selected_candidate": selected["name"],
        "selected_candidate_passes_validation_gates": selected[
            "passes_validation_gates"
        ],
        "candidate_summaries": {
            candidate["name"]: {
                "passes_validation_gates": candidate["passes_validation_gates"],
                "combined_evaluation": candidate["combined_evaluation"],
                "training": candidate["training"],
            }
            for candidate in candidates
        },
        "candidates": candidates,
        "history": selected["history"],
        "evaluations": selected["evaluations"],
        "combined_evaluation": combined,
        "structure_checks": selected["structure_checks"],
    }


def validation_payload(results: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for key, payload in results["runs"].items():
        combined = payload["combined_evaluation"]
        structure = payload["structure_checks"]
        checks[f"{key}_ess_fraction_gt_min"] = (
            combined["min_effective_sample_size_fraction"]
            > results["metadata"]["min_ess_fraction"]
        )
        checks[f"{key}_structure_exact"] = (
            structure["traceless_residual"] < 1.0e-10
            and structure["odd_parity_residual"] < 1.0e-10
            and structure["weyl_covariance_residual"] < 1.0e-10
        )
        checks[f"{key}_virial_abs_lt_tol"] = (
            combined["virial_residual_abs_max"]
            < results["metadata"]["virial_abs_tol"]
        )
        if "benchmark_energy" in combined:
            checks[f"{key}_benchmark_energy_error_lt_tol"] = (
                abs(combined["energy_error"])
                < results["metadata"]["energy_abs_tol"]
            )
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "note": (
            "The SU(2)/SU(3) radial finite-difference energies are external "
            "benchmarks only.  Training and evaluation use the same flexible "
            "Chebyshev ansatz and Sobol-Gaussian importance Monte Carlo "
            "estimator for every N."
        ),
    }


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    invalid = [n for n in args.n_values if n < 2]
    if invalid:
        raise ValueError(f"invalid N values: {invalid}")

    results: dict[str, Any] = {
        "metadata": {
            "ansatz": "shared flexible SU(N) Chebyshev spectral impurity",
            "hamiltonian": "-1/2 Delta_X + 1/2 omega^2 Tr X^2 + g Tr X^4",
            "omega": args.omega,
            "coupling": args.coupling,
            "seed": args.seed,
            "n_values": args.n_values,
            "training_mode": args.training_mode,
            "energy_abs_tol": args.energy_abs_tol,
            "virial_abs_tol": args.virial_abs_tol,
            "min_ess_fraction": args.min_ess_fraction,
            "shared_ansatz_config": shared_ansatz_config(args),
            "harmonic_exact_energies": {
                str(n): exact_suN_harmonic_adjoint_energy(n, args.omega)
                for n in args.n_values
            },
            "sampler": {
                "algorithm": "scrambled Sobol Gaussian importance sampling in orthonormal traceless eigenvalue coordinates",
                "n_train_samples": args.n_samples,
                "n_eval_samples": args.eval_samples,
                "n_eval_replicates": args.n_eval_replicates,
                "proposal_sigma": args.proposal_sigma,
                "independent_evaluation_samples": True,
            },
        },
        "runs": {},
    }
    for n in args.n_values:
        print(f"training SU({n}) with shared flexible ansatz")
        results["runs"][f"su{n}"] = train_one_n(n, args)

    results["validation_summary"] = validation_payload(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(results["validation_summary"], indent=2))


if __name__ == "__main__":
    main()
