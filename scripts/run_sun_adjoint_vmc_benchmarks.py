#!/usr/bin/env python
"""Run shared-ansatz SU(N) adjoint Monte Carlo benchmarks."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import platform
from pathlib import Path
import subprocess
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
    SUNAdjointLinearImpurityAnsatz,
    adjoint_eigenvalue_grid,
    adjoint_importance_energy,
    adjoint_importance_moments,
    adjoint_importance_observables,
    adjoint_profile_norm,
    adjoint_quadrature_energy,
    adjoint_quadrature_linear_impurity_eigenproblem,
    adjoint_quadrature_moments,
    adjoint_quadrature_observables,
    adjoint_structure_diagnostics,
    exact_suN_harmonic_adjoint_energy,
    initialize_full_chebyshev_head_from_linear_impurity,
    log_vandermonde_abs,
    sobol_gaussian_traceless_samples,
    su3_adjoint_polar_eigenvalue_grid,
    suN_adjoint_radial_finite_difference_energy,
    train_adjoint_importance,
)


@dataclass
class CandidateResult:
    payload: dict[str, Any]
    model: torch.nn.Module


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
    parser.add_argument(
        "--validated-candidates",
        action="store_const",
        dest="training_mode",
        const="validated-candidates",
        help="Alias for --training-mode validated-candidates.",
    )
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument(
        "--refresh-training-samples-every",
        type=int,
        default=0,
        help=(
            "Use a newly scrambled Sobol-Gaussian training sample set every K "
            "Adam steps. The default 0 reuses one fixed training sample set. "
            "Use 1 for independent randomized-QMC sample sets at every step."
        ),
    )
    parser.add_argument("--eval-samples", type=int, default=131072)
    parser.add_argument("--n-eval-replicates", type=int, default=3)
    parser.add_argument("--proposal-sigma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help=(
            "Print the cheap full-flexible Adam loss every this many steps; "
            "use 0 to disable progress printing. This does not control the "
            "saved full-diagnostic cadence."
        ),
    )
    parser.add_argument(
        "--per-step-error-samples",
        type=int,
        default=0,
        help=(
            "If positive, evaluate independent scrambled Sobol-Gaussian "
            "energy estimates with this many points per replicate for the "
            "importance-adam loss history."
        ),
    )
    parser.add_argument(
        "--per-step-error-replicates",
        type=int,
        default=0,
        help=(
            "Number of independent per-step error replicates. Requires "
            "--per-step-error-samples > 0."
        ),
    )
    parser.add_argument(
        "--per-step-error-every",
        type=int,
        default=1,
        help=(
            "Cadence for per-step independent error estimates when enabled. "
            "Use 1 to store/print an error bar at every Adam step."
        ),
    )
    parser.add_argument(
        "--refresh-per-step-error-samples",
        action="store_true",
        help=(
            "Use new independent scrambled Sobol-Gaussian sample sets for each "
            "per-step error estimate instead of reusing fixed diagnostic "
            "sample sets."
        ),
    )
    parser.add_argument("--envelope-lbfgs-max-iter", type=int, default=80)
    parser.add_argument("--include-linear-impurity-candidate", action="store_true")
    parser.add_argument("--include-linear-impurity-ladder", action="store_true")
    parser.add_argument(
        "--include-linear-initialized-neural-candidate",
        action="store_true",
    )
    parser.add_argument(
        "--linear-impurity-terms",
        nargs="+",
        default=["1", "rho", "u", "v", "rho2", "rho_u", "rho_v"],
    )
    parser.add_argument("--linear-impurity-overlap-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-initialized-neural-steps", type=int, default=40)
    parser.add_argument("--linear-initialized-neural-lr", type=float, default=2.0e-4)
    parser.add_argument(
        "--linear-initialized-neural-head-regularization",
        type=float,
        default=1.0e-3,
        help=(
            "Penalty on deviation of the neural impurity head from the exact "
            "linear-impurity initialization during optional neural refinement."
        ),
    )
    parser.add_argument(
        "--linear-initialized-neural-action-regularization",
        type=float,
        default=1.0e-4,
        help=(
            "Penalty on deviation of the scalar envelope action from the "
            "linear-impurity initialization during optional neural refinement."
        ),
    )
    parser.add_argument(
        "--linear-initialized-neural-regularization-samples",
        type=int,
        default=4096,
        help="Fixed Sobol sample count used for the linear-solution regularizer.",
    )
    parser.add_argument("--include-multicloud-candidate", action="store_true")
    parser.add_argument("--multicloud-steps", type=int, default=120)
    parser.add_argument("--multicloud-clouds-per-step", type=int, default=2)
    parser.add_argument("--multicloud-lr", type=float, default=5.0e-4)
    parser.add_argument(
        "--feature-mode",
        choices=["raw_moments", "shape", "shape_quadratic"],
        default="shape_quadratic",
    )
    parser.add_argument("--feature-scale-init", type=float, default=1.0)
    parser.add_argument("--learn-feature-scale", action="store_true")
    parser.add_argument("--parity", choices=["odd", "even"], default="odd")
    parser.add_argument("--chebyshev-degrees", type=int, nargs="+", default=None)
    parser.add_argument("--learn-coordinate-scale", action="store_true")
    parser.add_argument(
        "--action-quadratic-mode",
        choices=["explicit", "mlp"],
        default="explicit",
        help=(
            "Use 'explicit' for the standard positive alpha * Tr X^2 envelope "
            "term, or 'mlp' to remove that explicit term and append Tr X^2 to "
            "the scalar-action MLP inputs.  The 'mlp' mode is intended for "
            "NN-only importance-adam experiments."
        ),
    )
    parser.add_argument("--action-correction-scale", type=float, default=0.25)
    parser.add_argument("--head-correction-scale", type=float, default=0.25)
    parser.add_argument(
        "--head-coefficient-mode",
        choices=["full", "anchored"],
        default="full",
        help=(
            "Use 'full' for the note's c_k(m) Chebyshev coefficient network, "
            "or 'anchored' to keep the leading head coefficient fixed."
        ),
    )
    parser.add_argument("--energy-abs-tol", type=float, default=5.0e-4)
    parser.add_argument("--energy-spread-abs-tol", type=float, default=2.0e-3)
    parser.add_argument("--virial-abs-tol", type=float, default=5.0e-3)
    parser.add_argument("--min-ess-fraction", type=float, default=0.05)
    parser.add_argument("--include-hf-diagnostic", action="store_true")
    parser.add_argument("--hf-delta", type=float, default=1.0e-3)
    return parser.parse_args()


def quartic_tail_initialization(coupling: float) -> float:
    """Ray-WKB coefficient for ``S ~ c sqrt(p2) sqrt(p4)``."""

    return 2.0 * math.sqrt(2.0 * coupling) / 3.0


def git_commit_hash() -> str | None:
    """Return the current git commit hash when the repository is available."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_status_short() -> str | None:
    """Return short git status for reproducibility metadata."""

    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return parsed CLI arguments in JSON-serializable form."""

    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def shared_ansatz_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the identical flexible ansatz settings used for every N."""

    chebyshev_degrees = args.chebyshev_degrees
    if chebyshev_degrees is None:
        chebyshev_degrees = [1, 3] if args.parity == "odd" else [2, 4]
    return {
        "omega_init": args.omega,
        "quartic_tail_init": quartic_tail_initialization(args.coupling),
        "moment_cutoff": 6,
        "feature_mode": args.feature_mode,
        "feature_scale_init": args.feature_scale_init,
        "learn_feature_scale": args.learn_feature_scale,
        "hidden_layers": [24, 24],
        "head_hidden_layers": [24],
        "chebyshev_degrees": chebyshev_degrees,
        "parity": args.parity,
        "scale_init": 3.0,
        "learn_scale": False,
        "coordinate_scale_init": 1.0,
        "learn_coordinate_scale": args.learn_coordinate_scale,
        "action_quadratic_mode": getattr(args, "action_quadratic_mode", "explicit"),
        "action_correction_scale": args.action_correction_scale,
        "head_correction_scale": args.head_correction_scale,
        "head_coefficient_mode": args.head_coefficient_mode,
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
        feature_mode=config["feature_mode"],
        feature_scale_init=config["feature_scale_init"],
        learn_feature_scale=config["learn_feature_scale"],
        hidden_layers=tuple(config["hidden_layers"]),
        head_hidden_layers=tuple(config["head_hidden_layers"]),
        chebyshev_degrees=tuple(config["chebyshev_degrees"]),
        parity=config["parity"],
        scale_init=config["scale_init"],
        learn_scale=config["learn_scale"],
        coordinate_scale_init=config["coordinate_scale_init"],
        learn_coordinate_scale=config["learn_coordinate_scale"],
        action_quadratic_mode=config["action_quadratic_mode"],
        action_correction_scale=config["action_correction_scale"],
        head_correction_scale=config["head_correction_scale"],
        head_coefficient_mode=config["head_coefficient_mode"],
        dtype=torch.float64,
    )


def make_linear_initialized_neural_model(
    n: int,
    args: argparse.Namespace,
) -> SUNAdjointChebyshevSpectralAnsatz:
    """Return the neural ansatz shape needed to contain the linear impurity."""

    config = shared_ansatz_config(args)
    config["feature_mode"] = "shape_quadratic"
    config["head_hidden_layers"] = []
    config["head_coefficient_mode"] = "full"
    return SUNAdjointChebyshevSpectralAnsatz(
        n=n,
        omega_init=config["omega_init"],
        quartic_tail_init=config["quartic_tail_init"],
        moment_cutoff=config["moment_cutoff"],
        feature_mode=config["feature_mode"],
        feature_scale_init=config["feature_scale_init"],
        learn_feature_scale=config["learn_feature_scale"],
        hidden_layers=tuple(config["hidden_layers"]),
        head_hidden_layers=tuple(config["head_hidden_layers"]),
        chebyshev_degrees=tuple(config["chebyshev_degrees"]),
        parity=config["parity"],
        scale_init=config["scale_init"],
        learn_scale=config["learn_scale"],
        coordinate_scale_init=config["coordinate_scale_init"],
        learn_coordinate_scale=config["learn_coordinate_scale"],
        action_quadratic_mode=config["action_quadratic_mode"],
        action_correction_scale=config["action_correction_scale"],
        head_correction_scale=config["head_correction_scale"],
        head_coefficient_mode=config["head_coefficient_mode"],
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
        "parity": getattr(model, "parity", "odd"),
        "traceless_residual": diagnostics.traceless_residual,
        "parity_residual": diagnostics.parity_residual,
        "odd_parity_residual": diagnostics.parity_residual,
        "weyl_covariance_residual": diagnostics.weyl_residual,
        "head_collision_ratio_max_abs": diagnostics.head_collision_ratio_max_abs,
        "profile_collision_identity_residual": (
            diagnostics.profile_collision_identity_residual
        ),
    }


def _normalized_slice_direction(values: list[float]) -> torch.Tensor:
    direction = torch.tensor(values, dtype=torch.float64)
    direction = direction - torch.mean(direction)
    norm = torch.linalg.vector_norm(direction)
    if float(norm) <= 0.0:
        raise ValueError("slice direction must be nonzero after centering")
    return direction / norm


def profile_slice_payload(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Return normalized SU(4) profile slices for notebook inspection.

    The full SU(4) profile depends on three independent eigenvalue coordinates.
    This diagnostic records one-dimensional slices through the traceless
    eigenvalue plane; it is not a full wavefunction representation.
    """

    if getattr(model, "n", None) != 4:
        return None
    norm_lam, norm_weights, norm_grid = envelope_training_grid(4)
    with torch.no_grad():
        norm = adjoint_profile_norm(model, norm_lam, norm_weights)
        scale = 1.0 / torch.sqrt(norm)
        radii = torch.linspace(-4.0, 4.0, 401, dtype=torch.float64)
        directions = {
            "pair split": _normalized_slice_direction([1.0, -1.0, 0.0, 0.0]),
            "two-two split": _normalized_slice_direction([1.0, 1.0, -1.0, -1.0]),
            "one-three split": _normalized_slice_direction([3.0, -1.0, -1.0, -1.0]),
        }
        slices: dict[str, Any] = {}
        for label, direction in directions.items():
            lam = radii[:, None] * direction[None, :]
            action = model.action(lam)
            center_index = int(torch.argmin(torch.abs(radii)).detach())
            action_shifted = action - action[center_index]
            raw_profile = model.profile(lam)
            normalized_profile = scale * raw_profile
            slices[label] = {
                "direction": direction.tolist(),
                "r": radii.tolist(),
                "action": action.tolist(),
                "action_shifted": action_shifted.tolist(),
                "q_raw": raw_profile.tolist(),
                "q_normalized": normalized_profile.tolist(),
            }
    return {
        "description": (
            "One-dimensional slices of q_i(lambda) through SU(4) traceless "
            "eigenvalue space. q_normalized is divided by the deterministic "
            "grid adjoint-profile norm for this candidate. action is the "
            "scalar S_theta(lambda) in exp[-S_theta/2]; action_shifted "
            "subtracts S_theta at the slice point closest to r=0 so scalar "
            "envelope shapes can be compared modulo irrelevant additive "
            "constants."
        ),
        "normalization_grid": norm_grid,
        "profile_norm": float(norm.detach()),
        "slices": slices,
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
    mean_fields = {
        f"{field}_mean": sum(item[field] for item in evaluations) / len(evaluations)
        for field in [
            "local_energy_std",
            "radial",
            "angular",
            "potential",
            "tr_x2",
            "tr_x4",
            "kinetic",
            "virial_rhs",
        ]
    }
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
        **mean_fields,
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
        and structure["parity_residual"] < 1.0e-10
        and structure["weyl_covariance_residual"] < 1.0e-10
    ):
        return False
    if not math.isfinite(structure["head_collision_ratio_max_abs"]):
        return False
    if not math.isfinite(structure["profile_collision_identity_residual"]):
        return False
    if structure["head_collision_ratio_max_abs"] >= 1.0e8:
        return False
    if structure["profile_collision_identity_residual"] >= 1.0e-8:
        return False
    if combined["energy_spread_independent_runs"] >= args.energy_spread_abs_tol:
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
    profile_slices = profile_slice_payload(model, args)
    if profile_slices is not None:
        candidate["profile_slices"] = profile_slices
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
    training_batch_factory = None
    latest_training_batch: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "batch": (train_samples.lam, train_samples.log_prob)
    }
    latest_training_seed = {"seed": train_samples.seed}
    if args.refresh_training_samples_every > 0:
        training_cache: dict[str, Any] = {}

        def training_batch_factory(
            step: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            refresh_index = (step - 1) // args.refresh_training_samples_every
            if training_cache.get("refresh_index") != refresh_index:
                seed = args.seed + 10_000 + 100_000 * n + refresh_index
                refreshed = sobol_gaussian_traceless_samples(
                    n,
                    args.n_samples,
                    sigma=args.proposal_sigma,
                    seed=seed,
                    dtype=torch.float64,
                )
                training_cache["refresh_index"] = refresh_index
                training_cache["batch"] = (refreshed.lam, refreshed.log_prob)
                training_cache["seed"] = refreshed.seed
            latest_training_batch["batch"] = training_cache["batch"]
            latest_training_seed["seed"] = training_cache["seed"]
            return training_cache["batch"]

    error_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    error_batch_factory = None
    if args.per_step_error_samples > 0 and args.per_step_error_replicates > 0:
        if args.refresh_per_step_error_samples:

            def error_batch_factory(
                step: int,
            ) -> list[tuple[torch.Tensor, torch.Tensor]]:
                batches = []
                for replicate in range(args.per_step_error_replicates):
                    error_samples = sobol_gaussian_traceless_samples(
                        n,
                        args.per_step_error_samples,
                        sigma=args.proposal_sigma,
                        seed=(
                            args.seed
                            + 20_000
                            + 1_000_000 * n
                            + 10_000 * step
                            + replicate
                        ),
                        dtype=torch.float64,
                    )
                    batches.append((error_samples.lam, error_samples.log_prob))
                return batches

        else:
            error_batches = []
            for replicate in range(args.per_step_error_replicates):
                error_samples = sobol_gaussian_traceless_samples(
                    n,
                    args.per_step_error_samples,
                    sigma=args.proposal_sigma,
                    seed=args.seed + 20_000 + 1_000 * n + replicate,
                    dtype=torch.float64,
                )
                error_batches.append((error_samples.lam, error_samples.log_prob))
    start = time.perf_counter()
    history, loss_history = train_adjoint_importance(
        model,
        train_samples.lam,
        train_samples.log_prob,
        omega=args.omega,
        coupling=args.coupling,
        n_steps=args.n_steps,
        lr=args.lr,
        report_every=args.report_every,
        print_every=None if args.print_every == 0 else args.print_every,
        error_batches=error_batches,
        training_batch_factory=training_batch_factory,
        error_batch_factory=error_batch_factory,
        error_every=args.per_step_error_every,
    )
    wall_time = time.perf_counter() - start
    ess_lam, ess_log_prob = latest_training_batch["batch"]
    training_payload = {
        "method": (
            "refreshed scrambled Sobol-Gaussian Adam on the full flexible ansatz"
            if args.refresh_training_samples_every > 0
            else "fixed-proposal Adam on the full flexible ansatz"
        ),
        "sample_count": args.n_samples,
        "seed": train_samples.seed,
        "latest_training_seed": latest_training_seed["seed"],
        "refresh_training_samples_every": args.refresh_training_samples_every,
        "proposal_sigma": train_samples.sigma,
        "effective_sample_size": effective_sample_size(
            model,
            ess_lam,
            ess_log_prob,
        ),
        "loss_history": [record.__dict__ for record in loss_history],
        "loss_history_description": (
            "Per-step fixed-proposal Rayleigh quotient evaluated before the "
            "Adam update.  Full observable diagnostics are stored separately "
            "in the candidate history on report steps, including the virial "
            "residual and its Tr X^2, Tr X^4, and kinetic ingredients.  When "
            "enabled, error_energy_* fields are independent scrambled "
            "Sobol-Gaussian energy estimates on separate proposal sample sets; "
            "they are diagnostics and are not used as the optimizer loss."
        ),
        "per_step_error_samples": args.per_step_error_samples,
        "per_step_error_replicates": args.per_step_error_replicates,
        "per_step_error_every": args.per_step_error_every,
        "refresh_per_step_error_samples": args.refresh_per_step_error_samples,
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


def fit_envelope_lbfgs(
    model: SUNAdjointChebyshevSpectralAnsatz,
    n: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Optimize only the scalar envelope parameters of a shared model."""

    trainable_names = {"raw_alpha", "raw_cubic"}
    if args.learn_coordinate_scale:
        trainable_names.add("raw_coordinate_scale")
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_(parameter_name in trainable_names)
    lam, weights, grid_metadata = envelope_training_grid(n)
    optimizer = torch.optim.LBFGS(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.4,
        max_iter=args.envelope_lbfgs_max_iter,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    start = time.perf_counter()
    history: list[dict[str, Any]] = []
    closure_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        energy, *_ = adjoint_quadrature_energy(
            model,
            lam,
            weights,
            omega=args.omega,
            coupling=args.coupling,
        )
        energy.backward()
        closure_evaluations += 1
        history.append(
            {
                "step": closure_evaluations,
                "energy": float(energy.detach()),
                "alpha": float(model.alpha.detach()),
                "cubic": float(model.cubic.detach()),
                "coordinate_scale": float(model.coordinate_scale.detach()),
                "stage": "envelope_lbfgs_closure",
            }
        )
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
    final_record = {
        "step": closure_evaluations + 1,
        "energy": obs.energy,
        "radial": obs.radial,
        "angular": obs.angular,
        "potential": obs.potential,
        "local_energy_std": obs.local_energy_std,
        "virial_residual": moments.virial_residual,
        "alpha": obs.alpha,
        "cubic": obs.cubic,
        "coordinate_scale": obs.coordinate_scale,
        "stage": "envelope_lbfgs_final_observables",
    }
    history.append(final_record)
    payload = {
        "grid": grid_metadata,
        "wall_time_seconds": wall_time,
        "lbfgs_closure_evaluations": closure_evaluations,
    }
    return payload, history


def train_envelope_quadrature_candidate(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
) -> CandidateResult:
    """Train the stable envelope subspace inside the shared flexible model."""

    model = make_shared_model(n, args)
    envelope_payload, history = fit_envelope_lbfgs(model, n, args)
    training_payload = {
        "method": (
            "deterministic LBFGS warm start of the alpha/cubic envelope "
            "inside the full shared Chebyshev ansatz"
        ),
        **envelope_payload,
    }
    return CandidateResult(
        payload=summarize_candidate(
            "envelope_quadrature_lbfgs",
            model,
            args,
            benchmark=benchmark,
            training_payload=training_payload,
            history=history,
        ),
        model=model,
    )


def set_full_trainability(
    model: SUNAdjointChebyshevSpectralAnsatz,
    args: argparse.Namespace,
) -> None:
    """Restore trainability for the flexible parameters enabled by CLI flags."""

    for parameter_name, parameter in model.named_parameters():
        if parameter_name == "raw_alpha" and model.action_quadratic_mode == "mlp":
            parameter.requires_grad_(False)
        elif parameter_name == "raw_scale":
            parameter.requires_grad_(False)
        elif parameter_name == "raw_feature_scale":
            parameter.requires_grad_(args.learn_feature_scale)
        elif parameter_name == "raw_coordinate_scale":
            parameter.requires_grad_(args.learn_coordinate_scale)
        else:
            parameter.requires_grad_(True)


def train_multicloud_candidate(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
) -> dict[str, Any]:
    """Train flexible corrections after envelope warm start on refreshed clouds."""

    model = make_shared_model(n, args)
    envelope_payload, _ = fit_envelope_lbfgs(model, n, args)
    set_full_trainability(model, args)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.multicloud_lr)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for step in range(1, args.multicloud_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for cloud in range(args.multicloud_clouds_per_step):
            samples = sobol_gaussian_traceless_samples(
                n,
                args.n_samples,
                sigma=args.proposal_sigma,
                seed=args.seed + 1_000_000 * n + 10_000 * step + cloud,
                dtype=torch.float64,
            )
            energy, *_ = adjoint_importance_energy(
                model,
                samples.lam,
                samples.log_prob,
                omega=args.omega,
                coupling=args.coupling,
            )
            losses.append(energy)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
        optimizer.step()

        if (
            step == 1
            or step % args.report_every == 0
            or step == args.multicloud_steps
        ):
            eval_samples = sobol_gaussian_traceless_samples(
                n,
                args.n_samples,
                sigma=args.proposal_sigma,
                seed=args.seed + 2_000_000 * n + step,
                dtype=torch.float64,
            )
            obs = adjoint_importance_observables(
                model,
                eval_samples.lam,
                eval_samples.log_prob,
                omega=args.omega,
                coupling=args.coupling,
            )
            moments = adjoint_importance_moments(
                model,
                eval_samples.lam,
                eval_samples.log_prob,
                omega=args.omega,
                coupling=args.coupling,
            )
            history.append(
                {
                    "step": step,
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
            )

    wall_time = time.perf_counter() - start
    training_payload = {
        "method": (
            "envelope LBFGS warm start followed by flexible Adam on refreshed "
            "independent Sobol-Gaussian clouds"
        ),
        "warm_start": envelope_payload,
        "n_steps": args.multicloud_steps,
        "sample_count_per_cloud": args.n_samples,
        "clouds_per_step": args.multicloud_clouds_per_step,
        "proposal_sigma": args.proposal_sigma,
        "lr": args.multicloud_lr,
        "wall_time_seconds": wall_time + envelope_payload["wall_time_seconds"],
    }
    return summarize_candidate(
        "multicloud_flexible_after_envelope",
        model,
        args,
        benchmark=benchmark,
        training_payload=training_payload,
        history=history,
    )


def linear_impurity_ladder_specs(parity: str = "odd") -> list[dict[str, Any]]:
    """Return nested default bases for linear-impurity convergence checks."""

    if parity == "odd":
        low_degree = 1
        high_degree = 3
    elif parity == "even":
        low_degree = 2
        high_degree = 4
    else:
        raise ValueError("parity must be 'odd' or 'even'")
    return [
        {
            "name": f"linear_ladder_{parity}_constant_t{low_degree}",
            "terms": ("1",),
            "chebyshev_degrees": (low_degree,),
        },
        {
            "name": f"linear_ladder_{parity}_shape_t{low_degree}",
            "terms": ("1", "rho", "u", "v"),
            "chebyshev_degrees": (low_degree,),
        },
        {
            "name": f"linear_ladder_{parity}_shape_t{low_degree}{high_degree}",
            "terms": ("1", "rho", "u", "v"),
            "chebyshev_degrees": (low_degree, high_degree),
        },
        {
            "name": (
                f"linear_ladder_{parity}_shape_quadratic_t"
                f"{low_degree}{high_degree}"
            ),
            "terms": ("1", "rho", "u", "v", "rho2", "rho_u", "rho_v"),
            "chebyshev_degrees": (low_degree, high_degree),
        },
    ]


def make_linear_impurity_candidate(
    name: str,
    model: SUNAdjointChebyshevSpectralAnsatz,
    lam: torch.Tensor,
    weights: torch.Tensor,
    grid_metadata: dict[str, Any],
    envelope_payload: dict[str, Any],
    envelope_history: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    benchmark: float | None,
    terms: tuple[str, ...],
    chebyshev_degrees: tuple[int, ...],
) -> dict[str, Any]:
    """Solve and summarize one fixed-envelope linear impurity basis."""

    linear_start = time.perf_counter()
    result = adjoint_quadrature_linear_impurity_eigenproblem(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        chebyshev_degrees=chebyshev_degrees,
        chebyshev_scale=model.scale.detach(),
        feature_scale=model.feature_scale.detach(),
        terms=terms,
        tail_eps=model.tail_eps,
        overlap_rtol=args.linear_impurity_overlap_rtol,
    )
    linear_wall_time = time.perf_counter() - linear_start
    linear_model = SUNAdjointLinearImpurityAnsatz(
        envelope_model=model,
        coefficients=result.coefficients,
        chebyshev_degrees=chebyshev_degrees,
        parity=args.parity,
        chebyshev_scale=model.scale.detach(),
        feature_scale=model.feature_scale.detach(),
        terms=terms,
        tail_eps=model.tail_eps,
    )
    obs = adjoint_quadrature_observables(
        linear_model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    moments = adjoint_quadrature_moments(
        linear_model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    linear_step = envelope_history[-1]["step"] + 1
    history = envelope_history + [
        {
            "step": linear_step,
            "energy": obs.energy,
            "radial": obs.radial,
            "angular": obs.angular,
            "potential": obs.potential,
            "local_energy_std": obs.local_energy_std,
            "virial_residual": moments.virial_residual,
            "alpha": obs.alpha,
            "cubic": obs.cubic,
            "coordinate_scale": obs.coordinate_scale,
            "linear_impurity_energy": result.energy,
            "stage": "linear_impurity_gep_endpoint",
        }
    ]
    coefficient_payload = [
        {
            "basis": label,
            "coefficient": float(coefficient),
        }
        for label, coefficient in zip(result.basis_labels, result.coefficients)
    ]
    training_payload = {
        "method": (
            "envelope LBFGS warm start followed by a linear impurity "
            "generalized eigenproblem"
        ),
        "warm_start": envelope_payload,
        "grid": grid_metadata,
        "linear_impurity_energy": result.energy,
        "retained_basis_count": result.retained_basis_count,
        "requested_basis_count": len(terms) * len(chebyshev_degrees),
        "basis_labels": list(result.basis_labels),
        "coefficients": coefficient_payload,
        "lowest_eigenvalues": [
            float(value)
            for value in result.eigenvalues[: min(6, result.eigenvalues.numel())]
        ],
        "overlap_eigenvalue_min": float(torch.min(result.overlap_eigenvalues)),
        "overlap_eigenvalue_max": float(torch.max(result.overlap_eigenvalues)),
        "overlap_rtol": args.linear_impurity_overlap_rtol,
        "linear_solve_wall_time_seconds": linear_wall_time,
        "wall_time_seconds": envelope_payload["wall_time_seconds"] + linear_wall_time,
    }
    return summarize_candidate(
        name,
        linear_model,
        args,
        benchmark=benchmark,
        training_payload=training_payload,
        history=history,
    )


def train_linear_impurity_candidate(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
    envelope_result: CandidateResult | None = None,
) -> dict[str, Any]:
    """Solve the configured fixed-envelope linear impurity basis."""

    if envelope_result is None:
        envelope_result = train_envelope_quadrature_candidate(
            n,
            args,
            benchmark=benchmark,
        )
    model = envelope_result.model
    envelope_payload = envelope_result.payload["training"]
    envelope_history = envelope_result.payload["history"]
    lam, weights, grid_metadata = envelope_training_grid(n)
    return make_linear_impurity_candidate(
        "linear_impurity_after_envelope",
        model,
        lam,
        weights,
        grid_metadata,
        envelope_payload,
        envelope_history,
        args,
        benchmark=benchmark,
        terms=tuple(args.linear_impurity_terms),
        chebyshev_degrees=tuple(model.chebyshev_degrees),
    )


def train_linear_impurity_ladder_candidates(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
    envelope_result: CandidateResult | None = None,
) -> list[dict[str, Any]]:
    """Run a nested linear impurity basis ladder from one envelope warm start."""

    if envelope_result is None:
        envelope_result = train_envelope_quadrature_candidate(
            n,
            args,
            benchmark=benchmark,
        )
    model = envelope_result.model
    envelope_payload = envelope_result.payload["training"]
    envelope_history = envelope_result.payload["history"]
    lam, weights, grid_metadata = envelope_training_grid(n)
    candidates: list[dict[str, Any]] = []
    previous_grid_energy: float | None = None
    for spec in linear_impurity_ladder_specs(args.parity):
        candidate = make_linear_impurity_candidate(
            spec["name"],
            model,
            lam,
            weights,
            grid_metadata,
            envelope_payload,
            envelope_history,
            args,
            benchmark=benchmark,
            terms=tuple(spec["terms"]),
            chebyshev_degrees=tuple(spec["chebyshev_degrees"]),
        )
        grid_energy = candidate["training"]["linear_impurity_energy"]
        candidate["training"]["basis_ladder_index"] = len(candidates)
        candidate["training"]["previous_ladder_grid_energy"] = previous_grid_energy
        candidate["training"]["grid_energy_improvement_from_previous"] = (
            None if previous_grid_energy is None else previous_grid_energy - grid_energy
        )
        previous_grid_energy = grid_energy
        candidates.append(candidate)
    return candidates


def neural_refinement_regularization_setup(
    model: SUNAdjointChebyshevSpectralAnsatz,
    n: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Return fixed reference tensors for linear-solution regularization."""

    head_strength = args.linear_initialized_neural_head_regularization
    action_strength = args.linear_initialized_neural_action_regularization
    if head_strength == 0.0 and action_strength == 0.0:
        return None
    samples = sobol_gaussian_traceless_samples(
        n,
        args.linear_initialized_neural_regularization_samples,
        sigma=args.proposal_sigma,
        seed=args.seed + 3_500_000 * n,
        dtype=torch.float64,
    )
    with torch.no_grad():
        reference_head = model.head(samples.lam).detach()
        reference_action = model.action(samples.lam).detach()
        head_norm = torch.mean(reference_head**2).clamp_min(1.0e-12)
        action_norm = torch.mean(reference_action**2).clamp_min(1.0e-12)
    return {
        "lam": samples.lam,
        "seed": samples.seed,
        "sample_count": samples.lam.shape[0],
        "proposal_sigma": samples.sigma,
        "reference_head": reference_head,
        "reference_action": reference_action,
        "head_norm": head_norm,
        "action_norm": action_norm,
    }


def neural_refinement_regularization_loss(
    model: SUNAdjointChebyshevSpectralAnsatz,
    reference: dict[str, Any] | None,
    args: argparse.Namespace,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, head, and action penalties against the linear solution."""

    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    if reference is None:
        return zero, zero, zero
    head_penalty = zero
    action_penalty = zero
    if args.linear_initialized_neural_head_regularization > 0.0:
        head_delta = model.head(reference["lam"]) - reference["reference_head"]
        head_penalty = (
            args.linear_initialized_neural_head_regularization
            * torch.mean(head_delta**2)
            / reference["head_norm"]
        )
    if args.linear_initialized_neural_action_regularization > 0.0:
        action_delta = model.action(reference["lam"]) - reference["reference_action"]
        action_penalty = (
            args.linear_initialized_neural_action_regularization
            * torch.mean(action_delta**2)
            / reference["action_norm"]
        )
    return head_penalty + action_penalty, head_penalty, action_penalty


def neural_refinement_checkpoint_payload(
    model: SUNAdjointChebyshevSpectralAnsatz,
    n: int,
    args: argparse.Namespace,
    *,
    step: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one independent checkpoint cloud for neural early stopping."""

    samples = sobol_gaussian_traceless_samples(
        n,
        args.n_samples,
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
    checkpoint_passed = (
        math.isfinite(obs.energy)
        and math.isfinite(moments.virial_residual)
        and ess / args.n_samples > args.min_ess_fraction
        and abs(moments.virial_residual) < args.virial_abs_tol
    )
    return {
        "step": step,
        "energy": obs.energy,
        "radial": obs.radial,
        "angular": obs.angular,
        "potential": obs.potential,
        "local_energy_std": obs.local_energy_std,
        "virial_residual": moments.virial_residual,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / args.n_samples,
        "sample_count": args.n_samples,
        "seed": samples.seed,
        "proposal_sigma": samples.sigma,
        "alpha": obs.alpha,
        "cubic": obs.cubic,
        "coordinate_scale": obs.coordinate_scale,
        "single_cloud_checkpoint_passed": checkpoint_passed,
    }


def train_linear_initialized_neural_candidate(
    n: int,
    args: argparse.Namespace,
    *,
    benchmark: float | None,
) -> dict[str, Any]:
    """Initialize a neural head from the linear impurity and optionally refine."""

    model = make_linear_initialized_neural_model(n, args)
    envelope_payload, envelope_history = fit_envelope_lbfgs(model, n, args)
    lam, weights, grid_metadata = envelope_training_grid(n)
    terms = tuple(args.linear_impurity_terms)
    chebyshev_degrees = tuple(model.chebyshev_degrees)
    linear_start = time.perf_counter()
    result = adjoint_quadrature_linear_impurity_eigenproblem(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
        chebyshev_degrees=chebyshev_degrees,
        chebyshev_scale=model.scale.detach(),
        feature_scale=model.feature_scale.detach(),
        terms=terms,
        tail_eps=model.tail_eps,
        overlap_rtol=args.linear_impurity_overlap_rtol,
    )
    init_divisor = initialize_full_chebyshev_head_from_linear_impurity(
        model,
        result.coefficients,
        terms=terms,
        chebyshev_degrees=chebyshev_degrees,
    )
    linear_wall_time = time.perf_counter() - linear_start
    initialized_obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    initialized_moments = adjoint_quadrature_moments(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )
    linear_initialization_step = envelope_history[-1]["step"] + 1
    history = envelope_history + [
        {
            "step": linear_initialization_step,
            "energy": initialized_obs.energy,
            "radial": initialized_obs.radial,
            "angular": initialized_obs.angular,
            "potential": initialized_obs.potential,
            "local_energy_std": initialized_obs.local_energy_std,
            "virial_residual": initialized_moments.virial_residual,
            "alpha": initialized_obs.alpha,
            "cubic": initialized_obs.cubic,
            "coordinate_scale": initialized_obs.coordinate_scale,
            "linear_impurity_energy": result.energy,
            "stage": "linear_neural_initialization",
        }
    ]

    initial_state = copy.deepcopy(model.state_dict())
    regularization_reference = neural_refinement_regularization_setup(model, n, args)
    initial_checkpoint = neural_refinement_checkpoint_payload(
        model,
        n,
        args,
        step=linear_initialization_step,
        seed=args.seed + 4_000_000 * n,
    )
    initial_checkpoint["stage"] = "linear_neural_initialization_checkpoint"
    checkpoint_history = [initial_checkpoint]
    best_checkpoint_state = copy.deepcopy(initial_state)
    best_checkpoint_payload = copy.deepcopy(initial_checkpoint)
    best_checkpoint_source = "linear_initialization"
    best_checkpoint_passed = initial_checkpoint["single_cloud_checkpoint_passed"]

    set_full_trainability(model, args)
    train_start = time.perf_counter()
    if args.linear_initialized_neural_steps > 0:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=args.linear_initialized_neural_lr,
        )
        for step in range(1, args.linear_initialized_neural_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for cloud in range(args.multicloud_clouds_per_step):
                samples = sobol_gaussian_traceless_samples(
                    n,
                    args.n_samples,
                    sigma=args.proposal_sigma,
                    seed=args.seed + 3_000_000 * n + 10_000 * step + cloud,
                    dtype=torch.float64,
                )
                energy, *_ = adjoint_importance_energy(
                    model,
                    samples.lam,
                    samples.log_prob,
                    omega=args.omega,
                    coupling=args.coupling,
                )
                losses.append(energy)
            loss = torch.stack(losses).mean()
            regularization_loss, head_regularization, action_regularization = (
                neural_refinement_regularization_loss(
                    model,
                    regularization_reference,
                    args,
                    loss,
                )
            )
            total_loss = loss + regularization_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
            optimizer.step()

            if (
                step == 1
                or step % args.report_every == 0
                or step == args.linear_initialized_neural_steps
            ):
                checkpoint = neural_refinement_checkpoint_payload(
                    model,
                    n,
                    args,
                    step=linear_initialization_step + step,
                    seed=args.seed + 4_000_000 * n + step,
                )
                checkpoint["stage"] = "multicloud_neural_refinement_checkpoint"
                checkpoint_history.append(checkpoint)
                is_improving_checkpoint = (
                    checkpoint["single_cloud_checkpoint_passed"]
                    and (
                        not best_checkpoint_passed
                        or checkpoint["energy"] < best_checkpoint_payload["energy"]
                    )
                )
                if is_improving_checkpoint:
                    best_checkpoint_state = copy.deepcopy(model.state_dict())
                    best_checkpoint_payload = copy.deepcopy(checkpoint)
                    best_checkpoint_source = f"training_step_{step}"
                    best_checkpoint_passed = True
                history.append(
                    {
                        "step": linear_initialization_step + step,
                        "training_energy_loss": float(loss.detach()),
                        "regularization_loss": float(regularization_loss.detach()),
                        "head_regularization_loss": float(
                            head_regularization.detach()
                        ),
                        "action_regularization_loss": float(
                            action_regularization.detach()
                        ),
                        "total_loss": float(total_loss.detach()),
                        "energy": checkpoint["energy"],
                        "radial": checkpoint["radial"],
                        "angular": checkpoint["angular"],
                        "potential": checkpoint["potential"],
                        "local_energy_std": checkpoint["local_energy_std"],
                        "virial_residual": checkpoint["virial_residual"],
                        "effective_sample_size_fraction": checkpoint[
                            "effective_sample_size_fraction"
                        ],
                        "single_cloud_checkpoint_passed": checkpoint[
                            "single_cloud_checkpoint_passed"
                        ],
                        "alpha": checkpoint["alpha"],
                        "cubic": checkpoint["cubic"],
                        "coordinate_scale": checkpoint["coordinate_scale"],
                        "stage": "multicloud_neural_refinement",
                    }
                )
    train_wall_time = time.perf_counter() - train_start
    if best_checkpoint_passed:
        model.load_state_dict(best_checkpoint_state)
    else:
        model.load_state_dict(initial_state)
        best_checkpoint_source = "linear_initialization_fallback"
        best_checkpoint_payload = copy.deepcopy(initial_checkpoint)

    training_payload = {
        "method": (
            "envelope LBFGS plus linear impurity generalized eigenproblem, "
            "mapped exactly into a full Chebyshev neural coefficient head; "
            "optional refinement is regularized to the linear solution and "
            "restored by independent single-cloud checkpoint gates"
        ),
        "warm_start": envelope_payload,
        "grid": grid_metadata,
        "linear_impurity_energy": result.energy,
        "initialized_quadrature_energy": initialized_obs.energy,
        "initialized_virial_residual": initialized_moments.virial_residual,
        "head_initialization_scale_divisor": init_divisor,
        "retained_basis_count": result.retained_basis_count,
        "requested_basis_count": len(terms) * len(chebyshev_degrees),
        "basis_labels": list(result.basis_labels),
        "linear_solve_and_init_wall_time_seconds": linear_wall_time,
        "neural_refinement_steps": args.linear_initialized_neural_steps,
        "neural_refinement_lr": args.linear_initialized_neural_lr,
        "neural_refinement_head_regularization": (
            args.linear_initialized_neural_head_regularization
        ),
        "neural_refinement_action_regularization": (
            args.linear_initialized_neural_action_regularization
        ),
        "neural_refinement_regularization_sample_count": (
            0
            if regularization_reference is None
            else regularization_reference["sample_count"]
        ),
        "neural_refinement_regularization_seed": (
            None if regularization_reference is None else regularization_reference["seed"]
        ),
        "neural_refinement_checkpoint_history": checkpoint_history,
        "neural_refinement_restored_checkpoint_source": best_checkpoint_source,
        "neural_refinement_restored_checkpoint_passed_single_cloud_gates": (
            best_checkpoint_passed
        ),
        "neural_refinement_restored_checkpoint": best_checkpoint_payload,
        "sample_count_per_cloud": args.n_samples,
        "clouds_per_step": args.multicloud_clouds_per_step,
        "proposal_sigma": args.proposal_sigma,
        "wall_time_seconds": (
            envelope_payload["wall_time_seconds"]
            + linear_wall_time
            + train_wall_time
        ),
    }
    return summarize_candidate(
        "linear_initialized_neural_full_head",
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


def linear_ladder_summary(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize linear-impurity ladder monotonicity and validation results."""

    ladder = [
        candidate
        for candidate in candidates
        if candidate["name"].startswith("linear_ladder_")
    ]
    if not ladder:
        return None
    grid_energies = [
        candidate["training"]["linear_impurity_energy"]
        for candidate in ladder
    ]
    independent_energies = [
        candidate["combined_evaluation"]["energy_mean"]
        for candidate in ladder
    ]
    virial_abs_max = [
        candidate["combined_evaluation"]["virial_residual_abs_max"]
        for candidate in ladder
    ]
    energy_spreads = [
        candidate["combined_evaluation"]["energy_spread_independent_runs"]
        for candidate in ladder
    ]
    passing = [
        candidate["name"]
        for candidate in ladder
        if candidate["passes_validation_gates"]
    ]
    return {
        "rung_names": [candidate["name"] for candidate in ladder],
        "grid_energies": grid_energies,
        "grid_energy_monotone_nonincreasing": all(
            later <= earlier + 1.0e-10
            for earlier, later in zip(grid_energies, grid_energies[1:])
        ),
        "independent_energy_means": independent_energies,
        "independent_virial_abs_max": virial_abs_max,
        "independent_energy_spreads": energy_spreads,
        "passes_validation_gates": [
            candidate["passes_validation_gates"]
            for candidate in ladder
        ],
        "passing_rungs": passing,
        "lowest_passing_rung": (
            min(
                (
                    candidate
                    for candidate in ladder
                    if candidate["passes_validation_gates"]
                ),
                key=lambda candidate: candidate["combined_evaluation"]["energy_mean"],
            )["name"]
            if passing
            else None
        ),
    }


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
        envelope_result = (
            train_envelope_quadrature_candidate(n, args, benchmark=benchmark)
        )
        candidates.append(envelope_result.payload)
        if args.include_linear_impurity_candidate:
            candidates.append(
                train_linear_impurity_candidate(
                    n,
                    args,
                    benchmark=benchmark,
                    envelope_result=envelope_result,
                )
            )
        if args.include_linear_impurity_ladder:
            candidates.extend(
                train_linear_impurity_ladder_candidates(
                    n,
                    args,
                    benchmark=benchmark,
                    envelope_result=envelope_result,
                )
            )
        if args.include_linear_initialized_neural_candidate:
            candidates.append(
                train_linear_initialized_neural_candidate(
                    n,
                    args,
                    benchmark=benchmark,
                )
            )
        if args.include_multicloud_candidate:
            candidates.append(
                train_multicloud_candidate(n, args, benchmark=benchmark)
            )
    selected = select_candidate(candidates)
    combined = selected["combined_evaluation"]
    ladder_summary = linear_ladder_summary(candidates)
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
                "history": candidate["history"],
            }
            for candidate in candidates
        },
        "candidates": candidates,
        "history": selected["history"],
        "evaluations": selected["evaluations"],
        "combined_evaluation": combined,
        "structure_checks": selected["structure_checks"],
        "linear_ladder_summary": ladder_summary,
    }


def hellmann_feynman_payload(
    n: int,
    args: argparse.Namespace,
    central_payload: dict[str, Any],
) -> dict[str, Any]:
    """Finite-difference check of dE/dg against <Tr X^4>."""

    if args.hf_delta <= 0:
        raise ValueError("--hf-delta must be positive")
    if args.coupling - args.hf_delta < 0:
        return {
            "skipped": True,
            "reason": "coupling - hf_delta would be negative",
            "delta_g": args.hf_delta,
        }

    shifted: dict[str, dict[str, Any]] = {}
    for label, coupling in [
        ("minus", args.coupling - args.hf_delta),
        ("plus", args.coupling + args.hf_delta),
    ]:
        shifted_args = copy.copy(args)
        shifted_args.coupling = coupling
        shifted_args.include_hf_diagnostic = False
        shifted_args.seed = args.seed
        shifted[label] = train_one_n(n, shifted_args)

    target_name = central_payload["selected_candidate"]

    def branch_payload(label: str) -> dict[str, Any]:
        payload = shifted[label]
        candidate_summary = payload["candidate_summaries"].get(target_name)
        if candidate_summary is None:
            candidate_summary = {
                "combined_evaluation": payload["combined_evaluation"],
                "passes_validation_gates": payload[
                    "selected_candidate_passes_validation_gates"
                ],
            }
            candidate_name = payload["selected_candidate"]
            target_available = False
        else:
            candidate_name = target_name
            target_available = True
        return {
            "coupling": (
                args.coupling - args.hf_delta
                if label == "minus"
                else args.coupling + args.hf_delta
            ),
            "selected_candidate": payload["selected_candidate"],
            "finite_difference_candidate": candidate_name,
            "target_candidate_available": target_available,
            "energy_mean": candidate_summary["combined_evaluation"]["energy_mean"],
            "passes_validation_gates": candidate_summary["passes_validation_gates"],
        }

    minus_branch = branch_payload("minus")
    plus_branch = branch_payload("plus")
    e_plus = plus_branch["energy_mean"]
    e_minus = minus_branch["energy_mean"]
    derivative = (e_plus - e_minus) / (2.0 * args.hf_delta)
    central_tr_x4 = central_payload["combined_evaluation"]["tr_x4_mean"]
    return {
        "skipped": False,
        "delta_g": args.hf_delta,
        "central_tr_x4": central_tr_x4,
        "finite_difference_derivative": derivative,
        "error": central_tr_x4 - derivative,
        "central_selected_candidate": target_name,
        "minus": minus_branch,
        "plus": plus_branch,
        "same_selected_candidate": (
            shifted["minus"]["selected_candidate"]
            == target_name
            == shifted["plus"]["selected_candidate"]
        ),
        "same_finite_difference_candidate": (
            minus_branch["finite_difference_candidate"]
            == target_name
            == plus_branch["finite_difference_candidate"]
        ),
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
            and structure["parity_residual"] < 1.0e-10
            and structure["weyl_covariance_residual"] < 1.0e-10
        )
        checks[f"{key}_collision_regular"] = (
            math.isfinite(structure["head_collision_ratio_max_abs"])
            and math.isfinite(structure["profile_collision_identity_residual"])
            and structure["head_collision_ratio_max_abs"] < 1.0e8
            and structure["profile_collision_identity_residual"] < 1.0e-8
        )
        checks[f"{key}_virial_abs_lt_tol"] = (
            combined["virial_residual_abs_max"]
            < results["metadata"]["virial_abs_tol"]
        )
        checks[f"{key}_energy_spread_lt_tol"] = (
            combined["energy_spread_independent_runs"]
            < results["metadata"]["energy_spread_abs_tol"]
        )
        if "benchmark_energy" in combined:
            checks[f"{key}_benchmark_energy_error_lt_tol"] = (
                abs(combined["energy_error"])
                < results["metadata"]["energy_abs_tol"]
            )
        ladder_summary = payload.get("linear_ladder_summary")
        if ladder_summary is not None:
            checks[f"{key}_linear_ladder_grid_monotone"] = ladder_summary[
                "grid_energy_monotone_nonincreasing"
            ]
            checks[f"{key}_linear_ladder_has_passing_rung"] = (
                ladder_summary["lowest_passing_rung"] is not None
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
    if args.print_every < 0:
        raise ValueError("--print-every must be non-negative")
    if args.refresh_training_samples_every < 0:
        raise ValueError("--refresh-training-samples-every must be non-negative")
    if args.per_step_error_samples < 0:
        raise ValueError("--per-step-error-samples must be non-negative")
    if args.per_step_error_replicates < 0:
        raise ValueError("--per-step-error-replicates must be non-negative")
    if args.per_step_error_every < 1:
        raise ValueError("--per-step-error-every must be positive")
    if args.per_step_error_samples == 0 and args.per_step_error_replicates > 0:
        raise ValueError(
            "--per-step-error-samples must be positive when "
            "--per-step-error-replicates is positive"
        )
    if args.per_step_error_samples > 0 and args.per_step_error_replicates < 2:
        raise ValueError(
            "--per-step-error-replicates must be at least two when per-step "
            "error estimates are enabled"
        )
    if args.refresh_per_step_error_samples and args.per_step_error_samples == 0:
        raise ValueError(
            "--per-step-error-samples must be positive when refreshing "
            "per-step error samples"
        )
    if args.include_multicloud_candidate:
        if args.multicloud_steps < 1:
            raise ValueError("--multicloud-steps must be positive")
        if args.multicloud_clouds_per_step < 1:
            raise ValueError("--multicloud-clouds-per-step must be positive")
        if args.multicloud_lr <= 0:
            raise ValueError("--multicloud-lr must be positive")
    if args.include_linear_impurity_candidate:
        if args.linear_impurity_overlap_rtol <= 0:
            raise ValueError("--linear-impurity-overlap-rtol must be positive")
        if not args.linear_impurity_terms:
            raise ValueError("--linear-impurity-terms must be non-empty")
    if args.include_linear_impurity_ladder:
        if args.linear_impurity_overlap_rtol <= 0:
            raise ValueError("--linear-impurity-overlap-rtol must be positive")
    if args.include_linear_initialized_neural_candidate:
        if args.linear_impurity_overlap_rtol <= 0:
            raise ValueError("--linear-impurity-overlap-rtol must be positive")
        if not args.linear_impurity_terms:
            raise ValueError("--linear-impurity-terms must be non-empty")
        if args.linear_initialized_neural_steps < 0:
            raise ValueError("--linear-initialized-neural-steps must be non-negative")
        if args.linear_initialized_neural_lr <= 0:
            raise ValueError("--linear-initialized-neural-lr must be positive")
        if args.linear_initialized_neural_head_regularization < 0:
            raise ValueError(
                "--linear-initialized-neural-head-regularization must be non-negative"
            )
        if args.linear_initialized_neural_action_regularization < 0:
            raise ValueError(
                "--linear-initialized-neural-action-regularization must be non-negative"
            )
        if args.linear_initialized_neural_regularization_samples < 1:
            raise ValueError(
                "--linear-initialized-neural-regularization-samples must be positive"
            )
        if args.head_correction_scale <= 0:
            raise ValueError(
                "--head-correction-scale must be positive for linear initialization"
            )
    if args.include_hf_diagnostic and args.hf_delta <= 0:
        raise ValueError("--hf-delta must be positive")
    if args.action_quadratic_mode == "mlp" and args.training_mode != "importance-adam":
        raise ValueError(
            "--action-quadratic-mode mlp is currently supported only for "
            "--training-mode importance-adam, because the validated-candidates "
            "paths use an explicit alpha * Tr X^2 envelope."
        )

    results: dict[str, Any] = {
        "metadata": {
            "ansatz": "shared flexible SU(N) Chebyshev spectral impurity",
            "hamiltonian": "-1/2 Delta_X + 1/2 omega^2 Tr X^2 + g Tr X^4",
            "command_line": sys.argv,
            "git_commit": git_commit_hash(),
            "git_status_short": git_status_short(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "arguments": serializable_args(args),
            "omega": args.omega,
            "coupling": args.coupling,
            "seed": args.seed,
            "n_values": args.n_values,
            "training_mode": args.training_mode,
            "parity": args.parity,
            "include_linear_impurity_candidate": args.include_linear_impurity_candidate,
            "include_linear_impurity_ladder": args.include_linear_impurity_ladder,
            "include_linear_initialized_neural_candidate": (
                args.include_linear_initialized_neural_candidate
            ),
            "linear_impurity_ladder_specs": linear_impurity_ladder_specs(args.parity),
            "linear_impurity_terms": args.linear_impurity_terms,
            "linear_impurity_overlap_rtol": args.linear_impurity_overlap_rtol,
            "linear_initialized_neural_steps": args.linear_initialized_neural_steps,
            "linear_initialized_neural_lr": args.linear_initialized_neural_lr,
            "linear_initialized_neural_head_regularization": (
                args.linear_initialized_neural_head_regularization
            ),
            "linear_initialized_neural_action_regularization": (
                args.linear_initialized_neural_action_regularization
            ),
            "linear_initialized_neural_regularization_samples": (
                args.linear_initialized_neural_regularization_samples
            ),
            "include_multicloud_candidate": args.include_multicloud_candidate,
            "multicloud_steps": args.multicloud_steps,
            "multicloud_clouds_per_step": args.multicloud_clouds_per_step,
            "multicloud_lr": args.multicloud_lr,
            "energy_abs_tol": args.energy_abs_tol,
            "energy_spread_abs_tol": args.energy_spread_abs_tol,
            "virial_abs_tol": args.virial_abs_tol,
            "min_ess_fraction": args.min_ess_fraction,
            "include_hf_diagnostic": args.include_hf_diagnostic,
            "hf_delta": args.hf_delta,
            "shared_ansatz_config": shared_ansatz_config(args),
            "harmonic_exact_energies": {
                str(n): exact_suN_harmonic_adjoint_energy(n, args.omega)
                for n in args.n_values
            },
            "sampler": {
                "algorithm": "scrambled Sobol Gaussian importance sampling in orthonormal traceless eigenvalue coordinates",
                "n_train_samples": args.n_samples,
                "refresh_training_samples_every": (
                    args.refresh_training_samples_every
                ),
                "n_eval_samples": args.eval_samples,
                "n_eval_replicates": args.n_eval_replicates,
                "proposal_sigma": args.proposal_sigma,
                "per_step_error_samples": args.per_step_error_samples,
                "per_step_error_replicates": args.per_step_error_replicates,
                "per_step_error_every": args.per_step_error_every,
                "refresh_per_step_error_samples": (
                    args.refresh_per_step_error_samples
                ),
                "independent_evaluation_samples": True,
            },
        },
        "runs": {},
    }
    for n in args.n_values:
        print(f"training SU({n}) with shared flexible ansatz")
        run_payload = train_one_n(n, args)
        if args.include_hf_diagnostic:
            run_payload["hellmann_feynman"] = hellmann_feynman_payload(
                n,
                args,
                run_payload,
            )
        results["runs"][f"su{n}"] = run_payload

    results["validation_summary"] = validation_payload(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(results["validation_summary"], indent=2))


if __name__ == "__main__":
    main()
