#!/usr/bin/env python
"""Evaluate a saved SU(N) adjoint neural checkpoint.

This script is intentionally separate from training.  It estimates the same
variational Rayleigh quotient as the HMC training script, but uses independent
randomized Sobol-Gaussian importance samples in traceless eigenvalue
coordinates.  For the present SU(4) problem this gives a low-dimensional
quasi-Monte Carlo cross-check of the HMC validation energy without using virial
or other stationarity diagnostics as part of the energy estimator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch

from adjoint_qm import (
    adjoint_importance_moments,
    adjoint_importance_observables,
    sobol_gaussian_traceless_samples,
)
from run_sun_adjoint_vmc_benchmarks import make_shared_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=ROOT / "results" / "sun_adjoint_hmc_neural_su4_g1_1000.json",
        help="Training/evaluation JSON whose metadata defines the ansatz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint .pt file. Defaults to metadata.checkpoint in "
            "--result-json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the evaluation summary.",
    )
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--proposal-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1_240_000)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def load_model_from_result(
    result_json: Path,
    checkpoint: Path | None,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    result = json.loads(result_json.read_text())
    train_args = argparse.Namespace(**result["metadata"]["arguments"])
    checkpoint_path = (
        checkpoint
        if checkpoint is not None
        else Path(result["metadata"]["checkpoint"])
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = make_shared_model(train_args.n, train_args)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, result, checkpoint_path


def evaluate_replicate(
    model: torch.nn.Module,
    *,
    n: int,
    omega: float,
    coupling: float,
    sample_count: int,
    proposal_sigma: float,
    seed: int,
) -> dict[str, Any]:
    samples = sobol_gaussian_traceless_samples(
        n,
        sample_count,
        sigma=proposal_sigma,
        seed=seed,
        dtype=torch.float64,
    )
    obs = adjoint_importance_observables(
        model,
        samples.lam,
        samples.log_prob,
        omega=omega,
        coupling=coupling,
    )
    moments = adjoint_importance_moments(
        model,
        samples.lam,
        samples.log_prob,
        omega=omega,
        coupling=coupling,
    )
    return {
        "seed": seed,
        "sample_count": sample_count,
        "proposal_sigma": proposal_sigma,
        "energy": obs.energy,
        "radial": obs.radial,
        "angular": obs.angular,
        "potential": obs.potential,
        "norm": obs.norm,
        "local_energy_mean": obs.local_energy_mean,
        "local_energy_std": obs.local_energy_std,
        "tr_x2": moments.tr_x2,
        "tr_x4": moments.tr_x4,
        "kinetic": moments.kinetic,
        "virial_rhs": moments.virial_rhs,
        "virial_residual": moments.virial_residual,
    }


def main() -> dict[str, Any]:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least two")
    if args.replicates < 2:
        raise ValueError("--replicates must be at least two")
    if args.proposal_sigma <= 0.0:
        raise ValueError("--proposal-sigma must be positive")

    torch.set_default_dtype(torch.float64)
    model, result, checkpoint_path = load_model_from_result(
        args.result_json,
        args.checkpoint,
    )
    train_args = result["metadata"]["arguments"]
    n = int(train_args["n"])
    omega = float(train_args["omega"])
    coupling = float(train_args["coupling"])

    evaluations = [
        evaluate_replicate(
            model,
            n=n,
            omega=omega,
            coupling=coupling,
            sample_count=args.samples,
            proposal_sigma=args.proposal_sigma,
            seed=args.seed + replicate,
        )
        for replicate in range(args.replicates)
    ]
    energies = [item["energy"] for item in evaluations]
    virials = [item["virial_residual"] for item in evaluations]
    payload: dict[str, Any] = {
        "metadata": {
            "script": Path(__file__).name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "arguments": serializable_args(args),
            "source_result_json": str(args.result_json),
            "checkpoint": str(checkpoint_path),
            "method": (
                "randomized Sobol-Gaussian self-normalized importance sampling"
            ),
            "estimator_note": (
                "Estimates the same variational energy.  Virial residuals are "
                "reported only as independent diagnostics."
            ),
        },
        "evaluations": evaluations,
        "summary": {
            "energy_mean": statistics.mean(energies),
            "energy_sd_across_replicates": statistics.stdev(energies),
            "energy_standard_error": statistics.stdev(energies)
            / math.sqrt(len(energies)),
            "energy_min": min(energies),
            "energy_max": max(energies),
            "energy_spread": max(energies) - min(energies),
            "virial_mean": statistics.mean(virials),
            "virial_abs_max": max(abs(value) for value in virials),
            "replicates": len(evaluations),
            "sample_count_per_replicate": args.samples,
            "proposal_sigma": args.proposal_sigma,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")
    print(json.dumps(payload["summary"], indent=2))
    return payload


if __name__ == "__main__":
    main()
