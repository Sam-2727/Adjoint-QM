#!/usr/bin/env python
"""Train the shared SU(N) adjoint neural ansatz with lagged HMC refreshes."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch

from adjoint_qm import (
    adjoint_hmc_log_norm_density,
    adjoint_lagged_log_weights,
    adjoint_lagged_reweighted_energy,
    hmc_sample_randomized,
    hmc_warmup,
    minimum_ordered_gap,
    ordered_traceless_gaussian_initial,
    relative_effective_sample_size,
    tangent_project,
    traceless_hyperplane_basis,
)
from run_sun_adjoint_vmc_benchmarks import (
    make_shared_model,
    profile_slice_payload,
    shared_ansatz_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "sun_adjoint_hmc_neural_train.json")
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help=(
            "Path for the final model checkpoint. Defaults to the output JSON "
            "path with a .pt suffix."
        ),
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--anchor-refresh-ess", type=float, default=0.70)
    parser.add_argument("--train-beta", type=float, default=0.5)
    parser.add_argument("--validation-beta", type=float, default=1.0)
    parser.add_argument("--num-chains", type=int, default=512)
    parser.add_argument("--initial-sigma", type=float, default=1.5)
    parser.add_argument("--initial-step-size", type=float, default=None)
    parser.add_argument("--hmc-warmup", type=int, default=120)
    parser.add_argument("--hmc-transitions-per-refresh", type=int, default=10)
    parser.add_argument("--leapfrog-min", type=int, default=6)
    parser.add_argument("--leapfrog-max", type=int, default=14)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--production-step-scale", type=float, default=0.20)
    parser.add_argument("--validation-step-scale", type=float, default=None)
    parser.add_argument("--no-adapt-mass", dest="adapt_mass", action="store_false")
    parser.set_defaults(adapt_mass=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--validation-chains", type=int, default=512)
    parser.add_argument("--validation-warmup", type=int, default=120)
    parser.add_argument("--validation-transitions", type=int, default=80)
    parser.add_argument("--validation-replicates", type=int, default=3)
    parser.add_argument("--feature-mode", choices=["raw_moments", "shape", "shape_quadratic"], default="shape_quadratic")
    parser.add_argument("--feature-scale-init", type=float, default=1.0)
    parser.add_argument("--learn-feature-scale", action="store_true")
    parser.add_argument("--parity", choices=["odd", "even"], default="odd")
    parser.add_argument("--chebyshev-degrees", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--learn-coordinate-scale", action="store_true")
    parser.add_argument("--action-correction-scale", type=float, default=0.25)
    parser.add_argument("--head-correction-scale", type=float, default=0.25)
    parser.add_argument("--head-coefficient-mode", choices=["full", "anchored"], default="full")
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def detached_anchor(model: torch.nn.Module) -> torch.nn.Module:
    anchor = copy.deepcopy(model)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    return anchor


def hmc_log_prob_for_model(
    model: torch.nn.Module,
    *,
    beta: float,
    basis: torch.Tensor,
):
    def log_prob(z: torch.Tensor) -> torch.Tensor:
        return adjoint_hmc_log_norm_density(
            model,
            z,
            beta=beta,
            basis=basis,
            ordered=True,
        )

    return log_prob


def flatten_hmc_samples(samples: torch.Tensor, basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = samples.reshape(-1, samples.shape[-1]).detach()
    lam = tangent_project(z @ basis.T)
    return z, lam.detach()


def hmc_gap_diagnostics(samples: torch.Tensor, basis: torch.Tensor) -> dict[str, float]:
    z = samples.reshape(-1, samples.shape[-1]).detach()
    gaps = minimum_ordered_gap(z, basis)
    return {
        "minimum_gap_min": tensor_float(torch.min(gaps)),
        "minimum_gap_mean": tensor_float(torch.mean(gaps)),
    }


def tensor_float(value: torch.Tensor) -> float:
    return float(value.detach())


def pool_diagnostics(
    model: torch.nn.Module,
    anchor: torch.nn.Module,
    pool_lam: torch.Tensor,
    *,
    beta: float,
    omega: float,
    coupling: float,
) -> dict[str, float]:
    estimate = adjoint_lagged_reweighted_energy(
        model,
        anchor,
        pool_lam,
        beta=beta,
        omega=omega,
        coupling=coupling,
    )
    return {
        "energy": tensor_float(estimate.energy),
        "radial": tensor_float(estimate.radial),
        "angular": tensor_float(estimate.angular),
        "kinetic": tensor_float(estimate.kinetic),
        "potential": tensor_float(estimate.potential),
        "tr_x2": tensor_float(estimate.tr_x2),
        "tr_x4": tensor_float(estimate.tr_x4),
        "virial_rhs": tensor_float(estimate.virial_rhs),
        "virial_residual": tensor_float(estimate.virial_residual),
        "relative_ess": tensor_float(estimate.relative_ess),
    }


def train() -> dict[str, Any]:
    args = parse_args()
    if args.n < 2:
        raise ValueError("--n must be at least 2")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be positive")
    if args.refresh_every < 1:
        raise ValueError("--refresh-every must be positive")
    if not 0.0 < args.anchor_refresh_ess <= 1.0:
        raise ValueError("--anchor-refresh-ess must lie in (0, 1]")
    if args.train_beta < 0.0 or args.validation_beta < 0.0:
        raise ValueError("beta values must be non-negative")
    if args.leapfrog_max < args.leapfrog_min:
        raise ValueError("--leapfrog-max must be at least --leapfrog-min")
    if args.production_step_scale <= 0.0:
        raise ValueError("--production-step-scale must be positive")
    if args.validation_step_scale is not None and args.validation_step_scale <= 0.0:
        raise ValueError("--validation-step-scale must be positive")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = make_shared_model(args.n, args)
    model.train()
    basis = traceless_hyperplane_basis(args.n, dtype=torch.float64)
    initial_step_size = (
        args.initial_step_size
        if args.initial_step_size is not None
        else 0.02 / math.sqrt(float(args.n))
    )
    initial_z, _ = ordered_traceless_gaussian_initial(
        args.n,
        args.num_chains,
        sigma=args.initial_sigma,
        dtype=torch.float64,
        generator=generator,
    )
    initial_anchor = detached_anchor(model)
    initial_log_prob = hmc_log_prob_for_model(
        initial_anchor,
        beta=args.train_beta,
        basis=basis,
    )
    warmup_start = time.perf_counter()
    warmup = hmc_warmup(
        initial_log_prob,
        initial_z,
        initial_step_size=initial_step_size,
        leapfrog_min=args.leapfrog_min,
        leapfrog_max=args.leapfrog_max,
        n_warmup=args.hmc_warmup,
        target_acceptance=args.target_accept,
        adapt_mass=args.adapt_mass,
        generator=generator,
    )
    warmup_wall = time.perf_counter() - warmup_start
    z_state = warmup.final_z
    step_size = warmup.step_size * args.production_step_scale
    mass = warmup.mass

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    history: list[dict[str, Any]] = []
    loss_history: list[dict[str, float | int]] = []
    refresh_history: list[dict[str, Any]] = []
    force_refresh = True
    anchor = detached_anchor(model)
    pool_lam: torch.Tensor | None = None
    pool_sample_count = 0
    train_start = time.perf_counter()

    for step in range(1, args.n_steps + 1):
        if force_refresh or (step - 1) % args.refresh_every == 0:
            anchor = detached_anchor(model)
            log_prob = hmc_log_prob_for_model(anchor, beta=args.train_beta, basis=basis)
            refresh_start = time.perf_counter()
            chain = hmc_sample_randomized(
                log_prob,
                z_state,
                step_size=step_size,
                leapfrog_min=args.leapfrog_min,
                leapfrog_max=args.leapfrog_max,
                n_steps=args.hmc_transitions_per_refresh,
                mass=mass,
                generator=generator,
            )
            z_state = chain.final_z
            gap_diagnostics = hmc_gap_diagnostics(chain.samples, basis)
            _, pool_lam = flatten_hmc_samples(chain.samples, basis)
            pool_sample_count = int(pool_lam.shape[0])
            refresh_history.append(
                {
                    "step": step,
                    "beta": args.train_beta,
                    "sample_count": pool_sample_count,
                    "step_size": chain.step_size,
                    "acceptance_rate": chain.acceptance_rate,
                    "divergence_fraction": chain.divergence_fraction,
                    "mean_abs_hamiltonian_error": chain.mean_abs_hamiltonian_error,
                    "max_abs_hamiltonian_error": chain.max_abs_hamiltonian_error,
                    **gap_diagnostics,
                    "wall_time_seconds": time.perf_counter() - refresh_start,
                    "forced_by_low_ess": bool(force_refresh and step != 1),
                }
            )
            force_refresh = False

        assert pool_lam is not None
        if args.batch_size < pool_sample_count:
            indices = torch.randperm(pool_sample_count, generator=generator)[: args.batch_size]
            batch_lam = pool_lam[indices]
        else:
            batch_lam = pool_lam

        optimizer.zero_grad(set_to_none=True)
        estimate = adjoint_lagged_reweighted_energy(
            model,
            anchor,
            batch_lam,
            beta=args.train_beta,
            omega=args.omega,
            coupling=args.coupling,
        )
        loss = estimate.energy
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            full_log_weights = adjoint_lagged_log_weights(
                model,
                anchor,
                pool_lam,
                beta=args.train_beta,
            )
            full_relative_ess = relative_effective_sample_size(full_log_weights)
        relative_ess_value = tensor_float(full_relative_ess)
        if relative_ess_value < args.anchor_refresh_ess:
            force_refresh = True
        loss_history.append(
            {
                "step": step,
                "loss": tensor_float(loss),
                "batch_relative_ess": tensor_float(estimate.relative_ess),
                "pool_relative_ess_after_update": relative_ess_value,
                "grad_norm": float(grad_norm.detach()),
            }
        )

        if step == 1 or step % args.report_every == 0 or step == args.n_steps:
            diagnostics = pool_diagnostics(
                model,
                anchor,
                pool_lam,
                beta=args.train_beta,
                omega=args.omega,
                coupling=args.coupling,
            )
            record = {
                "step": step,
                "stage": "hmc_lagged_training",
                "sample_count": pool_sample_count,
                "loss": tensor_float(loss),
                "batch_relative_ess": tensor_float(estimate.relative_ess),
                "pool_relative_ess_after_update": relative_ess_value,
                "grad_norm": float(grad_norm.detach()),
                **diagnostics,
            }
            history.append(record)

        if args.print_every and (
            step == 1 or step % args.print_every == 0 or step == args.n_steps
        ):
            print(
                f"SU({args.n}) HMC Adam step {step}/{args.n_steps}: "
                f"loss={tensor_float(loss):.12g} "
                f"pool_ess={relative_ess_value:.3f}",
                flush=True,
            )

    train_wall = time.perf_counter() - train_start
    validation = validate_model(model, basis, args)
    profile_slices = profile_slice_payload(model, args)
    checkpoint_output = (
        args.checkpoint_output
        if args.checkpoint_output is not None
        else args.output.with_suffix(".pt")
    )
    energy_decreased = (
        len(loss_history) >= 2
        and loss_history[-1]["loss"] < loss_history[0]["loss"]
    )
    payload: dict[str, Any] = {
        "metadata": {
            "script": Path(__file__).name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "arguments": serializable_args(args),
            "ansatz_config": shared_ansatz_config(args),
            "hamiltonian": "-1/2 Delta_X + 1/2 omega^2 Tr X^2 + g Tr X^4",
            "sampler": "persistent ordered-chamber HMC with lagged reweighting",
            "checkpoint": str(checkpoint_output),
        },
        "warmup": {
            "step_size": warmup.step_size,
            "production_step_size": step_size,
            "production_step_scale": args.production_step_scale,
            "mass": warmup.mass.tolist(),
            "acceptance_rate": warmup.acceptance_rate,
            "divergence_fraction": warmup.divergence_fraction,
            "n_warmup": warmup.n_warmup,
            "leapfrog_min": warmup.leapfrog_min,
            "leapfrog_max": warmup.leapfrog_max,
            "target_acceptance": warmup.target_acceptance,
            "adapt_mass": args.adapt_mass,
            "wall_time_seconds": warmup_wall,
        },
        "training": {
            "history": history,
            "loss_history": loss_history,
            "refresh_history": refresh_history,
            "wall_time_seconds": train_wall,
            "energy_decreased": energy_decreased,
            "first_loss": loss_history[0]["loss"],
            "final_loss": loss_history[-1]["loss"],
        },
        "validation": validation,
        "profile_slices": profile_slices,
    }
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": payload["metadata"],
            "arguments": serializable_args(args),
            "ansatz_config": shared_ansatz_config(args),
            "validation": validation,
        },
        checkpoint_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {checkpoint_output}")
    print(f"wrote {args.output}")
    print(json.dumps({
        "energy_decreased": energy_decreased,
        "first_loss": loss_history[0]["loss"],
        "final_loss": loss_history[-1]["loss"],
        "validation_energy_mean": validation["energy_mean"],
        "validation_virial_abs_max": validation["virial_abs_max"],
    }, indent=2))
    return payload


def validate_model(
    model: torch.nn.Module,
    basis: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for replicate in range(args.validation_replicates):
        generator = torch.Generator().manual_seed(args.seed + 100_000 + replicate)
        initial_z, _ = ordered_traceless_gaussian_initial(
            args.n,
            args.validation_chains,
            sigma=args.initial_sigma,
            dtype=torch.float64,
            generator=generator,
        )
        anchor = detached_anchor(model)
        log_prob = hmc_log_prob_for_model(
            anchor,
            beta=args.validation_beta,
            basis=basis,
        )
        warmup = hmc_warmup(
            log_prob,
            initial_z,
            initial_step_size=(
                args.initial_step_size
                if args.initial_step_size is not None
                else 0.02 / math.sqrt(float(args.n))
            ),
            leapfrog_min=args.leapfrog_min,
            leapfrog_max=args.leapfrog_max,
            n_warmup=args.validation_warmup,
            target_acceptance=args.target_accept,
            adapt_mass=args.adapt_mass,
            generator=generator,
        )
        validation_step_scale = (
            args.validation_step_scale
            if args.validation_step_scale is not None
            else args.production_step_scale
        )
        validation_step_size = warmup.step_size * validation_step_scale
        chain = hmc_sample_randomized(
            log_prob,
            warmup.final_z,
            step_size=validation_step_size,
            leapfrog_min=args.leapfrog_min,
            leapfrog_max=args.leapfrog_max,
            n_steps=args.validation_transitions,
            mass=warmup.mass,
            generator=generator,
        )
        gap_diagnostics = hmc_gap_diagnostics(chain.samples, basis)
        _, lam = flatten_hmc_samples(chain.samples, basis)
        diagnostics = pool_diagnostics(
            model,
            anchor,
            lam,
            beta=args.validation_beta,
            omega=args.omega,
            coupling=args.coupling,
        )
        evaluations.append(
            {
                "replicate": replicate,
                "sample_count": int(lam.shape[0]),
                "warmup_acceptance_rate": warmup.acceptance_rate,
                "warmup_divergence_fraction": warmup.divergence_fraction,
                "warmup_step_size": warmup.step_size,
                "step_size": validation_step_size,
                "step_size_scale": validation_step_scale,
                "acceptance_rate": chain.acceptance_rate,
                "divergence_fraction": chain.divergence_fraction,
                "mean_abs_hamiltonian_error": chain.mean_abs_hamiltonian_error,
                "max_abs_hamiltonian_error": chain.max_abs_hamiltonian_error,
                **gap_diagnostics,
                **diagnostics,
            }
        )

    energy_values = [item["energy"] for item in evaluations]
    virial_values = [item["virial_residual"] for item in evaluations]
    return {
        "evaluations": evaluations,
        "energy_mean": sum(energy_values) / len(energy_values),
        "energy_min": min(energy_values),
        "energy_max": max(energy_values),
        "energy_spread": max(energy_values) - min(energy_values),
        "virial_mean": sum(virial_values) / len(virial_values),
        "virial_abs_max": max(abs(value) for value in virial_values),
        "validation_beta": args.validation_beta,
        "validation_step_scale": (
            args.validation_step_scale
            if args.validation_step_scale is not None
            else args.production_step_scale
        ),
    }


if __name__ == "__main__":
    train()
