#!/usr/bin/env python
"""Metropolis VMC benchmark for separable D-dimensional quartic oscillators."""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from adjoint_qm import (
    EvenFeatureMap,
    GaussianEnvelopeMLP,
    QuarticOscillatorPotential,
    diagonalize_separable_quartic_oscillator,
    metropolis_sample,
    train_vmc_metropolis,
    vmc_observables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=0.05)
    parser.add_argument("--basis-size", type=int, default=32)
    parser.add_argument("--basis-omega", type=float, default=1.2)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--train-samples", type=int, default=768)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--train-burn-in", type=int, default=150)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--train-thinning", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--lr", type=float, default=5.0e-3)
    parser.add_argument("--hidden-width", type=int, default=16)
    parser.add_argument("--step-prefactor", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)

    print("Separable D-dimensional quartic oscillator Metropolis VMC benchmark")
    print(f"omega                 {args.omega:.12g}")
    print(f"coupling              {args.coupling:.12g}")
    print(f"basis_size            {args.basis_size}")
    print(f"basis_omega           {args.basis_omega:.12g}")
    print(f"train_steps           {args.train_steps}")
    print(f"train_samples         {args.train_samples}")
    print(f"eval_samples          {args.eval_samples}")
    print(f"base_seed             {args.seed}")
    print()
    print(
        "dim seed step_size acceptance energy exact_energy energy_error "
        "relative_error error_per_dim local_energy_stderr local_energy_std "
        "virial_residual"
    )

    for offset, dim in enumerate(args.dims):
        seed = args.seed + offset
        step_size = args.step_prefactor / sqrt(dim)
        torch.manual_seed(seed)
        model = GaussianEnvelopeMLP(
            dim=dim,
            hidden_layers=(args.hidden_width,),
            feature_map=EvenFeatureMap(),
            init_alpha=args.omega + 2.0 * args.coupling,
            dtype=torch.float64,
        )
        potential = QuarticOscillatorPotential(
            omega=args.omega,
            coupling=args.coupling,
        )
        exact = diagonalize_separable_quartic_oscillator(
            dim=dim,
            n_basis=args.basis_size,
            n_levels=4,
            omega=args.omega,
            coupling=args.coupling,
            basis_omega=args.basis_omega,
            dtype=torch.float64,
        )

        train_vmc_metropolis(
            model,
            potential,
            dim=dim,
            n_steps=args.train_steps,
            n_samples=args.train_samples,
            n_chains=args.chains,
            step_size=step_size,
            burn_in=args.train_burn_in,
            thinning=args.train_thinning,
            lr=args.lr,
            seed=seed,
            report_every=args.train_steps,
            dtype=torch.float64,
        )
        result = metropolis_sample(
            model,
            n_samples=args.eval_samples,
            dim=dim,
            n_chains=args.chains,
            step_size=step_size,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=seed + 10_000,
            dtype=torch.float64,
        )
        obs = vmc_observables(model, potential, result.samples)
        exact_energy = float(exact.energies[0])

        print(
            f"{dim:d} "
            f"{seed:d} "
            f"{step_size:.8f} "
            f"{result.acceptance_rate:.8f} "
            f"{obs.local_energy_mean:.12f} "
            f"{exact_energy:.12f} "
            f"{obs.local_energy_mean - exact_energy:+.3e} "
            f"{(obs.local_energy_mean - exact_energy) / exact_energy:+.3e} "
            f"{(obs.local_energy_mean - exact_energy) / dim:+.3e} "
            f"{obs.local_energy_stderr:.3e} "
            f"{obs.local_energy_std:.3e} "
            f"{obs.virial_residual:+.3e}"
        )


if __name__ == "__main__":
    main()
