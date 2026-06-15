#!/usr/bin/env python
"""High-accuracy VMC check for a separable D-dimensional quartic oscillator."""

from __future__ import annotations

import argparse
from math import ceil, sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from adjoint_qm import (
    QuarticOscillatorPotential,
    SeparableGaussianEnvelopeMLP,
    diagonalize_quartic_oscillator,
    metropolis_sample,
    quadrature_observables,
    train_quadrature,
    trapezoid_grid,
    vmc_observables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=0.05)
    parser.add_argument("--basis-size", type=int, default=64)
    parser.add_argument("--basis-omega", type=float, default=1.2)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--n-grid", type=int, default=3001)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--chains", type=int, default=512)
    parser.add_argument("--burn-in", type=int, default=700)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--step-prefactor", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=5064)
    parser.add_argument("--eval-seed", type=int, default=268244)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--target-stderr", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    if args.dim < 1:
        raise ValueError("dim must be positive")
    if args.target_stderr <= 0.0:
        raise ValueError("target-stderr must be positive")

    torch.manual_seed(args.seed)
    potential = QuarticOscillatorPotential(
        omega=args.omega,
        coupling=args.coupling,
    )
    exact_1d = diagonalize_quartic_oscillator(
        args.basis_size,
        omega=args.omega,
        coupling=args.coupling,
        basis_omega=args.basis_omega,
        dtype=torch.float64,
    )
    exact_energy_1d = float(exact_1d.energies[0])
    exact_energy_d = args.dim * exact_energy_1d

    model = SeparableGaussianEnvelopeMLP(
        dim=args.dim,
        hidden_layers=(args.hidden_width, args.hidden_width),
        init_alpha=args.omega + 2.0 * args.coupling,
        dtype=torch.float64,
    )
    grid, weights = trapezoid_grid(
        args.x_max,
        args.n_grid,
        dtype=torch.float64,
    )
    train_quadrature(
        model.one_body,
        potential,
        grid,
        weights,
        n_steps=args.train_steps,
        lr=args.lr,
        report_every=args.train_steps,
    )
    one_dimensional_obs = quadrature_observables(
        model.one_body,
        potential,
        grid,
        weights,
    )

    step_size = args.step_prefactor / sqrt(args.dim)
    result = metropolis_sample(
        model,
        n_samples=args.samples,
        dim=args.dim,
        n_chains=args.chains,
        step_size=step_size,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.eval_seed,
        dtype=torch.float64,
    )
    obs = vmc_observables(
        model,
        potential,
        result.samples,
        batch_size=args.batch_size,
    )
    energy_error = obs.local_energy_mean - exact_energy_d
    estimated_samples_for_target = ceil(
        (obs.local_energy_std / args.target_stderr) ** 2
    )

    print("Separable quartic precision VMC benchmark")
    print(f"dim                         {args.dim}")
    print(f"omega                       {args.omega:.12g}")
    print(f"coupling                    {args.coupling:.12g}")
    print(f"basis_size                  {args.basis_size}")
    print(f"basis_omega                 {args.basis_omega:.12g}")
    print(f"train_steps                 {args.train_steps}")
    print(f"train_seed                  {args.seed}")
    print(f"samples                     {args.samples}")
    print(f"chains                      {args.chains}")
    print(f"step_size                   {step_size:.12g}")
    print(f"acceptance                  {result.acceptance_rate:.12g}")
    print(f"eval_seed                   {result.seed}")
    print()
    print(f"one_dim_basis_energy        {exact_energy_1d:.15f}")
    print(f"one_dim_quadrature_energy   {one_dimensional_obs.energy:.15f}")
    print(f"one_dim_energy_error        {one_dimensional_obs.energy - exact_energy_1d:+.3e}")
    print(f"one_dim_local_energy_std    {one_dimensional_obs.local_energy_variance**0.5:.3e}")
    print(f"one_dim_virial_residual     {one_dimensional_obs.virial_residual:+.3e}")
    print()
    print(f"d_dim_basis_energy          {exact_energy_d:.15f}")
    print(f"d_dim_vmc_energy            {obs.local_energy_mean:.15f}")
    print(f"d_dim_energy_error          {energy_error:+.3e}")
    print(f"d_dim_relative_error        {energy_error / exact_energy_d:+.3e}")
    print(f"d_dim_error_per_dimension   {energy_error / args.dim:+.3e}")
    print(f"local_energy_std            {obs.local_energy_std:.3e}")
    print(f"naive_stderr                {obs.local_energy_stderr:.3e}")
    print(f"naive_z_score               {energy_error / obs.local_energy_stderr:+.3e}")
    print(f"virial_residual             {obs.virial_residual:+.3e}")
    print()
    print(f"target_stderr               {args.target_stderr:.3e}")
    print(f"estimated_samples_for_target {estimated_samples_for_target}")
    print(
        "note                        stderr is naive and does not include "
        "autocorrelation corrections"
    )


if __name__ == "__main__":
    main()
