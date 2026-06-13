#!/usr/bin/env python
"""Train and diagnose a neural ground state for the 1D harmonic oscillator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from adjoint_qm import (
    GaussianEnvelopeMLP,
    HarmonicOscillatorPotential,
    exact_harmonic_benchmarks,
    metropolis_sample,
    quadrature_observables,
    train_quadrature,
    trapezoid_grid,
    vmc_observables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--n-grid", type=int, default=2001)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--init-alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vmc-samples", type=int, default=4096)
    parser.add_argument("--vmc-step-size", type=float, default=1.0)
    parser.add_argument("--vmc-burn-in", type=int, default=500)
    parser.add_argument("--vmc-thinning", type=int, default=5)
    parser.add_argument("--vmc-chains", type=int, default=64)
    parser.add_argument("--skip-vmc", action="store_true")
    return parser.parse_args()


def print_history(history: list) -> None:
    print("quadrature training")
    print("step energy kinetic potential alpha")
    for record in history:
        print(
            f"{record.step:5d} "
            f"{record.energy:.12f} "
            f"{record.kinetic:.12f} "
            f"{record.potential:.12f} "
            f"{record.alpha:.12f}"
        )


def print_quadrature_report(obs, exact: dict[str, float]) -> None:
    print("\nquadrature diagnostics")
    print(f"energy                 {obs.energy:.12f}  error {obs.energy - exact['energy']:+.3e}")
    print(f"kinetic                {obs.kinetic:.12f}  exact {exact['kinetic']:.12f}")
    print(f"potential              {obs.potential:.12f}  exact {exact['potential']:.12f}")
    print(f"x2                     {obs.x2:.12f}  error {obs.x2 - exact['x2']:+.3e}")
    print(f"x4                     {obs.x4:.12f}  error {obs.x4 - exact['x4']:+.3e}")
    print(f"local_energy_mean      {obs.local_energy_mean:.12f}")
    print(f"local_energy_variance  {obs.local_energy_variance:.12e}")
    print(f"parity_residual        {obs.parity_residual:.12e}")
    print(f"virial_residual        {obs.virial_residual:.12e}")
    print(f"G(0)=<x^2>             {obs.x2:.12f}  exact {exact['g0']:.12f}")


def print_vmc_report(result, obs, exact: dict[str, float]) -> None:
    print("\nvmc diagnostics")
    print(f"sample_count           {obs.sample_count}")
    print(f"seed                   {result.seed}")
    print(f"step_size              {result.step_size:.6f}")
    print(f"acceptance_rate        {result.acceptance_rate:.6f}")
    print(f"sample_mean_x          {obs.x_mean:.12f}")
    print(f"x2                     {obs.x2:.12f}  error {obs.x2 - exact['x2']:+.3e}")
    print(f"x4                     {obs.x4:.12f}  error {obs.x4 - exact['x4']:+.3e}")
    print(f"local_energy_mean      {obs.local_energy_mean:.12f}  error {obs.local_energy_mean - exact['energy']:+.3e}")
    print(f"local_energy_std       {obs.local_energy_std:.12e}")
    print(f"local_energy_stderr    {obs.local_energy_stderr:.12e}")


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    dtype = torch.float64
    potential = HarmonicOscillatorPotential(omega=args.omega)
    grid, weights = trapezoid_grid(args.x_max, args.n_grid, dtype=dtype)
    model = GaussianEnvelopeMLP(
        dim=1,
        hidden_layers=tuple([args.hidden_width] * args.hidden_depth),
        init_alpha=args.init_alpha,
        dtype=dtype,
    )
    exact = exact_harmonic_benchmarks(args.omega)

    history = train_quadrature(
        model,
        potential,
        grid,
        weights,
        n_steps=args.steps,
        lr=args.lr,
        report_every=args.report_every,
    )
    print_history(history)

    obs = quadrature_observables(model, potential, grid, weights)
    print_quadrature_report(obs, exact)

    if not args.skip_vmc:
        result = metropolis_sample(
            model,
            n_samples=args.vmc_samples,
            dim=1,
            n_chains=args.vmc_chains,
            step_size=args.vmc_step_size,
            burn_in=args.vmc_burn_in,
            thinning=args.vmc_thinning,
            seed=args.seed + 1,
            dtype=dtype,
        )
        vmc_obs = vmc_observables(model, potential, result.samples)
        print_vmc_report(result, vmc_obs, exact)


if __name__ == "__main__":
    main()
