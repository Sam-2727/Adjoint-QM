#!/usr/bin/env python
"""VMC diagnostics for the exact D-dimensional harmonic oscillator ground state."""

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
    GaussianEnvelopeMLP,
    HarmonicOscillatorPotential,
    RadialFeatureMap,
    exact_isotropic_harmonic_benchmarks,
    metropolis_sample,
    vmc_observables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--step-prefactor",
        type=float,
        default=1.6,
        help="Use step_size = step_prefactor / sqrt(dim).",
    )
    return parser.parse_args()


def exact_gaussian_model(dim: int, omega: float) -> GaussianEnvelopeMLP:
    return GaussianEnvelopeMLP(
        dim=dim,
        hidden_layers=(),
        feature_map=RadialFeatureMap(),
        init_alpha=omega,
        dtype=torch.float64,
        zero_final=True,
    )


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)

    print("D-dimensional harmonic oscillator VMC diagnostic")
    print(f"omega                 {args.omega:.12g}")
    print(f"samples               {args.samples}")
    print(f"chains                {args.chains}")
    print(f"burn_in               {args.burn_in}")
    print(f"thinning              {args.thinning}")
    print(f"base_seed             {args.seed}")
    print()
    header = (
        "dim seed step_size acceptance energy exact_energy energy_error "
        "local_energy_stderr local_energy_std r2 exact_r2 r2_error "
        "r4 exact_r4 r4_error coord_var exact_coord_var coord_var_error "
        "coord_mean_abs_max offdiag_abs_max virial_residual"
    )
    print(header)

    for offset, dim in enumerate(args.dims):
        seed = args.seed + offset
        step_size = args.step_prefactor / sqrt(dim)
        model = exact_gaussian_model(dim, args.omega)
        potential = HarmonicOscillatorPotential(args.omega)
        exact = exact_isotropic_harmonic_benchmarks(dim, args.omega)

        result = metropolis_sample(
            model,
            n_samples=args.samples,
            dim=dim,
            n_chains=args.chains,
            step_size=step_size,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=seed,
            dtype=torch.float64,
        )
        obs = vmc_observables(model, potential, result.samples)

        print(
            f"{dim:d} "
            f"{seed:d} "
            f"{step_size:.8f} "
            f"{result.acceptance_rate:.8f} "
            f"{obs.local_energy_mean:.12f} "
            f"{exact['energy']:.12f} "
            f"{obs.local_energy_mean - exact['energy']:+.3e} "
            f"{obs.local_energy_stderr:.3e} "
            f"{obs.local_energy_std:.3e} "
            f"{obs.x2:.12f} "
            f"{exact['r2']:.12f} "
            f"{obs.x2 - exact['r2']:+.3e} "
            f"{obs.r4:.12f} "
            f"{exact['r4']:.12f} "
            f"{obs.r4 - exact['r4']:+.3e} "
            f"{obs.coordinate_variance_mean:.12f} "
            f"{exact['coordinate_variance']:.12f} "
            f"{obs.coordinate_variance_mean - exact['coordinate_variance']:+.3e} "
            f"{obs.coordinate_mean_abs_max:.3e} "
            f"{obs.coordinate_offdiag_abs_max:.3e} "
            f"{obs.virial_residual:+.3e}"
        )


if __name__ == "__main__":
    main()
