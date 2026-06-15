#!/usr/bin/env python
"""Train the SU(2) adjoint-sector spectral wavefunction by quadrature."""

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
    SU2AdjointSpectralAnsatz,
    adjoint_quadrature_observables,
    exact_su2_harmonic_adjoint_energy,
    su2_adjoint_eigenvalue_grid,
    su2_adjoint_radial_finite_difference_energy,
    train_adjoint_quadrature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=8.0)
    parser.add_argument("--n-grid", type=int, default=3000)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--fd-grid", type=int, default=1600)
    parser.add_argument("--fd-r-max", type=float, default=9.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    _, lam, weights = su2_adjoint_eigenvalue_grid(
        args.z_max,
        args.n_grid,
        dtype=torch.float64,
    )
    model = SU2AdjointSpectralAnsatz(
        omega_init=args.omega,
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
    obs = adjoint_quadrature_observables(
        model,
        lam,
        weights,
        omega=args.omega,
        coupling=args.coupling,
    )

    if args.coupling == 0.0:
        benchmark = exact_su2_harmonic_adjoint_energy(args.omega)
        benchmark_label = "exact_harmonic"
    else:
        benchmark = su2_adjoint_radial_finite_difference_energy(
            omega=args.omega,
            coupling=args.coupling,
            r_max=args.fd_r_max,
            n_grid=args.fd_grid,
            dtype=torch.float64,
        )
        benchmark_label = "radial_finite_difference"

    print("SU(2) adjoint spectral quadrature training")
    print(f"omega                  {args.omega:.12g}")
    print(f"coupling               {args.coupling:.12g}")
    print(f"seed                   {args.seed}")
    print(f"z_max                  {args.z_max:.12g}")
    print(f"n_grid                 {args.n_grid}")
    print(f"steps                  {args.steps}")
    print(f"learning_rate          {args.lr:.12g}")
    print(f"benchmark              {benchmark_label}")
    print()
    print("step energy benchmark energy_error radial angular potential local_std alpha")
    for record in history:
        print(
            f"{record.step:d} "
            f"{record.energy:.12f} "
            f"{benchmark:.12f} "
            f"{record.energy - benchmark:+.3e} "
            f"{record.radial:.12f} "
            f"{record.angular:.12f} "
            f"{record.potential:.12f} "
            f"{record.local_energy_std:.3e} "
            f"{record.alpha:.12f}"
        )
    print()
    print(f"final_energy           {obs.energy:.15f}")
    print(f"benchmark_energy       {benchmark:.15f}")
    print(f"energy_error           {obs.energy - benchmark:+.3e}")
    print(f"traceless_residual     {obs.traceless_residual:.3e}")
    print(f"parity_residual        {obs.parity_residual:.3e}")
    print(f"local_energy_std       {obs.local_energy_std:.3e}")


if __name__ == "__main__":
    main()
