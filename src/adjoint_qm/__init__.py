"""Neural variational tools for quantum-mechanics benchmarks."""

from .ansatz import GaussianEnvelopeMLP
from .estimators import (
    exact_harmonic_benchmarks,
    harmonic_correlator,
    local_energy,
    parity_residual,
    quadrature_energy,
    quadrature_observables,
    trapezoid_grid,
    vmc_observables,
)
from .features import EvenFeatureMap, FeatureMap
from .potentials import HarmonicOscillatorPotential, Potential
from .sampler import MetropolisResult, metropolis_sample
from .time_evolution import (
    KrylovSpectrum,
    LanczosResult,
    dirichlet_grid_1d,
    krylov_correlator,
    krylov_spectrum,
    lanczos_tridiagonal,
    position_correlator_time_evolution_1d,
    position_source_state_1d,
    schrodinger_hamiltonian_action_1d,
    tridiagonal_matrix,
    weighted_inner,
)
from .training import TrainingRecord, train_quadrature

__all__ = [
    "EvenFeatureMap",
    "FeatureMap",
    "GaussianEnvelopeMLP",
    "harmonic_correlator",
    "HarmonicOscillatorPotential",
    "KrylovSpectrum",
    "LanczosResult",
    "MetropolisResult",
    "Potential",
    "TrainingRecord",
    "dirichlet_grid_1d",
    "exact_harmonic_benchmarks",
    "krylov_correlator",
    "krylov_spectrum",
    "lanczos_tridiagonal",
    "local_energy",
    "metropolis_sample",
    "parity_residual",
    "position_correlator_time_evolution_1d",
    "position_source_state_1d",
    "quadrature_energy",
    "quadrature_observables",
    "schrodinger_hamiltonian_action_1d",
    "train_quadrature",
    "tridiagonal_matrix",
    "trapezoid_grid",
    "vmc_observables",
    "weighted_inner",
]
