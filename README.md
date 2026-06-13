# Adjoint-QM

Early-stage research code for neural variational benchmarks in quantum mechanics.

The first benchmark is the one-dimensional harmonic oscillator ground state,
implemented with a parity-even Gaussian-envelope neural wavefunction.  It is
intended as the simplest exact check before moving to finite-`N` matrix quantum
mechanics.

The harmonic oscillator notebook may run training directly because the benchmark
is tiny.  Future expensive or production-quality runs should be performed by
reproducible Python scripts, with notebooks used only to load saved diagnostics,
compare against benchmarks, and explain the results.

Run the initial benchmark with:

```bash
python scripts/train_ho_ground.py
```

Tests are focused on exact harmonic-oscillator identities and smoke checks:

```bash
pytest
```

The library also includes a small Krylov/Lanczos time-evolution utility for
approximating Euclidean correlators of the form
`<phi|exp[-tau (H - E0)]|phi>` without explicitly solving for all excited
states.
