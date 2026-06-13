Instructions for future agents working in this repository.

This is a theoretical physics project. The standard here is not merely “working code”; it is mathematically and physically serious work with careful reasoning, explicit assumptions, and conservative claims. Treat every derivation, numerical approximation, literature statement, and implementation choice as something that may later be scrutinized by experts.

The required standard is research-grade rigor. Every contribution should aim to be:
- mathematically consistent
- physically well-motivated
- convention-checked
- benchmarked where possible
- explicitly qualified where not yet proven
- documented well enough that another expert can audit the logic

When in doubt, choose the more rigorous path over the faster or more convenient one. A partially slower but trustworthy result is better than a faster result with unclear conventions, weak diagnostics, or overstated interpretation.

The default attitude should be that mistakes in this project will most often come from:
- sign errors
- normalization mistakes
- hidden assumptions
- symmetry violations
- unjustified extrapolations
- numerics that look stable before they are actually trustworthy

So the burden of proof is high. Do not infer correctness from plausibility alone.

1. Physics first, code second.
Before implementing, identify the precise physical object being approximated. Do not code vague ideas.

2. Distinguish exact facts from heuristics.
Always separate:
- exact identities
- numerically motivated ansaetze
- asymptotic arguments
- conjectural expectations
- implementation conveniences
Never blur these categories in prose or code comments.

2a. Make work audit-ready.
Write derivations, code structure, and verification in a way that another careful reader could reconstruct why the result should be trusted. If a step cannot yet be justified cleanly, label it as provisional rather than smoothing it over.

3. Preserve symmetry structure whenever possible.
For matrix models, gauge symmetry and supersymmetry are not optional decorations. Any approximation that breaks them should do so knowingly, temporarily, and with a clear path to restoration or quantitative monitoring.

4. Benchmark the simplest exact case first.
Every new numerical mechanism should be validated first on a case with a known answer before being used in a harder setting. For this repository, the one-matrix harmonic problem is the default first benchmark.

5. Analytic control beats black-box optimization.
When introducing a loss, sampler, ansatz, or regularization, write down the governing formulas explicitly in the note and make sure the code reflects them directly. Avoid “magic” code paths whose physical meaning is unclear.

5a. Verification is part of the deliverable.
No substantial derivation, implementation, or numerical claim is complete without the strongest verification that is realistically available at that stage. A result without the right benchmark, diagnostic, or convention check is not finished work.

8. Write all notes to "codex.

9. Use references conservatively.
Only include references that are immediately relevant to the current argument or implementation. Do not pad the bibliography. If you add a new paper that materially shapes the project, update `references/README.md`.

10. Be explicit about assumptions and regimes.
If a statement depends on:
- finite `N`
- a mass deformation
- a gauge-fixed picture
- an asymptotic regime
- a truncation of fermion space
- a softened activation or regularized potential
say so explicitly in the note and in code comments where relevant.

11. Numerical results must come with diagnostics.
Never report an energy estimate without the context needed to interpret it. At a minimum, record or print:
- sample count
- acceptance rate
- error bar
- random seed when useful
- benchmark comparison when available
If autocorrelation may matter, say so.

11a. Numerical claims should be falsifiable.
Whenever possible, include a comparison that could have failed: an exact benchmark, a symmetry check, a limit in which the answer is known, or an independently motivated observable. Do not rely only on internal loss reduction.

12. Prefer small, testable modules.
Keep code split by physical role: basis, ansatz, potential, sampler, estimator, constraints, tests. Avoid monolithic scripts. New modules should come with at least one small verification test or smoke test.

13. Do not introduce heavyweight dependencies casually.
The current repository is intentionally NumPy-first. If you introduce PyTorch, JAX, or another framework, do so only for a concrete reason and document why the added complexity is justified.

14. Sanity-check formulas before implementation.
Before coding a term, check:
- indices
- normalization conventions
- signs
- factors of `1/2`
- basis conventions
- whether the chosen coordinates are Hermitian, adjoint, or gauge-fixed
Many “numerical bugs” in this project will actually be convention bugs.

14a. Signs deserve special paranoia.
Potential terms, Yukawa couplings, Gauss-law generators, supercharges, Hermitian conjugation, and commutator conventions should all be checked with exceptional care. If a sign convention is even slightly ambiguous, resolve it explicitly in the note before trusting the code.

15. Prefer reproducible scripts over notebook-only work.
Short scripts and tests are easier to review, rerun, and compare. If exploratory work is needed, make sure the durable result ends up in a script, module, or note.

16. Keep generated artifacts out of git.
Do not commit `__pycache__`, LaTeX auxiliary files, downloaded archives, or transient numerical outputs unless the user explicitly wants tracked artifacts.

17. Use focused verification after edits.
After changing code, run the smallest relevant test or smoke test and report it exactly. After changing the note, compile the TeX source if possible. Prefer targeted verification over broad but noisy commands.

18. Record open problems honestly.
If a derivation is incomplete, a convention is uncertain, or a numerical path is only provisional, say so directly. High-quality work here means preserving truth and clarity, not projecting false completeness.

18a. Never hide behind implementation.
If something is not yet at the standard needed for a serious physics claim, say so even if the code runs. Executable code is not evidence of conceptual correctness.


19. Leave the repository in a cleaner state than you found it.
If you add a prototype, also add the minimal test, documentation, and project-structure updates that make it understandable to the next person.

20. Optimize for the highest standard of work, not for quick closure.
Do not stop at “good enough” if a known weakness remains easy to address. Tighten claims, strengthen tests, sharpen formulas, and improve documentation until the result reaches the highest standard that is realistically attainable within the current pass.

21. Your notes should be put in a .tex document titled `codex notes.tex` in the `codex code reasoning` folder. The note `adjoint QM.tex` is written by humans and should not be touched. Use it as reference for things the humans have mostly verified. The note should be treated as mostly corrected, modulo some small errors/typos. Everytime you finish writing something, you should recompile the notes document.