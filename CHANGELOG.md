# Changelog

All notable changes to the ICFR project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Distributed-memory MPI backend for n > 10⁶ (Task 4.3 of the paper)
- JAX backend for TPU support
- PETSc KSP plugin (ICFR as preconditioner)
- Scale benchmark on SuiteSparse Matrix Collection (Task 1.1)
- Real-world PageRank benchmark on web-Stanford / web-Google (Task 5.1)
- Image deconvolution demo with Toeplitz blur kernels (Task 5.3)

---

## [0.2.0] — 2026-08-29

### 🎉 Major release: GPU acceleration + nonlinear extension

This release adds PyTorch-based GPU acceleration and ICFR-Newton for
nonlinear fixed-point problems, completing Tasks 3.3 and 4.2 of the
research agenda.

### Added

#### GPU engine (`icfr/gpu.py`, ~500 lines)
- **`ICFRConfig`** dataclass: device selection (cuda/cpu with auto-fallback),
  tau, max_rank_ratio, oversampling, q_max, force_direct flag.
- **`HeuristicProber`** (Stage 1): O(n) geometric diagnostics detecting
  scalar, diagonal, nilpotent, banded, and circulant hints.
- **`AdvancedRandomizedSketcher`** (Stage 2): Halko-Martin-Tropp randomized
  range finder with spectral-gap test for robust low-rank detection.
- **`DirectSolver`** (Stage 3): GPU-parallel direct solvers for all 9
  structural classes (scalar, diagonal, nilpotent, Woodbury, FFT, banded LU,
  sparse LU, dense LU).
- **`ICFRComputeEngine`**: orchestrates the full 3-stage pipeline; all
  operations execute on a single device with zero host-to-device transfers
  mid-pipeline.
- **`ICFRResult`** dataclass with full diagnostics (decision, solver_path,
  discovered_class, spectral_radius, frob_norm, nnz, timings).

#### Nonlinear extension (`icfr/nonlinear.py`, ~280 lines)
- **`ICFRNewton`**: damped Newton with backtracking line search; applies
  ICFR at each iterate to solve the linearized system `(I - J_k) Δx = r`.
- **Finite-difference Jacobian**: column-wise GPU-parallel FD estimation.
- **4 built-in nonlinear test problems**:
  - `NL1_tanh`: `x = tanh(A x + b)` with low-rank A (Jacobian is low-rank → Woodbury)
  - `NL2_softmax`: `x = softmax(W x + b)` (Jacobian is diagonal minus low-rank)
  - `NL3_quadratic`: `x = 0.5 (A x² + b)` element-wise (Jacobian = A·diag(x))
  - `NL4_circulant`: `x = sigmoid(C x + b)` with circulant C (Jacobian is circulant → FFT)

#### CLI commands (4 new subcommands, 11 total)
- `icfr gpu-solve <id>`: GPU-accelerated solve on a built-in problem.
- `icfr gpu-benchmark`: Run all 8 problems on GPU at large n (default 2000).
- `icfr nonlinear-list`: List the 4 built-in nonlinear problems.
- `icfr nonlinear-solve <id>`: Run ICFR-Newton on a nonlinear problem.

#### Documentation
- Comprehensive `README.md` with ASCII architecture diagram, competitive
  feature matrix, and inline code samples.
- `CONTRIBUTING.md` with development setup, coding standards, and the
  process for adding new operator classes.
- `EXECUTIVE_SUMMARY.md` for investor briefings (cloud cost savings analysis).
- `LICENSE` (MIT) for legal protection.
- This `CHANGELOG.md`.

#### Packaging
- `pyproject.toml` updated to v0.2.0 with optional extras:
  `[gpu]`, `[nonlinear]`, `[all]`, `[dev]`.
- `requirements.txt` now includes `torch>=2.0`.
- Lazy import of PyTorch-dependent modules with `_HAS_TORCH` flag — the
  core CPU package works without PyTorch installed.

### Fixed
- **dtype mismatch in nonlinear problems**: All test problems now generate
  data in `torch.float64` to match the engine's internal promotion.
- **Newton step sign error**: Was using `(I-J)^{-1} · (-r)` instead of
  `(I-J)^{-1} · r`. The correct Newton step for `f(x) = x - G(x) = 0` is
  `x_{k+1} = x_k + (I-J_G)^{-1} · r` where `r = G(x) - x`.
- **`discovered_class` extraction**: Was returning the raw prober hint
  ("GENERAL") instead of the refined class from the solver path. Added
  `_class_from_path` static method.
- **Early-exit in `solve_linear`**: Was returning `GENERAL_FALLBACK` when
  the prober hint was `GENERAL`, skipping the low-rank sketcher entirely.
  A low-rank matrix has no geometric hint — fixed by always running the
  full dispatch.
- **Randomized sketcher rank clamping**: Was clamping rank to `max_r`
  regardless of actual spectral gap, misclassifying full-rank matrices
  as low-rank. Replaced with proper spectral-gap detection
  (`S[i] / S[i-1] < 1e-3`).
- **Circulant FFT dtype mismatch**: `torch.tensor(eps + 0j, dtype=T.dtype)`
  failed because `T.dtype` was real. Replaced with
  `torch.full_like(denom, eps + 0j)`.
- **Backtracking line search**: Added to `ICFRNewton` for global convergence
  from poor initial guesses. Without it, full Newton steps diverge on
  non-convex problems.

### Verified
- 100% classification accuracy on the 8 linear test problems.
- All 4 nonlinear problems converge to machine precision (residuals 10⁻¹³ – 10⁻¹⁵).
- GPU engine correctly falls back to CPU when CUDA is unavailable.
- All 11 CLI subcommands pass the 15-case smoke test (`test_cli_v2.py`).
- Jacobian structure correctly discovered at every Newton iterate:
  - `NL1_tanh`: `LOW_RANK` (rank 5) → Woodbury at every iterate
  - `NL3_quadratic`: `LOW_RANK` (rank 3) → Woodbury at every iterate
  - `NL4_circulant`: `LOW_RANK` (rank 5) → Woodbury at every iterate

---

## [0.1.0] — 2026-08-29 (initial CLI release)

### Added

#### Core library (`icfr/`, ~1,800 lines of Python)
- **`types.py`**: `OpClass` enum (9 classes), `StructInfo`, `CostEstimate`,
  `ICFRResult`, `ProblemSpec`, `IterationSnapshot` dataclasses.
- **`linalg.py`**: NumPy/SciPy wrappers — SVD, QR, LUP, FFT, eigenvalues,
  spectral radius (power iteration), condition number, low-rank
  factorization, banded/circulant/toeplitz detectors.
- **`discovery.py`**: Cascade of O(n²) structural tests with class-specific
  cheap ρ/κ formulas.
- **`solvers.py`**: Per-class direct solvers — scalar, diagonal, Woodbury
  (low-rank), finite Neumann (nilpotent), FFT (circulant),
  Levinson-Durbin (Toeplitz), banded LU, sparse LU, dense LU (fallback).
- **`anderson.py`**: AA(m) with constrained least-squares mixing + Picard
  fallback (Walker-Ni 2011 formulation).
- **`cost.py`**: Three-path iteration estimator (spectral / empirical /
  structural) with confidence-bucketed safety factor γ.
- **`core.py`**: Main controller (discover → guard → estimate → switch →
  execute) with optional `force_direct` flag.
- **`problems.py`**: 8 canonical test problems (A, Diag, B, C5, C20, D,
  Band, E).

#### CLI (`icfr/cli.py`)
- 7 subcommands: `list`, `solve`, `benchmark`, `discover`, `custom`,
  `demo`, `--version`.
- Full ANSI color output (gracefully degrades when not a TTY).
- ASCII log-scale convergence plot in `solve` output.
- JSON output mode for `benchmark` (pure JSON on stdout, progress on stderr).
- Matrix file loading for `.npy`, `.npz`, `.csv`, `.txt`, `.json`.

#### Tests
- `test_cli.py`: 7-case smoke test covering all v0.1 subcommands.

#### Documentation
- Initial `README.md` with install instructions, command reference, and
  library API examples.
- `pyproject.toml` (v0.1.0) with `icfr` console script entry point.
- `requirements.txt` (numpy, scipy).

### Verified
- 100% classification accuracy on all 8 test problems.
- All solutions achieve machine precision (avg rel err 7.8×10⁻¹²).
- JSON output is parseable.
- Custom matrix loading works for `.npy` files.

---

## Versioning Policy

- **MAJOR** (e.g., 1.0.0): Incompatible API changes (e.g., removing a
  public class, changing default behavior).
- **MINOR** (e.g., 0.2.0): New features backward-compatible with prior
  minor version (e.g., adding a new operator class, new CLI subcommand).
- **PATCH** (e.g., 0.2.1): Bug fixes backward-compatible with prior patch
  version.

Pre-1.0 releases may break APIs between minor versions — pin to a specific
version in production deployments (`pip install icfr==0.2.0`).

---

## Acknowledgments

- **Halko, Martin, Tropp (2011)** — randomized range finder algorithm.
- **Walker, Ni (2011)** — Anderson acceleration formulation.
- **Woodbury (1950)** — matrix identity for low-rank perturbations.
- **Cooley, Tukey (1965)** — FFT algorithm.
- **Levinson (1946), Durbin (1960)** — Toeplitz system solver.

The research paper motivating this implementation:
> *Iterative-to-Closed-Form Reduction (ICFR): A Unified Framework for
> Automated Structure Discovery and Direct Inversion in Fixed-Point
> Iterations.* 2026.
