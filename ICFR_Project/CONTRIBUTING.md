# Contributing to ICFR

First off — **thank you** for taking the time to contribute! 🎉

ICFR is a research-driven numerical framework, and we welcome contributions
from numerical analysts, PyTorch engineers, and anyone interested in
high-performance linear algebra.

This document describes how to set up a development environment, the project's
coding standards, and the process for submitting changes.

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Project Layout](#-project-layout)
- [Development Workflow](#-development-workflow)
- [Coding Standards](#-coding-standards)
- [Testing](#-testing)
- [Adding a New Operator Class](#-adding-a-new-operator-class)
- [Adding a New Nonlinear Problem](#-adding-a-new-nonlinear-problem)
- [Submitting Changes](#-submitting-changes)
- [Roadmap](#-roadmap)

---

## 🤝 Code of Conduct

Be kind. Be precise. Cite your sources. Disagreements about numerical
methods are fine — personal attacks are not. We follow the
[Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.9
- NumPy ≥ 1.22, SciPy ≥ 1.8
- PyTorch ≥ 2.0 (optional — needed only for GPU + nonlinear features)
- Git

### Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/icfr.git
cd icfr

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in editable mode with all extras
pip install -e ".[all,dev]"

# Verify the install
icfr --version
python -m icfr.cli test  # if you add a test subcommand
python test_cli_v2.py
```

---

## 🗂 Project Layout

```
icfr-cli/
├── icfr/
│   ├── __init__.py        # Public API exports
│   ├── types.py           # OpClass enum, dataclasses
│   ├── linalg.py          # NumPy/SciPy wrappers (CPU baseline)
│   ├── discovery.py       # Structure discovery cascade
│   ├── solvers.py         # Per-class CPU direct solvers
│   ├── anderson.py        # Anderson acceleration AA(m)
│   ├── cost.py            # Three-path cost estimator
│   ├── core.py            # Main CPU ICFR controller
│   ├── problems.py        # 8 linear test problems
│   ├── gpu.py             # GPU engine (PyTorch)
│   ├── nonlinear.py       # ICFR-Newton extension
│   └── cli.py             # 11 subcommands
├── examples/              # Sample .npy matrices
├── tests/                 # Unit tests
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── EXECUTIVE_SUMMARY.md
├── CHANGELOG.md
└── test_cli.py / test_cli_v2.py
```

---

## 🛠 Development Workflow

1. **Pick an issue** from the GitHub issue tracker (or open one describing
   what you want to work on).
2. **Create a branch**: `git checkout -b feat/my-feature` or `fix/my-bugfix`.
3. **Make your changes**. Keep commits focused — one logical change per commit.
4. **Run tests locally**:
   ```bash
   python test_cli_v2.py         # 15 smoke tests
   python -m pytest tests/       # unit tests (if present)
   ```
5. **Lint your code**:
   ```bash
   pip install ruff
   ruff check icfr/
   ```
6. **Push and open a Pull Request** with a clear description of:
   - What changed
   - Why it changed
   - How it was tested
   - Any breaking changes

---

## 🎨 Coding Standards

### Python style

- **Type hints are mandatory** in all public functions.
- **Docstrings** in Google style for every public function/class:
  ```python
  def discover_structure(T: np.ndarray) -> StructInfo:
      """Probe an unknown operator T and return its discovered structural class.

      Args:
          T: Square operator of shape (n, n).

      Returns:
          StructInfo with op_class, params, spectral_radius, etc.
      """
  ```
- **Line length**: 100 characters max.
- **Imports**: stdlib first, then third-party, then local. Use absolute imports
  (`from icfr.linalg import frob_norm`), not relative ones, in tests and examples.
- **No `print()` in library code** — use logging or return structured data.
  `print()` is OK in the CLI module.

### Numerical conventions

- **dtype**: Always use `float64` (NumPy) or `torch.float64` (PyTorch) for
  numerical work. Mixed-dtype operations silently lose precision.
- **Reproducibility**: Every random construction takes a `seed` parameter.
  The default seed is `42` to match the paper.
- **Tolerances**: Use the `tau` parameter from `ICFRConfig` rather than
  hard-coded `1e-8` constants.
- **Shape checks**: Validate input shapes at the top of public functions
  with clear error messages.

### Git conventions

- **Commit messages**: Conventional Commits style:
  ```
  feat(gpu): add banded LU solver for Stage 3
  fix(nonlinear): correct Newton step sign (was -r, should be +r)
  docs: add EXECUTIVE_SUMMARY.md
  test: add smoke test for gpu-benchmark --json
  chore: bump version to 0.2.1
  ```
- **Branch names**: `<type>/<short-description>` — e.g. `feat/gpu-banded`,
  `fix/newton-sign`, `docs/exec-summary`.

---

## 🧪 Testing

### Test layers

1. **Smoke tests** (`test_cli_v2.py`): 15 end-to-end CLI tests — run before
   every commit. Must pass 100%.
2. **Unit tests** (`tests/test_*.py`): Per-module tests for individual
   functions (linalg, discovery, solvers, anderson, cost, nonlinear).
3. **Property tests** (optional, `tests/test_property.py`): Hypothesis-based
   tests verifying invariants like "ICFR with force_direct always produces
   rel err < 1e-10 on structured problems".

### Running tests

```bash
# Smoke tests (fast — 30 seconds)
python test_cli_v2.py

# Unit tests (slower — 2 minutes)
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_solvers.py::test_woodbury_rank_5 -v

# With coverage
python -m pytest tests/ --cov=icfr --cov-report=html
```

### Test problem convention

When adding a test problem, follow the existing pattern:

```python
ProblemSpec(
    id="F",                          # unique short id
    label="F - Symmetric PSD",       # human-readable label
    op_class=OpClass.LOW_RANK,       # expected class
    n=200,                           # default dimension
    rho_target=0.9,                  # target spectral radius
    description="T = 0.9 * Q Λ Q^T with rank-3 spectrum.",
    construction="0.9 * Q Λ Q^T, Q orthonormal, Λ = diag([1,1,1,0,0,...])",
)
```

---

## ➕ Adding a New Operator Class

This is the most impactful contribution. To add a new structural class
(say, "HAMILTONIAN"):

1. **Add the enum value** in `icfr/types.py`:
   ```python
   class OpClass(str, Enum):
       ...
       HAMILTONIAN = "HAMILTONIAN"
   ```
   Also add an entry to `OP_CLASS_INFO` with label, description, complexity,
   and a color.

2. **Add a detection test** in `icfr/discovery.py` (CPU) and
   `icfr/gpu.py:HeuristicProber` (GPU). The test should be O(n²) or faster.

3. **Add a direct solver** in `icfr/solvers.py` (CPU) and
   `icfr/gpu.py:DirectSolver` (GPU). Document the algebraic identity used.

4. **Add to the structural-K table** in `icfr/cost.py`:
   ```python
   STRUCTURAL_K[OpClass.HAMILTONIAN] = 100
   ```

5. **Add a test problem** in `icfr/problems.py` and update `PROBLEM_SPECS`.

6. **Run the full test suite** to verify nothing breaks.

7. **Update the README** with the new class in the taxonomy table.

---

## ➕ Adding a New Nonlinear Problem

1. **Write a constructor** in `icfr/nonlinear.py`:
   ```python
   def make_my_problem(n: int = 200, seed: int = 42) -> tuple[NonlinearMap, torch.Tensor, torch.Tensor]:
       """Problem NL5: x = G(x) where ..."""
       torch.manual_seed(seed)
       # ... build T, b in float64 ...
       def G(x: torch.Tensor) -> torch.Tensor:
           return ...
       x0 = torch.zeros(n, dtype=torch.float64)
       return G, x0, b
   ```

2. **Register it** in `NONLINEAR_PROBLEMS` dict at the bottom of `nonlinear.py`.

3. **Add a description** in `cmd_nonlinear_list` (in `cli.py`).

4. **Verify** it converges: `icfr nonlinear-solve NL5 --n 100 --max-iter 20`.

---

## 📤 Submitting Changes

- **Small PRs are easier to review**. If a change touches >500 lines, consider
  splitting it.
- **Include tests** for any new functionality. Bug fixes should include a
  regression test.
- **Update documentation** (`README.md`, `CHANGELOG.md`) as part of the PR.
- **Be patient** — reviews may take a few days. Reviewer feedback is about
  the code, not about you.

### Review criteria

Reviewers will check:

1. Does the change match the paper's mathematical framework?
2. Are there tests covering the new behavior?
3. Does the code follow the style guide?
4. Are numerical results reproducible (fixed seeds, declared tolerances)?
5. Is the documentation updated?

---

## 🗺 Roadmap

High-impact contributions we'd love help with:

### Theory
- [ ] **Discoverability Frontier** (Task 3.1 of the paper): formal
      characterization of what structures are detectable in O(n²).
- [ ] **Perturbation Analysis** (Task 3.2): robustness of direct solvers
      to "almost structured" operators (T = T_struct + E).
- [ ] **Convergence proof for ICFR-Newton**: when does the Jacobian
      structure preservation hold across iterates?

### Engineering
- [ ] **Distributed memory** (Task 4.3): MPI wrapper for n > 10⁶.
- [ ] **PETSc / SciPy integration** (Task 4.3): ICFR as a preconditioner.
- [ ] **JAX backend**: JAX equivalent of `gpu.py` for TPU support.
- [ ] **Benchmark suite** (Task 1.1): scale to n = 10⁶ on SuiteSparse matrices.

### Applications
- [ ] **PageRank** (Task 5.1): real-world graph datasets.
- [ ] **RL Policy Evaluation** (Task 5.2): OpenAI Gym environments.
- [ ] **Image Deconvolution** (Task 5.3): Toeplitz / circulant blur kernels.

See the [Detailed Research Agenda](README.md#-detailed-research-agenda)
in the README for the full 15-task roadmap.

---

## ❓ Questions?

- **Bug reports**: Open a GitHub issue with the `bug` label. Include the
  exact `icfr` command, the Python/PyTorch versions, and a minimal
  reproducer.
- **Feature requests**: Open an issue with the `enhancement` label.
- **Numerical questions**: Tag your issue with `numerics` — these often
  require discussion, not just code.

Happy hacking! 🚀
