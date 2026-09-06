"""ICFR — Iterative-to-Closed-Form Reduction.

A unified framework for automated structure discovery and direct inversion
in fixed-point iterations. Based on the paper of the same name.

Public API:
    from icfr import icfr, ICFROptions, build_problem, PROBLEM_SPECS, OpClass

GPU + nonlinear extensions (requires PyTorch):
    from icfr.gpu import ICFRComputeEngine, ICFRConfig
    from icfr.nonlinear import ICFRNewton, NONLINEAR_PROBLEMS
"""
# Core (NumPy/SciPy only)
from .types import (
    OpClass, StructInfo, CostEstimate, ICFRResult, IterationSnapshot,
    ProblemSpec, OP_CLASS_INFO,
)
from .linalg import (
    spectral_radius, condition_number, matrix_rank, low_rank_factor,
    fft, ifft, solve, dense_solve_resolvent,
)
from .discovery import discover_structure
from .solvers import direct_solve, DirectSolverError, residual
from .anderson import anderson_accelerate, fixed_point_iterate, AndersonResult
from .cost import estimate_cost, KAPPA_MAX
from .core import icfr, ICFROptions, baseline_fixed_point, baseline_anderson
from .problems import PROBLEM_SPECS, build_problem

__version__ = "0.2.0"

# Optional GPU + nonlinear extensions (require PyTorch)
try:
    from .gpu import ICFRConfig, ICFRComputeEngine, HeuristicProber, AdvancedRandomizedSketcher, DirectSolver
    from .nonlinear import ICFRNewton, NonlinearResult, NONLINEAR_PROBLEMS
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = [
    "OpClass", "StructInfo", "CostEstimate", "ICFRResult", "IterationSnapshot",
    "ProblemSpec", "OP_CLASS_INFO",
    "spectral_radius", "condition_number", "matrix_rank", "low_rank_factor",
    "fft", "ifft", "solve", "dense_solve_resolvent",
    "discover_structure", "direct_solve", "DirectSolverError", "residual",
    "anderson_accelerate", "fixed_point_iterate", "AndersonResult",
    "estimate_cost", "KAPPA_MAX",
    "icfr", "ICFROptions", "baseline_fixed_point", "baseline_anderson",
    "PROBLEM_SPECS", "build_problem",
    "__version__", "_HAS_TORCH",
]
if _HAS_TORCH:
    __all__.extend([
        "ICFRConfig", "ICFRComputeEngine", "HeuristicProber",
        "AdvancedRandomizedSketcher", "DirectSolver",
        "ICFRNewton", "NonlinearResult", "NONLINEAR_PROBLEMS",
    ])
