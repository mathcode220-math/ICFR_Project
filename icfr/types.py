"""ICFR — type definitions and structural enums.

Based on the paper "Iterative-to-Closed-Form Reduction (ICFR):
A Unified Framework for Automated Structure Discovery and Direct
Inversion in Fixed-Point Iterations".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class OpClass(str, Enum):
    """Structural classes discovered by ICFR, ordered from most specific
    (cheapest to invert) to most general."""

    SCALAR = "SCALAR"
    DIAGONAL = "DIAGONAL"
    NILPOTENT = "NILPOTENT"
    LOW_RANK = "LOW_RANK"
    CIRCULANT = "CIRCULANT"
    TOEPLITZ = "TOEPLITZ"
    BANDED = "BANDED"
    SPARSE = "SPARSE"
    GENERAL = "GENERAL"


OP_CLASS_INFO: dict[OpClass, dict[str, str]] = {
    OpClass.SCALAR: {
        "label": "Scalar",
        "description": "T = r * I. Simplest non-trivial case.",
        "direct_solver": "Element-wise scaling",
        "complexity": "O(n)",
        "color": "green",
    },
    OpClass.DIAGONAL: {
        "label": "Diagonal",
        "description": "T = diag(d_1, ..., d_n). Each dim evolves independently.",
        "direct_solver": "Element-wise division",
        "complexity": "O(n)",
        "color": "green",
    },
    OpClass.NILPOTENT: {
        "label": "Nilpotent",
        "description": "T^q = 0 for some q < n. Neumann series terminates after q terms.",
        "direct_solver": "Finite Neumann series",
        "complexity": "O(q * nnz)",
        "color": "cyan",
    },
    OpClass.LOW_RANK: {
        "label": "Low-Rank",
        "description": "T = U V^T with rank r << n. Woodbury reduces to r x r system.",
        "direct_solver": "Woodbury identity",
        "complexity": "O(r^2 n + r^3)",
        "color": "blue",
    },
    OpClass.CIRCULANT: {
        "label": "Circulant",
        "description": "T_{i,j} = c_{(j-i) mod n}. Diagonalized by FFT.",
        "direct_solver": "FFT diagonalization",
        "complexity": "O(n log n)",
        "color": "magenta",
    },
    OpClass.TOEPLITZ: {
        "label": "Toeplitz",
        "description": "T_{i,j} = c_{j-i}. Constant along diagonals.",
        "direct_solver": "Levinson-Durbin / FFT embedding",
        "complexity": "O(n log n) or O(n^2)",
        "color": "magenta",
    },
    OpClass.BANDED: {
        "label": "Banded",
        "description": "Sparse with nonzeros in a narrow band of width b << n.",
        "direct_solver": "Banded LU factorization",
        "complexity": "O(n * b^2)",
        "color": "red",
    },
    OpClass.SPARSE: {
        "label": "Sparse",
        "description": "T has O(n) or O(n*b) nonzeros.",
        "direct_solver": "Sparse LU factorization",
        "complexity": "O(nnz^1.5)",
        "color": "red",
    },
    OpClass.GENERAL: {
        "label": "General Dense",
        "description": "No exploitable structure. Fall back to Anderson acceleration.",
        "direct_solver": "Anderson acceleration (fallback)",
        "complexity": "O(k_anderson * n^2)",
        "color": "white",
    },
}


@dataclass
class StructInfo:
    """Metadata returned by the structure discovery module."""

    op_class: OpClass
    params: dict = field(default_factory=dict)
    spectral_radius: float = 0.0
    condition_number: float = 1.0
    frob_norm: float = 0.0
    nnz: int = 0
    discovery_time_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["op_class"] = self.op_class.value
        return d


@dataclass
class IterationSnapshot:
    """Snapshot of one fixed-point iteration."""

    k: int
    residual: float
    elapsed_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostEstimate:
    """Cost estimates from the three-path protocol."""

    k_spectral: int
    k_empirical: int
    k_structural: int
    k_iter: int
    spread: float
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    t_matvec_ms: float
    c_iter_ms: float
    c_discover_ms: float
    c_direct_ms: float
    c_direct_total_ms: float
    gamma: float
    switch_favorable: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ICFRResult:
    """Final ICFR run result."""

    solution: object  # numpy.ndarray
    relative_error: float
    decision: str  # "DIRECT" | "ITERATIVE"
    discovered_class: OpClass
    struct_info: StructInfo
    cost: CostEstimate
    iterations: list[IterationSnapshot]
    speedup: float
    reason: str
    total_time_ms: float
    ground_truth: Optional[object] = None  # numpy.ndarray

    def to_dict(self, include_solution: bool = False, include_iterations: bool = True) -> dict:
        d = {
            "relative_error": self.relative_error,
            "decision": self.decision,
            "discovered_class": self.discovered_class.value,
            "struct_info": self.struct_info.to_dict(),
            "cost": self.cost.to_dict(),
            "speedup": self.speedup,
            "reason": self.reason,
            "total_time_ms": self.total_time_ms,
        }
        if include_iterations:
            d["iterations"] = [it.to_dict() for it in self.iterations]
        if include_solution:
            import numpy as np
            d["solution"] = np.asarray(self.solution).tolist()
            if self.ground_truth is not None:
                d["ground_truth"] = np.asarray(self.ground_truth).tolist()
        return d


@dataclass
class ProblemSpec:
    """Specification for a test problem."""

    id: str
    label: str
    op_class: OpClass
    n: int
    rho_target: float
    description: str
    construction: str
