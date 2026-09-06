"""ICFR — Test Problem Generators.

Generates the canonical test problems (B, C5, C20, D, E) from Section 5.1
of the paper, plus additional problems exercising the SCALAR, DIAGONAL,
BANDED branches.
"""
from __future__ import annotations

import numpy as np

from .linalg import identity, rand_ortho, randn, roll, jordan_block, fft
from .types import OpClass, ProblemSpec


PROBLEM_SPECS: list[ProblemSpec] = [
    ProblemSpec(
        id="A",
        label="A - Scalar",
        op_class=OpClass.SCALAR,
        n=200,
        rho_target=0.9,
        description="T = 0.9 * I. The simplest non-trivial case.",
        construction="r * I with r = 0.9",
    ),
    ProblemSpec(
        id="diag",
        label="Diag - Diagonal",
        op_class=OpClass.DIAGONAL,
        n=200,
        rho_target=0.9,
        description="T = diag(d), with d_i drawn uniformly from [-0.9, 0.9].",
        construction="Uniform random diagonal in [-0.9, 0.9]",
    ),
    ProblemSpec(
        id="B",
        label="B - Low-Rank",
        op_class=OpClass.LOW_RANK,
        n=200,
        rho_target=0.9,
        description="T = 0.9 * U U^T with U orthonormal n x 5. Eigenvalues are exactly {0.9 (x5), 0 (x n-5)}.",
        construction="0.9 * U @ U^T, U orthonormal, rank 5",
    ),
    ProblemSpec(
        id="C5",
        label="C5 - Nilpotent (q=5)",
        op_class=OpClass.NILPOTENT,
        n=200,
        rho_target=0.0,
        description="Jordan block with superdiagonal 0.5, nilpotency index q = 5.",
        construction="5x5 Jordan block (superdiag = 0.5), padded to 200x200",
    ),
    ProblemSpec(
        id="C20",
        label="C20 - Nilpotent (q=20)",
        op_class=OpClass.NILPOTENT,
        n=200,
        rho_target=0.0,
        description="Jordan block with superdiagonal 0.5, nilpotency index q = 20.",
        construction="20x20 Jordan block (superdiag = 0.5), padded to 200x200",
    ),
    ProblemSpec(
        id="D",
        label="D - Circulant",
        op_class=OpClass.CIRCULANT,
        n=200,
        rho_target=0.8,
        description="Circulant matrix with random first column, scaled to rho = 0.8.",
        construction="Random first column scaled so spectral radius = 0.8",
    ),
    ProblemSpec(
        id="band",
        label="Band - Banded",
        op_class=OpClass.BANDED,
        n=200,
        rho_target=0.7,
        description="Tridiagonal matrix with diagonals in [-0.5, 0.5], rho ~ 0.7.",
        construction="Tridiagonal with random entries in [-0.5, 0.5]",
    ),
    ProblemSpec(
        id="E",
        label="E - General Dense",
        op_class=OpClass.GENERAL,
        n=200,
        rho_target=0.9,
        description="SVD-based: U Sigma V^T with Sigma = linspace(0.9, 0.09). No exploitable structure.",
        construction="U Sigma V^T with Sigma = linspace(0.9, 0.09) (full rank)",
    ),
]


def build_problem(spec: ProblemSpec, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build the matrix T and RHS b for a given problem spec."""
    n = spec.n
    b = randn(n, seed + 1)

    if spec.id == "A":
        T = 0.9 * np.eye(n)
        return T, b

    if spec.id == "diag":
        d = 0.9 * np.tanh(randn(n, seed + 2))
        T = np.diag(d)
        return T, b

    if spec.id == "B":
        r = 5
        U = rand_ortho(n, r, seed + 3)
        T = 0.9 * (U @ U.T)
        return T, b

    if spec.id == "C5":
        T = jordan_block(n, 0.5, 5)
        return T, b

    if spec.id == "C20":
        T = jordan_block(n, 0.5, 20)
        return T, b

    if spec.id == "D":
        c = randn(n, seed + 5)
        T = np.array([np.roll(c, i) for i in range(n)])
        # Scale so spectral radius = 0.8 using FFT (circulant eigenvalues = FFT of first column)
        eig_c = fft(c)
        rho = float(np.max(np.abs(eig_c))) or 1.0
        T = T * (0.8 / rho)
        return T, b

    if spec.id == "band":
        T = np.zeros((n, n))
        lo = 0.5 * np.tanh(randn(n, seed + 6))
        dg = 0.5 * np.tanh(randn(n, seed + 7))
        up = 0.5 * np.tanh(randn(n, seed + 8))
        for i in range(n):
            T[i, i] = dg[i]
            if i > 0:
                T[i, i - 1] = lo[i]
            if i < n - 1:
                T[i, i + 1] = up[i]
        return T, b

    if spec.id == "E":
        U = rand_ortho(n, n, seed + 9)
        V = rand_ortho(n, n, seed + 10)
        sigma = np.linspace(0.9, 0.09, n)
        T = (U * sigma[np.newaxis, :]) @ V.T
        return T, b

    raise ValueError(f"Unknown problem id: {spec.id}")
