"""ICFR — Direct Solvers.

Implements the structure-specific direct solvers from Section 2.3.
Each solver computes (I - T)^-1 b in the asymptotic complexity listed
in Theorem 2.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .linalg import solve, low_rank_factor, fft, ifft
from .types import OpClass, StructInfo


class DirectSolverError(Exception):
    """Raised when a direct solver detects a numerical breakdown."""


def direct_solve(T: np.ndarray, b: np.ndarray, info: StructInfo) -> np.ndarray:
    """Apply the structure-specific direct solver to compute x = (I - T)^-1 b."""
    cls = info.op_class
    if cls == OpClass.SCALAR:
        return _solve_scalar(b, info.params["r"])
    if cls == OpClass.DIAGONAL:
        return _solve_diagonal(b, np.array(info.params["diag"]))
    if cls == OpClass.LOW_RANK:
        return _solve_low_rank(T, b, info.params["rank"])
    if cls == OpClass.NILPOTENT:
        return _solve_nilpotent(T, b, info.params["q"])
    if cls == OpClass.CIRCULANT:
        return _solve_circulant(b, np.array(info.params["first_col"]))
    if cls == OpClass.TOEPLITZ:
        return _solve_toeplitz(T, b)
    if cls == OpClass.BANDED:
        return _solve_banded(T, b, info.params.get("bandwidth", 5))
    if cls == OpClass.SPARSE:
        return _solve_sparse(T, b)
    # GENERAL
    return _solve_dense(T, b)


# ---------- Per-class solvers ----------

def _solve_scalar(b: np.ndarray, r: float) -> np.ndarray:
    if abs(1 - r) < 1e-14:
        raise DirectSolverError("Scalar operator: 1 - r ~ 0 (rho >= 1)")
    return b / (1.0 - r)


def _solve_diagonal(b: np.ndarray, d: np.ndarray) -> np.ndarray:
    denom = 1.0 - d
    if np.any(np.abs(denom) < 1e-14):
        bad = int(np.argmin(np.abs(denom)))
        raise DirectSolverError(f"Diagonal operator: 1 - d[{bad}] ~ 0 (rho >= 1)")
    return b / denom


def _solve_low_rank(T: np.ndarray, b: np.ndarray, rank: int) -> np.ndarray:
    """Woodbury identity: (I - U V^T)^-1 b = b + U (I_r - V^T U)^-1 V^T b."""
    U, V, r_actual = low_rank_factor(T)
    r = min(rank, r_actual)
    U = U[:, :r]
    V = V[:, :r]

    # M = I_r - V^T U  (r x r)
    M = np.eye(r) - V.T @ U
    # y = V^T b
    y = V.T @ b
    # z = M^-1 y
    try:
        z = np.linalg.solve(M, y)
    except np.linalg.LinAlgError as e:
        raise DirectSolverError(f"Low-rank inner system singular: {e}")
    # out = b + U z
    return b + U @ z


def _solve_nilpotent(T: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Finite Neumann series: (I - T)^-1 b = sum_{k=0}^{q-1} T^k b."""
    result = b.copy()
    current = b.copy()
    for _ in range(1, q):
        current = T @ current
        if np.linalg.norm(current) < 1e-14:
            break
        result = result + current
    return result


def _solve_circulant(b: np.ndarray, first_col: np.ndarray) -> np.ndarray:
    """Circulant: x = IFFT( FFT(b) / (1 - FFT(c)) )."""
    B = fft(b)
    C = fft(first_col)
    denom = 1.0 - C
    if np.any(np.abs(denom) < 1e-14):
        bad = int(np.argmin(np.abs(denom)))
        raise DirectSolverError(f"Circulant: |1 - lambda[{bad}]| < 1e-14 (rho >= 1)")
    x_c = B / denom
    return np.real(ifft(x_c))


def _solve_toeplitz(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Toeplitz: solve via scipy's Levinson-Durbin (O(n^2)) for stability."""
    n = T.shape[0]
    c = T[:, 0]  # first column
    r = T[0, :]  # first row
    try:
        return sla.solve_toeplitz((c, r), b)
    except np.linalg.LinAlgError as e:
        raise DirectSolverError(f"Toeplitz solver failed: {e}")


def _solve_banded(T: np.ndarray, b: np.ndarray, bandwidth: int) -> np.ndarray:
    """Banded: use scipy's banded LU solver."""
    n = T.shape[0]
    A = np.eye(n) - T
    # Convert to banded storage (ab): upper diagonals then lower diagonals
    # scipy.linalg.solve_banded expects (l, u) where l = # sub-diagonals, u = # super-diagonals
    ab = np.zeros((2 * bandwidth + 1, n), dtype=np.float64)
    for i in range(-bandwidth, bandwidth + 1):
        # diag offset i goes into row (bandwidth - i) of ab
        diag_vals = np.diagonal(A, offset=i)
        row = bandwidth - i
        if i >= 0:
            ab[row, :n - i] = diag_vals
        else:
            ab[row, -i:] = diag_vals
    try:
        return sla.solve_banded((bandwidth, bandwidth), ab, b)
    except np.linalg.LinAlgError as e:
        raise DirectSolverError(f"Banded solver failed: {e}")


def _solve_sparse(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sparse: use scipy's sparse LU."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n = T.shape[0]
    A = sp.csc_matrix(np.eye(n) - T)
    try:
        return spla.spsolve(A, b)
    except Exception as e:
        raise DirectSolverError(f"Sparse solver failed: {e}")


def _solve_dense(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """General dense fallback: solve (I - T) x = b via LU."""
    n = T.shape[0]
    A = np.eye(n) - T
    return solve(A, b)


def residual(T: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Compute the residual ||(I - T) x - b|| / ||b||."""
    r = x - T @ x - b
    nb = np.linalg.norm(b)
    if nb < 1e-20:
        return 0.0
    return float(np.linalg.norm(r) / nb)


def direct_solve_op_count(info: StructInfo, n: int) -> int:
    """Estimate direct-solve op count (used by cost model)."""
    cls = info.op_class
    if cls in (OpClass.SCALAR, OpClass.DIAGONAL):
        return n
    if cls == OpClass.LOW_RANK:
        r = info.params.get("rank", 5)
        return r * r * n + r ** 3
    if cls == OpClass.NILPOTENT:
        q = info.params.get("q", 5)
        return q * n * n
    if cls == OpClass.CIRCULANT:
        return int(n * np.log2(max(n, 2)) * 3)
    if cls == OpClass.TOEPLITZ:
        return n * n
    if cls == OpClass.BANDED:
        return n * (info.params.get("bandwidth", 10)) ** 2
    if cls == OpClass.SPARSE:
        return int(info.nnz ** 1.5)
    return n * n * n
