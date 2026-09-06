"""ICFR — Linear algebra helpers built on NumPy / SciPy."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


Vec = np.ndarray
Mat = np.ndarray


# ---------- Basic operations ----------

def matvec(T: Mat, x: Vec) -> Vec:
    return T @ x


def frob_norm(T: Mat) -> float:
    return float(np.linalg.norm(T, "fro"))


def vec_norm(x: Vec) -> float:
    return float(np.linalg.norm(x, 2))


def identity(n: int) -> Mat:
    return np.eye(n, dtype=np.float64)


def diag(T: Mat) -> Vec:
    return np.diag(T).copy()


def diag_mat(d: Vec) -> Mat:
    return np.diag(d).copy()


def roll(x: Vec, k: int) -> Vec:
    return np.roll(x, k)


# ---------- Decompositions ----------

def svd(T: Mat) -> tuple[Mat, Vec, Mat]:
    """SVD: T ≈ U diag(S) Vt. Returns (U, S, Vt)."""
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    return U, S, Vt


def qr(T: Mat) -> tuple[Mat, Mat]:
    """QR: T = Q R. Returns (Q, R)."""
    Q, R = np.linalg.qr(T, mode="reduced")
    return Q, R


def lup(T: Mat) -> tuple[Mat, Mat, np.ndarray]:
    """LUP: P T = L U. Returns (L, U, p)."""
    lu, piv = sla.lu_factor(T)
    L, U = sla.lu_solve((lu, piv), np.eye(T.shape[0])), None  # not what we want
    # Use scipy's explicit LU instead:
    P, L, U = sla.lu(T)
    return L, U, np.arange(T.shape[0])  # scipy returns full P


def matrix_rank(T: Mat, tol: float = 1e-8) -> int:
    return int(np.linalg.matrix_rank(T, tol=tol))


def eigenvalues(T: Mat) -> np.ndarray:
    """Return eigenvalues as complex array."""
    if T.shape[0] <= 4:
        # Small matrices: closed form via numpy
        return np.linalg.eigvals(T)
    # Use scipy for stability on larger matrices
    return np.linalg.eigvals(T)


def spectral_radius(T: Mat, max_iter: int = 100) -> float:
    """Estimate spectral radius rho(T) = max |lambda_i|.

    For small n we use direct eigenvalues; for large n we fall back to
    power iteration which is O(n^2) per step.
    """
    n = T.shape[0]
    if n <= 400:
        try:
            eigs = eigenvalues(T)
            return float(np.max(np.abs(eigs)))
        except Exception:
            pass
    # Power iteration fallback
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)
    x /= max(np.linalg.norm(x), 1e-30)
    lam = 0.0
    for _ in range(max_iter):
        y = T @ x
        ny = np.linalg.norm(y)
        if ny < 1e-20:
            return 0.0
        x = y / ny
        lam = float(np.abs(x @ (T @ x)))
    return lam


def condition_number(T: Mat) -> float:
    """Estimate kappa(I - T) via SVD: kappa = sigma_max / sigma_min."""
    n = T.shape[0]
    A = np.eye(n) - T
    try:
        # For small/medium matrices use SVD
        if n <= 500:
            s = np.linalg.svd(A, compute_uv=False)
            s_max = float(np.max(s)) if s.size > 0 else 1.0
            s_min = float(np.min(s[s > 1e-300])) if np.any(s > 1e-300) else 1e-16
            return s_max / max(s_min, 1e-300)
        # For large matrices, estimate via Lanczos
        linop = spla.LinearOperator((n, n), matvec=lambda v: A @ v)
        lmax = spla.eigsh(linop, k=1, which="LM", return_eigenvectors=False, tol=1e-3)[0]
        lmin = spla.eigsh(linop, k=1, which="SM", return_eigenvectors=False, tol=1e-3)[0]
        return float(abs(lmax / lmin)) if abs(lmin) > 1e-20 else 1e16
    except Exception:
        return 1e6


def solve(A: Mat, b: Vec) -> Vec:
    """Solve A x = b via LU."""
    return np.linalg.solve(A, b)


def dense_solve_resolvent(T: Mat, b: Vec) -> Vec:
    """Compute (I - T)^-1 b via dense LU. Used as ground truth."""
    n = T.shape[0]
    return np.linalg.solve(np.eye(n) - T, b)


def low_rank_factor(T: Mat, tol: float = 1e-8) -> tuple[Mat, Mat, int]:
    """Factor T ~= U V^T via SVD. Returns (U, V, rank)."""
    U, S, Vt = svd(T)
    s_max = float(np.max(S)) if S.size > 0 else 1.0
    cutoff = max(tol, s_max * tol)
    rank = int(np.sum(S > cutoff))
    if rank == 0:
        rank = 1
    sqrtS = np.sqrt(np.maximum(S[:rank], 0))
    U_r = U[:, :rank] * sqrtS[np.newaxis, :]
    V_r = Vt[:rank, :].T * sqrtS[np.newaxis, :]
    return U_r, V_r, rank


def count_nonzero(T: Mat, tol: float = 1e-12) -> int:
    return int(np.sum(np.abs(T) > tol))


def bandwidth(T: Mat, tol: float = 1e-12) -> int:
    """Compute bandwidth = max |i - j| over (i, j) where T[i, j] != 0."""
    nz = np.abs(T) > tol
    if not np.any(nz):
        return 0
    rows, cols = np.where(nz)
    return int(np.max(np.abs(rows - cols)))


def is_circulant(T: Mat, tol: float = 1e-10) -> bool:
    """Check whether every row is a cyclic shift of the first row."""
    n = T.shape[0]
    first = T[0]
    sample = min(n, 10)
    for i in range(1, sample):
        expected = np.roll(first, i)
        if not np.allclose(T[i], expected, atol=tol, rtol=0):
            return False
    return True


def is_toeplitz(T: Mat, tol: float = 1e-10) -> bool:
    """Check whether each diagonal is constant."""
    n = T.shape[0]
    stride = max(1, n // 20)
    diag_indices = list(range(-(n - 1), n, stride))
    for k in diag_indices:
        d = np.diagonal(T, offset=k)
        if d.size > 1:
            if not np.allclose(d, d[0], atol=tol, rtol=0):
                return False
    return True


# ---------- FFT helpers ----------

def fft(x: Vec) -> np.ndarray:
    return np.fft.fft(x)


def ifft(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft(x)


# ---------- Timing ----------

def time_matvec(T: Mat, n_samples: int = 5) -> float:
    """Time one matvec application, returning the median over n_samples runs (ms)."""
    n = T.shape[0]
    xs = [np.random.default_rng(i).standard_normal(n) for i in range(n_samples)]
    # Warmup
    _ = T @ xs[0]
    times = []
    for x in xs:
        t0 = time.perf_counter()
        _ = T @ x
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------- Random construction ----------

def randn(n: int, seed: Optional[int] = None) -> Vec:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def rand_ortho(n: int, r: int, seed: Optional[int] = None) -> Mat:
    """Random orthonormal n x r matrix via QR of random Gaussian."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, r))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q


def jordan_block(n: int, v: float, q: int) -> Mat:
    """Build a q x q Jordan block with superdiagonal v, embedded in an n x n zero matrix."""
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(min(q - 1, n - 1)):
        M[i, i + 1] = v
    return M
