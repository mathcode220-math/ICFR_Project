"""ICFR — Anderson Acceleration.

Implements AA(m) acceleration for the fixed-point iteration x_{k+1} = T x_k + b.

Reference: Walker & Ni, "Anderson Acceleration for Fixed-Point Iterations",
SIAM J. Numer. Anal. 49(4), 2011.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .types import IterationSnapshot


@dataclass
class AndersonResult:
    x: np.ndarray
    iterations: list[IterationSnapshot]
    iteration_count: int
    converged: bool


def anderson_accelerate(
    T: np.ndarray,
    b: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-10,
    m: int = 5,
    track_history: bool = True,
) -> AndersonResult:
    """Run Anderson-accelerated fixed-point iteration for x = T x + b.

    Uses the standard "type 1" AA formulation with constrained least-squares
    mixing. Falls back to plain Picard iteration (x_{k+1} = g(x_k)) if the
    LS step produces a non-finite iterate or fails to reduce the residual.
    """
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64)            # x_0 = 0 (per the paper)
    g = T @ x + b                                # g(x) = T x + b
    f = g - x                                    # f(x) = g(x) - x
    fx_norm = float(np.linalg.norm(f))

    iterations: list[IterationSnapshot] = []
    if track_history:
        iterations.append(IterationSnapshot(k=0, residual=fx_norm, elapsed_ms=0.0))

    Xs: list[np.ndarray] = []   # past x's
    Fs: list[np.ndarray] = []   # past f's
    t0 = time.perf_counter()

    converged = False
    iter_count = 0
    for k in range(1, max_iter + 1):
        iter_count = k

        # Compute next iterate via AA mixing (if history available) or plain Picard.
        if not Xs:
            x_next = g
        else:
            x_next = _compute_aa_next(Xs, Fs, g, f, m)
            xn = float(np.linalg.norm(x_next))
            if not np.isfinite(xn) or xn > 1e15:
                x_next = g

        g_next = T @ x_next + b
        f_next = g_next - x_next
        fx_next_norm = float(np.linalg.norm(f_next))

        if track_history:
            iterations.append(
                IterationSnapshot(
                    k=k,
                    residual=fx_next_norm,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
            )

        # Update history
        Xs.append(x.copy())
        Fs.append(f.copy())
        if len(Xs) > m:
            Xs.pop(0)
            Fs.pop(0)

        x = x_next
        g = g_next
        f = f_next
        fx_norm = fx_next_norm

        if fx_norm < tol:
            converged = True
            break

    if not track_history and iter_count > 0:
        iterations.append(
            IterationSnapshot(
                k=iter_count,
                residual=fx_norm,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        )

    return AndersonResult(x=x, iterations=iterations, iteration_count=iter_count, converged=converged)


def _compute_aa_next(
    Xs: list[np.ndarray],
    Fs: list[np.ndarray],
    g_current: np.ndarray,
    f_current: np.ndarray,
    m_max: int,
) -> np.ndarray:
    """Compute the next Anderson iterate from history.

    Solves:  alpha = argmin ||sum alpha_i f_i||  s.t.  sum alpha_i = 1
    Then:    x_next = sum alpha_i (x_i + f_i) = sum alpha_i g_i
    """
    m_list = [f_current] + list(reversed(Fs))
    m = min(len(m_list), m_max + 1)
    m_list = m_list[:m]
    n = f_current.shape[0]

    if m < 2:
        return g_current

    # F (n x m): columns are m_list[0..m-1]
    F = np.array(m_list).T  # shape (n, m)
    FTF = F.T @ F

    # Augmented KKT system: [FTF, 1; 1^T, 0] [alpha; lambda] = [0; 1]
    aug = np.zeros((m + 1, m + 1))
    aug[:m, :m] = FTF
    aug[:m, m] = 1.0
    aug[m, :m] = 1.0
    rhs = np.zeros(m + 1)
    rhs[m] = 1.0

    try:
        sol = np.linalg.solve(aug, rhs)
        alpha = sol[:m]
    except np.linalg.LinAlgError:
        alpha = np.full(m, 1.0 / m)

    if not np.all(np.isfinite(alpha)):
        alpha = np.full(m, 1.0 / m)
    s = float(np.sum(alpha))
    if abs(s) > 1e-14:
        alpha = alpha / s
    else:
        alpha = np.full(m, 1.0 / m)

    # x_next = sum alpha_i g_i
    # g_0 = g_current; g_i for i >= 1 = Xs[len-1-(i-1)] + Fs[len-1-(i-1)]
    x_next = np.zeros(n, dtype=np.float64)
    x_next += alpha[0] * g_current
    for i in range(1, m):
        idx = len(Fs) - i
        if idx < 0:
            break
        g_past = Xs[idx] + Fs[idx]
        x_next += alpha[i] * g_past
    return x_next


def fixed_point_iterate(
    T: np.ndarray,
    b: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-10,
    track_history: bool = True,
) -> AndersonResult:
    """Run a plain (un-accelerated) fixed-point iteration for benchmarking."""
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64)
    iterations: list[IterationSnapshot] = []
    t0 = time.perf_counter()
    converged = False
    iter_count = 0
    last_residual = 0.0
    for k in range(1, max_iter + 1):
        iter_count = k
        x_next = T @ x + b
        r = float(np.linalg.norm(x_next - x))
        last_residual = r
        if track_history:
            iterations.append(
                IterationSnapshot(k=k, residual=r, elapsed_ms=(time.perf_counter() - t0) * 1000)
            )
        x = x_next
        if r < tol:
            converged = True
            break
    if not track_history and iter_count > 0:
        iterations.append(
            IterationSnapshot(
                k=iter_count,
                residual=last_residual,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        )
    return AndersonResult(x=x, iterations=iterations, iteration_count=iter_count, converged=converged)
