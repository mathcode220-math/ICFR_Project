"""ICFR — Structure Discovery Module.

Implements the cascade of O(n^2) diagnostic tests described in Section 3.2
of the paper. Tests run in order from cheapest/most-specific to most general,
with early termination on first match.
"""
from __future__ import annotations

import time

import numpy as np

from .linalg import (
    frob_norm,
    diag,
    diag_mat,
    identity,
    spectral_radius,
    count_nonzero,
    bandwidth,
    matrix_rank,
    is_circulant,
    is_toeplitz,
    fft,
)
from .types import OpClass, StructInfo


TAU = 1e-10
Q_MAX = 100


def discover_structure(T: np.ndarray) -> StructInfo:
    """Probe an unknown operator T and return its discovered structural class.

    Cascade order (per Section 3.2 of the paper):
      1. SCALAR     -- Frobenius distance from r*I
      2. DIAGONAL   -- Frobenius distance from diag(T)
      3. NILPOTENT  -- zero diagonal + T^q ~= 0 for some q < q_max
      4. LOW_RANK   -- numerical rank <= max(3, n/5)
      5. CIRCULANT  -- every row equals a cyclic roll of the first
      6. TOEPLITZ   -- every diagonal is constant
      7. BANDED     -- sparse with bandwidth < n/10
      8. SPARSE     -- nnz / n^2 < 0.1
      9. GENERAL    -- fallback
    """
    t0 = time.perf_counter()
    n = T.shape[0]
    t_frob = frob_norm(T) or 1.0
    info = StructInfo(
        op_class=OpClass.GENERAL,
        params={},
        spectral_radius=0.0,
        condition_number=1.0,
        frob_norm=t_frob,
        nnz=count_nonzero(T),
        discovery_time_ms=0.0,
    )

    # --- Test 1: SCALAR -----------------------------------------------------
    r = float(T[0, 0])
    scalar_dev = frob_norm(T - r * np.eye(n))
    if scalar_dev < TAU * t_frob:
        info.op_class = OpClass.SCALAR
        info.params = {"r": r}
        _finalize(info, T)
        info.discovery_time_ms = (time.perf_counter() - t0) * 1000
        return info

    # --- Test 2: DIAGONAL ---------------------------------------------------
    d = diag(T)
    off_diag = T - np.diag(d)
    if frob_norm(off_diag) < TAU * t_frob:
        info.op_class = OpClass.DIAGONAL
        info.params = {"diag": d.tolist()}
        _finalize(info, T)
        info.discovery_time_ms = (time.perf_counter() - t0) * 1000
        return info

    # --- Test 3: NILPOTENT --------------------------------------------------
    # Requires numerically zero diagonal.
    if frob_norm(np.diag(d)) < TAU * t_frob:
        P = T.copy()
        q_found = 0
        for q in range(1, min(Q_MAX, n)):
            if frob_norm(P) < TAU * t_frob:
                q_found = q
                break
            P = P @ T
        if q_found > 0:
            info.op_class = OpClass.NILPOTENT
            info.params = {"q": q_found}
            _finalize(info, T)
            info.discovery_time_ms = (time.perf_counter() - t0) * 1000
            return info

    # --- Test 4: LOW_RANK ---------------------------------------------------
    rank = matrix_rank(T, tol=1e-8)
    if rank <= max(3, n // 5):
        info.op_class = OpClass.LOW_RANK
        info.params = {"rank": rank}
        _finalize(info, T)
        info.discovery_time_ms = (time.perf_counter() - t0) * 1000
        return info

    # --- Test 5: CIRCULANT --------------------------------------------------
    if is_circulant(T, tol=TAU):
        info.op_class = OpClass.CIRCULANT
        info.params = {"first_col": T[:, 0].tolist()}
        _finalize(info, T)
        info.discovery_time_ms = (time.perf_counter() - t0) * 1000
        return info

    # --- Test 6: TOEPLITZ ---------------------------------------------------
    if is_toeplitz(T, tol=TAU):
        info.op_class = OpClass.TOEPLITZ
        info.params = {
            "first_col": T[:, 0].tolist(),
            "first_row": T[0, :].tolist(),
        }
        _finalize(info, T)
        info.discovery_time_ms = (time.perf_counter() - t0) * 1000
        return info

    # --- Test 7: SPARSE / BANDED -------------------------------------------
    nnz_ratio = info.nnz / (n * n)
    if nnz_ratio < 0.1:
        bw = bandwidth(T)
        if bw < n / 10:
            info.op_class = OpClass.BANDED
            info.params = {"bandwidth": bw, "nnz_ratio": nnz_ratio}
        else:
            info.op_class = OpClass.SPARSE
            info.params = {"nnz_ratio": nnz_ratio}
        _finalize(info, T)
        info.discovery_time_ms = (time.perf_counter() - t0) * 1000
        return info

    # --- Fallback: GENERAL --------------------------------------------------
    info.op_class = OpClass.GENERAL
    info.params = {"rank": rank, "nnz_ratio": nnz_ratio}
    _finalize(info, T)
    info.discovery_time_ms = (time.perf_counter() - t0) * 1000
    return info


def _finalize(info: StructInfo, T: np.ndarray) -> None:
    """Fill in spectral radius, condition number using class-specific cheap formulas."""
    n = T.shape[0]
    cls = info.op_class

    if cls == OpClass.SCALAR:
        r = info.params.get("r", 0.0)
        info.spectral_radius = abs(r)
        info.condition_number = 1.0 / max(abs(1 - r), 1e-300)
    elif cls == OpClass.DIAGONAL:
        d = np.array(info.params.get("diag", []))
        info.spectral_radius = float(np.max(np.abs(d))) if d.size > 0 else 0.0
        min_abs = float(np.min(np.abs(1 - d))) if d.size > 0 else 1.0
        info.condition_number = 1.0 / max(min_abs, 1e-300)
    elif cls == OpClass.NILPOTENT:
        info.spectral_radius = 0.0
        info.condition_number = 1.0
    elif cls == OpClass.CIRCULANT:
        c = info.params.get("first_col")
        if c is None:
            c = T[:, 0].tolist()
        eig_c = fft(np.array(c, dtype=np.float64))
        info.spectral_radius = float(np.max(np.abs(eig_c))) if eig_c.size > 0 else 0.0
        min_dist = float(np.min(np.abs(1 - eig_c.real))) if eig_c.size > 0 else 1e-12
        info.condition_number = 1.0 / max(min_dist, 1e-12)
    elif cls in (OpClass.TOEPLITZ, OpClass.BANDED, OpClass.SPARSE, OpClass.LOW_RANK):
        info.spectral_radius = spectral_radius(T)
        info.condition_number = 1e3
    else:  # GENERAL
        info.spectral_radius = spectral_radius(T)
        info.condition_number = 1e4
