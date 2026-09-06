"""ICFR — Cost Model & Iteration Estimation.

Implements the three-path iteration estimator (Section 3.3) and the
cost-benefit switch decision (Section 2.4).
"""
from __future__ import annotations

import time

import numpy as np

from .linalg import time_matvec
from .types import OpClass, StructInfo, CostEstimate


KAPPA_MAX = 1e12

STRUCTURAL_K: dict[OpClass, int] = {
    OpClass.SCALAR: 1,
    OpClass.DIAGONAL: 1,
    OpClass.LOW_RANK: 50,
    OpClass.NILPOTENT: 5,
    OpClass.CIRCULANT: 1,
    OpClass.TOEPLITZ: 5,
    OpClass.BANDED: 5000,
    OpClass.SPARSE: 5000,
    OpClass.GENERAL: 10000,
}


def _empirical_estimate(T: np.ndarray, b: np.ndarray, eps: float) -> int:
    """Path B: run 5 fixed-point iterations, extrapolate residual reduction."""
    n = b.shape[0]
    x = np.zeros(n)
    r0 = float(np.linalg.norm(b))
    if r0 < 1e-20:
        return 1
    residuals = [r0]
    for _ in range(5):
        x_next = T @ x + b
        residuals.append(float(np.linalg.norm(x_next - x)))
        x = x_next
    r5 = residuals[5] if len(residuals) > 5 else residuals[-1]
    if r5 < 1e-20:
        return 5
    beta = (r5 / r0) ** (1 / 5)
    if beta >= 0.99999:
        return 100000
    return int(np.ceil(np.log(eps / r0) / np.log(beta)))


def estimate_cost(
    T: np.ndarray,
    b: np.ndarray,
    info: StructInfo,
    *,
    eps: float = 1e-10,
    gamma: float = 2.0,
    direct_solve_ms: float = 1.0,
) -> CostEstimate:
    """Compute the full cost estimate using the three-path protocol."""
    # Path A — spectral
    rho = info.spectral_radius
    if 0 < rho < 0.9999:
        k_spectral = int(np.ceil(np.log(eps) / np.log(rho)))
    else:
        k_spectral = 10000

    # Path B — empirical
    k_empirical = _empirical_estimate(T, b, eps)

    # Path C — structural
    k_structural = STRUCTURAL_K[info.op_class]

    # Convergence filter
    estimates = [k for k in (k_spectral, k_empirical, k_structural) if k > 0]
    k_min = min(estimates)
    k_max = max(estimates)
    spread = k_max / max(k_min, 1)

    if spread <= 3:
        confidence = "HIGH"
        blended = [k_spectral, k_empirical, k_structural]
    elif spread <= 10:
        confidence = "MEDIUM"
        blended = [k_spectral, k_empirical, k_structural]
    else:
        confidence = "LOW"
        blended = sorted([k_spectral, k_empirical, k_structural])[:2]

    log_sum = sum(np.log(max(k, 1)) for k in blended)
    k_iter = int(round(np.exp(log_sum / len(blended))))

    t_matvec_ms = time_matvec(T, b.shape[0])
    adjusted_gamma = gamma if confidence == "HIGH" else (3.0 if confidence == "MEDIUM" else 4.0)
    c_iter_ms = adjusted_gamma * k_iter * t_matvec_ms
    c_discover_ms = info.discovery_time_ms
    c_direct_ms = direct_solve_ms
    c_direct_total_ms = c_discover_ms + c_direct_ms

    switch_favorable = (info.op_class != OpClass.GENERAL) and (c_direct_total_ms < c_iter_ms)

    return CostEstimate(
        k_spectral=k_spectral,
        k_empirical=k_empirical,
        k_structural=k_structural,
        k_iter=k_iter,
        spread=spread,
        confidence=confidence,
        t_matvec_ms=t_matvec_ms,
        c_iter_ms=c_iter_ms,
        c_discover_ms=c_discover_ms,
        c_direct_ms=c_direct_ms,
        c_direct_total_ms=c_direct_total_ms,
        gamma=adjusted_gamma,
        switch_favorable=switch_favorable,
    )
