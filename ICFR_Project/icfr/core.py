"""ICFR — Main Controller.

Orchestrates structure discovery, cost estimation, switching, and fallback.
Implements Algorithm: ICFR(T, b, eps) from Section 3.4 of the paper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .anderson import anderson_accelerate, fixed_point_iterate
from .cost import estimate_cost, KAPPA_MAX
from .discovery import discover_structure
from .linalg import dense_solve_resolvent
from .solvers import direct_solve, DirectSolverError
from .types import ICFRResult, IterationSnapshot, StructInfo, CostEstimate


@dataclass
class ICFROptions:
    eps: float = 1e-10
    gamma: float = 2.0
    max_iter: int = 5000
    m: int = 5
    compute_ground_truth: bool = True
    track_iterations: bool = True
    force_direct: bool = False   # If True, use direct solver even if switch is unfavorable


def icfr(T: np.ndarray, b: np.ndarray, opts: Optional[ICFROptions] = None) -> ICFRResult:
    """Run the full ICFR pipeline on (T, b, eps).

    Steps:
      1. DISCOVER  - structure probing
      2. STABILITY GUARD - if kappa > kappa_max, fall back to Anderson
      3. ESTIMATE ITERATIVE COST - three-path protocol
      4. ESTIMATE DIRECT COST
      5. SWITCH DECISION - direct if C_direct < gamma * C_iter and class != GENERAL
      6. EXECUTE - direct solver, or Anderson fallback
    """
    if opts is None:
        opts = ICFROptions()

    t0 = time.perf_counter()

    # 1. DISCOVER
    struct_info = discover_structure(T)

    # Ground truth for accuracy comparison
    ground_truth = None
    if opts.compute_ground_truth:
        try:
            ground_truth = dense_solve_resolvent(T, b)
        except Exception:
            ground_truth = None

    # 2. STABILITY GUARD
    if struct_info.condition_number > KAPPA_MAX:
        aa = anderson_accelerate(
            T, b, max_iter=opts.max_iter, tol=opts.eps, m=opts.m,
            track_history=opts.track_iterations,
        )
        rel_err = _rel_err(aa.x, ground_truth)
        return ICFRResult(
            solution=aa.x,
            relative_error=rel_err,
            decision="ITERATIVE",
            discovered_class=struct_info.op_class,
            struct_info=struct_info,
            cost=_zero_cost(struct_info, opts.gamma),
            iterations=aa.iterations,
            speedup=1.0,
            reason="Stability guard: kappa(I - T) > kappa_max, using Anderson acceleration.",
            total_time_ms=(time.perf_counter() - t0) * 1000,
            ground_truth=ground_truth,
        )

    # 3 + 4. ESTIMATE COSTS
    cost = estimate_cost(T, b, struct_info, eps=opts.eps, gamma=opts.gamma)

    # 5. SWITCH DECISION
    if struct_info.op_class.value == "GENERAL":
        use_iterative = True
        reason = "Operator classified as GENERAL - no structure to exploit; using Anderson acceleration."
    elif opts.force_direct:
        use_iterative = False
        reason = (
            f"Force-direct mode: bypassing cost-benefit switch. "
            f"Using {struct_info.op_class.value} direct solver "
            f"(C_direct={cost.c_direct_total_ms:.2f} ms vs γ·C_iter={cost.c_iter_ms:.2f} ms)."
        )
    elif cost.switch_favorable:
        use_iterative = False
        reason = (
            f"Switch favorable: C_direct ({cost.c_direct_total_ms:.2f} ms) < "
            f"gamma * C_iter ({cost.c_iter_ms:.2f} ms). Using {struct_info.op_class.value} direct solver."
        )
    else:
        use_iterative = True
        reason = (
            f"Switch unfavorable: C_direct ({cost.c_direct_total_ms:.2f} ms) >= "
            f"gamma * C_iter ({cost.c_iter_ms:.2f} ms). Using Anderson acceleration."
        )

    # 6. EXECUTE
    if not use_iterative:
        try:
            t_direct0 = time.perf_counter()
            x = direct_solve(T, b, struct_info)
            direct_ms = (time.perf_counter() - t_direct0) * 1000
            real_cost = estimate_cost(T, b, struct_info, eps=opts.eps, gamma=opts.gamma, direct_solve_ms=direct_ms)
            rel_err = _rel_err(x, ground_truth)
            speedup = real_cost.c_iter_ms / max(real_cost.c_direct_total_ms, 0.001)
            return ICFRResult(
                solution=x,
                relative_error=rel_err,
                decision="DIRECT",
                discovered_class=struct_info.op_class,
                struct_info=struct_info,
                cost=real_cost,
                iterations=[],
                speedup=speedup,
                reason=reason,
                total_time_ms=(time.perf_counter() - t0) * 1000,
                ground_truth=ground_truth,
            )
        except DirectSolverError as e:
            aa = anderson_accelerate(
                T, b, max_iter=opts.max_iter, tol=opts.eps, m=opts.m,
                track_history=opts.track_iterations,
            )
            rel_err = _rel_err(aa.x, ground_truth)
            return ICFRResult(
                solution=aa.x,
                relative_error=rel_err,
                decision="ITERATIVE",
                discovered_class=struct_info.op_class,
                struct_info=struct_info,
                cost=cost,
                iterations=aa.iterations,
                speedup=1.0,
                reason=f"Direct solver failed: {e}. Falling back to Anderson acceleration.",
                total_time_ms=(time.perf_counter() - t0) * 1000,
                ground_truth=ground_truth,
            )
    else:
        aa = anderson_accelerate(
            T, b, max_iter=opts.max_iter, tol=opts.eps, m=opts.m,
            track_history=opts.track_iterations,
        )
        rel_err = _rel_err(aa.x, ground_truth)
        return ICFRResult(
            solution=aa.x,
            relative_error=rel_err,
            decision="ITERATIVE",
            discovered_class=struct_info.op_class,
            struct_info=struct_info,
            cost=cost,
            iterations=aa.iterations,
            speedup=1.0,
            reason=reason,
            total_time_ms=(time.perf_counter() - t0) * 1000,
            ground_truth=ground_truth,
        )


def baseline_fixed_point(T: np.ndarray, b: np.ndarray, *, eps: float = 1e-10, max_iter: int = 5000):
    """Plain fixed-point iteration baseline."""
    return fixed_point_iterate(T, b, max_iter=max_iter, tol=eps, track_history=False)


def baseline_anderson(T: np.ndarray, b: np.ndarray, *, eps: float = 1e-10, max_iter: int = 5000, m: int = 5):
    """Anderson acceleration baseline."""
    return anderson_accelerate(T, b, max_iter=max_iter, tol=eps, m=m, track_history=False)


def _rel_err(x: np.ndarray, ground_truth: Optional[np.ndarray]) -> float:
    if ground_truth is None:
        return 0.0
    diff = x - ground_truth
    nx = float(np.linalg.norm(ground_truth))
    if nx < 1e-20:
        return 0.0
    return float(np.linalg.norm(diff) / nx)


def _zero_cost(struct_info: StructInfo, gamma: float) -> CostEstimate:
    return CostEstimate(
        k_spectral=0, k_empirical=0, k_structural=0, k_iter=0, spread=0.0,
        confidence="LOW", t_matvec_ms=0.0, c_iter_ms=0.0,
        c_discover_ms=struct_info.discovery_time_ms, c_direct_ms=0.0, c_direct_total_ms=0.0,
        gamma=gamma, switch_favorable=False,
    )
