#!/usr/bin/env python3
"""ICFR CLI — Iterative-to-Closed-Form Reduction command-line interface.

Usage:
    icfr solve B [--n N] [--eps EPS] [--no-track]
    icfr benchmark [--json]
    icfr discover <file>          # CSV/JSON/NPY matrix file
    icfr custom <file> [--b RHS]  # Run full ICFR on user matrix
    icfr demo                     # One-shot showcase on Problem B
    icfr list                     # List all built-in problems
    icfr --version

Examples:
    icfr solve B --n 200 --eps 1e-12
    icfr benchmark --json > results.json
    icfr custom my_matrix.npy --b my_rhs.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running both as `python -m icfr.cli` and `python icfr/cli.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from icfr import (
        PROBLEM_SPECS, build_problem, icfr, ICFROptions,
        baseline_fixed_point, baseline_anderson, discover_structure,
        OpClass, OP_CLASS_INFO,
    )
else:
    from . import (
        PROBLEM_SPECS, build_problem, icfr, ICFROptions,
        baseline_fixed_point, baseline_anderson, discover_structure,
        OpClass, OP_CLASS_INFO,
    )


# ----------------------------- ANSI colors -----------------------------

class C:
    """Minimal ANSI color codes (no external deps)."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    @staticmethod
    def color(text: str, color: str) -> str:
        if not sys.stdout.isatty():
            return text
        return f"{color}{text}{C.RESET}"


def class_color(cls: OpClass) -> str:
    return {
        OpClass.SCALAR: C.GREEN, OpClass.DIAGONAL: C.GREEN,
        OpClass.NILPOTENT: C.CYAN, OpClass.LOW_RANK: C.BLUE,
        OpClass.CIRCULANT: C.MAGENTA, OpClass.TOEPLITZ: C.MAGENTA,
        OpClass.BANDED: C.RED, OpClass.SPARSE: C.RED,
        OpClass.GENERAL: C.WHITE,
    }.get(cls, C.WHITE)


# ----------------------------- Formatting helpers -----------------------------

def fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    if not np.isfinite(x):
        return "inf" if x > 0 else "-inf"
    if abs(x) < 1e-4 and x != 0:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def fmt_time(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f} µs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def banner(title: str, subtitle: str = "") -> str:
    line = "═" * 60
    parts = [C.color(line, C.CYAN)]
    parts.append(C.color(f"  {title}", C.BOLD + C.CYAN))
    if subtitle:
        parts.append(C.color(f"  {subtitle}", C.DIM))
    parts.append(C.color(line, C.CYAN))
    return "\n".join(parts)


# ----------------------------- Commands -----------------------------

def cmd_list(args: argparse.Namespace) -> int:
    print(banner("ICFR built-in test problems", f"{len(PROBLEM_SPECS)} problems available"))
    print()
    print(f"  {'ID':<6} {'Label':<28} {'Class':<14} {'n':>5}  {'ρ target':>10}  Description")
    print(f"  {'─' * 6} {'─' * 28} {'─' * 14} {'─' * 5}  {'─' * 10}  {'─' * 40}")
    for spec in PROBLEM_SPECS:
        cls_color = class_color(spec.op_class)
        spec_id = C.color(f"{spec.id:<6}", C.BOLD)
        cls_str = C.color(f"{spec.op_class.value:<14}", cls_color)
        desc_str = C.color(spec.description, C.DIM)
        print(
            f"  {spec_id} "
            f"{spec.label:<28} "
            f"{cls_str} "
            f"{spec.n:>5}  "
            f"{spec.rho_target:>10.3f}  "
            f"{desc_str}"
        )
    print()
    print(C.color("Use `icfr solve <ID>` to run ICFR on any of these.", C.DIM))
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    spec = next((s for s in PROBLEM_SPECS if s.id == args.problem_id), None)
    if spec is None:
        print(C.color(f"Error: unknown problem id '{args.problem_id}'", C.RED), file=sys.stderr)
        print(f"Available: {[s.id for s in PROBLEM_SPECS]}", file=sys.stderr)
        return 2

    n = args.n if args.n is not None else spec.n
    n = max(8, min(n, 1000))  # cap for safety
    effective_spec = type(spec)(
        id=spec.id, label=spec.label, op_class=spec.op_class, n=n,
        rho_target=spec.rho_target, description=spec.description, construction=spec.construction,
    )

    print(banner(f"ICFR · {spec.label}", f"n = {n}, ε = {args.eps:g}"))
    print()
    print(C.color("Problem description:", C.BOLD))
    print(f"  {spec.description}")
    print(f"  {C.color('Construction:', C.DIM)} {spec.construction}")
    print(f"  {C.color('Expected class:', C.DIM)} {C.color(spec.op_class.value, class_color(spec.op_class))}")
    print()

    print(C.color("Building problem…", C.DIM), end=" ", flush=True)
    t0 = time.perf_counter()
    T, b = build_problem(effective_spec, seed=42)
    print(f"done in {fmt_time((time.perf_counter() - t0) * 1000)}")
    print(f"  T: {T.shape} ({T.dtype}), b: {b.shape}")
    print()

    print(C.color("Running ICFR pipeline…", C.BOLD))
    opts = ICFROptions(eps=args.eps, track_iterations=not args.no_track, force_direct=args.force_direct)
    result = icfr(T, b, opts)
    print()

    _print_result_summary(result, expected_class=spec.op_class, n=n)

    # Baselines
    print()
    print(C.color("Baselines:", C.BOLD))
    try:
        t0 = time.perf_counter()
        fp = baseline_fixed_point(T, b, eps=args.eps, max_iter=5000)
        fp_ms = (time.perf_counter() - t0) * 1000
        print(
            f"  {C.color('Plain fixed-point:', C.DIM)} "
            f"{fp.iteration_count} iters, {fmt_time(fp_ms)}, "
            f"final residual = {fp.iterations[-1].residual:.2e}, "
            f"converged = {C.color('yes', C.GREEN) if fp.converged else C.color('no', C.RED)}"
        )
    except Exception as e:
        print(f"  fixed-point failed: {e}")
    try:
        t0 = time.perf_counter()
        aa = baseline_anderson(T, b, eps=args.eps, max_iter=5000, m=5)
        aa_ms = (time.perf_counter() - t0) * 1000
        print(
            f"  {C.color('Anderson acceleration:', C.DIM)} "
            f"{aa.iteration_count} iters, {fmt_time(aa_ms)}, "
            f"final residual = {aa.iterations[-1].residual:.2e}, "
            f"converged = {C.color('yes', C.GREEN) if aa.converged else C.color('no', C.RED)}"
        )
    except Exception as e:
        print(f"  Anderson failed: {e}")

    # Convergence history
    if not args.no_track and len(result.iterations) > 1:
        print()
        _print_convergence(result.iterations)

    # Optional JSON output
    if args.json:
        print()
        print(json.dumps(result.to_dict(include_solution=False), indent=2))

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the full benchmark suite. With --json, emits pure JSON to stdout
    (progress goes to stderr)."""
    use_json = args.json
    if not use_json:
        print(banner("ICFR Benchmark Suite", f"{len(PROBLEM_SPECS)} problems"))
        print()

    rows = []
    t_total = time.perf_counter()
    for spec in PROBLEM_SPECS:
        if not use_json:
            print(f"  {C.color(spec.id, C.BOLD):<6} {spec.label:<32} ", end="", flush=True)
        else:
            print(f"  {spec.id} …", file=sys.stderr, flush=True)
        try:
            T, b = build_problem(spec, seed=42)
            t0 = time.perf_counter()
            result = icfr(T, b, ICFROptions(track_iterations=False))
            icfr_ms = (time.perf_counter() - t0) * 1000

            fp_iters = 0
            aa_iters = 0
            try:
                fp = baseline_fixed_point(T, b, eps=1e-10, max_iter=5000)
                fp_iters = fp.iteration_count
            except Exception:
                pass
            try:
                aa = baseline_anderson(T, b, eps=1e-10, max_iter=5000, m=5)
                aa_iters = aa.iteration_count
            except Exception:
                pass

            correct = result.struct_info.op_class == spec.op_class
            rows.append({
                "problem_id": spec.id,
                "problem_label": spec.label,
                "expected_class": spec.op_class.value,
                "n": spec.n,
                "rho_target": spec.rho_target,
                "discovered_class": result.struct_info.op_class.value,
                "classification_correct": correct,
                "decision": result.decision,
                "relative_error": result.relative_error,
                "speedup": result.speedup,
                "icfr_time_ms": icfr_ms,
                "discovery_time_ms": result.struct_info.discovery_time_ms,
                "k_iter_estimate": result.cost.k_iter,
                "confidence": result.cost.confidence,
                "fixed_point_iters": fp_iters,
                "anderson_iters": aa_iters,
                "spectral_radius": result.struct_info.spectral_radius,
                "condition_number": result.struct_info.condition_number,
                "reason": result.reason,
            })
            mark = C.color("✓", C.GREEN) if correct else C.color("✗", C.RED)
            dec = C.color(
                result.decision,
                C.GREEN if result.decision == "DIRECT" else C.YELLOW,
            )
            if not use_json:
                print(
                    f"{mark} {C.color(result.struct_info.op_class.value, class_color(result.struct_info.op_class)):<14} "
                    f"dec={dec:<10} err={result.relative_error:.1e} "
                    f"speedup={result.speedup:.2f}×  [{fmt_time(icfr_ms)}]"
                )
            else:
                print(f"  → {result.struct_info.op_class.value}, {result.decision}, err={result.relative_error:.1e}", file=sys.stderr)
        except Exception as e:
            if not use_json:
                print(f"{C.color('error:', C.RED)} {e}")
            else:
                print(f"  → error: {e}", file=sys.stderr)
            rows.append({"problem_id": spec.id, "error": str(e)})

    t_total = (time.perf_counter() - t_total) * 1000
    if not use_json:
        print()

    # Summary
    correct_count = sum(1 for r in rows if r.get("classification_correct"))
    direct_count = sum(1 for r in rows if r.get("decision") == "DIRECT")
    iter_count = sum(1 for r in rows if r.get("decision") == "ITERATIVE")
    max_speedup = max((r.get("speedup", 0) for r in rows), default=0)
    avg_err = sum((r.get("relative_error", 0) for r in rows)) / max(len(rows), 1)

    if not use_json:
        print(banner("Summary", f"total time: {fmt_time(t_total)}"))
        print(f"  Classification accuracy : {C.color(f'{correct_count}/{len(rows)} ({correct_count/len(rows)*100:.0f}%)', C.GREEN)}")
        print(f"  Direct decisions        : {direct_count}")
        print(f"  Iterative decisions     : {iter_count}")
        print(f"  Max speedup             : {C.color(f'{max_speedup:.2f}×', C.CYAN)}")
        print(f"  Avg relative error      : {avg_err:.2e}")
        print()
        print(C.color("Full results table:", C.BOLD))
        print()
        _print_results_table(rows)
    else:
        # JSON output (pure JSON to stdout, summary to stderr)
        summary = {
            "total_problems": len(rows),
            "classification_accuracy": correct_count / max(len(rows), 1),
            "direct_decisions": direct_count,
            "iterative_decisions": iter_count,
            "max_speedup": max_speedup,
            "avg_relative_error": avg_err,
            "total_time_ms": t_total,
        }
        print(json.dumps({"rows": rows, "summary": summary}, indent=2))

    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    T = _load_matrix(args.file)
    print(banner("ICFR · Structure Discovery", f"matrix: {T.shape}"))
    print()

    info = discover_structure(T)
    cls_color = class_color(info.op_class)
    print(f"  {C.color('Discovered class:', C.BOLD)} {C.color(info.op_class.value, cls_color + C.BOLD)}")
    op_info = OP_CLASS_INFO[info.op_class]
    print(f"  {C.color('Description:', C.DIM)} {op_info['description']}")
    print(f"  {C.color('Direct solver:', C.DIM)} {op_info['direct_solver']}")
    print(f"  {C.color('Complexity:', C.DIM)} {op_info['complexity']}")
    print()
    print(f"  {C.color('Spectral radius ρ(T):', C.BOLD)} {fmt_num(info.spectral_radius, 6)}")
    print(f"  {C.color('Condition number κ(I-T):', C.BOLD)} {fmt_num(info.condition_number, 2)}")
    print(f"  {C.color('Frobenius norm ‖T‖_F:', C.BOLD)} {fmt_num(info.frob_norm, 2)}")
    print(f"  {C.color('Nonzeros (nnz):', C.BOLD)} {info.nnz:,}")
    print(f"  {C.color('Discovery time:', C.BOLD)} {fmt_time(info.discovery_time_ms)}")
    print()

    if info.params:
        print(C.color("Parameters:", C.BOLD))
        for k, v in list(info.params.items())[:8]:
            if isinstance(v, list):
                v_str = f"[{len(v)} values]" if len(v) > 5 else f"{v}"
            else:
                v_str = str(v)
            print(f"  {k}: {C.color(v_str, C.DIM)}")

    if args.json:
        print()
        print(json.dumps(info.to_dict(), indent=2, default=str))

    return 0


def cmd_custom(args: argparse.Namespace) -> int:
    T = _load_matrix(args.file)
    if T.shape[0] != T.shape[1]:
        print(C.color(f"Error: matrix must be square, got {T.shape}", C.RED), file=sys.stderr)
        return 2
    n = T.shape[0]

    if args.b is not None:
        b = _load_matrix(args.b)
        if b.ndim > 1:
            b = b.ravel()
        if b.shape[0] != n:
            print(C.color(f"Error: b has length {b.shape[0]}, expected {n}", C.RED), file=sys.stderr)
            return 2
    else:
        # Default RHS: ones
        b = np.ones(n)

    print(banner("ICFR · Custom Matrix", f"T: {T.shape}, b: {b.shape}, ε = {args.eps:g}"))
    print()

    opts = ICFROptions(eps=args.eps, track_iterations=not args.no_track, force_direct=args.force_direct)
    result = icfr(T, b, opts)

    _print_result_summary(result, expected_class=None, n=n)

    if not args.no_track and len(result.iterations) > 1:
        print()
        _print_convergence(result.iterations)

    # Save solution
    if args.output:
        np.save(args.output, np.asarray(result.solution))
        print()
        print(C.color(f"Solution saved to: {args.output}", C.GREEN))

    if args.json:
        print()
        print(json.dumps(result.to_dict(include_solution=True), indent=2))

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """One-shot showcase: run ICFR on Problem B (low-rank)."""
    args.problem_id = "B"
    args.n = 200
    args.eps = 1e-10
    args.no_track = False
    args.force_direct = False
    args.json = False
    return cmd_solve(args)


# ============================================================================
# GPU-accelerated commands (PyTorch backend)
# ============================================================================

def _import_gpu():
    """Lazy-import torch + GPU engine to avoid hard dep at CLI startup."""
    try:
        import torch
        from .gpu import ICFRConfig, ICFRComputeEngine
        return torch, ICFRConfig, ICFRComputeEngine
    except ImportError as e:
        print(C.color(f"Error: PyTorch is required for GPU commands. Install with `pip install torch`.", C.RED), file=sys.stderr)
        print(f"  (Original error: {e})", file=sys.stderr)
        sys.exit(2)


def _import_nonlinear():
    """Lazy-import nonlinear module."""
    try:
        from .nonlinear import ICFRNewton, NONLINEAR_PROBLEMS
        from .gpu import ICFRConfig, ICFRComputeEngine
        return ICFRNewton, NONLINEAR_PROBLEMS, ICFRConfig, ICFRComputeEngine
    except ImportError as e:
        print(C.color(f"Error: PyTorch is required for nonlinear commands. Install with `pip install torch`.", C.RED), file=sys.stderr)
        print(f"  (Original error: {e})", file=sys.stderr)
        sys.exit(2)


def cmd_gpu_solve(args: argparse.Namespace) -> int:
    """Run GPU-accelerated ICFR on a built-in problem."""
    torch, ICFRConfig, ICFRComputeEngine = _import_gpu()
    spec = next((s for s in PROBLEM_SPECS if s.id == args.problem_id), None)
    if spec is None:
        print(C.color(f"Error: unknown problem id '{args.problem_id}'", C.RED), file=sys.stderr)
        return 2

    n = args.n if args.n is not None else spec.n
    n = max(8, min(n, 10000))

    # Build problem using NumPy then move to torch
    effective_spec = type(spec)(
        id=spec.id, label=spec.label, op_class=spec.op_class, n=n,
        rho_target=spec.rho_target, description=spec.description, construction=spec.construction,
    )
    T_np, b_np = build_problem(effective_spec, seed=42)
    T = torch.from_numpy(T_np).to(torch.float64)
    b = torch.from_numpy(b_np).to(torch.float64)

    actual_device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(banner(f"ICFR GPU · {spec.label}", f"n = {n}, ε = 1e-10, device = {actual_device}"))
    print()

    config = ICFRConfig(device=actual_device, tau=args.tau)
    print(f"  {C.color('Configured device:', C.DIM)} {C.color(actual_device, C.GREEN if actual_device == 'cuda' else C.YELLOW)}")
    if actual_device == "cpu" and args.device == "cuda":
        print(f"  {C.color('Note: CUDA not available; falling back to CPU.', C.YELLOW)}")
    print()

    engine = ICFRComputeEngine(config)
    print(C.color("Running GPU pipeline…", C.BOLD))
    result = engine.solve_linear_full(T, b)
    print()

    # Print summary
    print(C.color("─" * 60, C.DIM))
    dec_color = C.GREEN if result.decision == "DIRECT" else C.YELLOW
    print(f"  {C.color('Decision:', C.BOLD)}      {C.color(result.decision, dec_color + C.BOLD)}")
    print(f"  {C.color('Solver path:', C.DIM)}   {result.solver_path}")
    print(f"  {C.color('Discovered:', C.DIM)}    {result.discovered_class}")
    print(f"  {C.color('ρ(T):', C.DIM)}          {result.spectral_radius:.6f}")
    print(f"  {C.color('‖T‖_F:', C.DIM)}         {result.frob_norm:.4f}")
    print(f"  {C.color('nnz:', C.DIM)}           {result.nnz:,}")
    print(f"  {C.color('Probe time:', C.DIM)}    {fmt_time(result.discovery_time_ms)}")
    print(f"  {C.color('Solve time:', C.DIM)}    {fmt_time(result.solve_time_ms)}")
    print(f"  {C.color('Total time:', C.DIM)}    {fmt_time(result.total_time_ms)}")
    print()

    # Verify accuracy against ground truth (on CPU)
    try:
        T_ground = T.numpy()
        b_ground = b.numpy()
        x_star = np.linalg.solve(np.eye(n) - T_ground, b_ground)
        x_hat = result.solution.numpy()
        rel_err = float(np.linalg.norm(x_hat - x_star) / max(np.linalg.norm(x_star), 1e-20))
        err_color = C.GREEN if rel_err < 1e-10 else (C.YELLOW if rel_err < 1e-6 else C.RED)
        print(f"  {C.color('Relative error:', C.DIM)} {C.color(f'{rel_err:.2e}', err_color)}")
        if rel_err < 1e-10:
            print(f"  {C.color('→ Machine precision achieved', C.GREEN)}")
    except Exception as e:
        print(f"  {C.color('Ground truth check failed:', C.RED)} {e}")

    if args.json:
        out = {
            "problem_id": spec.id,
            "n": n,
            "device": actual_device,
            "decision": result.decision,
            "solver_path": result.solver_path,
            "discovered_class": result.discovered_class,
            "spectral_radius": result.spectral_radius,
            "frob_norm": result.frob_norm,
            "nnz": result.nnz,
            "discovery_time_ms": result.discovery_time_ms,
            "solve_time_ms": result.solve_time_ms,
            "total_time_ms": result.total_time_ms,
        }
        print()
        print(json.dumps(out, indent=2))

    return 0


def cmd_gpu_benchmark(args: argparse.Namespace) -> int:
    """Run GPU benchmark across all 8 problems at large n."""
    torch, ICFRConfig, ICFRComputeEngine = _import_gpu()
    n = max(50, min(args.n, 10000))
    actual_device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    use_json = args.json
    if not use_json:
        print(banner("ICFR GPU Benchmark Suite", f"n = {n}, device = {actual_device}, {len(PROBLEM_SPECS)} problems"))
        print()

    config = ICFRConfig(device=actual_device)
    engine = ICFRComputeEngine(config)

    rows = []
    t_total = time.perf_counter()
    for spec in PROBLEM_SPECS:
        if not use_json:
            print(f"  {C.color(spec.id, C.BOLD):<6} {spec.label:<32} ", end="", flush=True)
        else:
            print(f"  {spec.id} …", file=sys.stderr, flush=True)
        try:
            effective_spec = type(spec)(
                id=spec.id, label=spec.label, op_class=spec.op_class, n=n,
                rho_target=spec.rho_target, description=spec.description, construction=spec.construction,
            )
            T_np, b_np = build_problem(effective_spec, seed=42)
            T = torch.from_numpy(T_np).to(torch.float64)
            b = torch.from_numpy(b_np).to(torch.float64)

            result = engine.solve_linear_full(T, b)
            row = {
                "problem_id": spec.id,
                "problem_label": spec.label,
                "expected_class": spec.op_class.value,
                "n": n,
                "device": actual_device,
                "decision": result.decision,
                "solver_path": result.solver_path,
                "discovered_class": result.discovered_class,
                "spectral_radius": result.spectral_radius,
                "frob_norm": result.frob_norm,
                "nnz": result.nnz,
                "discovery_time_ms": result.discovery_time_ms,
                "solve_time_ms": result.solve_time_ms,
                "total_time_ms": result.total_time_ms,
            }
            rows.append(row)
            if not use_json:
                dec = C.color(result.decision, C.GREEN if result.decision == "DIRECT" else C.YELLOW)
                print(
                    f"{result.discovered_class:<14} dec={dec:<10} "
                    f"[probe={fmt_time(result.discovery_time_ms)}, solve={fmt_time(result.solve_time_ms)}, total={fmt_time(result.total_time_ms)}]"
                )
            else:
                print(f"  → {result.discovered_class}, {result.decision}, total={result.total_time_ms:.1f}ms", file=sys.stderr)
        except Exception as e:
            if not use_json:
                print(f"{C.color('error:', C.RED)} {e}")
            else:
                print(f"  → error: {e}", file=sys.stderr)
            rows.append({"problem_id": spec.id, "error": str(e)})

    t_total = (time.perf_counter() - t_total) * 1000
    if not use_json:
        print()
        print(banner("Summary", f"total time: {fmt_time(t_total)}"))
        direct = sum(1 for r in rows if r.get("decision") == "DIRECT")
        iterative = sum(1 for r in rows if r.get("decision") == "ITERATIVE")
        avg_solve = sum(r.get("solve_time_ms", 0) for r in rows) / max(len(rows), 1)
        print(f"  Direct decisions     : {direct}")
        print(f"  Iterative decisions  : {iterative}")
        print(f"  Avg solve time       : {fmt_time(avg_solve)}")
        print()
        # Compact results table
        print(f"  {'Problem':<10} {'Class':<14} {'Decision':<11} {'Probe':>10} {'Solve':>10} {'Total':>10}")
        print(f"  {'─' * 65}")
        for r in rows:
            if "error" in r:
                print(f"  {r['problem_id']:<10} {C.color('ERROR: ' + r['error'], C.RED)}")
                continue
            dec = r["decision"]
            dec_col = C.GREEN if dec == "DIRECT" else C.YELLOW
            print(
                f"  {r['problem_id']:<10} "
                f"{r['discovered_class']:<14} "
                f"{C.color(dec, dec_col):<11} "
                f"{fmt_time(r['discovery_time_ms']):>10} "
                f"{fmt_time(r['solve_time_ms']):>10} "
                f"{fmt_time(r['total_time_ms']):>10}"
            )
    else:
        print(json.dumps({"rows": rows, "device": actual_device, "n": n, "total_time_ms": t_total}, indent=2))

    return 0


# ============================================================================
# Nonlinear commands
# ============================================================================

def cmd_nonlinear_list(args: argparse.Namespace) -> int:
    """List built-in nonlinear problems."""
    ICFRNewton, NONLINEAR_PROBLEMS, _, _ = _import_nonlinear()
    print(banner("ICFR Nonlinear Test Problems", f"{len(NONLINEAR_PROBLEMS)} problems"))
    print()
    print(f"  {'ID':<20} {'Description'}")
    print(f"  {'─' * 20} {'─' * 60}")
    descriptions = {
        "NL1_tanh": "x = tanh(A x + b), A low-rank. Jacobian = (1-tanh^2) ⊙ A is low-rank → Woodbury.",
        "NL2_softmax": "x = softmax(W x + b). Jacobian is diagonal minus low-rank.",
        "NL3_quadratic": "x = 0.5 (A x² + b) element-wise. Jacobian = A diag(x) is low-rank.",
        "NL4_circulant": "x = sigmoid(C x + b), C circulant. Jacobian is circulant → FFT.",
    }
    for pid in NONLINEAR_PROBLEMS:
        print(f"  {C.color(pid, C.BOLD):<20} {descriptions.get(pid, '(no description)')}")
    print()
    print(C.color("Use `icfr nonlinear-solve <ID>` to run ICFR-Newton.", C.DIM))
    return 0


def cmd_nonlinear_solve(args: argparse.Namespace) -> int:
    """Solve a nonlinear fixed-point problem via ICFR-Newton."""
    ICFRNewton, NONLINEAR_PROBLEMS, ICFRConfig, ICFRComputeEngine = _import_nonlinear()
    import torch as _torch

    if args.problem_id not in NONLINEAR_PROBLEMS:
        print(C.color(f"Error: unknown nonlinear problem '{args.problem_id}'", C.RED), file=sys.stderr)
        print(f"Available: {list(NONLINEAR_PROBLEMS.keys())}", file=sys.stderr)
        return 2

    actual_device = "cuda" if (args.device == "cuda" and _torch.cuda.is_available()) else "cpu"
    descriptions = {
        "NL1_tanh": "x = tanh(A x + b), A low-rank (rank 5)",
        "NL2_softmax": "x = softmax(W x + b)",
        "NL3_quadratic": "x = 0.5 (A x² + b) element-wise, A low-rank (rank 3)",
        "NL4_circulant": "x = sigmoid(C x + b), C circulant",
    }
    print(banner(f"ICFR-Newton · {args.problem_id}", f"n = {args.n}, max_iter = {args.max_iter}, device = {actual_device}"))
    print()
    print(f"  {C.color('Problem:', C.DIM)} {descriptions.get(args.problem_id, '(no description)')}")
    print()

    # Build problem
    make_fn = NONLINEAR_PROBLEMS[args.problem_id]
    G, x0, b = make_fn(n=args.n, seed=42)

    # Setup engine
    config = ICFRConfig(device=actual_device)
    engine = ICFRComputeEngine(config)
    newton = ICFRNewton(
        engine,
        max_iter=args.max_iter,
        tol=args.tol,
        track_history=True,
    )

    print(C.color("Running ICFR-Newton…", C.BOLD))
    result = newton.solve(G, x0)
    print()

    # Summary
    print(C.color("─" * 60, C.DIM))
    conv_color = C.GREEN if result.converged else C.RED
    print(f"  {C.color('Converged:', C.BOLD)}      {C.color('yes' if result.converged else 'no', conv_color)}")
    print(f"  {C.color('Iterations:', C.DIM)}     {result.iterations}")
    print(f"  {C.color('Final residual:', C.DIM)} {result.final_residual:.2e}")
    print(f"  {C.color('Total time:', C.DIM)}     {fmt_time(result.total_time_ms)}")
    print()

    # Per-iteration Jacobian classes
    if result.jacobian_classes:
        print(C.color("Jacobian structure discovered per Newton iterate:", C.BOLD))
        for k, cls in enumerate(result.jacobian_classes, 1):
            cls_color_map = {
                "LOW_RANK": C.BLUE, "NILPOTENT": C.CYAN, "CIRCULANT": C.MAGENTA,
                "DIAGONAL": C.GREEN, "SCALAR": C.GREEN, "BANDED": C.RED,
                "SPARSE": C.RED, "GENERAL": C.WHITE,
            }
            col = cls_color_map.get(cls, C.WHITE)
            print(f"  iter {k}: {C.color(cls, col)}  ({result.paths[k-1]})")
        print()

    # Convergence history
    if result.per_iteration:
        print(C.color("Convergence History:", C.BOLD))
        print()
        print(f"  {'k':>4}  {'‖r‖':>14}  {'‖x‖':>14}")
        print(f"  {'─' * 4}  {'─' * 14}  {'─' * 14}")
        for it in result.per_iteration:
            res_col = C.GREEN if it["residual"] < args.tol else (C.YELLOW if it["residual"] < 1e-6 else C.RED)
            res_str = f"{it['residual']:>14.4e}"
            print(f"  {it['k']:>4}  {C.color(res_str, res_col)}  {it['x_norm']:>14.4e}")

    if args.json:
        out = {
            "problem_id": args.problem_id,
            "n": args.n,
            "device": actual_device,
            "converged": result.converged,
            "iterations": result.iterations,
            "final_residual": result.final_residual,
            "total_time_ms": result.total_time_ms,
            "jacobian_classes": result.jacobian_classes,
            "paths": result.paths,
            "per_iteration": result.per_iteration,
        }
        print()
        print(json.dumps(out, indent=2))

    return 0


# ----------------------------- Helpers -----------------------------

def _print_result_summary(result, *, expected_class: Optional[OpClass], n: int) -> None:
    cls_color = class_color(result.discovered_class)
    print(C.color("─" * 60, C.DIM))
    if result.decision == "DIRECT":
        dec_color = C.GREEN
    else:
        dec_color = C.YELLOW
    print(
        f"  {C.color('ICFR Decision:', C.BOLD)} "
        f"{C.color(result.decision, dec_color + C.BOLD)} "
        f"{C.color('via', C.DIM)} "
        f"{C.color(result.discovered_class.value, cls_color + C.BOLD)}"
    )
    if expected_class is not None:
        correct = result.discovered_class == expected_class
        mark = C.color("✓ correct", C.GREEN) if correct else C.color("✗ mismatch", C.RED)
        print(f"  {C.color('Classification:', C.DIM)} {mark} (expected {expected_class.value})")
    print()
    print(f"  {C.color('Reason:', C.DIM)} {result.reason}")
    print()
    print(f"  {C.color('Total time:', C.DIM)} {fmt_time(result.total_time_ms)}")
    print(f"  {C.color('Speedup:', C.DIM)} {C.color(f'{result.speedup:.2f}×', C.CYAN)} vs iterative baseline")
    err_color = C.GREEN if result.relative_error < 1e-10 else (C.YELLOW if result.relative_error < 1e-6 else C.RED)
    print(f"  {C.color('Relative error:', C.DIM)} {C.color(f'{result.relative_error:.2e}', err_color)}")
    if result.relative_error < 1e-10:
        print(f"  {C.color('→ Machine precision achieved', C.GREEN)}")
    print()

    # Discovered structure
    print(C.color("Discovered Structure:", C.BOLD))
    info = result.struct_info
    print(f"  {'Class':<26} {C.color(info.op_class.value, cls_color)}")
    print(f"  {'Spectral radius ρ(T)':<26} {fmt_num(info.spectral_radius, 6)}")
    print(f"  {'Condition number κ(I-T)':<26} {fmt_num(info.condition_number, 2)}")
    print(f"  {'Frobenius norm ‖T‖_F':<26} {fmt_num(info.frob_norm, 2)}")
    print(f"  {'Nonzeros (nnz)':<26} {info.nnz:,}")
    print(f"  {'Discovery time':<26} {fmt_time(info.discovery_time_ms)}")
    print()

    # Cost estimate
    print(C.color("Cost Estimate (3-Path Protocol):", C.BOLD))
    c = result.cost
    print(f"  {'Path A — Spectral k_A':<30} {c.k_spectral:>10,}")
    print(f"  {'Path B — Empirical k_B':<30} {c.k_empirical:>10,}")
    print(f"  {'Path C — Structural k_C':<30} {c.k_structural:>10,}")
    print(f"  {C.color('─' * 42, C.DIM)}")
    print(f"  {'Blended k_iter':<30} {c.k_iter:>10,}")
    print(f"  {'Spread ratio k_max/k_min':<30} {c.spread:>10.2f}")
    conf_color = {"HIGH": C.GREEN, "MEDIUM": C.YELLOW, "LOW": C.RED}.get(c.confidence, C.WHITE)
    print(f"  {'Confidence':<30} {C.color(c.confidence, conf_color)}")
    print()
    print(f"  {'Iterative cost C_iter':<30} {fmt_time(c.c_iter_ms):>12}  {C.color(f'(γ={c.gamma:.0f} · k · t_mv)', C.DIM)}")
    print(f"  {'Direct cost C_direct':<30} {fmt_time(c.c_direct_total_ms):>12}  {C.color('(discover + solve)', C.DIM)}")
    switch_color = C.GREEN if c.switch_favorable else C.YELLOW
    print(
        f"  {'Switch':<30} "
        f"{C.color('direct < γ·iter → DIRECT' if c.switch_favorable else 'direct ≥ γ·iter → ITERATIVE', switch_color)}"
    )


def _print_convergence(iterations) -> None:
    print(C.color("Convergence History:", C.BOLD))
    if len(iterations) == 0:
        print(f"  {C.color('(no iterations — direct solver was used)', C.DIM)}")
        return
    print()
    # ASCII log-scale plot
    residuals = [it.residual for it in iterations]
    log_res = [np.log10(max(r, 1e-16)) for r in residuals]
    y_max = max(log_res)
    y_min = min(log_res)
    y_range = max(y_max - y_min, 1)
    plot_w = 40
    print(f"  {'k':>5}  {'residual':>14}  {'log10':>7}  {'plot':<{plot_w + 2}}")
    print(f"  {'─' * 5}  {'─' * 14}  {'─' * 7}  {'─' * (plot_w + 2)}")
    # Sample iterations to fit ~20 rows
    step = max(1, len(iterations) // 20)
    for i, it in enumerate(iterations):
        if i % step != 0 and i != len(iterations) - 1:
            continue
        bar_len = int(((log_res[i] - y_min) / y_range) * plot_w)
        bar = "█" * bar_len
        color = C.GREEN if it.residual < 1e-10 else (C.YELLOW if it.residual < 1e-6 else C.RED)
        print(
            f"  {it.k:>5}  {it.residual:>14.4e}  {log_res[i]:>+7.1f}  "
            f"{C.color(bar, color)}"
        )


def _print_results_table(rows: list[dict]) -> None:
    headers = ["Problem", "Expected", "Discovered", "ρ(T)", "k̂_iter", "Decision",
               "FP", "AA", "Speedup", "Rel. error", "Conf."]
    widths = [10, 12, 12, 8, 8, 11, 6, 6, 9, 12, 6]
    print(f"  {' '.join(h.ljust(w) for h, w in zip(headers, widths))}")
    print(f"  {'─' * sum(widths + [len(widths) - 1])}")
    for r in rows:
        if "error" in r:
            print(f"  {r['problem_id']}: {C.color('ERROR: ' + r['error'], C.RED)}")
            continue
        cls = OpClass(r["discovered_class"])
        cls_col = class_color(cls)
        correct = r["classification_correct"]
        mark = C.color("✓", C.GREEN) if correct else C.color("✗", C.RED)
        dec = r["decision"]
        dec_col = C.GREEN if dec == "DIRECT" else C.YELLOW
        sp = r["speedup"]
        sp_col = C.CYAN if sp >= 1 else C.DIM
        err = r["relative_error"]
        err_col = C.GREEN if err < 1e-10 else (C.YELLOW if err < 1e-6 else C.RED)
        conf = r["confidence"]
        conf_col = {"HIGH": C.GREEN, "MEDIUM": C.YELLOW, "LOW": C.RED}.get(conf, C.WHITE)
        cells = [
            r["problem_label"][:10].ljust(10),
            C.color(r["expected_class"][:12].ljust(12), C.DIM),
            C.color((r["discovered_class"] + " " + mark)[:12].ljust(12), cls_col),
            f"{r['spectral_radius']:>8.3f}",
            f"{r['k_iter_estimate']:>8,}",
            C.color(dec.ljust(11), dec_col),
            f"{r['fixed_point_iters']:>6}",
            f"{r['anderson_iters']:>6}",
            C.color(f"{sp:>7.2f}×  ", sp_col),
            C.color(f"{err:>12.2e}", err_col),
            C.color(conf, conf_col),
        ]
        print("  " + " ".join(cells))


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = p.suffix.lower()
    if suffix in (".npy", ".npz"):
        if suffix == ".npz":
            data = np.load(p)
            key = list(data.keys())[0]
            return data[key]
        return np.load(p)
    if suffix in (".csv", ".txt"):
        return np.loadtxt(p, delimiter="," if suffix == ".csv" else None)
    if suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Take first array-valued field
            for k, v in data.items():
                if isinstance(v, list):
                    return np.array(v, dtype=np.float64)
        return np.array(data, dtype=np.float64)
    # Try numpy load as last resort
    return np.load(p)


# ----------------------------- Argparse setup -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="icfr",
        description="ICFR — Iterative-to-Closed-Form Reduction CLI. "
                    "Automated structure discovery and direct inversion in fixed-point iterations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version="icfr 0.2.0")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # list
    sp_list = sub.add_parser("list", help="List all built-in test problems")
    sp_list.set_defaults(func=cmd_list)

    # solve
    sp_solve = sub.add_parser("solve", help="Run ICFR on a built-in problem")
    sp_solve.add_argument("problem_id", help="Problem ID (e.g. B, C5, D)")
    sp_solve.add_argument("--n", type=int, default=None, help="Override problem dimension (default: from spec)")
    sp_solve.add_argument("--eps", type=float, default=1e-10, help="Convergence tolerance (default: 1e-10)")
    sp_solve.add_argument("--no-track", action="store_true", help="Don't track per-iteration history")
    sp_solve.add_argument("--force-direct", action="store_true", help="Bypass cost-benefit switch; always use direct solver (for demo)")
    sp_solve.add_argument("--json", action="store_true", help="Also print JSON result")
    sp_solve.set_defaults(func=cmd_solve)

    # benchmark
    sp_bench = sub.add_parser("benchmark", help="Run the full benchmark suite (all 8 problems)")
    sp_bench.add_argument("--json", action="store_true", help="Output JSON instead of table")
    sp_bench.set_defaults(func=cmd_benchmark)

    # discover
    sp_disc = sub.add_parser("discover", help="Run only structure discovery on a matrix file")
    sp_disc.add_argument("file", help="Matrix file (.npy, .npz, .csv, .json)")
    sp_disc.add_argument("--json", action="store_true", help="Also print JSON result")
    sp_disc.set_defaults(func=cmd_discover)

    # custom
    sp_cust = sub.add_parser("custom", help="Run full ICFR on a user-supplied matrix")
    sp_cust.add_argument("file", help="Matrix file (.npy, .npz, .csv, .json)")
    sp_cust.add_argument("--b", help="RHS vector file (default: ones)")
    sp_cust.add_argument("--eps", type=float, default=1e-10, help="Convergence tolerance")
    sp_cust.add_argument("--no-track", action="store_true", help="Don't track per-iteration history")
    sp_cust.add_argument("--force-direct", action="store_true", help="Bypass cost-benefit switch; always use direct solver (for demo)")
    sp_cust.add_argument("--output", "-o", help="Save solution to .npy file")
    sp_cust.add_argument("--json", action="store_true", help="Also print JSON result")
    sp_cust.set_defaults(func=cmd_custom)

    # demo
    sp_demo = sub.add_parser("demo", help="One-shot showcase on Problem B (low-rank, n=200)")
    sp_demo.set_defaults(func=cmd_demo)

    # ===== GPU commands =====

    # gpu-solve
    sp_gpu = sub.add_parser("gpu-solve", help="Run GPU-accelerated ICFR on a built-in problem (PyTorch backend)")
    sp_gpu.add_argument("problem_id", help="Problem ID (e.g. B, C5, D)")
    sp_gpu.add_argument("--n", type=int, default=None, help="Override problem dimension")
    sp_gpu.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device (default: cuda, falls back to cpu)")
    sp_gpu.add_argument("--tau", type=float, default=1e-8, help="Classification threshold")
    sp_gpu.add_argument("--json", action="store_true", help="Also print JSON result")
    sp_gpu.set_defaults(func=cmd_gpu_solve)

    # gpu-benchmark
    sp_gpube = sub.add_parser("gpu-benchmark", help="Run GPU benchmark across all 8 problems")
    sp_gpube.add_argument("--n", type=int, default=2000, help="Dimension (default: 2000)")
    sp_gpube.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device")
    sp_gpube.add_argument("--json", action="store_true", help="Output JSON instead of table")
    sp_gpube.set_defaults(func=cmd_gpu_benchmark)

    # ===== Nonlinear commands =====

    # nonlinear-solve
    sp_nl = sub.add_parser("nonlinear-solve", help="Solve a nonlinear fixed-point problem x = G(x) via ICFR-Newton")
    sp_nl.add_argument("problem_id", help="Nonlinear problem ID (NL1_tanh, NL2_softmax, NL3_quadratic, NL4_circulant)")
    sp_nl.add_argument("--n", type=int, default=200, help="Dimension (default: 200)")
    sp_nl.add_argument("--max-iter", type=int, default=20, help="Max Newton iterations (default: 20)")
    sp_nl.add_argument("--tol", type=float, default=1e-10, help="Convergence tolerance (default: 1e-10)")
    sp_nl.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device")
    sp_nl.add_argument("--json", action="store_true", help="Also print JSON result")
    sp_nl.set_defaults(func=cmd_nonlinear_solve)

    # nonlinear-list
    sp_nllist = sub.add_parser("nonlinear-list", help="List built-in nonlinear problems")
    sp_nllist.set_defaults(func=cmd_nonlinear_list)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
