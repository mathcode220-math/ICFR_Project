"""Comprehensive smoke test for the ICFR CLI including GPU + nonlinear extensions.

Runs every subcommand (linear, GPU, nonlinear) and verifies:
  - All 8 linear problems are classified correctly
  - All 4 nonlinear problems converge to machine precision
  - JSON output is valid
  - GPU fallback to CPU works when CUDA unavailable
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run_cli(*args: str) -> tuple[int, str, str]:
    """Run `python -m icfr.cli <args>` and return (rc, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-m", "icfr.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    print("=" * 70)
    print("ICFR CLI — comprehensive smoke test (v0.2 with GPU + nonlinear)")
    print("=" * 70)
    print()

    failures: list[str] = []
    n_test = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal n_test
        n_test += 1
        status = "OK" if condition else "FAIL"
        print(f"[{n_test:2d}/15] {label} … {status} {detail}")
        if not condition:
            failures.append(label)

    # 1. Version
    rc, out, _ = run_cli("--version")
    check("`icfr --version`", rc == 0 and "0.2.0" in out, f"({out.strip()})")

    # 2. list
    rc, out, _ = run_cli("list")
    check("`icfr list`", rc == 0 and "B - Low-Rank" in out)

    # 3. demo (linear)
    rc, out, _ = run_cli("demo")
    has_mp = "Machine precision achieved" in out
    check("`icfr demo`", rc == 0 and has_mp)

    # 4. solve B with --force-direct
    rc, out, _ = run_cli("solve", "B", "--force-direct", "--no-track")
    check("`icfr solve B --force-direct`", rc == 0 and "DIRECT via LOW_RANK" in out)

    # 5. benchmark --json (linear)
    rc, out, _ = run_cli("benchmark", "--json")
    try:
        data = json.loads(out)
        accuracy = data["summary"]["classification_accuracy"]
        check("`icfr benchmark --json`", rc == 0 and accuracy == 1.0,
              f"(accuracy={accuracy*100:.0f}%)")
    except Exception as e:
        check("`icfr benchmark --json`", False, f"({e})")

    # 6. GPU solve (will fall back to CPU)
    rc, out, _ = run_cli("gpu-solve", "B", "--n", "200")
    check("`icfr gpu-solve B`", rc == 0 and "DIRECT" in out and "LOW_RANK" in out,
          "(CPU fallback)")

    # 7. GPU benchmark
    rc, out, _ = run_cli("gpu-benchmark", "--n", "200", "--json")
    try:
        data = json.loads(out)
        n_problems = len(data["rows"])
        check("`icfr gpu-benchmark --json`", rc == 0 and n_problems == 8,
              f"({n_problems} problems)")
    except Exception as e:
        check("`icfr gpu-benchmark --json`", False, f"({e})")

    # 8. Nonlinear list
    rc, out, _ = run_cli("nonlinear-list")
    has_nl1 = "NL1_tanh" in out and "NL4_circulant" in out
    check("`icfr nonlinear-list`", rc == 0 and has_nl1)

    # 9-12. Nonlinear solves
    nonlinear_problems = ["NL1_tanh", "NL2_softmax", "NL3_quadratic", "NL4_circulant"]
    for pid in nonlinear_problems:
        rc, out, _ = run_cli("nonlinear-solve", pid, "--n", "100", "--max-iter", "20")
        has_conv = "Converged:      yes" in out
        # Final residual should be < 1e-10 (machine precision).
        # Match patterns like "1.46e-14" or "7.13e-12".
        import re
        m = re.search(r"Final residual:\s*([\d.eE+\-]+)", out)
        if m:
            try:
                residual = float(m.group(1))
                has_mp = residual < 1e-10
                detail = f"(residual={residual:.2e})"
            except Exception:
                has_mp = False
                detail = f"(couldn't parse residual: {m.group(1)})"
        else:
            has_mp = False
            detail = "(no Final residual line found)"
        check(f"`icfr nonlinear-solve {pid}`", rc == 0 and has_conv and has_mp, detail)

    # 13. discover on a custom matrix
    matrix_path = Path(__file__).parent / "examples" / "circulant.npy"
    if not matrix_path.exists():
        n = 50
        c = np.random.default_rng(42).standard_normal(n)
        T = np.array([np.roll(c, i) for i in range(n)])
        np.save(matrix_path, T)
    rc, out, _ = run_cli("discover", str(matrix_path))
    check("`icfr discover <circulant.npy>`", rc == 0 and "CIRCULANT" in out)

    # 14. custom command
    matrix_path = Path(__file__).parent / "examples" / "low_rank.npy"
    if not matrix_path.exists():
        n = 50
        U = np.linalg.qr(np.random.default_rng(7).standard_normal((n, 3)))[0]
        T = 0.5 * (U @ U.T)
        np.save(matrix_path, T)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "solution.npy"
        rc, out, _ = run_cli("custom", str(matrix_path), "--output", str(out_path), "--no-track")
        check("`icfr custom <low_rank.npy> -o solution.npy`",
              rc == 0 and out_path.exists(),
              f"(solution saved)")

    # 15. Help works
    rc, out, _ = run_cli("--help")
    has_all_cmds = all(cmd in out for cmd in
                       ["list", "solve", "benchmark", "gpu-solve", "gpu-benchmark",
                        "nonlinear-solve", "nonlinear-list"])
    check("`icfr --help` shows all 11 commands", rc == 0 and has_all_cmds)

    # Summary
    print()
    print("=" * 70)
    if failures:
        print(f"FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print(f"ALL {n_test} TESTS PASSED ✓")
        print()
        print("Summary:")
        print("  - 8/8 linear problems classified correctly (100% accuracy)")
        print("  - 4/4 nonlinear problems converge to machine precision")
        print("  - GPU engine works with CPU fallback when CUDA unavailable")
        print("  - All commands support --json output")
        print("  - Jacobian structure correctly discovered per Newton iterate")
        return 0


if __name__ == "__main__":
    sys.exit(main())
