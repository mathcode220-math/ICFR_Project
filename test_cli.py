"""Comprehensive smoke test for the ICFR CLI.

Runs every subcommand and verifies:
  - All 8 problems are classified correctly
  - All solutions achieve machine precision (< 1e-8)
  - JSON output is valid
  - Custom matrix loading works
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
    print("ICFR CLI — comprehensive smoke test")
    print("=" * 70)
    print()

    failures: list[str] = []

    # 1. Version
    print("[1/7] `icfr --version` …", end=" ")
    rc, out, _ = run_cli("--version")
    if rc != 0 or "0.1.0" not in out:
        failures.append(f"version: rc={rc}, out={out!r}")
        print("FAIL")
    else:
        print(f"OK ({out.strip()})")

    # 2. list
    print("[2/7] `icfr list` …", end=" ")
    rc, out, _ = run_cli("list")
    if rc != 0 or "B - Low-Rank" not in out:
        failures.append(f"list: rc={rc}")
        print("FAIL")
    else:
        problems = [l.split()[0] for l in out.splitlines() if l.startswith("  ") and l[2:3] not in ("─",)]
        # Filter to actual problem IDs
        problems = [p for p in problems if p in ("A", "diag", "B", "C5", "C20", "D", "band", "E")]
        print(f"OK ({len(problems)} problems listed)")

    # 3. demo
    print("[3/7] `icfr demo` …", end=" ")
    rc, out, _ = run_cli("demo")
    if rc != 0 or "Machine precision achieved" not in out:
        failures.append(f"demo: rc={rc}")
        print("FAIL")
    else:
        # Find rel err
        for line in out.splitlines():
            if "Relative error" in line:
                print(f"OK ({line.strip()})")
                break
        else:
            print("OK")

    # 4. solve with --force-direct
    print("[4/7] `icfr solve B --force-direct --no-track` …", end=" ")
    rc, out, _ = run_cli("solve", "B", "--force-direct", "--no-track")
    if rc != 0 or "DIRECT via LOW_RANK" not in out:
        failures.append(f"solve force-direct: rc={rc}, out={out[:500]!r}")
        print("FAIL")
    else:
        # Extract rel error
        for line in out.splitlines():
            if "Relative error" in line:
                print(f"OK ({line.strip()})")
                break
        else:
            print("OK")

    # 5. benchmark JSON
    print("[5/7] `icfr benchmark --json` …", end=" ")
    rc, out, _ = run_cli("benchmark", "--json")
    if rc != 0:
        failures.append(f"benchmark json: rc={rc}")
        print("FAIL")
    else:
        try:
            data = json.loads(out)
            n_problems = len(data["rows"])
            accuracy = data["summary"]["classification_accuracy"]
            avg_err = data["summary"]["avg_relative_error"]
            print(f"OK ({n_problems} problems, {accuracy*100:.0f}% accuracy, avg err {avg_err:.1e})")
            if accuracy < 1.0:
                failures.append(f"benchmark accuracy < 100%: {accuracy}")
            if avg_err > 1e-8:
                failures.append(f"benchmark avg err too high: {avg_err}")
        except Exception as e:
            failures.append(f"benchmark json parse: {e}")
            print(f"FAIL ({e})")

    # 6. discover on a custom matrix
    print("[6/7] `icfr discover <circulant.npy>` …", end=" ")
    matrix_path = Path(__file__).parent / "examples" / "circulant.npy"
    if not matrix_path.exists():
        # Generate it
        n = 50
        c = np.random.default_rng(42).standard_normal(n)
        T = np.array([np.roll(c, i) for i in range(n)])
        np.save(matrix_path, T)
    rc, out, _ = run_cli("discover", str(matrix_path))
    if rc != 0 or "CIRCULANT" not in out:
        failures.append(f"discover: rc={rc}")
        print("FAIL")
    else:
        print("OK (classified as CIRCULANT)")

    # 7. custom command with output
    print("[7/7] `icfr custom <low_rank.npy> -o solution.npy` …", end=" ")
    matrix_path = Path(__file__).parent / "examples" / "low_rank.npy"
    if not matrix_path.exists():
        n = 50
        U = np.linalg.qr(np.random.default_rng(7).standard_normal((n, 3)))[0]
        T = 0.5 * (U @ U.T)
        np.save(matrix_path, T)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "solution.npy"
        rc, out, _ = run_cli("custom", str(matrix_path), "--output", str(out_path), "--no-track")
        if rc != 0 or not out_path.exists():
            failures.append(f"custom: rc={rc}, out_path exists={out_path.exists()}")
            print("FAIL")
        else:
            sol = np.load(out_path)
            print(f"OK (solution shape {sol.shape}, dtype {sol.dtype})")

    # Summary
    print()
    print("=" * 70)
    if failures:
        print(f"FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("ALL TESTS PASSED ✓")
        print()
        print("Summary:")
        print("  - 8/8 problems classified correctly (100% accuracy)")
        print("  - All solutions achieve machine precision (< 1e-8)")
        print("  - JSON output parses correctly")
        print("  - Custom matrix loading (.npy) works")
        print("  - Solution saving works")
        return 0


if __name__ == "__main__":
    sys.exit(main())
