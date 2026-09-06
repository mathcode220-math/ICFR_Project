"""ICFR — GPU-accelerated engine built on PyTorch.

Implements the three-stage pipeline described in the project README:

    Stage 1: Heuristic Probing       — O(n) geometric diagnostics
    Stage 2: Randomized Sketching    — O(n*r) RandNLA spectrum reduction
    Stage 3: Parallel Direct Solver  — Woodbury / FFT / Neumann / banded LU

All operations execute on a single device (CPU or CUDA) so that no
host-to-device transfers occur during the pipeline. The result is moved
back to CPU only at the very end.

Reference: "Iterative-to-Closed-Form Reduction (ICFR): A Unified
Framework for Automated Structure Discovery and Direct Inversion in
Fixed-Point Iterations".
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ICFRConfig:
    """Manages structural tolerances, execution criteria, and target device."""

    tau: float = 1e-8                # classification threshold
    max_rank_ratio: float = 0.05     # max r/n ratio for low-rank classification
    oversampling: int = 5            # RandNLA oversampling parameter
    device: str = "cuda"             # requested device
    q_max: int = 100                 # max nilpotency index to probe
    banded_bw_max_ratio: float = 0.1  # max bandwidth/n ratio for banded class
    nnz_ratio_max: float = 0.1       # max nnz/n^2 ratio for sparse class
    force_direct: bool = False       # bypass cost-benefit switch (for demo)

    def __post_init__(self) -> None:
        # Resolve actual device: fall back to CPU if CUDA not available.
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.torch_device = torch.device(self.device)


# ============================================================================
# Stage 1: Heuristic Prober
# ============================================================================

class HeuristicProber:
    """Stage 1: Executes sub-quadratic O(n) trace analysis to early-abort general matrices.

    Returns a hint string that triggers Stage 2 to look for a specific structure.
    """

    def __init__(self, config: ICFRConfig) -> None:
        self.config = config

    def quick_scan(self, T: torch.Tensor) -> tuple[bool, str]:
        n = T.shape[0]
        device = T.device
        dtype = T.dtype

        diag_elements = torch.diagonal(T, dim1=-2, dim2=-1)
        diag_norm = torch.linalg.vector_norm(diag_elements)
        total_norm = torch.linalg.matrix_norm(T)

        # Guard against zero matrices
        if total_norm.item() < self.config.tau:
            return True, "ZERO"

        # Nilpotent hint: diagonal is (near) zero
        if diag_norm.item() < self.config.tau * total_norm.item():
            return True, "POTENTIAL_NILPOTENT"

        # Diagonal hint: off-diagonal entries are negligible
        if diag_norm.item() > 0.95 * total_norm.item():
            return True, "POTENTIAL_DIAGONAL"

        # Scalar hint: all diagonal entries are equal and off-diagonal is zero
        if n > 1:
            diag_var = torch.var(diag_elements)
            if diag_var.item() < self.config.tau and diag_norm.item() > 0.95 * total_norm.item():
                return True, "POTENTIAL_SCALAR"

        # Banded hint: tridiagonal capture > 90% of mass
        if n > 2:
            super_diag = torch.diagonal(T, offset=1, dim1=-2, dim2=-1)
            sub_diag = torch.diagonal(T, offset=-1, dim1=-2, dim2=-1)
            tri_mass = (
                torch.linalg.vector_norm(super_diag)
                + torch.linalg.vector_norm(sub_diag)
                + diag_norm
            )
            if tri_mass.item() > 0.90 * total_norm.item():
                return True, "POTENTIAL_BANDED"

        # Circulant hint: row 1 is a cyclic shift of row 0
        if n > 4:
            sample_rows = min(5, n - 1)
            is_circ = True
            for i in range(1, sample_rows):
                expected = torch.roll(T[0], shifts=i)
                if not torch.allclose(T[i], expected, atol=self.config.tau):
                    is_circ = False
                    break
            if is_circ:
                return True, "POTENTIAL_CIRCULANT"

        return False, "GENERAL"


# ============================================================================
# Stage 2: Randomized Matrix Sketcher
# ============================================================================

class AdvancedRandomizedSketcher:
    """Stage 2: RandNLA spectrum dimension-reduction to catch hidden low-rank matrices.

    Uses the Halko-Martin-Tropp randomized range finder:
        1. Draw Omega ~ N(0, I) of shape (n, k) with k = r + oversampling.
        2. Compute Y = T @ Omega.
        3. Orthonormalize Y -> Q via QR.
        4. Project B = Q^T @ T @ Q (k x k).
        5. SVD of B -> approximate singular values of T.
    """

    def __init__(self, config: ICFRConfig) -> None:
        self.config = config

    def estimate_rank(self, T: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Return (estimated_rank, Q, S_sketch) where Q projects onto the range of T.

        Uses the Halko-Martin-Tropp randomized range finder with a sharp
        spectral-gap test: a matrix is classified as low-rank only if there
        is a clear gap between the top-r singular values and the rest.
        """
        n = T.shape[0]
        max_r = max(5, int(n * self.config.max_rank_ratio))
        k = min(max_r + self.config.oversampling, n - 1)

        if k >= n:
            # Fall back to full SVD
            U, S, Vh = torch.linalg.svd(T, full_matrices=False)
            s_max = S[0].item() if S.numel() > 0 else 1.0
            cutoff = max(self.config.tau, s_max * 1e-10)
            rank = int(torch.sum(S > cutoff).item())
            return rank, U[:, :rank], S[:rank]

        # Randomized range finder: Y = T @ Omega, then QR
        omega = torch.randn(n, k, device=T.device, dtype=T.dtype)
        Y = T @ omega
        Q, _ = torch.linalg.qr(Y)
        B = Q.T @ T @ Q  # k x k

        # SVD of small matrix B
        S = torch.linalg.svdvals(B)
        s_max = S[0].item() if S.numel() > 0 else 1.0

        # Detect low-rank via spectral gap: there must be an index r where
        # S[r] / S[r-1] is very small (<< 1e-3). This distinguishes a true
        # rank-r matrix from a full-rank matrix that just happens to project
        # to k dimensions.
        s_list = S.tolist()
        rank = 0
        for i in range(1, len(s_list)):
            if s_list[i - 1] > 1e-12 and s_list[i] / s_list[i - 1] < 1e-3:
                rank = i
                break
        # If no gap found, the matrix is NOT low-rank (rank = n)
        if rank == 0:
            rank = n  # signal "not low-rank"
        return rank, Q, S

    def check_low_rank(self, T: torch.Tensor) -> tuple[bool, int]:
        """Return (is_low_rank, rank)."""
        n = T.shape[0]
        rank, _, _ = self.estimate_rank(T)
        max_r = max(5, int(n * self.config.max_rank_ratio))
        return 0 < rank <= max_r, rank


# ============================================================================
# Stage 3: Direct Solvers (all on GPU)
# ============================================================================

class DirectSolver:
    """Stage 3: Parallel direct linear inversions for each discovered structure."""

    def __init__(self, config: ICFRConfig) -> None:
        self.config = config

    def solve_scalar(self, T: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(I - r I)^-1 b = b / (1 - r)."""
        r = T[0, 0].item()
        return b / (1.0 - r)

    def solve_diagonal(self, T: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(I - diag(d))^-1 b = b / (1 - d) element-wise."""
        d = torch.diagonal(T, dim1=-2, dim2=-1)
        return b / (1.0 - d)

    def solve_nilpotent(self, T: torch.Tensor, b: torch.Tensor, q: int) -> torch.Tensor:
        """Finite Neumann series: (I - T)^-1 b = sum_{k=0}^{q-1} T^k b."""
        result = b.clone()
        current = b.clone()
        for _ in range(1, q):
            current = T @ current
            if torch.linalg.vector_norm(current).item() < 1e-14:
                break
            result = result + current
        return result

    def solve_low_rank(self, T: torch.Tensor, b: torch.Tensor, rank: int) -> torch.Tensor:
        """Woodbury identity: (I - U V^T)^-1 b = b + U (I_r - V^T U)^-1 V^T b.

        The inner r x r system is solved via LU on the GPU.
        """
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
        r = min(rank, S.shape[0])
        sqrtS = torch.sqrt(torch.clamp(S[:r], min=0.0))
        Ur = U[:, :r] * sqrtS.unsqueeze(0)
        Vr = Vh[:r, :].T * sqrtS.unsqueeze(0)

        I_r = torch.eye(r, device=T.device, dtype=T.dtype)
        M = I_r - Vr.T @ Ur  # r x r
        # Solve M z = V^T b
        y = Vr.T @ b
        z = torch.linalg.solve(M, y)
        return b + Ur @ z

    def solve_circulant(self, T: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Circulant: x = IFFT(FFT(b) / (1 - FFT(c)))."""
        c = T[:, 0]
        # Real-input FFT
        B = torch.fft.fft(b)
        C = torch.fft.fft(c)
        denom = 1.0 - C
        # Guard against near-zero denominators
        eps = 1e-14
        # Use the absolute value of denom to find small entries; replace with eps
        denom_abs = torch.abs(denom)
        # Avoid complex dtype mismatch: build the replacement as complex
        safe_denom = torch.where(
            denom_abs < eps,
            torch.full_like(denom, eps + 0j),
            denom,
        )
        x_c = B / safe_denom
        return torch.real(torch.fft.ifft(x_c))

    def solve_banded(self, T: torch.Tensor, b: torch.Tensor, bandwidth: int) -> torch.Tensor:
        """Banded: solve (I - T) x = b via dense LU (torch has no native banded solver)."""
        n = T.shape[0]
        A = torch.eye(n, device=T.device, dtype=T.dtype) - T
        return torch.linalg.solve(A, b)

    def solve_sparse(self, T: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Sparse: dense LU fallback (torch has no native sparse direct solver)."""
        n = T.shape[0]
        A = torch.eye(n, device=T.device, dtype=T.dtype) - T
        return torch.linalg.solve(A, b)

    def solve_dense(self, T: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """General dense: solve (I - T) x = b via LU on the GPU."""
        n = T.shape[0]
        A = torch.eye(n, device=T.device, dtype=T.dtype) - T
        return torch.linalg.solve(A, b)


# ============================================================================
# Main Compute Engine
# ============================================================================

@dataclass
class ICFRResult:
    """Result of a single ICFR run on the GPU."""

    solution: torch.Tensor
    decision: str          # "DIRECT" | "ITERATIVE"
    solver_path: str       # e.g. "DIRECT_WOODBURY_(RANK_5)"
    discovered_class: str  # e.g. "LOW_RANK"
    spectral_radius: float
    frob_norm: float
    nnz: int
    discovery_time_ms: float
    solve_time_ms: float
    total_time_ms: float
    reason: str
    iterations: list = field(default_factory=list)  # for ITERATIVE path


class ICFRComputeEngine:
    """The unified pipeline. All operations execute on a single device."""

    def __init__(self, config: Optional[ICFRConfig] = None) -> None:
        self.config = config or ICFRConfig()
        self.prober = HeuristicProber(self.config)
        self.sketcher = AdvancedRandomizedSketcher(self.config)
        self.solver = DirectSolver(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_linear(self, T_cpu: torch.Tensor, b_cpu: torch.Tensor) -> tuple[torch.Tensor, str]:
        """Solve (I - T) x = b. Inputs may be on CPU; they're moved to the engine's device.

        Returns (solution_on_cpu, decision_path_string).
        """
        T = T_cpu.to(self.config.torch_device)
        b = b_cpu.to(self.config.torch_device)
        if T.dtype != b.dtype:
            T = T.to(b.dtype)

        t0 = time.perf_counter()
        T = T.contiguous()
        b = b.contiguous()

        # Stage 1: Heuristic probing
        has_signal, hint = self.prober.quick_scan(T)
        t_probe = time.perf_counter() - t0

        # Always try low-rank detection (via randomized sketching) even if
        # the geometric probe returns GENERAL — a matrix can be low-rank
        # without having zero diagonal or any other geometric hint.
        # The full dispatch in _dispatch() handles this case.
        t_solve0 = time.perf_counter()
        x, path = self._dispatch(T, b, hint)
        t_solve = time.perf_counter() - t_solve0
        total_ms = (time.perf_counter() - t0) * 1000
        return (
            x.cpu(),
            f"{path} (probe={t_probe*1000:.1f}ms, solve={t_solve*1000:.1f}ms, total={total_ms:.1f}ms)",
        )

    def solve_linear_full(self, T_cpu: torch.Tensor, b_cpu: torch.Tensor) -> ICFRResult:
        """Like solve_linear but returns a full ICFRResult with diagnostics."""
        T = T_cpu.to(self.config.torch_device)
        b = b_cpu.to(self.config.torch_device)
        if T.dtype != b.dtype:
            T = T.to(b.dtype)
        T = T.contiguous()
        b = b.contiguous()
        n = T.shape[0]

        t0 = time.perf_counter()

        # Diagnostics (always computed, regardless of decision)
        frob_norm = torch.linalg.matrix_norm(T).item()
        nnz = int(torch.count_nonzero(torch.abs(T) > 1e-12).item())

        # Stage 1
        has_signal, hint = self.prober.quick_scan(T)
        t_probe = time.perf_counter() - t0

        # Stage 2 + 3
        t_solve0 = time.perf_counter()
        x, path = self._dispatch(T, b, hint)
        t_solve = time.perf_counter() - t_solve0

        total_ms = (time.perf_counter() - t0) * 1000

        # Spectral radius via power iteration (cheap, GPU-parallel)
        rho = self._spectral_radius(T)

        # Extract discovered class from solver path (more accurate than hint)
        discovered_class = self._class_from_path(path)

        return ICFRResult(
            solution=x.detach().cpu(),
            decision="DIRECT" if path.startswith("DIRECT") or path in ("SCALAR", "DIAGONAL", "NILPOTENT", "CIRCULANT") else "ITERATIVE",
            solver_path=path,
            discovered_class=discovered_class,
            spectral_radius=rho,
            frob_norm=frob_norm,
            nnz=nnz,
            discovery_time_ms=t_probe * 1000,
            solve_time_ms=t_solve * 1000,
            total_time_ms=total_ms,
            reason=f"Hint={hint}; path={path}",
            iterations=[],
        )

    @staticmethod
    def _class_from_path(path: str) -> str:
        """Extract the discovered class name from the solver path string."""
        if "WOODBURY" in path:
            return "LOW_RANK"
        if "NILPOTENT" in path:
            return "NILPOTENT"
        if "CIRCULANT" in path:
            return "CIRCULANT"
        if "DIAGONAL" in path:
            return "DIAGONAL"
        if "SCALAR" in path:
            return "SCALAR"
        if "BANDED" in path:
            return "BANDED"
        if "SPARSE" in path:
            return "SPARSE"
        if "ZERO" in path:
            return "ZERO"
        return "GENERAL"

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, T: torch.Tensor, b: torch.Tensor, hint: str) -> tuple[torch.Tensor, str]:
        """Refine hint to exact class and dispatch to the matching solver."""
        n = T.shape[0]

        if hint == "ZERO":
            return b.clone(), "DIRECT_ZERO"

        if hint == "POTENTIAL_SCALAR":
            # Verify
            r = T[0, 0].item()
            if torch.allclose(T, r * torch.eye(n, device=T.device, dtype=T.dtype), atol=self.config.tau):
                return self.solver.solve_scalar(T, b), "DIRECT_SCALAR"

        if hint == "POTENTIAL_DIAGONAL":
            d = torch.diagonal(T, dim1=-2, dim2=-1)
            off = T - torch.diag(d)
            if torch.linalg.matrix_norm(off).item() < self.config.tau * torch.linalg.matrix_norm(T).item():
                return self.solver.solve_diagonal(T, b), "DIRECT_DIAGONAL"

        if hint == "POTENTIAL_NILPOTENT":
            q = self._find_nilpotency_index(T)
            if q > 0:
                return self.solver.solve_nilpotent(T, b, q), f"DIRECT_NILPOTENT_(Q={q})"

        # Low-rank test via randomized sketching
        is_lr, rank = self.sketcher.check_low_rank(T)
        if is_lr and rank > 0:
            return self.solver.solve_low_rank(T, b, rank), f"DIRECT_WOODBURY_(RANK_{rank})"

        if hint == "POTENTIAL_CIRCULANT":
            return self.solver.solve_circulant(T, b), "DIRECT_CIRCULANT_FFT"

        if hint == "POTENTIAL_BANDED":
            bw = self._estimate_bandwidth(T)
            if bw < n * self.config.banded_bw_max_ratio:
                return self.solver.solve_banded(T, b, bw), f"DIRECT_BANDED_(BW={bw})"

        # Sparse test
        nnz_ratio = float(torch.count_nonzero(torch.abs(T) > 1e-12).item()) / (n * n)
        if nnz_ratio < self.config.nnz_ratio_max:
            return self.solver.solve_sparse(T, b), f"DIRECT_SPARSE_(NNZ_RATIO={nnz_ratio:.3f})"

        # Fallback: dense LU
        return self.solver.solve_dense(T, b), "GENERAL_FALLBACK"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_nilpotency_index(self, T: torch.Tensor) -> int:
        """Probe T^q = 0 for q = 1, 2, ..., q_max. Returns 0 if not nilpotent."""
        n = T.shape[0]
        P = T.clone()
        norm_T = torch.linalg.matrix_norm(T).item()
        if norm_T < self.config.tau:
            return 1
        for q in range(1, min(self.config.q_max, n)):
            if torch.linalg.matrix_norm(P).item() < self.config.tau * norm_T:
                return q
            P = P @ T
        return 0

    def _estimate_bandwidth(self, T: torch.Tensor) -> int:
        """Estimate bandwidth = max |i - j| where T[i, j] != 0."""
        n = T.shape[0]
        nz = torch.abs(T) > 1e-12
        # Vectorized: for each diagonal offset k, check if any nonzero
        max_bw = 0
        for k in range(n):
            diag_k = torch.diagonal(T, offset=k, dim1=-2, dim2=-1)
            if torch.any(torch.abs(diag_k) > 1e-12):
                max_bw = k
            elif k > max_bw + 5:
                break
        return max_bw

    def _spectral_radius(self, T: torch.Tensor, max_iter: int = 50) -> float:
        """Power iteration estimate of rho(T)."""
        n = T.shape[0]
        x = torch.ones(n, device=T.device, dtype=T.dtype) / math.sqrt(n)
        lam = 0.0
        for _ in range(max_iter):
            y = T @ x
            ny = torch.linalg.vector_norm(y).item()
            if ny < 1e-20:
                return 0.0
            x = y / ny
            Tx = T @ x
            lam = abs(float((x @ Tx).item()))
            upper = torch.linalg.vector_norm(Tx).item()
            if upper > lam:
                lam = upper
        return lam
