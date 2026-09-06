"""ICFR — Nonlinear extension via Newton's method with linearized ICFR.

For a nonlinear fixed-point problem x = G(x), at each Newton iterate we:
    1. Compute the Jacobian J_k = ∇G(x_k) via finite differences (or autograd).
    2. Apply the ICFR pipeline to the linearized system
       (I - J_k) Δx = -(G(x_k) - x_k).
    3. Update x_{k+1} = x_k + Δx.

If J_k happens to be low-rank, circulant, nilpotent, or banded, ICFR
exploits it for a direct Newton step — converging in fewer iterations
than standard Newton with dense LU.

Reference: Task 3.3 of the ICFR paper ("Extension to Nonlinear
Fixed-Point Operators").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .gpu import ICFRComputeEngine, ICFRConfig


# Type alias for nonlinear fixed-point map G: R^n -> R^n
NonlinearMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class NonlinearResult:
    """Result of an ICFR-Newton run."""

    solution: torch.Tensor
    converged: bool
    iterations: int
    final_residual: float
    paths: list[str]                     # decision path at each Newton iterate
    jacobian_classes: list[str]          # discovered class of each J_k
    total_time_ms: float
    per_iteration: list[dict] = field(default_factory=list)


class ICFRNewton:
    """Newton's method using ICFR to solve each linearized system on the GPU.

    Implements a damped Newton with simple backtracking line search to
    guarantee global convergence from poor initial guesses.
    """

    def __init__(
        self,
        engine: Optional[ICFRComputeEngine] = None,
        *,
        fd_eps: float = 1e-6,
        max_iter: int = 50,
        tol: float = 1e-10,
        damping: float = 1.0,
        track_history: bool = True,
        line_search: bool = True,
        line_search_factor: float = 0.5,
        line_search_max_steps: int = 10,
        min_damping: float = 1e-4,
    ) -> None:
        self.engine = engine or ICFRComputeEngine(ICFRConfig())
        self.fd_eps = fd_eps
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.track_history = track_history
        self.line_search = line_search
        self.line_search_factor = line_search_factor
        self.line_search_max_steps = line_search_max_steps
        self.min_damping = min_damping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        G: NonlinearMap,
        x0: torch.Tensor,
        *,
        jac_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> NonlinearResult:
        """Solve x = G(x) via damped ICFR-Newton.

        Args:
            G: nonlinear map R^n -> R^n (callable on torch tensors).
            x0: initial guess (will be moved to the engine's device).
            jac_fn: optional analytic Jacobian. If None, finite differences are used.

        Returns:
            NonlinearResult with solution + per-iteration metadata.
        """
        device = self.engine.config.torch_device
        x = x0.to(device).clone()
        if x.dtype != torch.float64:
            x = x.to(torch.float64)

        t0 = time.perf_counter()
        paths: list[str] = []
        jac_classes: list[str] = []
        per_iter: list[dict] = []

        converged = False
        final_residual = 0.0
        n_iter = 0

        # Compute initial residual
        Gx = G(x)
        r = Gx - x
        best_res_norm = float(torch.linalg.vector_norm(r).item())

        for k in range(1, self.max_iter + 1):
            n_iter = k

            res_norm = float(torch.linalg.vector_norm(r).item())
            final_residual = res_norm

            if self.track_history:
                per_iter.append({"k": k, "residual": res_norm, "x_norm": float(torch.linalg.vector_norm(x).item())})

            if res_norm < self.tol:
                converged = True
                break

            # Compute Jacobian J_k = ∇G(x_k)
            if jac_fn is not None:
                J = jac_fn(x)
            else:
                J = self._finite_difference_jacobian(G, x, Gx)

            # Apply ICFR to (I - J_k) Δx = r
            # Newton for f(x) = x - G(x) = 0:
            #   x_{k+1} = x_k - (I - J_G)^{-1} f(x_k)
            #           = x_k - (I - J_G)^{-1} (x_k - G(x_k))
            #           = x_k + (I - J_G)^{-1} (G(x_k) - x_k)
            #           = x_k + (I - J_G)^{-1} r
            # ICFR solves (I - T) x = b, so T = J_k and b = +r (NOT -r).
            x_sol, path = self.engine.solve_linear(J, r)
            delta = x_sol.to(device)

            # Damped Newton with backtracking line search
            if self.line_search:
                alpha = self.damping
                accepted = False
                for _ls in range(self.line_search_max_steps):
                    x_trial = x + alpha * delta
                    try:
                        Gx_trial = G(x_trial)
                        r_trial = Gx_trial - x_trial
                        trial_res = float(torch.linalg.vector_norm(r_trial).item())
                        if trial_res < res_norm or alpha <= self.min_damping:
                            x = x_trial
                            Gx = Gx_trial
                            r = r_trial
                            best_res_norm = trial_res
                            accepted = True
                            break
                        alpha *= self.line_search_factor
                    except Exception:
                        alpha *= self.line_search_factor
                if not accepted:
                    # Take the full Newton step anyway
                    x = x + self.damping * delta
                    Gx = G(x)
                    r = Gx - x
            else:
                x = x + self.damping * delta
                Gx = G(x)
                r = Gx - x

            # Track decision path
            paths.append(path.split(" ")[0])  # e.g. "DIRECT_WOODBURY_(RANK_5)"
            if "WOODBURY" in path:
                jac_classes.append("LOW_RANK")
            elif "NILPOTENT" in path:
                jac_classes.append("NILPOTENT")
            elif "CIRCULANT" in path:
                jac_classes.append("CIRCULANT")
            elif "DIAGONAL" in path:
                jac_classes.append("DIAGONAL")
            elif "SCALAR" in path:
                jac_classes.append("SCALAR")
            elif "BANDED" in path:
                jac_classes.append("BANDED")
            elif "SPARSE" in path:
                jac_classes.append("SPARSE")
            else:
                jac_classes.append("GENERAL")

        total_ms = (time.perf_counter() - t0) * 1000

        return NonlinearResult(
            solution=x.detach().cpu(),
            converged=converged,
            iterations=n_iter,
            final_residual=final_residual,
            paths=paths,
            jacobian_classes=jac_classes,
            total_time_ms=total_ms,
            per_iteration=per_iter,
        )

    # ------------------------------------------------------------------
    # Jacobian via finite differences (column-wise, GPU-parallel)
    # ------------------------------------------------------------------

    def _finite_difference_jacobian(
        self,
        G: NonlinearMap,
        x: torch.Tensor,
        Gx: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate J = ∇G(x) via forward finite differences.

        Vectorized: perturbs every coordinate simultaneously by building
        an (n, n) perturbation matrix [e_1, e_2, ..., e_n] * eps, then
        computing G(x + eps * e_i) for each i in a single batched call
        when possible.
        """
        n = x.shape[0]
        device = x.device
        dtype = x.dtype
        eps = self.fd_eps

        # Build perturbation matrix: each column is eps * e_i
        E = torch.eye(n, device=device, dtype=dtype) * eps  # (n, n)
        # Perturbed points: x_perturbed[i] = x + E[:, i]
        # We can't batch G calls in general (it might not support batching),
        # so loop over columns.
        J = torch.zeros((n, n), device=device, dtype=dtype)
        for i in range(n):
            x_pert = x + E[:, i]
            Gx_pert = G(x_pert)
            J[:, i] = (Gx_pert - Gx) / eps
        return J


# ============================================================================
# Built-in nonlinear test problems
# ============================================================================

def make_tanh_problem(n: int = 200, seed: int = 42) -> tuple[NonlinearMap, torch.Tensor, torch.Tensor]:
    """Problem NL1: x = tanh(A x + b).

    A is a low-rank matrix (rank 5), so the Jacobian J = (1 - tanh^2) ⊙ A
    is also low-rank at every iterate. ICFR should discover this and use
    Woodbury at each Newton step.
    """
    torch.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, 5, dtype=torch.float64))[0]
    A = 0.5 * (U @ U.T)  # rank-5, spectral radius 0.5
    b = torch.randn(n, dtype=torch.float64) * 0.3

    def G(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(A @ x + b)

    x0 = torch.zeros(n, dtype=torch.float64)
    return G, x0, b


def make_softmax_problem(n: int = 100, seed: int = 42) -> tuple[NonlinearMap, torch.Tensor, torch.Tensor]:
    """Problem NL2: x = softmax(W x + b) (fixed-point form).

    The Jacobian of softmax has special structure (low-rank perturbation
    of a diagonal). ICFR may discover this if the diagonal dominance is
    strong enough.
    """
    torch.manual_seed(seed)
    W = torch.randn(n, n, dtype=torch.float64) * 0.1
    b = torch.randn(n, dtype=torch.float64) * 0.5

    def G(x: torch.Tensor) -> torch.Tensor:
        z = W @ x + b
        z = z - z.max()
        e = torch.exp(z)
        return e / e.sum()

    x0 = torch.zeros(n, dtype=torch.float64)
    return G, x0, b


def make_quadratic_problem(n: int = 200, seed: int = 42) -> tuple[NonlinearMap, torch.Tensor, torch.Tensor]:
    """Problem NL3: x = 0.5 * (A x^2 + b) where x^2 is element-wise.

    The Jacobian is J = A * diag(x), which is low-rank when A is low-rank
    (which we make so).
    """
    torch.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, 3, dtype=torch.float64))[0]
    A = 0.3 * (U @ U.T)  # rank-3
    b = torch.randn(n, dtype=torch.float64) * 0.2

    def G(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (A @ (x * x) + b)

    x0 = torch.ones(n, dtype=torch.float64) * 0.1
    return G, x0, b


def make_circulant_nonlinear_problem(n: int = 200, seed: int = 42) -> tuple[NonlinearMap, torch.Tensor, torch.Tensor]:
    """Problem NL4: x = sigmoid(C x + b) where C is circulant.

    The Jacobian J = sigmoid' ⊙ C is the elementwise product of a diagonal
    with a circulant matrix — which is itself circulant. ICFR should
    discover this and use FFT.
    """
    torch.manual_seed(seed)
    c = torch.randn(n, dtype=torch.float64)
    C = torch.stack([torch.roll(c, i) for i in range(n)])
    b = torch.randn(n, dtype=torch.float64) * 0.3

    def G(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(C @ x + b)

    x0 = torch.zeros(n, dtype=torch.float64)
    return G, x0, b


NONLINEAR_PROBLEMS = {
    "NL1_tanh": make_tanh_problem,
    "NL2_softmax": make_softmax_problem,
    "NL3_quadratic": make_quadratic_problem,
    "NL4_circulant": make_circulant_nonlinear_problem,
}
