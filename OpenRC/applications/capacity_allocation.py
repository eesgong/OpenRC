r"""Optimal capacity allocation: the largest orthotope that fits inside the region.

Each point of connection is granted a capacity ``dP_i``.  Because a capacity is
a *range*, the grant is honoured only if every combination of the connected
units operating anywhere within their grants stays feasible - that is, if the
orthotope the grants span lies inside the region.

.. math::

    \\max \\sum_i \\omega_i \\Delta P_i \\quad\\text{s.t.}\\quad
    (P_1 + B_1 \\alpha_1 \\Delta P_1, \\ldots) \\in \\mathcal{R}
    \;\; \\forall B \\in \\{0,1\\}^m

``alpha_i`` is the loading ratio: units rarely all sit at full output at once,
so the corner actually tested is a fraction of the granted range.  ``P_lo``
(the orthotope's lower corner) is optimised jointly with ``dP``.

Because ``alpha_i >= 0`` and ``dP_i >= 0``, the maximum of ``a @ P`` over the
orthotope is ``a @ P_lo + max(a, 0) @ (alpha * dP)``, yielding a single linear
constraint per facet instead of ``2^m`` corner constraints.

Capacities are in the region's own coordinates: MW on a P axis, Mvar on a Q
axis (``BatchResult.info["units"]`` lists them when the region carries axes).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from OpenRC.applications.base import (APPLICATIONS, Application, BatchResult, SolveResult, _units,
                                      solve_lp)
from OpenRC.geometry.polytope import Polytope
import time


@APPLICATIONS.register(
    "capacity-allocation",
    doc="Largest weighted capacity orthotope that fits inside the region.",
)
class CapacityAllocationProblem(Application):
    """Maximise the weighted sum of capacities whose orthotope fits in the region.

    Parameters
    ----------
    weights:
        Per-axis preference ``omega_i``.  ``None`` means equal weights.
    min_loading_ratio:
        Lower bound ``alpha_hat`` applied element-wise to scenario loading ratios
        ``alpha = maximum(scenario, min_loading_ratio)`` to prevent near-zero
        loading factors from producing unbounded capacity allocations.
    """

    name = "capacity-allocation"

    def __init__(self, *, weights: Optional[np.ndarray] = None,
                 min_loading_ratio: float = 0.8) -> None:
        if weights is not None:
            raw_w = np.asarray(weights)
            if raw_w.dtype == bool or raw_w.ndim != 1:
                raise ValueError("weights must be a 1-D array of finite non-negative numbers")
            w = np.array(raw_w, dtype=float, copy=True)
            if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
                raise ValueError("weights must be finite and non-negative")
            if not np.any(w > 0.0):
                raise ValueError("at least one capacity weight must be positive")
            w.flags.writeable = False
            self.weights: Optional[np.ndarray] = w
        else:
            self.weights = None
        if isinstance(min_loading_ratio, (bool, np.bool_)):
            raise ValueError("min_loading_ratio must be finite and strictly positive")
        self.min_loading_ratio = float(min_loading_ratio)
        if not np.isfinite(self.min_loading_ratio) or self.min_loading_ratio <= 0.0:
            raise ValueError("min_loading_ratio must be finite and strictly positive")

    def _validated_weights(self, dim: int) -> np.ndarray:
        weights = np.ones(dim) if self.weights is None else self.weights
        if weights.shape != (dim,):
            raise ValueError(f"weights have shape {weights.shape}, expected ({dim},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one capacity weight must be positive")
        return weights

    @staticmethod
    def _solve_prepared(dim: int, cost: np.ndarray, A_pos: np.ndarray,
                        constraints: np.ndarray, b: np.ndarray, bounds: list,
                        scenario, min_loading_ratio: float) -> SolveResult:
        start = time.perf_counter()
        raw_alpha = np.asarray(scenario)
        if raw_alpha.dtype == bool:
            raise ValueError("loading ratios must be finite and non-negative")
        alpha = np.asarray(raw_alpha, dtype=float)
        if alpha.shape != (dim,):
            raise ValueError(f"scenario has shape {alpha.shape}, expected ({dim},)")
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.0):
            raise ValueError("loading ratios must be finite and non-negative")
        alpha = np.maximum(alpha, min_loading_ratio)
        constraints[:, dim:] = A_pos * alpha[None, :]
        x, objective, ok, status = solve_lp(cost, constraints, b, bounds=bounds)
        elapsed = time.perf_counter() - start
        if not ok:
            return SolveResult(float("nan"), {}, False, elapsed, status)
        return SolveResult(
            objective=-objective,   # undo the minimisation sign
            decision={"lower_corner": x[:dim], "capacity": x[dim:],
                      "loading_ratio": alpha},
            success=True,
            seconds=elapsed,
            status=status,
        )

    def solve(self, region: Polytope, scenario) -> SolveResult:
        """Solve for one scenario of loading ratios ``alpha`` (one per axis)."""
        dim = region.dim
        raw_alpha = np.asarray(scenario)
        if raw_alpha.dtype == bool:
            raise ValueError("loading ratios must be finite and non-negative")
        alpha = np.asarray(raw_alpha, dtype=float)
        if alpha.shape != (dim,):
            raise ValueError(f"scenario has shape {alpha.shape}, expected ({dim},)")
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.0):
            raise ValueError("loading ratios must be finite and non-negative")
        weights = self._validated_weights(dim)
        A, b = region.facets
        A_pos = np.maximum(A, 0.0)
        # variables x = [P_lo (dim), dP (dim)], maximise w.dP  ->  minimise -w.dP
        cost = np.concatenate([np.zeros(dim), -weights])
        # Worst-case corner per facet: A @ P_lo + max(A, 0) @ (alpha * dP) <= b.
        constraints = np.empty((A.shape[0], 2 * dim), dtype=float)
        constraints[:, :dim] = A
        bounds = [(None, None)] * dim + [(0.0, None)] * dim
        return self._solve_prepared(dim, cost, A_pos, constraints, b, bounds,
                                    alpha, self.min_loading_ratio)

    def solve_batch(self, region: Polytope, scenarios) -> BatchResult:
        """Solve a sequence of loading-ratio scenarios reusing the region's halfspaces."""
        if isinstance(scenarios, (str, bytes, bytearray)):
            raise TypeError("scenarios must be a sequence of loading-ratio vectors, not str or bytes")
        scenarios = list(scenarios)
        dim = region.dim
        weights = self._validated_weights(dim)
        if not scenarios:
            n_fac = region.n_facets
            return BatchResult(
                results=[],
                region_n_vertices=region.n_vertices,
                region_n_facets=n_fac,
                region_n_constraints=n_fac,
                info={"application": self.name} | self.describe() | _units(region),
            )
        A, b = region.facets
        A_pos = np.maximum(A, 0.0)
        cost = np.concatenate([np.zeros(dim), -weights])
        constraints = np.empty((A.shape[0], 2 * dim), dtype=float)
        constraints[:, :dim] = A
        bounds = [(None, None)] * dim + [(0.0, None)] * dim
        results = [
            self._solve_prepared(dim, cost, A_pos, constraints, b, bounds,
                                 s, self.min_loading_ratio)
            for s in scenarios
        ]
        return BatchResult(
            results=results,
            region_n_vertices=region.n_vertices,
            region_n_facets=region.n_facets,
            region_n_constraints=A.shape[0],
            info={"application": self.name} | self.describe() | _units(region),
        )

    def describe(self) -> dict:
        return {
            "weights": None if self.weights is None else self.weights.tolist(),
            "min_loading_ratio": self.min_loading_ratio,
        }
