"""Optimal storage sizing: push an infeasible operating point back into the region.

When the network is congested the prevailing operating point lies outside the
region.  Storage installed at the points of connection can shift it back in, and
the question is how much of it is needed:

.. math::

    \\min \\sum_i |\\hat{P}_i - P_i| \\quad\\text{s.t.}\\quad P \\in \\mathcal{R}

The objective is the total regulating power the storage must provide, and the
only constraint is region membership.  Coordinates carry axis units (MW on ``p``
axes, Mvar on ``q`` axes); on mixed P/Q regions the L1 objective sums active
and reactive shifts linearly (as opposed to apparent power
``sqrt(dP^2 + dQ^2)`` per bus).  Written with ``A P <= b`` and an epigraph
split of the absolute value, the problem is a linear program whose constraint
count scales with ``region.facets``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from OpenRC.applications.base import (APPLICATIONS, Application, BatchResult, SolveResult, _units,
                                      solve_lp)
from OpenRC.geometry.polytope import Polytope
import time


@APPLICATIONS.register(
    "storage-sizing",
    doc="Minimum regulating power that brings an operating point back into the region.",
)
class StorageSizingProblem(Application):
    """Minimum total ``|shift|`` that moves a point into the region.

    Parameters
    ----------
    weights:
        Per-axis cost of regulating power.  ``None`` means equal weights.
    """

    name = "storage-sizing"

    def __init__(self, *, weights: Optional[np.ndarray] = None) -> None:
        if weights is not None:
            raw_w = np.asarray(weights)
            if raw_w.dtype == bool or raw_w.ndim != 1:
                raise ValueError("weights must be a 1-D array of finite non-negative numbers")
            w = np.array(raw_w, dtype=float, copy=True)
            if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
                raise ValueError("weights must be finite and non-negative")
            if not np.any(w > 0.0):
                raise ValueError("at least one storage weight must be positive")
            w.flags.writeable = False
            self.weights: Optional[np.ndarray] = w
        else:
            self.weights = None

    def _validated_weights(self, dim: int) -> np.ndarray:
        weights = np.ones(dim) if self.weights is None else self.weights
        if weights.shape != (dim,):
            raise ValueError(f"weights have shape {weights.shape}, expected ({dim},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one storage weight must be positive")
        return weights

    @staticmethod
    def _prepare_lp(region: Polytope, weights: np.ndarray) -> tuple:
        dim = region.dim
        A, b = region.facets
        # variables x = [P (dim), c (dim)] with c the epigraph of |P - point|
        cost = np.concatenate([np.zeros(dim), weights])
        eye = np.eye(dim)
        A_ub = np.vstack([
            np.hstack([A, np.zeros((A.shape[0], dim))]),     # A P <= b
            np.hstack([eye, -eye]),                          #  P - c <=  point
            np.hstack([-eye, -eye]),                         # -P - c <= -point
        ])
        rhs = np.empty(len(b) + 2 * dim, dtype=float)
        rhs[:len(b)] = b
        bounds = [(None, None)] * dim + [(0.0, None)] * dim
        return dim, len(b), cost, A_ub, rhs, bounds

    @staticmethod
    def _solve_prepared(dim: int, n_facets: int, cost: np.ndarray, A_ub: np.ndarray,
                        rhs: np.ndarray, bounds: list, scenario,
                        weights: np.ndarray) -> SolveResult:
        start = time.perf_counter()
        raw_point = np.asarray(scenario)
        if raw_point.dtype == bool:
            raise ValueError("scenario coordinates must be finite")
        point = np.asarray(raw_point, dtype=float)
        if point.shape != (dim,):
            raise ValueError(f"scenario has shape {point.shape}, expected ({dim},)")
        if not np.all(np.isfinite(point)):
            raise ValueError("scenario coordinates must be finite")
        rhs[n_facets:n_facets + dim] = point
        rhs[n_facets + dim:] = -point
        x, objective, ok, status = solve_lp(cost, A_ub, rhs, bounds=bounds)
        if ok and np.any(weights == 0.0):
            # Among optimal primary shifts, pick the minimum unweighted shift on
            # zero-cost axes so unpriced coordinates do not drift arbitrarily.
            tie_cost = np.concatenate([np.zeros(dim), (weights == 0.0).astype(float)])
            tie_A = np.vstack([A_ub, cost[None, :]])
            tie_rhs = np.append(rhs, objective + 1e-9 * max(1.0, abs(objective)))
            x2, _, ok2, _ = solve_lp(tie_cost, tie_A, tie_rhs, bounds=bounds)
            if ok2:
                x = x2
        elapsed = time.perf_counter() - start
        if not ok:
            return SolveResult(float("nan"), {}, False, elapsed, status)
        shift = x[:dim] - point
        if np.any(weights == 0.0):
            objective = float(weights @ np.abs(shift))
        return SolveResult(
            objective=objective,
            decision={"operating_point": x[:dim], "shift": shift,
                      "storage": np.abs(shift)},
            success=True,
            seconds=elapsed,
            status=status,
        )

    def solve(self, region: Polytope, scenario) -> SolveResult:
        """Solve for one operating point ``scenario`` (an array of region coordinates)."""
        dim = region.dim
        raw_point = np.asarray(scenario)
        if raw_point.dtype == bool:
            raise ValueError("scenario coordinates must be finite")
        point = np.asarray(raw_point, dtype=float)
        if point.shape != (dim,):
            raise ValueError(f"scenario has shape {point.shape}, expected ({dim},)")
        if not np.all(np.isfinite(point)):
            raise ValueError("scenario coordinates must be finite")
        weights = self._validated_weights(dim)
        dim, n_facets, cost, A_ub, rhs, bounds = self._prepare_lp(region, weights)
        return self._solve_prepared(dim, n_facets, cost, A_ub, rhs, bounds, point, weights)

    def solve_batch(self, region: Polytope, scenarios) -> BatchResult:
        """Solve a sequence of operating points reusing the scenario-invariant LP matrix."""
        if isinstance(scenarios, (str, bytes, bytearray)):
            raise TypeError("scenarios must be a sequence of operating-point vectors, not str or bytes")
        scenarios = list(scenarios)
        weights = self._validated_weights(region.dim)
        if not scenarios:
            n_fac = region.n_facets
            return BatchResult(
                results=[],
                region_n_vertices=region.n_vertices,
                region_n_facets=n_fac,
                region_n_constraints=n_fac,
                info={"application": self.name} | self.describe() | _units(region),
            )
        dim, n_facets, cost, A_ub, rhs, bounds = self._prepare_lp(region, weights)
        results = [
            self._solve_prepared(dim, n_facets, cost, A_ub, rhs, bounds, s, weights)
            for s in scenarios
        ]
        return BatchResult(
            results=results,
            region_n_vertices=region.n_vertices,
            region_n_facets=region.n_facets,
            region_n_constraints=n_facets,
            info={"application": self.name} | self.describe() | _units(region),
        )

    def describe(self) -> dict:
        return {"weights": None if self.weights is None else self.weights.tolist()}
