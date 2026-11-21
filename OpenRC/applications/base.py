"""Linear-programming decision problems posed over a polytope security region.

Representing the feasible injection set as ``A x <= b`` reduces grid-constrained
decision problems over the points of connection to linear programs without
power-flow variables or non-convex constraints.  LP row count scales with the
polytope's half-space count (:attr:`Polytope.facets`).  All problems are solved
via ``scipy.optimize.linprog(..., method="highs")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from OpenRC.geometry.polytope import Polytope
from OpenRC.registry import Registry

#: Registry of application classes.
APPLICATIONS: Registry = Registry("application")


@dataclass
class SolveResult:
    """Result of a single scenario LP solve: objective value, decision variables, status, and runtime."""

    objective: float
    decision: dict
    success: bool
    seconds: float
    status: str = ""

    def as_dict(self) -> dict:
        out = {"objective": self.objective, "success": self.success,
               "seconds": self.seconds, "status": self.status}
        out.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in self.decision.items()})
        return out


@dataclass
class BatchResult:
    """Aggregated results and region metadata across a batch of scenarios."""

    results: List[SolveResult] = field(default_factory=list)
    region_n_vertices: int = 0
    region_n_facets: int = 0
    #: Rows of the region's halfspace system the LP uses (one per hull simplex,
    #: so it can exceed ``region_n_facets`` on a non-simplicial hull).
    region_n_constraints: Optional[int] = None
    info: dict = field(default_factory=dict)

    @property
    def n_solved(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def mean_objective(self) -> float:
        values = [r.objective for r in self.results if r.success]
        return float(np.mean(values)) if values else float("nan")

    @property
    def mean_ms(self) -> float:
        values = [r.seconds for r in self.results if r.success]
        return float(1000.0 * np.mean(values)) if values else float("nan")

    @property
    def total_seconds(self) -> float:
        return float(sum(r.seconds for r in self.results))

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([r.as_dict() for r in self.results])

    def __repr__(self) -> str:
        app = self.info.get("application", "application")
        return (
            f"BatchResult(application={app!r}, solved={self.n_solved}/{len(self.results)}, "
            f"mean_objective={self.mean_objective:.4g}, mean_ms={self.mean_ms:.3f})"
        )

    def summary(self) -> str:
        n_constraints = (self.region_n_facets if self.region_n_constraints is None
                         else self.region_n_constraints)
        return (
            f"{self.info.get('application', 'application')} over {len(self.results)} "
            f"scenarios ({self.n_solved} solved)\n"
            f"    region      : {self.region_n_vertices} vertices, "
            f"{self.region_n_facets} facets -> {n_constraints} linear constraints\n"
            f"    mean objective: {self.mean_objective:.4f}\n"
            f"    mean solve time: {self.mean_ms:.3f} ms   "
            f"(total {self.total_seconds:.2f} s)"
        )


def _units(region) -> dict:
    """``{"units": ["MW" | "Mvar", ...]}`` per axis when the region carries axes."""
    axes = getattr(region, "axes", None)
    return {"units": [axis.unit for axis in axes]} if axes is not None else {}


class Application(ABC):
    """A decision problem posed over a region."""

    name: str = "application"

    @abstractmethod
    def solve(self, region: Polytope, scenario) -> SolveResult:
        """Solve one instance against ``region``."""

    def solve_batch(self, region: Polytope, scenarios) -> BatchResult:
        """Solve ``scenarios`` sequentially against ``region`` and aggregate into a :class:`BatchResult`."""
        if isinstance(scenarios, (str, bytes, bytearray)):
            raise TypeError("scenarios must be a sequence of scenarios, not str or bytes")
        results = [self.solve(region, s) for s in scenarios]
        return BatchResult(
            results=results,
            region_n_vertices=region.n_vertices,
            region_n_facets=region.n_facets,
            info={"application": self.name} | self.describe(),
        )

    def describe(self) -> dict:
        return {}


def get_application(name: str, **kwargs) -> Application:
    return APPLICATIONS.get(name)(**kwargs)


def list_applications() -> List[tuple]:
    return APPLICATIONS.describe()


def solve_lp(c: np.ndarray, A_ub: np.ndarray, b_ub: np.ndarray,
             bounds) -> tuple:
    """Solve ``min c @ x`` subject to ``A_ub @ x <= b_ub`` and ``bounds`` via HiGHS.

    Returns ``(x, objective, ok, status)``.
    """
    from scipy.optimize import linprog

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success or result.x is None or result.fun is None:
        return None, float("nan"), False, str(result.message)
    return result.x, float(result.fun), True, "optimal"
