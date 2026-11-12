"""Modelling layers for the solver-based oracles: Pyomo, or the solver's own API.

The DistFlow oracle (with ``backend="solver"``), the SOCP relaxation and the
exact oracle each pose one feasibility problem per query, and only the
injections change from one query to the next.  Each oracle accepts a
``modeling`` argument:

``"pyomo"`` (default)
    The oracle builds its own Pyomo model, and ``solver=`` picks any Pyomo
    solver that can handle it.  This is the most portable path and the
    reference the direct layers agree with.  Pyomo re-translates the whole
    model for the solver at every solve.
``"mosek"``
    MOSEK Fusion, built once, with the injections as a Fusion ``Parameter``: a
    persistent model.  Convex models only (not the exact oracle).
``"gurobi"``
    gurobipy, built once; a query only rewrites the right-hand side of the
    balance equations.  Handles the exact oracle's non-convex equality too.

The direct layers do not see the oracle's Pyomo model.  The oracle states its
problem once more, in a form any solver API can take directly
(:class:`ConicForm`)::

    minimise  c . x
    s.t.      A x = b + B u              u = [p_mw; q_mvar], one entry per bus
              lb <= x <= ub
              P_k^2 + Q_k^2 <= f_k g_k   (or = for the exact model)

where ``f_k`` and ``g_k`` are affine in one variable each (a voltage, a
squared current, or a constant).  :func:`branch_flow_form` writes that form for
all three oracles.  Additional solver backends can be registered in
:data:`MODELING_LAYERS` by subclassing :class:`DirectModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import wraps
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import numpy as np

from OpenRC.powerflow.base import SolverUnavailableError, Verdict
from OpenRC.registry import Registry

#: Direct modelling layers: ``name -> class(form, *, time_limit, options)``.
MODELING_LAYERS: Registry = Registry("modelling layer")

#: Coordinate process-global solver output capture. Pyomo operations and quiet
#: AC Iwamoto retries share this lock; independent model handles are not enough.
SOLVER_OUTPUT_LOCK = threading.RLock()
_PyomoCallable = TypeVar("_PyomoCallable", bound=Callable[..., Any])


def synchronized_pyomo(function: _PyomoCallable) -> _PyomoCallable:
    """Serialize a complete Pyomo operation, preserving its public signature.

    Guard model preparation, parameter updates, solving and result extraction
    together. Reentrancy lets helpers call other guarded helpers. This lock is
    process-local: spawned processes remain parallel, and direct modelling
    layers never acquire it. Custom Pyomo integrations can use the same guard.
    """
    @wraps(function)
    def guarded(*args, **kwargs):
        with SOLVER_OUTPUT_LOCK:
            return function(*args, **kwargs)

    return cast(_PyomoCallable, guarded)


def modeling_choices() -> Tuple[str, ...]:
    """What ``modeling=`` accepts: Pyomo, or a registered direct layer."""
    return ("pyomo",) + tuple(MODELING_LAYERS.names())


#: An affine factor ``coef * x[index] + const``; ``index`` is None for a constant.
Factor = Tuple[Optional[int], float, float]


@dataclass
class FlowCone:
    """``x[p]^2 + x[q]^2 <= f * g``, or ``==`` when ``exact`` (non-convex)."""

    p: int
    q: int
    f: Factor
    g: Factor
    exact: bool = False


@dataclass
class ConicForm:
    """One feasibility problem, in a form every direct layer accepts.

    Besides feeding persistent solver APIs, the form is an executable public
    model.  :meth:`scipy_problem` exposes its equalities as SciPy callables and
    keeps the bounds/cones in a separate acceptor: a root of ``A x = rhs`` is
    accepted as feasible only when bound and cone residuals also satisfy
    ``tolerance``.
    """

    names: List[str]
    lb: np.ndarray
    ub: np.ndarray
    A: Any                      # scipy.sparse, (rows, n)
    b: np.ndarray               # (rows,)
    B: Any                      # scipy.sparse, (rows, 2 * n_bus): rhs = b + B @ u
    c: np.ndarray               # (n,), minimised
    cones: List[FlowCone] = field(default_factory=list)

    @property
    def convex(self) -> bool:
        return not any(cone.exact for cone in self.cones)

    def rhs(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> np.ndarray:
        return self.b + self.B @ np.concatenate([p_mw, q_mvar])

    def with_case_charging(self, case: Any) -> "ConicForm":
        """Embed constant line-charging reactive offsets into ``b`` when active."""
        if case.line_charging == "omit":
            return self
        offset = np.concatenate([
            np.zeros(case.n_bus, dtype=float),
            np.asarray(case.charging_q_mvar, dtype=float),
        ])
        return ConicForm(
            names=list(self.names), lb=self.lb.copy(), ub=self.ub.copy(),
            A=self.A, b=np.asarray(self.b - self.B @ offset, dtype=float),
            B=self.B, c=self.c.copy(), cones=list(self.cones),
        )

    def initial_guess(self) -> np.ndarray:
        """A finite, bounds-respecting start for equation algorithms."""
        finite_lo = np.isfinite(self.lb)
        finite_hi = np.isfinite(self.ub)
        x0 = np.zeros(len(self.names), dtype=float)
        both = finite_lo & finite_hi
        x0[both] = 0.5 * self.lb[both] + 0.5 * self.ub[both]
        only_lo = finite_lo & ~finite_hi
        only_hi = finite_hi & ~finite_lo
        x0[only_lo] = self.lb[only_lo]
        x0[only_hi] = self.ub[only_hi]
        for cone in self.cones:
            f_idx, f_coef, f_const = cone.f
            if (cone.exact and f_idx is not None and f_coef != 0.0 and f_const == 0.0
                    and x0[cone.p] == 0.0 and x0[cone.q] == 0.0
                    and self.lb[f_idx] <= 0.0 <= self.ub[f_idx]):
                x0[f_idx] = 0.0
        return x0

    @staticmethod
    def _factor_value(x: np.ndarray, factor: Factor) -> float:
        index, coefficient, constant = factor
        return float(constant if index is None else coefficient * x[index] + constant)

    def constraint_residuals(
        self,
        x: np.ndarray,
        p_mw: np.ndarray,
        q_mvar: np.ndarray,
    ):
        """Return scaled equality, bound and cone violations for ``x``."""
        from OpenRC.powerflow.equations import ConstraintResiduals

        raw_x = np.asarray(x)
        raw_p = np.asarray(p_mw)
        raw_q = np.asarray(q_mvar)
        if raw_x.dtype == bool or raw_p.dtype == bool or raw_q.dtype == bool:
            return ConstraintResiduals(math.inf, math.inf, math.inf)
        x = np.asarray(raw_x, dtype=float)
        if x.shape != (len(self.names),) or not np.all(np.isfinite(x)):
            return ConstraintResiduals(math.inf, math.inf, math.inf)
        with np.errstate(over="ignore", invalid="ignore"):
            rhs = np.asarray(self.rhs(p_mw, q_mvar), dtype=float)
            lhs = np.asarray(self.A @ x, dtype=float)
        if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(lhs)):
            return ConstraintResiduals(math.inf, math.inf, math.inf)
        with np.errstate(over="ignore", invalid="ignore"):
            equality = float(np.max(np.abs(lhs - rhs) / (1.0 + np.maximum(
                np.abs(lhs), np.abs(rhs))))) if rhs.size else 0.0

        bound_violation = 0.0
        finite_lo = np.isfinite(self.lb)
        finite_hi = np.isfinite(self.ub)
        if np.any(finite_lo):
            bound_violation = max(
                bound_violation,
                float(np.max(np.maximum(self.lb[finite_lo] - x[finite_lo], 0.0)
                             / (1.0 + np.abs(self.lb[finite_lo])))),
            )
        if np.any(finite_hi):
            bound_violation = max(
                bound_violation,
                float(np.max(np.maximum(x[finite_hi] - self.ub[finite_hi], 0.0)
                             / (1.0 + np.abs(self.ub[finite_hi])))),
            )

        cone_violation = 0.0
        for cone in self.cones:
            f = self._factor_value(x, cone.f)
            g = self._factor_value(x, cone.g)
            with np.errstate(over="ignore", invalid="ignore"):
                lhs_cone = float(x[cone.p] ** 2 + x[cone.q] ** 2)
            rhs_cone = f * g
            scale = 1.0 + abs(lhs_cone) + abs(rhs_cone)
            if not all(math.isfinite(value) for value in (f, g, lhs_cone, rhs_cone, scale)):
                return ConstraintResiduals(equality, bound_violation, math.inf)
            if cone.exact:
                violation = abs(lhs_cone - rhs_cone) / scale
            else:
                violation = max(lhs_cone - rhs_cone, 0.0) / scale
            # A rotated cone also requires both affine factors to be
            # non-negative.  Solver APIs impose this as part of the cone domain.
            factor_violation = max(-f / (1.0 + abs(f)), -g / (1.0 + abs(g)), 0.0)
            cone_violation = max(cone_violation, violation, factor_violation)
        return ConstraintResiduals(equality, bound_violation, cone_violation)

    def scipy_problem(
        self,
        p_mw: np.ndarray,
        q_mvar: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
        tolerance: float = 1e-9,
    ):
        """Return an :class:`EquationProblem` usable by SciPy or custom code.

        ``problem.fun`` and ``problem.jac`` have the signatures expected by
        :func:`scipy.optimize.root`.  For bounded or under-determined systems,
        :func:`scipy.optimize.least_squares` can instead consume ``fun``,
        ``jac``, ``x0`` and ``bounds``.  In either case, call
        ``problem.accepts(solution.x)`` before treating a root as feasible.
        """
        from OpenRC.powerflow.equations import EquationProblem

        raw_p = np.asarray(p_mw)
        raw_q = np.asarray(q_mvar)
        if raw_p.dtype == bool or raw_q.dtype == bool:
            raise ValueError("p_mw and q_mvar must be numeric arrays, not boolean")
        p = np.asarray(raw_p, dtype=float).copy()
        q = np.asarray(raw_q, dtype=float).copy()
        expected_injections = (self.B.shape[1] // 2,)
        if p.shape != expected_injections or q.shape != expected_injections:
            raise ValueError(
                "p_mw and q_mvar must both have shape "
                f"{expected_injections}; got {p.shape} and {q.shape}"
            )
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("power injections must contain only finite values")
        rhs = np.asarray(self.rhs(p, q), dtype=float)
        if x0 is not None and np.asarray(x0).dtype == bool:
            raise ValueError("x0 must be a numeric array, not boolean")
        start = self.initial_guess() if x0 is None else np.asarray(x0, dtype=float).copy()
        expected = (len(self.names),)
        if start.shape != expected or not np.all(np.isfinite(start)):
            raise ValueError(f"x0 must be finite with shape {expected}, got {start.shape}")
        if isinstance(tolerance, (bool, np.bool_)) or tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be a positive finite number")

        exact_cones = tuple(cone for cone in self.cones if cone.exact)
        if not exact_cones:
            def fun(x: np.ndarray) -> np.ndarray:
                return np.asarray(self.A @ np.asarray(x, dtype=float) - rhs, dtype=float)

            def jac(x: np.ndarray):
                del x
                return self.A
        else:
            import scipy.sparse as sp

            n_var = len(self.names)
            n_exact = len(exact_cones)

            def fun(x: np.ndarray) -> np.ndarray:
                arr = np.asarray(x, dtype=float)
                lin = np.asarray(self.A @ arr - rhs, dtype=float)
                quad = np.empty(n_exact, dtype=float)
                for k, cone in enumerate(exact_cones):
                    f_val = self._factor_value(arr, cone.f)
                    g_val = self._factor_value(arr, cone.g)
                    quad[k] = arr[cone.p] ** 2 + arr[cone.q] ** 2 - f_val * g_val
                return np.concatenate([lin, quad])

            def jac(x: np.ndarray):
                arr = np.asarray(x, dtype=float)
                j_rows: List[int] = []
                j_cols: List[int] = []
                j_vals: List[float] = []
                for k, cone in enumerate(exact_cones):
                    f_val = self._factor_value(arr, cone.f)
                    g_val = self._factor_value(arr, cone.g)
                    j_rows.extend((k, k))
                    j_cols.extend((cone.p, cone.q))
                    j_vals.extend((2.0 * float(arr[cone.p]), 2.0 * float(arr[cone.q])))
                    if cone.f[0] is not None:
                        j_rows.append(k)
                        j_cols.append(cone.f[0])
                        j_vals.append(-cone.f[1] * g_val)
                    if cone.g[0] is not None:
                        j_rows.append(k)
                        j_cols.append(cone.g[0])
                        j_vals.append(-cone.g[1] * f_val)
                quad_jac: Any = sp.csr_matrix((j_vals, (j_rows, j_cols)), shape=(n_exact, n_var))
                return sp.vstack([self.A, quad_jac], format="csr")

        def violations(x: np.ndarray):
            return self.constraint_residuals(x, p, q)

        return EquationProblem(
            tuple(self.names), fun, jac, start, (self.lb.copy(), self.ub.copy()),
            violations, float(tolerance),
        )


class DirectModel(ABC):
    """A persistent model of one :class:`ConicForm`; ``verdict`` per query."""

    def __init__(self, form: ConicForm, *, time_limit: Optional[float] = None,
                 options: Optional[dict] = None) -> None:
        if not isinstance(form, ConicForm):
            raise TypeError("form must be a ConicForm")
        if time_limit is not None and (
                isinstance(time_limit, (bool, np.bool_))
                or not math.isfinite(time_limit)
                or time_limit <= 0.0):
            raise ValueError("time_limit must be finite and strictly positive")
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dict")
        self.form = form
        self.time_limit = None if time_limit is None else float(time_limit)
        self.options = dict(options or {})

    @abstractmethod
    def verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """FEASIBLE with a solution, INFEASIBLE with a proof, else INCONCLUSIVE."""

    def close(self) -> None:
        """Release the solver's memory (optional)."""


# ---------------------------------------------------------------------------
# The form, for the three branch-flow models
# ---------------------------------------------------------------------------
def branch_flow_form(topology, *, scale: float, v_slack: float,
                     v_bounds: Tuple[float, float], losses: str,
                     l_upper: Optional[Dict[int, float]] = None,
                     thermal: Optional[Dict[int, float]] = None,
                     thermal_voltage: Optional[float] = None,
                     balance_const: Optional[Dict[int, Tuple[float, float]]] = None,
                     drop_const: Optional[Dict[int, float]] = None,
                     base: Optional[Tuple[float, float]] = None,
                     objective: str = "none") -> ConicForm:
    """The branch-flow equations of a radial network as a :class:`ConicForm`.

    Variables: ``P_k, Q_k`` per branch, ``v_i`` (squared voltage) per non-slack
    bus and, unless ``losses == "none"``, ``l_k`` (squared current) per
    branch.  Per branch ``k`` from ``i`` to ``j``::

        P_k - sum P_down - r l_k = scale * p_j + balance_const   (and Q)
        v_i - v_j - 2 (r P_k + x Q_k) + (r^2 + x^2) l_k = drop_const

    ``losses``: ``"none"`` (LinDistFlow: no ``l``; ``thermal`` caps
    ``P^2 + Q^2 <= cap v_i``, or ``<= cap thermal_voltage`` when that
    squared voltage is given), ``"relaxed"`` (``P^2 + Q^2 <= l v_i``) or
    ``"exact"`` (``=``).  ``objective`` is ``"none"``, ``"voltage"`` (sum of
    ``v``) or ``"losses"`` (sum of ``r l``).

    Every argument is in SI per phase (W, V^2, A^2, ohm).  ``base = (s_base, v_base)``
    (VA per phase, V) rescales the system into per-unit (``z_base = v^2 / s``,
    ``i_base = s / v``), avoiding the ~10^10 coefficient dynamic range of SI
    units in interior-point solvers.
    """
    import scipy.sparse as sp

    if losses not in ("none", "relaxed", "exact"):
        raise ValueError(f"losses must be 'none', 'relaxed' or 'exact', got {losses!r}")
    if (isinstance(scale, (bool, np.bool_))
            or not isinstance(scale, (int, float, np.number))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0):
        raise ValueError("scale must be a finite positive number")
    if (isinstance(v_slack, (bool, np.bool_))
            or not isinstance(v_slack, (int, float, np.number))
            or not math.isfinite(float(v_slack))
            or float(v_slack) <= 0.0):
        raise ValueError("v_slack must be a finite positive number")
    if thermal_voltage is not None and (
            isinstance(thermal_voltage, (bool, np.bool_))
            or not isinstance(thermal_voltage, (int, float, np.number))
            or not math.isfinite(float(thermal_voltage))
            or float(thermal_voltage) <= 0.0):
        raise ValueError("thermal_voltage must be a finite positive number or None")
    if isinstance(v_bounds, (str, bytes, bytearray)):
        raise TypeError("v_bounds must be a (v_min, v_max) pair, not str or bytes")
    try:
        v_lo, v_hi = v_bounds
    except (TypeError, ValueError) as exc:
        raise ValueError("v_bounds must be a (v_min, v_max) pair") from exc
    if (isinstance(v_lo, (bool, np.bool_)) or isinstance(v_hi, (bool, np.bool_))
            or not isinstance(v_lo, (int, float, np.number))
            or not isinstance(v_hi, (int, float, np.number))
            or not (math.isfinite(float(v_lo)) and math.isfinite(float(v_hi)))
            or float(v_lo) < 0.0 or float(v_lo) > float(v_hi)):
        raise ValueError("v_bounds must be finite non-negative numbers with v_min <= v_max")
    if base is not None:
        if isinstance(base, (str, bytes, bytearray)):
            raise TypeError("base must be a (s_base, v_base) pair, not str or bytes")
        try:
            s_b, v_b = base
        except (TypeError, ValueError) as exc:
            raise ValueError("base must be a (s_base, v_base) pair") from exc
        if (isinstance(s_b, (bool, np.bool_)) or isinstance(v_b, (bool, np.bool_))
                or not isinstance(s_b, (int, float, np.number))
                or not isinstance(v_b, (int, float, np.number))
                or not (math.isfinite(float(s_b)) and math.isfinite(float(v_b)))
                or float(s_b) <= 0.0 or float(v_b) <= 0.0):
            raise ValueError("base must be finite positive (s_base, v_base) numbers")
        s_base, v_base = float(s_b), float(v_b)
    else:
        s_base, v_base = 1.0, 1.0
    branches = list(topology.branches)
    buses = list(topology.buses)
    n_bus = topology.n_bus
    z_base, i_base = v_base ** 2 / s_base, s_base / v_base
    scale = float(scale) / s_base
    v_slack = float(v_slack) / v_base ** 2
    v_bounds = (float(v_lo) / v_base ** 2, float(v_hi) / v_base ** 2)
    names: List[str] = []

    def add(name: str) -> int:
        names.append(name)
        return len(names) - 1

    iP = {br.index: add(f"P[{br.index}]") for br in branches}
    iQ = {br.index: add(f"Q[{br.index}]") for br in branches}
    iV = {bus: add(f"v[{bus}]") for bus in buses}
    with_l = losses != "none"
    iL = {br.index: add(f"l[{br.index}]") for br in branches} if with_l else {}
    n = len(names)
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    for bus in buses:
        lb[iV[bus]], ub[iV[bus]] = v_bounds
    for k, idx in iL.items():
        lb[idx] = 0.0
        if l_upper and k in l_upper:
            ub[idx] = l_upper[k] / i_base ** 2

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    b: List[float] = []
    b_rows: List[int] = []
    b_cols: List[int] = []
    b_vals: List[float] = []

    def row(entries, const, injection=None) -> None:
        r_index = len(b)
        for col, val in entries:
            rows.append(r_index)
            cols.append(col)
            vals.append(val)
        b.append(const)
        if injection is not None:
            b_rows.append(r_index)
            b_cols.append(injection)
            b_vals.append(scale)

    cones: List[FlowCone] = []
    for br in branches:
        k, child, parent = br.index, br.child, br.parent
        r, x = br.r_ohm / z_base, br.x_ohm / z_base
        down = topology.children_of[child]
        c_p, c_q = balance_const.get(k, (0.0, 0.0)) if balance_const else (0.0, 0.0)
        c_p, c_q = c_p / s_base, c_q / s_base
        loss_p = [(iL[k], -r)] if with_l else []
        loss_q = [(iL[k], -x)] if with_l else []
        row([(iP[k], 1.0)] + [(iP[d.index], -1.0) for d in down] + loss_p, c_p, child)
        row([(iQ[k], 1.0)] + [(iQ[d.index], -1.0) for d in down] + loss_q, c_q,
            n_bus + child)
        entries = [(iV[child], -1.0), (iP[k], -2.0 * r), (iQ[k], -2.0 * x)]
        if with_l:
            entries.append((iL[k], r * r + x * x))
        const = (drop_const.get(k, 0.0) if drop_const else 0.0) / v_base ** 2
        if parent == topology.slack:
            const -= v_slack
        else:
            entries.append((iV[parent], 1.0))
        row(entries, const)

        v_parent: Factor = ((None, 0.0, v_slack) if parent == topology.slack
                            else (iV[parent], 1.0, 0.0))
        if with_l:
            cones.append(FlowCone(iP[k], iQ[k], (iL[k], 1.0, 0.0), v_parent,
                                  exact=losses == "exact"))
        elif thermal and k in thermal:
            cap_at: Factor = (v_parent if thermal_voltage is None
                              else (None, 0.0, float(thermal_voltage) / v_base ** 2))
            cones.append(FlowCone(iP[k], iQ[k], (None, 0.0, thermal[k] / i_base ** 2),
                                  cap_at))

    c = np.zeros(n)
    if objective == "voltage":
        c[[iV[bus] for bus in buses]] = 1.0
    elif objective == "losses":
        for br in branches:
            c[iL[br.index]] = br.r_ohm / z_base
    elif objective != "none":
        raise ValueError(f"unknown objective {objective!r}")
    m = len(b)
    return ConicForm(
        names=names, lb=lb, ub=ub,
        A=sp.csr_matrix((vals, (rows, cols)), shape=(m, n)), b=np.asarray(b, dtype=float),
        B=sp.csr_matrix((b_vals, (b_rows, b_cols)), shape=(m, 2 * n_bus)),
        c=c, cones=cones)


# ---------------------------------------------------------------------------
# MOSEK Fusion
# ---------------------------------------------------------------------------
class MosekFusionModel(DirectModel):
    """MOSEK Fusion model with the injections as a parameter (convex forms only)."""

    def __init__(self, form: ConicForm, *, time_limit: Optional[float] = None,
                 options: Optional[dict] = None) -> None:
        super().__init__(form, time_limit=time_limit, options=options)
        if not form.convex:
            raise ValueError("MOSEK solves convex problems only; the exact model needs "
                             "modeling='gurobi' (or Pyomo with a global solver)")
        try:
            import mosek.fusion as mf
        except ImportError as exc:  # environment dependent
            raise SolverUnavailableError(
                "modeling='mosek' needs the mosek package (MOSEK Fusion), which is "
                "not installed"
            ) from exc

        self._mf = mf
        M = mf.Model()
        try:
            n = len(form.names)
            x = M.variable("x", n, mf.Domain.unbounded())
            for bound, domain in ((form.lb, mf.Domain.greaterThan),
                                  (form.ub, mf.Domain.lessThan)):
                idx = np.flatnonzero(np.isfinite(bound))
                if len(idx):
                    M.constraint(x.pick(idx.astype(np.int32).tolist()),
                                 domain(bound[idx].tolist()))
            u = M.parameter("u", form.B.shape[1])
            A, B = form.A.tocoo(), form.B.tocoo()
            A_f = mf.Matrix.sparse(A.shape[0], A.shape[1], A.row.tolist(), A.col.tolist(),
                                   A.data.tolist())
            B_f = mf.Matrix.sparse(B.shape[0], B.shape[1], B.row.tolist(), B.col.tolist(),
                                   B.data.tolist())
            M.constraint(mf.Expr.sub(mf.Expr.mul(A_f, x), mf.Expr.mul(B_f, u)),
                         mf.Domain.equalsTo(form.b.tolist()))
            if form.cones:
                M.constraint(self._cone_rows(form, x, n), mf.Domain.inRotatedQCone())
            M.objective(mf.ObjectiveSense.Minimize, mf.Expr.dot(form.c.tolist(), x))
            if time_limit is not None:
                M.setSolverParam("optimizerMaxTime", float(time_limit))
            for key, value in self.options.items():
                M.setSolverParam(key, value)
        except BaseException:
            M.dispose()
            raise
        self._model, self._u = M, u

    def _cone_rows(self, form: ConicForm, x, n: int):
        """``[f/2, g, P, Q]`` per cone, stacked: MOSEK's rotated cone is 2 a b >= |c|^2."""
        mf = self._mf
        m = len(form.cones)

        def affine(factor_of, weight):
            rows, cols, vals, const = [], [], [], np.zeros(m)
            for i, cone in enumerate(form.cones):
                index, coef, c0 = factor_of(cone)
                if index is not None:
                    rows.append(i)
                    cols.append(index)
                    vals.append(weight * coef)
                const[i] = weight * c0
            mat = mf.Matrix.sparse(m, n, rows, cols, vals)
            return mf.Expr.add(mf.Expr.mul(mat, x), const.tolist())

        pick = [x.pick([cone.p for cone in form.cones]), x.pick([cone.q for cone in form.cones])]
        return mf.Expr.hstack([affine(lambda c: c.f, 0.5), affine(lambda c: c.g, 1.0)] + pick)

    def verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        mf = self._mf
        self._u.setValue(np.concatenate([p_mw, q_mvar]).tolist())
        try:
            self._model.solve()
            status = self._model.getProblemStatus()
        except mf.FusionException:
            return Verdict.INCONCLUSIVE
        if status == mf.ProblemStatus.PrimalAndDualFeasible:
            return Verdict.FEASIBLE
        if status == mf.ProblemStatus.PrimalInfeasible:
            return Verdict.INFEASIBLE
        return Verdict.INCONCLUSIVE

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        model = getattr(self, "_model", None)
        if model is not None:
            model.dispose()


# ---------------------------------------------------------------------------
# gurobipy
# ---------------------------------------------------------------------------
class GurobiModel(DirectModel):
    """gurobipy model; a query rewrites the balance right-hand sides."""

    def __init__(self, form: ConicForm, *, time_limit: Optional[float] = None,
                 options: Optional[dict] = None) -> None:
        super().__init__(form, time_limit=time_limit, options=options)
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:  # environment dependent
            raise SolverUnavailableError(
                "modeling='gurobi' needs the gurobipy package, which is not installed"
            ) from exc

        self._GRB = GRB
        self._env = gp.Env(empty=True)
        m = None
        try:
            self._env.setParam("OutputFlag", 0)
            self._env.start()
            m = gp.Model(env=self._env)
            x = m.addMVar(len(form.names), lb=form.lb, ub=form.ub)
            self._eq = m.addMConstr(form.A, x, GRB.EQUAL, form.b)
            xs = x.tolist()

            def expr(factor: Factor):
                index, coef, const = factor
                return const if index is None else coef * xs[index] + const

            for cone in form.cones:
                # A rotated cone requires f >= 0 and g >= 0 as well as the
                # quadratic inequality.  The latter alone also accepts two
                # negative factors.  Built-in forms already imply the domains
                # through variable bounds; custom affine factors may not.
                for factor in (cone.f, cone.g):
                    index, coef, const = factor
                    implied = const >= 0.0 and (
                        index is None or coef == 0.0
                        or (coef > 0.0 and form.lb[index] >= 0.0)
                        or (coef < 0.0 and form.ub[index] <= 0.0)
                    )
                    if not implied:
                        m.addConstr(gp.LinExpr(expr(factor)) >= 0.0)
                lhs = xs[cone.p] * xs[cone.p] + xs[cone.q] * xs[cone.q]
                rhs = expr(cone.f) * expr(cone.g)
                m.addQConstr(lhs == rhs if cone.exact else lhs <= rhs)
            m.setObjective(form.c @ x, GRB.MINIMIZE)
            if not form.convex:
                m.Params.NonConvex = 2
            m.Params.DualReductions = 0          # tell infeasible from unbounded
            if time_limit is not None:
                m.Params.TimeLimit = float(time_limit)
            for key, value in self.options.items():
                m.setParam(key, value)
        except BaseException:
            # Environment release must not be skipped if model disposal also
            # fails while unwinding an incomplete construction.
            with ExitStack() as resources:
                resources.callback(self._env.dispose)
                if m is not None:
                    resources.callback(m.dispose)
                raise
        self._model = m

    def verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        GRB = self._GRB
        self._eq.setAttr("RHS", self.form.rhs(p_mw, q_mvar))
        self._model.reset(0)
        self._model.optimize()
        status = self._model.Status
        if status == GRB.OPTIMAL:
            return Verdict.FEASIBLE
        if status == GRB.INFEASIBLE:
            return Verdict.INFEASIBLE
        return Verdict.INCONCLUSIVE

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        model = getattr(self, "_model", None)
        env = getattr(self, "_env", None)
        with ExitStack() as resources:
            if env is not None:
                resources.callback(env.dispose)
            if model is not None:
                resources.callback(model.dispose)


MODELING_LAYERS.add("mosek", MosekFusionModel,
                    doc="MOSEK Fusion, persistent; the injections are a parameter")
MODELING_LAYERS.add("gurobi", GurobiModel,
                    doc="gurobipy, persistent; handles the exact (non-convex) model too")


def direct_model(modeling: str, form: ConicForm, **kwargs) -> DirectModel:
    """The registered layer ``modeling``, built for ``form``."""
    return MODELING_LAYERS.get(modeling)(form, **kwargs)


def check_modeling(modeling: str) -> None:
    if modeling not in modeling_choices():
        raise ValueError(f"modeling must be one of {modeling_choices()}, got {modeling!r}")


__all__ = ["MODELING_LAYERS", "ConicForm", "DirectModel", "FlowCone", "GurobiModel",
           "MosekFusionModel", "branch_flow_form", "check_modeling", "direct_model",
           "modeling_choices"]
