"""Global operating-limit bounds, existence of solutions, inner-region certification.

Monte Carlo sampling (:func:`~OpenRC.validation.quality.evaluate_region`) can
detect violations at tested points but cannot certify their absence over a
continuous polytope, such as a narrow infeasible corner.  Certifying an inner
(conservative) approximation requires global bounds over all :math:`x \\in R`.

:func:`certify_inner_region` provides the global part of that argument.  It
poses one optimisation per operating limit over the **entire polytope**::

    maximise   loading of branch k          (or: min / max voltage at bus i)
    subject to x in region                  (A x <= b, the region's facets)
               exact branch-flow equations  (with the injection at x)
               voltages inside a window     (vm_min - margin, vm_max + margin)

and solves it to global optimality with Gurobi's spatial branch-and-bound
(``NonConvex=2``).  The solver returns a certified dual bound on the objective
alongside its incumbent solution.  When every bound respects its limit, no
solution within the voltage window violates any operating constraint.  When a
bound exceeds its limit, the solver's incumbent provides a concrete *witness*: a
point of the region together with an exact power-flow solution that violates
the limit.  Because non-linear power flow can admit multiple roots, this refutes
the universal limit bound while leaving injection feasibility to witness
confirmation: another solution at the same injection may satisfy every
limit.  A region is refuted only when an injection is proven to have no safe
solution, or no solution exists anywhere in the voltage window.

What the certificate covers, precisely
--------------------------------------
The equations and the per-unit scaling are those of
:class:`~OpenRC.powerflow.exact.ExactDistFlowOracle`.  Two facts are proven
separately:

1. **Limits.**  Every branch-flow solution whose voltages lie inside the window
   satisfies every thermal and voltage limit.  This is the global part.
2. **Existence**, in one of two strengths:

   a. *At every point* (``existence="contraction"``, the default): the
      fixed-point form of power flow is a contraction at every vertex, hence -
      the condition being convex - at every point, which guarantees a solution
      in a voltage band inside the window
      (:mod:`OpenRC.validation.existence`).  Steps 1 and 2a together prove the
      region inner with no further assumption; the status is ``certified``
      with scope ``"inner"``.
   b. *At sampled points*: a solution exists at points spread uniformly over
      the region's *surface*, each decided by the escalating
      :class:`~OpenRC.validation.pointcheck.PointVerifier` (AC power flow, then
      the global exact solve, then the method's sub-models).  The vertices are
      not re-checked: the construction already evaluated them, and a surface
      sample probes the facets between them, which is where a region built
      from linearised models is most likely to overreach.  The surface sample
      is drawn in both cases; after 2a it is an independent cross-check.

When 2a does not hold (heavily loaded or long feeders), 2b covers the whole
region under one standard assumption: the physical (high-voltage) solution
varies continuously over the region without a voltage-collapse fold inside the
window.  Under that assumption the solution cannot leave the window without
passing through its margin band, which step 1 would have seen as a voltage
violation.  A voltage window cannot in general exclude every alternative
power-flow root; an unsafe root gives ``limit-violated`` unless the all-limits
exact solve proves the injection infeasible.  Such a result has scope
``"conditional_inner"``.

Relax first, then solve globally
--------------------------------
A global solve per limit is expensive (seconds each on a few hundred buses,
some time out).  The SOC relaxation - ``l v >= P^2 + Q^2`` instead of ``=`` -
contains every exact solution, so its optimum is a valid bound too, and a
convex solve costs about a hundredth of a second.  On its own it settles only
the upper voltage limits: it can inflate the losses, which lowers voltages and
raises currents without limit.  A few rounds of **bound tightening** fix that:
bound every ``P`` and ``Q`` over the relaxation, and add the valid cut
``v_lo l <= secant(P) + secant(Q)`` (``P^2 <= (a+b) P - a b`` on ``[a, b]``),
which caps the losses at what the flows allow.  Every bound and cut holds for
every exact solution, so the tightened relaxation is still a relaxation.  A
limit it settles is certified; the others - normally only the violated ones and
a few close calls - go to the global solve, which also starts from the
tightened bounds.  Those global solves can run in parallel threads.  A
settled limit reports the relaxation's bound, an over-estimate; the limits
that could still hold the reported worst case of their family are therefore
re-solved globally, so the reported worst cases are exact.  Termination uses a
fixed round count rather than wall-clock time so bounds are deterministic across
machines.

"Proven" is up to the solver's feasibility tolerance (about 1e-6 per unit);
limits are compared with the same tolerance.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from OpenRC.geometry.polytope import Polytope
from OpenRC.validation.existence import ExistenceProof
from OpenRC.powerflow.base import SolverUnavailableError, Verdict
from OpenRC.validation.pointcheck import PointOutcome, PointVerification

#: Outcomes of one limit check.
CERTIFIED, LIMITS_CERTIFIED, VIOLATED, UNDECIDED = (
    "certified", "limits-certified", "violated", "undecided"
)
#: A universal limit bound failed, without proving an injection infeasible.
LIMIT_VIOLATED = "limit-violated"


@dataclass
class LimitCheck:
    """The worst case of one operating limit over the whole region.

    Values are in natural units: a thermal check reports the current as a
    fraction of the rating, a voltage check reports per-unit magnitude.
    """

    kind: str                    # "thermal", "v_min" or "v_max"
    element: int                 # branch index (thermal) or bus index (voltage)
    limit: float
    bound: float                 # proven worst case (nan if the solver gave none)
    incumbent: float             # worst case actually attained by a solution found
    status: str                  # CERTIFIED, VIOLATED or UNDECIDED
    solver_status: str
    runtime: float
    witness: Optional[np.ndarray] = None   # region point attaining ``incumbent``
    note: str = ""
    #: No operating point satisfying all limits exists at ``witness``.  An
    #: unsafe solution alone does not prove this when power flow has many roots.
    witness_infeasible: bool = False

    @property
    def name(self) -> str:
        return f"{self.kind}[{'branch' if self.kind == 'thermal' else 'bus'} {self.element}]"

    @property
    def margin(self) -> float:
        """Proven slack to the limit; negative when the bound breaches it."""
        if self.kind == "v_min":
            return self.bound - self.limit
        return self.limit - self.bound

    def as_dict(self) -> dict:
        return {
            "check": self.name, "limit": self.limit, "bound": self.bound,
            "incumbent": self.incumbent, "margin": self.margin,
            "status": self.status, "solver_status": self.solver_status,
            "runtime_s": self.runtime,
            "witness": None if self.witness is None else self.witness.tolist(),
            "note": self.note, "witness_infeasible": self.witness_infeasible,
        }


@dataclass
class InnerCertificate:
    """Result of :func:`certify_inner_region`."""

    case: str
    axes: List[str]
    voltage_window: Tuple[float, float]
    checks: List[LimitCheck]
    surface_checks: List[PointVerification]
    runtime: float
    no_solution: bool = False    # no power-flow solution anywhere in the window
    notes: List[str] = field(default_factory=list)
    #: The contraction proof of existence (:mod:`OpenRC.validation.existence`),
    #: when it was attempted.
    existence: Optional["ExistenceProof"] = None
    #: The limit families that were bounded; an inner-region claim needs all three.
    families: Tuple[str, ...] = ("thermal", "v_min", "v_max")

    # ------------------------------------------------------------- verdicts
    @property
    def all_families(self) -> bool:
        """Whether thermal, lower- and upper-voltage limits were all bounded."""
        return set(self.families) >= {"thermal", "v_min", "v_max"}

    @property
    def violations(self) -> List[LimitCheck]:
        return [c for c in self.checks if c.status == VIOLATED]

    @property
    def undecided(self) -> List[LimitCheck]:
        return [c for c in self.checks if c.status == UNDECIDED]

    @property
    def infeasible_witnesses(self) -> List[LimitCheck]:
        """Checks whose injection, rather than just one solution, is infeasible."""
        return [c for c in self.checks if c.witness_infeasible]

    @property
    def surface_points(self) -> np.ndarray:
        if not self.surface_checks:
            return np.empty((0, len(self.axes)), dtype=float)
        return np.array([c.point for c in self.surface_checks], dtype=float)

    @property
    def surface_infeasible(self) -> List[PointVerification]:
        """Surface points proven infeasible (by the exact solve or the relaxation)."""
        return [c for c in self.surface_checks if c.outcome is PointOutcome.INFEASIBLE]

    @property
    def surface_computationally_infeasible(self) -> List[PointVerification]:
        """Surface points no stage could settle; recorded with their worst overrun."""
        return [c for c in self.surface_checks
                if c.outcome is PointOutcome.COMPUTATIONALLY_INFEASIBLE]

    @property
    def status(self) -> str:
        """Overall result, including the scope of the checks performed.

        *Refuted* needs evidence: an injection proven infeasible, a surface
        point proven infeasible, or no solution in the window at all.  An
        unsafe alternative power-flow solution gives *limit-violated* only.
        Time-outs and inconclusive surface points yield ``"undecided"`` rather
        than ``"certified"``.  *Certified* needs the
        limit bounds plus existence - proven at every point by contraction, or
        at the surface sample.  With neither, or when a limit family was left
        out of ``limits``, the result is ``"limits-certified"``.
        """
        if self.no_solution or self.infeasible_witnesses or self.surface_infeasible:
            return "refuted"
        if self.violations:
            return LIMIT_VIOLATED
        if (not self.checks or self.undecided
                or self.surface_computationally_infeasible):
            return UNDECIDED
        if self.all_families and (self.existence_proven or self.surface_checks):
            return CERTIFIED
        return LIMITS_CERTIFIED

    @property
    def existence_proven(self) -> bool:
        """Whether a solution is proven to exist at every point of the region."""
        return bool(self.existence is not None and getattr(self.existence, "proven", False))

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED

    @property
    def limits_certified(self) -> bool:
        """Whether every requested global limit bound is settled and safe."""
        return bool(self.checks and not self.no_solution
                    and not self.violations and not self.undecided)

    @property
    def scope(self) -> str:
        """Coverage: ``"inner"`` (existence proven everywhere), ``"conditional_inner"``
        (existence sampled, continuity assumed) or ``"limits_only"``."""
        if not self.all_families:
            return "limits_only"
        if self.existence_proven:
            return "inner"
        return "conditional_inner" if self.surface_checks else "limits_only"

    # ---------------------------------------------------------- worst cases
    def _worst(self, kind: str) -> Optional[LimitCheck]:
        rows = [c for c in self.checks if c.kind == kind and math.isfinite(c.bound)]
        if not rows:
            return None
        pick = min if kind == "v_min" else max
        return pick(rows, key=lambda c: c.bound)

    @property
    def worst_loading(self) -> Optional[LimitCheck]:
        return self._worst("thermal")

    @property
    def worst_v_min(self) -> Optional[LimitCheck]:
        return self._worst("v_min")

    @property
    def worst_v_max(self) -> Optional[LimitCheck]:
        return self._worst("v_max")

    # --------------------------------------------------------------- output
    def summary(self) -> str:
        label = {"inner": "inner-region", "conditional_inner": "conditional inner-region",
                 "limits_only": "global limit"}[self.scope]
        lines = [f"{label} certificate for {self.case} over "
                 f"{', '.join(self.axes)}: {self.status.upper()}"]
        if self.no_solution:
            lines.append("  no power-flow solution exists inside the voltage window "
                         f"{self.voltage_window} for any point of the region")
        loading, low, high = self.worst_loading, self.worst_v_min, self.worst_v_max
        if loading is not None:
            lines.append(f"  worst loading      {100 * loading.bound:8.3f} %   "
                         f"(limit 100 %, {loading.name})")
        if low is not None:
            lines.append(f"  lowest voltage     {low.bound:8.5f} pu  "
                         f"(limit {low.limit:.3f}, {low.name})")
        if high is not None:
            lines.append(f"  highest voltage    {high.bound:8.5f} pu  "
                         f"(limit {high.limit:.3f}, {high.name})")
        n_ok = sum(c.status == CERTIFIED for c in self.checks)
        lines.append(f"  limits proven      {n_ok} / {len(self.checks)}  "
                     f"(voltage window {self.voltage_window[0]:.2f}-"
                     f"{self.voltage_window[1]:.2f} pu)")
        if self.existence is not None:
            lines.append("  existence          " + self.existence.describe())
        if self.surface_checks:
            lines.append("  surface points     " + _outcome_counts(self.surface_checks))
        for limit_check in self.violations:
            witness = ("unavailable" if limit_check.witness is None
                       else str(np.round(limit_check.witness, 4).tolist()))
            if limit_check.solver_status == "split_infeasible":
                lines.append(f"  INFEASIBLE injection {witness}: no solution inside the "
                             "voltage window on part of the region")
                continue
            lines.append(f"  VIOLATED {limit_check.name}: worst "
                         f"{limit_check.incumbent:.5f} vs limit "
                         f"{limit_check.limit:.5f} at {witness}"
                         + (f"  [{limit_check.note}]" if limit_check.note else ""))
        if self.status == LIMIT_VIOLATED:
            lines.append("  note: an unsafe solution disproves the universal limit bound; "
                         "no injection has been proven infeasible, so the inner-region "
                         "claim remains undecided")
        for limit_check in self.undecided:
            lines.append(f"  undecided {limit_check.name}: bound "
                         f"{limit_check.bound:.5f}, attained "
                         f"{limit_check.incumbent:.5f} ({limit_check.solver_status})")
        for point_check in self.surface_infeasible[:5]:
            lines.append(f"  surface point proven infeasible ({point_check.stage}): "
                         f"{np.round(point_check.point, 4).tolist()}"
                         + (f"; {point_check.worst.describe()}" if point_check.worst else ""))
        for point_check in self.surface_computationally_infeasible[:5]:
            lines.append(f"  surface point computationally infeasible: "
                         f"{np.round(point_check.point, 4).tolist()}"
                         + (f"; {point_check.worst.describe()}"
                            if point_check.worst else ""))
        lines.extend(f"  note: {n}" for n in self.notes)
        lines.append(f"  runtime            {self.runtime:.1f} s")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "case": self.case, "axes": self.axes, "status": self.status,
            "target_model": "branch-flow",
            "scope": self.scope,
            "families": list(self.families),
            "voltage_window": list(self.voltage_window),
            "no_solution": self.no_solution,
            "checks": [c.as_dict() for c in self.checks],
            "surface_checks": [c.as_dict() for c in self.surface_checks],
            "existence": None if self.existence is None else self.existence.as_dict(),
            "runtime_s": self.runtime, "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The certifier
# ---------------------------------------------------------------------------
def certify_inner_region(
    region,
    case,
    axes=None,
    *,
    window_margin: float = 0.15,
    limits: Sequence[str] = ("thermal", "v_min", "v_max"),
    surface_points: int = 32,
    time_limit: float = 60.0,
    stop_at_limit: bool = False,
    confirm_witnesses: bool = True,
    tol: float = 1e-6,
    threads: int = 0,
    rng=0,
    point_verifier=None,
    existence: str = "contraction",
    split_depth: int = 2,
    relaxation_rounds: Optional[int] = 3,
    workers: int = 1,
) -> InnerCertificate:
    """Certify global limits and prove (or sample) existence of solutions.

    The limit bounds are global.  Existence is proven at every point by the
    contraction condition whenever it holds; the result then has scope
    ``"inner"`` and needs no assumption.  Otherwise the surface sample and the
    continuity/no-fold assumption described in the module documentation give
    a ``"conditional_inner"`` result, and with ``surface_points=0`` only
    ``"limits-certified"``.

    Parameters
    ----------
    region:
        A full-dimensional :class:`~OpenRC.geometry.polytope.Polytope`.
    case, axes:
        The network and the axes of the region's coordinates.  ``axes`` defaults
        to ``region.axes``.
    window_margin:
        How far (per unit) outside ``[vm_min, vm_max]`` the voltage window
        reaches.  It must be positive - the band between the limit and the
        window edge is where a violation shows up.  No choice of width generally
        guarantees exclusion of all alternative roots.  0.15 suits
        distribution feeders.  The window is clipped to ``[0.5, 1.5]`` only
        where that still strictly contains the allowed voltage range; unusual
        case limits expand it so no permitted solution is excluded.
    limits:
        Which families of limits to check.  Thermal checks skip unrated branches,
        which have no limit.
    surface_points:
        Points drawn uniformly from the region's surface for the existence
        check.  0 skips it; the certificate then covers the limits only, and a
        note says so.
    time_limit:
        Seconds per limit.  A time-out still yields a valid bound; the check is
        *undecided* only when that bound does not settle the question.
    stop_at_limit:
        Stop each solve as soon as its outcome is decided instead of computing
        the exact worst case.  Much faster on large networks; the reported
        bounds are then only as tight as needed.
    confirm_witnesses:
        At each violation witness, solve the exact branch-flow model with all
        limits imposed.  Only proven infeasibility refutes the inner-region
        claim; a safe alternative solution leaves the result ``limit-violated``.
        Also run an independent AC power flow as a diagnostic cross-check.
        Queries reuse their oracles and have the same ``time_limit``.
    tol:
        Tolerance on the squared quantities (loading ratio, ``v^2``).
    threads:
        Gurobi threads (0 = solver default).
    rng:
        Seed for the surface sample.
    point_verifier:
        The :class:`~OpenRC.validation.pointcheck.PointVerifier` that decides
        each surface point (AC, then exact, then the method's sub-models, then
        the worst overrun).  Its target must match this certificate's exact
        branch-flow physics.  Defaults to one matched to the region's method
        with ``target_model="branch-flow"``.  A custom verifier without a
        ``reference_oracle`` must explicitly declare that target.
    existence:
        ``"contraction"`` (default) attempts the proof of existence at every
        point (:func:`~OpenRC.validation.existence.prove_existence`), which
        costs one small matrix product per vertex; ``"surface"`` relies on the
        surface sample alone.
    split_depth:
        A limit left undecided (time-out or numerical trouble) is re-solved on
        the two halves of the region, split across its widest coordinate, and
        so on up to this depth (at most ``2**split_depth`` pieces).  The worst
        case over the region is the worst over the pieces, so the pieces'
        bounds combine into a valid bound for the whole region; smaller pieces
        give tighter relaxations and often settle what the whole could not.
        ``0`` disables splitting.  Settled checks are never split.
    relaxation_rounds:
        Screen every limit with the SOC relaxation first, and solve globally
        only the limits it cannot settle (see the module documentation).  The
        relaxation is tightened between screens for at most this many rounds:
        the first always (the plain relaxation settles no current or
        lower-voltage limit), each further one while the previous round
        settled at least one more limit.  Stopping by iteration count rather
        than wall-clock time keeps the bounds deterministic.  ``0`` screens with the
        plain relaxation, which settles upper voltage limits only.  ``None``
        disables the screen: every limit is solved globally, and every bound is
        the exact worst case rather than a valid over-estimate of it.
    workers:
        Global solves run in this many threads, each with its own Gurobi
        environment and model.  With ``threads=0`` each solve then gets an
        equal share of the cores.
    """
    start = time.perf_counter()
    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    axes = axes if axes is not None else region.axes
    if axes is None:
        raise ValueError("the region carries no axes; pass them explicitly")
    region.validate_axes(axes)
    if isinstance(limits, (str, bytes, bytearray)):
        raise TypeError("limits must be a sequence of limit family names, not a string")
    limits = tuple(limits)
    if not limits:
        raise ValueError("limits must contain at least one limit family")
    unknown = set(limits) - {"thermal", "v_min", "v_max"}
    if unknown:
        raise ValueError(f"unknown limit families {sorted(unknown)}")
    if (isinstance(window_margin, (bool, np.bool_))
            or not isinstance(window_margin, (int, float, Real))
            or not np.isfinite(float(window_margin)) or float(window_margin) <= 0.0):
        raise ValueError("window_margin must be finite and strictly positive")
    if (isinstance(surface_points, (bool, np.bool_))
            or not isinstance(surface_points, (int, float, Real))
            or not np.isfinite(float(surface_points))
            or int(float(surface_points)) != float(surface_points) or float(surface_points) < 0):
        raise ValueError("surface_points must be a non-negative integer")
    if (isinstance(time_limit, (bool, np.bool_))
            or not isinstance(time_limit, (int, float, Real))
            or not np.isfinite(float(time_limit)) or float(time_limit) <= 0.0):
        raise ValueError("time_limit must be finite and strictly positive")
    if (isinstance(tol, (bool, np.bool_))
            or not isinstance(tol, (int, float, Real))
            or not np.isfinite(float(tol)) or float(tol) < 0.0):
        raise ValueError("tol must be finite and non-negative")
    if (isinstance(threads, (bool, np.bool_))
            or not isinstance(threads, (int, float, Real))
            or not np.isfinite(float(threads))
            or int(float(threads)) != float(threads) or float(threads) < 0):
        raise ValueError("threads must be a non-negative integer")
    if existence not in ("contraction", "surface"):
        raise ValueError("existence must be 'contraction' or 'surface'")
    if (isinstance(split_depth, (bool, np.bool_))
            or not isinstance(split_depth, (int, float, Real))
            or not np.isfinite(float(split_depth))
            or int(float(split_depth)) != float(split_depth) or float(split_depth) < 0):
        raise ValueError("split_depth must be a non-negative integer")
    if relaxation_rounds is not None and (
            isinstance(relaxation_rounds, (bool, np.bool_))
            or not isinstance(relaxation_rounds, (int, float, Real))
            or not np.isfinite(float(relaxation_rounds))
            or int(float(relaxation_rounds)) != float(relaxation_rounds)
            or float(relaxation_rounds) < 0):
        raise ValueError("relaxation_rounds must be None or a non-negative integer")
    if (isinstance(workers, (bool, np.bool_))
            or not isinstance(workers, (int, float, Real))
            or not np.isfinite(float(workers))
            or int(float(workers)) != float(workers) or float(workers) < 1):
        raise ValueError("workers must be a positive integer")
    if not isinstance(stop_at_limit, (bool, np.bool_)):
        raise ValueError("stop_at_limit must be a boolean")
    if not isinstance(confirm_witnesses, (bool, np.bool_)):
        raise ValueError("confirm_witnesses must be a boolean")
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must not be a boolean")
    if point_verifier is not None and not callable(getattr(point_verifier, "verify_many", None)):
        raise TypeError("point_verifier must be a PointVerifier or an object with a .verify_many() method")
    from OpenRC.cases.base import validate_case_configuration

    validate_case_configuration(case)
    if not case.is_radial:
        raise ValueError(f"{case.name!r} is not radial; the branch-flow certificate "
                         "needs a radial network")
    if point_verifier is not None:
        _validate_surface_reference(point_verifier, case)
    window_margin = float(window_margin)
    time_limit = float(time_limit)
    tol = float(tol)
    threads = int(float(threads))
    # No solution in this window can refute a region only if the window
    # contains *every* permitted voltage.  The strict margin is also needed
    # by the conditional continuation argument: a solution leaving the window
    # must first violate a voltage limit.  Keep the usual numerical clips for
    # distribution feeders, expanding them when the case's limits require it.
    floor = 0.5 if case.vm_min_pu > 0.5 else 0.5 * case.vm_min_pu
    window = (max(floor, case.vm_min_pu - window_margin),
              min(1.5, case.vm_max_pu + window_margin) if case.vm_max_pu < 1.5
              else case.vm_max_pu + window_margin)

    # Defer gurobipy import until arguments and network radiality are validated.
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # environment dependent
        raise SolverUnavailableError("certify_inner_region needs gurobipy (a global "
                                     "non-convex QCQP solver)") from exc

    todo = _limit_list(case, limits, GRB)
    solve_kw = dict(time_limit=time_limit, threads=threads, tol=tol,
                    stop_at_limit=bool(stop_at_limit), split_depth=int(split_depth))
    done: Dict[int, LimitCheck] = {}
    pending = list(range(len(todo)))
    bounds: Optional[_Bounds] = None
    no_solution = False
    if relaxation_rounds is not None:
        done, pending, bounds, no_solution = _relax_stage(
            gp, GRB, region, case, axes, window, todo, int(relaxation_rounds), tol,
            int(workers))
    n_relaxed = sum(c.solver_status == "relaxation" for c in done.values())
    if not no_solution:
        exact, no_solution = _solve_globally(
            gp, GRB, region, case, axes, window, [todo[i] for i in pending], bounds,
            workers=int(workers), **solve_kw)
        for i, check in zip(pending, exact):
            if check is not None and check.solver_status != "infeasible":
                done[i] = check
    n_extremes = 0
    if relaxation_rounds is not None and not no_solution and not stop_at_limit:
        n_extremes, no_solution = _exact_extremes(
            gp, GRB, region, case, axes, window, todo, done, bounds, solve_kw)
        n_relaxed = sum(c.solver_status == "relaxation" for c in done.values())
    checks = [done[i] for i in sorted(done)]

    cert = InnerCertificate(
        case=case.name, axes=[a.label for a in axes], voltage_window=window,
        checks=checks, surface_checks=[], runtime=0.0, no_solution=no_solution,
        families=limits,
    )
    unchecked = [family for family in ("thermal", "v_min", "v_max") if family not in limits]
    if unchecked:
        cert.notes.append(f"limit families {unchecked} were not checked; the certificate "
                          "covers the requested limits only")
    if relaxation_rounds is not None and not no_solution:
        tightened = (f" after {bounds.rounds} bound-tightening round(s) "
                     f"({bounds.seconds:.1f} s)" if bounds is not None else "")
        cert.notes.append(f"{n_relaxed} of {len(todo)} limits settled by the SOC "
                          f"relaxation{tightened}; the rest solved globally"
                          + (f", {n_extremes} of them to make each reported worst case "
                             "exact" if n_extremes else ""))
    if case.topology.unrated_branches and "thermal" in limits:
        cert.notes.append(f"{len(case.topology.unrated_branches)} unrated branch(es) have no "
                          "thermal limit and were not checked")

    if confirm_witnesses and cert.violations:
        _confirm_witnesses(case, axes, cert.violations,
                           time_limit=time_limit, threads=threads)

    if existence == "contraction" and not no_solution:
        from OpenRC.validation.existence import prove_existence

        cert.existence = prove_existence(region, case, axes, window=window)

    if surface_points and not no_solution:
        from OpenRC.validation.samplers import SurfaceSampler

        points = SurfaceSampler().draw(int(surface_points), region=region, rng=rng)
        created_verifier = point_verifier is None
        verifier = (default_point_verifier(region, case, exact_time_limit=time_limit,
                                           target_model="branch-flow")
                    if created_verifier else point_verifier)
        try:
            cert.surface_checks = verifier.verify_many(axes, points)
        finally:
            if created_verifier:
                verifier.close()
    elif not surface_points and not cert.existence_proven:
        cert.notes.append("existence on the surface was not checked (surface_points=0); "
                          "the certificate covers the limits only")

    cert.runtime = time.perf_counter() - start
    return cert


def _validate_surface_reference(verifier, case) -> None:
    """Surface existence evidence must concern the global model being bounded."""
    from OpenRC.powerflow.base import _model_compatibility
    from OpenRC.powerflow.exact import ExactDistFlowOracle

    reference = getattr(verifier, "reference_oracle", None)
    if reference is None:
        if getattr(verifier, "target_model", None) == "branch-flow":
            return   # explicit contract of a caller-supplied verifier
        raise ValueError("a custom certificate point_verifier must declare "
                         "target_model='branch-flow' or expose a compatible reference_oracle")
    target = ExactDistFlowOracle(case)
    try:
        matches, why = _model_compatibility(reference, target)
    finally:
        target.close()
    if not matches:
        raise ValueError("the certificate point_verifier targets different physics or limits "
                         f"({why}); use PointVerifier(..., target_model='branch-flow')")


def default_point_verifier(region, case, **kwargs):
    """A :class:`~OpenRC.validation.pointcheck.PointVerifier` matched to ``region``.

    The theory stage uses the sub-models of the conservative method recorded in
    the region's metadata (with its loss-shift or thermal-tightening scope); a
    region of unknown origin gets the Method-2 pair with the default path
    scope, whose joint acceptance is recorded but never settles a point.
    """
    from OpenRC.validation.pointcheck import PointVerifier

    meta = getattr(region, "metadata", {}) or {}
    kwargs.setdefault("method", meta.get("conservative_method", "tight-intersect"))
    kwargs.setdefault("shift_scope", meta.get("shift_scope", "path"))
    kwargs.setdefault("tighten_scope", meta.get("tighten_scope", "upstream"))
    return PointVerifier(case, **kwargs)


def _outcome_counts(records) -> str:
    counts = {o: sum(r.outcome is o for r in records) for o in PointOutcome}
    parts = [f"{counts[o]} {o.value}" for o in PointOutcome if counts[o]]
    return f"{len(records)} points: " + ", ".join(parts)


def shrink_to_certified(
    region: Polytope,
    case,
    axes=None,
    *,
    center: Optional[Sequence[float]] = None,
    rel_tol: float = 1e-3,
    max_probes: int = 20,
    **certify_kwargs,
) -> Tuple[Polytope, InnerCertificate, float]:
    """Largest contraction of ``region`` towards ``center`` that is certified inner.

    A constructed region can fail its certificate by a small margin - the
    ``intersect-shift`` construction, for instance, does not compensate every
    loss it drops (see the guide).  Contracting the region homothetically,
    ``center + lam * (region - center)``, keeps its shape and trades a little
    volume for stronger evidence.  ``lam`` is found by bisection; probes use
    global limit bounds, and the returned region receives the full certificate
    (existence proof and the requested surface checks).

    Parameters
    ----------
    center:
        Fixed point of the contraction.  Defaults to the base operating point
        (the origin of region coordinates) when it lies inside the region, and
        to the region's Chebyshev centre otherwise.  It must itself be feasible.
    rel_tol:
        Bisection stops once the bracket on ``lam`` is narrower than this.
    max_probes:
        Hard cap on the number of certificates computed after the first one
        (probes and full certificates of contracted regions together; one full
        certificate of the bisected contraction is always attempted).
    certify_kwargs:
        Passed to :func:`certify_inner_region`.  Probes use
        ``stop_at_limit=True`` and no surface sample; the final region gets the
        full certificate with the requested options.

    Returns
    -------
    ``(region, certificate, lam)``.  ``lam == 1`` means the region was already
    certified and is returned unchanged.

    Raises
    ------
    RuntimeError
        If no contraction can be certified: the limit bounds fail down to
        ``rel_tol`` (the centre itself is not provably feasible), or the full
        certificate's existence/surface checks fail for every contraction tried.
    """
    from OpenRC.geometry.ops import chebyshev_center

    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    if (isinstance(rel_tol, (bool, np.bool_))
            or not isinstance(rel_tol, (int, float, Real))
            or not np.isfinite(float(rel_tol)) or not 0.0 < float(rel_tol) < 1.0):
        raise ValueError("rel_tol must be finite and lie strictly between 0 and 1")
    if (isinstance(max_probes, (bool, np.bool_))
            or not isinstance(max_probes, (int, float, Real))
            or not np.isfinite(float(max_probes))
            or int(float(max_probes)) != float(max_probes) or float(max_probes) <= 0):
        raise ValueError("max_probes must be a positive integer")
    families = certify_kwargs.get("limits", ("thermal", "v_min", "v_max"))
    if (not isinstance(families, (str, bytes, bytearray))
            and not {"thermal", "v_min", "v_max"} <= set(families)):
        raise ValueError("shrink_to_certified needs every limit family; leave 'limits' out, "
                         "or use certify_inner_region for a limits-only certificate")
    surface_points = certify_kwargs.get("surface_points", 32)
    if (not isinstance(surface_points, (bool, np.bool_))
            and surface_points == 0
            and certify_kwargs.get("existence", "contraction") == "surface"):
        raise ValueError(
            "shrink_to_certified needs existence evidence for its final certificate: "
            "surface_points > 0, or existence='contraction'"
        )
    axes = axes if axes is not None else region.axes
    chosen: Optional[np.ndarray] = None
    if center is not None:
        # Validate an explicit contraction centre before running certification.
        raw_center = np.asarray(center)
        if raw_center.dtype == bool:
            raise ValueError("center must be a numeric array, not boolean")
        given = np.asarray(raw_center, dtype=float)
        if given.shape != (region.dim,) or not np.all(np.isfinite(given)):
            raise ValueError(f"center must contain {region.dim} finite coordinates; got {given}")
        if not region.contains(given[None, :])[0]:
            raise ValueError("the contraction center must lie inside the region")
        chosen = given
    full = certify_inner_region(region, case, axes, **certify_kwargs)
    if full.certified:
        return region, full, 1.0

    if chosen is None:
        origin = np.zeros(region.dim)
        A_fac, b_fac = region.facets
        span = max(float(np.ptp(region.vertices, axis=0).max()), 1.0)
        strict_interior = bool(np.all(A_fac @ origin < b_fac - 1e-7 * span))
        chosen = origin if strict_interior else chebyshev_center(A_fac, b_fac)
    if chosen is None:
        raise ValueError("could not determine a contraction centre")
    centre_arr = np.asarray(chosen, dtype=float)
    if centre_arr.shape != (region.dim,) or not np.all(np.isfinite(centre_arr)):
        raise ValueError(
            f"center must contain {region.dim} finite coordinates; got {centre_arr}"
        )
    if not region.contains(centre_arr[None, :])[0]:
        raise ValueError("the contraction center must lie inside the region")

    def contracted(lam: float) -> Polytope:
        shrunk = region.replace(vertices=centre_arr + lam * (region.vertices - centre_arr),
                                certified_contraction=lam)
        return shrunk

    probe_kwargs = dict(certify_kwargs, stop_at_limit=True, surface_points=0,
                        confirm_witnesses=False)
    require_contraction_in_probe = (
        not isinstance(surface_points, (bool, np.bool_)) and int(surface_points) == 0
    )
    lo, hi = 0.0, 1.0            # lo: safe bounds, hi: not certified
    probes = 0
    while hi - lo > rel_tol and probes < max_probes:
        mid = 0.5 * (lo + hi)
        probe_cert = certify_inner_region(contracted(mid), case, axes, **probe_kwargs)
        if probe_cert.certified if require_contraction_in_probe else probe_cert.limits_certified:
            lo = mid
        else:
            hi = mid
        probes += 1

    # The final certificate is the full one, surface sample included.  Should
    # it disagree with the probe (a surface point, or a looser time limit),
    # step down the bracket rather than return an uncertified region.
    lam = lo
    attempted = False
    while lam > 0.0 and (not attempted or probes < max_probes):
        shrunk = contracted(lam)
        cert = certify_inner_region(shrunk, case, axes, **certify_kwargs)
        attempted = True
        probes += 1
        if cert.certified:
            cert.notes.append(f"region contracted by lam = {lam:.4f} towards "
                              f"{np.round(centre_arr, 4).tolist()} to obtain this certificate")
            return shrunk, cert, lam
        lam = lam - max(rel_tol, 0.05 * lam)
    if lo == 0.0:
        cause = (f"the limit bounds failed for every probed contraction, so the contraction "
                 f"centre {np.round(centre_arr, 4).tolist()} is not provably feasible")
    else:
        cause = (f"the limit bounds hold down to lam = {lo:.4f}, but the full certificate "
                 "(existence proof and surface checks) failed for the contractions tried "
                 f"within max_probes={max_probes}")
    raise RuntimeError(f"no contraction of the region could be certified: {cause}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
@dataclass
class _FlowModel:
    """A branch-flow model over the region, with handles on its variables."""

    model: Any
    d: Any
    v2: Any
    l2: Any
    P: Any
    Q: Any
    loading_scale: Dict[int, float]

    def objective(self, kind: str, element: int):
        """The quantity a limit bounds: ``(I / I_max)^2`` or ``v^2``, per unit."""
        if kind == "thermal":
            return self.l2[element] * self.loading_scale[element]
        return self.v2[element]


@dataclass
class _Bounds:
    """Bounds that hold for every exact solution, from the tightened relaxation."""

    P: Dict[int, Tuple[float, float]]
    Q: Dict[int, Tuple[float, float]]
    v2_low: Dict[int, float]
    rounds: int
    seconds: float

    def apply(self, flow: _FlowModel) -> None:
        """Narrow ``flow``'s variable bounds (valid for every exact solution)."""
        for k, (lo, hi) in self.P.items():
            flow.P[k].LB, flow.P[k].UB = lo, hi
        for k, (lo, hi) in self.Q.items():
            flow.Q[k].LB, flow.Q[k].UB = lo, hi
        for bus, lo in self.v2_low.items():
            flow.v2[bus].LB = max(flow.v2[bus].LB, lo)


def _limit_list(case, limits, GRB) -> List[Tuple[str, int, float, int]]:
    """``(kind, element, squared limit, sense)`` for every limit to check."""
    topo = case.topology
    todo = []
    if "thermal" in limits:
        todo += [("thermal", br.index, 1.0, GRB.MAXIMIZE) for br in topo.branches
                 if br.rated]
    for bus in topo.buses:
        if "v_min" in limits:
            todo.append(("v_min", bus, case.vm_min_pu ** 2, GRB.MINIMIZE))
        if "v_max" in limits:
            todo.append(("v_max", bus, case.vm_max_pu ** 2, GRB.MAXIMIZE))
    return todo


def _silent_env(gp):
    """A private environment: no licence banner, memory released on dispose."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def _exact_extremes(gp, GRB, region, case, axes, window, todo, done, bounds, solve_kw):
    """Re-solve globally the relaxation-settled limits that could hold a reported worst case.

    A limit the relaxation settles keeps the relaxation's bound: valid, but an
    over-estimate (Method 1 on ``9bus`` 1,3,5,6: 97.4 % for an exact 89.8 %).
    The certificate reports the worst case of each family (loading, lowest and
    highest voltage), so per family the relaxation-settled limits are solved
    globally, worst bound first, until no remaining relaxation bound is worse
    than the exact worst case found.  Each such limit stays proven either way;
    a global solve that does not settle keeps the relaxation's check.  One
    model serves every re-solve.  Updates ``done`` in place; returns the
    number of limits re-solved and whether a solve found that no solution
    exists.
    """
    families = []
    for kind in ("thermal", "v_min", "v_max"):
        relaxed = sorted((i for i, c in done.items() if c.kind == kind
                          and c.solver_status == "relaxation" and math.isfinite(c.bound)),
                         key=lambda i: done[i].bound, reverse=kind != "v_min")
        if relaxed:
            families.append((kind, relaxed))
    if not families:
        return 0, False
    solved = 0
    env = _silent_env(gp)
    try:
        flow = _branch_flow_model(gp, env, region, case, axes, window)
        try:
            if bounds is not None:
                bounds.apply(flow)
            _configure_global(flow.model, solve_kw["time_limit"], solve_kw["threads"])
            for kind, relaxed in families:
                lowest = kind == "v_min"
                exact = [c.bound for c in done.values() if c.kind == kind
                         and c.solver_status != "relaxation" and c.status != UNDECIDED
                         and math.isfinite(c.bound)]
                best = (min(exact) if lowest else max(exact)) if exact else None
                for i in relaxed:
                    bound = done[i].bound
                    if best is not None and (bound >= best if lowest else bound <= best):
                        break
                    _, element, limit_sq, sense = todo[i]
                    check = _check_limit(flow, GRB, kind, element, limit_sq, sense,
                                         solve_kw["tol"], solve_kw["stop_at_limit"], region,
                                         solve_kw["split_depth"])
                    solved += 1
                    if check.solver_status == "infeasible":
                        return solved, True
                    # A split can prove an injection infeasible without a
                    # finite objective bound (there is no objective value on
                    # the empty-solution piece).  Keep that witness, and any
                    # attained violation, before filtering numerical bounds.
                    if check.status == VIOLATED:
                        done[i] = check
                        continue
                    if check.status == UNDECIDED or not math.isfinite(check.bound):
                        continue
                    done[i] = check
                    if best is None or (check.bound < best if lowest else check.bound > best):
                        best = check.bound
        finally:
            flow.model.dispose()
    finally:
        env.dispose()
    return solved, False


def _relax_stage(gp, GRB, region, case, axes, window, todo, rounds, tol, workers):
    """Screen with the relaxation, tightening it while that still settles limits.

    The first tightening round always runs: without it the relaxation settles
    no current or lower-voltage limit at all, and one global solve of such a
    limit can take a time-out.  Each further round runs while the previous
    one settled at least one more limit, up to ``rounds``.  Rounds are counted
    rather than weighed against the time of a global solve, so every bound is
    reproducible on any machine; the price is an occasional round that a fast
    global solve would have made unnecessary.  Returns the settled checks by
    index into ``todo``, the indices still open, the bounds for the global
    solves, and whether a solve showed that no solution exists.
    """
    relax = _Relaxation(gp, GRB, region, case, axes, window, tol, workers)
    try:
        done = relax.screen(todo, range(len(todo)))
        pending = [i for i in range(len(todo)) if i not in done]
        no_solution = False
        while pending and relax.rounds < rounds:
            if not relax.tighten():               # infeasible: the global solve says so
                break
            settled = relax.screen(todo, pending)
            if not settled:
                break
            done.update(settled)
            pending = [i for i in pending if i not in settled]
        return done, pending, relax.bounds(), no_solution
    finally:
        relax.close()


class _Relaxation:
    """The SOC relaxation over the region, kept open to screen and tighten.

    It holds one replica of the relaxed model per worker; screening and each
    phase of a tightening round fan their solves out over the replicas in
    threads (Gurobi releases the interpreter lock while it solves), and the
    replicas are kept identical in between.

    A tightening round bounds every bus voltage from below, then every branch
    flow from both sides, over the current relaxation, and finally replaces
    the loss cuts ``v_lo(parent) l <= secant(P) + secant(Q)``.  Each
    cut holds for every exact solution (``l v = P^2 + Q^2``, ``v >= v_lo``,
    and a square lies below its secant on ``[a, b]``), so every round's
    relaxation still contains them all, and so do the bounds it returns.
    """

    def __init__(self, gp, GRB, region, case, axes, window, tol, workers: int = 1) -> None:
        self.GRB, self.tol, self.topology = GRB, tol, case.topology
        self.slack_v2 = float(case.slack_vm_pu) ** 2
        self.rounds = 0
        self.tighten_seconds = 0.0
        self._replicas: List[Tuple[Any, _FlowModel]] = []
        self._cuts: List[list] = []
        try:
            for _ in range(max(1, int(workers))):
                env = _silent_env(gp)
                self._replicas.append((env, _branch_flow_model(
                    gp, env=env, region=region, case=case, axes=axes, window=window,
                    relaxed=True)))
                self._cuts.append([])
        except BaseException:
            self.close()
            raise

    @property
    def flow(self) -> _FlowModel:
        return self._replicas[0][1]

    def _map(self, jobs: Sequence, solve) -> list:
        """``solve(flow, job)`` for every job, spread over the replicas."""
        jobs = list(jobs)
        results: list = [None] * len(jobs)
        if len(self._replicas) == 1 or len(jobs) < 2:
            for k, job in enumerate(jobs):
                results[k] = solve(self.flow, job)
        else:
            work: "queue.Queue[int]" = queue.Queue()
            for k in range(len(jobs)):
                work.put(k)
            errors: List[BaseException] = []

            def run(flow: _FlowModel) -> None:
                try:
                    while True:
                        try:
                            k = work.get_nowait()
                        except queue.Empty:
                            return
                        results[k] = solve(flow, jobs[k])
                except BaseException as exc:      # re-raised in the calling thread
                    errors.append(exc)

            threads = [threading.Thread(target=run, args=(flow,), daemon=True)
                       for _, flow in self._replicas]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
        return results

    def screen(self, todo, indices) -> Dict[int, LimitCheck]:
        """The limits among ``indices`` that the relaxation proves."""
        def one(flow: _FlowModel, i: int) -> Optional[LimitCheck]:
            kind, element, limit_sq, sense = todo[i]
            return _screen(flow, self.GRB, kind, element, limit_sq, sense, self.tol,
                           self.rounds)

        indices = list(indices)
        checks = self._map(indices, one)
        return {i: c for i, c in zip(indices, checks) if c is not None}

    def tighten(self) -> bool:
        """One round of bound tightening; ``False`` if the relaxation is infeasible.

        Voltages first, then flows, each **level by level from the slack**: the
        bounds found at one depth are applied before the next depth is solved.
        A bound on an upstream flow caps how far every loss below it can
        inflate, so this order propagates tightening down the tree within one
        round; the solves of one level run in parallel.
        """
        GRB, topo = self.GRB, self.topology
        start = time.perf_counter()
        depth = {br.index: topo.depth(br.child) for br in topo.branches}
        for level in sorted(set(depth.values())):
            branches = [br for br in topo.branches if depth[br.index] == level]
            buses = [br.child for br in branches]
            lows = self._map(buses, lambda flow, bus: _relaxed_bound(
                flow.model, GRB, flow.v2[bus], GRB.MINIMIZE))
            if any(low is _INFEASIBLE for low in lows):
                return False
            self._apply([("v2", bus, GRB.MINIMIZE) for bus in buses], lows)
        for level in sorted(set(depth.values())):
            jobs = [(which, br.index, sense) for br in topo.branches
                    if depth[br.index] == level for which in ("P", "Q")
                    for sense in (GRB.MINIMIZE, GRB.MAXIMIZE)]
            found = self._map(jobs, lambda flow, job: _relaxed_bound(
                flow.model, GRB, getattr(flow, job[0])[job[1]], job[2]))
            if any(value is _INFEASIBLE for value in found):
                return False
            self._apply(jobs, found)
        for r, (_, flow) in enumerate(self._replicas):
            flow.model.remove(self._cuts[r])
            self._cuts[r] = []
            for br in topo.branches:
                if not self._finite(flow, br.index):
                    continue
                P, Q = flow.P[br.index], flow.Q[br.index]
                v_low = (self.slack_v2 if br.parent == topo.slack
                         else flow.v2[br.parent].LB)
                self._cuts[r].append(flow.model.addConstr(
                    v_low * flow.l2[br.index]
                    <= (P.LB + P.UB) * P - P.LB * P.UB + (Q.LB + Q.UB) * Q - Q.LB * Q.UB))
            flow.model.update()
        self.rounds += 1
        self.tighten_seconds += time.perf_counter() - start
        return True

    def _apply(self, jobs, values) -> None:
        """Narrow ``(variable family, index, sense)`` bounds in every replica."""
        GRB = self.GRB
        for _, flow in self._replicas:
            for (which, index, sense), value in zip(jobs, values):
                if value is None:
                    continue
                var = getattr(flow, which)[index]
                if sense == GRB.MINIMIZE:
                    var.LB = max(var.LB, value)
                else:
                    var.UB = min(var.UB, value)
            flow.model.update()

    @staticmethod
    def _finite(flow: _FlowModel, k: int) -> bool:
        P, Q = flow.P[k], flow.Q[k]
        return all(map(math.isfinite, (P.LB, P.UB, Q.LB, Q.UB)))

    def bounds(self) -> Optional["_Bounds"]:
        """The tightened bounds, or ``None`` before the first round."""
        if not self.rounds:
            return None
        flow = self.flow
        finite = [br.index for br in self.topology.branches if self._finite(flow, br.index)]
        return _Bounds(
            P={k: (flow.P[k].LB, flow.P[k].UB) for k in finite},
            Q={k: (flow.Q[k].LB, flow.Q[k].UB) for k in finite},
            v2_low={bus: flow.v2[bus].LB for bus in self.topology.buses},
            rounds=self.rounds, seconds=self.tighten_seconds,
        )

    def close(self) -> None:
        for env, flow in self._replicas:
            flow.model.dispose()
            env.dispose()
        self._replicas = []


#: Returned by :func:`_relaxed_bound` when the relaxation itself is infeasible.
_INFEASIBLE = object()


def _relaxed_bound(model, GRB, expr, sense):
    """A valid bound on ``expr`` over the relaxation, padded for solver accuracy.

    ``None`` when the solve gives no bound, :data:`_INFEASIBLE` when the
    relaxation - and so the exact model - has no solution at all.
    """
    model.reset(0)
    model.setObjective(expr, sense)
    model.optimize()
    if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return _INFEASIBLE
    if model.Status != GRB.OPTIMAL:
        return None
    try:
        value = float(model.ObjBound)       # the dual bound
    except Exception:
        value = float(model.ObjVal)
    pad = 1e-6 * (1.0 + abs(value))
    return value + pad if sense == GRB.MAXIMIZE else value - pad


def _screen(flow: _FlowModel, GRB, kind, element, limit_sq, sense, tol,
            rounds) -> Optional[LimitCheck]:
    """The limit, if the relaxation proves it; ``None`` if it cannot."""
    t0 = time.perf_counter()
    bound_sq = _relaxed_bound(flow.model, GRB, flow.objective(kind, element), sense)
    runtime = time.perf_counter() - t0
    if bound_sq is None or bound_sq is _INFEASIBLE:
        return None
    maximise = sense == GRB.MAXIMIZE
    if not (bound_sq <= limit_sq + tol if maximise else bound_sq >= limit_sq - tol):
        return None
    return LimitCheck(
        kind, element, math.sqrt(limit_sq), bound=math.sqrt(max(bound_sq, 0.0)),
        incumbent=math.nan, status=CERTIFIED, solver_status="relaxation",
        runtime=runtime,
        note=f"settled by the SOC relaxation ({rounds} tightening round(s))")


def _solve_globally(gp, GRB, region, case, axes, window, todo, bounds, *, time_limit,
                    threads, tol, stop_at_limit, split_depth,
                    workers) -> Tuple[Sequence[Optional[LimitCheck]], bool]:
    """Solve each limit in ``todo`` to global optimality, possibly in parallel.

    Each worker thread owns a Gurobi environment and a model (Gurobi releases
    the interpreter lock while it solves).  Results keep the order of ``todo``.
    Once a solve shows that no solution exists inside the window, no new solve
    starts; the polytope is non-empty, so that is the only way a solve can be
    infeasible.
    """
    if not todo:
        return [], False
    workers = min(workers, len(todo))
    if workers > 1 and not threads:
        threads = max(1, (os.cpu_count() or 1) // workers)
    results: List[Optional[LimitCheck]] = [None] * len(todo)
    work: "queue.Queue[int]" = queue.Queue()
    for i in range(len(todo)):
        work.put(i)
    stop = threading.Event()
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            env = _silent_env(gp)
        except BaseException as exc:          # re-raised in the calling thread
            errors.append(exc)
            stop.set()
            return
        try:
            flow = _branch_flow_model(gp, env, region, case, axes, window)
            try:
                if bounds is not None:
                    bounds.apply(flow)
                _configure_global(flow.model, time_limit, threads)
                while not stop.is_set():
                    try:
                        i = work.get_nowait()
                    except queue.Empty:
                        return
                    kind, element, limit_sq, sense = todo[i]
                    results[i] = check = _check_limit(
                        flow, GRB, kind, element, limit_sq, sense, tol, stop_at_limit,
                        region, split_depth)
                    if check.solver_status == "infeasible":
                        stop.set()
            finally:
                flow.model.dispose()
        except BaseException as exc:
            errors.append(exc)
            stop.set()
        finally:
            env.dispose()

    if workers == 1:
        worker()
    else:
        pool = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()
    if errors:
        raise errors[0]
    done = [c for c in results if c is not None]
    no_solution = any(c.solver_status == "infeasible" for c in done)
    if no_solution:
        return [
            c if (c is not None and c.solver_status != "infeasible") else None
            for c in results
        ], True
    return done, False


def _configure_global(model, time_limit, threads) -> None:
    """Spatial branch-and-bound to a proven global optimum."""
    model.Params.NonConvex = 2
    model.Params.MIPGap = 1e-6
    # Retain Gurobi's 1e-6 feasibility tolerance to match the relaxation: the
    # relaxation's tightened bounds are themselves valid only to that
    # tolerance, and at 1e-9 a global solve on a larger network can report no
    # solution inside the window, which ends the certificate early.
    model.Params.TimeLimit = float(time_limit)
    if threads:
        model.Params.Threads = int(threads)


def _check_limit(flow: _FlowModel, GRB, kind, element, limit_sq, sense, tol,
                 stop_at_limit, region, split_depth) -> LimitCheck:
    """One global solve, re-solved on pieces of the region if it stays undecided."""
    expr = flow.objective(kind, element)
    check = _solve_limit(flow.model, GRB, flow.d, kind, element, expr, limit_sq, sense,
                         tol, stop_at_limit)
    if check.status == UNDECIDED and split_depth > 0 and check.solver_status != "infeasible":
        check = _split_limit(flow.model, GRB, flow.d, kind, element, expr, limit_sq, sense,
                             tol, stop_at_limit, region, region.bounding_box(), split_depth,
                             check)
    return check


def _split_limit(model, GRB, d, kind, element, expr, limit_sq, sense, tol,
                 stop_at_limit, region, box, depth, whole: LimitCheck) -> LimitCheck:
    """Settle an undecided limit by solving it on pieces of the region.

    ``max over R = max over the pieces`` for any cover of ``R``, so the pieces'
    bounds combine (max, or min for ``v_min``) into a valid bound for the whole
    region.  Pieces are boxes cut across the widest coordinate; each is added
    to the model as two bound constraints, solved, and removed again.  An
    empty piece (checked by LP) contributes nothing; a non-empty piece with no
    solution inside the window refutes the region there.
    """
    from scipy.optimize import linprog

    A, b = region.facets
    maximise = sense == GRB.MAXIMIZE
    pieces: List[LimitCheck] = []
    n_solved = 0

    def solve(lower, upper, level) -> None:
        nonlocal n_solved
        k = int(np.argmax(upper - lower))
        mid = 0.5 * (lower[k] + upper[k])
        for low_side in (True, False):
            lo, hi = lower.copy(), upper.copy()
            if low_side:
                hi[k] = mid
            else:
                lo[k] = mid
            inside = linprog(np.zeros(len(lo)), A_ub=A, b_ub=b,
                             bounds=list(zip(lo, hi)), method="highs")
            if inside.status == 2:
                continue
            if inside.status != 0 or inside.x is None:
                pieces.append(LimitCheck(
                    kind, element, whole.limit, whole.bound, whole.incumbent,
                    UNDECIDED, "lp_undecided", 0.0,
                    note=f"piece intersection LP undecided (status {inside.status})",
                ))
                continue
            added = [model.addConstr(d[i] >= float(lo[i])) for i in range(len(lo))]
            added += [model.addConstr(d[i] <= float(hi[i])) for i in range(len(lo))]
            try:
                piece = _solve_limit(model, GRB, d, kind, element, expr, limit_sq,
                                     sense, tol, stop_at_limit)
            finally:
                model.remove(added)
                model.update()
            n_solved += 1
            if piece.solver_status == "infeasible":
                # Every point of this piece of the region lacks a solution in
                # the voltage window; confirm_witnesses checks the all-limits
                # exact model at inside.x before labelling it infeasible.
                witness = np.asarray(inside.x, dtype=float)
                piece = LimitCheck(kind, element, whole.limit, math.nan, math.nan, VIOLATED,
                                   "split_infeasible", piece.runtime, witness=witness,
                                   note="no solution inside the voltage window on part of "
                                        "the region", witness_infeasible=False)
            if piece.status == UNDECIDED and level > 1:
                solve(lo, hi, level - 1)
            else:
                pieces.append(piece)

    solve(np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float), depth)
    if not pieces:
        return whole
    runtime = whole.runtime + sum(p.runtime for p in pieces)
    violated = [p for p in pieces if p.status == VIOLATED]
    bounds = [p.bound for p in pieces if math.isfinite(p.bound)]
    finite = len(bounds) == len(pieces)
    bound = (max(bounds) if maximise else min(bounds)) if finite else whole.bound
    incumbents = [p.incumbent for p in pieces if math.isfinite(p.incumbent)]
    incumbent = ((max(incumbents) if maximise else min(incumbents)) if incumbents
                 else whole.incumbent)
    note = f"split into {n_solved} pieces (depth {depth})"
    if violated:
        # An infeasible piece proves that some injections lack a solution,
        # whereas a limit witness may be only an unsafe alternative solution.
        # Preserve that stronger evidence without labelling the *whole* model
        # infeasible (the unsplit region may also contain feasible injections).
        worst = next((p for p in violated if p.solver_status == "split_infeasible"),
                     violated[0])
        return LimitCheck(kind, element, whole.limit, bound, worst.incumbent, VIOLATED,
                          worst.solver_status, runtime, witness=worst.witness,
                          note=f"{note}; {worst.note}".rstrip("; "),
                          witness_infeasible=worst.witness_infeasible)
    if all(p.status == CERTIFIED for p in pieces):
        return LimitCheck(kind, element, whole.limit, bound, incumbent, CERTIFIED,
                          "split", runtime, note=note)
    return LimitCheck(kind, element, whole.limit, bound if finite else whole.bound,
                      incumbent, UNDECIDED, whole.solver_status, runtime,
                      note=f"{note}; still undecided")


def _branch_flow_model(gp, env, region, case, axes, window, *,
                       relaxed: bool = False) -> _FlowModel:
    """Exact branch-flow equations with the injection free over the region.

    Mirrors :class:`~OpenRC.powerflow.exact.ExactDistFlowOracle` - per-unit on
    a 1 MVA three-phase base - except that the injection is an affine function
    of region coordinates ``d`` constrained to the polytope, and the voltage
    bounds are the (wider) window instead of the limits.  ``relaxed`` replaces
    ``l v = P^2 + Q^2`` by ``l v >= P^2 + Q^2``: a convex (rotated-cone)
    relaxation whose feasible set contains every exact solution.
    """
    topo = case.topology
    s_base_va = 1.0e6
    v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
    s_base_phase = s_base_va / 3.0
    z_base = v_base ** 2 / s_base_phase
    i_base = s_base_phase / v_base
    scale = 1.0e6 / s_base_va                      # MW -> pu

    p_base = case.effective_base_p_mw * scale
    q_base = case.branch_flow_q(case.effective_base_q_mvar) * scale
    A, b = region.facets            # one row per distinct facet
    lower, upper = region.bounding_box()

    m = gp.Model(f"{case.name}:inner-certificate", env=env)
    d = m.addVars(region.dim, lb=lower.tolist(), ub=upper.tolist(), name="d")
    for row, rhs in zip(A, b):
        m.addConstr(gp.quicksum(float(row[k]) * d[k] for k in range(region.dim))
                    <= float(rhs))

    p_inj: Dict[int, object] = {bus: p_base[bus] for bus in topo.buses}
    q_inj: Dict[int, object] = {bus: q_base[bus] for bus in topo.buses}
    for k, axis in enumerate(axes):
        target = p_inj if axis.quantity == "p" else q_inj
        target[axis.bus] = target[axis.bus] + scale * d[k]

    v2 = m.addVars(topo.buses, lb=window[0] ** 2, ub=window[1] ** 2, name="v2")
    l2 = m.addVars([br.index for br in topo.branches], lb=0.0, name="l2")
    P = m.addVars([br.index for br in topo.branches], lb=-gp.GRB.INFINITY, name="P")
    Q = m.addVars([br.index for br in topo.branches], lb=-gp.GRB.INFINITY, name="Q")

    slack_v2 = float(case.slack_vm_pu) ** 2

    def v_of(bus):
        return slack_v2 if bus == topo.slack else v2[bus]

    loading_scale = {}
    for br in topo.branches:
        k, child = br.index, br.child
        down = topo.children_of[child]
        r_pu, x_pu = br.r_ohm / z_base, br.x_ohm / z_base
        m.addConstr(P[k] == gp.quicksum(P[dn.index] for dn in down)
                    + r_pu * l2[k] + p_inj[child])
        m.addConstr(Q[k] == gp.quicksum(Q[dn.index] for dn in down)
                    + x_pu * l2[k] + q_inj[child])
        m.addConstr(v_of(br.parent) - v2[child]
                    == 2.0 * (r_pu * P[k] + x_pu * Q[k]) - (r_pu ** 2 + x_pu ** 2) * l2[k])
        if relaxed:
            m.addQConstr(l2[k] * v_of(br.parent) >= P[k] * P[k] + Q[k] * Q[k])
        else:
            m.addQConstr(l2[k] * v_of(br.parent) == P[k] * P[k] + Q[k] * Q[k])
        if br.rated:
            loading_scale[k] = i_base ** 2 / br.l_max    # l2 * scale = (I / I_max)^2
    m.update()
    return _FlowModel(m, d, v2, l2, P, Q, loading_scale)


def _solve_limit(model, GRB, d, kind, element, expr, limit_sq, sense, tol, stop_at_limit):
    """Bound one limit over the region and classify the outcome."""
    maximise = sense == GRB.MAXIMIZE
    model.reset(0)
    model.setObjective(expr, sense)
    if stop_at_limit:
        # Stop once the bound proves the limit, or a solution breaches it.
        model.Params.BestBdStop = limit_sq + tol if maximise else limit_sq - tol
        model.Params.BestObjStop = limit_sq + 2 * tol if maximise else limit_sq - 2 * tol
    t0 = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - t0

    status = model.Status
    names = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible",
             GRB.INF_OR_UNBD: "inf_or_unbd", GRB.TIME_LIMIT: "time_limit",
             GRB.USER_OBJ_LIMIT: "stopped_at_limit", GRB.NUMERIC: "numeric",
             GRB.INTERRUPTED: "interrupted", GRB.SUBOPTIMAL: "suboptimal"}
    solver_status = names.get(status, f"status_{status}")
    def natural(s):
        return math.sqrt(max(s, 0.0))
    limit = natural(limit_sq)
    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return LimitCheck(kind, element, limit, math.nan, math.nan, UNDECIDED,
                          solver_status, runtime)

    # ObjBound is a valid bound whenever the solve ended normally, including a
    # time-out: branch-and-bound never lets it cross the true optimum.
    try:
        bound_sq = float(model.ObjBound)
    except Exception:
        bound_sq = math.inf if maximise else -math.inf
    has_incumbent = model.SolCount > 0
    incumbent_sq = float(model.ObjVal) if has_incumbent else math.nan
    witness = np.array([d[k].X for k in range(len(d))]) if has_incumbent else None
    if status == GRB.NUMERIC:
        bound_sq = math.inf if maximise else -math.inf

    if maximise:
        proven = bound_sq <= limit_sq + tol
        breached = has_incumbent and incumbent_sq > limit_sq + tol
    else:
        proven = bound_sq >= limit_sq - tol
        breached = has_incumbent and incumbent_sq < limit_sq - tol
    outcome = CERTIFIED if proven else VIOLATED if breached else UNDECIDED

    to_natural = natural if math.isfinite(bound_sq) else (lambda s: s)
    return LimitCheck(
        kind, element, limit,
        bound=to_natural(bound_sq),
        incumbent=natural(incumbent_sq) if has_incumbent else math.nan,
        status=outcome, solver_status=solver_status, runtime=runtime,
        witness=witness if outcome == VIOLATED else None,
    )


def _confirm_witnesses(case, axes, violations: Sequence[LimitCheck], *,
                       time_limit: float, threads: int) -> None:
    """Distinguish an unsafe solution from an infeasible injection.

    The all-limits exact model uses the same physics as the global bounds.
    AC is an independent diagnostic, and never supplies the infeasibility
    proof: its model can differ, and a converged unsafe root need not be the
    only root.  Repeated witness coordinates are solved only once.
    """
    from OpenRC.powerflow.base import get_oracle

    with ExitStack() as resources:
        exact = get_oracle("exact", case, modeling="gurobi", time_limit=time_limit,
                           solver_options={"Threads": threads} if threads else {})
        resources.callback(exact.close)
        try:
            acpf = get_oracle("acpf", case)
            resources.callback(acpf.close)
            ac_error = ""
        except Exception as exc:   # no pandapower model, or pandapower missing
            acpf, ac_error = None, str(exc)
        decisions = {}
        for check in violations:
            if check.witness is None:
                check.note = "no witness point to cross-check"
                continue
            notes = [check.note] if check.note else []
            if not check.witness_infeasible:
                key = tuple(float(x) for x in check.witness)
                if key not in decisions:
                    decisions[key] = exact.verdict_at(axes, check.witness)
                answer = decisions[key]
                check.witness_infeasible = answer is Verdict.INFEASIBLE
                notes.append({
                    Verdict.INFEASIBLE: "exact all-limits solve proves this injection infeasible",
                    Verdict.FEASIBLE: "exact all-limits solve finds a safe alternative solution; "
                                      "the unsafe root does not refute the region",
                    Verdict.INCONCLUSIVE: "exact all-limits solve could not decide whether a "
                                          "safe alternative solution exists",
                }[answer])
            if acpf is not None:
                verdict = acpf.verdict_at(axes, check.witness)
                notes.append({
                    Verdict.INFEASIBLE: "AC power flow also finds a limit violation",
                    Verdict.FEASIBLE: "AC power flow finds this injection feasible",
                    Verdict.INCONCLUSIVE: "AC power flow did not converge at the witness",
                }[verdict])
            else:
                notes.append(f"AC cross-check unavailable ({ac_error})")
            check.note = "; ".join(notes)
