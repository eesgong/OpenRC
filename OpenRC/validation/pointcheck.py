"""Deciding one operating point, as far as the available evidence allows.

Pointwise region verification tests whether a given nodal injection admits a
feasible operating point.  Because local AC solvers can fail to converge on
feasible points, global non-convex solves can time out, and convex relaxations
certify only one direction, :class:`PointVerifier` escalates through four stages
and records which one settled the point:

1. **AC** (``acpf`` by default, ``acopf`` selectable).  A converged solution
   within every limit settles the point as *feasible*.
2. **Exact** - a global solve of the exact branch-flow equations with Gurobi
   (:class:`~OpenRC.powerflow.exact.ExactDistFlowOracle`).  It either finds an
   operating point within limits (*feasible*) or proves none exists
   (*infeasible*).
3. **Theory** - the two sub-models of the conservative method that built the
   region, evaluated in closed form (their equations leave no free variable),
   plus the SOC relaxation, solved with MOSEK:

   * ``intersect-shift`` (Method 1): the point must be accepted by both linear
     sub-models under every combination of the P and Q loss shifts - every
     corner of the on-axis shift box, at every corner of the off-axis one
     (the base injection raised).  When every lossy branch feeds a region bus
     that is exactly the construction's argument, so acceptance is
     *theory-feasible*; otherwise a branch loss no shift represents can
     escape both models, and acceptance is recorded without settling the
     point.
   * ``tight-intersect`` (Method 2): with ``tighten_scope="subtree"`` S1 bounds
     every combination of branch losses (see
     :func:`~OpenRC.region.conservative.method2_covers_losses`), so joint
     acceptance is *theory-feasible*.  With the path scopes the two models
     bound the zero-loss and the maximum-loss corner, not every mixed
     combination of branch losses: acceptance is recorded but does not settle
     the point.
   * the SOC relaxation is an outer model; if even it has no solution within
     the limits, the point is proven *infeasible*.
4. **Worst violation.**  If nothing above settled the point, it is recorded as
   *computationally infeasible*: no tool produced a feasible operating point.
   The most severe limit overrun is reported with it - from the AC solution
   when the power flow converged, otherwise from a relaxed operating point
   minimising the common overrun.  That element's value is diagnostic; only
   the solver's dual bound on the common overrun is universal.

Each stage settles a point only in the direction mathematically implied by its
model.
"""

from __future__ import annotations

import enum
import math
import threading
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, List, Optional, Tuple

import numpy as np

from OpenRC.powerflow.base import (SolverUnavailableError, Verdict, _close_solver,
                                  _model_compatibility, _reference_evidence)
from OpenRC.powerflow.modeling import synchronized_pyomo
from OpenRC.utils import LOGGER

#: Maximum loss-shift box corners evaluated by the Method-1 theory stage before
#: skipping enumeration.
_MAX_THEORY_CORNERS = 4096

#: A relaxation overrun this small is within reach of the conic solver's noise
#: at a degenerate optimum: at points the exact model proves feasible, MOSEK
#: can still return a small positive overrun, more so on a power base far from
#: the network's load.  Such an overrun proves infeasibility only together
#: with the solver's infeasibility certificate for the relaxation with every
#: limit enforced.
_CONFIRM_BELOW = 1e-4


class PointOutcome(enum.Enum):
    """How a point was finally classified."""

    FEASIBLE = "feasible"                      # an operating point within limits was found
    THEORY_FEASIBLE = "theory-feasible"        # construction sub-models accept where it is a theorem
    INFEASIBLE = "infeasible"                  # proven: no operating point within limits
    COMPUTATIONALLY_INFEASIBLE = "computationally-infeasible"   # nothing found one

    @property
    def is_feasible(self) -> bool:
        return self in (PointOutcome.FEASIBLE, PointOutcome.THEORY_FEASIBLE)


@dataclass
class WorstViolation:
    """The most severe limit overrun associated with a point."""

    source: str              # "acpf" or an illustrative "socp-relaxation" solution
    quantity: str            # "loading", "v_min", "v_max", or "existence" (no solution)
    element: Optional[int]   # line / branch index, or bus index
    value: float             # loading as a fraction of rating, or voltage in pu
    limit: float
    excess: float            # how far past the limit, in the same units (>= 0)
    branch: str = "line"     # loading only: "line", "trafo" or "trafo3w"

    def describe(self) -> str:
        if self.quantity == "existence":
            band = (f"{self.value:g}-{self.limit:g} pu" if math.isfinite(self.value)
                    else "its voltage band")
            return (f"no operating point exists even in the SOC relaxation with voltages "
                    f"in {band} (from {self.source})")
        what = "loading" if self.quantity == "loading" else "voltage"
        where = (self.branch if self.quantity == "loading" else "bus") + f" {self.element}"
        unit = "" if self.quantity == "loading" else " pu"
        shown = (f"{100 * self.value:.2f} %" if self.quantity == "loading"
                 else f"{self.value:.4f}{unit}")
        return (f"{what} {shown} at {where} vs limit "
                f"{(100 * self.limit if self.quantity == 'loading' else self.limit):.4g}"
                f"{' %' if self.quantity == 'loading' else unit} "
                f"(excess {self.excess:.4g}, from {self.source})")

    def as_dict(self) -> dict:
        return dict(source=self.source, quantity=self.quantity, element=self.element,
                    value=self.value, limit=self.limit, excess=self.excess, branch=self.branch)


@dataclass
class PointVerification:
    """The full record of how one point was decided."""

    point: np.ndarray
    outcome: PointOutcome
    stage: str                                    # the stage that settled it
    trail: List[Tuple[str, str]] = field(default_factory=list)
    worst: Optional[WorstViolation] = None
    note: str = ""
    #: Whether the first stage's verdict is evidence about the network itself
    #: (its ``supports_adjudication``): AC power flow's is, a linear model's not.
    primary_decides: bool = True
    #: Target physical model for the adjudicated verdict.
    target_model: str = "primary"

    @property
    def model_disagreement(self) -> bool:
        """The first stage *decided* infeasible, yet a later stage accepted the point.

        Occurs when AC power flow converges to a limit-violating root while the
        exact model (or theorem-backed sub-model pair) admits a feasible
        operating point.  The
        point counts as feasible; see
        :attr:`~OpenRC.validation.adjudicate.Adjudication.model_disagreement`.
        """
        return (self.primary_decides and self.outcome.is_feasible and bool(self.trail)
                and self.trail[0][1] == Verdict.INFEASIBLE.value)


    @property
    def singular(self) -> bool:
        """A *singular point*: the same condition as :attr:`model_disagreement`.

        AC power flow converged outside a limit, yet the exact model has an
        operating point within every limit.  The point counts as feasible and
        is reported separately (``singular_points``) for inspection.
        """
        return self.model_disagreement

    def as_dict(self) -> dict:
        return {
            "point": self.point.tolist(), "outcome": self.outcome.value,
            "model_disagreement": self.model_disagreement,
            "singular": self.singular,
            "stage": self.stage, "trail": [list(t) for t in self.trail],
            "target_model": self.target_model,
            "worst": None if self.worst is None else self.worst.as_dict(),
            "note": self.note,
        }


class PointVerifier:
    """Escalating feasibility check for single points (see the module docstring).

    Parameters
    ----------
    case:
        The network.
    primary:
        First-stage AC oracle: ``"acpf"`` (default) or ``"acopf"`` (needs an
        OPF-ready network), or an oracle instance.
    method:
        Which conservative method's sub-models form the theory stage:
        ``"tight-intersect"``, ``"intersect-shift"``, or ``None`` to skip the
        sub-models (the SOC relaxation still runs).
    shift_scope:
        For ``intersect-shift`` only: the scope of the loss shifts, as in
        :class:`~OpenRC.region.conservative.ConservativeBuilder`.
    tighten_scope:
        For ``tight-intersect`` only: S1's thermal-tightening scope, as passed
        to the builder through ``oracle_kwargs`` (see
        :class:`~OpenRC.powerflow.distflow.DistFlowOracle`).
    exact_time_limit:
        Seconds for each global exact solve.
    theory_solver:
        Conic solver for the SOC relaxation (MOSEK by default).  The linear
        sub-models are evaluated in closed form and need none.
    target_model:
        ``"primary"`` (default) checks the physical model underlying the
        first-stage oracle (a weak linear/DC model is only a screen for its
        canonical branch-flow/native-AC problem).
        ``"branch-flow"`` checks the exact radial series-impedance model, as
        required by the global region certificate.  Evidence from another
        model settles a point only when its physics and limits are compatible.
    tol:
        The relaxation's minimum overrun must exceed this (relative, squared
        quantities) to count as a proof of infeasibility.  Overruns up to
        ``1e-4`` must in addition be confirmed by the solver's infeasibility
        certificate for the relaxation with every limit enforced: at points
        the exact model proves feasible, MOSEK can return overruns up to
        ``4.7e-5``.
    """

    def __init__(self, case, *, primary="acpf", method: Optional[str] = "tight-intersect",
                 shift_scope: str = "path", tighten_scope: str = "upstream",
                 exact_time_limit: float = 60.0,
                 theory_solver: str = "mosek", tol: float = 1e-6,
                 target_model: str = "primary") -> None:
        from OpenRC.powerflow.distflow import TIGHTEN_SCOPES
        from OpenRC.region.conservative import CONSERVATIVE_METHODS, _METHOD_ALIASES

        if method is not None:
            method = _METHOD_ALIASES.get(method, method)
        if method is not None and method not in CONSERVATIVE_METHODS:
            raise ValueError(f"method must be one of {tuple(CONSERVATIVE_METHODS)} or None")
        if target_model not in ("primary", "branch-flow"):
            raise ValueError("target_model must be 'primary' or 'branch-flow'")
        if shift_scope not in ("path", "feeding_branch"):
            raise ValueError("shift_scope must be 'path' or 'feeding_branch'")
        if tighten_scope not in TIGHTEN_SCOPES:
            raise ValueError(f"tighten_scope must be one of {TIGHTEN_SCOPES}")
        if (isinstance(exact_time_limit, (bool, np.bool_))
                or not isinstance(exact_time_limit, (int, float, Real))
                or not math.isfinite(float(exact_time_limit)) or float(exact_time_limit) <= 0.0):
            raise ValueError("exact_time_limit must be finite and strictly positive")
        if (isinstance(tol, (bool, np.bool_))
                or not isinstance(tol, (int, float, Real))
                or not math.isfinite(float(tol)) or float(tol) < 0.0):
            raise ValueError("tol must be finite and non-negative")
        if not isinstance(theory_solver, str) or not theory_solver.strip():
            raise ValueError("theory_solver must be a non-empty string")
        from OpenRC.cases.base import GridCase, validate_case_configuration
        from OpenRC.powerflow.base import ORACLES

        if isinstance(primary, str):
            ORACLES.get(primary)
        elif not callable(getattr(primary, "verdict_at", None)):
            raise TypeError("primary must be an oracle name or an oracle")
        validate_case_configuration(case, primary if not isinstance(primary, str) else None)
        self.case = case.snapshot() if isinstance(case, GridCase) else case
        self.primary_spec = primary
        self.target_model = target_model
        self.method = method
        self.shift_scope = shift_scope
        self.tighten_scope = tighten_scope
        self.exact_time_limit = float(exact_time_limit)
        self.theory_solver = theory_solver
        self.tol = float(tol)
        self._lock = threading.RLock()
        # Built lazily - most points never need the later stages.
        self._primary: Any = None
        self._exact: Any = None
        self._submodels: Optional[List[Tuple[str, Any]]] = None
        self._relaxation: Optional["_SoftRelaxation"] = None
        self._exact_unavailable: Optional[str] = None
        self._relaxation_unavailable: Optional[str] = None

    def close(self) -> None:
        """Release owned stages once; later queries can lazily rebuild them.

        A caller-supplied primary remains open.  Acquires ``self._lock`` across
        the full point check because the soft-relaxation and diagnostic stages
        also hold mutable solver state.
        """
        with self._lock:
            owned = []
            if isinstance(self.primary_spec, str):
                owned.append(self._primary)
            owned.append(self._exact)
            if self._submodels is not None:
                owned.extend(oracle for _, oracle in self._submodels)
            owned.append(self._relaxation)
            self._primary = self._exact = self._submodels = self._relaxation = None
            self._exact_unavailable = self._relaxation_unavailable = None
            external = None if isinstance(self.primary_spec, str) else self.primary_spec
            seen = set()
            with ExitStack() as resources:
                for stage in owned:
                    if stage is None or stage is external or id(stage) in seen:
                        continue
                    seen.add(id(stage))
                    resources.callback(stage.close)

    def __enter__(self) -> "PointVerifier":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ oracles
    @property
    def _branch_flow_available(self) -> bool:
        """Can the exact and theory stages be built for this network?"""
        return bool(getattr(self.case, "is_radial", True))

    @property
    def primary(self):
        with self._lock:
            if self._primary is None:
                from OpenRC.powerflow.base import get_oracle

                spec = self.primary_spec
                self._primary = get_oracle(spec, self.case) if isinstance(spec, str) else spec
            return self._primary

    def _exact_oracle(self):
        if self._exact is None:
            from OpenRC.powerflow.exact import ExactDistFlowOracle

            self._exact = ExactDistFlowOracle(self.case, time_limit=self.exact_time_limit)
        return self._exact

    @property
    def reference_oracle(self):
        """The model whose feasibility this verifier is trying to establish."""
        return self._exact_oracle() if self.target_model == "branch-flow" else self.primary

    def _submodel_oracles(self):
        if self._submodels is None:
            from OpenRC.powerflow.base import get_oracle
            from OpenRC.region.conservative import CONSERVATIVE_METHODS

            if self.method is None:
                raise ValueError("no conservative method: the theory stage is disabled")
            names = CONSERVATIVE_METHODS[self.method][:2]
            models = []
            with ExitStack() as pending:
                extra = ({"tighten_scope": self.tighten_scope}
                         if self.method == "tight-intersect" else {})
                for name in names:
                    oracle = get_oracle(name, self.case, solver=self.theory_solver, **extra)
                    pending.callback(oracle.close)
                    models.append((name, oracle))
                pending.pop_all()          # ownership transfers to this verifier
            self._submodels = models
        return self._submodels

    # ------------------------------------------------------------- verify
    def verify(self, axes, point, *, primary_verdict: Optional[Verdict] = None
               ) -> PointVerification:
        """Classify one point in region coordinates."""
        with self._lock:
            return self._verify(axes, point, primary_verdict=primary_verdict)

    def _verify(self, axes, point, *, primary_verdict: Optional[Verdict] = None
                ) -> PointVerification:
        """The serialised implementation; subclasses may extend this hook."""
        if isinstance(axes, (str, bytes, bytearray)):
            raise TypeError("axes must be a sequence of Axis objects, not a string")
        raw_pt = np.asarray(point)
        if raw_pt.dtype == bool:
            raise ValueError("point must be a numeric 1-D vector, not boolean")
        point = np.asarray(raw_pt, dtype=float)
        dim = None if axes is None else len(axes)
        if point.ndim != 1 or (dim is not None and point.shape[0] != dim):
            raise ValueError(f"point must be a 1-D vector of length {dim}, got shape {point.shape}")
        if not np.all(np.isfinite(point)):
            raise ValueError("point must contain only finite coordinates")
        if primary_verdict is not None and not isinstance(primary_verdict, Verdict):
            if isinstance(primary_verdict, (bool, np.bool_)):
                raise ValueError("primary_verdict must be a Verdict, not a boolean")
            primary_verdict = Verdict(primary_verdict)
        record = PointVerification(point=point, outcome=PointOutcome.COMPUTATIONALLY_INFEASIBLE,
                                   stage="unresolved", target_model=self.target_model)

        # ---- 1. AC --------------------------------------------------------
        primary = self.primary
        verdict = (primary_verdict if primary_verdict is not None
                   else primary.verdict_at(axes, point))
        record.trail.append((primary.name, verdict.value))
        reference = self.reference_oracle
        primary_matches, mismatch = _model_compatibility(primary, reference)
        record.primary_decides = (bool(getattr(primary, "supports_adjudication", True))
                                  and primary_matches)
        evidence = _reference_evidence(primary, verdict, reference=reference)
        if not primary_matches:
            record.trail.append(("primary-model", f"diagnostic only: {mismatch}"))
        if evidence is Verdict.FEASIBLE:
            record.outcome, record.stage = PointOutcome.FEASIBLE, primary.name
            return record
        if evidence is Verdict.INFEASIBLE:
            record.outcome, record.stage = PointOutcome.INFEASIBLE, primary.name
            record.note = f"{primary.name} supplied an infeasibility certificate"
            return record
        ac_converged = verdict is Verdict.INFEASIBLE

        if not self._branch_flow_available:
            # Stages 2 and 3 are branch-flow models: on a meshed network (or one
            # with elements they cannot represent) they cannot be built.  Record
            # the point and go on rather than abort the whole verification.
            record.trail.append(("exact", "unavailable: no branch-flow model of this network"))
            record.worst = self._ac_worst(axes, point) if ac_converged else None
            record.note = ("the exact and theory stages need a radial network the branch-flow "
                           "models can represent; recorded as computationally infeasible - "
                           "confirm this point manually")
            return record

        branch_matches, mismatch = _model_compatibility(self._exact_oracle(), reference)
        if not branch_matches:
            record.trail.append(("branch-flow stages", f"not comparable: {mismatch}"))
            record.worst = self._ac_worst(axes, point) if ac_converged else None
            record.note = ("the exact, construction and SOC stages use a different physical "
                           "model or limits from the target; they cannot settle this point")
            return record

        # ---- 2. exact, global -------------------------------------------
        exact = Verdict.INCONCLUSIVE
        if self._exact_unavailable is None:
            try:
                exact = self._exact_oracle().verdict_at(axes, point)
            except SolverUnavailableError as exc:
                self._exact_unavailable = str(exc)
        record.trail.append(("exact", exact.value if self._exact_unavailable is None
                             else f"unavailable: {self._exact_unavailable}"))
        if exact is Verdict.FEASIBLE:
            record.outcome, record.stage = PointOutcome.FEASIBLE, "exact"
            record.note = (f"{primary.name} reported {verdict.value}; the global solve found "
                           "an operating point within every limit")
            return record
        if exact is Verdict.INFEASIBLE:
            record.outcome, record.stage = PointOutcome.INFEASIBLE, "exact"
            record.worst = self._worst_violation(axes, point, ac_converged)
            record.note = "the global solve proved that no operating point satisfies the limits"
            return record

        # ---- 3. theory: the method's sub-models, then the relaxation -----
        if self.method is not None:
            from OpenRC.region.conservative import method1_covers_losses, method2_covers_losses

            accepted = self._submodels_accept(axes, point, record)
            if (accepted and self.method == "intersect-shift"
                    and method1_covers_losses(self.case, axes)):
                record.outcome, record.stage = PointOutcome.THEORY_FEASIBLE, "theory"
                record.note = ("both Method-1 sub-models accept the point under every "
                               "combination of the P and Q loss shifts, and every lossy "
                               "branch feeds a region bus; feasible by the "
                               "construction's argument")
                return record
            if (accepted and self.method == "tight-intersect"
                    and method2_covers_losses(self.case, self._submodel_oracles()[0][1])):
                record.outcome, record.stage = PointOutcome.THEORY_FEASIBLE, "theory"
                record.note = ("both Method-2 sub-models accept the point, and S1's subtree "
                               "tightening, checked at v_min, bounds every combination of "
                               "branch losses; feasible by the construction's argument")
                return record
            if accepted:
                record.note = ("both construction sub-models accept the point, but a "
                               "combination of branch losses can escape them here, so "
                               "this does not settle the point; ")
        relaxed_excess, relaxed_worst = self._relaxation_min_violation(axes, point)
        record.trail.append(("socp-relaxation",
                             "unavailable" if relaxed_excess is None
                             else f"min worst excess {relaxed_excess:.3g}"))
        if (relaxed_excess is not None and self.tol < relaxed_excess <= _CONFIRM_BELOW
                and not self._relaxation_certifies(axes, point)):
            record.trail.append(("socp-relaxation", "excess within solver noise, "
                                 "not certified infeasible"))
            relaxed_excess = None
        if relaxed_excess is not None and relaxed_excess > self.tol:
            record.outcome, record.stage = PointOutcome.INFEASIBLE, "theory"
            record.worst = (self._ac_worst(axes, point) if ac_converged else None) or relaxed_worst
            record.note += ("even the SOC relaxation (an outer model) has no solution within "
                            "the limits, so the point is infeasible")
            return record

        # ---- 4. nothing settled it: record the worst overrun --------------
        record.worst = (self._ac_worst(axes, point) if ac_converged else None) or relaxed_worst
        record.note += ("no stage produced a feasible operating point or a proof of "
                        "infeasibility; recorded as computationally infeasible - confirm "
                        "this point manually")
        if record.worst is not None and record.worst.excess == 0.0:
            record.note += (" (no overrun is proven: the reported worst case is the "
                            "limit closest to binding)")
        return record

    def verify_many(self, axes, points, *, primary_verdicts=None) -> List[PointVerification]:
        """Check a finite ``(n_points, n_axes)`` batch, including empty input."""
        if isinstance(axes, (str, bytes, bytearray)):
            raise TypeError("axes must be a sequence of Axis objects, not a string")
        raw_in = np.asarray(points)
        if raw_in.dtype == bool:
            raise ValueError("points must be a numeric array, not boolean")
        raw = np.asarray(raw_in, dtype=float)
        dim = None if axes is None else len(axes)
        if raw.size == 0 and (
                raw.ndim == 1 or (raw.ndim == 2 and raw.shape[0] == 0
                                  and (dim is None or raw.shape[1] in (0, dim)))):
            raw = np.empty((0, 0 if dim is None else dim))
        points = np.atleast_2d(raw)
        if points.ndim != 2 or (dim is not None and points.shape[1] != dim):
            raise ValueError(f"points must have shape (n_points, {dim})")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite coordinates")
        if primary_verdicts is not None and isinstance(primary_verdicts, (str, bytes, bytearray)):
            raise TypeError("primary_verdicts must be a sequence of Verdicts, not a string")
        verdicts = (list(primary_verdicts) if primary_verdicts is not None
                    else [None] * len(points))
        if len(verdicts) != len(points):
            raise ValueError(
                f"primary_verdicts has {len(verdicts)} entries for {len(points)} points"
            )
        return [self.verify(axes, p, primary_verdict=v) for p, v in zip(points, verdicts)]

    # ---------------------------------------------------------- internals
    def _submodels_accept(self, axes, point, record) -> bool:
        """Whether both sub-models accept the point (Method 1: at every shift corner)."""
        if self.method != "intersect-shift":
            accepted = True
            for name, oracle in self._submodel_oracles():
                answer = oracle.verdict_at(axes, point)
                record.trail.append((name, answer.value))
                accepted &= answer is Verdict.FEASIBLE
            return accepted

        import itertools

        from OpenRC.region.conservative import ConservativeBuilder, OffsetOracle

        builder = ConservativeBuilder(shift_scope=self.shift_scope)
        shifts = builder._axis_shifts(self.case, axes)
        on_axis = [(k, shifts[k]) for k in range(len(axes)) if shifts[k] > 0.0]
        off_axis = builder._off_axis_shifts(self.case, axes)
        n_corners = 2 ** (len(on_axis) + len(off_axis))
        if n_corners > _MAX_THEORY_CORNERS:
            record.trail.append(("theory", f"skipped: {n_corners} shift corners"))
            return False
        for name, model in self._submodel_oracles():
            verdict = "feasible"
            for off_corner in itertools.product((False, True), repeat=len(off_axis)):
                dp, dq = np.zeros(self.case.n_bus), np.zeros(self.case.n_bus)
                for used, (bus, quantity, amount) in zip(off_corner, off_axis):
                    if used:
                        (dp if quantity == "p" else dq)[bus] += amount
                oracle = OffsetOracle(model, dp_mw=dp, dq_mvar=dq) if any(off_corner) else model
                for on_corner in itertools.product((False, True), repeat=len(on_axis)):
                    probe = point.copy()
                    for used, (k, amount) in zip(on_corner, on_axis):
                        if used:
                            probe[k] += amount
                    if oracle.verdict_at(axes, probe) is not Verdict.FEASIBLE:
                        verdict = "infeasible at a shift corner"
                        break
                if verdict != "feasible":
                    break
            record.trail.append((f"{name}[{n_corners} shift corners]", verdict))
            if verdict != "feasible":
                return False
        return True

    def _worst_violation(self, axes, point, ac_converged) -> Optional[WorstViolation]:
        worst = self._ac_worst(axes, point) if ac_converged else None
        if worst is None:
            _, worst = self._relaxation_min_violation(axes, point)
        return worst

    def _ac_worst(self, axes, point) -> Optional[WorstViolation]:
        """Largest overrun in the converged AC solution at ``point``.

        Judged against the limits the AC oracle itself applies: lines and
        transformers each at their own loading limit, buses at their own
        voltage limits.
        """
        oracle = self.primary
        if getattr(oracle, "name", "") != "acpf":
            return None
        # Hold the primary oracle's lock across both runpp and result readout
        # in case the oracle instance is shared across threads; a
        # different query must not replace the diagnostic's solved network.
        with getattr(oracle, "_lock", None) or nullcontext():
            return self._ac_worst_locked(oracle, axes, point)

    def _ac_worst_locked(self, oracle, axes, point) -> Optional[WorstViolation]:
        """Inspect one solved state while its oracle's reentrant lock is held."""
        if oracle.verdict_at(axes, point) is Verdict.INCONCLUSIVE:
            return None
        net = oracle._net
        candidates = []
        own_loading = getattr(oracle, "_branch_loading_limits", None)
        for table, result_table in (("line", "res_line"), ("trafo", "res_trafo"),
                                    ("trafo3w", "res_trafo3w")):
            elements = getattr(net, table, None)
            results = getattr(net, result_table, None)
            if elements is None or results is None or len(elements) == 0:
                continue
            loading = results.loading_percent.to_numpy(dtype=float) / 100.0
            if not np.isfinite(loading).any():
                continue
            limits = (np.asarray(own_loading(elements), dtype=float) / 100.0
                      if callable(own_loading)
                      else np.full(len(loading), oracle.max_loading_percent / 100.0))
            excess = np.where(np.isfinite(loading), loading - limits, -np.inf)
            i = int(np.argmax(excess))
            candidates.append(WorstViolation("acpf", "loading", int(elements.index[i]),
                                             float(loading[i]), float(limits[i]),
                                             max(0.0, float(excess[i])), branch=table))
        vm = net.res_bus.vm_pu.to_numpy(dtype=float)
        if getattr(oracle, "check_voltage", True) and vm.size and np.isfinite(vm).any():
            own_voltage = getattr(oracle, "_bus_vm_limits", None)
            if callable(own_voltage):
                lower, upper = (np.asarray(v, dtype=float) for v in own_voltage(net))
            else:
                lower = np.full(len(vm), oracle.vm_min_pu)
                upper = np.full(len(vm), oracle.vm_max_pu)
            # As the AC oracle judges them: infeed buses hold their setpoint.
            try:
                infeeds = list(self.case.infeed_buses)
            except Exception:
                infeeds = []
            judged = np.isfinite(vm) & ~np.isin(net.bus.index.to_numpy(), infeeds)
            low_gap = np.where(judged, lower - vm, -np.inf)
            high_gap = np.where(judged, vm - upper, -np.inf)
            lo, hi = int(np.argmax(low_gap)), int(np.argmax(high_gap))
            candidates.append(WorstViolation("acpf", "v_min", int(net.bus.index[lo]),
                                             float(vm[lo]), float(lower[lo]),
                                             max(0.0, float(low_gap[lo]))))
            candidates.append(WorstViolation("acpf", "v_max", int(net.bus.index[hi]),
                                             float(vm[hi]), float(upper[hi]),
                                             max(0.0, float(high_gap[hi]))))
        return max(candidates, key=_severity) if candidates else None

    def _relaxation_certifies(self, axes, point) -> bool:
        """Does the solver certify the relaxation infeasible within every limit?"""
        try:
            assert self._relaxation is not None   # solved just before
            p, q = self.case.injection_vector(axes, point)
            return self._relaxation.certifies_infeasible(p, q)
        except Exception as exc:   # solver missing or crashed: no certificate
            LOGGER.warning("SOC relaxation certificate unavailable at %s: %s",
                           np.round(point, 4).tolist(), exc)
            return False

    def _relaxation_min_violation(self, axes, point):
        """Smallest worst-case overrun the SOC relaxation admits at ``point``.

        Minimises ``t`` subject to the relaxed branch-flow model with every
        limit loosened by the factor ``(1 + t)`` (squared quantities).  The
        relaxation contains every exact operating point, so a positive proven
        lower bound on ``t`` proves that each breaches some limit by at least
        that much.  A primal value is an upper bound and cannot prove this.
        Returns ``(lower_bound, WorstViolation)``; ``None`` in the first entry
        means no bound was available.  The diagnostic describes the relaxed
        primal solution, not a universal bound on that particular element.
        """
        if self._relaxation_unavailable is not None:
            return None, None
        try:
            if self._relaxation is None:
                self._relaxation = _SoftRelaxation(self.case, self.theory_solver)
            p, q = self.case.injection_vector(axes, point)
            return self._relaxation.solve(p, q)
        except SolverUnavailableError as exc:
            self._relaxation_unavailable = str(exc)
            LOGGER.warning("SOC relaxation solver unavailable: %s", exc)
            return None, None
        except Exception as exc:   # solver error; treat stage as inconclusive
            LOGGER.warning("SOC relaxation stage unavailable at %s: %s",
                           np.round(point, 4).tolist(), exc)
            return None, None


class _SoftRelaxation:
    """SOC relaxation of branch flow with a common relative slack on all limits.

    Per unit on a three-phase power base matched to the network: a decade below
    its total base load.  On a fixed 1 MVA base a feeder of tens of MVA
    carries squared currents of several thousand per unit and MOSEK can stall
    without an answer; on 100 MVA an LV village grid loses digits.
    ``t`` is a relative overrun, so the base does not change it.  Written with
    ``pyomo.kernel`` because the rotated cone must be declared as a cone.
    """

    @synchronized_pyomo
    def __init__(self, case, solver_name: str) -> None:
        import pyomo.kernel as pmo
        from pyomo.opt import SolverFactory

        self.case = case
        topo = case.topology
        v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        load = float(np.abs(case.effective_base_p_mw).sum()
                     + np.abs(case.effective_base_q_mvar).sum())
        s_base_mva = (min(max(10.0 ** math.floor(math.log10(load)), 1e-3), 1e4)
                      if load > 0.0 else 1.0)
        self.scale = 1.0 / s_base_mva                  # MW -> per unit
        s_base_phase = s_base_mva * 1.0e6 / 3.0
        z_base = v_base ** 2 / s_base_phase
        i_base = s_base_phase / v_base
        self.rated = {br.index: br.l_max / i_base ** 2 for br in topo.branches if br.rated}

        # The voltage band keeps the model bounded as ``t`` grows.  It must
        # contain the case's own limits, or the relaxation is no longer outer
        # and its "no solution" would not prove infeasibility.
        self.band = (min(0.5, case.vm_min_pu), max(1.5, case.vm_max_pu))

        m = pmo.block()
        m.t = pmo.variable(lb=0.0)
        m.p = pmo.parameter_dict()
        m.q = pmo.parameter_dict()
        m.v2 = pmo.variable_dict()
        for bus in topo.buses:
            m.p[bus] = pmo.parameter(value=0.0)
            m.q[bus] = pmo.parameter(value=0.0)
            m.v2[bus] = pmo.variable(lb=self.band[0] ** 2, ub=self.band[1] ** 2)
        m.P, m.Q, m.l2, m.r1 = (pmo.variable_dict() for _ in range(4))
        m.cons = pmo.constraint_list()
        m.cones = pmo.constraint_list()
        vmin2, vmax2 = case.vm_min_pu ** 2, case.vm_max_pu ** 2
        for bus in topo.buses:
            m.cons.append(pmo.constraint(m.v2[bus] >= vmin2 - vmin2 * m.t))
            m.cons.append(pmo.constraint(m.v2[bus] <= vmax2 + vmax2 * m.t))

        slack_v2 = float(case.slack_vm_pu) ** 2

        def v_of(bus):
            return slack_v2 if bus == topo.slack else m.v2[bus]

        for br in topo.branches:            # all variables first: constraints
            k = br.index                     # reference downstream branches
            m.P[k], m.Q[k] = pmo.variable(), pmo.variable()
            m.l2[k], m.r1[k] = pmo.variable(lb=0.0), pmo.variable(lb=0.0)
        for br in topo.branches:
            k, child = br.index, br.child
            r_pu, x_pu = br.r_ohm / z_base, br.x_ohm / z_base
            down = topo.children_of[child]
            m.cons.append(pmo.constraint(
                m.P[k] == sum(m.P[d.index] for d in down) + r_pu * m.l2[k] + m.p[child]))
            m.cons.append(pmo.constraint(
                m.Q[k] == sum(m.Q[d.index] for d in down) + x_pu * m.l2[k] + m.q[child]))
            m.cons.append(pmo.constraint(
                v_of(br.parent) - m.v2[child]
                == 2.0 * (r_pu * m.P[k] + x_pu * m.Q[k]) - (r_pu ** 2 + x_pu ** 2) * m.l2[k]))
            m.cons.append(pmo.constraint(2.0 * m.r1[k] == v_of(br.parent)))
            m.cones.append(pmo.conic.rotated_quadratic(r1=m.r1[k], r2=m.l2[k],
                                                       x=[m.P[k], m.Q[k]]))
            if k in self.rated:
                lim = self.rated[k]
                m.cons.append(pmo.constraint(m.l2[k] <= lim + lim * m.t))
        # Minimise only the common violation.  Adding even a tiny loss penalty
        # can trade a positive ``t`` against lower losses and falsely turn a
        # feasible point into a proof of infeasibility.  A secondary objective
        # is unnecessary because only the optimum value of ``t`` is evidence.
        m.obj = pmo.objective(m.t)
        solver = SolverFactory(solver_name)
        if not solver.available(exception_flag=False):
            raise SolverUnavailableError(f"solver {solver_name!r} is not available")
        self.model, self.solver = m, solver
        # A plugin's field called "lower_bound" need not contain a dual
        # bound: Pyomo's GurobiDirect fills it with ObjVal on continuous
        # models.  MOSEK's plugin uses getdualobj and Gurobi's file interface
        # uses ObjBound; direct Gurobi needs its native ObjBound explicitly.
        plugin = type(solver).__module__
        if plugin in ("pyomo.solvers.plugins.solvers.gurobi_direct",
                      "pyomo.solvers.plugins.solvers.gurobi_persistent"):
            self._dual_bound_kind = "native-gurobi"
        elif (plugin.startswith("pyomo.solvers.plugins.solvers.mosek_")
              or plugin == "pyomo.solvers.plugins.solvers.GUROBI"):
            self._dual_bound_kind = "reported"
        else:
            self._dual_bound_kind = "unavailable"

    def close(self) -> None:
        """Release the native task behind Pyomo's direct solver, idempotently.

        Pyomo's generic solver context is a no-op.  The owned native model is
        therefore detached explicitly and closed using its supported resource
        protocol (Gurobi dispose, MOSEK Task context exit).
        """
        solver, self.solver = getattr(self, "solver", None), None
        self.model = None
        _close_solver(solver)

    def _set_injections(self, p_mw, q_mvar) -> None:
        q_mvar = self.case.branch_flow_q(q_mvar)
        m = self.model
        for bus in self.case.topology.buses:
            m.p[bus].value = float(p_mw[bus]) * self.scale
            m.q[bus].value = float(q_mvar[bus]) * self.scale

    @synchronized_pyomo
    def certifies_infeasible(self, p_mw, q_mvar) -> bool:
        """``True`` when the solver proves no solution with ``t = 0`` exists."""
        m = self.model
        self._set_injections(p_mw, q_mvar)
        m.t.fix(0.0)
        try:
            results = self.solver.solve(m, load_solutions=False)
        finally:
            m.t.free()
        return str(results.solver.termination_condition) == "infeasible"

    @synchronized_pyomo
    def solve(self, p_mw, q_mvar):
        m, topo, case = self.model, self.case.topology, self.case
        self._set_injections(p_mw, q_mvar)
        results = self.solver.solve(m, load_solutions=False)
        tc = str(results.solver.termination_condition)
        if tc == "infeasible":
            # With every limit softened, infeasibility means the relaxation has
            # no operating point at all inside its voltage band.
            return math.inf, WorstViolation("socp-relaxation", "existence", None,
                                            self.band[0], self.band[1], math.inf)
        if tc != "optimal":
            return None, None
        m.load_solution(results.solution(0))
        try:
            if self._dual_bound_kind == "native-gurobi":
                lower = float(self.solver._solver_model.ObjBound)
            elif self._dual_bound_kind == "reported":
                lower = float(results.problem.lower_bound)
            else:
                lower = math.nan
        except (AttributeError, TypeError, ValueError):
            lower = math.nan
        # For minimisation the solver's dual/global lower bound is the
        # certificate; m.t.value is only an attained upper bound.  Some solver
        # interfaces omit the lower bound even with an optimal termination.
        proven = max(0.0, lower) if math.isfinite(lower) else None
        candidates = []
        for k, lim in self.rated.items():
            value = math.sqrt(max(m.l2[k].value, 0.0) / lim)
            candidates.append(WorstViolation("socp-relaxation", "loading", k, value, 1.0,
                                             max(0.0, value - 1.0)))
        for bus in topo.buses:
            v = math.sqrt(max(m.v2[bus].value, 0.0))
            candidates.append(WorstViolation("socp-relaxation", "v_min", bus, v,
                                             case.vm_min_pu, max(0.0, case.vm_min_pu - v)))
            candidates.append(WorstViolation("socp-relaxation", "v_max", bus, v,
                                             case.vm_max_pu, max(0.0, v - case.vm_max_pu)))
        return proven, max(candidates, key=_severity)


def _severity(w: WorstViolation):
    """Order overruns by excess, and limits that hold by how close they are."""
    if w.quantity == "v_min":
        utilisation = w.limit / w.value if w.value > 0 else math.inf
    else:
        utilisation = w.value / w.limit if w.limit > 0 else math.inf
    return (w.excess, utilisation)
