"""Abstract base class :class:`FeasibilityOracle`, three-valued :class:`Verdict`,
and oracle compatibility checks used by region builders and validators.

Implementing a new oracle
-------------------------
Subclass :class:`FeasibilityOracle`, implement :meth:`_check`, and register it::

    @ORACLES.register("my-model", doc="...")
    class MyOracle(FeasibilityOracle):
        def _check(self, p_mw, q_mvar) -> bool:
            ...
"""

from __future__ import annotations

import enum
import math
import threading
from abc import ABC, abstractmethod
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from OpenRC.registry import Registry

_OracleT = TypeVar("_OracleT", bound="FeasibilityOracle")


class SolverUnavailableError(RuntimeError):
    """An optional solver is absent; no mathematical verdict was obtained."""


class Verdict(enum.Enum):
    """Three-valued outcome of a feasibility evaluation.

    Local solver non-convergence maps to :attr:`INCONCLUSIVE` because it does
    not establish infeasibility.  Validation can escalate unresolved points
    without counting numerical failures as demonstrated region errors.
    """

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INCONCLUSIVE = "inconclusive"

    def __bool__(self) -> bool:
        """Return ``True`` iff the verdict is :attr:`FEASIBLE`.

        Region construction keeps uncertain points outside the region, so an
        inconclusive verdict evaluates to ``False`` in boolean context.
        """
        return self is Verdict.FEASIBLE

#: Registry of oracle classes; constructed as ``ORACLES.get(name)(case, **kwargs)``.
ORACLES: Registry = Registry("oracle")

_REPORTED_SOLVER_ERRORS: set = set()


def report_solver_error(oracle_name: str, exc: BaseException) -> None:
    """Log an exception mapped to :attr:`Verdict.INCONCLUSIVE` by a solver-backed oracle.

    Logs each distinct ``(oracle_name, exception_type, message)`` tuple at most
    once per process (up to 100 entries) so configuration errors such as invalid
    solver options remain visible.
    """
    key = (oracle_name, type(exc).__name__, str(exc))
    if key in _REPORTED_SOLVER_ERRORS or len(_REPORTED_SOLVER_ERRORS) >= 100:
        return
    _REPORTED_SOLVER_ERRORS.add(key)
    from OpenRC.utils import LOGGER

    LOGGER.warning("%s: the solver raised %s: %s - the query is INCONCLUSIVE "
                   "(each distinct error is logged once)", oracle_name, type(exc).__name__, exc)


@dataclass
class OracleStats:
    """Cumulative query count, feasible count, and solver wall-clock time."""

    n_calls: int = 0
    n_feasible: int = 0
    seconds: float = 0.0

    def record(self, feasible: bool, seconds: float) -> None:
        if not isinstance(feasible, (bool, np.bool_)):
            raise ValueError("feasible must be a boolean")
        if isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, (int, float, np.number)):
            raise ValueError("seconds must be a finite non-negative float")
        sec = float(seconds)
        if not math.isfinite(sec) or sec < 0.0:
            raise ValueError("seconds must be a finite non-negative float")
        self.n_calls += 1
        self.n_feasible += int(bool(feasible))
        self.seconds += sec

    def record_many(self, feasible: ArrayLike, seconds: float) -> None:
        """Record a batch in constant Python work, preserving custom callbacks.

        ``seconds`` is the total batch evaluation time.  A subclass or instance
        that overrides :meth:`record` still receives one event per point, with
        an equal time share; the ordinary counter uses a NumPy reduction.
        Callers hold the same oracle lock as for single-point bookkeeping.
        """
        if isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, (int, float, np.number)):
            raise ValueError("seconds must be a finite non-negative float")
        sec = float(seconds)
        if not math.isfinite(sec) or sec < 0.0:
            raise ValueError("seconds must be a finite non-negative float")
        raw = np.asarray(feasible)
        if raw.ndim != 1:
            raise ValueError("feasible must be a 1-D boolean array")
        if len(raw) == 0:
            return
        if raw.dtype != bool:
            raise ValueError("feasible must be a 1-D boolean array")
        values = raw
        if getattr(self.record, "__func__", None) is not OracleStats.record:
            share = sec / len(values)
            for ok in values:
                self.record(bool(ok), share)
        else:
            self.n_calls += len(values)
            self.n_feasible += int(np.count_nonzero(values))
            self.seconds += sec

    def reset(self) -> None:
        self.n_calls = 0
        self.n_feasible = 0
        self.seconds = 0.0

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_feasible": self.n_feasible,
            "seconds": round(self.seconds, 4),
            "ms_per_call": round(1000.0 * self.seconds / self.n_calls, 3) if self.n_calls else 0.0,
        }


class FeasibilityOracle(ABC):
    """Decides whether an injection vector is feasible for one network.

    Parameters
    ----------
    case:
        The :class:`~OpenRC.cases.base.GridCase` this oracle is bound to.
    name:
        Label used in provenance metadata; defaults to the class name.
    """

    #: Set by subclasses.  ``True`` means "every point this oracle accepts is
    #: also feasible for the exact branch-flow model", i.e. the oracle induces an
    #: *inner* approximation of the true region.  ``False`` means an *outer*
    #: relaxation containing that region.  ``None`` does not establish either
    #: inclusion; component models must also disable ``supports_adjudication``.
    inner_approximation: Optional[bool] = None

    #: Whether this oracle's verdicts can certify the reference problem, in
    #: the direction stated above.  Disable for individual construction models
    #: whose relationship to the reference is unknown.  Their own feasibility
    #: queries and region-building behaviour are unchanged.
    supports_adjudication: bool = True
    #: A reference oracle must opt in before a negative answer is a global
    #: infeasibility certificate.  A local power flow outside a limit is not
    #: such a proof.  An outer relaxation already establishes this direction.
    certifies_infeasibility: bool = False

    #: Whether :meth:`margin_at` returns a signed constraint slack suitable for
    #: ITP root-finding along rays.
    supports_margin: bool = False
    #: Whether ``ray_boundary(..., search="auto")`` uses margin interpolation
    #: instead of pure bisection.
    prefers_margin_search: bool = False

    #: Registry name this instance was built from, set by :func:`get_oracle`.
    #: ``None`` for oracles constructed directly, which cannot be parallelised.
    registry_name: Optional[str] = None
    registry_kwargs: dict = {}

    def __init__(self, case, name: Optional[str] = None) -> None:
        from OpenRC.cases.base import GridCase

        if case is None or isinstance(case, (str, bytes, bytearray, int, float, bool, np.number)):
            raise TypeError(f"case must be a GridCase; got {type(case).__name__}")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("oracle name must be a non-empty string")
        # Snapshot once at oracle construction so individual queries do not
        # re-hash network tables. Duck-typed extension cases keep their own
        # lifetime contract.
        self.case = case.snapshot() if isinstance(case, GridCase) else case
        self.name = name or type(self).__name__
        self.stats = OracleStats()
        # Directly constructed oracles must not share the class-level empty
        # recipe; registered construction replaces this with its own snapshot.
        self.registry_kwargs = {}
        # A single oracle owns one stateful solver model.  Serialising access
        # keeps it correct if a caller does share it between threads; real
        # parallelism comes from one oracle per worker (see OpenRC.parallel).
        self._lock = threading.RLock()

    # ------------------------------------------------------------- public API
    def close(self) -> None:
        """Release optional solver resources; subclasses may implement cleanup.

        Cleanup must be idempotent.  The default owns no external handle.
        Built-in solver-backed oracles can lazily rebuild after being closed.
        """

    def __enter__(self: _OracleT) -> _OracleT:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def is_feasible(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        """Feasibility of a full per-bus injection vector (MW / Mvar, load positive).

        An inconclusive verdict returns ``False`` here, which keeps uncertain
        points outside a region under construction.  Use :meth:`verdict` when
        the distinction matters.
        """
        return self.verdict(p_mw, q_mvar) is Verdict.FEASIBLE

    def verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """Three-valued feasibility, with timing and call counting."""
        import time

        raw_p = np.asarray(p_mw)
        raw_q = np.asarray(q_mvar)
        if raw_p.dtype == bool or raw_q.dtype == bool:
            raise ValueError("power injections must be numeric arrays, not boolean")
        p = np.asarray(raw_p, dtype=float)
        q = np.asarray(raw_q, dtype=float)
        expected = (self.case.n_bus,)
        if p.shape != expected or q.shape != expected:
            raise ValueError(
                f"p_mw and q_mvar must both have shape {expected}; "
                f"got {p.shape} and {q.shape}"
            )
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("power injections must contain only finite values")

        start = time.perf_counter()
        with self._lock:
            try:
                answer = self._verdict(p, q)
            finally:
                elapsed = time.perf_counter() - start
            self.stats.record(answer is Verdict.FEASIBLE, elapsed)
        return answer

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """Default: promote the boolean :meth:`_check` to a two-valued verdict.

        Oracles that can tell "no solution found" from "no solution exists"
        override this - see :class:`~OpenRC.powerflow.acpf.AcPowerFlowOracle`.
        """
        return Verdict.FEASIBLE if self._check(p_mw, q_mvar) else Verdict.INFEASIBLE

    def verdict_margin(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Tuple[Verdict, float]:
        """:meth:`verdict` plus a signed margin, from one evaluation (one call).

        The margin is ``>= 0`` exactly when the verdict is FEASIBLE and measures
        the normalised slack of the closest limit; it is ``nan`` when the
        oracle has none (see :attr:`supports_margin`) or the solve failed.
        """
        import time

        raw_p = np.asarray(p_mw)
        raw_q = np.asarray(q_mvar)
        if raw_p.dtype == bool or raw_q.dtype == bool:
            raise ValueError("power injections must be numeric arrays, not boolean")
        p = np.asarray(raw_p, dtype=float)
        q = np.asarray(raw_q, dtype=float)
        expected = (self.case.n_bus,)
        if p.shape != expected or q.shape != expected:
            raise ValueError(
                f"p_mw and q_mvar must both have shape {expected}; "
                f"got {p.shape} and {q.shape}"
            )
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("power injections must contain only finite values")
        start = time.perf_counter()
        with self._lock:
            try:
                if self.margin_available():
                    answer, margin = self._verdict_margin(p, q)
                else:
                    answer, margin = self._verdict(p, q), math.nan
            finally:
                elapsed = time.perf_counter() - start
            self.stats.record(answer is Verdict.FEASIBLE, elapsed)
        return answer, margin

    def margin_available(self) -> bool:
        """Whether :meth:`margin_at` returns margins that match the verdicts.

        A subclass that overrides ``_verdict`` (e.g. to add a constraint) while
        inheriting ``_verdict_margin`` would compute margins for the parent's
        verdicts.  The margin is enabled only when ``_verdict_margin`` is defined
        at the same or a more derived class level than ``_verdict``.

        Public single-point wrappers and instance-level verdict patches cannot
        inherit that guarantee: they conservatively fall back to bisection.
        Prefer paired ``_verdict`` / ``_verdict_margin`` subclass hooks. An
        extension with another matched path may explicitly override this method.
        """
        if not self.supports_margin:
            return False
        if "_verdict" in self.__dict__:
            return False
        public_hooks = (("feasible_at", FeasibilityOracle.feasible_at),
                        ("is_feasible", FeasibilityOracle.is_feasible),
                        ("verdict", FeasibilityOracle.verdict))
        if any(getattr(getattr(self, name), "__func__", None) is not default
               for name, default in public_hooks):
            return False
        mro = type(self).__mro__

        def owner(name: str) -> type:
            return next(k for k in mro if name in vars(k))

        return issubclass(owner("_verdict_margin"), owner("_verdict"))

    def reset_warm_state(self) -> None:
        """Clear any query-to-query warm-start state (no-op by default).

        Oracles that warm-start from previous solutions (such as
        :class:`~OpenRC.powerflow.acpf.AcPowerFlowOracle` with ``recycle=True``)
        override this so ray searches produce identical trajectories in serial
        and parallel execution.
        """

    def margin_at(self, axes, delta: ArrayLike) -> Tuple[bool, float]:
        """Feasibility and signed margin of a point in region coordinates."""
        if delta is None:
            raise ValueError("delta cannot be None")
        p, q = self.case.injection_vector(axes, delta)
        answer, margin = self.verdict_margin(p, q)
        return answer is Verdict.FEASIBLE, margin

    def _verdict_margin(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Tuple[Verdict, float]:
        """Subclass hook: the verdict and its margin.  Default: no margin."""
        return self._verdict(p_mw, q_mvar), math.nan

    def verdict_at(self, axes, delta: ArrayLike) -> Verdict:
        """Three-valued verdict for a point in region coordinates."""
        if delta is None:
            raise ValueError("delta cannot be None")
        p, q = self.case.injection_vector(axes, delta)
        return self.verdict(p, q)

    def feasible_at(self, axes, delta: ArrayLike) -> bool:
        """Feasibility of a point expressed in *region* coordinates.

        This is the form the geometry code uses: ``delta`` is one value per
        region axis, added on top of the case's base injection.
        """
        if delta is None:
            raise ValueError("delta cannot be None")
        p, q = self.case.injection_vector(axes, delta)
        return self.is_feasible(p, q)

    def feasible_mask(self, axes, points: np.ndarray) -> np.ndarray:
        """Vectorised convenience: boolean verdict for each row of ``points``."""
        raw_input = np.asarray(points)
        if raw_input.dtype == bool:
            raise ValueError("points must be a numeric array, not boolean")
        raw = np.asarray(raw_input, dtype=float)
        if raw.size == 0:
            if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[0] == 0
                                 and raw.shape[1] in (0, len(axes))):
                return np.zeros(0, dtype=bool)
            if raw.ndim != 2:
                raise ValueError(f"points must be 1D or 2D, got shape {raw.shape}")
            raise ValueError(f"points have dimension {raw.shape[1]}, expected {len(axes)}")
        pts = np.atleast_2d(raw)
        if pts.ndim != 2 or pts.shape[1] != len(axes):
            raise ValueError(f"points have dimension {pts.shape[-1]}, expected {len(axes)}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("region coordinates must contain only finite values")
        return np.array([self.feasible_at(axes, row) for row in pts], dtype=bool)

    # --------------------------------------------------------------- subclass
    @abstractmethod
    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        """Subclass hook: the actual feasibility test, without bookkeeping.

        Oracles that can distinguish a numerical failure from a proof of
        infeasibility should also override :meth:`_verdict`.
        """

    # ------------------------------------------------------------------ repr
    def describe(self) -> dict:
        """Provenance block stored alongside every region this oracle produced."""
        return {
            "oracle": self.name,
            "class": type(self).__name__,
            "case": getattr(self.case, "name", "?"),
            "case_configuration_id": getattr(self.case, "configuration_id", None),
            "slack_vm_pu": getattr(self.case, "slack_vm_pu", None),
            "line_charging": getattr(self.case, "line_charging", None),
            "inner_approximation": self.inner_approximation,
            "supports_adjudication": self.supports_adjudication,
            "certifies_infeasibility": self.certifies_infeasibility,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} case={getattr(self.case, 'name', '?')!r}>"


def _close_solver(solver: Any) -> None:
    """Release an owned Pyomo solver without importing optional solver APIs.

    Prefer a solver's public close protocol (Gurobi also manages environments
    there).  Pyomo's MOSEK adapter has no effective context exit: detach its
    native task and environment, then release them in that order.  Callers
    detach the solver itself before invoking this helper.
    """
    if solver is None:
        return
    close = getattr(solver, "close", None)
    if callable(close):
        close()
        return
    native = getattr(solver, "_solver_model", None)
    environment = getattr(solver, "_mosek_env", None)
    if native is not None:
        solver._solver_model = None
    if environment is not None:
        solver._mosek_env = None
    seen = set()
    with ExitStack() as resources:
        for handle in (environment, native):
            if handle is None or id(handle) in seen:
                continue
            seen.add(id(handle))
            dispose = getattr(handle, "dispose", None)
            close_handle = getattr(handle, "close", None)
            exit_context = getattr(handle, "__exit__", None)
            if callable(dispose):
                resources.callback(dispose)
            elif callable(close_handle):
                resources.callback(close_handle)
            elif callable(exit_context):
                resources.callback(exit_context, None, None, None)


def _model_compatibility(source: Any, reference: Any) -> Tuple[bool, str]:
    """Whether two oracles have the same underlying physical reference model.

    A common case hash alone is insufficient: native AC includes line shunts,
    its infeed setpoint and per-bus limits, whereas branch flow uses the case's
    scalar bounds and its own slack voltage.  OPF additionally dispatches
    resources and enforces source limits absent from fixed-injection branch
    flow.  Only sufficient, explicitly checked equivalences are used here.
    Approximation direction is handled separately by ``_reference_evidence``.
    A construction or DC screening oracle identifies its canonical physical
    problem here; this does not equate its approximation with the exact set.
    Its capability metadata still prevents an unsupported verdict being used.
    """
    if source is reference:
        return True, "same oracle"
    from OpenRC.powerflow.acpf import AcPowerFlowOracle, DcPowerFlowOracle
    from OpenRC.powerflow.distflow import DistFlowOracle
    from OpenRC.powerflow.exact import ExactDistFlowOracle
    from OpenRC.powerflow.opf import AcOpfOracle, DcOpfOracle
    from OpenRC.powerflow.socp import SocpBranchFlowOracle

    def family(oracle):
        if isinstance(oracle, (ExactDistFlowOracle, SocpBranchFlowOracle, DistFlowOracle)):
            return "branch-flow"
        if isinstance(oracle, (AcPowerFlowOracle, DcPowerFlowOracle)):
            return "acpf"
        if isinstance(oracle, (AcOpfOracle, DcOpfOracle)):
            return "acopf"
        return None

    left, right = family(source), family(reference)
    if left is None or right is None:
        return False, "no physical-model equivalence is declared for these oracle types"
    case = source.case
    source_id = getattr(case, "configuration_id", None)
    target_id = getattr(reference.case, "configuration_id", None)
    if source_id is None or source_id != target_id:
        return False, "the oracles use different network configurations"
    if left == right == "branch-flow":
        return True, "same branch-flow problem"

    def enforces_voltage(oracle) -> bool:
        return not (isinstance(oracle, (AcPowerFlowOracle, AcOpfOracle))
                    and not oracle.check_voltage)

    def native_limits_match(a, b):
        if enforces_voltage(a) != enforces_voltage(b):
            return False
        if enforces_voltage(a):
            lo_a, hi_a = a._bus_vm_limits(case.net)
            lo_b, hi_b = b._bus_vm_limits(case.net)
            if not (np.array_equal(lo_a, lo_b) and np.array_equal(hi_a, hi_b)):
                return False
        for table in ("line", "trafo", "trafo3w"):
            frame = getattr(case.net, table, None)
            if frame is not None and len(frame):
                active = (frame[frame.in_service.astype(bool)]
                          if "in_service" in frame.columns else frame)
                if len(active) and not np.array_equal(a._branch_loading_limits(active),
                                                      b._branch_loading_limits(active)):
                    return False
        return True

    if left == right:
        if native_limits_match(source, reference):
            return True, f"same native {left} problem"
        return False, "the native oracles apply different operating limits"
    if "acopf" in (left, right):
        return False, ("OPF dispatch and source power limits are absent from the "
                       "fixed-injection branch-flow/AC power-flow problem")

    ac = source if left == "acpf" else reference
    if not enforces_voltage(ac):
        return False, "the AC oracle disables voltage magnitude limits"
    if not case.is_radial:
        return False, "the native AC network has no equivalent radial branch-flow model"
    native_infeeds: set[int] = set()
    for table, slack_column in (("ext_grid", None), ("gen", "slack")):
        frame = getattr(case.net, table, None)
        if frame is None or not len(frame):
            continue
        active = frame[frame.in_service.astype(bool)] if "in_service" in frame.columns else frame
        if slack_column is not None:
            if slack_column not in active.columns:
                continue
            active = active[active[slack_column].astype(bool)]
        native_infeeds.update(int(bus) for bus in active.bus)
    if native_infeeds != {case.slack_bus}:
        return False, "the native AC infeed location differs from the branch-flow root"
    if "in_service" in case.net.bus.columns and not case.net.bus.in_service.astype(bool).all():
        return False, "native out-of-service buses are not states of the branch-flow tree"
    pf_options = case.net.get("user_pf_options", {})
    if isinstance(pf_options, dict) and any(
            pf_options.get(option, False) for option in ("consider_line_temperature", "tdpf")):
        return False, "native AC temperature-dependent line impedances differ from branch flow"
    if np.any(case.net.bus.vn_kv.to_numpy(dtype=float) != case.v_nom_kv):
        return False, "native bus voltage bases differ from the branch-flow voltage base"
    lines = case.net.line
    live = (lines[lines.in_service.astype(bool)] if "in_service" in lines.columns else lines)
    for column in ("c_nf_per_km", "g_us_per_km"):
        if column in live.columns:
            values = live[column].to_numpy(dtype=float)
            length = live.length_km.to_numpy(dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values * length != 0.0):
                return False, ("native AC line shunts differ from the series branch-flow "
                               "model; constant charging is also an approximation")
    from OpenRC.cases.pandapower_adapter import _slack_setpoint

    setpoint = _slack_setpoint(case.net)
    if setpoint is None or setpoint != float(case.slack_vm_pu):
        return False, "the native AC and branch-flow infeed voltage setpoints differ"
    # Both fixed-dispatch AC and branch flow treat the infeed as a setpoint,
    # rather than imposing an additional voltage-envelope check there.
    buses = np.asarray(case.non_slack_buses, dtype=int)
    lower, upper = ac._bus_vm_limits(case.net)
    if (np.any(lower[buses] != case.vm_min_pu)
            or np.any(upper[buses] != case.vm_max_pu)):
        return False, "native per-bus voltage limits differ from the scalar branch-flow limits"
    from OpenRC.powerflow.topology import _open_switch_elements

    conductive = live.loc[~live.index.isin(_open_switch_elements(case.net, "l"))]
    limits = ac._branch_loading_limits(conductive)
    stated = (conductive.max_loading_percent.to_numpy(dtype=float)
              if "max_loading_percent" in conductive.columns else np.full(len(conductive), 100.0))
    expected = np.where(np.isfinite(stated) & (stated >= 0.0),
                        np.minimum(stated, 100.0), 100.0)
    if not np.array_equal(limits, expected):
        return False, "the AC oracle's thermal limits differ from the branch-flow limits"
    # Placeholder ratings are omitted by branch flow.  Their native AC check
    # is equivalent only if the rating exceeds a voltage-implied current bound
    # and is therefore redundant even for an arbitrary allowed phase angle.
    v_nom = float(case.v_nom_kv) * 1e3 / math.sqrt(3.0)
    for branch in case.topology.unrated_branches:
        if math.isinf(branch.i_max_a):
            continue
        impedance = math.hypot(branch.r_ohm, branch.x_ohm)
        parent_max = (case.slack_vm_pu if branch.parent == case.slack_bus
                      else case.vm_max_pu)
        if (impedance == 0.0 or not math.isfinite(branch.i_max_a)
                or (parent_max + case.vm_max_pu) * v_nom / impedance > branch.i_max_a):
            return False, "an omitted branch-flow rating is not redundant in the native AC model"
    return True, "native AC and radial series branch flow have matching data and limits"


def _reference_evidence(oracle: Any, answer: Verdict, *, reference: Any = None,
                        legacy_infeasibility: bool = False) -> Verdict:
    """Filter a model verdict to the direction it proves for the reference.

    This never evaluates a model.  ``legacy_infeasibility`` is the verdict
    trusted from duck-typed reference providers that declare no capability
    metadata; subclasses of ``FeasibilityOracle`` have explicit safe defaults.
    """
    if not getattr(oracle, "supports_adjudication", True):
        return Verdict.INCONCLUSIVE
    if reference is not None and not _model_compatibility(oracle, reference)[0]:
        return Verdict.INCONCLUSIVE
    approximation = getattr(oracle, "inner_approximation", None)
    if answer is Verdict.FEASIBLE and approximation is not False:
        return Verdict.FEASIBLE
    if (answer is Verdict.INFEASIBLE and approximation is not True
            and (approximation is False
                 or getattr(oracle, "certifies_infeasibility", legacy_infeasibility))):
        return Verdict.INFEASIBLE
    return Verdict.INCONCLUSIVE


_ORACLE_ALIASES = {
    "distflow": "lindistflow",
    "method2-1": "tight-voltage",
    "method2-2": "tight-flow",
    "method2-1-script": "legacy-tight-voltage",
    "method2-2-script": "legacy-tight-flow",
}


def get_oracle(name: str, case, **kwargs) -> FeasibilityOracle:
    """Instantiate a registered oracle by name: ``get_oracle("socp", case)``.

    The registry name and keyword arguments are recorded on the instance so that
    :mod:`OpenRC.parallel` can rebuild an equivalent oracle inside a worker
    process without pickling native solver handles or ``pandapower`` networks.
    """
    canonical = _ORACLE_ALIASES.get(name, name)
    factory = ORACLES.get(canonical)
    recipe: Optional[dict]
    try:
        # Deep-copy kwargs so subsequent caller mutations to solver_options or
        # candidate_options do not alter worker configurations.
        recipe = deepcopy(kwargs)
    except Exception:
        # Fall back to serial-only execution when kwargs contain non-copyable handles.
        recipe = None
    oracle = factory(case, **kwargs)
    oracle.registry_name = canonical if recipe is not None else None
    oracle.registry_kwargs = recipe if recipe is not None else {}
    if recipe is None:
        from OpenRC.utils import LOGGER

        LOGGER.warning("oracle %r options cannot be safely copied for workers; "
                       "parallel helpers will use serial evaluation", name)
    return oracle


def list_oracles() -> List[Tuple[str, str]]:
    """``[(name, one-line description), ...]``."""
    return ORACLES.describe()
