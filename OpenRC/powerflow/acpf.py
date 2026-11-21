"""Full AC and linear DC power-flow feasibility oracles via ``pandapower``.

The validation routines in :mod:`OpenRC.validation` use :class:`AcPowerFlowOracle`
as the default reference model to evaluate false-positive error rates and
volume coverage of derived regions.

Injection bookkeeping
---------------------
Because a bus may host multiple ``load``, ``sgen``, or ``storage`` rows, each
oracle attaches one dedicated control load per non-slack bus at initialisation
and sets it to ``target - base`` so net nodal injections match the requested
target vector.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import contextlib
import io
import math

import numpy as np

from OpenRC.powerflow.base import ORACLES, FeasibilityOracle, Verdict
from OpenRC.powerflow.modeling import SOLVER_OUTPUT_LOCK

# pandapower's Iwamoto implementation prints every multiplier directly to
# stdout. Redirecting stdout is process-global, so protect only that rare retry
# with the same lock as Pyomo; separate locks would still race each other's
# redirection. Normal NR/BFSW attempts remain fully parallel.
_IWAMOTO_OUTPUT_LOCK = SOLVER_OUTPUT_LOCK


def _native_vm_envelope(case) -> Tuple[float, float]:
    """Return the strictest voltage envelope stated on ``case.net.bus``.

    Infeed buses are excluded (their voltage is held at a setpoint), and
    unspecified sides default to ``case.vm_min_pu`` / ``case.vm_max_pu``.  If
    ``case.vm_min_pu`` / ``case.vm_max_pu`` match this envelope, per-bus limits
    from ``net.bus`` are used; if overridden, the scalar limits apply to all
    non-infeed buses.  When bus intervals have no common intersection, falls
    back to ``net["_openrc_case_vm_limits"]``.
    """
    from OpenRC.cases.pandapower_adapter import _voltage_envelope

    net = case.net
    fallback = (case.vm_min_pu, case.vm_max_pu)
    if getattr(net, "bus", None) is None:
        return fallback
    low, high, common = _voltage_envelope(net, case.vm_min_pu, case.vm_max_pu)
    if common:
        return low, high
    return net.get("_openrc_case_vm_limits", fallback) if hasattr(net, "get") else fallback


class _PandapowerOracle(FeasibilityOracle):
    """Base class for ``pandapower`` fixed-dispatch AC and DC power-flow oracles."""

    def __init__(self, case, *, name: str,
                 check_voltage: bool,
                 max_loading_percent: float = 100.0,
                 vm_min_pu: Optional[float] = None,
                 vm_max_pu: Optional[float] = None,
                 numba: Optional[bool] = None) -> None:
        super().__init__(case, name=name)
        case = self.case
        if not isinstance(check_voltage, (bool, np.bool_)):
            raise ValueError("check_voltage must be a boolean")
        self.check_voltage = bool(check_voltage)
        if numba is not None and not isinstance(numba, (bool, np.bool_)):
            raise ValueError("numba must be True, False or None")
        #: pandapower's numba-compiled power flow: 1.2-2.1x per AC query after
        #: a one-off compile of about 0.6 s per process, voltages equal to
        #: 6.5e-13 p.u.  ``None`` (default) uses it when pandapower finds numba.
        self._numba_option = None if numba is None else bool(numba)
        if (
            isinstance(max_loading_percent, (bool, np.bool_))
            or not isinstance(max_loading_percent, (int, float, np.number))
        ):
            raise ValueError("max_loading_percent must be strictly positive and not NaN")
        if (
            (vm_min_pu is not None and (
                isinstance(vm_min_pu, (bool, np.bool_))
                or not isinstance(vm_min_pu, (int, float, np.number))
            ))
            or (vm_max_pu is not None and (
                isinstance(vm_max_pu, (bool, np.bool_))
                or not isinstance(vm_max_pu, (int, float, np.number))
            ))
        ):
            raise ValueError("voltage limits must be finite and satisfy 0 < min < max")
        self.max_loading_percent = float(max_loading_percent)
        self.vm_min_pu = case.vm_min_pu if vm_min_pu is None else float(vm_min_pu)
        self.vm_max_pu = case.vm_max_pu if vm_max_pu is None else float(vm_max_pu)
        if math.isnan(self.max_loading_percent) or self.max_loading_percent <= 0.0:
            raise ValueError("max_loading_percent must be strictly positive and not NaN")
        if not (math.isfinite(self.vm_min_pu) and math.isfinite(self.vm_max_pu)
                and 0.0 < self.vm_min_pu < self.vm_max_pu):
            raise ValueError("voltage limits must be finite and satisfy 0 < min < max")
        base_vm_min, base_vm_max = _native_vm_envelope(case)
        self._vm_min_overridden = vm_min_pu is not None or float(case.vm_min_pu) != float(base_vm_min)
        self._vm_max_overridden = vm_max_pu is not None or float(case.vm_max_pu) != float(base_vm_max)
        self._net: Any = None          # the pandapower net, prepared lazily
        self._control_load: dict[int, int] = {}
        self._control_buses = np.empty(0, dtype=np.intp)
        self._control_rows = np.empty(0, dtype=np.intp)
        self._native_base_p = np.zeros(case.n_bus)
        self._native_base_q = np.zeros(case.n_bus)

    def _bus_vm_limits(self, net) -> Tuple[np.ndarray, np.ndarray]:
        """Per-bus ``(lower, upper)`` voltage magnitude limits in p.u.

        The bus table's own limits apply while the case's scalar envelope
        equals the one the table states (:func:`_native_vm_envelope`); a case
        whose envelope differs - or an explicit ``vm_min_pu``/``vm_max_pu`` -
        sets every non-infeed bus, as it does for the branch-flow models.
        """
        n = len(net.bus)
        lower = np.full(n, self.vm_min_pu, dtype=float)
        upper = np.full(n, self.vm_max_pu, dtype=float)
        infeed_mask = np.zeros(n, dtype=bool)
        if self._vm_min_overridden or self._vm_max_overridden:
            try:
                infeeds = self.case.infeed_buses
            except Exception:
                infeeds = set()
            if infeeds and len(infeeds) < n:
                infeed_mask = np.isin(net.bus.index.to_numpy(), list(infeeds))
        if "min_vm_pu" in net.bus.columns:
            stated = net.bus.min_vm_pu.to_numpy(dtype=float)
            use = np.isfinite(stated) & (stated > 0.0)
            if self._vm_min_overridden:
                use = use & infeed_mask
            lower[use] = stated[use]
        if "max_vm_pu" in net.bus.columns:
            stated = net.bus.max_vm_pu.to_numpy(dtype=float)
            use = np.isfinite(stated) & (stated < 2.0)
            if self._vm_max_overridden:
                use = use & infeed_mask
            upper[use] = stated[use]
        return lower, upper

    @staticmethod
    def _held_buses(net) -> np.ndarray:
        """Buses whose magnitude an in-service ``gen`` holds at its setpoint."""
        gen = getattr(net, "gen", None)
        if gen is None or not len(gen):
            return np.zeros(len(net.bus), dtype=bool)
        live = gen[gen.in_service.astype(bool)] if "in_service" in gen.columns else gen
        return np.isin(net.bus.index.to_numpy(), live.bus.to_numpy())

    def _branch_loading_limits(self, elements) -> np.ndarray:
        """Per-element thermal loading limit in percent."""
        limits = np.full(len(elements), self.max_loading_percent, dtype=float)
        if "max_loading_percent" in elements.columns:
            stated = elements.max_loading_percent.to_numpy(dtype=float)
            use = np.isfinite(stated) & (stated >= 0.0)
            limits[use] = np.minimum(limits[use], stated[use])
        return limits

    def _prepare_net(self):
        """Deep-copy the network and attach one controllable load per bus."""
        import copy as _copy

        import pandapower as pp

        net = _copy.deepcopy(self.case.net)
        # A feasibility query needs branch currents, even if the caller's
        # ordinary load-flow configuration requests voltage results only.
        if "user_pf_options" in net:
            net.user_pf_options["only_v_results"] = False
        # GridCase's base vectors define the requested operating point and
        # may legitimately differ from the native tables (for example after
        # dataclasses.replace).  Offset the injections actually in this net,
        # so the public absolute-P/Q query is realised in either case.
        base_p, base_q = np.zeros(self.case.n_bus), np.zeros(self.case.n_bus)
        for table, sign in (("load", 1.0), ("sgen", -1.0), ("storage", 1.0)):
            frame = getattr(net, table, None)
            if frame is None or not len(frame):
                continue
            live = frame[frame.in_service.astype(bool)] if "in_service" in frame.columns else frame
            scaling = (live.scaling.to_numpy(dtype=float) if "scaling" in live.columns
                       else np.ones(len(live)))
            buses = live.bus.to_numpy(dtype=int)
            values = {column: live[column] for column in ("p_mw", "q_mvar")
                      if column in live.columns}
            if table == "sgen" and isinstance(self, AcPowerFlowOracle):
                from OpenRC.cases.pandapower_adapter import _effective_sgen_pq

                eff_p, eff_q = _effective_sgen_pq(net, live)
                if eff_p is not None:
                    values["p_mw"] = eff_p
                if eff_q is not None:
                    values["q_mvar"] = eff_q
            for column, aggregate in (("p_mw", base_p), ("q_mvar", base_q)):
                if column in values:
                    np.add.at(aggregate, buses, sign * values[column].to_numpy(dtype=float) * scaling)
        self._native_base_p, self._native_base_q = base_p, base_q
        buses = np.asarray(self.case.non_slack_buses, dtype=np.intp)
        names = [f"__openrc_control_{bus}" for bus in buses]
        create_many = getattr(pp, "create_loads", None)
        if len(buses) and callable(create_many):
            rows = np.asarray(create_many(net, buses=buses, p_mw=0.0, q_mvar=0.0,
                                          name=names), dtype=np.intp)
        else:
            # Keep older pandapower installations and extension builds usable.
            rows = np.asarray([
                pp.create_load(net, bus=int(bus), p_mw=0.0, q_mvar=0.0, name=name)
                for bus, name in zip(buses, names)
            ], dtype=np.intp)
        self._control_buses, self._control_rows = buses, rows
        self._control_load = {int(bus): int(row) for bus, row in zip(buses, rows)}
        self._net = net
        return net

    def _apply(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> None:
        net = self._net if self._net is not None else self._prepare_net()
        # These are the native, already-scaled fixed injections, measured
        # before the control loads were attached; PV generators remain native.
        base_p, base_q = self._native_base_p, self._native_base_q
        if len(self._control_buses) >= 32:
            # Scalar .at is cheaper for tiny feeders.  On large networks one
            # indexed assignment avoids hundreds of Python/pandas lookups.
            buses = self._control_buses
            net.load.loc[self._control_rows, ["p_mw", "q_mvar"]] = np.column_stack(
                (p_mw[buses] - base_p[buses], q_mvar[buses] - base_q[buses]))
            return
        for bus, load_idx in self._control_load.items():
            net.load.at[load_idx, "p_mw"] = float(p_mw[bus]) - float(base_p[bus])
            net.load.at[load_idx, "q_mvar"] = float(q_mvar[bus]) - float(base_q[bus])

    def _limits_respected(self, net, *, with_voltage: bool) -> bool:
        return self._limit_margin(net, with_voltage=with_voltage)[0]

    def _requests_supplied(self, net, *, with_reactive: bool) -> bool:
        """Whether every nonzero target injection belongs to an energised bus.

        Pandapower removes unfed islands before solving, so convergence alone
        does not establish that a connection request there was served.  The
        control load is only ``target - base``: inspect its sum with the fixed
        base, allowing a request that exactly cancels an island's base load.
        A DC model has no reactive-power balance and therefore checks P only.
        """
        if not len(self._control_rows):
            return True
        dead = ~np.isfinite(net.res_bus.vm_pu.reindex(self._control_buses)
                            .to_numpy(dtype=float))
        if not np.any(dead):
            return True
        buses = self._control_buses[dead]
        rows = self._control_rows[dead]
        p = (net.load.loc[rows, "p_mw"].to_numpy(dtype=float)
             + self._native_base_p[buses])
        if np.any(p != 0.0):
            return False
        if with_reactive:
            q = (net.load.loc[rows, "q_mvar"].to_numpy(dtype=float)
                 + self._native_base_q[buses])
            if np.any(q != 0.0):
                return False
        return True

    def _limit_margin(self, net, *, with_voltage: bool) -> Tuple[bool, float]:
        """Return ``(feasible, margin)`` for voltage and branch thermal limits.

        Voltage slacks are in p.u. and thermal loading slacks in fractions of
        100%.  ``margin >= 0`` iff ``feasible`` is ``True``, and ``nan`` when
        power-flow results are non-finite or leave requested injections unserved.
        """
        slacks = []
        # pandapower omits loads on isolated/de-energised buses without raising;
        # treat unserved non-zero connection requests as violations.
        if not self._requests_supplied(net, with_reactive=with_voltage):
            return False, math.nan
        if with_voltage:
            vm = net.res_bus.vm_pu.to_numpy(dtype=float)
            # pandapower reports de-energised (isolated or out-of-service)
            # buses as NaN; they have no voltage to violate.  Infeed buses
            # hold their setpoint, which the branch-flow models do not check
            # either; checking it only pins the margin at zero.
            live = np.isfinite(vm)
            try:
                infeeds = list(self.case.infeed_buses)
            except Exception:
                infeeds = []
            live &= ~np.isin(net.bus.index.to_numpy(), infeeds)
            if not np.any(np.isfinite(vm)):
                return False, math.nan
            lower, upper = self._bus_vm_limits(net)
            if np.any(live):
                # A PV bus held exactly at its limit returns 1.05 + 2e-16:
                # clear an overrun inside a 1e-9 relative band, at such buses
                # only, so a margin search that converges onto a PQ bus's
                # limit sees the same margins as before.
                tol = 1e-9
                low = vm[live] - lower[live]
                high = upper[live] - vm[live]
                held = self._held_buses(net)[live]
                low = np.where(held & (low < 0.0) & (low >= -tol * lower[live]), 0.0, low)
                high = np.where(held & (high < 0.0) & (high >= -tol * upper[live]), 0.0, high)
                slacks += [float(np.min(low)), float(np.min(high))]
        # ``runpp`` reports each branch family separately: ``res_line`` alone
        # would miss overloaded transformers, and a transformer-only network
        # has no line results at all.
        for table, result_table in (("line", "res_line"),
                                    ("trafo", "res_trafo"),
                                    ("trafo3w", "res_trafo3w")):
            elements = getattr(net, table, None)
            results = getattr(net, result_table, None)
            if elements is None or results is None or len(elements) == 0:
                continue
            loading = results.loading_percent.to_numpy(dtype=float)
            if np.any(np.isinf(loading)):
                return False, math.nan
            limits = self._branch_loading_limits(elements)
            energised = np.isfinite(loading)      # NaN: element of a de-energised island
            if np.any(energised):
                slacks.append(float(np.min(limits[energised] - loading[energised])) / 100.0)
        if not slacks:
            return True, math.inf
        margin = min(slacks)
        return margin >= 0.0, margin

    def _judge(self, *, with_voltage: bool) -> Tuple[bool, float]:
        """The verdict by :meth:`_limits_respected`, the margin by :meth:`_limit_margin`.

        They agree unless a subclass overrides one of them; the verdict then
        follows the override and the margin is dropped (``nan``).
        """
        checker = self._limits_respected
        if getattr(checker, "__func__", None) is _PandapowerOracle._limits_respected:
            # The normal checker already delegates to this same margin scan.
            # Custom checkers still run and keep their independent verdict.
            return self._limit_margin(self._net, with_voltage=with_voltage)
        feasible = checker(self._net, with_voltage=with_voltage)
        agrees, margin = self._limit_margin(self._net, with_voltage=with_voltage)
        return feasible, (margin if agrees == feasible else math.nan)

    @property
    def numba(self) -> bool:
        """Whether pandapower's numba-compiled power flow is used."""
        if self._numba_option is None:
            try:
                from pandapower.auxiliary import NUMBA_INSTALLED
            except ImportError:  # pandapower is always present
                NUMBA_INSTALLED = False
            self._numba_option = bool(NUMBA_INSTALLED)
        return self._numba_option

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            numba=self.numba,
            max_loading_percent=self.max_loading_percent,
            vm_limits=[self.vm_min_pu, self.vm_max_pu] if self.check_voltage else None,
        )
        return info


def _network_has_angle_shift(net) -> bool:
    """Whether ``net`` specifies nonzero slack or transformer phase angles."""
    for table, slack_only, columns in (
        ("ext_grid", False, ("va_degree",)),
        ("gen", True, ("va_degree",)),
        ("trafo", False, ("shift_degree",)),
        ("trafo3w", False, ("shift_mv_degree", "shift_lv_degree")),
    ):
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0:
            continue
        live = frame[frame["in_service"].astype(bool)] if "in_service" in frame.columns else frame
        if slack_only:
            if "slack" not in live.columns:
                continue
            live = live[live["slack"].astype(bool)]
        if len(live) == 0:
            continue
        for col in columns:
            if col in live.columns:
                vals = live[col].to_numpy(dtype=float)
                if np.any(np.isfinite(vals) & (np.abs(vals) > 1e-9)):
                    return True
    return False


class AcPowerFlowOracle(_PandapowerOracle):
    """Newton-Raphson AC power flow plus voltage and thermal limit checks.

    Reference AC power-flow model used by default in
    :func:`OpenRC.validation.evaluate_region`.
    """

    inner_approximation = None   # reference model (neither inner nor outer approximation)
    supports_margin = True
    prefers_margin_search = True  # a query costs milliseconds: interpolate

    def __init__(self, case, *, check_voltage: bool = True,
                 max_iteration: int = 100, retry: bool = True,
                 quiet: bool = True, recycle: bool = True, **kwargs) -> None:
        super().__init__(case, name="acpf", check_voltage=check_voltage, **kwargs)
        if (not isinstance(max_iteration, (int, np.integer))
                or isinstance(max_iteration, (bool, np.bool_)) or max_iteration < 1):
            raise ValueError("max_iteration must be a positive integer")
        for flag_name, flag_val in (("retry", retry), ("quiet", quiet), ("recycle", recycle)):
            if not isinstance(flag_val, (bool, np.bool_)):
                raise ValueError(f"{flag_name} must be a boolean")
        self.max_iteration = int(max_iteration)
        #: Try the alternative initialisations in :attr:`RETRIES` before
        #: declaring a point inconclusive.
        self.retry = bool(retry)
        self.quiet = bool(quiet)
        #: Reuse pandapower's Ybus/internal structures across queries
        #: (``recycle={"bus_pq": True, ...}``) when the previous solve was
        #: feasible on the high-voltage branch.  After an infeasible or
        #: non-convergent query, warm-starting can converge onto a low-voltage
        #: branch (e.g. 0.16 p.u. on ``case33bw`` where flat start finds
        #: 0.94 p.u.), so the next query starts flat, and any failed recycled
        #: solve falls back through :attr:`RETRIES`.
        self.recycle = bool(recycle)
        self._recyclable = False       # the stored solution is a feasible one
        self._has_angle_shift = False

    def _prepare_net(self):
        net = super()._prepare_net()
        self._has_angle_shift = _network_has_angle_shift(net)
        return net

    #: Fallback ``(init, algorithm)`` sequence tried when flat-start Newton-Raphson
    #: fails to converge.
    RETRIES = (
        {"init": "flat", "algorithm": "nr"},
        {"init": "dc", "algorithm": "nr"},
        {"init": "flat", "algorithm": "iwamoto_nr"},
        {"init": "flat", "algorithm": "bfsw"},
    )

    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE

    def reset_warm_state(self) -> None:
        """The next query starts flat (see :meth:`FeasibilityOracle.reset_warm_state`)."""
        with self._lock:
            self._recyclable = False

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """Evaluate AC power-flow convergence and operating limits.

        Returns :attr:`Verdict.INCONCLUSIVE` on non-convergence so validation
        can escalate unresolved points to an adjudicator instead of counting
        them as infeasible.
        """
        return self._verdict_margin(p_mw, q_mvar)[0]

    def _verdict_margin(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Tuple[Verdict, float]:
        newton = self._run(p_mw, q_mvar)
        if newton is None:
            return Verdict.INCONCLUSIVE, math.nan
        feasible, margin = self._judge(with_voltage=self.check_voltage)
        # Only a Newton-Raphson model can be recycled, and only from a feasible
        # state (see ``recycle``).
        self._recyclable = feasible and newton
        return (Verdict.FEASIBLE if feasible else Verdict.INFEASIBLE), margin

    def _run(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Optional[bool]:
        """Solve the power flow: ``None`` if no attempt converged, else whether
        the converged attempt was Newton-Raphson."""
        import pandapower as pp

        self._apply(p_mw, q_mvar)
        if self.recycle and self._recyclable:
            self._recyclable = False
            try:
                pp.runpp(self._net, algorithm="nr", max_iteration=self.max_iteration,
                         numba=self.numba, only_v_results=False,
                         recycle={"bus_pq": True, "trafo": False, "gen": False})
            except Exception:
                pass                        # fall back to the full chain below
            else:
                # Numerical convergence is not enough: on a multi-solution
                # power-flow problem a warm start can reach a low-voltage,
                # limit-violating branch even though a flat start reaches the
                # physical feasible branch.  Accept a recycled result only if
                # it is already feasible; otherwise confirm it from flat.
                if self._limits_respected(self._net, with_voltage=self.check_voltage):
                    return True
        retries = self.RETRIES
        if self._has_angle_shift and retries is AcPowerFlowOracle.RETRIES:
            # Nonzero slack or transformer phase angles put va=0 far from the
            # physical branch; a DC angle seed prevents Newton-Raphson from
            # crossing onto a slipped-pole solution.
            retries = (retries[1], retries[0]) + retries[2:]
        for attempt in (retries if self.retry else retries[:1]):
            try:
                if self.quiet and attempt["algorithm"] == "iwamoto_nr":
                    with _IWAMOTO_OUTPUT_LOCK, contextlib.redirect_stdout(io.StringIO()), \
                            contextlib.redirect_stderr(io.StringIO()):
                        pp.runpp(
                            self._net, init=attempt["init"],
                            algorithm=attempt["algorithm"],
                            max_iteration=self.max_iteration, numba=self.numba,
                            only_v_results=False,
                        )
                else:
                    pp.runpp(
                        self._net, init=attempt["init"], algorithm=attempt["algorithm"],
                        max_iteration=self.max_iteration, numba=self.numba,
                        only_v_results=False,
                    )
            except Exception:
                continue
            return attempt["algorithm"] == "nr"
        return None

    def describe(self) -> dict:
        info = super().describe()
        info.update(max_iteration=self.max_iteration, retry=self.retry, quiet=self.quiet,
                    recycle=self.recycle)
        return info


class DcPowerFlowOracle(_PandapowerOracle):
    """DC power flow with thermal limits only - fast, no voltage information.

    Useful as a cheap screening oracle and for reproducing the DC-based accuracy
    studies of the random-acceleration experiments.
    """

    inner_approximation = None
    supports_adjudication = False   # no AC voltage/reactive feasibility guarantee
    supports_margin = True
    prefers_margin_search = True

    def __init__(self, case, *, check_voltage: bool = False, **kwargs) -> None:
        if not isinstance(check_voltage, (bool, np.bool_)):
            raise ValueError("check_voltage must be a boolean")
        if check_voltage:
            raise ValueError("dcpf is a DC model and cannot check voltage magnitude limits")
        super().__init__(case, name="dcpf", check_voltage=False, **kwargs)

    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        return self._verdict_margin(p_mw, q_mvar)[0]

    def _verdict_margin(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Tuple[Verdict, float]:
        import pandapower as pp

        self._apply(p_mw, q_mvar)
        try:
            pp.rundcpp(self._net, numba=self.numba)
        except Exception:
            return Verdict.INCONCLUSIVE, math.nan
        feasible, margin = self._judge(with_voltage=False)
        return (Verdict.FEASIBLE if feasible else Verdict.INFEASIBLE), margin


@ORACLES.register("acpf", doc="Newton-Raphson AC power flow with limit checks (ground truth).")
def _make_acpf(case, **kwargs) -> AcPowerFlowOracle:
    return AcPowerFlowOracle(case, **kwargs)


@ORACLES.register("dcpf", doc="DC power flow, thermal limits only (fast screening).")
def _make_dcpf(case, **kwargs) -> DcPowerFlowOracle:
    return DcPowerFlowOracle(case, **kwargs)
