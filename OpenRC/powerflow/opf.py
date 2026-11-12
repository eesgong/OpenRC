"""Optimal-power-flow feasibility oracles for networks with dispatchable generation.

Whereas fixed-dispatch power-flow and branch-flow models evaluate a fixed
generator schedule, optimal-power-flow oracles test whether an operating point
exists when controllable generators are dispatched within limits:

``dcopf``
    Linear DC optimal power flow.  A converged solve is feasible; solver
    exceptions return :attr:`~OpenRC.powerflow.base.Verdict.INCONCLUSIVE`
    because ``pandapower`` does not distinguish solver failure from certified
    infeasibility in its exception path.
``acopf``
    Full non-convex AC optimal power flow.  Non-convergence returns
    :attr:`~OpenRC.powerflow.base.Verdict.INCONCLUSIVE`.

Neither oracle requires radiality, enabling region construction on meshed
networks.  Networks must define generator costs (``poly_cost`` or ``pwl_cost``)
and finite active-power bounds on controllable generators; missing data raises
:class:`NotOpfReadyError`.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np

from OpenRC.powerflow.acpf import _PandapowerOracle
from OpenRC.powerflow.base import ORACLES, Verdict


class NotOpfReadyError(ValueError):
    """The network lacks the data an optimal power flow needs."""


class _OpfOracle(_PandapowerOracle):
    """Base class for ``pandapower`` DC and AC optimal-power-flow oracles."""

    #: An OPF enforces its operating limits as constraints.  Its island
    #: preprocessing can still discard a firm connection request, so supply
    #: is checked independently after convergence.
    inner_approximation = None

    def __init__(self, case, *, name: str, check_voltage: bool,
                 quiet: bool = True, **kwargs) -> None:
        super().__init__(case, name=name, check_voltage=check_voltage, **kwargs)
        if not isinstance(quiet, (bool, np.bool_)):
            raise ValueError("quiet must be a boolean")
        #: Suppress pandapower's per-solve log chatter - for instance the notice
        #: that a generator voltage setpoint was clamped to its bus limit, which
        #: is benign in an OPF and would otherwise be printed thousands of times
        #: during a region build.  Set False to see it while diagnosing.
        self.quiet = bool(quiet)

    @contextlib.contextmanager
    def _solver_logging(self):
        """Temporarily raise pandapower's log level while a solve runs."""
        if not self.quiet:
            yield
            return
        logger = logging.getLogger("pandapower")
        previous = logger.level
        logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            logger.setLevel(previous)

    def _prepare_net(self):
        net = super()._prepare_net()
        self._require_opf_ready(net)
        # Write voltage and branch-loading limits into net tables read by pandapower OPF.
        if self.check_voltage:
            lower, upper = self._bus_vm_limits(net)
            net.bus["min_vm_pu"] = lower
            net.bus["max_vm_pu"] = upper
        else:
            net.bus["min_vm_pu"] = np.nan
            net.bus["max_vm_pu"] = np.nan
        for table in ("line", "trafo", "trafo3w"):
            frame = getattr(net, table, None)
            if frame is not None and len(frame):
                frame["max_loading_percent"] = self._branch_loading_limits(frame)
        # Fix controllable=False on loads so the OPF cannot curtail POC injections.
        if "controllable" in net.load.columns:
            net.load["controllable"] = False
        return net

    @staticmethod
    def _require_opf_ready(net) -> None:
        # A DataFrame has no truth value, so test for None explicitly.
        costs = 0
        for table in ("poly_cost", "pwl_cost"):
            frame = getattr(net, table, None)
            if frame is not None:
                costs += len(frame)
        if costs == 0:
            raise NotOpfReadyError(
                "this network carries no generator costs, so an optimal power flow "
                "has no objective. Add a pandapower poly_cost/pwl_cost for at least "
                "one generator or the external grid, or use the 'acpf'/'dcpf' "
                "oracles, which evaluate a fixed dispatch instead."
            )
        gen = getattr(net, "gen", None)
        if gen is not None and len(gen):
            live = gen[gen.in_service.astype(bool)] if "in_service" in gen.columns else gen
            if "controllable" in live.columns:
                live = live[live.controllable.fillna(True).astype(bool)]
            if len(live):
                required = ("min_p_mw", "max_p_mw")
                missing = [c for c in required if c not in live.columns]
                incomplete = [] if missing else [int(i) for i in live.index[
                    ~np.isfinite(live.loc[:, required].to_numpy(dtype=float)).all(axis=1)
                ]]
                if missing or incomplete:
                    detail = (f"missing columns {', '.join(missing)}" if missing
                              else f"non-finite limits on generator rows {incomplete[:10]}")
                    raise NotOpfReadyError(
                        f"the generators have incomplete active-power limits ({detail}), "
                        "so the optimal power flow has no well-defined dispatch range. Set "
                        "finite limits on every generator, or use the 'acpf'/'dcpf' "
                        "oracles."
                    )

    def _run(self, net) -> None:  # subclasses implement this
        raise NotImplementedError

    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE


def opf_ready(net) -> bool:
    """Whether ``net`` carries what an optimal power flow needs.

    That is a generator cost (``poly_cost`` or ``pwl_cost``) and finite active
    power limits on every generator - the check the OPF oracles make before
    their first solve, which otherwise raise :class:`NotOpfReadyError`.
    """
    try:
        _OpfOracle._require_opf_ready(net)
    except NotOpfReadyError:
        return False
    return True


@ORACLES.register(
    "dcopf",
    doc="DC optimal power flow: can the network be dispatched to host this injection?",
)
class DcOpfOracle(_OpfOracle):
    """Linear DC optimal power flow; no radiality requirement.

    Enforces branch thermal limits and generator active-power bounds under the
    linear DC approximation (reactive power and voltage magnitudes are omitted).
    """

    supports_adjudication = False   # DC dispatch is not an AC operating-point certificate

    def __init__(self, case, *, check_voltage: bool = False, **kwargs) -> None:
        if not isinstance(check_voltage, (bool, np.bool_)):
            raise ValueError("check_voltage must be a boolean")
        if check_voltage:
            raise ValueError("dcopf is a DC model and cannot check voltage magnitude limits")
        super().__init__(case, name="dcopf", check_voltage=False, **kwargs)

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        import pandapower as pp

        self._apply(p_mw, q_mvar)
        try:
            with self._solver_logging():
                pp.rundcopp(self._net)
        except Exception:
            return Verdict.INCONCLUSIVE
        if not bool(self._net.OPF_converged):
            return Verdict.INCONCLUSIVE
        return (Verdict.FEASIBLE if self._requests_supplied(self._net, with_reactive=False)
                else Verdict.INFEASIBLE)


@ORACLES.register(
    "acopf",
    doc="AC optimal power flow: full model, no radiality requirement.",
)
class AcOpfOracle(_OpfOracle):
    """Full AC optimal power flow.

    Enforces voltage limits, thermal limits and generator limits simultaneously.
    Because the program is non-convex, failure to converge is reported as
    :attr:`Verdict.INCONCLUSIVE`: the solver has not shown that no dispatch
    exists, only that it did not find one.

    Tries each initialisation in ``inits`` sequentially until ``pp.runopp``
    converges (default ``("pf", "flat")``, which initialises from an AC
    power-flow solution before falling back to a flat start).
    """

    INITS = ("pf", "flat", "results")

    def __init__(self, case, *, check_voltage: bool = True,
                 inits=("pf", "flat"), **kwargs) -> None:
        super().__init__(case, name="acopf", check_voltage=check_voltage, **kwargs)
        if isinstance(inits, (bytes, bytearray)):
            raise ValueError(f"inits must be a non-empty sequence of {self.INITS}")
        try:
            inits = (inits,) if isinstance(inits, str) else tuple(dict.fromkeys(inits))
        except TypeError as exc:
            raise ValueError(f"inits must be a non-empty sequence of {self.INITS}") from exc
        if not inits or any(init not in self.INITS for init in inits):
            raise ValueError(f"inits must be a non-empty sequence of {self.INITS}")
        self.inits = inits

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        import pandapower as pp

        self._apply(p_mw, q_mvar)
        for init in self.inits:
            try:
                with self._solver_logging():
                    pp.runopp(self._net, init=init)
            except Exception:
                continue
            if bool(self._net.OPF_converged):
                return (Verdict.FEASIBLE if self._requests_supplied(self._net, with_reactive=True)
                        else Verdict.INFEASIBLE)
        return Verdict.INCONCLUSIVE

    def describe(self) -> dict:
        info = super().describe()
        info.update(inits=list(self.inits))
        return info
