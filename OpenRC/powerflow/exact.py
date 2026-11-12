"""Exact radial branch-flow feasibility via global non-convex QCQP.

Whereas the SOCP relaxation in :mod:`OpenRC.powerflow.socp` relaxes
:math:`l_{ij} = (P^2 + Q^2)/v_i` to an inequality, that relaxation is not
guaranteed to be exact when upper voltage bounds bind under reverse power flow.
This oracle retains the quadratic equality and solves the resulting non-convex
QCQP globally (e.g. Gurobi with ``NonConvex=2``).

The equations are formulated in per-unit on a 1 MVA three-phase base, avoiding
the large coefficient dynamic range of SI units in spatial branch-and-bound.

Intended for adjudicating contested or ``INCONCLUSIVE`` validation points:
a global solve either finds a feasible operating point or proves infeasibility
for the exact branch-flow equations, at higher computational cost than convex
or Newton-Raphson oracles.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np

from OpenRC.powerflow.base import (ORACLES, FeasibilityOracle, SolverUnavailableError,
                                  Verdict, report_solver_error)
from OpenRC.powerflow.equations import CandidateMixin
from OpenRC.powerflow.modeling import (ConicForm, DirectModel, branch_flow_form,
                                       check_modeling, direct_model, synchronized_pyomo)

# Termination conditions and what they prove.
_PROVEN_FEASIBLE = {"optimal", "globallyOptimal", "feasible"}
_PROVEN_INFEASIBLE = {"infeasible"}
#: Pyomo solver names whose "infeasible" is a global proof for this QCQP.
_GLOBAL_SOLVERS = ("gurobi", "scip", "baron", "couenne", "antigone", "octeract", "lindoglobal")
#: Match full solver identifiers (e.g. ``gurobi_direct``, ``appsi_gurobi``)
#: while excluding unrelated prefix matches such as ``scipy``.
_GLOBAL_SOLVER_NAME = re.compile(r"(?:appsi_)?(?:%s)(?:[_-][a-z0-9]+)*" % "|".join(_GLOBAL_SOLVERS))


class ExactDistFlowOracle(CandidateMixin, FeasibilityOracle):
    """Global solve of the exact branch-flow equations with all limits imposed.

    Parameters
    ----------
    case:
        The network.
    solver:
        A global non-convex QCQP solver, for ``modeling="pyomo"``.  ``"gurobi"``
        is the default; it is run with ``NonConvex=2``.  Only a known global
        solver's infeasibility status counts as a proof: a local solver
        (``ipopt``, ...) can still prove a point feasible, but its
        "infeasible" is :attr:`Verdict.INCONCLUSIVE`.
    modeling:
        ``"pyomo"`` (default) or ``"gurobi"``: the model stated directly and
        persistently in gurobipy (:mod:`OpenRC.powerflow.modeling`).  MOSEK
        solves convex problems only and cannot hold this one.  Gurobi keeps its
        default ``FeasibilityTol`` of 1e-6 per per-unit row.  Those errors can
        add up along a feeder, so within about 1e-5 p.u. of a voltage limit a
        point can be reported feasible that AC power flow places just outside.
        ``solver_options={"FeasibilityTol": 1e-9}`` is stricter but not the
        default: on ``9bus`` it reached the 60 s time limit at a point the
        default decided in 0.1 s.
    time_limit:
        Seconds before the solver terminates.  Timeouts return
        :attr:`Verdict.INCONCLUSIVE`.
    candidate_solver:
        Optional pre-solver tried before the global solve; it can only prove a
        point feasible.  ``"sweep"`` (:mod:`OpenRC.powerflow.sweep`) certifies
        a solution within every limit by closed-form sweeps and a contraction
        bound.  A registered equation solver also works on this model's
        ``ConicForm``.  ``candidate_min_rate`` and ``candidate_patience``
        disable the pre-solver if its acceptance rate falls below
        ``candidate_min_rate``.
    """

    inner_approximation = None   # exact branch-flow reference model
    certifies_infeasibility = True   # only a global infeasibility status is decisive

    def __init__(self, case, *, solver: str = "gurobi", modeling: str = "pyomo",
                 time_limit: float = 60.0, solver_options: Optional[dict] = None,
                 candidate_solver: Optional[str] = None, candidate_starts: int = 1,
                 candidate_seed: Optional[int] = 0, candidate_options: Optional[dict] = None,
                 candidate_tolerance: float = 1e-9, candidate_min_rate: float = 0.05,
                 candidate_patience: int = 20, name: Optional[str] = None) -> None:
        check_modeling(modeling)
        if modeling == "mosek":
            raise ValueError("the exact model is non-convex; MOSEK cannot solve it - use "
                             "modeling='gurobi' or 'pyomo'")
        if not isinstance(solver, str) or not solver.strip():
            raise ValueError("solver must be a non-empty string")
        if solver_options is not None and not isinstance(solver_options, dict):
            raise TypeError("solver_options must be a dict")
        super().__init__(case, name=name or "exact")
        self.solver_name = solver
        self.modeling = modeling
        self._direct: Optional[DirectModel] = None
        if isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, (int, float, np.number)):
            raise ValueError("time_limit must be finite and strictly positive")
        self.time_limit = float(time_limit)
        if not math.isfinite(self.time_limit) or self.time_limit <= 0.0:
            raise ValueError("time_limit must be finite and strictly positive")
        self.solver_options = dict(solver_options or {})
        # Built lazily by _build(); Pyomo objects carry no type information.
        self._model: Any = None
        self._solver: Any = None
        self._p_param: Any = None
        self._q_param: Any = None
        self._s_base_va = 1.0e6
        self._p_scale = 1.0
        self.last_solve: Optional[dict] = None
        self._init_candidate(candidate_solver, starts=candidate_starts, seed=candidate_seed,
                             options=candidate_options, tolerance=candidate_tolerance,
                             min_rate=candidate_min_rate, patience=candidate_patience,
                             structured=True)

    def _build(self):
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory

        case = self.case
        topo = case.topology
        v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        s_base_phase = self._s_base_va / 3.0
        z_base = v_base ** 2 / s_base_phase
        i_base = s_base_phase / v_base
        self._p_scale = 1.0e6 / self._s_base_va
        v_slack_sq = case.slack_vm_pu ** 2

        m = pyo.ConcreteModel(name=f"{case.name}:exact")
        m.BUS = pyo.Set(initialize=topo.buses, ordered=True)
        m.BR = pyo.Set(initialize=[br.index for br in topo.branches], ordered=True)

        m.p_inj = pyo.Param(m.BUS, mutable=True, initialize=0.0, within=pyo.Reals)
        m.q_inj = pyo.Param(m.BUS, mutable=True, initialize=0.0, within=pyo.Reals)

        m.v2 = pyo.Var(m.BUS, bounds=(case.vm_min_pu ** 2,
                                      case.vm_max_pu ** 2))
        m.l2 = pyo.Var(m.BR, domain=pyo.NonNegativeReals)
        m.P = pyo.Var(m.BR)
        m.Q = pyo.Var(m.BR)
        for br in topo.branches:
            if br.rated:               # an unrated branch has no thermal limit
                m.l2[br.index].setub(br.l_max / i_base ** 2)

        def v_of(bus):
            return v_slack_sq if bus == topo.slack else m.v2[bus]

        m.cons = pyo.ConstraintList()
        for br in topo.branches:
            k, child = br.index, br.child
            downstream = topo.children_of[child]
            r_pu = br.r_ohm / z_base
            x_pu = br.x_ohm / z_base
            m.cons.add(m.P[k] == sum(m.P[d.index] for d in downstream)
                       + r_pu * m.l2[k] + m.p_inj[child])
            m.cons.add(m.Q[k] == sum(m.Q[d.index] for d in downstream)
                       + x_pu * m.l2[k] + m.q_inj[child])
            m.cons.add(v_of(br.parent) - m.v2[child]
                       == 2.0 * (r_pu * m.P[k] + x_pu * m.Q[k])
                       - (r_pu ** 2 + x_pu ** 2) * m.l2[k])
            # Non-convex branch-current definition: l_k * v_parent == P_k^2 + Q_k^2.
            m.cons.add(m.l2[k] * v_of(br.parent) == m.P[k] ** 2 + m.Q[k] ** 2)

        m.obj = pyo.Objective(expr=0.0)   # pure feasibility problem

        solver = SolverFactory(self.solver_name)
        if not solver.available(exception_flag=False):
            raise SolverUnavailableError(
                f"solver {self.solver_name!r} is not available; the exact oracle needs a "
                "global non-convex QCQP solver such as Gurobi"
            )
        if self.solver_name.startswith("gurobi"):
            solver.options["NonConvex"] = 2
            solver.options["TimeLimit"] = self.time_limit
        for key, value in self.solver_options.items():
            solver.options[key] = value

        self._model, self._solver = m, solver
        self._p_param, self._q_param = m.p_inj, m.q_inj
        return m

    def _conic_form(self) -> ConicForm:
        """The model of :meth:`_build` (per unit), for a direct modelling layer."""
        case = self.case
        topo = case.topology
        v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        return branch_flow_form(
            topo, scale=1e6 / 3.0, v_slack=(v_base * case.slack_vm_pu) ** 2,
            v_bounds=((v_base * case.vm_min_pu) ** 2, (v_base * case.vm_max_pu) ** 2),
            losses="exact",
            l_upper={br.index: br.l_max for br in topo.branches if br.rated},
            base=(self._s_base_va / 3.0, v_base))

    def conic_form(self) -> ConicForm:
        """Public solver-neutral form of the exact branch-flow model."""
        return self._conic_form().with_case_charging(self.case)

    def equation_problem(
        self, p_mw: np.ndarray, q_mvar: np.ndarray, *,
        x0: Optional[np.ndarray] = None, tolerance: float = 1e-9,
    ):
        """Return this query as a public SciPy-compatible equation problem."""
        return self._conic_form().scipy_problem(
            p_mw, self.case.branch_flow_q(q_mvar), x0=x0, tolerance=tolerance
        )

    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        q_mvar = self.case.branch_flow_q(q_mvar)
        candidate = self._candidate_verdict(p_mw, q_mvar)
        if candidate is not None:
            self.last_solve = {"verdict": candidate.name, "candidate": self.candidate_solver}
            return candidate
        if self.modeling != "pyomo":
            if self._direct is None:
                self._direct = direct_model(self.modeling, self._conic_form(),
                                            time_limit=self.time_limit,
                                            options=self.solver_options)
            try:
                verdict = self._direct.verdict(p_mw, q_mvar)
            except Exception as exc:
                report_solver_error(self.name, exc)
                self.last_solve = {
                    "verdict": Verdict.INCONCLUSIVE.name,
                    "modeling": self.modeling,
                    "termination_condition": "exception",
                    "message": str(exc),
                }
                return Verdict.INCONCLUSIVE
            self.last_solve = {"verdict": verdict.name, "modeling": self.modeling}
            return verdict
        return self._pyomo_verdict(p_mw, q_mvar)

    @synchronized_pyomo
    def _pyomo_verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """Protect Pyomo's global capture state across independent oracles."""
        if self._model is None:
            self._build()
        for bus in self.case.topology.buses:
            # On a 1 MVA three-phase base, a public MW value has the same
            # numeric value in per-unit. Keep the expression explicit so a
            # future base change remains correct.
            self._p_param[bus].value = float(p_mw[bus]) * self._p_scale
            self._q_param[bus].value = float(q_mvar[bus]) * self._p_scale

        try:
            results = self._solver.solve(self._model, load_solutions=False)
        except Exception as exc:
            report_solver_error(self.name, exc)
            self.last_solve = {"termination_condition": "exception", "message": str(exc)}
            return Verdict.INCONCLUSIVE

        tc = str(results.solver.termination_condition)
        self.last_solve = {
            "termination_condition": tc,
            "status": str(results.solver.status),
            "message": str(getattr(results.solver, "message", "")),
        }
        if tc in _PROVEN_FEASIBLE:
            return Verdict.FEASIBLE
        if tc in _PROVEN_INFEASIBLE:
            # A local NLP solver (ipopt, ...) reports "infeasible" for a locally
            # infeasible point; only a global solver's status is a proof.
            if _GLOBAL_SOLVER_NAME.fullmatch(self.solver_name.lower()):
                return Verdict.INFEASIBLE
            return Verdict.INCONCLUSIVE
        # Unresolved termination (maxTimeLimit, maxIterations, unknown, error): return INCONCLUSIVE.
        return Verdict.INCONCLUSIVE

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            modeling=self.modeling,
            solver=self.solver_name if self.modeling == "pyomo" else None,
            time_limit=self.time_limit,
            scaling="per-unit on a 1 MVA three-phase base",
            last_solve=self.last_solve,
            **self._candidate_info(),
        )
        return info


@ORACLES.register(
    "exact",
    doc="Global solve of the exact branch-flow equations (Gurobi NonConvex=2); adjudicator.",
)
def _make_exact(case, **kwargs) -> ExactDistFlowOracle:
    return ExactDistFlowOracle(case, **kwargs)
