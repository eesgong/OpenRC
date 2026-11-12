"""Exact branch-flow feasibility via its second-order-cone relaxation.

It is the convex outer counterpart of the exact model
(:mod:`OpenRC.powerflow.exact`).  Unlike the LinDistFlow variants it keeps the
squared branch current :math:`l_{ij}` as a decision variable and enforces

.. math::  l_{ij} v_i \\ge P_{ij}^2 + Q_{ij}^2

which, being a product of two variables, is a *rotated* quadratic cone
:math:`\\|x\\|^2 \\le 2 r_1 r_2` with :math:`r_1 = v_i/2`, :math:`r_2 = l_{ij}`.
It contains every exact solution, so an *infeasible* verdict is a proof
(``inner_approximation = False``).  A *feasible* verdict does not guarantee
exact feasibility: when upper voltage limits bind under reverse power flow, the
relaxation can become inexact by dissipating surplus power in artificially
inflated ``l_{ij}`` (e.g. on ``case33bw``, generation-direction ray boundaries
can exceed exact/ACPF boundaries by 8-12x).

Because the cone must be declared explicitly (passing ``P**2 + Q**2 <= l*v`` as
a general quadratic constraint causes solvers to reject the bilinear term
``l * v`` as an indefinite quadratic constraint), the Pyomo backend builds the
model with ``pyomo.kernel`` rather than ``pyomo.environ``.

Like the exact oracle, the model is stated in per-unit on a 1 MVA three-phase
base.  The SI form is algebraically identical, but its coefficients span about
eight orders of magnitude, enough for Gurobi's barrier to stop short of its
optimality tolerances at feasible points (an inconclusive verdict).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from OpenRC.powerflow.base import (ORACLES, FeasibilityOracle, SolverUnavailableError,
                                  Verdict, report_solver_error)
from OpenRC.powerflow.equations import CandidateMixin, EquationProblem
from OpenRC.powerflow.modeling import (ConicForm, DirectModel, branch_flow_form,
                                       check_modeling, direct_model, synchronized_pyomo)


class SocpBranchFlowOracle(CandidateMixin, FeasibilityOracle):
    """Feasibility under the SOC relaxation of the exact branch-flow model.

    ``modeling`` chooses how the model is stated: ``"pyomo"`` (default; any
    conic solver through ``solver=``), or directly and persistently in
    ``"mosek"`` (Fusion) or ``"gurobi"`` - see :mod:`OpenRC.powerflow.modeling`.
    """

    #: Outer relaxation of the exact branch-flow feasible set.
    inner_approximation = False

    def __init__(self, case, *, solver: str = "mosek", modeling: str = "pyomo",
                 solver_options: Optional[dict] = None,
                 candidate_solver: Optional[str] = None,
                 candidate_starts: int = 1,
                 candidate_seed: Optional[int] = 0,
                 candidate_options: Optional[dict] = None,
                 candidate_tolerance: float = 1e-9,
                 candidate_min_rate: float = 0.05,
                 candidate_patience: int = 20,
                 name: Optional[str] = None) -> None:
        check_modeling(modeling)
        if not isinstance(solver, str) or not solver.strip():
            raise ValueError("solver must be a non-empty string")
        if solver_options is not None and not isinstance(solver_options, dict):
            raise TypeError("solver_options must be a dict")
        super().__init__(case, name=name or "socp")
        self.solver_name = solver
        self.modeling = modeling
        self._direct: Optional[DirectModel] = None
        self.solver_options = dict(solver_options or {})
        # ``"sweep"`` (structured) proves the exact model feasible, which the
        # relaxation contains.
        self._init_candidate(candidate_solver, starts=candidate_starts, seed=candidate_seed,
                             options=candidate_options, tolerance=candidate_tolerance,
                             min_rate=candidate_min_rate, patience=candidate_patience,
                             structured=True)
        self._model = None
        self._solver = None
        self._s_base_va = 1.0e6
        self._p_param: dict = {}
        self._q_param: dict = {}

    def _build(self):
        import pyomo.kernel as pmo
        from pyomo.opt import SolverFactory

        case = self.case
        topo = case.topology
        v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        s_base_phase = self._s_base_va / 3.0
        z_base = v_base ** 2 / s_base_phase
        i_base = s_base_phase / v_base
        v_slack_sq = case.slack_vm_pu ** 2
        v_lo_sq = case.vm_min_pu ** 2
        v_hi_sq = case.vm_max_pu ** 2

        m = pmo.block()
        m.v2 = pmo.variable_dict()
        m.p_inj = pmo.parameter_dict()
        m.q_inj = pmo.parameter_dict()
        for bus in topo.buses:
            m.v2[bus] = pmo.variable(lb=v_lo_sq, ub=v_hi_sq)
            m.p_inj[bus] = pmo.parameter(value=0.0)
            m.q_inj[bus] = pmo.parameter(value=0.0)

        m.P = pmo.variable_dict()
        m.Q = pmo.variable_dict()
        m.l2 = pmo.variable_dict()
        # r1 mirrors half the sending-end squared voltage; the cone needs a
        # variable there, so it is tied to v2 (or to the slack constant) below.
        m.r1 = pmo.variable_dict()
        for br in topo.branches:
            m.P[br.index] = pmo.variable()
            m.Q[br.index] = pmo.variable()
            # Leave l2 unbounded above on unrated branches; large placeholder
            # upper bounds degrade barrier conditioning and can cause false infeasibility.
            m.l2[br.index] = pmo.variable(
                lb=0.0, ub=br.l_max / i_base ** 2 if br.rated else None)
            m.r1[br.index] = pmo.variable(lb=0.0)

        m.cons = pmo.constraint_list()
        # Constructed directly (not via as_domain), a conic domain IS a constraint,
        # so it belongs in a constraint_list.
        m.cones = pmo.constraint_list()

        def v_of(bus):
            return v_slack_sq if bus == topo.slack else m.v2[bus]

        for br in topo.branches:
            k = br.index
            child = br.child
            downstream = topo.children_of[child]
            r_pu = br.r_ohm / z_base
            x_pu = br.x_ohm / z_base

            # exact power balance - losses are carried by l
            m.cons.append(pmo.constraint(
                m.P[k] == sum(m.P[d.index] for d in downstream)
                + r_pu * m.l2[k] + m.p_inj[child]
            ))
            m.cons.append(pmo.constraint(
                m.Q[k] == sum(m.Q[d.index] for d in downstream)
                + x_pu * m.l2[k] + m.q_inj[child]
            ))

            # exact voltage drop
            m.cons.append(pmo.constraint(
                v_of(br.parent) - m.v2[child]
                == 2.0 * (r_pu * m.P[k] + x_pu * m.Q[k])
                - (r_pu ** 2 + x_pu ** 2) * m.l2[k]
            ))

            # rotated cone  P^2 + Q^2 <= 2 * r1 * l  with  r1 = v_parent / 2
            m.cons.append(pmo.constraint(2.0 * m.r1[k] == v_of(br.parent)))
            m.cones.append(
                pmo.conic.rotated_quadratic(r1=m.r1[k], r2=m.l2[k],
                                            x=[m.P[k], m.Q[k]])
            )

        # Minimising ohmic losses drives the relaxation towards tightness, which
        # is the behaviour wanted from a reference model.
        m.obj = pmo.objective(
            sum(br.r_ohm / z_base * m.l2[br.index] for br in topo.branches),
            sense=pmo.minimize,
        )

        solver = SolverFactory(self.solver_name)
        if not solver.available(exception_flag=False):
            raise SolverUnavailableError(
                f"solver {self.solver_name!r} is not available; the SOCP oracle needs a "
                "conic solver such as MOSEK"
            )
        for key, value in self.solver_options.items():
            solver.options[key] = value

        self._model, self._solver = m, solver
        self._p_param, self._q_param = m.p_inj, m.q_inj
        return m

    def _conic_form(self) -> ConicForm:
        """The model of :meth:`_build` (per unit), for a direct modelling layer."""
        case = self.case
        topo = case.topology
        v_nom = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        return branch_flow_form(
            topo, scale=1e6 / 3.0, v_slack=(v_nom * case.slack_vm_pu) ** 2,
            v_bounds=((v_nom * case.vm_min_pu) ** 2, (v_nom * case.vm_max_pu) ** 2),
            losses="relaxed",
            l_upper={br.index: br.l_max for br in topo.branches if br.rated},
            base=(1e6 / 3.0, v_nom), objective="losses")

    def conic_form(self) -> ConicForm:
        """Public solver-neutral form of the SOCP relaxation."""
        return self._conic_form().with_case_charging(self.case)

    def equation_problem(
        self, p_mw: np.ndarray, q_mvar: np.ndarray, *,
        x0: Optional[np.ndarray] = None, tolerance: float = 1e-9,
    ) -> EquationProblem:
        """Return this query as a SciPy-compatible equation problem."""
        return self._conic_form().scipy_problem(
            p_mw, self.case.branch_flow_q(q_mvar), x0=x0, tolerance=tolerance
        )

    def _optimizer_verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        if self.modeling != "pyomo":
            if self._direct is None:
                self._direct = direct_model(self.modeling, self._conic_form(),
                                            options=self.solver_options)
            try:
                return self._direct.verdict(p_mw, q_mvar)
            except Exception as exc:
                report_solver_error(self.name, exc)
                return Verdict.INCONCLUSIVE
        return self._pyomo_verdict(p_mw, q_mvar)

    @synchronized_pyomo
    def _pyomo_verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """Keep all Pyomo adapter work inside the shared process-wide guard."""
        if self._model is None:
            self._build()
        scale = 1e6 / self._s_base_va     # three-phase MW -> per unit
        for bus in self.case.topology.buses:
            self._p_param[bus].value = float(p_mw[bus]) * scale
            self._q_param[bus].value = float(q_mvar[bus]) * scale

        from OpenRC.powerflow.distflow import _solve_verdict

        return _solve_verdict(self._solver, self._model, name=self.name)

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        q_mvar = self.case.branch_flow_q(q_mvar)
        candidate = self._candidate_verdict(p_mw, q_mvar)
        return candidate if candidate is not None else self._optimizer_verdict(p_mw, q_mvar)

    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE

    def describe(self) -> dict:
        info = super().describe()
        info.update(modeling=self.modeling,
                    solver=self.solver_name if self.modeling == "pyomo" else None,
                    **self._candidate_info())
        return info


@ORACLES.register(
    "socp",
    doc="Exact branch flow via its second-order-cone relaxation (reference model).",
)
def _make_socp(case, **kwargs) -> SocpBranchFlowOracle:
    return SocpBranchFlowOracle(case, **kwargs)
