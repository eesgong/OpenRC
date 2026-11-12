r"""The LinDistFlow model family - four convex models built from one implementation.

All four differ only in how they handle the loss terms that the exact branch-flow
equations carry.  Writing the exact equations first makes the differences easy to
see (per phase, SI units, consumption positive):

.. math::

    P_{ij} &= \\textstyle\\sum_{k:(j,k)\\in E} P_{jk} + r_{ij} l_{ij} + P_j \\\\
    Q_{ij} &= \\textstyle\\sum_{k:(j,k)\\in E} Q_{jk} + x_{ij} l_{ij} + Q_j \\\\
    v_j    &= v_i - 2 (r_{ij} P_{ij} + x_{ij} Q_{ij}) + (r_{ij}^2 + x_{ij}^2) l_{ij} \\\\
    l_{ij} &= (P_{ij}^2 + Q_{ij}^2) / v_i \;\\le\; l^{max}_{ij}

with :math:`v^{min} \\le v_j \\le v^{max}`.  Every model below keeps the thermal
constraint in its convex form :math:`P^2 + Q^2 \\le l^{max} v_i` and differs in
three independent switches: whether the balance is charged the loss, what the
voltage-drop equation does with :math:`(r^2+x^2) l`, and whether the thermal
limit is the nameplate one or a tightened one.

======================  ==============  ==========================  ==============
oracle                  balance loss    voltage-drop loss term      thermal limit
======================  ==============  ==========================  ==============
``lindistflow``         dropped         dropped                     nameplate
``lindistflow-vdrop``   dropped         ``+(r^2+x^2) l_max``        nameplate
``tight-voltage``       dropped         ``+(r^2+x^2) l_max``        tightened
``tight-flow``          ``+r l_max``    dropped                     nameplate
======================  ==============  ==========================  ==============

Closed-form evaluation vs. conic solver
---------------------------------------
On a radial tree, branch flows and bus voltages in all four variants are
unique affine functions of the nodal injections::

    P = T p + c_P,     v = v_0 - 2 T^T (r*P + x*Q) + s T^T ((r^2+x^2) l_max)

with ``T`` the branch-by-bus subtree incidence (``T[b, j] = 1`` when bus ``j``
lies below branch ``b``), followed by the voltage bounds and the thermal
inequalities.  ``backend="analytic"`` (default) evaluates these equations
directly in sparse matrix form and vectorises across batches in
:meth:`DistFlowOracle.feasible_mask`.  ``backend="solver"`` solves the
equivalent conic formulation via Pyomo, MOSEK Fusion, or Gurobi.

How the two published constructions use the loss bounds
-------------------------------------------------------
The two conservative inner-approximation methods in :mod:`OpenRC.region.conservative`
combine pairs of these variants:

*Bracketing* (``lindistflow`` + ``lindistflow-vdrop``).  The exact voltage
carries :math:`+(r^2+x^2) l` with :math:`l \\in [0, l^{max}]`.  Model-I takes the
low end of that interval and Model-II the high end, so the two **bracket** the
true voltage.  Their intersection is therefore contained in the feasible set of
the exact voltage-drop equation used in Method 1.

*Margining* (``tight-voltage`` + ``tight-flow``).  Published Method 2
uses two complementary conservative models.  Model-I drops balance losses,
evaluates the voltage term at ``l_max``, and tightens path thermal limits.
Model-II charges balance losses, drops the voltage term, and keeps nameplate
thermal limits.  Their direct intersection is the published inner
approximation; with ``tighten_scope="upstream"`` it bounds the zero-loss and
all-``l_max`` extremes rather than arbitrary mixed branch-loss vectors (use
:mod:`OpenRC.validation` to certify exact inclusion).  With ``tighten_scope="subtree"`` Model-I tightens each
branch by its subtree's losses and checks the lossless flow at ``v_min``,
which bounds every combination (:func:`OpenRC.region.conservative.method2_covers_losses`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from OpenRC.powerflow.base import (ORACLES, FeasibilityOracle, SolverUnavailableError,
                                  Verdict, report_solver_error)
from OpenRC.powerflow.equations import CandidateMixin, EquationProblem
from OpenRC.powerflow.modeling import (ConicForm, DirectModel, branch_flow_form,
                                       check_modeling, direct_model, synchronized_pyomo)
from OpenRC.powerflow.topology import Branch

# Termination conditions that mean "a feasible point exists".
_FEASIBLE_TC = {"optimal", "locallyOptimal", "globallyOptimal", "feasible"}
# Only an explicit infeasibility proof can reject a point definitively.
# Pyomo's noSolution means no feasible solution was found, not that none exists.
_INFEASIBLE_TC = {"infeasible"}

#: How the tightened model accumulates the loss margin used to shrink I_max.
TIGHTEN_SCOPES = ("upstream_legacy", "upstream", "subtree")

#: How a query is answered: evaluating the equations, or a conic solve.
BACKENDS = ("analytic", "solver")


@dataclass(frozen=True)
class DistFlowVariant:
    """The three switches that define one member of the family.

    Attributes
    ----------
    charge_loss_in_balance:
        Add ``r * l_max`` / ``x * l_max`` to each branch's power balance, so the
        computed flow is never below the true one.
    vdrop_sign:
        Coefficient of ``(r^2 + x^2) * l_max`` in ``v_child``: ``0`` drops the
        term, ``+1`` reproduces the exact equation at ``l = l_max`` (bracketing),
        ``-1`` reproduces the original implementation (``legacy-*`` variants
        only; the conservative builders never use it).
    tighten_thermal:
        Replace the nameplate ``l_max`` with the loss-corrected ``l_max_tau``.
    inner_approximation:
        Inclusion relationship to the exact branch-flow feasible set (``True``
        for inner, ``False`` for outer, ``None`` when valid only as part of a
        paired construction).
    """

    charge_loss_in_balance: bool
    vdrop_sign: int
    tighten_thermal: bool
    inner_approximation: Optional[bool] = None
    doc: str = ""


#: Paper-consistent DistFlow variants plus two ``legacy-*`` reproduction variants.
VARIANTS: Dict[str, DistFlowVariant] = {
    "model1": DistFlowVariant(
        charge_loss_in_balance=False, vdrop_sign=0, tighten_thermal=False,
        doc="Model-I: every loss term dropped.",
    ),
    "model2": DistFlowVariant(
        charge_loss_in_balance=False, vdrop_sign=+1, tighten_thermal=False,
        doc="Model-II: voltage drop evaluated at the largest permissible loss.",
    ),
    "tight-voltage": DistFlowVariant(
        charge_loss_in_balance=False, vdrop_sign=+1, tighten_thermal=True,
        doc="Method-2 Model-I: lmax voltage term and path-tightened thermal limits.",
    ),
    "tight-flow": DistFlowVariant(
        charge_loss_in_balance=True, vdrop_sign=0, tighten_thermal=False,
        doc="Method-2 Model-II: maximum losses charged to the balance.",
    ),
    # Legacy variants retained for exact baseline reproduction; ConservativeBuilder
    # uses the paper-consistent Method-2 models above by default.
    "legacy-tight-voltage": DistFlowVariant(
        charge_loss_in_balance=False, vdrop_sign=-1, tighten_thermal=False,
        doc="Legacy Method-2 Model-I variant with vdrop_sign=-1 and un-tightened thermal limits.",
    ),
    "legacy-tight-flow": DistFlowVariant(
        charge_loss_in_balance=True, vdrop_sign=0, tighten_thermal=True,
        doc="Legacy Method-2 Model-II variant with balance loss charge and tightened thermal limits.",
    ),
}

_VARIANT_ALIASES: Dict[str, str] = {
    "method2-1": "tight-voltage",
    "method2-2": "tight-flow",
    "method2-1-script": "legacy-tight-voltage",
    "method2-2-script": "legacy-tight-flow",
}


class DistFlowOracle(CandidateMixin, FeasibilityOracle):
    """LinDistFlow feasibility test: a closed-form evaluation, or a conic solve.

    With ``backend="analytic"`` (default) a query evaluates the model's
    equations directly (see the module docstring: the models have no free
    variable).  With ``backend="solver"`` the Pyomo model is assembled once in
    :meth:`_build`, the injections enter as mutable parameters, and each query
    is one conic solve.  Both give the same verdicts, up to the conic solver's
    tolerance at the boundary; the analytic one needs no solver licence.

    Parameters
    ----------
    case:
        The network.
    variant:
        One of :data:`VARIANTS` - see the module docstring for the table.
    backend:
        ``"analytic"`` (default) or ``"solver"``.  A subclass that changes the
        model in :meth:`_build` should use ``"solver"`` or override
        :meth:`_evaluate` to match.
    solver:
        Pyomo solver name for ``backend="solver"`` with ``modeling="pyomo"``.
        Needs conic support: MOSEK (default) or Gurobi.
    modeling:
        How ``backend="solver"`` states the model: ``"pyomo"`` (default, any
        Pyomo solver), or directly and persistently in ``"mosek"`` (Fusion) or
        ``"gurobi"`` (gurobipy) - see :mod:`OpenRC.powerflow.modeling`.
    tighten_scope:
        Only used when the variant tightens the thermal limit.  Which branches
        contribute to the loss margin that shrinks the limit of branch *b*:

        ``"upstream"``
            the whole path to the root, including the root branch.  This is the
            paper-consistent default.  The losses below *b*, which enter *b*'s
            flow, are not part of it.
        ``"upstream_legacy"``
            upstream path excluding the root branch incident to the slack bus
            (retained for legacy reproduction).
        ``"subtree"``
            *b* itself and everything downstream of it - the scope that matches
            where the neglected losses physically accumulate in the flow on *b*.
            The thermal check is then taken at the lowest permissible voltage,
            ``P^2 + Q^2 <= l_max_tau * v_min^2``, rather than at this model's
            own parent voltage, which bounds the true one from above (see
            :attr:`thermal_cap_at_v_min`).  With ``tight-flow`` this makes
            Method 2 an inner approximation (:mod:`OpenRC.region.conservative`).
    solver_options:
        Passed straight through to the solver, in the chosen layer's own naming.
    candidate_solver:
        Optional ``"newton"``, ``"gauss-seidel"`` or registered equation
        solver, used only before ``backend="solver"``.  A candidate is accepted
        only after every equality, bound and cone constraint passes an
        independent residual check.  Otherwise the optimiser is always called.
        The default is ``None``: the solver path remains the reference.  The
        analytic backend is already the exact and faster solution of this
        determined affine model, so it does not need a candidate solver.
    """

    def __init__(
        self,
        case,
        *,
        variant: str = "model1",
        backend: str = "analytic",
        solver: str = "mosek",
        modeling: str = "pyomo",
        tighten_scope: str = "upstream",
        solver_options: Optional[dict] = None,
        candidate_solver: Optional[str] = None,
        candidate_starts: int = 1,
        candidate_seed: Optional[int] = 0,
        candidate_options: Optional[dict] = None,
        candidate_tolerance: float = 1e-9,
        candidate_min_rate: float = 0.05,
        candidate_patience: int = 20,
        name: Optional[str] = None,
    ) -> None:
        variant = _VARIANT_ALIASES.get(variant, variant)
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown DistFlow variant {variant!r}; available: "
                f"{', '.join(sorted(VARIANTS))}"
            )
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if candidate_solver is not None and backend != "solver":
            raise ValueError(
                "candidate_solver is only used with backend='solver'; "
                "backend='analytic' already evaluates the equations exactly"
            )
        check_modeling(modeling)
        if tighten_scope not in TIGHTEN_SCOPES:
            raise ValueError(
                f"tighten_scope must be one of {TIGHTEN_SCOPES}, got {tighten_scope!r}"
            )
        if not isinstance(solver, str) or not solver.strip():
            raise ValueError("solver must be a non-empty string")
        if solver_options is not None and not isinstance(solver_options, dict):
            raise TypeError("solver_options must be a dict")
        super().__init__(case, name=name or f"distflow:{variant}")
        self.variant = variant
        self.spec = VARIANTS[variant]
        self.backend = backend
        self.modeling = modeling
        self.solver_name = solver
        self.tighten_scope = tighten_scope
        self.solver_options = dict(solver_options or {})
        self._init_candidate(candidate_solver, starts=candidate_starts, seed=candidate_seed,
                             options=candidate_options, tolerance=candidate_tolerance,
                             min_rate=candidate_min_rate, patience=candidate_patience)
        self.inner_approximation = self.spec.inner_approximation
        # Single DistFlow variants do not certify reference feasibility unless
        # inner_approximation is explicitly set.
        self.supports_adjudication = self.inner_approximation is not None
        # Closed-form constraint margin is available when _evaluate is not overridden.
        self.supports_margin = (backend == "analytic"
                                and type(self)._evaluate is DistFlowOracle._evaluate)

        # Built lazily by _build(); Pyomo objects carry no type information.
        self._model: Any = None
        self._solver: Any = None
        self._p_param: Any = None
        self._q_param: Any = None
        self._s_base_va = 1.0e6
        self._linear: Optional[_LinearDistFlow] = None
        self._direct: Optional[DirectModel] = None

    # --------------------------------------------------------------- geometry
    @property
    def topology(self):
        return self.case.topology

    def _tightened_margins_a(self) -> Dict[int, float]:
        """Signed current margin ``i_max_a - sqrt(d_l)`` per branch (in A)."""
        topo = self.topology
        if not self.spec.tighten_thermal:
            return {br.index: br.i_max_a for br in topo.branches}

        v_min = (self.case.v_nom_kv * 1e3 / math.sqrt(3.0)) * self.case.vm_min_pu
        margins: Dict[int, float] = {}
        for br in topo.branches:
            if self.tighten_scope == "subtree":
                contributors = _subtree_branches(topo, br)
            else:
                path = topo.path_to_root(br.child)      # [br, parent-of-br, ..., root]
                # "upstream_legacy" excludes the root branch incident to the slack bus.
                contributors = path[:-1] if self.tighten_scope == "upstream_legacy" else path

            d_p = sum(c.r_ohm * c.l_max for c in contributors)
            d_q = sum(c.x_ohm * c.l_max for c in contributors)
            # Loss power -> equivalent current magnitude at the worst-case voltage.
            d_l = (d_p ** 2 + d_q ** 2) / (v_min ** 2)
            margins[br.index] = br.i_max_a - math.sqrt(d_l)
        return margins

    def _effective_l_max(self) -> Dict[int, float]:
        """Squared thermal limit per branch, after tightening if requested.

        Returns a mapping ``branch.index -> l_max`` in A^2.
        """
        if not self.spec.tighten_thermal:
            return {br.index: br.l_max for br in self.topology.branches}
        # Clamp negative current margins to 0.0 before squaring so over-tightened
        # branches receive l_max = 0; has_clamped_tightening() reports whether
        # accumulated losses exceed nameplate rating on any branch.
        return {
            idx: float(max(margin, 0.0)) ** 2
            for idx, margin in self._tightened_margins_a().items()
        }

    def has_clamped_tightening(self) -> bool:
        """Whether thermal tightening exceeded any branch's nameplate rating."""
        if not self.spec.tighten_thermal:
            return False
        return any(
            not math.isfinite(margin) or margin < 0.0
            for margin in self._tightened_margins_a().values()
        )

    @property
    def thermal_cap_at_v_min(self) -> bool:
        """Whether the thermal check is ``P^2 + Q^2 <= l_max_tau * v_min^2``.

        True for a tightened variant with ``tighten_scope="subtree"``.  The
        exact flow on a branch lies within ``|S^LL| + |D|`` of the lossless one
        (``D``: the subtree's largest losses, the margin ``l_max_tau`` takes
        off), and the true parent voltage is at least ``v_min``; checking the
        lossless flow against the tightened limit at ``v_min`` therefore keeps
        the true current within the nameplate rating for every combination of
        branch losses up to ``l_max``.  This model's own parent voltage is an
        upper bound on the true one, so a check there would not.
        """
        return bool(self.spec.tighten_thermal and self.tighten_scope == "subtree")

    def binding_tightening(self) -> Dict[int, float]:
        """Ratio of tightened to nameplate ``l_max`` per branch, for diagnosis.

        A ratio of ``0.0`` indicates that the loss margin exceeds the nameplate
        thermal limit on that branch, making zero-flow the only permissible state.
        """
        nameplate = {br.index: br.l_max for br in self.topology.branches}
        tightened = self._effective_l_max()
        return {k: (tightened[k] / nameplate[k] if nameplate[k] > 0 else 0.0)
                for k in nameplate}

    # ------------------------------------------------------------------ build
    @property
    def needs_ratings(self) -> bool:
        """Whether this variant uses branch ratings to bound neglected losses.

        ``False`` for ``model1`` (which uses ratings only for thermal limits,
        omitted on unrated branches); ``True`` for variants that bound neglected
        losses via ``l_max`` in voltage-drop, power-balance, or tightened
        thermal terms.
        """
        spec = self.spec
        return bool(spec.vdrop_sign or spec.charge_loss_in_balance or spec.tighten_thermal)

    def _check_ratings(self) -> None:
        """Refuse a variant that bounds losses by ratings the network lacks."""
        from OpenRC.powerflow.topology import UnratedBranchError

        case = self.case
        topo = self.topology
        if self.needs_ratings and topo.unrated_branches:
            names = [br.index for br in topo.unrated_branches[:6]]
            raise UnratedBranchError(
                f"the {self.variant!r} model bounds the neglected losses by each "
                f"branch's thermal rating, but {len(topo.unrated_branches)} branch(es) "
                f"of case {case.name!r} (lines {names}...) carry a placeholder rather "
                "than a rating - typically MATPOWER's 9900 MVA or pandapower's "
                "99999 kA. Without a real limit there is no loss bound. Supply one "
                "with case_from_pandapower(..., default_rating_ka=...), or use "
                "'lindistflow', 'socp' or 'acpf', which treat an unrated branch as "
                "simply unconstrained."
            )

    def _build(self):
        """Assemble the Pyomo model once, with injections as mutable parameters."""
        import pyomo.environ as pyo

        self._check_ratings()
        case = self.case
        topo = self.topology
        spec = self.spec
        l_max = self._effective_l_max()

        # Stated in per-unit on a 1 MVA three-phase base (matching _conic_form,
        # SOCP and exact) so high-voltage networks do not hand 1e10-1e17 SI
        # quadratic coefficients to conic solvers.
        v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        s_base_phase = self._s_base_va / 3.0
        z_base = v_base ** 2 / s_base_phase
        i_base_sq = (s_base_phase / v_base) ** 2
        v_slack_sq = case.slack_vm_pu ** 2
        v_lo_sq = case.vm_min_pu ** 2
        v_hi_sq = case.vm_max_pu ** 2

        m = pyo.ConcreteModel(name=f"{case.name}:{self.variant}")
        m.BUS = pyo.Set(initialize=topo.buses, ordered=True)
        m.BR = pyo.Set(initialize=[br.index for br in topo.branches], ordered=True)

        # Per-unit injection on s_base_va; updated in place per query.
        m.p_inj = pyo.Param(m.BUS, mutable=True, initialize=0.0, within=pyo.Reals)
        m.q_inj = pyo.Param(m.BUS, mutable=True, initialize=0.0, within=pyo.Reals)

        m.v2 = pyo.Var(m.BUS, bounds=(v_lo_sq, v_hi_sq))
        m.P = pyo.Var(m.BR)
        m.Q = pyo.Var(m.BR)

        def v_of(bus):
            """Squared voltage of a bus - constant at the slack, a variable elsewhere."""
            return v_slack_sq if bus == topo.slack else m.v2[bus]

        m.balance_p = pyo.ConstraintList()
        m.balance_q = pyo.ConstraintList()
        m.v_drop = pyo.ConstraintList()
        m.thermal = pyo.ConstraintList()

        for br in topo.branches:
            child = br.child
            downstream = topo.children_of[child]
            r_pu = br.r_ohm / z_base
            x_pu = br.x_ohm / z_base
            # Balance loss bounds use nameplate l_max even when thermal limits are tightened.
            nameplate_l_pu = br.l_max / i_base_sq

            # (1) active power balance at the child bus
            rhs_p = sum(m.P[d.index] for d in downstream) + m.p_inj[child]
            if spec.charge_loss_in_balance:
                rhs_p = rhs_p + r_pu * nameplate_l_pu
            m.balance_p.add(m.P[br.index] == rhs_p)

            # (2) reactive power balance at the child bus
            rhs_q = sum(m.Q[d.index] for d in downstream) + m.q_inj[child]
            if spec.charge_loss_in_balance:
                rhs_q = rhs_q + x_pu * nameplate_l_pu
            m.balance_q.add(m.Q[br.index] == rhs_q)

            # (3) voltage drop across the branch.  Written as
            #     v_parent - v_child == 2 (rP + xQ) - sign * (r^2+x^2) l_max
            # so that sign = +1 reproduces the exact equation at l = l_max.
            drop = 2.0 * (r_pu * m.P[br.index] + x_pu * m.Q[br.index])
            if spec.vdrop_sign:
                drop = drop - spec.vdrop_sign * (r_pu ** 2 + x_pu ** 2) * nameplate_l_pu
            m.v_drop.add(v_of(br.parent) - m.v2[child] == drop)

            # (4) thermal limit in convex second-order-cone form (evaluated at
            # v_min for subtree scope, otherwise at v_parent).  Omit thermal
            # constraints on unrated branches to avoid large placeholder coefficients.
            if br.rated:
                eff_l_pu = l_max[br.index] / i_base_sq
                m.thermal.add(
                    m.P[br.index] ** 2 + m.Q[br.index] ** 2
                    <= eff_l_pu * (v_lo_sq if self.thermal_cap_at_v_min
                                   else v_of(br.parent))
                )

        # A linear objective keeps the problem a plain conic program.  Its value
        # is irrelevant - only the termination status is read.
        m.obj = pyo.Objective(expr=sum(m.v2[b] for b in m.BUS), sense=pyo.minimize)

        self._model = m
        self._p_param = m.p_inj
        self._q_param = m.q_inj
        self._solver = self._make_solver()
        return m

    def _make_solver(self):
        from pyomo.opt import SolverFactory

        solver = SolverFactory(self.solver_name)
        if not solver.available(exception_flag=False):
            raise SolverUnavailableError(
                f"solver {self.solver_name!r} is not available. The DistFlow oracles need "
                "a conic solver; install MOSEK or Gurobi, or pass solver=... explicitly."
            )
        for key, value in self.solver_options.items():
            solver.options[key] = value
        return solver

    # ------------------------------------------------------------------ query
    def _linear_model(self) -> "_LinearDistFlow":
        """The closed-form model (built once, on first use)."""
        if self._linear is None:
            self._check_ratings()
            self._linear = _LinearDistFlow(self)
        return self._linear

    def _evaluate(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> np.ndarray:
        """Closed-form verdicts for ``(n, n_bus)`` injection arrays (MW, Mvar)."""
        return self._linear_model().feasible(p_mw, q_mvar)

    def _conic_form(self) -> ConicForm:
        """The model of :meth:`_build`, for a direct modelling layer."""
        self._check_ratings()
        case, spec = self.case, self.spec
        topo = self.topology
        v_nom = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        effective = self._effective_l_max()
        return branch_flow_form(
            topo, scale=1e6 / 3.0, v_slack=(v_nom * case.slack_vm_pu) ** 2,
            v_bounds=((v_nom * case.vm_min_pu) ** 2, (v_nom * case.vm_max_pu) ** 2),
            losses="none",
            thermal={br.index: effective[br.index] for br in topo.branches if br.rated},
            thermal_voltage=((v_nom * case.vm_min_pu) ** 2 if self.thermal_cap_at_v_min
                             else None),
            balance_const=({br.index: (br.r_ohm * br.l_max, br.x_ohm * br.l_max)
                            for br in topo.branches} if spec.charge_loss_in_balance
                           else None),
            drop_const=({br.index: -spec.vdrop_sign * (br.r_ohm ** 2 + br.x_ohm ** 2)
                         * br.l_max for br in topo.branches} if spec.vdrop_sign else None),
            base=(1e6 / 3.0, v_nom), objective="voltage")

    def conic_form(self) -> ConicForm:
        """Public solver-neutral form of this DistFlow variant.

        The returned object can create a SciPy-compatible equation problem with
        :meth:`ConicForm.scipy_problem`, or feed a custom modelling layer.
        """
        return self._conic_form().with_case_charging(self.case)

    def equation_problem(
        self, p_mw: np.ndarray, q_mvar: np.ndarray, *,
        x0: Optional[np.ndarray] = None, tolerance: float = 1e-9,
    ) -> EquationProblem:
        """Return this query as a public SciPy-compatible equation problem."""
        return self._conic_form().scipy_problem(
            p_mw, self.case.branch_flow_q(q_mvar), x0=x0, tolerance=tolerance
        )

    def _optimizer_verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        """Reference solver verdict after an optional candidate has failed."""
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
        """Serialize model preparation and solving, not analytic/direct queries."""
        if self._model is None:
            self._build()

        topo = self.topology
        scale = 1e6 / self._s_base_va
        for bus in topo.buses:
            self._p_param[bus].value = float(p_mw[bus]) * scale
            self._q_param[bus].value = float(q_mvar[bus]) * scale
        return _solve_verdict(self._solver, self._model, name=self.name)

    def _verdict_margin(self, p_mw: np.ndarray, q_mvar: np.ndarray):
        if not self.supports_margin:
            return self._verdict(p_mw, q_mvar), math.nan
        q_mvar = self.case.branch_flow_q(q_mvar)
        linear = self._linear_model()
        if (getattr(linear.feasible, "__func__", None) is _LinearDistFlow.feasible
                and getattr(linear.margin, "__func__", None) is _LinearDistFlow.margin):
            feasible, margin = linear.verdict_margin_1d(p_mw, q_mvar)
        else:
            feasible = bool(linear.feasible(p_mw[None, :], q_mvar[None, :])[0])
            margin = float(linear.margin(p_mw[None, :], q_mvar[None, :])[0])
        # The verdict is the reference; the margin only takes its sign on a tie.
        if feasible and margin < 0.0:
            margin = 0.0
        elif not feasible and margin >= 0.0:
            margin = -1e-300
        return (Verdict.FEASIBLE if feasible else Verdict.INFEASIBLE), margin

    def _verdict(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> Verdict:
        q_mvar = self.case.branch_flow_q(q_mvar)
        if self.backend == "analytic":
            if getattr(self._evaluate, "__func__", None) is DistFlowOracle._evaluate:
                feasible = self._linear_model().feasible_1d(p_mw, q_mvar)
            else:
                feasible = bool(self._evaluate(p_mw[None, :], q_mvar[None, :])[0])
            return Verdict.FEASIBLE if feasible else Verdict.INFEASIBLE
        candidate = self._candidate_verdict(p_mw, q_mvar)
        return candidate if candidate is not None else self._optimizer_verdict(p_mw, q_mvar)

    def _check(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE

    def feasible_mask(self, axes, points: np.ndarray) -> np.ndarray:
        """Bounded-memory batch verdicts, respecting custom single-point hooks.

        Evaluates rows in vectorised blocks when ``backend == "analytic"`` and
        falls back to row-by-row evaluation when ``backend != "analytic"`` or
        any single-point verdict method is overridden.
        """
        hooks = (("feasible_at", FeasibilityOracle.feasible_at),
                 ("is_feasible", FeasibilityOracle.is_feasible),
                 ("verdict", FeasibilityOracle.verdict),
                 ("_verdict", DistFlowOracle._verdict))
        if (self.backend != "analytic"
                or any(getattr(getattr(self, name), "__func__", None) is not default
                       for name, default in hooks)):
            return super().feasible_mask(axes, points)
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
        points = np.atleast_2d(raw)
        if points.ndim != 2 or points.shape[1] != len(axes):
            raise ValueError(f"points have dimension {points.shape[1]}, expected {len(axes)}")
        if not np.all(np.isfinite(points)):
            raise ValueError("region coordinates must contain only finite values")
        base_p, base_q = self.case.injection_vector(axes)
        base_q = self.case.branch_flow_q(base_q)
        # Cap block size to ~16 MiB across injections, branch flows, voltages, and slacks.
        rows = max(1, min(4096, 16 * 2 ** 20 // (12 * 8 * self.case.n_bus)))
        mask = np.empty(len(points), dtype=bool)
        start = time.perf_counter()
        with self._lock:
            for first in range(0, len(points), rows):
                last = min(first + rows, len(points))
                p = np.repeat(base_p[None, :], last - first, axis=0)
                q = np.repeat(base_q[None, :], last - first, axis=0)
                for k, axis in enumerate(axes):
                    (p if axis.quantity == "p" else q)[:, axis.bus] += points[first:last, k]
                mask[first:last] = self._evaluate(p, q)
            self.stats.record_many(mask, time.perf_counter() - start)
        return mask

    def describe(self) -> dict:
        info = super().describe()
        info.update(
            variant=self.variant,
            backend=self.backend,
            modeling=self.modeling if self.backend == "solver" else None,
            solver=(self.solver_name if self.backend == "solver" and self.modeling == "pyomo"
                    else None),
            tighten_scope=self.tighten_scope if self.spec.tighten_thermal else None,
            **self._candidate_info(),
        )
        return info


class _LinearDistFlow:
    """Sparse-matrix representation of the affine DistFlow equations.

    Branch flows follow from ``T`` (``T[b, j] = 1`` when bus ``j`` lies in the
    subtree of branch ``b``) and bus voltages from ``T^T`` applied to branch
    voltage drops, giving ``nnz(T)`` equal to the sum of bus depths
    (``O(n_bus)`` on bounded-depth trees).

    Constraints are checked with relative tolerance ``TOL = 1e-9``.
    """

    TOL = 1e-9

    def __init__(self, oracle: "DistFlowOracle") -> None:
        import scipy.sparse as sp

        case, topo, spec = oracle.case, oracle.topology, oracle.spec
        branches = list(topo.branches)
        row = {br.index: k for k, br in enumerate(branches)}
        entries = [(row[br.index], bus) for bus in topo.buses
                   for br in topo.path_to_root(bus)]
        rows, cols = np.array(entries, dtype=int).reshape(-1, 2).T
        self.T: Any = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                    shape=(len(branches), case.n_bus))
        self.Tt = self.T.T.tocsr()
        self.r = np.array([br.r_ohm for br in branches])
        self.x = np.array([br.x_ohm for br in branches])
        nameplate = np.array([br.l_max if br.rated else 0.0 for br in branches])
        effective = oracle._effective_l_max()

        # Constants the variant adds to the balances and to the voltage drops.
        # A branch's loss charge enters the balance of every branch above it,
        # exactly like an injection at its child bus: so place it there.
        children = np.array([br.child for br in branches], dtype=int)

        def at_child(per_branch: np.ndarray) -> np.ndarray:
            per_bus = np.zeros(case.n_bus)
            per_bus[children] = per_branch
            return per_bus

        zero = np.zeros(len(branches))
        charge = spec.charge_loss_in_balance
        self.c_p = self.T @ at_child(self.r * nameplate) if charge else zero
        self.c_q = self.T @ at_child(self.x * nameplate) if charge else zero
        self.c_v = spec.vdrop_sign * (self.Tt @ ((self.r ** 2 + self.x ** 2) * nameplate))

        v_nom = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        self.v_slack = (v_nom * case.slack_vm_pu) ** 2
        self.v_lo = (v_nom * case.vm_min_pu) ** 2
        self.v_hi = (v_nom * case.vm_max_pu) ** 2
        self.buses = np.array(topo.buses, dtype=int)
        self.parent = np.array([br.parent for br in branches], dtype=int)
        self.rated = np.array([br.rated for br in branches], dtype=bool)
        self.l_max = np.array([effective[br.index] for br in branches])
        # Squared voltage of every thermal cap: v_min for the subtree scope,
        # otherwise the parent bus voltage of the state (None).
        self.cap_v: Optional[float] = self.v_lo if oracle.thermal_cap_at_v_min else None

    def _cap_voltage(self, v: np.ndarray) -> Any:
        """The voltage each branch's thermal cap is taken at (per row for 2D ``v``)."""
        return v[self.parent] if self.cap_v is None else self.cap_v

    def _state_1d(self, p_mw: np.ndarray, q_mvar: np.ndarray):
        """Single-point flows ``(P, Q)`` and squared voltages ``v`` as 1D vectors."""
        p_raw = np.asarray(p_mw)
        q_raw = np.asarray(q_mvar)
        if p_raw.dtype == bool or q_raw.dtype == bool:
            raise ValueError("injections must be numeric, not boolean")
        scale = 1e6 / 3.0                                    # MW three-phase -> W per phase
        P = self.T @ (p_raw * scale) + self.c_p
        Q = self.T @ (q_raw * scale) + self.c_q
        drop = 2.0 * (self.r * P + self.x * Q)
        v = self.v_slack - self.Tt @ drop + self.c_v
        return P, Q, v

    def feasible_1d(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> bool:
        """Single-point verdict for 1D ``(n_bus,)`` injections in MW / Mvar."""
        P, Q, v = self._state_1d(p_mw, q_mvar)
        v_bus = v[self.buses]
        if not np.all((v_bus >= self.v_lo * (1.0 - self.TOL))
                      & (v_bus <= self.v_hi * (1.0 + self.TOL))):
            return False
        cap = self.l_max * self._cap_voltage(v)
        thermal = (P ** 2 + Q ** 2 <= cap * (1.0 + self.TOL) + self.TOL) | ~self.rated
        return bool(np.all(thermal))

    def verdict_margin_1d(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> tuple[bool, float]:
        """Single-point ``(feasible, margin)`` from one state evaluation."""
        P, Q, v = self._state_1d(p_mw, q_mvar)
        v_bus = v[self.buses]
        voltage = bool(np.all((v_bus >= self.v_lo * (1.0 - self.TOL))
                              & (v_bus <= self.v_hi * (1.0 + self.TOL))))
        cap_f = self.l_max * self._cap_voltage(v)
        thermal_f = (P ** 2 + Q ** 2 <= cap_f * (1.0 + self.TOL) + self.TOL) | ~self.rated
        feasible = voltage and bool(np.all(thermal_f))

        low = v_bus / (self.v_lo * (1.0 - self.TOL)) - 1.0
        high = 1.0 - v_bus / (self.v_hi * (1.0 + self.TOL))
        cap_m = cap_f * (1.0 + self.TOL) + self.TOL
        s_sq_1d = P ** 2 + Q ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_th_1d = np.where(cap_m > 0.0, 1.0 - s_sq_1d / cap_m, (cap_m - s_sq_1d) / self.TOL)
        thermal_m = np.where(self.rated, raw_th_1d, np.inf)
        margin = float(min(float(np.minimum(low, high).min(initial=np.inf)),
                           float(thermal_m.min(initial=np.inf))))
        return feasible, margin

    def _state(self, p_mw: np.ndarray, q_mvar: np.ndarray):
        """Flows ``(P, Q)`` per branch and squared voltages ``v`` per bus, per row."""
        p_raw = np.asarray(p_mw)
        q_raw = np.asarray(q_mvar)
        if p_raw.dtype == bool or q_raw.dtype == bool:
            raise ValueError("injections must be numeric, not boolean")
        scale = 1e6 / 3.0                                    # MW three-phase -> W per phase
        p = np.atleast_2d(p_raw).T * scale                   # (n_bus, n)
        q = np.atleast_2d(q_raw).T * scale
        P = self.T @ p + self.c_p[:, None]                   # (n_branch, n)
        Q = self.T @ q + self.c_q[:, None]
        drop = 2.0 * (self.r[:, None] * P + self.x[:, None] * Q)
        v = self.v_slack - self.Tt @ drop + self.c_v[:, None]   # (n_bus, n); slack row = v_slack
        return P, Q, v

    def feasible(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> np.ndarray:
        """Verdict per row of ``(n, n_bus)`` injections in MW / Mvar."""
        P, Q, v = self._state(p_mw, q_mvar)
        v_bus = v[self.buses]
        voltage = np.all((v_bus >= self.v_lo * (1.0 - self.TOL))
                         & (v_bus <= self.v_hi * (1.0 + self.TOL)), axis=0)
        # The absolute term only matters for a limit tightened to zero.
        cap = self.l_max[:, None] * self._cap_voltage(v)
        thermal = (P ** 2 + Q ** 2 <= cap * (1.0 + self.TOL) + self.TOL) | ~self.rated[:, None]
        return voltage & np.all(thermal, axis=0)

    def margin(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> np.ndarray:
        """Smallest relative slack per row, against the same thresholds as :meth:`feasible`."""
        P, Q, v = self._state(p_mw, q_mvar)
        v_bus = v[self.buses]
        low = v_bus / (self.v_lo * (1.0 - self.TOL)) - 1.0
        high = 1.0 - v_bus / (self.v_hi * (1.0 + self.TOL))
        cap = self.l_max[:, None] * self._cap_voltage(v) * (1.0 + self.TOL) + self.TOL
        s_sq = P ** 2 + Q ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_th = np.where(cap > 0.0, 1.0 - s_sq / cap, (cap - s_sq) / self.TOL)
        thermal = np.where(self.rated[:, None], raw_th, np.inf)
        return np.minimum(np.minimum(low, high).min(axis=0, initial=np.inf),
                          thermal.min(axis=0, initial=np.inf))


def _subtree_branches(topo, root: Branch):
    """``root`` plus every branch downstream of it (``tighten_scope="subtree"``)."""
    out = [root]
    stack = [root.child]
    while stack:
        bus = stack.pop()
        for child_branch in topo.children_of[bus]:
            out.append(child_branch)
            stack.append(child_branch.child)
    return out


@synchronized_pyomo
def _solve_verdict(solver, model, *, name: str = "pyomo") -> Verdict:
    """Solve ``model`` with ``solver`` and map termination conditions to :class:`Verdict`.

    Inspects ``results.solver.termination_condition`` to distinguish proven
    infeasibility (which often accompanies ``SolverStatus.ok``) from feasible
    solutions and unresolved solver terminations.
    """
    from pyomo.opt import SolverStatus

    try:
        results = solver.solve(model, load_solutions=False)
    except Exception as exc:
        report_solver_error(name, exc)
        return Verdict.INCONCLUSIVE

    tc = str(results.solver.termination_condition)
    if tc in _INFEASIBLE_TC:
        return Verdict.INFEASIBLE
    if tc in _FEASIBLE_TC:
        return Verdict.FEASIBLE
    if results.solver.status == SolverStatus.ok and tc == "optimal":
        return Verdict.FEASIBLE
    return Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------
# Registry entries - one per variant
# --------------------------------------------------------------------------
def _register(oracle_name: str, variant: str, doc: str) -> None:
    def _factory(case, **kwargs):
        kwargs.setdefault("variant", variant)
        return DistFlowOracle(case, **kwargs)

    _factory.__doc__ = doc
    ORACLES.add(oracle_name, _factory, doc=doc)


_register(
    "lindistflow", "model1",
    "Model-I: LinDistFlow with all loss terms dropped (fast, the default).",
)
_register(
    "lindistflow-vdrop", "model2",
    "Model-II: voltage drop at the largest permissible loss; brackets Model-I.",
)
_register(
    "tight-voltage", "tight-voltage",
    "Method-2 Model-I: lmax voltage term and path-tightened thermal limits.",
)
_register(
    "tight-flow", "tight-flow",
    "Method-2 Model-II: maximum losses charged to the balance; nameplate limits.",
)
_register(
    "legacy-tight-voltage", "legacy-tight-voltage",
    "Legacy Method-2 Model-I (vdrop_sign=-1, nameplate thermal limits).",
)
_register(
    "legacy-tight-flow", "legacy-tight-flow",
    "Legacy Method-2 Model-II (balance loss charge, tightened thermal limits).",
)
