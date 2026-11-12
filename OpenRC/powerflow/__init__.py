"""Power-flow feasibility oracles and radial topology extraction.

Every :class:`FeasibilityOracle` evaluates whether a full per-bus active and
reactive injection vector satisfies network operating limits, decoupling region
construction and validation algorithms from the underlying power-flow model.

Registered oracles (see :data:`ORACLES`)::

    lindistflow        Model-I  - LinDistFlow, every loss term dropped
    lindistflow-vdrop  Model-II - voltage drop at the largest permissible loss
    tight-voltage      S1 - voltages bounded from above, thermal limits tightened
    tight-flow         S2 - losses charged to the balance, voltages bounded from below
    socp               exact branch flow, second-order-cone relaxation
    exact              exact branch flow solved globally (Gurobi) - the adjudicator
    acpf               Newton-Raphson AC power flow with limit checking (ground truth)
    dcpf               DC power flow, thermal limits only (fast screening)
    dcopf              DC optimal power flow - dispatchable, no radiality needed
    acopf              AC optimal power flow - dispatchable, no radiality needed

The four LinDistFlow variants form **two pairs**, and each pair is one of the
inner-approximation methods in :mod:`OpenRC.region.conservative`:
``lindistflow`` with ``lindistflow-vdrop`` brackets the exact voltage (Method 1,
which then needs a shift), while ``tight-voltage`` with ``tight-flow`` margins it
from both sides (Method 2, intersected directly).
"""

from OpenRC.powerflow.base import (
    ORACLES,
    FeasibilityOracle,
    SolverUnavailableError,
    Verdict,
    get_oracle,
    list_oracles,
)
from OpenRC.powerflow.equations import (
    EQUATION_SOLVERS,
    STRUCTURED_SOLVERS,
    ConstraintResiduals,
    EquationProblem,
    EquationResult,
    EquationSolverStats,
    PreparedEquationSolver,
    list_equation_solvers,
    list_structured_solvers,
    solve_equations,
)
from OpenRC.powerflow.modeling import ConicForm
from OpenRC.powerflow.topology import (
    AmbiguousSlackError,
    Branch,
    MeshedNetworkError,
    RadialityReport,
    RadialTopology,
    UnratedBranchError,
    infer_slack_bus,
    radiality_report,
    unrated_lines,
)

# Import concrete oracle modules to populate ORACLES.
from OpenRC.powerflow import acpf, distflow, exact, opf, socp, sweep  # noqa: F401  (side-effect import)
from OpenRC.powerflow.sweep import RadialSweep, SweepCertificate

__all__ = [
    "AmbiguousSlackError",
    "Branch",
    "ConicForm",
    "ConstraintResiduals",
    "EQUATION_SOLVERS",
    "STRUCTURED_SOLVERS",
    "EquationProblem",
    "EquationResult",
    "EquationSolverStats",
    "MeshedNetworkError",
    "RadialityReport",
    "FeasibilityOracle",
    "ORACLES",
    "RadialTopology",
    "RadialSweep",
    "SolverUnavailableError",
    "SweepCertificate",
    "PreparedEquationSolver",
    "UnratedBranchError",
    "Verdict",
    "infer_slack_bus",
    "radiality_report",
    "unrated_lines",
    "get_oracle",
    "list_oracles",
    "list_equation_solvers",
    "list_structured_solvers",
    "solve_equations",
]
