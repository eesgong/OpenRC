"""OpenRC - security regions for power networks.

A *security region* (also hosting capacity region, feasible operation region,
dynamic operating envelope) is the joint feasible set of power injections at a
subset of points of connection.  Representing that set as a convex polytope
reduces grid-constrained decision problems over those points to linear programs.

Subpackages:

==============================  ===========================================
:mod:`OpenRC.cases`            networks, and the axes a region is built over
:mod:`OpenRC.powerflow`        feasibility oracles - linearised, conic, exact, AC
:mod:`OpenRC.geometry`         polytopes, ray search, set operations
:mod:`OpenRC.region`           builders, seeding strategies, facet policies
:mod:`OpenRC.compression`      vertex pruning, and when to prune while growing
:mod:`OpenRC.validation`       loss/error rate, adjudication, inner-region proof
:mod:`OpenRC.applications`     decision problems posed over a region
==============================  ===========================================

Quick start
-----------
::

    import OpenRC as orc

    case   = orc.get_case("9bus")
    axes   = case.resolve_axes("247")           # POC 2,4,7 of the papers
    oracle = orc.get_oracle("lindistflow", case)

    region = orc.get_builder("expansive").build(case, axes, oracle,
                                                max_vertices=200).polytope
    small  = orc.get_compressor("local-search").compress(region,
                                                         target_vertices=15).polytope
    print(orc.evaluate_region(small, case, axes).summary())

Each component family (networks, oracles, builders, facet-selection policies,
compressors, samplers, and applications) uses a :class:`~OpenRC.registry.Registry`
decorator and is listed by ``python scripts/demo.py list``.

Conventions
-----------
Powers are **MW / Mvar**, consumption positive.  A region coordinate is the
*additional* injection a point of connection may take, so negative coordinates
mean export.
"""

from OpenRC.applications import (
    APPLICATIONS,
    Application,
    BatchResult,
    SolveResult,
    get_application,
    list_applications,
)
from OpenRC.cases import (
    CASES,
    PP_PREFIX,
    GridCase,
    case_from_pandapower,
    classify_pandapower_networks,
    list_feeders,
    list_pandapower_networks,
    RegionAxes,
    RegionAxis,
    build_case,
    get_case,
    list_cases,
)
from OpenRC.compression import (
    COMPRESSORS,
    SCHEDULES,
    CompressionResult,
    CompressionSchedule,
    Compressor,
    get_compressor,
    get_schedule,
    list_compressors,
    list_schedules,
)
from OpenRC.geometry import Polytope, intersect, ray_boundary, shutdown_geometry_workers
from OpenRC.powerflow import (
    EQUATION_SOLVERS,
    STRUCTURED_SOLVERS,
    ORACLES,
    AmbiguousSlackError,
    ConicForm,
    ConstraintResiduals,
    EquationProblem,
    EquationResult,
    EquationSolverStats,
    FeasibilityOracle,
    MeshedNetworkError,
    RadialityReport,
    RadialTopology,
    SolverUnavailableError,
    Verdict,
    get_oracle,
    infer_slack_bus,
    list_equation_solvers,
    list_structured_solvers,
    list_oracles,
    PreparedEquationSolver,
    RadialSweep,
    radiality_report,
    solve_equations,
    SweepCertificate,
)
from OpenRC.region import (
    BUILDERS,
    FACET_POLICIES,
    FACET_SCREENS,
    RAY_STRATEGIES,
    REFINERS,
    SEEDS,
    RegionBuilder,
    RegionResult,
    get_builder,
    get_facet_policy,
    get_ray_strategy,
    get_refiner,
    get_seed,
    list_builders,
    list_facet_policies,
    list_ray_strategies,
    list_refiners,
    list_seeds,
)
from OpenRC.benchmark import BENCHMARK_METHODS, list_benchmark_methods
from OpenRC.powerflow.modeling import MODELING_LAYERS
from OpenRC.region.facet_policy import list_facet_screens
from OpenRC.utils import configure_logging
from OpenRC.validation import (
    SAMPLERS,
    Adjudicator,
    InnerCertificate,
    RegionQuality,
    certify_inner_region,
    evaluate_region,
    get_sampler,
    list_samplers,
    prove_existence,
    shrink_to_certified,
    verify_inner_region,
)

__version__ = "0.1.1"

__all__ = [
    "APPLICATIONS", "Application", "BatchResult", "SolveResult",
    "get_application", "list_applications",
    "CASES", "GridCase", "RegionAxes", "RegionAxis", "build_case", "get_case", "list_cases",
    "PP_PREFIX", "case_from_pandapower", "classify_pandapower_networks", "list_feeders",
    "list_pandapower_networks",
    "COMPRESSORS", "CompressionResult", "Compressor", "get_compressor", "list_compressors",
    "SCHEDULES", "CompressionSchedule", "get_schedule", "list_schedules",
    "Polytope", "intersect", "ray_boundary",
    "shutdown_geometry_workers",
    "ORACLES", "EQUATION_SOLVERS", "STRUCTURED_SOLVERS", "FeasibilityOracle", "RadialTopology", "Verdict",
    "ConicForm", "ConstraintResiduals", "EquationProblem", "EquationResult",
    "EquationSolverStats",
    "PreparedEquationSolver", "solve_equations", "list_equation_solvers",
    "list_structured_solvers", "RadialSweep", "SweepCertificate",
    "AmbiguousSlackError", "MeshedNetworkError", "RadialityReport", "SolverUnavailableError",
    "get_oracle", "list_oracles", "infer_slack_bus", "radiality_report",
    "BUILDERS", "FACET_POLICIES", "FACET_SCREENS", "RAY_STRATEGIES", "REFINERS", "SEEDS",
    "RegionBuilder", "RegionResult",
    "get_builder", "get_facet_policy", "get_ray_strategy", "get_refiner", "get_seed",
    "list_builders", "list_facet_policies", "list_facet_screens", "list_ray_strategies",
    "list_refiners",
    "list_seeds",
    "SAMPLERS", "Adjudicator", "RegionQuality", "evaluate_region",
    "InnerCertificate", "certify_inner_region", "prove_existence", "shrink_to_certified",
    "verify_inner_region",
    "get_sampler", "list_samplers",
    "BENCHMARK_METHODS", "list_benchmark_methods", "MODELING_LAYERS",
    "configure_logging", "registries", "__version__",
]


def registries() -> dict:
    """Return a mapping from registry category to ``[(name, doc), ...]`` entries.

    Used by ``scripts/demo.py list``.
    """
    return {
        "cases": list_cases(),
        "oracles": list_oracles(),
        "equation solvers": list_equation_solvers(),
        "structured solvers": list_structured_solvers(),
        "modelling layers": MODELING_LAYERS.describe(),
        "builders": list_builders(),
        "seeds": list_seeds(),
        "facet policies": list_facet_policies(),
        "facet screens": list_facet_screens(),
        "refiners": list_refiners(),
        "ray strategies": list_ray_strategies(),
        "compressors": list_compressors(),
        "compression schedules": list_schedules(),
        "samplers": list_samplers(),
        "applications": list_applications(),
        "benchmark methods": list_benchmark_methods(),
    }
