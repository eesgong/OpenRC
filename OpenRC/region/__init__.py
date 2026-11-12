"""Region construction: builders, seeding strategies and facet-selection policies."""

from OpenRC.region.base import (
    BUILDERS,
    SEEDS,
    RegionBuilder,
    RegionResult,
    SeedResult,
    get_builder,
    get_seed,
    list_builders,
    list_seeds,
)
from OpenRC.region.facet_policy import (
    FACET_POLICIES,
    FACET_SCREENS,
    FacetScreen,
    FacetPolicy,
    SelectionContext,
    get_facet_policy,
    list_facet_policies,
)

from OpenRC.region.refine import REFINERS, Refiner, get_refiner, list_refiners
from OpenRC.region.ray_growth import (
    RAY_STRATEGIES,
    RayProposal,
    RayStrategy,
    get_ray_strategy,
    list_ray_strategies,
)

# Side-effect imports populate the registries.
from OpenRC.region import conservative, expansive, memory_aware, ray_growth  # noqa: F401,E402

__all__ = [
    "BUILDERS",
    "FACET_POLICIES",
    "FACET_SCREENS",
    "FacetPolicy",
    "FacetScreen",
    "RAY_STRATEGIES",
    "REFINERS",
    "RayProposal",
    "RayStrategy",
    "Refiner",
    "RegionBuilder",
    "RegionResult",
    "SEEDS",
    "SeedResult",
    "SelectionContext",
    "get_builder",
    "get_facet_policy",
    "get_ray_strategy",
    "get_refiner",
    "get_seed",
    "list_builders",
    "list_facet_policies",
    "list_ray_strategies",
    "list_refiners",
    "list_seeds",
]
