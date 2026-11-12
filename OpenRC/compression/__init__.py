"""Polytope vertex-set compression and growth-time compression schedules.

Two registries are provided:

* :data:`COMPRESSORS`: algorithms selecting vertices to prune (local search,
  hybrid VNS, star-screened greedy, and random / roulette baselines);
* :data:`SCHEDULES`: policies controlling when and how far an active expansion
  compresses its live hull (vertex budget, memory budget, or adaptive).
"""

from OpenRC.compression.base import (
    COMPRESSORS,
    CompressionResult,
    Compressor,
    get_compressor,
    hull_adjacency,
    list_compressors,
    marginal_losses,
    polytope_from_live,
)
from OpenRC.compression.schedule import (
    SCHEDULES,
    AdaptiveSchedule,
    CompressionPlan,
    CompressionSchedule,
    GrowthState,
    MemoryBudgetSchedule,
    VertexBudgetSchedule,
    get_schedule,
    list_schedules,
    resolve_schedule,
)

# Side-effect imports populate the registry.
from OpenRC.compression import baselines, hybrid_vns, local_search  # noqa: F401,E402

__all__ = [
    "COMPRESSORS",
    "SCHEDULES",
    "AdaptiveSchedule",
    "CompressionPlan",
    "CompressionResult",
    "CompressionSchedule",
    "Compressor",
    "GrowthState",
    "MemoryBudgetSchedule",
    "VertexBudgetSchedule",
    "get_compressor",
    "get_schedule",
    "hull_adjacency",
    "list_compressors",
    "list_schedules",
    "marginal_losses",
    "polytope_from_live",
    "resolve_schedule",
]
