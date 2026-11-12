"""Grid-aware optimisation problems posed over a region."""

from OpenRC.applications.base import (
    APPLICATIONS,
    Application,
    BatchResult,
    SolveResult,
    get_application,
    list_applications,
)

# Side-effect imports populate the registry.
from OpenRC.applications import capacity_allocation, storage_sizing  # noqa: F401,E402

__all__ = [
    "APPLICATIONS",
    "Application",
    "BatchResult",
    "SolveResult",
    "get_application",
    "list_applications",
]
