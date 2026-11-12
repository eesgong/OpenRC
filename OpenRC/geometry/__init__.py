"""Geometry layer: polytopes, measures, ray search and set operations."""

from OpenRC.geometry.bisection import (
    InfeasibleOriginError,
    UnboundedRayError,
    RayResult,
    axis_box,
    ray_boundary,
)
from OpenRC.geometry.measures import (
    bounding_box,
    cone_volume,
    hull_volume,
    simplex_volume,
    simplex_volumes,
    volume_gain,
)
from OpenRC.geometry.ops import (
    EmptyIntersectionError,
    GeometryNumericsError,
    chebyshev_center,
    clip_to_box,
    erode_by_box,
    from_halfspaces,
    intersect,
    intersect_all,
)
from OpenRC.geometry.live_hull import (
    geometry_workers,
    set_geometry_workers,
    shutdown_geometry_workers,
)
from OpenRC.geometry.polytope import Polytope

__all__ = [
    "geometry_workers",
    "set_geometry_workers",
    "shutdown_geometry_workers",
    "EmptyIntersectionError",
    "GeometryNumericsError",
    "erode_by_box",
    "from_halfspaces",
    "InfeasibleOriginError",
    "UnboundedRayError",
    "Polytope",
    "RayResult",
    "axis_box",
    "bounding_box",
    "chebyshev_center",
    "clip_to_box",
    "cone_volume",
    "hull_volume",
    "intersect",
    "intersect_all",
    "ray_boundary",
    "simplex_volume",
    "simplex_volumes",
    "volume_gain",
]
