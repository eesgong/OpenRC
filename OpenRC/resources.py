"""Resource management: memory budgets, hull-size estimates, and chunked iteration.

Utilities for bounding memory usage during high-dimensional convex-hull
construction and batch evaluation:

* :class:`MemoryGuard` polls process RSS against a MiB budget.
* :func:`hull_memory_mb`, :func:`measure_polytope_memory`, and
  :func:`extrapolate_vertex_budget` estimate and measure convex-hull memory.
* :func:`chunked` bounds intermediate batch sizes.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Iterator, List, Optional, Sequence, TypeVar

import numpy as np

LOGGER = logging.getLogger("OpenRC")

T = TypeVar("T")


class MemoryBudgetExceeded(RuntimeError):
    """A configured RSS or modelled hull-memory budget has been exceeded."""


def memory_report() -> dict:
    """Current process and system memory, in MiB.

    Returns zeroed fields with ``available=False`` when ``psutil`` is not installed.
    """
    try:
        import psutil
    except ImportError:  # optional dependency
        return {"rss_mb": 0.0, "available_mb": 0.0, "percent": 0.0, "available": False}

    process = psutil.Process(os.getpid())
    virtual = psutil.virtual_memory()
    return {
        "rss_mb": process.memory_info().rss / (1024 * 1024),
        "available_mb": virtual.available / (1024 * 1024),
        "percent": virtual.percent,
        "available": True,
    }


@dataclass
class MemoryGuard:
    """Cooperative memory budget, polled at safe points by the caller.

    Callers poll :meth:`under_pressure` or :meth:`check` between iterations to
    trigger vertex compression or stop before exceeding ``limit_mb``.

    Parameters
    ----------
    limit_mb:
        Resident-set-size budget in MiB.  ``None`` disables the guard.
    headroom:
        Fraction of the budget at which :meth:`under_pressure` starts returning
        ``True``, so callers can react (compress, flush, shrink a batch) before
        the hard limit is hit.
    """

    limit_mb: Optional[float] = None
    headroom: float = 0.85
    peak_mb: float = 0.0
    _process: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (isinstance(self.headroom, (bool, np.bool_))
                or not isinstance(self.headroom, (int, float, Real))
                or not math.isfinite(float(self.headroom))
                or not 0.0 < float(self.headroom) <= 1.0):
            raise ValueError("headroom must lie in (0, 1]")
        self.headroom = float(self.headroom)
        if self.limit_mb is None:
            return
        if (isinstance(self.limit_mb, (bool, np.bool_))
                or not isinstance(self.limit_mb, (int, float, Real))):
            raise ValueError("limit_mb must be a finite positive number, not a boolean")
        self.limit_mb = float(self.limit_mb)
        if not math.isfinite(self.limit_mb) or self.limit_mb <= 0.0:
            raise ValueError("limit_mb must be finite and strictly positive")
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except ImportError:  # optional dependency
            LOGGER.warning("psutil is not installed; the memory budget will be ignored")
            self.limit_mb = None

    def current_mb(self) -> float:
        """Resident set size in MiB (``0.0`` when unavailable)."""
        if self._process is None:
            return 0.0
        mb = self._process.memory_info().rss / (1024 * 1024)
        self.peak_mb = max(self.peak_mb, mb)
        return mb

    def under_pressure(self) -> bool:
        """Return ``True`` when current RSS exceeds ``headroom * limit_mb``."""
        if self.limit_mb is None:
            return False
        return self.current_mb() > self.headroom * self.limit_mb

    def exceeded(self) -> bool:
        if self.limit_mb is None:
            return False
        return self.current_mb() > self.limit_mb

    def check(self) -> None:
        """Raise :class:`MemoryBudgetExceeded` if the budget has been passed."""
        if self.exceeded():
            raise MemoryBudgetExceeded(
                f"resident memory {self.current_mb():.1f} MiB exceeds the "
                f"{self.limit_mb:.1f} MiB budget"
            )


def hull_memory_mb(n_vertices: int, dim: int, n_facets: int) -> float:
    """Memory a ``scipy`` convex hull of this shape needs, in MiB.

    A polytope owns its vertex snapshot and a private Qhull point buffer, both
    ``(N, d)`` float64 arrays, plus one hyperplane per facet
    ``(F, d+1)`` as float64, and the simplices and neighbour lists ``(F, d)`` as
    int32, so memory scales primarily with ``n_facets``.

    For ``d >= 4``, ``n_facets`` grows combinatorially and cannot be inferred
    from ``n_vertices`` alone (e.g. a 60-vertex 8-D region can exceed 22,000
    facets); obtain ``n_facets`` from :func:`measure_polytope_memory` or
    :func:`extrapolate_vertex_budget`.
    """
    if (isinstance(dim, (bool, np.bool_)) or not isinstance(dim, Integral) or dim < 1
            or isinstance(n_vertices, (bool, np.bool_)) or not isinstance(n_vertices, Integral) or n_vertices < 0
            or isinstance(n_facets, (bool, np.bool_)) or not isinstance(n_facets, Integral) or n_facets < 0):
        raise ValueError("n_vertices, dim and n_facets must be non-negative integers with dim >= 1")
    n_v, d, n_f = int(n_vertices), int(dim), int(n_facets)
    if n_v <= d:
        return 0.0
    doubles = 2 * n_v * d + n_f * (d + 1)
    ints = n_v + 2 * n_f * d   # hull vertex indices, simplices, neighbours
    return (doubles * 8 + ints * 4) / (1024 * 1024)


def measure_polytope_memory(polytope) -> dict:
    """Actual bytes held by a polytope's vertices and cached hull.

    Sums unique ``ndarray.nbytes`` across vertices, Qhull arrays, and cached
    geometry.  ``total_mb`` is a lower bound on process RSS because Qhull's
    internal C allocations are not exposed as NumPy arrays.
    """
    import numpy as np

    if not hasattr(polytope, "vertices") or not hasattr(polytope, "hull"):
        raise TypeError("polytope must be a Polytope with 'vertices' and 'hull' attributes")
    vertices = np.asarray(polytope.vertices)
    if vertices.ndim != 2 or vertices.dtype == bool:
        raise ValueError("polytope.vertices must be a 2-D numeric array")
    seen = [vertices]

    def unique_bytes(value) -> int:
        if not isinstance(value, np.ndarray):
            return 0
        if any(np.shares_memory(value, previous) for previous in seen):
            return 0
        seen.append(value)
        return int(value.nbytes)

    out = {
        "n_vertices": int(vertices.shape[0]),
        "dim": int(vertices.shape[1]),
        "vertices_mb": vertices.nbytes / (1024 * 1024),
        "n_facets": 0,
        "hull_mb": 0.0,
    }
    hull = polytope.hull
    if hull is not None:
        # Qhull receives a private copy for numerical safety.  Count it unless
        # a custom hull shares its point buffer with the polytope.
        hull_bytes = sum(
            unique_bytes(getattr(hull, attr, None))
            for attr in ("points", "equations", "simplices", "neighbors", "vertices",
                         "coplanar", "good", "min_bound", "max_bound")
        )
        out["n_facets"] = int(len(hull.equations))
        out["hull_mb"] = hull_bytes / (1024 * 1024)
    cached_bytes = 0
    for name in ("hrep", "halfspaces", "facets", "centroid", "hull_vertices"):
        cached = polytope.__dict__.get(name)
        for value in cached if isinstance(cached, tuple) else (cached,):
            cached_bytes += unique_bytes(value)
    out["cached_geometry_mb"] = cached_bytes / (1024 * 1024)
    out["total_mb"] = out["vertices_mb"] + out["hull_mb"] + out["cached_geometry_mb"]
    return out


def extrapolate_vertex_budget(observed: Sequence[tuple], dim: int,
                              limit_mb: float) -> int:
    """Largest vertex count expected to fit in ``limit_mb``, from measurements.

    Parameters
    ----------
    observed:
        At least two ``(n_vertices, n_facets)`` pairs measured on the *same*
        problem - for instance from the first two rounds of a memory-aware run.
    dim:
        Region dimension.
    limit_mb:
        Memory budget for the hull.

    The facet count is modelled as a power law ``F ~ a * N**k`` fitted to the
    observations.  Because this is an empirical fit to ``observed``, use
    alongside a :class:`MemoryGuard` to enforce hard RSS limits.
    """
    import numpy as np

    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, Integral) or not 1 <= dim < 10 ** 7:
        raise ValueError("dim must be a positive integer below the vertex search cap")
    if (isinstance(limit_mb, (bool, np.bool_))
            or not isinstance(limit_mb, (int, float, Real))
            or not math.isfinite(float(limit_mb)) or float(limit_mb) <= 0.0):
        raise ValueError("limit_mb must be finite and strictly positive")
    limit_mb = float(limit_mb)
    if isinstance(observed, (str, bytes, bytearray)):
        raise TypeError("observed must be a sequence of (n_vertices, n_facets) pairs, not str or bytes")
    raw_in = np.asarray(observed)
    if raw_in.dtype == bool:
        raise ValueError("observed must contain finite integer (n_vertices, n_facets) pairs, not booleans")
    raw = np.asarray(raw_in, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.all(np.isfinite(raw)):
        raise ValueError("observed must contain finite (n_vertices, n_facets) pairs")
    if np.any(raw != np.floor(raw)):
        raise ValueError("observed vertex and facet counts must be integers")
    points = [(float(n), float(f)) for n, f in raw if n > dim and f > 0]
    if len(points) < 2:
        raise ValueError("need at least two (n_vertices, n_facets) measurements")
    if len({n for n, _ in points}) < 2:
        raise ValueError("need at least two distinct observed vertex counts")

    logs_n = np.log([p[0] for p in points])
    logs_f = np.log([p[1] for p in points])
    k, log_a = np.polyfit(logs_n, logs_f, 1)
    if not (np.isfinite(k) and np.isfinite(log_a)) or k < -1e-12:
        raise ValueError("observations must support a finite, non-decreasing facet model")
    k = max(float(k), 0.0)
    facet_bytes = 8 * (dim + 1) + 8 * dim

    def fits(n: int) -> bool:
        remaining = limit_mb - (16 * n * dim + 4 * n) / 2 ** 20
        if remaining <= 0.0:
            return False
        # Stay in log space: a can underflow while N**k overflows even when
        # their product (the fitted facet count) is perfectly representable.
        log_facets = max(math.log(dim + 1), float(log_a) + k * math.log(n))
        return log_facets + math.log(facet_bytes) <= math.log(remaining) + math.log(2 ** 20)

    # Bracket the limit by geometric growth (x1.05), then bisect [low, high] to
    # find the exact largest integer N satisfying fits(N).
    low, high, cap = dim + 1, dim + 2, 10 ** 7
    if not fits(low):
        raise MemoryBudgetExceeded("even the estimated smallest full-dimensional hull exceeds limit_mb")
    while fits(high):
        low = high
        if high == cap:
            return cap
        high = min(cap, int(high * 1.05) + 1)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def chunked(items: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield ``items`` in lists of at most ``size`` elements.

    Used to keep a long validation sweep from materialising every intermediate
    result at once, and to hand work to a worker pool in sensible batches.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
