"""Test-point samplers for Monte Carlo region validation.

Because the exact feasible injection set has no closed-form description, region
error and loss rates are estimated by drawing points in injection space and
comparing polytope membership against reference power-flow verdicts.

Sampler choice determines which metrics are observable.  Sampling only inside
the polytope (:class:`InRegionSampler`, :class:`SurfaceSampler`) concentrates
all draws on estimating the **error rate** without observing feasible points
omitted outside the region; estimating the **loss rate** requires sampling a
domain extending beyond the polytope, such as its bounding box padded by a
2 MW margin (:class:`BoxSampler`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from OpenRC.registry import Registry

#: Registry of sampler classes.
SAMPLERS: Registry = Registry("sampler")


def _sample_count(n: int) -> int:
    """Validate that ``n`` is a non-negative integer (excluding booleans)."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a non-negative integer")
    return int(n)


def _make_rng(rng) -> np.random.Generator:
    """Normalise ``rng`` into a Generator, rejecting boolean seeds."""
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must not be a boolean")
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _scale_box(unit: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Interpolate finite endpoints without requiring a representable span.

    Ordinary boxes use ``lower + unit * (upper - lower)``.  Where that width
    overflows (opposite extreme endpoints), a convex combination is used
    instead, whose terms stay finite.
    """
    with np.errstate(over="ignore"):
        width = upper - lower
    safe = np.isfinite(width)
    if np.all(safe):
        return lower[None, :] + unit * width[None, :]
    points = np.empty_like(unit)
    points[:, safe] = lower[safe] + unit[:, safe] * width[safe]
    fraction = unit[:, ~safe]
    points[:, ~safe] = (1.0 - fraction) * lower[~safe] + fraction * upper[~safe]
    return points


class Sampler(ABC):
    """Produces test points in region coordinates."""

    name: str = "sampler"
    #: Whether the draws reach outside the region, so that the share of
    #: feasible points it misses (the loss rate) is observable; ``False`` for
    #: samplers restricted to the polytope or its boundary.
    measures_loss: bool = True
    #: Whether draws are independent with one common sampling distribution.
    #: Binomial confidence intervals require this; extensions must opt in.
    iid_sampling: bool = False

    @abstractmethod
    def draw(self, n: int, *, region=None, rng=None) -> np.ndarray:
        """Return an ``(n, dim)`` array of test points."""

    def describe(self) -> dict:
        return {"sampler": self.name}


@SAMPLERS.register(
    "uniform-box",
    doc="Uniform points in the region's bounding box, grown by a margin (the standard).",
)
class BoxSampler(Sampler):
    """Independent uniform draws from the region's bounding box plus a margin.

    Parameters
    ----------
    margin:
        Outward padding added to each coordinate of ``region.bounding_box()``,
        in MW, so feasible points outside the polytope are sampled when
        estimating the loss rate.
    lower, upper:
        Explicit box, overriding the region's own bounds.
    """

    name = "uniform-box"
    iid_sampling = True

    def __init__(self, *, margin: float = 2.0,
                 lower: Optional[ArrayLike] = None,
                 upper: Optional[ArrayLike] = None) -> None:
        if isinstance(margin, (bool, np.bool_)) or not isinstance(margin, (int, float, Real)):
            raise ValueError("margin must be finite and non-negative")
        self.margin = float(margin)
        if not np.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError("margin must be finite and non-negative")
        if (lower is None) != (upper is None):
            raise ValueError("explicit lower and upper bounds must be supplied together")
        if (lower is not None and np.asarray(lower).dtype == bool) or (
                upper is not None and np.asarray(upper).dtype == bool):
            raise ValueError("box bounds must be numeric, not boolean")
        self.lower = None if lower is None else np.array(lower, dtype=float, copy=True)
        self.upper = None if upper is None else np.array(upper, dtype=float, copy=True)
        if self.lower is not None:
            self.bounds(None)

    def bounds(self, region):
        if self.lower is not None and self.upper is not None:
            lower, upper = self.lower, self.upper
        else:
            if region is None:
                raise ValueError("BoxSampler needs either a region or explicit bounds")
            lower, upper = region.bounding_box(self.margin)
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lower.ndim != 1 or lower.size == 0 or upper.shape != lower.shape
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(upper < lower)):
            raise ValueError("box bounds must be finite, non-empty vectors with lower <= upper")
        if region is not None and len(lower) != region.dim:
            raise ValueError("box bounds must match the region dimension")
        return lower, upper

    def draw(self, n: int, *, region=None, rng=None) -> np.ndarray:
        n = _sample_count(n)
        lower, upper = self.bounds(region)
        generator = _make_rng(rng)
        unit = generator.random((n, len(lower)))
        return _scale_box(unit, lower, upper)

    def describe(self) -> dict:
        if self.lower is not None and self.upper is not None:
            # Explicit bounds replace the margin; record the box actually used.
            return {"sampler": self.name, "lower": self.lower.tolist(),
                    "upper": self.upper.tolist()}
        return {"sampler": self.name, "margin": self.margin}


@SAMPLERS.register(
    "sobol-box",
    doc="Low-discrepancy (Sobol) points in the grown bounding box - tighter estimates.",
)
class SobolBoxSampler(BoxSampler):
    """Quasi-random version of :class:`BoxSampler`.

    A scrambled Sobol sequence covers the box far more evenly than independent
    draws, so the same number of (expensive) power-flow evaluations yields a
    noticeably tighter estimate of the loss and error rates.
    """

    name = "sobol-box"
    iid_sampling = False

    def draw(self, n: int, *, region=None, rng=None) -> np.ndarray:
        from scipy.stats import qmc

        n = _sample_count(n)
        lower, upper = self.bounds(region)
        # Pass a Generator on to SciPy, so seeded draws stay reproducible.
        if isinstance(rng, (bool, np.bool_)):
            raise TypeError("rng must not be a boolean")
        seed = int(rng) if isinstance(rng, (int, np.integer)) else rng
        engine = qmc.Sobol(d=len(lower), scramble=True, seed=seed)
        unit = engine.random(n)
        return _scale_box(unit, lower, upper)


@SAMPLERS.register(
    "in-region",
    doc="Uniform points inside the region itself - measures correctness only.",
)
class InRegionSampler(Sampler):
    """Uniform points drawn from inside the polytope.

    Concentrates all samples inside the polytope when estimating the error rate
    without spending reference evaluations outside the region.

    ``method``:

    ``"auto"`` (default)
        Rejection sampling from the bounding box; if its work budget
        (``max_attempts_factor`` draws per point) runs out - a slender region
        can fill under a millionth of its box - the remaining points are drawn
        by ``"cones"``.  Where rejection succeeds, the sample is the one
        ``"rejection"`` draws.
    ``"rejection"``
        Rejection sampling only; raises when the budget runs out.
    ``"cones"``
        Exact and rejection-free: the cones from an interior point over the
        triangulated facets partition the body, so choosing a cone with
        probability proportional to its volume and a point inside it with flat
        Dirichlet weights on its corners is uniform on the whole region.
    """

    name = "in-region"
    measures_loss = False
    iid_sampling = True
    METHODS = ("auto", "rejection", "cones")

    def __init__(self, *, max_attempts_factor: int = 200, method: str = "auto") -> None:
        self.max_attempts_factor = _sample_count(max_attempts_factor)
        if self.max_attempts_factor == 0:
            raise ValueError("max_attempts_factor must be strictly positive")
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.method = method

    def draw(self, n: int, *, region=None, rng=None) -> np.ndarray:
        n = _sample_count(n)
        if region is None:
            raise ValueError("InRegionSampler needs a region")
        generator = _make_rng(rng)
        if n == 0:
            return np.empty((0, region.dim), dtype=float)
        if self.method == "cones":
            return _cone_draw(region, n, generator)
        lower, upper = region.bounding_box()
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lower.ndim != 1 or lower.shape != (region.dim,) or upper.shape != lower.shape
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(upper < lower)):
            raise ValueError(
                "box bounds must be finite, non-empty vectors matching region.dim with lower <= upper"
            )
        kept: List[np.ndarray] = []
        attempts = 0
        budget = n * self.max_attempts_factor
        while len(kept) < n and attempts < budget:
            # Bound both rejection work and temporary storage.  In high
            # dimensions an unsuccessful sampler must not overshoot its budget.
            batch = min(max(n - len(kept), 32) * 4, budget - attempts, 4096)
            attempts += batch
            points = _scale_box(generator.random((batch, len(lower))), lower, upper)
            inside = region.contains(points)
            kept.extend(points[inside][: n - len(kept)])
        if len(kept) < n:
            failure = RuntimeError(
                f"rejection sampling produced only {len(kept)} of {n} points; the region "
                "occupies a very small share of its bounding box (method='cones' samples "
                "it exactly)"
            )
            if self.method != "auto":
                raise failure
            found = np.array(kept, dtype=float).reshape(-1, region.dim)
            try:
                rest = _cone_draw(region, n - len(found), generator)
            except (AttributeError, ValueError):
                raise failure from None       # no full-dimensional body to decompose
            return np.vstack([found, rest])
        return np.array(kept)

    def describe(self) -> dict:
        info = super().describe()
        info.update(method=self.method)
        return info


def _cone_draw(region, n: int, generator: np.random.Generator) -> np.ndarray:
    """``n`` uniform points of ``region`` by cone decomposition (no rejection)."""
    from OpenRC.geometry.live_hull import cone_volumes

    hull = region.hull
    if hull is None:
        raise ValueError("in-region sampling needs a full-dimensional region")
    points = np.asarray(hull.points, dtype=float)
    simplices = np.asarray(hull.simplices)
    centre = points[np.asarray(hull.vertices)].mean(axis=0)
    volumes = cone_volumes(points, simplices, centre)
    total = float(volumes.sum())
    if not (np.isfinite(total) and total > 0.0):
        raise ValueError("the region's cone decomposition has no volume")
    dim = points.shape[1]
    pick = generator.choice(len(simplices), size=n, p=volumes / total)
    out = np.empty((n, dim))
    batch = max(1, min(4096, 16 * 2 ** 20 // (8 * (dim + 1) * dim)))
    for first in range(0, n, batch):
        last = min(first + batch, n)
        weights = generator.dirichlet(np.ones(dim + 1), size=last - first)
        corners = np.concatenate([points[simplices[pick[first:last]]],
                                  np.broadcast_to(centre, (last - first, 1, dim))], axis=1)
        out[first:last] = np.einsum("nk,nkd->nd", weights, corners)
    return out


@SAMPLERS.register(
    "surface",
    doc="Uniform points on the region's boundary - probes the facets between vertices.",
)
class SurfaceSampler(Sampler):
    """Points distributed uniformly over the surface of the polytope.

    Region construction evaluates feasibility at the hull vertices, whereas
    open facets between vertices are interpolated by convexity.  When the true
    feasible set is non-convex, boundary violations typically occur on these
    facets.

    ``ConvexHull`` triangulates every facet into ``(dim-1)``-simplices.  A
    simplex is chosen with probability proportional to its area and a point is
    drawn uniformly inside it (flat Dirichlet weights on its corners), which
    makes the sample uniform on the whole boundary.
    """

    name = "surface"
    measures_loss = False
    iid_sampling = True

    def draw(self, n: int, *, region=None, rng=None) -> np.ndarray:
        from OpenRC.geometry.measures import simplex_volumes

        n = _sample_count(n)
        if region is None:
            raise ValueError("SurfaceSampler needs a region")
        generator = _make_rng(rng)
        if n == 0:
            return np.empty((0, region.dim), dtype=float)
        hull = region.hull
        if hull is None:
            raise ValueError("SurfaceSampler needs a full-dimensional region")
        dim = region.dim
        if dim == 1:
            endpoints = region.hull_vertices
            return endpoints[generator.integers(0, len(endpoints), size=n)].copy()
        # Never materialise (n_facets, dim, dim) for a high-dimensional hull.
        # Account for corners, edges, Gram matrices and scaled copies together.
        batch = max(1, min(4096, 16 * 2 ** 20 // (8 * 4 * dim * dim)))
        areas = np.empty(len(hull.simplices))
        for first in range(0, len(areas), batch):
            last = min(first + batch, len(areas))
            corners = region.vertices[hull.simplices[first:last]]
            # A single ill-conditioned facet can disappear in a Gram
            # determinant even when the total area remains positive.
            areas[first:last] = simplex_volumes(corners)
        with np.errstate(over="ignore"):
            total = float(areas.sum())
        if not np.all(np.isfinite(areas)) or not np.isfinite(total) or total == 0.0:
            # Absolute area may not fit float64 even though relative sampling
            # probabilities do.  Scale each simplex and evaluate log areas.
            logs = np.full(len(areas), -np.inf)
            for first in range(0, len(areas), batch):
                last = min(first + batch, len(areas))
                corners = region.vertices[hull.simplices[first:last]]
                edges = corners[:, 1:, :] - corners[:, :1, :]
                scale = np.abs(edges).max(axis=2)
                valid = np.all(scale > 0.0, axis=1)
                if np.any(valid):
                    scaled = edges[valid] / scale[valid, :, None]
                    singular = np.linalg.svd(scaled, compute_uv=False)
                    with np.errstate(divide="ignore"):
                        # The common factorial cancels when normalising.
                        values = np.log(singular).sum(axis=1) + np.log(scale[valid]).sum(axis=1)
                    deficient = singular[:, -1] <= np.finfo(float).eps * dim * singular[:, 0]
                    values[deficient] = -np.inf
                    logs[first:last][valid] = values
            largest = logs.max(initial=-np.inf)
            if np.isfinite(largest):
                with np.errstate(under="ignore"):
                    areas = np.exp(logs - largest)
        if not areas.sum() > 0.0:
            raise ValueError("the region's surface has zero area")
        pick = generator.choice(len(areas), size=n, p=areas / areas.sum())
        points = np.empty((n, dim))
        for first in range(0, n, batch):
            last = min(first + batch, n)
            weights = generator.dirichlet(np.ones(dim), size=last - first)
            corners = region.vertices[hull.simplices[pick[first:last]]]
            points[first:last] = np.einsum("nk,nkj->nj", weights, corners)
        return points


@SAMPLERS.register(
    "scenarios",
    doc="Replay recorded operating scenarios instead of synthetic points.",
)
class ScenarioSampler(Sampler):
    """Draw from a supplied matrix of real operating points.

    Evaluates region membership and feasibility on historical or simulated
    operating profiles rather than synthetic uniform draws.
    """

    name = "scenarios"

    def __init__(self, scenarios: np.ndarray, *, replace: bool = False) -> None:
        if np.asarray(scenarios).dtype == bool:
            raise ValueError("scenarios must be a numeric matrix, not boolean")
        self.scenarios = np.atleast_2d(np.array(scenarios, dtype=float, copy=True))
        if (self.scenarios.ndim != 2 or 0 in self.scenarios.shape
                or not np.all(np.isfinite(self.scenarios))):
            raise ValueError("scenarios must be a finite (n, dim) matrix with n > 0 and dim > 0")
        if not isinstance(replace, (bool, np.bool_)):
            raise ValueError("replace must be a boolean")
        self.replace = bool(replace)
        # With replacement, independent draws estimate probabilities under the
        # empirical scenario distribution.  Without replacement, counts are
        # dependent (or a full census), and a binomial interval is not valid.
        self.iid_sampling = self.replace

    def draw(self, n: int, *, region=None, rng=None) -> np.ndarray:
        n = _sample_count(n)
        if region is not None and self.scenarios.shape[1] != region.dim:
            raise ValueError("scenarios must match the region dimension")
        generator = _make_rng(rng)
        total = len(self.scenarios)
        if n >= total and not self.replace:
            return self.scenarios.copy()
        index = generator.choice(total, size=n, replace=self.replace)
        return self.scenarios[index]

    def describe(self) -> dict:
        return {"sampler": self.name, "n_available": int(len(self.scenarios))}


def get_sampler(name: str, **kwargs) -> Sampler:
    return SAMPLERS.get(name)(**kwargs)


def list_samplers() -> List[tuple]:
    return SAMPLERS.describe()
