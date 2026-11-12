"""Region builders: the interface, the result object, and the seeding strategies.

A *builder* turns ``(case, axes, oracle)`` into a :class:`Polytope` plus a
record of how it got there.  Three are registered:

``expansive``     grow a polytope by repeatedly breaking a facet (the method of
                  the hosting-capacity-region and facet-selection papers)
``memory-aware``  the same, interleaved with compression so the vertex set never
                  outgrows a memory budget
``conservative``  build an inner approximation of the exact feasible set, by
                  either of the two published routes

Seeding strategies construct the initial polytope for region growth and are
registered separately::

    simplex   d+1 rays (the unit axes plus one negative diagonal)
    axis      2d rays along +/- each axis; also yields an axis-intercept box
    random    the axis seed, plus N uniformly sampled points inside that box
              whose feasible members are added as vertices.  Starting from a
              rough hull rather than a simplex removes a large number of early,
              low-value facet breaks.
"""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from OpenRC.geometry.bisection import axis_box, ray_boundary
from OpenRC.geometry.polytope import Polytope
from OpenRC.registry import Registry

if TYPE_CHECKING:
    from OpenRC.geometry.live_hull import LiveHull

#: Registry of region-builder classes.
BUILDERS: Registry = Registry("region builder")

#: Registry of seeding strategies (callables, see :func:`seed_simplex`).
SEEDS: Registry = Registry("seed")

#: ``seed="auto"`` uses the axis seed from this dimension up, the simplex below.
#: Across random benchmark instances (``OpenRC.benchmark``), the axis seed
#: yields 2-66x the simplex seed's volume in 4-D; 3-D retains the simplex to
#: reproduce published low-dimensional results.
AXIS_SEED_FROM_DIM = 4


def resolve_seed(name: str, dim: int) -> str:
    """The seeding strategy ``name`` stands for in ``dim`` dimensions.

    Only ``"auto"`` depends on ``dim``; explicit seed names are returned
    unchanged.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"seed name must be a non-empty string, got {name!r}")
    if (isinstance(dim, (bool, np.bool_))
            or not isinstance(dim, (int, np.integer))
            or int(dim) < 1):
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    if name == "auto":
        return "axis" if int(dim) >= AXIS_SEED_FROM_DIM else "simplex"
    return name


def check_hull_budget(live: "LiveHull", max_hull_mb: Optional[float]) -> None:
    """Reject a starting hull that already exceeds its modelled memory cap.

    Callers validate the positive finite cap. This check does not own/release
    the hull: a builder releases a rejected hull it created, while a resumed
    caller-owned hull remains usable. Seed construction has already happened;
    this is not a bound on Qhull's temporary allocations or process RSS.
    """
    if max_hull_mb is None:
        return
    required = live.memory_mb()
    if required > max_hull_mb:
        from OpenRC.resources import MemoryBudgetExceeded

        raise MemoryBudgetExceeded(
            f"the starting hull needs {required:.6g} MiB of modelled memory, "
            f"above max_hull_mb={max_hull_mb:.6g}; use a smaller seed, compress "
            "the supplied hull, or raise the cap")


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class RegionResult:
    """A region plus the provenance needed to compare runs fairly."""

    polytope: Polytope
    trace: List[dict] = field(default_factory=list)
    oracle_stats: Dict[str, Any] = field(default_factory=dict)
    builder_info: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    stopped_because: str = ""
    #: The working hull, when the builder was asked to hand it on
    #: (``keep_hull=True``).  Not part of the result proper: call
    #: :meth:`release_hull` once it is no longer needed.
    live_hull: Optional["LiveHull"] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.polytope, Polytope):
            raise TypeError(f"polytope must be a Polytope, got {type(self.polytope).__name__}")

    def release_hull(self) -> None:
        """Free the working hull, keeping only the vertex set."""
        if self.live_hull is not None:
            self.live_hull.release()
            self.live_hull = None

    @property
    def volume(self) -> float:
        return self.polytope.volume

    @property
    def n_vertices(self) -> int:
        return self.polytope.n_vertices

    def to_frame(self):
        """The growth trace as a ``pandas.DataFrame`` (volume vs vertices vs time)."""
        import pandas as pd

        return pd.DataFrame(self.trace)

    def summary(self) -> str:
        calls = self.oracle_stats.get("n_calls", "?")
        return (
            f"{self.builder_info.get('builder', 'region')}: "
            f"{self.n_vertices} vertices, volume {self.volume:.6g}, "
            f"{self.seconds:.1f} s, {calls} oracle calls"
            + (f"\n    stopped: {self.stopped_because}" if self.stopped_because else "")
        )

    def save(self, path) -> None:
        """Persist the polytope (CSV + JSON) and the trace (CSV) side by side."""
        from pathlib import Path

        path = Path(path)
        if path.suffix.lower() in (".csv", ".json"):
            path = path.with_suffix("")
        self.polytope.metadata.update(
            builder=self.builder_info,
            oracle_stats=self.oracle_stats,
            seconds=self.seconds,
            stopped_because=self.stopped_because,
        )
        self.polytope.save(path)
        if self.trace:
            self.to_frame().to_csv(path.with_name(path.name + "_trace.csv"), index=False)


# --------------------------------------------------------------------------
# Builder interface
# --------------------------------------------------------------------------
class RegionBuilder(ABC):
    """Turns a case + axes + oracle into a :class:`RegionResult`."""

    name: str = "builder"

    @abstractmethod
    def build(self, case, axes, oracle, **kwargs) -> RegionResult:
        """Compute the region.  Implementations must fill in :class:`RegionResult`."""

    def describe(self) -> dict:
        return {"builder": self.name}


def get_builder(name: str, **kwargs) -> RegionBuilder:
    return BUILDERS.get(name)(**kwargs)


def list_builders() -> List[tuple]:
    return BUILDERS.describe()


# --------------------------------------------------------------------------
# Seeding strategies
# --------------------------------------------------------------------------
@dataclass
class SeedResult:
    """Initial vertices for a region build, plus what they cost."""

    vertices: np.ndarray
    n_oracle_calls: int
    seconds: float
    info: Dict[str, Any] = field(default_factory=dict)


#: Keywords a seed forwards to :func:`~OpenRC.geometry.bisection.ray_boundary`.
_RAY_OPTIONS = ("origin", "initial_ratio", "max_ratio", "tol", "max_iter", "require_bracket",
                "hint", "hint_width", "check_origin", "search")


def _check_ray_options(seed: str, options: dict) -> None:
    """Validate that ``options`` contains only supported ray-search keywords."""
    unknown = sorted(set(options) - set(_RAY_OPTIONS))
    if unknown:
        raise TypeError(
            f"the {seed!r} seed does not take {', '.join(unknown)}; besides its own "
            f"options it forwards only ray-search options ({', '.join(_RAY_OPTIONS)}). "
            "n_samples, sampler and batch_size belong to seed='random'.")


@SEEDS.register("simplex", doc="d+1 rays: each positive axis plus one negative diagonal.")
def seed_simplex(oracle, axes, *, rng=None, **kwargs) -> SeedResult:
    """The classical seed: a ``d+1``-vertex simplex around the origin.

    Directions are the ``d`` positive unit axes plus the normalised negative
    diagonal, which is the smallest set that spans a body of non-zero volume.
    """
    _check_ray_options("simplex", kwargs)
    start = time.perf_counter()
    dim = len(axes)
    directions = list(np.eye(dim))
    directions.append(-np.ones(dim) / np.sqrt(dim))

    vertices, calls = [], 0
    for direction in directions:
        result = ray_boundary(oracle, axes, direction, **kwargs)
        vertices.append(result.boundary)
        calls += result.n_oracle_calls
    return SeedResult(np.array(vertices), calls, time.perf_counter() - start,
                      {"seed": "simplex", "n_rays": len(directions)})


@SEEDS.register("axis", doc="2d rays along +/- every axis; also gives an axis-intercept box.")
def seed_axis(oracle, axes, *, rng=None, n_workers=1, backend="process",
              **kwargs) -> SeedResult:
    """Probe both directions of every axis.

    Costs twice as many rays as ``simplex`` but starts from a body that reaches
    the boundary in both directions along every coordinate axis.  These
    intercepts do not generally give the full region's coordinate extrema.
    The ``2d`` rays are independent, so ``n_workers`` spreads them across
    processes.
    """
    _check_ray_options("axis", kwargs)
    start = time.perf_counter()
    lower, upper, vertices, calls = _axis_probe(oracle, axes, n_workers=n_workers,
                                                backend=backend, **kwargs)
    return SeedResult(vertices, calls, time.perf_counter() - start,
                      {"seed": "axis", "box_lower": lower.tolist(),
                       "box_upper": upper.tolist(), "n_workers": n_workers})


@SEEDS.register(
    "random",
    doc="Axis seed plus N random points from its box; the feasible ones become vertices.",
)
def seed_random(oracle, axes, *, n_samples: int = 100, rng=None,
                sampler: str = "uniform", n_workers=1, backend="process",
                batch_size: int = 10_000, **kwargs) -> SeedResult:
    """Random acceleration: start from the hull of a cloud of feasible points.

    The expansive loop spends its early iterations discovering the coarse shape
    of the region one ray at a time.  A batch of random feasibility tests finds
    that shape far more cheaply, because each test is a single oracle call
    rather than the ~30 a bisection needs.  The hull of the feasible samples
    (together with the axis boundary points, which guarantee a non-degenerate
    body) is handed to the expansive loop as its starting polytope.

    Parameters
    ----------
    n_samples:
        Number of random points to test.
    sampler:
        ``"uniform"`` for independent uniform draws, ``"sobol"`` for a
        low-discrepancy sequence that covers the box more evenly for the same
        budget.
    n_workers:
        Worker count for evaluating the independent random samples.  ``"auto"``
        estimates the per-call cost and starts a worker pool only when the
        expected speedup exceeds pool start-up overhead (see
        :func:`OpenRC.parallel.plan_workers`).
    batch_size:
        Maximum random points materialised at once.  Each batch is reduced to
        its hull before the next one, bounding temporary array memory without
        changing the convex hull of all feasible samples.
    """
    import numpy as _np

    start = time.perf_counter()
    if (isinstance(n_samples, (bool, _np.bool_))
            or not isinstance(n_samples, (int, float, _np.number))
            or not np.isfinite(n_samples)
            or int(n_samples) != n_samples or n_samples < 0):
        raise ValueError("n_samples must be a non-negative integer")
    if (isinstance(batch_size, (bool, _np.bool_))
            or not isinstance(batch_size, (int, float, _np.number))
            or not np.isfinite(batch_size)
            or int(batch_size) != batch_size or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer")
    # Validate arguments before running the 2d axis boundary searches.
    _check_sampler(sampler)
    _check_rng(rng)
    _check_ray_options("random", kwargs)
    n_samples = int(n_samples)
    batch_size = int(batch_size)

    # 1. axis probes: the sampling box, and a guaranteed non-degenerate seed
    lower, upper, axis_vertices, calls = _axis_probe(oracle, axes, n_workers=n_workers,
                                                     backend=backend, **kwargs)

    # 2. Draw and evaluate bounded batches.  Discarding points interior to a
    # batch hull is exact: such a point is also interior to the hull of the
    # union, so it can never become a final vertex.
    retained = _np.asarray(axis_vertices, dtype=float)
    n_feasible = 0
    with _BatchFeasibility(oracle, axes, n_workers=n_workers, backend=backend,
                           total=n_samples) as feasible_of:
        for points in _draw_batches(
            lower, upper, n_samples, batch_size=batch_size, sampler=sampler, rng=rng
        ):
            mask = feasible_of(points)
            calls += len(points)
            feasible = points[mask]
            n_feasible += len(feasible)
            if len(feasible):
                retained = _hull_points(_np.vstack([retained, feasible]))

    vertices = retained
    return SeedResult(
        vertices,
        calls,
        time.perf_counter() - start,
        {
            "seed": "random",
            "n_samples": int(n_samples),
            "n_feasible": int(n_feasible),
            "hit_rate": round(n_feasible / n_samples, 4) if n_samples else 0.0,
            "batch_size": batch_size,
            "sampler": sampler,
            "n_workers": n_workers,
            "box_lower": lower.tolist(),
            "box_upper": upper.tolist(),
        },
    )


def _charge_calls(oracle, remote_stats) -> None:
    """Merge complete worker query statistics under the caller's oracle lock.

    The worker timings exclude executor start-up.  A temporary statistics
    object keeps remote evaluation from racing with queries on the live oracle.
    """
    with getattr(oracle, "_lock", None) or contextlib.nullcontext():
        oracle.stats.n_calls += remote_stats.n_calls
        oracle.stats.n_feasible += remote_stats.n_feasible
        oracle.stats.seconds += remote_stats.seconds


def _axis_probe(oracle, axes, *, n_workers=1, backend="process", **kwargs):
    """``axis_box`` with an optional parallel sweep over the 2d ray directions.

    ``"auto"`` fires the first ray here and starts a pool for the others only
    if their measured cost repays it (:func:`~OpenRC.parallel.plan_workers`):
    on a closed-form model all 2d rays take milliseconds, while spawning a
    pool takes seconds.  The rays do not depend on one another, so the points
    are the same whichever way they are found.
    """
    from OpenRC.parallel import OracleSpec, map_rays, plan_workers, resolve_workers
    from OpenRC.powerflow.base import OracleStats

    spec = OracleSpec.from_oracle(oracle, axes) if resolve_workers(n_workers) != 1 else None
    if spec is None:
        return axis_box(oracle, axes, **kwargs)

    dim = len(axes)
    directions = []
    for k in range(dim):
        for sign in (+1.0, -1.0):
            direction = np.zeros(dim)
            direction[k] = sign
            directions.append(direction)
    points, calls = [], 0
    workers = n_workers
    if n_workers == "auto":
        start = time.perf_counter()
        first = ray_boundary(oracle, axes, directions[0], **kwargs)
        points, calls = [first.boundary], first.n_oracle_calls
        workers = plan_workers(len(directions) - 1, time.perf_counter() - start, "auto",
                               key=(spec.case, spec.oracle))
        if workers == 1:
            for direction in directions[1:]:
                result = ray_boundary(oracle, axes, direction, **kwargs)
                points.append(result.boundary)
                calls += result.n_oracle_calls
        directions = directions[1:] if workers > 1 else []
    if directions:
        remote_stats = OracleStats()
        remote, remote_calls = map_rays(oracle.case, axes, spec.oracle, directions,
                                        n_workers=workers, backend=backend,
                                        oracle_kwargs=spec.oracle_options(),
                                        ray_kwargs=kwargs, stats=remote_stats)
        _charge_calls(oracle, remote_stats)
        points += list(remote)
        calls += remote_calls
    boundary = np.array(points, dtype=float)
    lower = np.zeros(dim)
    upper = np.zeros(dim)
    for k in range(dim):
        upper[k] = boundary[2 * k][k]
        lower[k] = boundary[2 * k + 1][k]
    return lower, upper, boundary, calls


class _BatchFeasibility:
    """Feasibility of successive batches, starting at most one worker pool.

    Every query is charged to ``oracle``, whoever answers it.  In this process
    a batch goes through ``oracle.feasible_mask`` (vectorised for the
    closed-form DistFlow models, verdict for verdict the same as one query at
    a time).  ``"auto"`` times the first ``calibration`` queries on the warmed
    ``oracle`` (retaining their verdicts) and amortises pool start-up across
    all ``total`` remaining samples (:func:`~OpenRC.parallel.plan_workers`).

    A pool, once started, serves every later batch, so the workers are spawned
    once per seed rather than once per batch.
    """

    def __init__(self, oracle, axes, *, n_workers=1, backend="process",
                 total: Optional[int] = None, calibration: int = 12) -> None:
        from OpenRC.parallel import resolve_workers

        if backend not in ("process", "thread"):
            raise ValueError("backend must be 'process' or 'thread'")
        if (isinstance(calibration, (bool, np.bool_))
                or not isinstance(calibration, (int, np.integer))
                or int(calibration) < 2):
            raise ValueError("calibration must be an integer >= 2")
        self.oracle, self.axes, self.backend = oracle, axes, backend
        self.calibration = int(calibration)
        self.remaining = total
        self._auto = n_workers == "auto"
        self._workers: Optional[int] = None if self._auto else resolve_workers(n_workers)
        self._spec: Any = None
        self._spec_checked = False
        self._pool: Any = None

    def __enter__(self) -> "_BatchFeasibility":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.close()

    def _local(self, batch: np.ndarray) -> np.ndarray:
        if len(batch) == 0:
            return np.zeros(0, dtype=bool)
        return np.asarray(self.oracle.feasible_mask(self.axes, batch), dtype=bool)

    def __call__(self, points) -> np.ndarray:
        from OpenRC.parallel import FeasibilityPool, OracleSpec, plan_workers
        from OpenRC.powerflow.base import OracleStats

        points = np.atleast_2d(np.asarray(points, dtype=float))
        still_to_come = len(points) if self.remaining is None else max(self.remaining,
                                                                        len(points))
        if self.remaining is not None:
            self.remaining = max(0, self.remaining - len(points))
        if len(points) == 0 or self._workers == 1:
            return self._local(points)
        if not self._spec_checked:
            self._spec_checked = True
            self._spec = OracleSpec.from_oracle(self.oracle, self.axes)
        if self._spec is None:
            return self._local(points)

        head = np.zeros(0, dtype=bool)
        rest = points
        if self._workers is None:                     # "auto", not decided yet
            count = max(2, min(self.calibration, len(points)))
            # The first query may build a model, so it is evaluated but not timed.
            first = self._local(points[:1])
            start = time.perf_counter()
            timed = self._local(points[1:count])
            per_item = (time.perf_counter() - start) / max(count - 1, 1)
            head, rest = np.concatenate([first, timed]), points[count:]
            self._workers = plan_workers(still_to_come - count, per_item, "auto",
                                         key=(self._spec.case, self._spec.oracle))
        if self._workers == 1 or len(rest) == 0:
            return np.concatenate([head, self._local(rest)])

        if self._pool is None:
            self._pool = FeasibilityPool(self._spec, n_workers=self._workers,
                                         backend=self.backend).__enter__()
        remote_stats = OracleStats()
        tail = np.asarray(self._pool.feasible(rest, stats=remote_stats), dtype=bool)
        _charge_calls(self.oracle, remote_stats)
        return np.concatenate([head, tail])


def _insertion_subset(points: np.ndarray, target: int) -> np.ndarray:
    """``target`` of ``points`` chosen by greedy volume insertion.

    Initialises a ``(d + 1)``-simplex by farthest-point orthogonal selection
    from the centroid, then greedily inserts the candidate with largest volume
    gain until the hull reaches ``target`` vertices.  Candidate gains equal
    ``sum(area_f * distance_f) / d`` over visible facets (evaluated via a
    batched matrix product); interior candidates are permanently discarded.
    Returns the hull vertices.
    """
    from OpenRC.geometry.live_hull import LiveHull, cone_volumes

    d = points.shape[1]
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    while len(chosen) < d + 1:
        offsets = points - points[chosen[0]]
        if len(chosen) > 1:
            basis, _ = np.linalg.qr((points[chosen[1:]] - points[chosen[0]]).T)
            offsets = offsets - (offsets @ basis) @ basis.T
        chosen.append(int(np.argmax(np.linalg.norm(offsets, axis=1))))
    hull = LiveHull(points[chosen])
    try:
        alive = np.ones(len(points), dtype=bool)
        alive[chosen] = False
        while hull.n_vertices < target and alive.any():
            candidates = np.flatnonzero(alive)
            eq = hull.equations
            inner = hull.vertices.mean(axis=0)
            denom = np.maximum(-(eq[:, :-1] @ inner + eq[:, -1]), 1e-300)
            areas = d * cone_volumes(hull.points, hull.simplices, inner) / denom
            gains = np.empty(len(candidates))
            step = max(1, (1 << 22) // len(eq))           # at most 32 MiB of distances
            for first in range(0, len(candidates), step):
                block = candidates[first:first + step]
                distances = eq[:, :-1] @ points[block].T + eq[:, -1:]
                gains[first:first + step] = areas @ np.maximum(distances, 0.0) / d
            outside = gains > 1e-12 * hull.volume
            alive[candidates[~outside]] = False
            if not outside.any():
                break
            best = candidates[int(np.argmax(np.where(outside, gains, -np.inf)))]
            hull.add(points[best])
            alive[best] = False
        return hull.vertices.copy()
    finally:
        hull.release()


def _compress_oversized_seed(live, max_vertices: int, axes, seed_info: dict) -> dict:
    """Reduce a seed hull that already exceeds ``max_vertices`` to it, in place.

    A random seed with many samples can start above the budget.  It keeps
    ``max_vertices`` of its vertices, chosen by greedy volume insertion
    (:func:`_insertion_subset`): within about 2 percentage points of the volume
    greedy removal keeps, at a small fraction of its cost in high dimension
    (an 8-D seed of 1,010 vertices: under a second).  The ``"auto"`` compressor
    is the fallback for a degenerate start.  The loss is logged and recorded;
    returns the updated ``seed_info``.
    """
    from OpenRC.utils import LOGGER

    n_seed, volume = live.n_vertices, live.volume
    try:
        live.reset(_insertion_subset(live.vertices, max_vertices))
        method = "greedy insertion"
    except (ValueError, RuntimeError):               # a degenerate start simplex
        from OpenRC.compression.base import auto_compressor_name, get_compressor

        method = auto_compressor_name(live.dim)
        get_compressor(method).compress_live(live, target_vertices=max_vertices, axes=axes)
    loss = 100.0 * (1.0 - live.volume / volume)
    LOGGER.warning("the seed had %d hull vertices, more than max_vertices=%d; compressed it "
                   "to %d by %s, losing %.3f %% of its volume", n_seed, max_vertices,
                   live.n_vertices, method, loss)
    return dict(seed_info, exceeds_max_vertices=n_seed, seed_compressed_to=live.n_vertices,
                seed_volume_loss_pct=float(loss), seed_compression=method)


def _check_rng(rng) -> None:
    if not (rng is None or isinstance(rng, np.random.Generator)
            or callable(getattr(rng, "randrange", None))):
        raise TypeError(f"rng must be a random.Random, a numpy Generator or None, "
                        f"got {type(rng).__name__}")


def _check_sampler(sampler: str) -> None:
    if sampler not in ("uniform", "sobol"):
        raise ValueError(f"unknown sampler {sampler!r}; use 'uniform' or 'sobol'")


def _hull_points(points: np.ndarray) -> np.ndarray:
    """Return only extreme points, or all points while the set is degenerate."""
    points = np.asarray(points, dtype=float)
    if len(points) <= points.shape[1]:
        return points
    from scipy.spatial import ConvexHull, QhullError

    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return points


def _draw_batches(lower: np.ndarray, upper: np.ndarray, n: int, *,
                  batch_size: int, sampler: str, rng):
    """Yield reproducible random points without materialising all ``n`` rows."""
    dim = len(lower)
    _check_sampler(sampler)
    _check_rng(rng)
    if rng is None:
        seed = None
    elif isinstance(rng, np.random.Generator):
        seed = int(rng.integers(2 ** 31))
    else:                                                  # random.Random
        seed = rng.randrange(2 ** 31)
    if sampler == "sobol":
        from scipy.stats import qmc

        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)

        def draw(size: int) -> np.ndarray:
            return engine.random(size)
    else:
        generator = np.random.default_rng(seed)

        def draw(size: int) -> np.ndarray:
            return generator.random((size, dim))

    remaining = int(n)
    while remaining:
        size = min(batch_size, remaining)
        unit = draw(size)
        yield lower[None, :] + unit * (upper - lower)[None, :]
        remaining -= size


def get_seed(name: str):
    return SEEDS.get(name)


def list_seeds() -> List[tuple]:
    return SEEDS.describe()
