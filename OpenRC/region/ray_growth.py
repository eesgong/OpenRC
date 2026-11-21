r"""Region growth driven by a pluggable *ray strategy*.

Every construction method in the library does the same three things over and
over: choose a direction, find where the region's boundary lies along it
(one bisection), and decide what the boundary point does to the polytope.  The
facet-breaking builder (:mod:`OpenRC.region.expansive`) hard-wires the first
and the last: the direction is a facet's centroid, the point is added.  This
module separates them:

* a :class:`RayStrategy` **proposes** a ray - a direction and an action (add a
  vertex, or replace vertex ``i``) - and learns from the outcome;
* the engine (:func:`run_rays`) **executes** it: the boundary search (cached,
  warm-started), the exact volume the action would give (read off the live
  hull), the acceptance test, the vertex cap and the hard memory cap, and the
  bookkeeping.  It keeps the best state it has seen, so no strategy can lose a
  better region than the one it returns.

Registered ray strategies:

``facet``       add the boundary point behind the facet with the best score
                (cone volume by default, or any function of the facets) - the
                facet-selection idea on the live hull's real facets.
``gradient``    move vertices along the boundary up the exact volume gradient
                (see :mod:`OpenRC.region.refine`) and, while the vertex budget
                allows, *split* the most promising vertex into two - growth in
                the manner of the Linde-Buzo-Gray codebook algorithm.
``relocation``  the moves of ``gradient`` without splitting: improvement at a
                fixed vertex count.

A strategy of your own is a class with ``propose`` (and optionally
``observe``/``accept``); pass an instance, or register it in
:data:`RAY_STRATEGIES`.  :class:`RayContext` gives it everything it may need -
the hull, the volume gradient, facet directions and cone volumes - computed at
most once per hull state.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from OpenRC.geometry.live_hull import (
    LiveHull,
    cone_volumes,
    hull_volume_and_facets,
    live_hull_memory_mb,
    parallel_map,
)
from OpenRC.region.base import (BUILDERS, RegionBuilder, RegionResult, _RAY_OPTIONS, get_seed,
                                check_hull_budget, resolve_seed, single_oracle_guarantee)
from OpenRC.registry import Registry
from OpenRC.utils import LOGGER, MemoryBudgetExceeded, MemoryGuard

#: Registry of ray strategies.
RAY_STRATEGIES: Registry = Registry("ray strategy")


# --------------------------------------------------------------------------
# What a strategy proposes, and what happens to it
# --------------------------------------------------------------------------
@dataclass
class RayProposal:
    """One ray to fire and what to do with its boundary point.

    ``replace=None`` adds the point as a new vertex; an integer replaces the
    vertex at that position of ``ctx.vertices``.  ``hint`` is an optional
    initial boundary-distance estimate to warm-start ray search (an inaccurate
    hint adds at most two bracketing oracle calls without affecting accuracy).
    ``tag`` is handed back in the outcome.
    """

    direction: np.ndarray
    replace: Optional[int] = None
    hint: Optional[float] = None
    tag: Any = None


@dataclass
class RayOutcome:
    """What the engine did with a proposal."""

    proposal: RayProposal
    point: Optional[np.ndarray]
    accepted: bool
    volume_before: float
    volume_after: float            # volume the action gives (whether accepted or not)
    reason: str                    # accepted / no gain / vertex cap / memory cap / ray failed
    oracle_calls: int

    @property
    def gain(self) -> float:
        return self.volume_after - self.volume_before


class RayContext:
    """What a strategy may look at.  Derived quantities are cached per hull state.

    Attributes
    ----------
    live, axes, dim, rng:
        Read-only reference to the live hull, the region axes, dimension, and a
        :class:`numpy.random.Generator` for the strategy's own randomness.
    max_vertices, max_hull_mb:
        The caps the engine enforces; an ``add`` at the vertex cap, or one that
        would pass the memory cap, is refused.
    step, oracle_calls, trace:
        Moves executed so far, oracle calls so far, and the engine's trace.
    """

    def __init__(self, live: LiveHull, axes, *, rng: np.random.Generator,
                 max_vertices: int, max_hull_mb: Optional[float], oracle,
                 rays: Optional[dict] = None, ray_kwargs: Optional[dict] = None,
                 warm_start: bool = True, origin_checked: bool = False) -> None:
        if not isinstance(live, LiveHull):
            raise TypeError(f"live must be a LiveHull, got {type(live).__name__}")
        if not isinstance(warm_start, (bool, np.bool_)):
            raise ValueError("warm_start must be a boolean")
        if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
            raise TypeError("ray_kwargs must be a dict")
        if rays is not None and not isinstance(rays, dict):
            raise TypeError("rays must be a dict or None")
        if (isinstance(max_vertices, (bool, np.bool_))
                or not isinstance(max_vertices, (int, float, np.number))
                or not np.isfinite(max_vertices)
                or int(max_vertices) != max_vertices or max_vertices < 1):
            raise ValueError("max_vertices must be a positive integer")
        self.live = live
        self._rays = rays if rays is not None else {}
        self._ray_kwargs = dict(ray_kwargs or {})
        #: Where every ray starts; radii and directions are measured from it.
        raw_origin = self._ray_kwargs.get("origin")
        if raw_origin is None:
            self.origin = np.zeros(live.dim)
        else:
            arr_origin = np.asarray(raw_origin)
            if arr_origin.dtype == bool:
                raise ValueError(f"origin must be a finite ({live.dim},) vector, not boolean")
            self.origin = np.asarray(arr_origin, dtype=float)
            if self.origin.shape != (live.dim,) or not np.all(np.isfinite(self.origin)):
                raise ValueError(f"origin must be a finite ({live.dim},) vector")
        self._origin_checked = bool(origin_checked)
        self._warm_start = bool(warm_start)
        self.probes = 0
        self.axes = axes
        self.dim = live.dim
        self.rng = rng
        self.max_vertices = int(max_vertices)
        self.max_hull_mb = max_hull_mb
        self._oracle = oracle
        self.step = 0
        self.trace: List[dict] = []
        self.version = 0
        #: Set when an ``add`` was refused by the memory cap; cleared by the
        #: next accepted replacement (which may shrink the hull).
        self.memory_full = False
        self._cache: Dict[str, Any] = {}

    # --------------------------------------------------------- bookkeeping
    def _changed(self) -> None:
        self.version += 1
        self._cache.clear()

    def _cached(self, key: str, compute: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def oracle_calls(self) -> int:
        return int(self._oracle.stats.n_calls)

    def boundary(self, direction, hint: Optional[float] = None) -> Optional[np.ndarray]:
        """The boundary point along ``direction`` - a probe the strategy pays for.

        Shares the engine's ray cache (a later proposal in the same direction
        costs nothing) and counts toward every oracle budget.  Returns ``None``
        when the ray cannot be resolved (unbounded, solver failure).
        """
        from OpenRC.geometry.bisection import InfeasibleOriginError, ray_boundary

        raw_dir = np.asarray(direction)
        if raw_dir.dtype == bool:
            raise ValueError(f"direction must be a finite non-zero ({self.dim},) vector")
        direction = np.asarray(raw_dir, dtype=float)
        if (direction.shape != (self.dim,) or not np.all(np.isfinite(direction))
                or not np.linalg.norm(direction) > 0.0):
            raise ValueError(f"direction must be a finite non-zero ({self.dim},) vector")
        key = _ray_cache_key(self.origin, direction)
        if key not in self._rays:
            try:
                self._rays[key] = ray_boundary(
                    self._oracle, self.axes, direction,
                    hint=hint if self._warm_start else None,
                    check_origin=not self._origin_checked,
                    **self._ray_kwargs).boundary
                self._origin_checked = True
            except InfeasibleOriginError:
                raise
            except Exception:
                self._origin_checked = True
                return None
            self.probes += 1
        return self._rays[key]

    @property
    def room(self) -> bool:
        """Whether an ``add`` can still be accepted: vertex cap not reached and
        the memory cap has not refused the last attempt."""
        return self.live.n_vertices < self.max_vertices and not self.memory_full

    # ----------------------------------------------------------- geometry
    @property
    def vertices(self) -> np.ndarray:
        return self._cached("vertices", lambda: self.live.vertices)

    @property
    def radii(self) -> np.ndarray:
        """Distance of every vertex from the origin (the base point)."""
        return self._cached("radii", lambda: np.linalg.norm(self.vertices - self.origin, axis=1))

    @property
    def directions(self) -> np.ndarray:
        """Unit direction of every vertex."""
        return self._cached("directions", lambda: (self.vertices - self.origin)
                            / np.maximum(self.radii, 1e-300)[:, None])

    def volume_gradient(self) -> np.ndarray:
        """Exact ``d vol / d v_i`` for every vertex (see :mod:`OpenRC.region.refine`)."""
        from OpenRC.region.refine import volume_gradient

        return self._cached("gradient", lambda: volume_gradient(self.live))

    def tangent_gradient(self) -> np.ndarray:
        """The gradient's component tangent to the sphere of directions."""
        def compute():
            grad, u = self.volume_gradient(), self.directions
            return grad - np.sum(grad * u, axis=1)[:, None] * u
        return self._cached("tangent", compute)

    def facet_rows(self) -> np.ndarray:
        """``(n_facets, dim)`` vertex positions of every triangulated facet."""
        return self._cached("facets", self.live.facet_positions)

    def facet_directions(self) -> np.ndarray:
        """Unit direction of every facet's centroid (or outward normal when degenerate)."""
        def compute():
            centroids = self.vertices[self.facet_rows()].mean(axis=1) - self.origin
            norms = np.linalg.norm(centroids, axis=1)
            normals = self.live.equations[:, :-1]
            scale = max(float(self.radii.max(initial=1.0)), 1.0)
            bad = (norms <= 1e-12 * scale) | (np.sum(centroids * normals, axis=1) <= 0.0)
            dirs = np.empty_like(centroids)
            good = ~bad
            if np.any(good):
                dirs[good] = centroids[good] / norms[good, None]
            if np.any(bad):
                dirs[bad] = normals[bad] / np.maximum(
                    np.linalg.norm(normals[bad], axis=1, keepdims=True), 1e-300
                )
            return dirs
        return self._cached("facet_directions", compute)

    def predicted_gains(self) -> np.ndarray:
        """Volume each facet's ray is expected to add, before firing it.

        The radial function of the region is continuous, so the boundary along
        a facet's centroid direction is predicted at the mean radius ``r`` of the
        facet's vertices (all boundary points).  The hull reaches ``t`` along that
        direction, and the cap the new point would add over the facet is the
        facet's origin cone scaled by ``r / t - 1`` - the gap between the facet's
        chord and the arc through its vertices.  No oracle call is needed.
        """
        def compute():
            eq = self.live.equations
            scale = max(float(self.radii.max(initial=1.0)), 1.0)
            signed_h = -(eq[:, :-1] @ self.origin + eq[:, -1])
            cos = np.sum(eq[:, :-1] * self.facet_directions(), axis=1)
            r = self.radii[self.facet_rows()].mean(axis=1)
            ratio = np.zeros(len(eq), dtype=float)
            valid = (signed_h > 1e-12 * scale) & (cos > 1e-12)
            if np.any(valid):
                t = signed_h[valid] / cos[valid]
                ratio[valid] = np.maximum(r[valid] / t - 1.0, 0.0)
            return self.facet_cones() * ratio
        return self._cached("predicted_gains", compute)

    def facet_cones(self) -> np.ndarray:
        """Volume of the cone each facet forms with the origin."""
        def compute():
            return cone_volumes(self.live.points, self.live.simplices, self.origin)
        return self._cached("facet_cones", compute)


class RayStrategy(ABC):
    """Proposes rays; learns from what they gave.

    Implement :meth:`propose`.  Optional hooks: :meth:`reset` (start of a
    run), :meth:`observe` (after every proposal), :meth:`accept` (the
    acceptance rule; default: strictly more volume).
    """

    name: str = "strategy"

    def reset(self, ctx: RayContext) -> None:
        """Called once before the first proposal of a run."""

    @abstractmethod
    def propose(self, ctx: RayContext) -> Optional[RayProposal]:
        """The next ray, or ``None`` when the strategy has nothing left to try."""

    def observe(self, ctx: RayContext, outcome: RayOutcome) -> None:
        """Feedback after the engine executed a proposal."""

    def propose_batch(self, ctx: RayContext, k: int) -> List[RayProposal]:
        """Up to ``k`` proposals from the current state (``run_rays(batch=k)``).

        The default is one :meth:`propose`.  A strategy that overrides it must
        accept ``"outranked"`` outcomes in :meth:`observe`: improving proposals
        that lost to a better one of the same batch.
        """
        proposal = self.propose(ctx)
        return [] if proposal is None else [proposal]

    def accept(self, ctx: RayContext, outcome: RayOutcome) -> bool:
        """Whether to apply the action; the engine refuses caps before asking."""
        return outcome.volume_after > outcome.volume_before * (1.0 + 1e-12)

    def report(self) -> dict:
        return {"strategy": self.name}


def get_ray_strategy(name: str, **kwargs) -> RayStrategy:
    return RAY_STRATEGIES.get(name)(**kwargs)


def list_ray_strategies() -> List[tuple]:
    return RAY_STRATEGIES.describe()


def resolve_ray_strategy(strategy, **kwargs) -> RayStrategy:
    if isinstance(strategy, str):
        made = get_ray_strategy(strategy, **kwargs)
        if not isinstance(made, RayStrategy):
            raise TypeError(f"ray strategy {strategy!r} did not build a RayStrategy")
        return made
    if not isinstance(strategy, RayStrategy):
        raise TypeError("strategy must be a registered name or a RayStrategy instance")
    if kwargs:
        raise TypeError("strategy options apply only when the strategy is given by name")
    return strategy


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
@dataclass
class RayRun:
    """Result of :func:`run_rays`."""

    trace: List[dict] = field(default_factory=list)
    stopped_because: str = ""
    best_volume: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)


def _check_ray_kwargs(ray_kwargs: dict) -> None:
    """Validate ``ray_kwargs`` against supported ray-search options.

    Rejects unknown keywords as well as ``hint`` and ``check_origin``, which
    :func:`run_rays` manages internally (since ray-search exceptions are caught
    as ray failures during execution).
    """
    accepted = [name for name in _RAY_OPTIONS if name not in ("hint", "check_origin")]
    unknown = sorted(set(ray_kwargs) - set(accepted))
    if unknown:
        raise TypeError(f"ray_kwargs {unknown} cannot be used here (the engine sets hint and "
                        f"check_origin itself); accepted: {', '.join(accepted)}")


def _direction_key(direction: np.ndarray) -> tuple:
    return tuple(np.round(direction / np.linalg.norm(direction), 12))


def _ray_cache_key(origin: np.ndarray, direction: np.ndarray) -> tuple:
    dir_key = _direction_key(direction)
    if not np.any(origin):
        return dir_key
    return (tuple(np.round(origin, 12)), dir_key)


def run_rays(live: LiveHull, oracle, axes, strategy: RayStrategy, *,
             max_moves: Optional[int] = None, max_vertices: int = 10 ** 9,
             max_hull_mb: Optional[float] = None, max_oracle_calls: Optional[int] = None,
             max_seconds: Optional[float] = None, volume_tol: Optional[float] = None,
             patience: int = 25, ray_kwargs: Optional[dict] = None,
             ray_cache: Optional[dict] = None, tabu: Optional[set] = None,
             monitor: Optional[Callable[[Any], Optional[str]]] = None,
             guard: Optional[MemoryGuard] = None, rng: Optional[np.random.Generator] = None,
             warm_start: bool = True, max_refused: Optional[int] = None,
             first_order_screen: bool = False,
             state_fn: Optional[Callable[[RayContext], Any]] = None,
             batch: int = 1, origin_checked: bool = False) -> RayRun:
    """Let ``strategy`` improve ``live`` in place, one ray at a time.

    The volume of the returned hull is the best the run reached: if a custom
    acceptance rule accepted a loss that was never recovered, the best vertex
    set is restored at the end.

    ``first_order_screen`` is an opt-in speed heuristic for replacements.  It
    refuses a candidate when ``grad_i . (p - v_i) <= 0`` without rebuilding
    the hull.  This local derivative is not a global upper bound: a far-away
    candidate can change the hull's facets and still improve the exact volume.
    The default is therefore ``False``; enable it only when missed improvements
    are an acceptable trade-off.  Every candidate that is not screened is
    accepted or rejected using its exact replacement volume.

    ``batch > 1`` asks the strategy for up to that many proposals at once
    (:meth:`RayStrategy.propose_batch`), fires their rays, measures the exact
    replacement hulls in parallel threads (Qhull releases the interpreter
    lock) and applies the best acceptable one; the others are reported as
    ``"outranked"``.  In high dimension, where one replacement hull costs tens
    of milliseconds, this trades oracle calls for wall-clock time.  The default
    ``1`` fires one ray at a time.
    """
    if not isinstance(warm_start, (bool, np.bool_)):
        raise ValueError("warm_start must be a boolean")
    if not isinstance(first_order_screen, (bool, np.bool_)):
        raise ValueError("first_order_screen must be a boolean")
    if (isinstance(batch, (bool, np.bool_))
            or not isinstance(batch, (int, float, np.number))
            or not np.isfinite(batch)
            or int(batch) != batch or batch < 1):
        raise ValueError("batch must be a positive integer")
    if (max_moves is not None
            and (isinstance(max_moves, (bool, np.bool_))
                 or not isinstance(max_moves, (int, float, np.number))
                 or not np.isfinite(max_moves)
                 or int(max_moves) != max_moves or max_moves < 0)):
        raise ValueError("max_moves must be a non-negative integer when provided")
    if (isinstance(max_vertices, (bool, np.bool_))
            or not isinstance(max_vertices, (int, float, np.number))
            or not np.isfinite(max_vertices)
            or int(max_vertices) != max_vertices or max_vertices < 1):
        raise ValueError("max_vertices must be a positive integer")
    if (max_oracle_calls is not None
            and (isinstance(max_oracle_calls, (bool, np.bool_))
                 or not isinstance(max_oracle_calls, (int, float, np.number))
                 or not np.isfinite(max_oracle_calls)
                 or int(max_oracle_calls) != max_oracle_calls or max_oracle_calls <= 0)):
        raise ValueError("max_oracle_calls must be a positive integer when provided")
    if max_seconds is not None and (
        isinstance(max_seconds, (bool, np.bool_))
        or not isinstance(max_seconds, (int, float, np.number))
        or not np.isfinite(max_seconds)
        or max_seconds <= 0.0
    ):
        raise ValueError("max_seconds must be finite and positive when provided")
    if volume_tol is not None and (
        isinstance(volume_tol, (bool, np.bool_))
        or not isinstance(volume_tol, (int, float, np.number))
        or not np.isfinite(volume_tol)
        or volume_tol < 0.0
    ):
        raise ValueError("volume_tol must be finite and non-negative when provided")
    if max_hull_mb is not None and (
        isinstance(max_hull_mb, (bool, np.bool_))
        or not isinstance(max_hull_mb, (int, float, np.number))
        or not (np.isfinite(max_hull_mb) and max_hull_mb > 0.0)
    ):
        raise ValueError("max_hull_mb must be finite and positive when provided")
    if (isinstance(patience, (bool, np.bool_))
            or not isinstance(patience, (int, float, np.number))
            or not np.isfinite(patience)
            or int(patience) != patience or patience <= 0):
        raise ValueError("patience must be a positive integer")
    if (max_refused is not None
            and (isinstance(max_refused, (bool, np.bool_))
                 or not isinstance(max_refused, (int, float, np.number))
                 or not np.isfinite(max_refused)
                 or int(max_refused) != max_refused or max_refused <= 0)):
        raise ValueError("max_refused must be a positive integer when provided")
    if ray_cache is not None and not isinstance(ray_cache, dict):
        raise TypeError("ray_cache must be a dict or None")
    if monitor is not None and not callable(monitor):
        raise TypeError("monitor must be callable or None")
    if state_fn is not None and not callable(state_fn):
        raise TypeError("state_fn must be callable or None")
    from OpenRC.geometry.bisection import InfeasibleOriginError, ray_boundary
    from OpenRC.region.expansive import point_key

    if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
        raise TypeError("ray_kwargs must be a dict")
    ray_kwargs = dict(ray_kwargs or {})
    _check_ray_kwargs(ray_kwargs)
    if not isinstance(live, LiveHull):
        raise TypeError(f"live must be a LiveHull, got {type(live).__name__}")
    if axes is None or len(axes) != live.dim:
        raise ValueError(
            f"axes must have length {live.dim} to match the live hull, "
            f"got {None if axes is None else len(axes)}"
        )
    if not isinstance(strategy, RayStrategy):
        raise TypeError(f"strategy must be a RayStrategy, got {type(strategy).__name__}")
    check_hull_budget(live, max_hull_mb)
    rays = ray_cache if ray_cache is not None else {}
    ctx = RayContext(live, axes, rng=rng if rng is not None else np.random.default_rng(0),
                     max_vertices=max_vertices, max_hull_mb=max_hull_mb, oracle=oracle,
                     rays=rays, ray_kwargs=ray_kwargs, warm_start=bool(warm_start),
                     origin_checked=origin_checked)
    strategy.reset(ctx)
    start = time.perf_counter()
    calls_at_start = oracle.stats.n_calls
    run = RayRun(counts={"moves": 0, "accepted": 0, "adds": 0, "replaces": 0,
                         "refused": 0, "ray_failures": 0, "cached_rays": 0, "tabu": 0})
    best_volume, best_vertices = live.volume, None
    refused_in_a_row = 0
    limit_refused = int(max_refused) if max_refused is not None else 4 * live.dim + 10
    stagnant, last_volume = 0, live.volume

    def fire(proposal: RayProposal):
        """The boundary point along the proposal's direction (cached), or a reason."""
        direction = np.asarray(proposal.direction, dtype=float).ravel()
        key = _ray_cache_key(ctx.origin, direction)
        if key in rays:
            run.counts["cached_rays"] += 1
            return rays[key], ""
        try:
            hint = proposal.hint if warm_start else None
            point = ray_boundary(oracle, axes, direction, hint=hint,
                                 check_origin=not ctx._origin_checked,
                                 **ray_kwargs).boundary
            ctx._origin_checked = True
            rays[key] = point
            return point, ""
        except InfeasibleOriginError:
            raise
        except Exception as exc:          # unbounded ray, solver failure ...
            ctx._origin_checked = True
            run.counts["ray_failures"] += 1
            return None, f"ray failed: {type(exc).__name__}"

    def assess(proposal: RayProposal, point, reason: str, volume_before: float) -> dict:
        """What the action would give, short of the exact replacement hull."""
        item = {"point": point, "reason": reason, "volume_after": volume_before,
                "replacement_mb": None, "candidate": None}
        if point is None or reason:
            return item
        if tabu and point_key(point) in tabu:
            run.counts["tabu"] += 1
            item["reason"] = "tabu point"
        elif proposal.replace is None:
            if not ctx.room:
                item["reason"] = "vertex cap"
            elif max_hull_mb is not None and live.memory_after_mb(point) > max_hull_mb:
                item["reason"] = "memory cap"
                ctx.memory_full = True       # strategies see ``room`` turn False
            else:
                item["volume_after"] = volume_before + live.gain(point)
        else:
            i = int(proposal.replace)
            if live.gain(point) <= 0.0:
                item["reason"] = "no gain"   # an interior point can only lose v_i's cap
            elif first_order_screen and float(
                    ctx.volume_gradient()[i] @ (point - ctx.vertices[i])) <= 0.0:
                item["reason"] = "no first-order gain"
            else:
                item["candidate"] = np.vstack([np.delete(ctx.vertices, i, axis=0), point])
        return item

    run.stopped_because = "move budget used"
    while max_moves is None or ctx.step < max_moves:
        if max_seconds is not None and time.perf_counter() - start > max_seconds:
            run.stopped_because = "time budget reached"
            break
        if max_oracle_calls is not None and oracle.stats.n_calls - calls_at_start >= max_oracle_calls:
            run.stopped_because = "oracle-call budget reached"
            break
        if guard is not None:
            try:
                guard.check()
            except MemoryBudgetExceeded as exc:
                run.stopped_because = f"memory budget reached ({exc})"
                break
        if monitor is not None and state_fn is not None:
            reason = monitor(state_fn(ctx))
            if reason:
                run.stopped_because = str(reason)
                break

        k = 1 if batch == 1 else batch
        if max_moves is not None:
            k = max(1, min(k, max_moves - ctx.step))
        if k == 1:
            single = strategy.propose(ctx)
            proposals = [] if single is None else [single]
        else:
            proposals = [p for p in strategy.propose_batch(ctx, k) if p is not None][:k]
        if not proposals:
            # A strategy that stops because ``room`` turned False on the memory
            # cap must say so in the wording the memory-aware builder triggers on.
            run.stopped_because = (
                f"memory cap: strategy {strategy.name!r} stopped after an insertion was "
                f"refused by the {max_hull_mb} MiB cap" if ctx.memory_full
                else f"strategy {strategy.name!r} has nothing left to propose")
            break
        for proposal in proposals:
            raw_dir = np.asarray(proposal.direction)
            if raw_dir.dtype == bool:
                raise ValueError(f"strategy {strategy.name!r} proposed an invalid direction "
                                 f"{proposal.direction!r}")
            direction = np.asarray(raw_dir, dtype=float)
            if direction.shape != (live.dim,) or not np.all(np.isfinite(direction)) \
                    or not np.linalg.norm(direction) > 0:
                raise ValueError(f"strategy {strategy.name!r} proposed an invalid direction "
                                 f"{direction!r}")
            if proposal.replace is not None:
                if (isinstance(proposal.replace, (bool, np.bool_))
                        or not isinstance(proposal.replace, (int, np.integer))
                        or not 0 <= int(proposal.replace) < live.n_vertices):
                    raise ValueError(f"strategy {strategy.name!r} proposed to replace vertex "
                                     f"{proposal.replace!r} of {live.n_vertices}")

        # ---- fire every ray, then assess what each action would give --------
        volume_before = live.volume
        items = []
        for proposal in proposals:
            ctx.step += 1
            run.counts["moves"] += 1
            calls = oracle.stats.n_calls
            point, reason = fire(proposal)
            item = assess(proposal, point, reason, volume_before)
            item.update(proposal=proposal, calls=oracle.stats.n_calls - calls, step=ctx.step)
            items.append(item)

        # ---- exact replacement volumes: independent hulls, threaded ---------
        pending = [item for item in items if item["candidate"] is not None]
        measured = parallel_map(hull_volume_and_facets,
                                [item["candidate"] for item in pending])
        for item, result in zip(pending, measured):
            if result is None:
                item["reason"] = "unmeasurable replacement"
            else:
                item["volume_after"] = result[0]
                item["replacement_mb"] = live_hull_memory_mb(len(item["candidate"]), live.dim,
                                                             result[1])

        # ---- decide: the best acceptable action is applied ------------------
        outcomes = []
        acceptable = []
        for item in items:
            outcome = RayOutcome(proposal=item["proposal"], point=item["point"], accepted=False,
                                 volume_before=volume_before,
                                 volume_after=item["volume_after"],
                                 reason=item["reason"] or "no gain",
                                 oracle_calls=item["calls"])
            outcomes.append(outcome)
            if item["reason"]:
                continue
            if (item["replacement_mb"] is not None and max_hull_mb is not None
                    and item["replacement_mb"] > max_hull_mb):
                outcome.reason = "memory cap"
            elif strategy.accept(ctx, outcome):
                acceptable.append(outcome)
        if acceptable:
            chosen = max(acceptable, key=lambda o: o.volume_after)
            proposal, point = chosen.proposal, chosen.point
            assert point is not None
            if chosen.volume_after < volume_before and volume_before >= best_volume:
                best_vertices = ctx.vertices.copy()     # the best state, before a loss
            undone = False
            if proposal.replace is None:
                rollback = ctx.vertices.copy() if max_hull_mb is not None else None
                live.add(point)
                if (rollback is not None and max_hull_mb is not None
                        and live.memory_mb() > max_hull_mb):
                    # facets_after() is exact only in general position: undo.
                    live.reset(rollback)
                    ctx.memory_full = True
                    undone = True
                else:
                    run.counts["adds"] += 1
            else:
                kept = np.delete(ctx.vertices, int(proposal.replace), axis=0)
                live.reset(np.vstack([kept, point]))
                run.counts["replaces"] += 1
                ctx.memory_full = False
            if undone:
                chosen.reason = "memory cap"
            else:
                chosen.accepted, chosen.reason = True, "accepted"
                run.counts["accepted"] += 1
            ctx._changed()
            for other in acceptable:
                if other is not chosen:
                    other.reason = "outranked"      # improving, but another was better

        # ---- feedback, trace and stopping rules, one row per proposal -------
        stop = False
        for item, outcome in zip(items, outcomes):
            if not outcome.accepted and outcome.reason in ("vertex cap", "memory cap"):
                run.counts["refused"] += 1
                refused_in_a_row += 1
            else:
                refused_in_a_row = 0
            strategy.observe(ctx, outcome)
            if live.volume > best_volume:
                best_volume, best_vertices = live.volume, None
            proposal = outcome.proposal
            row = {"step": item["step"],
                   "action": "add" if proposal.replace is None else "replace",
                   "accepted": outcome.accepted, "reason": outcome.reason,
                   "volume": live.volume, "gain": outcome.gain,
                   "n_vertices": live.n_vertices, "n_hull_vertices": live.n_vertices,
                   "n_facets": live.n_facets, "hull_mb": live.memory_mb(),
                   "added": max(live.volume - volume_before, 0.0) if outcome.accepted else 0.0,
                   "oracle_calls": oracle.stats.n_calls,
                   "seconds": time.perf_counter() - start, "iteration": item["step"]}
            ctx.trace.append(row)
            if refused_in_a_row >= limit_refused:
                # Worded as the other builders word these stops, so that a
                # memory-aware caller recognises them as its triggers.
                run.stopped_because = (
                    f"max_vertices reached ({refused_in_a_row} proposals in a row needed "
                    "another vertex)" if outcome.reason == "vertex cap" else
                    f"memory cap: {refused_in_a_row} proposals in a row would have passed "
                    f"{max_hull_mb} MiB")
                stop = True
                break
            if volume_tol is not None:
                gain = ((live.volume - last_volume) / last_volume if last_volume > 0
                        else math.inf)
                stagnant = stagnant + 1 if gain < volume_tol else 0
                last_volume = live.volume
                if stagnant >= patience:
                    run.stopped_because = (f"volume converged (< {volume_tol} for "
                                           f"{patience} moves)")
                    stop = True
                    break
        if stop:
            break

    if best_vertices is not None and live.volume < best_volume:
        live.reset(best_vertices)                       # restore highest-volume state
    run.trace = ctx.trace
    run.best_volume = live.volume
    return run


# --------------------------------------------------------------------------
# Built-in strategies
# --------------------------------------------------------------------------
@RAY_STRATEGIES.register(
    "facet",
    doc="Add the boundary point behind the facet with the largest predicted gain.",
)
class FacetRays(RayStrategy):
    """The facet-selection idea as a ray strategy, on the live hull's real facets.

    Each proposal scores every facet, skips facets whose ray has already been
    fired, and adds the boundary point behind the best one.  Scores:
    ``"predicted"`` (default) - the gain :meth:`RayContext.predicted_gains`
    expects without firing; ``"cone"`` - the facet-to-origin cone, which
    favours facets already close to the boundary and is a poor guide;
    ``"area"``; or a function ``score(ctx) -> array`` of your own.  The
    richer, stateful facet policies (bandits, generations) live in
    :class:`~OpenRC.region.expansive.ExpansiveBuilder`.
    """

    name = "facet"

    def __init__(self, *, score: Any = "predicted", lookahead: int = 3) -> None:
        if not (callable(score) or score in ("predicted", "cone", "area")):
            raise ValueError("score must be 'predicted', 'cone', 'area' or a callable")
        if (isinstance(lookahead, (bool, np.bool_))
                or not isinstance(lookahead, (int, float, np.number))
                or not np.isfinite(lookahead)
                or int(lookahead) != lookahead or int(lookahead) < 1):
            raise ValueError("lookahead must be a positive integer")
        self.score = score
        self.lookahead = int(lookahead)
        self._fired: set = set()

    def reset(self, ctx: RayContext) -> None:
        self._fired = set()

    def _scores(self, ctx: RayContext) -> np.ndarray:
        if callable(self.score):
            return np.asarray(self.score(ctx), dtype=float)
        if self.score == "predicted":
            return ctx.predicted_gains()
        if self.score == "cone":
            return ctx.facet_cones()
        # area of each (d-1)-facet: its cone from the centroid over the height
        eq = ctx.live.equations
        c = ctx.vertices.mean(axis=0)
        heights = np.abs(eq[:, :-1] @ c + eq[:, -1])
        return (cone_volumes(ctx.live.points, ctx.live.simplices, c) * ctx.dim
                / np.maximum(heights, 1e-300))

    def propose(self, ctx: RayContext) -> Optional[RayProposal]:
        """The best of the ``lookahead`` best-scoring unfired facets, by realised gain.

        With ``lookahead > 1`` the rays of the short list are probed through
        :meth:`RayContext.boundary` (cached, so the chosen one is not paid for
        twice) and the facet yielding the largest realised volume gain is
        selected, so the heuristic score only needs to order candidates.  Probed facets that lose stay
        unfired: their rays are cached for later.
        """
        if not ctx.room:
            return None
        scores = self._scores(ctx)
        directions = ctx.facet_directions()
        rows = ctx.facet_rows()
        shortlist = []
        for f in np.argsort(-scores, kind="stable"):
            if _direction_key(directions[f]) not in self._fired:
                shortlist.append(int(f))
                if len(shortlist) == self.lookahead:
                    break
        if not shortlist:
            return None
        best, best_gain = shortlist[0], -1.0
        if len(shortlist) > 1:
            for f in shortlist:
                point = ctx.boundary(directions[f], hint=float(ctx.radii[rows[f]].mean()))
                gain = ctx.live.gain(point) if point is not None else -1.0
                if gain > best_gain:
                    best, best_gain = f, gain
        self._fired.add(_direction_key(directions[best]))
        return RayProposal(direction=directions[best], hint=float(ctx.radii[rows[best]].mean()))


@RAY_STRATEGIES.register(
    "gradient",
    doc="Move vertices up the volume gradient along the boundary; split vertices "
        "while the budget allows (LBG-style growth).",
)
class GradientRays(RayStrategy):
    """Boundary moves up the volume gradient, with splitting for growth.

    A *move* rotates vertex ``i``'s direction by its step angle towards the
    tangential volume gradient and replaces the vertex with the boundary point
    there, if that raises the volume.  A *split* adds that point instead of
    replacing - the vertex and its rotated copy both stay and then move apart,
    as codewords do in the Linde-Buzo-Gray algorithm.  While the vertex budget
    has room, every ``split_every``-th proposal is a split; with
    ``split_every=None`` the strategy only moves (``"relocation"``).

    The vertex to move is the one whose tangent gradient promises most
    (``|tangent| x radius x step``).  Each vertex has its own step angle,
    adapted by Rechenberg's 1/5 success rule; the strategy is done when every
    step is below ``min_step`` and it may not split - the first-order form of
    Property Z at that resolution.  Rays are warm-started at the moving
    vertex's radius.  At a symmetric stationary point (every tangent gradient
    zero) a split goes along a random tangent direction, which breaks the
    symmetry as the LBG algorithm's perturbed split does.
    """

    name = "gradient"

    def __init__(self, *, step: float = 0.2, min_step: float = 0.01, grow: float = 1.5,
                 split_every: Optional[int] = 3, split_step: float = 0.3) -> None:
        if (isinstance(min_step, (bool, np.bool_)) or isinstance(step, (bool, np.bool_))
                or not isinstance(min_step, (int, float, np.number))
                or not isinstance(step, (int, float, np.number))
                or not (math.isfinite(min_step) and math.isfinite(step)
                        and 0.0 < min_step < step < math.pi / 2)):
            raise ValueError("need 0 < min_step < step < pi/2")
        if (isinstance(grow, (bool, np.bool_))
                or not isinstance(grow, (int, float, np.number))
                or not (math.isfinite(grow) and grow > 1.0)):
            raise ValueError("grow must exceed 1")
        if (split_every is not None
                and (isinstance(split_every, (bool, np.bool_))
                     or not isinstance(split_every, (int, float, np.number))
                     or not np.isfinite(split_every)
                     or int(split_every) != split_every or int(split_every) < 1)):
            raise ValueError("split_every must be a positive integer or None")
        if (isinstance(split_step, (bool, np.bool_))
                or not isinstance(split_step, (int, float, np.number))
                or not (math.isfinite(split_step) and 0.0 < split_step < math.pi / 2)):
            raise ValueError("split_step must be finite and in (0, pi/2)")
        self.step = float(step)
        self.min_step = float(min_step)
        self.grow = float(grow)
        self.split_every = None if split_every is None else int(split_every)
        self.split_step = float(split_step)
        self._steps: Dict[tuple, float] = {}
        self._since_split = 0

    def reset(self, ctx: RayContext) -> None:
        self._steps = {}
        self._since_split = 0

    def _theta(self, ctx: RayContext) -> np.ndarray:
        from OpenRC.region.expansive import point_key

        return np.array([self._steps.setdefault(point_key(v), self.step)
                         for v in ctx.vertices])

    def propose(self, ctx: RayContext) -> Optional[RayProposal]:
        theta = self._theta(ctx)
        tangent = ctx.tangent_gradient()
        norms = np.linalg.norm(tangent, axis=1)
        promise = norms * ctx.radii * theta
        splitting = (self.split_every is not None and ctx.room
                     and self._since_split + 1 >= self.split_every)
        if not splitting:
            promise = np.where(theta >= self.min_step, promise, -1.0)
        i = int(np.argmax(promise))
        if promise[i] <= 0.0 or norms[i] == 0.0:
            if self.split_every is not None and ctx.room:
                splitting, i = True, int(np.argmax(norms * ctx.radii))
            else:
                return None
        u = ctx.directions[i]
        if norms[i] > 1e-12 * max(float(norms.max()), 1e-300):
            t = tangent[i] / norms[i]
        elif splitting:
            # A symmetric configuration (e.g. the axis seed of a symmetric
            # region) is a stationary point: every tangent gradient vanishes.
            # Split along a random tangent direction instead, as the LBG
            # algorithm perturbs codewords, to break the symmetry.
            if ctx.dim == 1:
                # A one-dimensional sphere has no tangent direction and its
                # two boundary points already form the complete interval.
                return None
            for _ in range(4):
                noise = ctx.rng.normal(size=ctx.dim)
                noise -= (noise @ u) * u
                noise_norm = float(np.linalg.norm(noise))
                if noise_norm > 1e-14:
                    t = noise / noise_norm
                    break
            else:  # degenerate random projection; skip rather than emit NaN
                return None
        else:
            return None
        angle = self.split_step if splitting else float(theta[i])
        direction = math.cos(angle) * u + math.sin(angle) * t
        return RayProposal(direction=direction, replace=None if splitting else i,
                           hint=float(ctx.radii[i]),
                           tag=("split" if splitting else "move", float(theta[i]),
                                ctx.vertices[i].copy()))

    def propose_batch(self, ctx: RayContext, k: int) -> List[RayProposal]:
        """Moves for the ``k`` most promising vertices; a due split goes alone."""
        splitting = (self.split_every is not None and ctx.room
                     and self._since_split + 1 >= self.split_every)
        if k <= 1 or splitting:
            return super().propose_batch(ctx, k)
        theta = self._theta(ctx)
        tangent = ctx.tangent_gradient()
        norms = np.linalg.norm(tangent, axis=1)
        promise = np.where(theta >= self.min_step, norms * ctx.radii * theta, -1.0)
        order = [int(i) for i in np.argsort(-promise)[:k]
                 if promise[i] > 0.0 and norms[i] > 1e-12 * max(float(norms.max()), 1e-300)]
        if not order:
            return super().propose_batch(ctx, k)
        proposals = []
        for i in order:
            u, t = ctx.directions[i], tangent[i] / norms[i]
            angle = float(theta[i])
            proposals.append(RayProposal(
                direction=math.cos(angle) * u + math.sin(angle) * t, replace=i,
                hint=float(ctx.radii[i]), tag=("move", angle, ctx.vertices[i].copy())))
        return proposals

    def observe(self, ctx: RayContext, outcome: RayOutcome) -> None:
        from OpenRC.region.expansive import point_key

        if outcome.reason == "outranked":
            return                    # it would have improved: no step change
        kind, theta, origin_vertex = outcome.proposal.tag
        if kind == "split":
            self._since_split = 0
            if outcome.accepted and outcome.point is not None:
                self._steps[point_key(outcome.point)] = self.step
            return
        self._since_split += 1
        if outcome.accepted and outcome.point is not None:
            self._steps[point_key(outcome.point)] = theta * self.grow
        else:
            self._steps[point_key(origin_vertex)] = theta / self.grow ** 0.25

    def report(self) -> dict:
        return {"strategy": self.name, "step": self.step, "min_step": self.min_step,
                "split_every": self.split_every}


@RAY_STRATEGIES.register(
    "relocation",
    doc="Gradient moves only: improve the region at a fixed vertex count.",
)
class RelocationRays(GradientRays):
    """:class:`GradientRays` without splitting."""

    name = "relocation"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("split_every", None)
        super().__init__(**kwargs)


@RAY_STRATEGIES.register(
    "facet-queue",
    doc="The expansive builder's facet queue and facet-selection policies, as a ray strategy.",
)
class FacetQueueRays(RayStrategy):
    """Facet breaking with the facet queue of :class:`~OpenRC.region.expansive.ExpansiveBuilder`.

    One difference: the engine refuses a point that adds no volume, and the
    facet is then retired without children, where the expansive builder keeps
    such a point and splits the facet when its bulge exceeds ``min_bulge``.

    The strategy keeps the expansive builder's *virtual* facet queue: a broken
    facet is replaced by its ``d`` children through the new point, whose
    centroids subdivide the directions under the old facet.  With a policy
    that favours old facets (``generation``, or the ``adaptive-*`` bandits that
    include it), the queue refines directional coverage breadth-first.  Any registered
    :class:`~OpenRC.region.facet_policy.FacetPolicy`, or an instance, can be
    used.  The strategy is done when the vertex budget is full or the queue is
    empty.  ``refresh_gains`` (default ``True``) re-measures cached gains
    after every accepted point, as :class:`~OpenRC.region.expansive.ExpansiveBuilder`
    does; ``False`` keeps the gain measured when a facet was first scored, as
    the published method did.
    """

    name = "facet-queue"

    def __init__(self, *, policy: Any = "adaptive-ema", policy_kwargs: Optional[dict] = None,
                 min_bulge: float = 1e-3, refresh_gains: bool = True) -> None:
        from OpenRC.region.facet_policy import FacetPolicy, get_facet_policy

        if not isinstance(refresh_gains, (bool, np.bool_)):
            raise ValueError("refresh_gains must be a boolean")
        if policy_kwargs is not None and not isinstance(policy_kwargs, dict):
            raise TypeError("policy_kwargs must be a dict")
        self.refresh_gains = bool(refresh_gains)
        self._policy_arg = policy
        self.policy_kwargs = dict(policy_kwargs or {})
        if isinstance(policy, str):
            self.policy = get_facet_policy(policy, **self.policy_kwargs)
        elif not isinstance(policy, FacetPolicy):
            raise TypeError(f"policy must be a registered name or a FacetPolicy; got {policy!r}")
        elif self.policy_kwargs:
            raise TypeError("policy_kwargs apply only when policy is given by name")
        else:
            self.policy = policy
        if (isinstance(min_bulge, (bool, np.bool_))
                or not isinstance(min_bulge, (int, float, np.number))
                or not (np.isfinite(min_bulge) and min_bulge >= 0.0)):
            raise ValueError("min_bulge must be finite and non-negative")
        self.min_bulge = float(min_bulge)

    def reset(self, ctx: RayContext) -> None:
        from OpenRC.region.expansive import _Facet

        self.policy.reset()
        self._origin = ctx.origin
        self._rng = random.Random(int(ctx.rng.integers(2 ** 62)))
        self._points: List[np.ndarray] = [np.asarray(v, dtype=float) for v in ctx.vertices]
        self._facets = [_Facet(fid=i, vertex_idx=tuple(int(v) for v in row), generation=0)
                        for i, row in enumerate(ctx.facet_rows())]
        self._by_fid = {f.fid: f for f in self._facets}
        self._next = len(self._facets)
        self._candidate: Dict[int, Optional[np.ndarray]] = {}
        self._gain: Dict[int, float] = {}
        self._measures: Dict[str, Dict[int, float]] = {"area": {}, "cone": {}}
        self._pending: Optional[tuple] = None
        # Refreshed gains: what an insertion changes, measured cheaply (see
        # the expansive builder); without refresh, gains come straight from
        # LiveHull.gain.
        from OpenRC.region.expansive import _GainMeter

        self._measure = _GainMeter(ctx.live).gain if self.refresh_gains else ctx.live.gain
        self._delta: Any = None

    def _centroid(self, facet) -> np.ndarray:
        from OpenRC.region.expansive import _outward_normal

        pts = np.array([self._points[i] for i in facet.vertex_idx]) - self._origin
        centroid = pts.mean(axis=0)
        return centroid if np.linalg.norm(centroid) > 0.0 else _outward_normal(pts)

    def _retire(self, position: int) -> None:
        facet = self._facets.pop(position)
        del self._by_fid[facet.fid]
        self._gain.pop(facet.fid, None)
        self._candidate.pop(facet.fid, None)
        for memo in self._measures.values():
            memo.pop(facet.fid, None)

    def propose(self, ctx: RayContext) -> Optional[RayProposal]:
        from OpenRC.region.facet_policy import SelectionContext

        def gain_of(fid: int) -> float:
            if fid not in self._gain and fid in self._candidate:
                point = self._candidate[fid]      # searched already; the hull moved
                self._gain[fid] = self._measure(point) if point is not None else 0.0
            if fid not in self._gain:
                facet = self._by_fid[fid]
                hint = float(np.mean([np.linalg.norm(self._points[i] - self._origin)
                                      for i in facet.vertex_idx]))
                point = ctx.boundary(self._centroid(facet), hint=hint)
                self._candidate[fid] = point
                self._gain[fid] = self._measure(point) if point is not None else 0.0
            return self._gain[fid]

        while ctx.room and self._facets:
            selection = SelectionContext(
                vertices=np.array(self._points) - self._origin,
                facets=[f.vertex_idx for f in self._facets],
                generations=[f.generation for f in self._facets],
                facet_ids=[f.fid for f in self._facets],
                gain_fn=gain_of, rng=self._rng, cache=self._measures)
            position = self.policy.select(selection)
            chosen = self._facets[position]
            gain = gain_of(chosen.fid)
            self.policy.observe(selection, position, gain)
            point = self._candidate.get(chosen.fid)
            if point is None:                               # the ray failed: retire it
                self._retire(position)
                continue
            self._pending = (chosen, point)
            self._delta = None              # never one left over from a refused point
            if self.refresh_gains and self._gain:
                from OpenRC.region.expansive import _InsertionDelta

                self._delta = _InsertionDelta(ctx.live, point)
            return RayProposal(direction=self._centroid(chosen),
                               hint=float(np.linalg.norm(point - self._origin)), tag="facet")
        return None

    def observe(self, ctx: RayContext, outcome: RayOutcome) -> None:
        from OpenRC.region.expansive import _Facet, _distance_to_hyperplane

        if self._pending is None:
            return
        chosen, point = self._pending
        self._pending = None
        facet_points = np.array([self._points[i] for i in chosen.vertex_idx])
        self._retire(self._facets.index(chosen))
        # A refused point is not in the real hull.  Retire the attempted facet,
        # but do not let that point enter the virtual queue or parent facets
        # whose vertex indices have no counterpart in ``ctx.live``.
        if not outcome.accepted:
            return
        delta, self._delta = self._delta, None
        if self.refresh_gains and delta is not None:
            # Gains of failed rays (no candidate) stay 0; the others are
            # re-measured only where the insertion changed what they see.
            fids = [f for f in self._gain if self._candidate.get(f) is not None]
            if fids:
                stale = delta.stale(ctx.live, np.array([self._candidate[f] for f in fids]))
                for fid, is_stale in zip(fids, stale):
                    if is_stale:
                        del self._gain[fid]
        elif self.refresh_gains:
            self._gain.clear()
        self._points.append(np.asarray(point, dtype=float))
        new_index = len(self._points) - 1
        if _distance_to_hyperplane(facet_points, point) > self.min_bulge:
            for dropped in chosen.vertex_idx:
                child = tuple(v for v in chosen.vertex_idx if v != dropped) + (new_index,)
                facet = _Facet(self._next, child, chosen.generation + 1)
                self._facets.append(facet)
                self._by_fid[self._next] = facet
                self._next += 1

    def report(self) -> dict:
        return {"strategy": self.name, "policy": self.policy.report(),
                "min_bulge": self.min_bulge, "refresh_gains": self.refresh_gains}


class SequentialRays(RayStrategy):
    """Run strategies one after another.

    Each stage proposes until it returns ``None``; the next stage then starts
    (its :meth:`~RayStrategy.reset` is called on the hull as it is by then).
    Outcomes and acceptance go to the stage that made the proposal.
    """

    name = "sequence"

    def __init__(self, stages: List[Any], *, name: Optional[str] = None) -> None:
        if isinstance(stages, (str, bytes, bytearray)):
            raise TypeError("stages must be a sequence of ray strategies, not a string")
        if not stages:
            raise ValueError("a sequence needs at least one stage")
        self.stages = [resolve_ray_strategy(s) for s in stages]
        if name:
            self.name = name
        self._index = 0

    def reset(self, ctx: RayContext) -> None:
        self._index = 0
        self.stages[0].reset(ctx)

    def propose(self, ctx: RayContext) -> Optional[RayProposal]:
        while self._index < len(self.stages):
            proposal = self.stages[self._index].propose(ctx)
            if proposal is not None:
                return proposal
            self._index += 1
            if self._index < len(self.stages):
                self.stages[self._index].reset(ctx)
        return None

    def propose_batch(self, ctx: RayContext, k: int) -> List[RayProposal]:
        """The active stage's batch; an exhausted stage hands over as in :meth:`propose`."""
        while self._index < len(self.stages):
            proposals = self.stages[self._index].propose_batch(ctx, k)
            if proposals:
                return proposals
            self._index += 1
            if self._index < len(self.stages):
                self.stages[self._index].reset(ctx)
        return []

    def observe(self, ctx: RayContext, outcome: RayOutcome) -> None:
        self.stages[min(self._index, len(self.stages) - 1)].observe(ctx, outcome)

    def accept(self, ctx: RayContext, outcome: RayOutcome) -> bool:
        return self.stages[min(self._index, len(self.stages) - 1)].accept(ctx, outcome)

    def report(self) -> dict:
        return {"strategy": self.name, "stage": self._index,
                "stages": [s.report() for s in self.stages]}


@RAY_STRATEGIES.register(
    "grow-and-polish",
    doc="Facet-queue growth up to the vertex budget, then gradient relocation at that size.",
)
def _grow_and_polish(*, policy: Any = "adaptive-ema", step: float = 0.2,
                     min_step: float = 0.01, refresh_gains: bool = True) -> SequentialRays:
    return SequentialRays([FacetQueueRays(policy=policy, refresh_gains=refresh_gains),
                           RelocationRays(step=step, min_step=min_step)],
                          name="grow-and-polish")


# --------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------
@BUILDERS.register(
    "ray-growth",
    doc="Grow a region with a pluggable ray strategy (facet queue, then relocation, by default).",
)
class RayGrowthBuilder(RegionBuilder):
    """Region construction by any :class:`RayStrategy`.

    Parameters
    ----------
    strategy, strategy_kwargs:
        A registered name with options, or an instance.  The default,
        ``"grow-and-polish"``, grows with the facet queue until the vertex
        budget is full and then relocates vertices at that size.
    seed, seed_kwargs:
        Seeding strategy (as for :class:`~OpenRC.region.expansive.ExpansiveBuilder`).
        ``"axis"`` by default: the 2d signed axis rays span a body around the
        base point in every direction, which in high dimension is worth orders
        of magnitude of volume over the one-sided simplex seed.  ``"auto"``
        follows :class:`~OpenRC.region.expansive.ExpansiveBuilder` (axis from
        4-D up).
    ray_kwargs:
        Passed to :func:`~OpenRC.geometry.bisection.ray_boundary`; ``hint`` and
        ``check_origin`` are set by the engine and refused here.
    warm_start:
        Pass each proposal's ``hint`` to the bisection.
    seed_rng:
        Seed of the strategy's random stream and of the seeding.
    batch:
        Proposals per engine step (see :func:`run_rays`): their replacement
        hulls are measured in parallel threads and the best is applied.  ``1``
        (default) is one ray at a time.
    first_order_screen:
        Opt-in replacement heuristic.  It can save hull rebuilds but may miss
        gains after the hull's facet structure changes; by default every
        replacement is measured exactly.
    """

    name = "ray-growth"
    supports_live_hull = True

    def __init__(self, *, strategy: Any = "grow-and-polish",
                 strategy_kwargs: Optional[dict] = None,
                 seed: str = "axis", seed_kwargs: Optional[dict] = None,
                 ray_kwargs: Optional[dict] = None, warm_start: bool = True,
                 seed_rng: int = 0, first_order_screen: bool = False,
                 batch: int = 1) -> None:
        if (isinstance(batch, (bool, np.bool_))
                or not isinstance(batch, (int, float, np.number))
                or not np.isfinite(batch)
                or int(batch) != batch or batch < 1):
            raise ValueError("batch must be a positive integer")
        if strategy_kwargs is not None and not isinstance(strategy_kwargs, dict):
            raise TypeError("strategy_kwargs must be a dict")
        if seed_kwargs is not None and not isinstance(seed_kwargs, dict):
            raise TypeError("seed_kwargs must be a dict")
        if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
            raise TypeError("ray_kwargs must be a dict")
        self.batch = int(batch)
        self._strategy_arg = strategy
        self.strategy_kwargs = dict(strategy_kwargs or {})
        resolve_ray_strategy(strategy, **self.strategy_kwargs)   # validate early
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError("seed must be a non-empty string")
        if seed != "auto":
            get_seed(seed)
        self.seed_name = seed
        self.seed_kwargs = dict(seed_kwargs or {})
        self.ray_kwargs = dict(ray_kwargs or {})
        _check_ray_kwargs(self.ray_kwargs)
        if not isinstance(warm_start, (bool, np.bool_)):
            raise ValueError("warm_start must be a boolean")
        self.warm_start = bool(warm_start)
        if (isinstance(seed_rng, (bool, np.bool_))
                or not isinstance(seed_rng, (int, float, np.number))
                or not np.isfinite(seed_rng)
                or int(seed_rng) != seed_rng):
            raise ValueError("seed_rng must be an integer")
        self.seed_rng = int(seed_rng)
        if not isinstance(first_order_screen, (bool, np.bool_)):
            raise ValueError("first_order_screen must be a boolean")
        self.first_order_screen = bool(first_order_screen)

    def build(self, case, axes, oracle, *, max_vertices: int = 200,
              max_moves: Optional[int] = None, max_oracle_calls: Optional[int] = None,
              max_seconds: Optional[float] = None, volume_tol: Optional[float] = None,
              patience: int = 25, memory_limit_mb: Optional[float] = None,
              max_hull_mb: Optional[float] = None,
              initial_vertices: Optional[np.ndarray] = None,
              initial_hull: Optional[LiveHull] = None, keep_hull: bool = False,
              monitor: Optional[Callable[[Any], Optional[str]]] = None,
              rng: Optional[random.Random] = None, ray_cache: Optional[dict] = None,
              tabu: Optional[set] = None, progress_every: int = 0,
              **kwargs: Any) -> RegionResult:
        """Grow (or improve) a region until a budget binds or the strategy is done.

        ``max_moves`` defaults to ``4 * max_vertices``.  The remaining
        keywords mirror :meth:`ExpansiveBuilder.build` so that
        :class:`~OpenRC.region.memory_aware.MemoryAwareBuilder` can drive this
        builder too; unknown keywords are rejected.
        """
        from OpenRC.compression.base import polytope_from_live
        from OpenRC.compression.schedule import GrowthState

        if kwargs:
            raise TypeError(f"unknown RayGrowthBuilder options: {sorted(kwargs)}")
        if not isinstance(keep_hull, (bool, np.bool_)):
            raise ValueError("keep_hull must be a boolean")
        from OpenRC.cases.base import validate_case_configuration

        validate_case_configuration(case, oracle)
        if len(axes) < 2:
            raise ValueError("Qhull-backed region construction needs at least two axes; "
                             "single-axis power-flow queries remain supported")
        if (isinstance(max_vertices, (bool, np.bool_))
                or not isinstance(max_vertices, (int, float, np.number))
                or not np.isfinite(max_vertices)
                or int(max_vertices) != max_vertices or max_vertices < len(axes) + 1):
            raise ValueError(f"max_vertices must be an integer >= {len(axes) + 1}")
        if (max_moves is not None
                and (isinstance(max_moves, (bool, np.bool_))
                     or not isinstance(max_moves, (int, float, np.number))
                     or not np.isfinite(max_moves)
                     or int(max_moves) != max_moves or max_moves < 0)):
            raise ValueError("max_moves must be a non-negative integer when provided")
        if (max_oracle_calls is not None
                and (isinstance(max_oracle_calls, (bool, np.bool_))
                     or not isinstance(max_oracle_calls, (int, float, np.number))
                     or not np.isfinite(max_oracle_calls)
                     or int(max_oracle_calls) != max_oracle_calls or max_oracle_calls <= 0)):
            raise ValueError("max_oracle_calls must be a positive integer when provided")
        if max_seconds is not None and (
            isinstance(max_seconds, (bool, np.bool_))
            or not isinstance(max_seconds, (int, float, np.number))
            or not np.isfinite(max_seconds)
            or max_seconds <= 0.0
        ):
            raise ValueError("max_seconds must be finite and positive when provided")
        if volume_tol is not None and (
            isinstance(volume_tol, (bool, np.bool_))
            or not isinstance(volume_tol, (int, float, np.number))
            or not np.isfinite(volume_tol)
            or volume_tol < 0.0
        ):
            raise ValueError("volume_tol must be finite and non-negative when provided")
        if (isinstance(patience, (bool, np.bool_))
                or not isinstance(patience, (int, float, np.number))
                or not np.isfinite(patience)
                or int(patience) != patience or patience <= 0):
            raise ValueError("patience must be a positive integer")
        if max_hull_mb is not None and (
            isinstance(max_hull_mb, (bool, np.bool_))
            or not isinstance(max_hull_mb, (int, float, np.number))
            or not (np.isfinite(max_hull_mb) and max_hull_mb > 0.0)
        ):
            raise ValueError("max_hull_mb must be finite and positive when provided")
        if (isinstance(progress_every, (bool, np.bool_))
                or not isinstance(progress_every, (int, float, np.number))
                or not np.isfinite(progress_every)
                or int(progress_every) != progress_every or progress_every < 0):
            raise ValueError("progress_every must be a non-negative integer")
        if monitor is not None and not callable(monitor):
            raise TypeError("monitor must be callable or None")
        if ray_cache is not None and not isinstance(ray_cache, dict):
            raise TypeError("ray_cache must be a dict or None")
        if initial_hull is not None and initial_vertices is not None:
            raise ValueError("give initial_vertices or initial_hull, not both")
        max_vertices = int(max_vertices)
        start = time.perf_counter()
        calls_at_start = oracle.stats.n_calls
        strategy = resolve_ray_strategy(self._strategy_arg, **self.strategy_kwargs)
        guard = MemoryGuard(limit_mb=memory_limit_mb)
        guard.check()
        stream = rng if rng is not None else random.Random(self.seed_rng)
        np_rng = np.random.default_rng(stream.randrange(2 ** 63))

        if initial_hull is not None:
            live = initial_hull
            live.compact()
            seed_info: Dict[str, Any] = {"seed": "resumed", "n_vertices": live.n_vertices}
        else:
            if initial_vertices is None:
                seed_kwargs = dict(self.seed_kwargs)
                name = resolve_seed(self.seed_name, len(axes))
                seeded = get_seed(name)(oracle, axes, rng=stream, **seed_kwargs,
                                        **self.ray_kwargs)
                points, seed_info = np.asarray(seeded.vertices, dtype=float), seeded.info
            else:
                raw_init = np.asarray(initial_vertices)
                if raw_init.dtype == bool or raw_init.ndim != 2:
                    raise ValueError("initial_vertices must be a 2-D numeric array")
                points = np.asarray(raw_init, dtype=float)
                seed_info = {"seed": "resumed", "n_vertices": len(points)}
            try:
                live = LiveHull(points)
            except ValueError as exc:
                raise ValueError("the seed polytope is degenerate (zero hypervolume)") from exc
            live.compact()
        if live.dim != len(axes):
            if initial_hull is None:
                live.release()
            raise ValueError(f"the starting hull is {live.dim}-D but there are {len(axes)} axes")
        if live.n_vertices > max_vertices:
            # The seed alone (typically ``seed="random"`` with many samples)
            # already passes the budget: compress it to the budget, and say so.
            from OpenRC.region.base import _compress_oversized_seed

            seed_info = _compress_oversized_seed(live, max_vertices, axes, seed_info)
        try:
            check_hull_budget(live, max_hull_mb)
        except MemoryBudgetExceeded:
            if initial_hull is None:
                live.release()
            raise

        def state_fn(ctx: RayContext) -> GrowthState:
            return GrowthState(
                dim=live.dim, iteration=ctx.step, n_points=live.n_vertices,
                n_vertices=live.n_vertices, n_facets=live.n_facets, volume=live.volume,
                hull_mb=live.memory_mb(), rss_mb=guard.current_mb(),
                memory_limit_mb=memory_limit_mb, oracle_calls=oracle.stats.n_calls,
                seconds=time.perf_counter() - start, max_vertices=max_vertices,
                trace=tuple(ctx.trace))

        seeded_origin = initial_hull is None and initial_vertices is None
        remaining_calls = (None if max_oracle_calls is None
                           else max_oracle_calls - (oracle.stats.n_calls - calls_at_start))
        remaining_seconds = (None if max_seconds is None
                             else max_seconds - (time.perf_counter() - start))
        try:
            if remaining_seconds is not None and remaining_seconds <= 0.0:
                run = RayRun(
                    counts={"moves": 0, "accepted": 0, "adds": 0, "replaces": 0,
                            "refused": 0, "ray_failures": 0, "cached_rays": 0, "tabu": 0},
                    stopped_because="time budget reached",
                    best_volume=live.volume,
                )
            elif remaining_calls is not None and remaining_calls <= 0:
                run = RayRun(
                    counts={"moves": 0, "accepted": 0, "adds": 0, "replaces": 0,
                            "refused": 0, "ray_failures": 0, "cached_rays": 0, "tabu": 0},
                    stopped_because="oracle-call budget reached",
                    best_volume=live.volume,
                )
            else:
                run = run_rays(
                    live, oracle, axes, strategy,
                    max_moves=max_moves if max_moves is not None else 4 * max_vertices,
                    max_vertices=max_vertices, max_hull_mb=max_hull_mb,
                    max_oracle_calls=remaining_calls, max_seconds=remaining_seconds,
                    volume_tol=volume_tol, patience=patience, ray_kwargs=self.ray_kwargs,
                    ray_cache=ray_cache, tabu=tabu, monitor=monitor, guard=guard,
                    rng=np_rng, warm_start=self.warm_start,
                    first_order_screen=self.first_order_screen, state_fn=state_fn,
                    batch=self.batch, origin_checked=seeded_origin,
                )
        except BaseException:
            if initial_hull is None:
                live.release()
            raise
        if progress_every:
            LOGGER.info("ray-growth: %d moves, %d vertices, volume %.6g (%s)",
                        run.counts["moves"], live.n_vertices, live.volume, run.stopped_because)
        stopped = run.stopped_because
        result = RegionResult(
            polytope=polytope_from_live(live, axes=axes),
            trace=run.trace,
            oracle_stats=oracle.stats.as_dict() | oracle.describe(),
            builder_info={
                "builder": self.name, "strategy": strategy.report(), "seed": seed_info,
                "case": getattr(case, "name", "?"), "axes": list(axes.labels),
                "counts": dict(run.counts), "warm_start": self.warm_start,
                "first_order_screen": self.first_order_screen,
                "hull": dict(live.stats, n_facets=live.n_facets,
                             hull_mb=round(live.memory_mb(), 3)),
                "cached_rays": run.counts["cached_rays"], "tabu_skips": run.counts["tabu"],
                "guarantee": single_oracle_guarantee(oracle), "theorem_backed": False,
            },
            seconds=time.perf_counter() - start,
            stopped_because=stopped,
        )
        if keep_hull:
            result.live_hull = live
        elif initial_hull is None:
            live.release()
        return result


__all__ = [
    "RAY_STRATEGIES",
    "FacetRays",
    "GradientRays",
    "RayContext",
    "RayGrowthBuilder",
    "RayOutcome",
    "RayProposal",
    "RayRun",
    "RayStrategy",
    "FacetQueueRays",
    "RelocationRays",
    "SequentialRays",
    "get_ray_strategy",
    "list_ray_strategies",
    "resolve_ray_strategy",
    "run_rays",
]
