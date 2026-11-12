"""Expansive polytope construction - grow a region by breaking facets.

The algorithm
-------------
1. Seed a small polytope (see :mod:`OpenRC.region.base` for the strategies).
2. Repeat:

   a. a :class:`~OpenRC.region.facet_policy.FacetPolicy` picks a facet from the
      queue;
   b. a ray is fired from the origin through that facet's centroid, and
      bisection returns the last feasible point on it - the *candidate*;
   c. if the candidate sticks out far enough past the facet's own hyperplane,
      the facet is replaced by the ``d`` facets formed by dropping one of its
      vertices and adding the candidate; otherwise the facet is retired as
      fully explored;
   d. the candidate joins the vertex set.

3. Stop on a vertex budget, an oracle-call budget, a time budget, or when the
   volume stops moving.

Candidates are cached per facet, so a facet nominated by multiple policy arms
in one round incurs a single boundary search and none in subsequent rounds,
keeping multi-arm adaptive policies practical.

Incremental live hull
---------------------
The polytope is maintained in a single :class:`~OpenRC.geometry.live_hull.LiveHull`.
Candidate volume gains are evaluated from the cones over visible facets, and
accepted vertices are inserted incrementally, avoiding ``2 * n_arms + 1`` full
hull reconstructions per iteration.  Facet areas and cone volumes are cached
for the lifetime of each facet.

:meth:`ExpansiveBuilder.build` releases the working hull before returning
unless ``keep_hull=True`` transfers ownership to ``result.live_hull`` (used by
:class:`~OpenRC.region.memory_aware.MemoryAwareBuilder` to alternate growth and
compression in place).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from OpenRC.geometry.bisection import RayResult, ray_boundary
from OpenRC.geometry.live_hull import LiveHull, cone_volumes
from OpenRC.geometry.ops import GeometryNumericsError
from OpenRC.geometry.measures import simplex_volume
from OpenRC.region.base import (BUILDERS, RegionBuilder, RegionResult, check_hull_budget,
                                get_seed, resolve_seed)
from OpenRC.region.facet_policy import (
    FacetPolicy,
    SelectionContext,
    facet_debts,
    get_facet_policy,
    resolve_facet_screen,
)
from OpenRC.utils import LOGGER, MemoryBudgetExceeded, MemoryGuard


@dataclass
class _Facet:
    """One facet awaiting exploration."""

    fid: int                       # stable id, used as the cache key
    vertex_idx: Tuple[int, ...]    # indices into the builder's vertex list
    generation: int                # 0 for the seed, parent + 1 afterwards
    debt: float = 0.0              # see facet_policy.DebtScreen


@BUILDERS.register(
    "expansive",
    doc="Grow the region by repeatedly breaking a facet (the standard construction).",
)
class ExpansiveBuilder(RegionBuilder):
    """Facet-breaking region construction, in any dimension.

    Parameters
    ----------
    policy:
        Facet-selection policy name or instance.  Defaults to ``"adaptive-ema"``
        with the cheap arms, which is a good all-round choice; pass
        ``"volume-gain"`` for the greedy reference or ``"generation"`` for the
        cheapest sensible one.
    policy_kwargs:
        Forwarded to the policy factory when ``policy`` is a name.
    seed:
        Seeding strategy: ``"simplex"``, ``"axis"``, ``"random"``, or
        ``"auto"`` (default): ``"axis"`` from 4-D up, ``"simplex"`` in 3-D and
        below (:data:`~OpenRC.region.base.AXIS_SEED_FROM_DIM`); an explicit
        name is always used as given.  The simplex seed is one-sided - the
        positive unit axes plus a single negative diagonal - and
        facet-centroid rays never undo that bias; the 2d signed axis rays span
        the base point in every direction.  At 50 vertices on ``9bus`` the
        axis seed gives 1.3, 3.8, 21, 101 and 3 120 times the volume in 3- to
        7-D for the same oracle calls.  In 3-D ``"auto"`` keeps the simplex so
        that published low-dimensional results reproduce; pass ``"axis"`` for
        more volume there too.
    seed_kwargs:
        Forwarded to the seeding strategy - for example ``n_samples`` for
        ``seed="random"``.
    min_bulge:
        A candidate must lie at least this far (in MW) outside its parent
        facet's hyperplane for the facet to be subdivided.  Below it, the facet
        is considered fully explored and retired.  The original method tested
        ``volume/area > 1e-3`` for the cone over the facet, a quantity
        proportional to this distance.
    ray_kwargs:
        Forwarded to :func:`~OpenRC.geometry.bisection.ray_boundary`
        (``tol``, ``max_ratio``, ...).
    seed_rng:
        Seed for the internal RNG, so runs are reproducible.
    n_workers:
        Workers for the seeding phase, which is a batch of independent
        feasibility tests.  ``"auto"`` decides by measuring.  The growth loop
        stays sequential by construction: each new vertex changes which facet
        should be broken next, so there is nothing to overlap.
    backend:
        ``"process"`` (default) or ``"thread"``.
    refresh_gains:
        Re-measure a queued facet's gain on the current hull whenever a
        policy asks for it after an insertion.  The boundary point is cached,
        so this costs no oracle call, only a pass over the hull's facets.
        ``True`` (default).  ``False`` keeps the gain measured when the facet
        was first scored - stale once later vertices have covered part of its
        cap - which is how the published method scored; pass it to reproduce
        published results.
    screen, screen_kwargs:
        Optional step 1 of a two-step selection
        (:class:`~OpenRC.region.facet_policy.FacetScreen`, e.g. ``"debt"``):
        each iteration the screen picks the pool the policy then chooses from.
        Facets only carry debt when ``build`` is given ``debt_points``, so a
        screen changes nothing in a fresh build.
    """

    name = "expansive"
    #: ``build`` accepts ``initial_hull``/``keep_hull``/``monitor``, so the
    #: memory-aware builder can grow and compress one hull in place.
    supports_live_hull = True
    #: ``build`` accepts ``screen``/``screen_rng``/``debt_points``.
    supports_facet_screen = True

    def __init__(
        self,
        *,
        policy="adaptive-ema",
        policy_kwargs: Optional[dict] = None,
        seed: str = "auto",
        seed_kwargs: Optional[dict] = None,
        min_bulge: float = 1e-3,
        ray_kwargs: Optional[dict] = None,
        seed_rng: int = 0,
        n_workers=1,
        backend: str = "process",
        screen=None,
        screen_kwargs: Optional[dict] = None,
        refresh_gains: bool = True,
    ) -> None:
        if not isinstance(refresh_gains, (bool, np.bool_)):
            raise ValueError("refresh_gains must be a boolean")
        self.refresh_gains = bool(refresh_gains)
        if screen_kwargs is not None and not isinstance(screen_kwargs, dict):
            raise TypeError("screen_kwargs must be a dict")
        if policy_kwargs is not None and not isinstance(policy_kwargs, dict):
            raise TypeError("policy_kwargs must be a dict")
        if seed_kwargs is not None and not isinstance(seed_kwargs, dict):
            raise TypeError("seed_kwargs must be a dict")
        if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
            raise TypeError("ray_kwargs must be a dict")
        self.screen = resolve_facet_screen(screen, **dict(screen_kwargs or {}))
        self.policy_name = policy if isinstance(policy, str) else getattr(policy, "name", "custom")
        self._policy_arg = policy
        self.policy_kwargs = dict(policy_kwargs or {})
        if isinstance(policy, str):
            get_facet_policy(policy, **self.policy_kwargs)
        elif not isinstance(policy, FacetPolicy):
            raise TypeError(f"policy must be a registered name or a FacetPolicy; got {policy!r}")
        elif self.policy_kwargs:
            raise TypeError("policy_kwargs apply only when policy is given by name")
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError("seed must be a non-empty string")
        if seed != "auto":
            get_seed(seed)
        self.seed_name = seed
        self.seed_kwargs = dict(seed_kwargs or {})
        if (isinstance(min_bulge, (bool, np.bool_))
                or not isinstance(min_bulge, (int, float, np.number))):
            raise ValueError("min_bulge must be finite and non-negative")
        self.min_bulge = float(min_bulge)
        if not np.isfinite(self.min_bulge) or self.min_bulge < 0.0:
            raise ValueError("min_bulge must be finite and non-negative")
        self.ray_kwargs = dict(ray_kwargs or {})
        if (isinstance(seed_rng, (bool, np.bool_))
                or not isinstance(seed_rng, (int, float, np.number))
                or not np.isfinite(seed_rng)
                or int(seed_rng) != seed_rng):
            raise ValueError("seed_rng must be an integer")
        self.seed_rng = int(seed_rng)
        # Only the seeding phase is batched; the growth loop is inherently
        # sequential because each new vertex changes which facet to break next.
        self.n_workers = n_workers
        if backend not in ("process", "thread"):
            raise ValueError("backend must be one of the two supported execution modes: process or thread.")
        self.backend = backend

    def seed_for(self, dim: int) -> str:
        """The seeding strategy used in ``dim`` dimensions (resolves ``"auto"``)."""
        return resolve_seed(self.seed_name, dim)

    # ------------------------------------------------------------------ build
    def build(
        self,
        case,
        axes,
        oracle,
        *,
        max_vertices: int = 200,
        max_oracle_calls: Optional[int] = None,
        max_seconds: Optional[float] = None,
        volume_tol: Optional[float] = None,
        patience: int = 25,
        memory_limit_mb: Optional[float] = None,
        initial_vertices: Optional[np.ndarray] = None,
        initial_hull: Optional[LiveHull] = None,
        keep_hull: bool = False,
        monitor: Optional[Callable[[Any], Optional[str]]] = None,
        rng: Optional[random.Random] = None,
        ray_cache: Optional[Dict[tuple, np.ndarray]] = None,
        tabu: Optional[set] = None,
        debt_points: Optional[np.ndarray] = None,
        screen=None,
        screen_rng: Optional[np.random.Generator] = None,
        max_hull_mb: Optional[float] = None,
        progress_every: int = 0,
        **kwargs: Any,
    ) -> RegionResult:
        """Grow a region until one of the budgets is exhausted.

        Parameters
        ----------
        max_vertices:
            Stop once the vertex set - the seed points plus one boundary point
            per iteration - reaches this size.  A stored point can later fall
            inside the hull, so the result may have fewer hull vertices
            (``n_hull_vertices`` in the trace); the ray-growth builder caps the
            hull vertices themselves.
        max_oracle_calls, max_seconds:
            Additional, cooperative budgets; whichever binds first wins.  They
            are checked between growth iterations, so a ray search or seed
            batch already in progress is allowed to finish.
        volume_tol, patience:
            Stop early when the relative volume gain stays below ``volume_tol``
            for ``patience`` consecutive iterations.
        memory_limit_mb:
            Cooperative resident-set-size budget in MiB (see :class:`MemoryGuard`):

            * **Exceeded before seeding**: raises :class:`MemoryBudgetExceeded`
              immediately because no partial region exists yet.
            * **Exceeded during growth**: terminates the loop, returns the
              region constructed so far, and records the reason in
              ``stopped_because``.
        initial_vertices:
            Skip seeding and start from these vertices.
        initial_hull:
            Skip seeding and continue growing this live hull in place.
            This is how the memory-aware builder resumes after compressing.
        keep_hull:
            Hand the live hull on in ``result.live_hull`` instead of releasing
            it.  The caller then owns it and must call ``release()``.
        monitor:
            Called once per iteration with a
            :class:`~OpenRC.compression.schedule.GrowthState`; returning a
            string stops the expansion with that string as
            ``stopped_because``.  Compression schedules plug in here.
        rng:
            Random stream to draw from instead of a fresh
            ``random.Random(seed_rng)``.  A caller that resumes growth several
            times passes one stream, so that successive rounds do not replay
            the same tie-breaks from the same hull.
        ray_cache:
            ``{direction: boundary point}`` shared across calls.  A facet that
            survives a compression has the same centroid ray as before, so its
            boundary search is not repeated.
        tabu:
            Keys (:func:`point_key`) of points a compression already judged
            not worth keeping.  A facet whose candidate is tabu is retired
            without adding the point, so the budget goes to new directions
            instead of re-adding what was just removed.
        debt_points:
            Points a compression discarded.  With a screen, every starting
            facet is charged the debt it owes them
            (:func:`~OpenRC.region.facet_policy.facet_debts`).
        screen, screen_rng:
            Override the builder's screen for this call, and the numpy stream
            it draws from (default: derived from ``seed_rng``, independent of
            the policy's stream).
        max_hull_mb:
            Hard cap on the live hull's modelled memory.  Checked *before*
            each insertion with the exact post-insertion facet count
            (:meth:`~OpenRC.geometry.live_hull.LiveHull.facets_after`); an
            insertion that would pass it stops the expansion instead, so the
            returned hull stays within the cap. An oversized starting hull
            raises ``MemoryBudgetExceeded`` after seed construction; transient
            seed/Qhull allocations and process RSS are not covered by this cap.
            In 16-D one insertion can add 15 MiB.
        progress_every:
            Log a progress line every N iterations (0 disables).
        """
        if kwargs:
            raise TypeError(f"unknown ExpansiveBuilder options: {sorted(kwargs)}")
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
        max_vertices = int(max_vertices)
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
        if (isinstance(progress_every, (bool, np.bool_))
                or not isinstance(progress_every, (int, float, np.number))
                or not np.isfinite(progress_every)
                or int(progress_every) != progress_every or progress_every < 0):
            raise ValueError("progress_every must be a non-negative integer")
        if max_hull_mb is not None and (
            isinstance(max_hull_mb, (bool, np.bool_))
            or not isinstance(max_hull_mb, (int, float, np.number))
            or not (np.isfinite(max_hull_mb) and max_hull_mb > 0)
        ):
            raise ValueError("max_hull_mb must be finite and positive when provided")
        if monitor is not None and not callable(monitor):
            raise TypeError("monitor must be callable or None")
        if ray_cache is not None and not isinstance(ray_cache, dict):
            raise TypeError("ray_cache must be a dict or None")

        wall_start = time.perf_counter()
        calls_at_start = oracle.stats.n_calls
        rng = rng if rng is not None else random.Random(self.seed_rng)
        rays: Dict[tuple, np.ndarray] = ray_cache if ray_cache is not None else {}
        n_cached_rays = 0
        n_tabu = 0
        policy: FacetPolicy = (
            get_facet_policy(self._policy_arg, **self.policy_kwargs)
            if isinstance(self._policy_arg, str) else self._policy_arg
        )
        policy.reset()
        guard = MemoryGuard(limit_mb=memory_limit_mb)
        guard.check()

        # ---------------------------------------------------------- seeding
        if initial_hull is not None and initial_vertices is not None:
            raise ValueError("give initial_vertices or initial_hull, not both")
        if initial_hull is not None:
            live = initial_hull
            live.compact()
            seed_info = {"seed": "resumed", "n_vertices": live.n_vertices}
        else:
            if initial_vertices is None:
                seed_name = self.seed_for(len(axes))
                seeder = get_seed(seed_name)
                seed_kwargs = dict(self.seed_kwargs)
                if seed_name in ("axis", "random"):
                    seed_kwargs.setdefault("n_workers", self.n_workers)
                    seed_kwargs.setdefault("backend", self.backend)
                seed_result = seeder(oracle, axes, rng=rng, **seed_kwargs,
                                     **self.ray_kwargs)
                seed_points = np.asarray(seed_result.vertices, dtype=float)
                seed_info = seed_result.info
            else:
                raw_init = np.asarray(initial_vertices)
                if raw_init.dtype == bool or raw_init.ndim != 2:
                    raise ValueError("initial_vertices must be a 2-D numeric array")
                seed_points = np.asarray(raw_init, dtype=float)
                seed_info = {"seed": "resumed", "n_vertices": len(seed_points)}
            try:
                live = LiveHull(seed_points)
            except ValueError as exc:
                raise ValueError(
                    "the seed polytope is degenerate (zero hypervolume). The region may "
                    "be flat in some axis, or the oracle may reject the base point."
                ) from exc
            # Only hull vertices are kept: interior points contribute nothing to
            # the body and would otherwise eat into the vertex budget.  This
            # matters for the random seed, most of whose feasible samples land
            # inside the hull.
            if live.n_vertices < len(seed_points):
                seed_info = dict(seed_info, n_seed_vertices=len(seed_points),
                                 n_seed_on_hull=live.n_vertices)
                live.compact()
        if live.dim != len(axes):
            if initial_hull is None:
                live.release()
            raise ValueError(f"the starting hull is {live.dim}-D but there are {len(axes)} axes")
        if live.n_vertices > max_vertices:
            # If the seed hull exceeds max_vertices (e.g. seed="random"),
            # compress it in place to max_vertices and log the reduction.
            from OpenRC.region.base import _compress_oversized_seed

            seed_info = _compress_oversized_seed(live, max_vertices, axes, seed_info)
        try:
            check_hull_budget(live, max_hull_mb)
        except MemoryBudgetExceeded:
            if initial_hull is None:
                live.release()
            raise

        vertices = [np.asarray(v, dtype=float) for v in live.vertices]
        screen = resolve_facet_screen(screen) if screen is not None else self.screen
        if screen is not None:
            screen.reset()
            if screen_rng is None:
                screen_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed_rng, 0x5C4EE9]))
        debts = (facet_debts(live, debt_points,
                             weighting=getattr(screen, "weighting", "count"))
                 if screen is not None and debt_points is not None and len(debt_points)
                 else np.zeros(live.n_facets))
        facets = [
            _Facet(fid=i, vertex_idx=tuple(int(v) for v in row), generation=0,
                   debt=float(debts[i]))
            for i, row in enumerate(live.facet_positions())
        ]
        by_fid: Dict[int, _Facet] = {f.fid: f for f in facets}
        next_fid = len(facets)

        # Per-facet caches keyed by the stable facet id.
        candidate: Dict[int, np.ndarray] = {}
        gain_cache: Dict[int, float] = {}
        measure_cache: Dict[str, Dict[int, float]] = {"area": {}, "cone": {}}

        trace: List[dict] = []
        stopped = ("max_vertices reached" if "exceeds_max_vertices" not in seed_info else
                   f"max_vertices reached by the seed: {seed_info['exceeds_max_vertices']} "
                   f"hull vertices compressed to {len(vertices)}, nothing grown")
        stagnant = 0
        last_volume = live.volume

        # Without refresh, gains come straight from LiveHull.gain.
        measured_gain = _GainMeter(live).gain if self.refresh_gains else live.gain
        ray_origin = (np.zeros(len(axes)) if self.ray_kwargs.get("origin") is None
                      else np.asarray(self.ray_kwargs["origin"], dtype=float))

        def gain_of(fid: int) -> float:
            """Hypervolume the facet's candidate would add (memoised).

            Boundary search dominates the cost; the volume gain is evaluated
            directly from the live hull facets without reconstructing the hull.
            """
            if fid in gain_cache:
                return gain_cache[fid]
            if fid in candidate:
                # The ray was searched already; only the hull has moved since.
                gain_cache[fid] = measured_gain(candidate[fid])
                return gain_cache[fid]
            facet = by_fid[fid]
            points = np.array([vertices[i] for i in facet.vertex_idx])
            # The ray goes from the ray origin *through* the centroid.
            shifted = points - ray_origin
            centroid = shifted.mean(axis=0)
            interior = live.vertices.mean(axis=0) - ray_origin
            outward = _outward_normal(shifted, interior=interior)
            scale = max(float(np.abs(shifted).max(initial=1.0)), 1.0)
            if np.linalg.norm(centroid) <= 1e-12 * scale or float(centroid @ outward) <= 0.0:
                # A facet whose centroid sits on (or behind) the origin gives no
                # outward centroid ray; use its outward normal instead.
                centroid = outward
            from OpenRC.region.ray_growth import _ray_cache_key

            key = _ray_cache_key(ray_origin, centroid)
            if key in rays:
                nonlocal n_cached_rays
                n_cached_rays += 1
                boundary = rays[key]
            else:
                ray: RayResult = ray_boundary(oracle, axes, centroid, **self.ray_kwargs)
                boundary = ray.boundary
                rays[key] = boundary
            candidate[fid] = boundary
            gain_cache[fid] = measured_gain(boundary)
            return gain_cache[fid]

        def snapshot(n_iter: int) -> Any:
            from OpenRC.compression.schedule import GrowthState

            return GrowthState(
                dim=live.dim, iteration=n_iter, n_points=len(vertices),
                n_vertices=live.n_vertices, n_facets=live.n_facets, volume=live.volume,
                hull_mb=live.memory_mb(), rss_mb=guard.current_mb(),
                memory_limit_mb=memory_limit_mb, oracle_calls=oracle.stats.n_calls,
                seconds=time.perf_counter() - wall_start, max_vertices=max_vertices,
                trace=tuple(trace))

        # ------------------------------------------------------------- loop
        iteration = 0
        while len(vertices) < max_vertices:
            if not facets:
                stopped = "facet queue exhausted (region fully explored)"
                break
            if max_seconds is not None and (time.perf_counter() - wall_start) > max_seconds:
                stopped = "time budget reached"
                break
            if (max_oracle_calls is not None
                    and oracle.stats.n_calls - calls_at_start >= max_oracle_calls):
                stopped = "oracle-call budget reached"
                break
            try:
                guard.check()
            except MemoryBudgetExceeded as exc:
                stopped = f"memory budget reached ({exc})"
                break
            if monitor is not None:
                reason = monitor(snapshot(iteration))
                if reason:
                    stopped = str(reason)
                    break

            iteration += 1
            vertex_array = np.array(vertices)
            # Step 1 (optional): the screen decides which facets may be chosen.
            pool = np.arange(len(facets))
            if screen is not None:
                debt = np.array([f.debt for f in facets])
                if debt.any():
                    mask = screen.pool(debt, screen_rng)
                    if not mask.any():
                        mask = debt <= debt.min()     # never leave the policy nothing
                    pool = np.flatnonzero(mask)
            candidates = [facets[i] for i in pool]
            # Step 2: the facet-selection policy chooses within the pool.
            ctx = SelectionContext(
                vertices=vertex_array - ray_origin,    # cone measures from the ray origin
                facets=[f.vertex_idx for f in candidates],
                generations=[f.generation for f in candidates],
                facet_ids=[f.fid for f in candidates],
                gain_fn=gain_of,
                rng=rng,
                cache=measure_cache,
            )
            picked = policy.select(ctx)
            position = int(pool[picked])
            chosen = facets[position]

            gain = gain_of(chosen.fid)          # cached if the policy already asked
            new_point = candidate[chosen.fid]
            policy.observe(ctx, picked, gain)

            if tabu and point_key(new_point) in tabu:
                n_tabu += 1
                facets.pop(position)
                del by_fid[chosen.fid]
                gain_cache.pop(chosen.fid, None)
                candidate.pop(chosen.fid, None)
                for cache in measure_cache.values():
                    cache.pop(chosen.fid, None)
                continue

            if max_hull_mb is not None:
                after_mb = live.memory_after_mb(new_point)
                if after_mb > max_hull_mb:
                    stopped = (f"memory cap: the next insertion would take the hull to "
                               f"{after_mb:.1f} MiB (cap {max_hull_mb:.1f} MiB)")
                    break

            # How far did the candidate push past the parent facet's plane?
            facet_points = vertex_array[list(chosen.vertex_idx)]
            bulge = _distance_to_hyperplane(facet_points, new_point)

            vertices.append(new_point)
            new_index = len(vertices) - 1

            facets.pop(position)
            del by_fid[chosen.fid]
            gain_cache.pop(chosen.fid, None)
            candidate.pop(chosen.fid, None)
            for cache in measure_cache.values():
                cache.pop(chosen.fid, None)

            if bulge > self.min_bulge:
                # Replace the facet by its d children.
                for dropped in chosen.vertex_idx:
                    child_idx = tuple(v for v in chosen.vertex_idx if v != dropped) + (new_index,)
                    child = _Facet(next_fid, child_idx, chosen.generation + 1,
                                   screen.inherit(chosen.debt) if screen is not None else 0.0)
                    facets.append(child)
                    by_fid[next_fid] = child
                    next_fid += 1

            delta = (_InsertionDelta(live, new_point)
                     if self.refresh_gains and gain_cache else None)
            rollback = live.vertices if max_hull_mb is not None else None
            try:
                volume = live.add(new_point)
            except GeometryNumericsError as exc:
                # The rare vertex set that Qhull cannot hull in any of its
                # modes: LiveHull.add left the hull as it was, so growth stops
                # at the last valid region instead of losing it.
                vertices.pop()
                LOGGER.warning("growth stopped at %d vertices: %s", live.n_vertices,
                               str(exc)[:200])
                stopped = ("geometry: Qhull could not insert the next vertex; "
                           "the last valid hull is returned")
                break
            if rollback is not None and max_hull_mb is not None and live.memory_mb() > max_hull_mb:
                # facets_after() is exact only in general position; a merge of
                # coplanar facets can leave more.  The cap is a hard one.
                after_mb = live.memory_mb()
                live.reset(rollback)
                stopped = (f"memory cap: the insertion took the hull to {after_mb:.1f} MiB "
                           f"(cap {max_hull_mb:.1f} MiB) and was undone")
                break
            if delta is not None:
                # Gains measured on the hull before this insertion: drop the
                # ones it can have changed (re-measured when next asked).
                fids = list(gain_cache)
                stale = delta.stale(live, np.array([candidate[f] for f in fids]))
                for fid, is_stale in zip(fids, stale):
                    if is_stale:
                        del gain_cache[fid]
            added = volume - last_volume
            relative_gain = (volume - last_volume) / last_volume if last_volume > 0 else np.inf
            trace.append({
                "iteration": iteration,
                "n_vertices": len(vertices),
                "n_hull_vertices": live.n_vertices,
                "n_facets": live.n_facets,
                "hull_mb": live.memory_mb(),
                "n_queue": len(facets),
                "volume": volume,
                "gain": gain,
                "added": added,
                "pool": len(candidates),
                "debt": chosen.debt,
                "bulge": bulge,
                "generation": chosen.generation,
                "oracle_calls": oracle.stats.n_calls,
                "seconds": time.perf_counter() - wall_start,
            })

            if volume_tol is not None:
                stagnant = stagnant + 1 if relative_gain < volume_tol else 0
                if stagnant >= patience:
                    stopped = f"volume converged (< {volume_tol} for {patience} iterations)"
                    break
            last_volume = volume

            if progress_every and iteration % progress_every == 0:
                LOGGER.info(
                    "expansive: %d vertices, %d hull facets, volume %.6g, %d oracle calls",
                    len(vertices), live.n_facets, volume, oracle.stats.n_calls,
                )

        from OpenRC.compression.base import polytope_from_live

        polytope = polytope_from_live(live, axes=axes)
        result = RegionResult(
            polytope=polytope,
            trace=trace,
            oracle_stats=oracle.stats.as_dict() | oracle.describe(),
            builder_info={
                "builder": self.name,
                "policy": policy.report(),
                "seed": seed_info,
                "min_bulge": self.min_bulge,
                "refresh_gains": self.refresh_gains,
                "iterations": iteration,
                "case": getattr(case, "name", "?"),
                "axes": list(axes.labels),
                "hull": dict(live.stats, n_facets=live.n_facets,
                             hull_mb=round(live.memory_mb(), 3)),
                "cached_rays": n_cached_rays,
                "tabu_skips": n_tabu,
                "screen": screen.report() if screen is not None else None,
                "initial_debt": float(debts.sum()),
            },
            seconds=time.perf_counter() - wall_start,
            stopped_because=stopped,
        )
        if keep_hull:
            result.live_hull = live
        elif initial_hull is None:
            live.release()      # the caller's hull is the caller's to release
        return result

def point_key(point: np.ndarray) -> tuple:
    """Hashable identity of a boundary point (rounded to 1e-9 of its scale)."""
    raw = np.asarray(point)
    if raw.dtype == bool:
        raise ValueError("point must be numeric, not boolean")
    p = np.asarray(raw, dtype=float)
    scale = max(float(np.abs(p).max()), 1.0)
    return tuple(np.round(p / scale, 9)) + (round(scale, 6),)


def _facet_areas(live) -> np.ndarray:
    """Area of every facet of ``live``: ``|det(edges, unit normal)| / (d-1)!``."""
    points, simplices, planes = live.points, live.simplices, live.equations
    dim = points.shape[1]
    areas = np.empty(len(simplices))
    scale = 1.0 / math.factorial(dim - 1)
    for first in range(0, len(simplices), 16384):
        corners = points[simplices[first:first + 16384]]
        frame = np.empty_like(corners)
        frame[:, :-1, :] = corners[:, 1:, :] - corners[:, :1, :]
        frame[:, -1, :] = planes[first:first + 16384, :-1]
        areas[first:first + 16384] = np.abs(np.linalg.det(frame)) * scale
    return areas


class _GainMeter:
    """Gains on a live hull for a policy that re-measures its queue often.

    :meth:`LiveHull.gain` spends a determinant per facet the point sees.  With
    every facet's area a gain is ``sum(area * distance) / d`` over the facets
    it sees - one product with the planes (on an 8-D hull of 42,673 facets,
    200 gains take 25 ms instead of 697 ms).  The areas are
    computed once the determinants spent on the current hull exceed one per
    facet, so a policy that asks for a few gains pays at most twice the cheaper
    way.  Any change of the hull (insertion, rebuild) resets both.
    """

    def __init__(self, live) -> None:
        self.live = live
        self._state: Optional[tuple] = None
        self._areas: Optional[np.ndarray] = None
        self._det_facets = 0

    def gain(self, point: np.ndarray) -> float:
        live = self.live
        state = (live.stats["builds"], live.stats["inserts"], live.n_facets)
        if state != self._state:
            self._state, self._areas, self._det_facets = state, None, 0
        planes = live.equations
        distance = planes[:, :-1] @ point + planes[:, -1]
        if self._areas is None:
            visible = distance > 0.0
            if self._det_facets < live.n_facets:
                self._det_facets += int(visible.sum())
                if not visible.any():
                    return 0.0
                return float(cone_volumes(live.points, live.simplices[visible], point).sum())
            self._areas = _facet_areas(live)
        return float(self._areas @ np.maximum(distance, 0.0)) / live.dim


class _InsertionDelta:
    """Which cached gains one :meth:`LiveHull.add` of ``point`` can change.

    A candidate's gain is the sum of the cones over the facets it sees.  A
    facet that survives an insertion keeps its vertices, hence its plane and
    its cone, so only a candidate that sees a facet the new point removed (one
    it sees) or one the insertion created (it contains the new point) can
    change.  A facet Qhull merges or re-triangulates is coplanar with such a
    new facet, so a candidate seeing it sees the new facet too.  Anything
    unexpected - a rebuild (repair, fallback or compaction renumbers the
    points), another point than ``point``, facet counts that do not add up -
    marks every gain stale.
    """

    def __init__(self, live, point: np.ndarray) -> None:
        planes = live.equations
        seen = planes[:, :-1] @ point + planes[:, -1] > 0.0
        self.point = np.array(point, dtype=float)
        self.removed = np.array(planes[seen])
        self.n_kept = live.n_facets - int(seen.sum())
        self.n_points = live.n_points
        self.builds = live.stats["builds"]

    def stale(self, live, points: np.ndarray) -> np.ndarray:
        """Boolean mask over ``points``: whose gain must be measured again."""
        everything = np.ones(len(points), dtype=bool)
        if (live.stats["builds"] != self.builds or live.n_points != self.n_points + 1
                or not np.array_equal(live.points[-1], self.point)):
            return everything
        created = np.any(live.simplices == live.n_points - 1, axis=1)
        if live.n_facets - int(created.sum()) != self.n_kept:
            return everything
        planes = np.vstack([self.removed, live.equations[created]])
        # Points within rounding of a changed plane count as seeing it.
        tolerance = 1e-9 * max(1.0, float(np.abs(points).max(initial=0.0)))
        distance = planes[:, :-1] @ points.T + planes[:, -1:]
        return np.any(distance > -tolerance, axis=0)


def _distance_to_hyperplane(facet_points: np.ndarray, point: np.ndarray) -> float:
    """Distance from ``point`` to the hyperplane through ``facet_points``.

    Equivalent to ``d * volume(cone) / volume(facet)`` - the quantity the
    original ``volume / area`` test was proportional to - but computed directly
    and stable when the facet is nearly degenerate.
    """
    pts = np.asarray(facet_points, dtype=float)
    dim = pts.shape[1]
    base_area = simplex_volume(pts)
    if base_area <= 0.0:
        return 0.0
    cone = simplex_volume(np.vstack([pts, np.asarray(point, dtype=float)]))
    return float(dim * cone / base_area)


def _outward_normal(facet_points: np.ndarray,
                    interior: Optional[np.ndarray] = None) -> np.ndarray:
    """Outward unit normal of the hyperplane spanned by ``facet_points``.

    Used when a facet's centroid sits on (or behind) the ray origin, leaving
    the centroid ray undefined or inward.  ``interior`` supplies an interior
    point of the shifted hull so a hyperplane passing through ``0`` is still
    oriented away from the body rather than by rounding noise on ``pts[0]``.
    """
    pts = np.asarray(facet_points, dtype=float)
    vectors = pts[1:] - pts[0]
    # The null space of the spanning vectors is the facet normal.
    _, _, vh = np.linalg.svd(vectors)
    normal = vh[-1]
    ref = pts[0] if interior is None else (pts[0] - np.asarray(interior, dtype=float))
    return normal if float(normal @ ref) >= 0.0 else -normal
