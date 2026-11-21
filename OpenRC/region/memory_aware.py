"""Memory-aware region construction via in-place expansion and compression.

In high dimensions, convex-hull facet storage becomes the primary bottleneck
before boundary accuracy converges.  Facet counts grow combinatorially with
vertex count (e.g., on the 9-bus network in 16-D: 36 vertices produce 105,000
facets; 42 vertices produce 960,000 facets and ~1.8 GiB), exhausting memory
during a single unpruned expansion.

This builder interleaves the two halves of the library on **one live hull**
(:class:`~OpenRC.geometry.live_hull.LiveHull`): the expansion inserts points
into it incrementally, a :class:`~OpenRC.compression.base.Compressor` removes
vertices from it in place, and the expansion resumes on it.  No stage rebuilds
the hull from scratch for its own bookkeeping, and only the vertex set is
returned.

*When* to compress and *how far* is a pluggable
:class:`~OpenRC.compression.schedule.CompressionSchedule`:

``"vertex-budget"`` (default)  the published rule, grow to ``max_vertices``
                               and compress to ``target_vertices``;
``"memory-budget"``            keep the live hull under a memory budget;
``"adaptive"``                 memory decides when, the exploration decides
                               how much and when to stop;
``None``                       never compress (one expansion).

A schedule instance, or any registered name with ``schedule_kwargs``, can be
passed; see :mod:`OpenRC.compression.schedule` for writing one.

Between rounds
--------------
State preserved across rounds:

* **Random streams.**  Round ``r`` draws from its own stream, derived from
  ``(seed_rng, r)`` by :class:`numpy.random.SeedSequence` - statistically
  independent of every other round and reproducible on its own.  (A fresh
  ``random.Random(seed_rng)`` per round would replay round 0's tie-breaks on a
  similar hull; in 16-D that re-adds and re-removes the same two vertices for
  eight rounds.)
* **What compression discarded.**  Its points are *tabu* - a resumed expansion
  does not re-add them - and, with ``facet_screen="debt"``, they charge a debt
  to the facets they lie beyond, which the screen uses to keep those facets
  (and, by inheritance, their children) out of the selection pool with high
  probability.  Boundary searches are cached by ray direction, so a facet that
  survives a compression costs no oracle call when it is explored again.
* **The best volume so far**, which is the potential that bounds the run: see
  :meth:`~OpenRC.compression.schedule.CompressionSchedule.finished`.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from OpenRC.geometry.polytope import Polytope
from OpenRC.region.base import BUILDERS, RegionBuilder, RegionResult, single_oracle_guarantee
from OpenRC.utils import LOGGER, MemoryBudgetExceeded


@BUILDERS.register(
    "memory-aware",
    doc="Alternate expansion and compression on one live hull, under a pluggable schedule.",
)
class MemoryAwareBuilder(RegionBuilder):
    """Expansion/compression cycles under a compression schedule.

    Parameters
    ----------
    inner_builder:
        The expansion engine; defaults to :class:`ExpansiveBuilder`.  A builder
        that does not take a live hull still works: it is resumed from the
        compressed vertex set instead.
    compressor:
        Name or instance of a :class:`~OpenRC.compression.base.Compressor`.
        ``"auto"`` (default) picks ``"local-search"`` up to 6-D and
        ``"star-screened"`` above, where every vertex neighbours almost every
        other and exact greedy costs a full hull per vertex per removal: 3.2-3.9x
        faster in 7- and 8-D at volume ratios 0.93-1.12.  Pass
        ``"local-search"`` for exact greedy in every dimension.
    schedule, schedule_kwargs:
        When and how far to compress (see the module docstring).  ``None``
        switches compression off.
    max_vertices:
        Hard cap on the points an expansion may hold (``M`` of the paper).
        Every schedule respects it; ``vertex-budget`` compresses exactly when
        it is reached.
    target_vertices:
        ``M_hat`` of the paper - the vertex-budget schedule's target.
    rounds:
        Maximum number of compressions.
    facet_screen, screen_kwargs:
        Optional step 1 of the two-step facet selection of resumed expansions
        (:class:`~OpenRC.region.facet_policy.DebtScreen`): ``"debt"`` screens
        out facets that lie over discarded vertices.  Off by default: from the
        axis seed its effect on the 8-D benchmark is within the spread between
        random seeds.  Only inner builders with a facet queue
        (``supports_facet_screen``) are screened.
    refiner, refine_kwargs, refine_moves:
        Optional refinement after every expansion: a
        :class:`~OpenRC.region.refine.Refiner` (``"relocation"``) moves
        vertices along the boundary to increase volume at fixed vertex count
        within the memory cap, improving accuracy once further vertex additions
        or removals are unprofitable.
        ``refine_moves`` caps the moves per round (default: the refiner's
        own, twice the vertex count for ``relocation``).  ``None`` (default)
        switches refinement off.
    """

    name = "memory-aware"

    def __init__(
        self,
        *,
        inner_builder=None,
        compressor="auto",
        schedule="vertex-budget",
        schedule_kwargs: Optional[dict] = None,
        max_vertices: int = 150,
        target_vertices: int = 75,
        rounds: int = 10,
        facet_screen=None,
        screen_kwargs: Optional[dict] = None,
        refiner=None,
        refine_kwargs: Optional[dict] = None,
        refine_moves: Optional[int] = None,
    ) -> None:
        from OpenRC.region.refine import resolve_refiner

        if refine_kwargs is not None and not isinstance(refine_kwargs, dict):
            raise TypeError("refine_kwargs must be a dict")
        if screen_kwargs is not None and not isinstance(screen_kwargs, dict):
            raise TypeError("screen_kwargs must be a dict")
        if schedule_kwargs is not None and not isinstance(schedule_kwargs, dict):
            raise TypeError("schedule_kwargs must be a dict")
        self.refiner = resolve_refiner(refiner, **dict(refine_kwargs or {}))
        if (refine_moves is not None
                and (isinstance(refine_moves, (bool, np.bool_))
                     or not isinstance(refine_moves, (int, float, np.number))
                     or not np.isfinite(refine_moves)
                     or int(refine_moves) != refine_moves or refine_moves < 0)):
            raise ValueError("refine_moves must be a non-negative integer or None")
        self.refine_moves = None if refine_moves is None else int(refine_moves)
        from OpenRC.region.facet_policy import resolve_facet_screen

        self.facet_screen = resolve_facet_screen(facet_screen, **dict(screen_kwargs or {}))
        if (isinstance(max_vertices, (bool, np.bool_))
                or not isinstance(max_vertices, (int, float, np.number))
                or not np.isfinite(max_vertices)
                or int(max_vertices) != max_vertices or max_vertices < 2):
            raise ValueError("max_vertices must be an integer >= 2")
        if (isinstance(target_vertices, (bool, np.bool_))
                or not isinstance(target_vertices, (int, float, np.number))
                or not np.isfinite(target_vertices)
                or int(target_vertices) != target_vertices or target_vertices < 1):
            raise ValueError("target_vertices must be a positive integer")
        if (isinstance(rounds, (bool, np.bool_))
                or not isinstance(rounds, (int, float, np.number))
                or not np.isfinite(rounds)
                or int(rounds) != rounds or rounds < 0):
            raise ValueError("rounds must be a non-negative integer")
        if inner_builder is None:
            from OpenRC.region.expansive import ExpansiveBuilder

            inner_builder = ExpansiveBuilder()
        elif not callable(getattr(inner_builder, "build", None)):
            raise TypeError("inner_builder must have a callable build(case, axes, oracle, ...) method")
        self.inner_builder = inner_builder
        self._compressor_arg = compressor
        self._schedule_arg = schedule
        self.schedule_kwargs = dict(schedule_kwargs or {})
        self.max_vertices = int(max_vertices)
        self.target_vertices = int(target_vertices)
        self.rounds = int(rounds)
        sched = self._schedule()
        if sched is not None and sched.name == "vertex-budget":
            sched_target = getattr(sched, "target_vertices", self.target_vertices)
            if sched_target >= self.max_vertices:
                raise ValueError("target_vertices must be smaller than max_vertices")
        if compressor != "auto":
            from OpenRC.compression.base import Compressor, get_compressor

            if isinstance(compressor, str):
                get_compressor(compressor)
            elif not isinstance(compressor, Compressor):
                raise TypeError(
                    f"compressor must be 'auto', a registered name, or a Compressor; got {compressor!r}"
                )

    # ------------------------------------------------------------- plumbing
    def _compressor(self, dim: int):
        from OpenRC.compression import get_compressor
        from OpenRC.compression.base import auto_compressor_name

        arg = self._compressor_arg
        if arg == "auto":
            arg = auto_compressor_name(dim)
        return get_compressor(arg) if isinstance(arg, str) else arg

    def _schedule(self):
        from OpenRC.compression.schedule import resolve_schedule

        if self._schedule_arg == "vertex-budget":
            kwargs = {"target_vertices": self.target_vertices} | self.schedule_kwargs
            return resolve_schedule("vertex-budget", **kwargs)
        return resolve_schedule(self._schedule_arg, **self.schedule_kwargs)

    # ---------------------------------------------------------------- build
    def build(self, case, axes, oracle, *, memory_limit_mb: Optional[float] = None,
              max_seconds: Optional[float] = None,
              max_oracle_calls: Optional[int] = None, **kwargs: Any) -> RegionResult:
        """Run expansion/compression cycles.

        ``max_seconds`` and ``max_oracle_calls`` are cooperative budgets for
        the **whole** run, not each round.  A ray or seed batch already in
        progress may finish before the next check.  Other keywords go to the
        inner builder.  If the memory budget is already exceeded when a later
        round starts (freed memory is not always returned to the operating
        system), the run stops and returns the best region so far instead of
        raising.
        """
        from OpenRC.compression.schedule import GrowthState
        from OpenRC.geometry.live_hull import LiveHull
        from OpenRC.region.expansive import point_key
        from OpenRC.cases.base import validate_case_configuration

        validate_case_configuration(case, oracle)
        if len(axes) < 2:
            raise ValueError(
                "Qhull-backed region construction needs at least two axes; "
                "single-axis power-flow queries remain supported"
            )
        if self.max_vertices < len(axes) + 1:
            raise ValueError(f"max_vertices must be an integer >= {len(axes) + 1}")

        if max_seconds is not None and (
            isinstance(max_seconds, (bool, np.bool_))
            or not isinstance(max_seconds, (int, float, np.number))
            or not np.isfinite(max_seconds)
            or max_seconds <= 0.0
        ):
            raise ValueError("max_seconds must be finite and positive when provided")
        if (max_oracle_calls is not None
                and (isinstance(max_oracle_calls, (bool, np.bool_))
                     or not isinstance(max_oracle_calls, (int, float, np.number))
                     or not np.isfinite(max_oracle_calls)
                     or int(max_oracle_calls) != max_oracle_calls or max_oracle_calls <= 0)):
            raise ValueError("max_oracle_calls must be a positive integer when provided")
        start = time.perf_counter()
        calls_at_start = oracle.stats.n_calls
        requested = kwargs.pop("max_vertices", None)     # the cycle owns this budget
        if requested is not None and requested != self.max_vertices:
            raise TypeError(
                f"max_vertices={requested!r} was passed to build(), but the memory-aware "
                f"cycle grows to the constructor's max_vertices={self.max_vertices}; set it "
                "there: MemoryAwareBuilder(max_vertices=...)")
        # A starting vertex set seeds round 0 only; later rounds resume on the hull.
        initial_vertices = kwargs.pop("initial_vertices", None)
        for key in ("initial_hull", "keep_hull", "monitor"):
            if key in kwargs:
                raise TypeError(f"{key!r} is managed by the memory-aware builder")
        schedule = self._schedule()
        target = getattr(schedule, "target_vertices", None)
        if (schedule is not None and schedule.name == "vertex-budget"
                and isinstance(target, int) and target <= len(axes)):
            # The schedule would clamp it to a simplex, discarding most of the
            # region at every compression; refuse it like a compressor does.
            raise ValueError(f"target_vertices={target} is below the {len(axes) + 1} "
                             f"vertices a {len(axes)}-D body needs")
        if schedule is not None:
            schedule.reset()
        if self.refiner is not None:
            self.refiner.reset()
        compressor = self._compressor(len(axes))
        live_capable = bool(getattr(self.inner_builder, "supports_live_hull", False))
        inner_ray_kw = getattr(self.inner_builder, "ray_kwargs", None)
        inner_origin = inner_ray_kw.get("origin") if isinstance(inner_ray_kw, dict) else None

        seed = int(getattr(self.inner_builder, "seed_rng", 0))
        ray_cache: dict = {}
        tabu: set = set()
        discarded: List[np.ndarray] = []
        seen_states: set = set()
        revisits = 0
        screened_total = 0
        excluded_total = 0

        cycle_trace: List[dict] = []
        combined_trace: List[dict] = []
        live: Optional[LiveHull] = None
        best: Optional[dict] = None
        last: Optional[RegionResult] = None
        stopped = ""

        def state_of(result: RegionResult, *, at_cap: bool) -> GrowthState:
            rows = result.trace
            tail = rows[-1] if rows else {}
            return GrowthState(
                dim=len(axes), iteration=len(rows), n_points=int(tail.get(
                    "n_vertices", result.polytope.n_vertices)),
                n_vertices=live.n_vertices if live is not None else result.polytope.n_vertices,
                n_facets=live.n_facets if live is not None else 0,
                volume=result.polytope.volume,
                hull_mb=live.memory_mb() if live is not None else 0.0,
                rss_mb=float(tail.get("rss_mb", 0.0)), memory_limit_mb=memory_limit_mb,
                oracle_calls=oracle.stats.n_calls, seconds=time.perf_counter() - start,
                max_vertices=self.max_vertices, at_cap=at_cap, trace=tuple(rows))

        try:
            for round_index in range(self.rounds + 1):
                remaining = None
                if max_seconds is not None:
                    remaining = max_seconds - (time.perf_counter() - start)
                    if remaining <= 0:
                        stopped = "time budget reached"
                        break
                remaining_calls = None
                if max_oracle_calls is not None:
                    remaining_calls = max_oracle_calls - (oracle.stats.n_calls - calls_at_start)
                    if remaining_calls <= 0:
                        stopped = "oracle-call budget reached"
                        break
                pending: dict = {}

                def monitor(state, _pending=pending, _round=round_index):
                    if schedule is None:
                        return None
                    state = dataclasses.replace(state, round=_round)
                    plan = schedule.decide(state)
                    if plan is None:
                        return None
                    _pending["plan"], _pending["state"] = plan, state
                    return f"compression scheduled: {plan.reason}"

                inner_kwargs = dict(kwargs, max_vertices=self.max_vertices,
                                    memory_limit_mb=memory_limit_mb, max_seconds=remaining,
                                    max_oracle_calls=remaining_calls)
                cap = schedule.memory_cap(memory_limit_mb) if schedule is not None else None
                if live_capable:
                    rng, screen_rng = _round_streams(seed, round_index)
                    inner_kwargs.update(initial_hull=live, keep_hull=True, monitor=monitor,
                                        rng=rng, ray_cache=ray_cache, tabu=tabu,
                                        max_hull_mb=cap)
                    if (self.facet_screen is not None
                            and getattr(self.inner_builder, "supports_facet_screen", False)):
                        inner_kwargs.update(
                            screen=self.facet_screen, screen_rng=screen_rng,
                            debt_points=np.array(discarded) if discarded else None)
                elif live is not None:
                    inner_kwargs.update(initial_vertices=live.vertices)
                if live is None and initial_vertices is not None:
                    inner_kwargs.update(initial_vertices=initial_vertices)
                try:
                    result = self.inner_builder.build(case, axes, oracle, **inner_kwargs)
                except MemoryBudgetExceeded as exc:
                    if best is None:
                        raise
                    stopped = f"memory budget exceeded before round {round_index} ({exc})"
                    break
                last = result
                round_screen = result.builder_info.get("screen")
                if isinstance(round_screen, dict):
                    screened_total += int(round_screen.get("screened", 0))
                    excluded_total += int(round_screen.get("excluded", 0))
                if live_capable:
                    live = result.live_hull
                    result.live_hull = None
                else:
                    if live is not None:
                        live.release()
                    live = LiveHull(result.polytope.vertices)

                for row in result.trace:
                    combined_trace.append(row | {"round": round_index, "phase": "expand"})
                expanded_volume = float(live.volume)
                refined = None
                if self.refiner is not None:
                    left = (None if max_seconds is None
                            else max(max_seconds - (time.perf_counter() - start), 0.0))
                    calls_left = (None if max_oracle_calls is None else
                                  max_oracle_calls - (oracle.stats.n_calls - calls_at_start))
                    if calls_left is not None and calls_left <= 0:
                        stopped = "oracle-call budget reached"
                    elif left is not None and left <= 0.0:
                        stopped = "time budget reached"
                    else:
                        ref_ray_kw = getattr(self.refiner, "ray_kwargs", None)
                        extra_refine = (
                            {"ray_kwargs": {"origin": inner_origin}}
                            if inner_origin is not None
                            and isinstance(ref_ray_kw, dict)
                            and "origin" not in ref_ray_kw
                            else {}
                        )
                        refined = self.refiner.refine(
                            live, oracle, axes, max_moves=self.refine_moves,
                            max_hull_mb=cap, ray_cache=ray_cache, max_seconds=left,
                            max_oracle_calls=calls_left, **extra_refine)
                        for row in refined["trace"]:
                            combined_trace.append(
                                row | {"round": round_index, "phase": "refine"})
                volume = float(live.volume)
                # A state within the vertex cap beats any state beyond it (a
                # random seed can start past the cap); the potential the
                # schedule watches counts capped states only, so it never falls.
                within = live.n_vertices <= self.max_vertices
                if (best is None or (within, volume) > (best["within"], best["volume"])):
                    best = {"vertices": live.vertices, "volume": volume, "within": within,
                            "round": round_index, "stopped": result.stopped_because}
                cycle_trace.append({
                    "round": round_index, "phase": "expand",
                    "n_vertices": live.n_vertices, "n_facets": live.n_facets,
                    "hull_mb": live.memory_mb(), "volume": expanded_volume,
                    "best_volume": best["volume"] if best["within"] else 0.0,
                    "oracle_calls": oracle.stats.n_calls,
                    "seconds": time.perf_counter() - start,
                    "stopped_because": result.stopped_because,
                    "cached_rays": result.builder_info.get("cached_rays", 0),
                })
                if refined is not None:
                    cycle_trace.append({
                        "round": round_index, "phase": "refine",
                        "n_vertices": live.n_vertices, "n_facets": live.n_facets,
                        "hull_mb": live.memory_mb(), "volume": volume,
                        "moves": refined["moves"], "accepted": refined["accepted"],
                        "oracle_calls": oracle.stats.n_calls,
                        "seconds": time.perf_counter() - start,
                    })
                LOGGER.info("memory-aware round %d: expanded to %d vertices (%d facets), "
                            "volume %.6g", round_index, live.n_vertices, live.n_facets, volume)

                if stopped:
                    break
                if schedule is None:
                    stopped = result.stopped_because
                    break
                plan = pending.get("plan")
                state = pending.get("state")
                if plan is None and result.stopped_because.startswith("memory cap"):
                    # The next insertion would have passed the hard cap: that
                    # is the memory trigger, one step early.
                    state = dataclasses.replace(state_of(result, at_cap=False),
                                                round=round_index, hull_mb=float(cap or 0.0))
                    plan = schedule.decide(state) or schedule.at_cap(state)
                # Ray-growth strategies may report strategy-specific stop strings
                # at the vertex cap, so check live.n_vertices directly as well.
                at_vertex_cap = (result.stopped_because.startswith("max_vertices")
                                 or (live.n_vertices >= self.max_vertices
                                     and not result.stopped_because.startswith(
                                         ("time budget", "oracle-call budget",
                                          "memory budget", "volume converged"))))
                if plan is None:
                    if not at_vertex_cap:
                        # Time, oracle calls, memory guard, convergence or a
                        # fully explored queue: compressing would not help.
                        stopped = result.stopped_because
                        break
                    state = dataclasses.replace(state_of(result, at_cap=True),
                                                round=round_index)
                    plan = schedule.at_cap(state)
                    if plan is None:
                        stopped = "vertex cap reached and the schedule declined to compress"
                        break
                if round_index == self.rounds:
                    stopped = "round limit reached"
                    break
                reason = schedule.finished(cycle_trace)
                if reason:
                    stopped = reason
                    break

                before = live.vertices
                compressed = compressor.compress_live(live, axes=axes, **plan.rules())
                # What the compression threw away is not re-added by the next
                # expansion (see ExpansiveBuilder's ``tabu``).
                survivors = {point_key(v) for v in live.vertices}
                for v in before:
                    pkey = point_key(v)
                    if pkey not in survivors and pkey not in tabu:
                        tabu.add(pkey)
                        discarded.append(v)
                schedule.observe(state, plan, compressed)
                cycle_trace.append({
                    "round": round_index, "phase": "compress",
                    "n_vertices": live.n_vertices, "n_facets": live.n_facets,
                    "hull_mb": live.memory_mb(), "volume": float(live.volume),
                    "volume_loss_pct": compressed.volume_loss_pct,
                    "target_vertices": plan.target_vertices,
                    "max_volume_loss_pct": plan.max_volume_loss_pct,
                    "max_memory_mb": plan.max_memory_mb,
                    "max_vertex_loss": plan.max_vertex_loss,
                    "reason": plan.reason,
                    "oracle_calls": oracle.stats.n_calls,
                    "seconds": time.perf_counter() - start,
                })
                LOGGER.info("memory-aware round %d: compressed to %d vertices (%.3f%% volume "
                            "lost; %s)", round_index, live.n_vertices,
                            compressed.volume_loss_pct, plan.reason)
                reason = schedule.finished(cycle_trace)
                if reason:
                    stopped = reason
                    # Stop before starting an unprofitable next round; the best
                    # expanded vertex set recorded so far is returned below.
                    break
                # Revisiting a previous vertex set does not cause an infinite
                # loop because discarded points are tabu and each round uses an
                # independent RNG stream.  Termination comes from the
                # schedule's potential (best volume); revisits are only counted.
                fingerprint = _fingerprint(live.vertices)
                revisits += fingerprint in seen_states
                seen_states.add(fingerprint)
        finally:
            if live is not None:
                live.release()

        assert best is not None and last is not None
        # Only the vertex set leaves this builder; the hull was released above.
        polytope = Polytope.with_known_volume(best["vertices"], best["volume"], axes=axes)
        screen_report = None
        if self.facet_screen is not None:
            screen_report = dict(self.facet_screen.report())
            if "screened" in screen_report:
                screen_report["screened"] = screened_total
            if "excluded" in screen_report:
                screen_report["excluded"] = excluded_total
        info = dict(last.builder_info)
        info.update(
            builder=self.name,
            inner_builder=getattr(self.inner_builder, "name", str(self.inner_builder)),
            compressor=getattr(compressor, "name", str(compressor)),
            schedule=schedule.report() if schedule is not None else None,
            max_vertices=self.max_vertices,
            target_vertices=self.target_vertices,
            rounds=self.rounds,
            cycles=cycle_trace,
            best_round=best["round"],
            n_compressions=sum(c["phase"] == "compress" for c in cycle_trace),
            cached_rays=sum(c.get("cached_rays", 0) for c in cycle_trace),
            tabu_points=len(tabu),
            revisited_states=revisits,
            screen=screen_report,
            facet_screen=screen_report,
            refiner=self.refiner.report() if self.refiner is not None else None,
            guarantee=single_oracle_guarantee(oracle),
            theorem_backed=False,
        )
        return RegionResult(
            polytope=polytope,
            trace=combined_trace,
            oracle_stats=oracle.stats.as_dict() | oracle.describe(),
            builder_info=info,
            seconds=time.perf_counter() - start,
            stopped_because=stopped or best["stopped"],
        )


def _fingerprint(vertices: np.ndarray) -> bytes:
    """Order-independent identity of a vertex set, robust to 1e-9 relative noise."""
    scale = max(float(np.abs(vertices).max()), 1.0)
    rounded = np.round(vertices / scale, 9)
    return np.ascontiguousarray(rounded[np.lexsort(rounded.T[::-1])]).tobytes()


def _round_streams(seed: int, round_index: int) -> Tuple[random.Random, np.random.Generator]:
    """Independent, reproducible random streams for one round.

    Round 0 uses ``random.Random(seed)`` - exactly what a plain expansion
    uses - so the first round of a memory-aware run equals a plain run.  Every
    later round hashes ``(seed, round)`` through :class:`numpy.random.SeedSequence`,
    whose outputs for distinct inputs are designed to be independent: a
    round's randomness depends neither on the other rounds nor on how many
    draws they made.
    """
    sequence = np.random.SeedSequence([seed, round_index, 0x4D454D])
    policy = (random.Random(seed) if round_index == 0
              else random.Random(int(sequence.generate_state(1, dtype=np.uint64)[0])))
    return policy, np.random.default_rng(sequence.spawn(1)[0])
