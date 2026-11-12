"""Facet-selection policies - deciding which facet of the polytope to break next.

The expansive construction repeatedly picks a facet, fires a ray through its
centroid, and replaces it with the facets formed by the boundary point it finds.
Facet selection governs growth efficiency: repeatedly subdividing one region of
the boundary triggers a Matthew effect where its small child facets crowd the
queue and stall overall boundary coverage.

Four single measures and three adaptive policies are provided.

Single measures
---------------
``volume-gain``   maximise the hypervolume the candidate point would add.  The
                  greedy choice, and the most informative - but it needs a
                  boundary search for *every* facet in the queue, so its cost
                  per iteration grows with the queue.
``facet-area``    maximise the facet's own ``(d-1)``-volume.  Cheap.
``cone-volume``   maximise the volume of the cone the facet forms with the
                  origin; rewards facets that are both large and far out. Cheap.
``generation``    minimise the facet's generation counter, i.e. always work on
                  the oldest part of the boundary.  This is the direct
                  anti-Matthew choice, and it is cheap.
``random``        uniform pick; the baseline the others are measured against.

Adaptive policies
-----------------
The adaptive policies treat the cheap measures as *arms* of a bandit: each arm
nominates a facet, the realised hypervolume gain of a nomination is the reward,
and the policy learns which measure is paying off at this stage of the growth.

``adaptive-ema``       exponential recency-weighted average of each arm's
                       reward, then pick the best arm.  Full-information: every
                       arm's nomination is scored each round.
``adaptive-egreedy``   the same, but with probability ``epsilon`` a uniformly
                       random arm is taken instead.
``adaptive-gradient``  a gradient bandit that samples an arm before the expensive
                       boundary search; preferences are updated by
                       ``H_a += alpha (R - R_bar)(1 - pi_a)`` for the arm played
                       and ``H_k -= alpha (R - R_bar) pi_k`` for the others,
                       with the arm sampled from ``softmax(H)``.

``lazy=True`` (the default) evaluates boundary gains only for the facets
nominated by the arms, reducing per-iteration cost from ``O(|queue|)``
boundary searches to at most ``n_arms``.  ``lazy=False`` reproduces the original,
unaccelerated method: every facet is scored (one boundary search) as soon as
it is in the queue.  The rule that picks the facet is the same; only the cost
differs, and the time at which each gain is measured.

Two-step selection
------------------
A :class:`FacetScreen` can run first and decide which facets enter the pool the
policy chooses from.  The registered ``debt`` screen charges facets for the
vertices a compression discarded beyond them and passes that debt on to their
children, so a resumed expansion stops re-exploring what was just cut away.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence

import numpy as np

from OpenRC.registry import Registry

#: Registry of facet-selection policy factories.
FACET_POLICIES: Registry = Registry("facet policy")


class SelectionContext:
    """What a policy may look at, with the expensive parts computed on demand.

    Read-only view over the builder's facet queue that memoises per-facet
    geometric measures.  Evaluating :meth:`gain` triggers a boundary ray search
    on first access, whereas area, cone, and generation policies avoid ray
    searches altogether.

    ``cache`` lets the builder keep the area and cone memos *across*
    iterations (``{"area": {fid: value}, "cone": {...}}``): a facet's geometry
    never changes while it is in the queue, so each is computed once in its
    lifetime instead of once per iteration.  :meth:`areas` and :meth:`cones`
    return the measure of every facet at once.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        facets: Sequence[Sequence[int]],
        generations: Sequence[int],
        facet_ids: Sequence[int],
        gain_fn: Callable[[int], float],
        rng: random.Random,
        cache: Optional[dict] = None,
    ) -> None:
        raw_vertices = np.asarray(vertices)
        if raw_vertices.dtype == bool or raw_vertices.ndim != 2:
            raise ValueError("vertices must be a 2-D numeric array")
        if (
            isinstance(facets, (str, bytes, bytearray))
            or isinstance(generations, (str, bytes, bytearray))
            or isinstance(facet_ids, (str, bytes, bytearray))
        ):
            raise TypeError("facets, generations, and facet_ids must be sequences, not strings")
        if not callable(gain_fn):
            raise TypeError("gain_fn must be callable")
        if cache is not None and not isinstance(cache, dict):
            raise TypeError("cache must be a dict or None")
        self.vertices = vertices
        self.facets = facets
        self.generations = list(generations)
        self.facet_ids = list(facet_ids)
        if not (len(self.facets) == len(self.generations) == len(self.facet_ids)):
            raise ValueError("facets, generations, and facet_ids must have the same length")
        self._gain_fn = gain_fn        # keyed by facet id, memoised by the builder
        self.rng = rng
        cache = cache if cache is not None else {}
        self._area: dict[int, float] = cache.setdefault("area", {})
        self._cone: dict[int, float] = cache.setdefault("cone", {})

    def _fill(self, memo: dict, measure,
              batch_measure: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Fill missing measures in bounded batches, preserving cache/order semantics."""
        if batch_measure is None:
            for i, fid in enumerate(self.facet_ids):
                if fid not in memo:
                    memo[fid] = measure(self.facet_points(i))
        else:
            pending: dict[int, int] = {}    # facet ID -> first queue position
            size, limit = -1, 1

            def flush() -> None:
                if len(pending) >= 4:
                    points = np.array([self.facet_points(i) for i in pending.values()])
                    values = batch_measure(points)
                    for fid, value in zip(pending, values):
                        memo[fid] = float(value)
                else:                     # batching tiny tails would cost more
                    for fid, i in pending.items():
                        memo[fid] = measure(self.facet_points(i))
                pending.clear()

            for i, fid in enumerate(self.facet_ids):
                if fid in memo or fid in pending:
                    continue
                k = len(self.facets[i])
                if pending and k != size:
                    flush()                # a batch holds facets of one size only
                if not pending:
                    size = k
                    # Include a possible cone apex and the batch helper's
                    # edges/Grams.  Never materialise all queued facet points.
                    per_row = 8 * (3 * (k + 1) * self.vertices.shape[1]
                                   + 2 * (k + 1) ** 2 + 8 * (k + 1))
                    limit = max(1, min(1024, 2 ** 20 // per_row))
                pending[fid] = i
                if len(pending) >= limit:
                    flush()
            flush()
        return np.array([memo[fid] for fid in self.facet_ids], dtype=float)

    def areas(self) -> np.ndarray:
        """:meth:`area` of every facet, in queue order."""
        from OpenRC.geometry.measures import simplex_volume, simplex_volumes

        # Small 3-D updates usually add only three facets; retain their cheap
        # scalar path.  Higher-dimensional queues repay batched determinants.
        batch = simplex_volumes if self.vertices.shape[1] >= 4 else None
        return self._fill(self._area, simplex_volume, batch)

    def cones(self) -> np.ndarray:
        """:meth:`cone` of every facet, in queue order."""
        from OpenRC.geometry.measures import cone_volume, simplex_volumes

        def batch(points: np.ndarray) -> np.ndarray:
            apex = np.zeros((len(points), 1, points.shape[2]))
            return simplex_volumes(np.concatenate([points, apex], axis=1))

        return self._fill(self._cone, cone_volume,
                          batch if self.vertices.shape[1] >= 4 else None)

    @property
    def n_facets(self) -> int:
        return len(self.facets)

    def facet_points(self, i: int) -> np.ndarray:
        return self.vertices[list(self.facets[i])]

    def area(self, i: int) -> float:
        """``(d-1)``-volume of facet ``i`` (cheap, memoised)."""
        fid = self.facet_ids[i]
        if fid not in self._area:
            from OpenRC.geometry.measures import simplex_volume

            self._area[fid] = simplex_volume(self.facet_points(i))
        return self._area[fid]

    def cone(self, i: int) -> float:
        """Volume of the cone facet ``i`` forms with the origin (cheap, memoised)."""
        fid = self.facet_ids[i]
        if fid not in self._cone:
            from OpenRC.geometry.measures import cone_volume

            self._cone[fid] = cone_volume(self.facet_points(i))
        return self._cone[fid]

    def generation(self, i: int) -> int:
        """Generation counter of facet ``i`` (0 for the initial polytope)."""
        return self.generations[i]

    def gain(self, i: int) -> float:
        """Hypervolume the candidate boundary point of facet ``i`` would add.

        **Expensive**: triggers a boundary search the first time it is asked for
        a given facet.  The builder memoises the result by facet id.
        """
        return self._gain_fn(self.facet_ids[i])


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class FacetPolicy(ABC):
    """Chooses the next facet to break."""

    name: str = "policy"

    @abstractmethod
    def select(self, ctx: SelectionContext) -> int:
        """Return the index of the facet to break, in ``ctx.facets``."""

    def observe(self, ctx: SelectionContext, chosen: int, realised_gain: float) -> None:
        """Feedback hook; adaptive policies learn here, fixed measures ignore it."""

    def reset(self) -> None:
        """Clear learned state so one policy object can be reused across runs."""

    def report(self) -> dict:
        """Diagnostics for the run record (arm probabilities, pick counts, ...)."""
        return {"policy": self.name}


# --------------------------------------------------------------------------
# Single-measure policies
# --------------------------------------------------------------------------
class _ArgMaxPolicy(FacetPolicy):
    """Pick the facet maximising a scalar measure; ties broken at random.

    ``batch`` optionally names an accessor of the context (``"areas"``,
    ``"cones"``) that returns the measure for every facet at once; it must
    agree with ``measure`` element by element.
    """

    def __init__(self, measure: Callable[[SelectionContext, int], float], name: str,
                 maximise: bool = True, rng: Optional[random.Random] = None,
                 batch: Optional[str] = None) -> None:
        if not isinstance(maximise, (bool, np.bool_)):
            raise ValueError("maximise must be a boolean")
        self._measure = measure
        self.name = name
        self._maximise = bool(maximise)
        self._rng = rng
        self._batch = batch

    def select(self, ctx: SelectionContext) -> int:
        rng = self._rng or ctx.rng
        if self._batch is not None and hasattr(ctx, self._batch):
            values = list(getattr(ctx, self._batch)())
        else:
            values = [self._measure(ctx, i) for i in range(ctx.n_facets)]
        best = max(values) if self._maximise else min(values)
        winners = [i for i, v in enumerate(values) if v == best]
        return rng.choice(winners)


@FACET_POLICIES.register("volume-gain", doc="Greedy: maximise the hypervolume the new vertex adds.")
def _volume_gain(**kwargs) -> FacetPolicy:
    return _ArgMaxPolicy(lambda ctx, i: ctx.gain(i), "volume-gain", **kwargs)


@FACET_POLICIES.register("facet-area", doc="Maximise the facet's own (d-1)-volume (cheap).")
def _facet_area(**kwargs) -> FacetPolicy:
    return _ArgMaxPolicy(lambda ctx, i: ctx.area(i), "facet-area", batch="areas", **kwargs)


@FACET_POLICIES.register("cone-volume", doc="Maximise the facet-to-origin cone volume (cheap).")
def _cone_volume(**kwargs) -> FacetPolicy:
    return _ArgMaxPolicy(lambda ctx, i: ctx.cone(i), "cone-volume", batch="cones", **kwargs)


@FACET_POLICIES.register("generation", doc="Always break the oldest facet (anti-Matthew, cheap).")
def _generation(**kwargs) -> FacetPolicy:
    return _ArgMaxPolicy(lambda ctx, i: ctx.generation(i), "generation",
                         maximise=False, **kwargs)


@FACET_POLICIES.register("random", doc="Uniformly random facet - the baseline.")
class RandomFacetPolicy(FacetPolicy):
    """Uniformly random facet choice."""

    name = "random"

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng

    def select(self, ctx: SelectionContext) -> int:
        rng = self._rng or ctx.rng
        return rng.randrange(ctx.n_facets)


# --------------------------------------------------------------------------
# Adaptive (bandit) policies
# --------------------------------------------------------------------------
DEFAULT_ARMS = ("facet-area", "cone-volume", "generation")
DEFAULT_ARMS_WITH_GREEDY = ("facet-area", "cone-volume", "generation", "volume-gain")


class _BanditPolicy(FacetPolicy):
    """Shared machinery for the adaptive policies.

    Each arm is itself a :class:`FacetPolicy`; a round consists of asking every
    arm for a nomination, choosing one arm, and breaking that arm's facet.
    """

    def __init__(self, arms: Sequence[str] = DEFAULT_ARMS, *,
                 lazy: bool = True, rng: Optional[random.Random] = None) -> None:
        if isinstance(arms, (str, bytes, bytearray)):
            raise TypeError("arms must be a sequence of arm names, not a string")
        if not isinstance(lazy, (bool, np.bool_)):
            raise ValueError("lazy must be a boolean")
        self.arm_names = list(arms)
        if not self.arm_names:
            raise ValueError("an adaptive policy needs at least one arm")
        self.arms: List[FacetPolicy] = [FACET_POLICIES.get(a)() for a in self.arm_names]
        self.lazy = bool(lazy)
        self._rng = rng
        self.n_arms = len(self.arms)
        self.pick_counts = [0] * self.n_arms
        self._last_arm: Optional[int] = None
        self._last_nominations: List[int] = []

    # ---- to be provided by the concrete policy ---------------------------
    @abstractmethod
    def _choose_arm(self, rewards: Sequence[float], rng: random.Random) -> int:
        """Pick an arm given this round's nomination rewards."""

    def _update(self, arm: int, reward: float) -> None:
        """Learn from the realised reward of the played arm."""

    # ---- common flow ------------------------------------------------------
    def _score_queue(self, ctx: SelectionContext) -> None:
        """``lazy=False``: score every queued facet, as the unaccelerated method did.

        Gains are memoised per facet by the builder, so each facet costs one
        boundary search, paid when it first meets the policy.
        """
        if not self.lazy:
            for i in range(ctx.n_facets):
                ctx.gain(i)

    def select(self, ctx: SelectionContext) -> int:
        rng = self._rng or ctx.rng
        self._score_queue(ctx)
        nominations = [arm.select(ctx) for arm in self.arms]
        self._last_nominations = nominations

        # In lazy mode, score only the arm-nominated facets (cached by facet ID).
        rewards = [ctx.gain(i) for i in nominations]

        arm = self._choose_arm(rewards, rng)
        self._last_arm = arm
        self.pick_counts[arm] += 1
        self._update(arm, rewards[arm])
        return nominations[arm]

    def reset(self) -> None:
        self.pick_counts = [0] * self.n_arms
        self._last_arm = None

    def report(self) -> dict:
        return {
            "policy": self.name,
            "arms": self.arm_names,
            "pick_counts": list(self.pick_counts),
            "lazy": self.lazy,
        }


@FACET_POLICIES.register(
    "adaptive-ema",
    doc="Bandit over the cheap measures, exponential recency-weighted rewards.",
)
class AdaptiveEmaPolicy(_BanditPolicy):
    """Exponential recency-weighted average over arms, then greedy on the average.

    ``w_k <- alpha * w_k + (1 - alpha) * reward_k`` for **every** arm each round
    (full information), then play ``argmax w``.  ``alpha = 0`` collapses to
    "play whichever arm nominated the best facet this round", which is the
    setting the geometric-programming study used.
    """

    name = "adaptive-ema"

    def __init__(self, arms: Sequence[str] = DEFAULT_ARMS, *, alpha: float = 0.5,
                 **kwargs) -> None:
        super().__init__(arms, **kwargs)
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.number)):
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.weights = [0.0] * self.n_arms

    def _choose_arm(self, rewards: Sequence[float], rng: random.Random) -> int:
        for k in range(self.n_arms):
            self.weights[k] = self.alpha * self.weights[k] + (1.0 - self.alpha) * rewards[k]
        best = max(self.weights)
        return rng.choice([k for k, w in enumerate(self.weights) if w == best])

    def reset(self) -> None:
        super().reset()
        self.weights = [0.0] * self.n_arms

    def report(self) -> dict:
        info = super().report()
        info.update(alpha=self.alpha, weights=[round(w, 6) for w in self.weights])
        return info


@FACET_POLICIES.register(
    "adaptive-egreedy",
    doc="Recency-weighted bandit with an epsilon-greedy exploration step.",
)
class AdaptiveEpsilonGreedyPolicy(AdaptiveEmaPolicy):
    """:class:`AdaptiveEmaPolicy` that explores uniformly with probability ``epsilon``."""

    name = "adaptive-egreedy"

    def __init__(self, arms: Sequence[str] = DEFAULT_ARMS, *, alpha: float = 0.5,
                 epsilon: float = 0.1, **kwargs) -> None:
        super().__init__(arms, alpha=alpha, **kwargs)
        if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, (int, float, np.number)):
            raise ValueError("epsilon must lie in [0, 1]")
        self.epsilon = float(epsilon)
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")

    def _choose_arm(self, rewards: Sequence[float], rng: random.Random) -> int:
        greedy = super()._choose_arm(rewards, rng)
        if rng.random() < self.epsilon:
            return rng.randrange(self.n_arms)
        return greedy

    def report(self) -> dict:
        info = super().report()
        info.update(epsilon=self.epsilon)
        return info


@FACET_POLICIES.register(
    "adaptive-gradient",
    doc="Gradient bandit over the cheap measures (softmax preferences).",
)
class AdaptiveGradientPolicy(_BanditPolicy):
    """Gradient bandit with a running-average reward baseline.

    Rewards are hypervolume gains, whose scale depends on the region's units and
    dimension.  Feeding them to a softmax raw would saturate it, so they are
    divided by ``reward_scale``; when ``None``, rewards are normalised by
    their running mean absolute value so preference updates remain
    well-scaled across dimensions and units (e.g. MW^3 vs. MW^8).
    """

    name = "adaptive-gradient"

    def __init__(self, arms: Sequence[str] = DEFAULT_ARMS, *, alpha: float = 0.1,
                 reward_scale: Optional[float] = None, **kwargs) -> None:
        super().__init__(arms, **kwargs)
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.number)):
            raise ValueError("alpha must be finite and strictly positive")
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and strictly positive")
        if reward_scale is not None and (
            isinstance(reward_scale, (bool, np.bool_))
            or not isinstance(reward_scale, (int, float, np.number))
            or not np.isfinite(reward_scale)
            or reward_scale <= 0.0
        ):
            raise ValueError("reward_scale must be finite and strictly positive")
        self.reward_scale = None if reward_scale is None else float(reward_scale)
        self.preferences = [0.0] * self.n_arms
        self._probs = [1.0 / self.n_arms] * self.n_arms
        self._reward_sum = 0.0
        self._magnitude_sum = 0.0
        self._n = 0

    def _softmax(self) -> List[float]:
        top = max(self.preferences)
        weights = [math.exp(h - top) for h in self.preferences]  # shifted for stability
        total = sum(weights)
        return [w / total for w in weights]

    def _scale(self, reward: float) -> float:
        if self.reward_scale:
            return reward / self.reward_scale
        # Adaptive normalisation by the running mean |reward|, including this
        # observation.  The observation count itself is advanced in _update.
        self._magnitude_sum += abs(reward)
        mean_magnitude = self._magnitude_sum / (self._n + 1)
        return reward / mean_magnitude if mean_magnitude > 0 else 0.0

    def select(self, ctx: SelectionContext) -> int:
        """Sample one arm before computing the selected facet boundary gain.

        Unlike the full-information EMA policies, gradient bandits require only
        the reward of the arm actually played.  Deferring that reward to
        :meth:`observe` avoids three redundant solver-backed ray searches per
        iteration when the default three cheap arms are used.
        """
        rng = self._rng or ctx.rng
        self._score_queue(ctx)
        nominations = [arm.select(ctx) for arm in self.arms]
        self._last_nominations = nominations
        arm = self._choose_arm((), rng)
        self._last_arm = arm
        self.pick_counts[arm] += 1
        return nominations[arm]

    def observe(self, ctx: SelectionContext, chosen: int, realised_gain: float) -> None:
        if self._last_arm is not None:
            self._update(self._last_arm, float(realised_gain))

    def _choose_arm(self, rewards: Sequence[float], rng: random.Random) -> int:
        self._probs = self._softmax()
        draw = rng.random()
        cumulative = 0.0
        for k, p in enumerate(self._probs):
            cumulative += p
            if draw <= cumulative:
                return k
        return self.n_arms - 1

    def _update(self, arm: int, reward: float) -> None:
        scaled = self._scale(reward)
        self._n += 1
        self._reward_sum += scaled
        baseline = self._reward_sum / max(self._n, 1)
        advantage = scaled - baseline
        for k in range(self.n_arms):
            if k == arm:
                self.preferences[k] += self.alpha * advantage * (1.0 - self._probs[k])
            else:
                self.preferences[k] -= self.alpha * advantage * self._probs[k]

    def reset(self) -> None:
        super().reset()
        self.preferences = [0.0] * self.n_arms
        self._probs = [1.0 / self.n_arms] * self.n_arms
        self._reward_sum = 0.0
        self._magnitude_sum = 0.0
        self._n = 0

    def report(self) -> dict:
        info = super().report()
        info.update(
            alpha=self.alpha,
            preferences=[round(h, 4) for h in self.preferences],
            probabilities=[round(p, 4) for p in self._probs],
        )
        return info


def get_facet_policy(name: str, **kwargs) -> FacetPolicy:
    """Instantiate a registered policy: ``get_facet_policy("adaptive-ema", alpha=0.5)``."""
    return FACET_POLICIES.get(name)(**kwargs)


def list_facet_policies() -> List[tuple]:
    return FACET_POLICIES.describe()


# --------------------------------------------------------------------------
# Facet screens: step 1 of a two-step selection
# --------------------------------------------------------------------------
#: Registry of facet screens (see :class:`FacetScreen`).
FACET_SCREENS: Registry = Registry("facet screen")


class FacetScreen(ABC):
    """Step 1 of a two-step facet selection: who enters this round's pool.

    Each facet carries a *debt* (0 unless something assigned one).  Every
    iteration the screen decides which facets form the selection pool; the
    facet-selection policy (step 2) then picks among the pool only.  A screen
    that keeps every debt-free facet and draws its randomness from its own
    stream leaves a debt-free run bit-for-bit unchanged.
    """

    name: str = "screen"

    def reset(self) -> None:
        """Clear per-run counters."""

    @abstractmethod
    def pool(self, debts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Boolean mask over the queue: which facets may be selected now."""

    def inherit(self, debt: float) -> float:
        """Debt a child facet takes over from the facet it was split from."""
        return 0.0

    def report(self) -> dict:
        return {"screen": self.name}


@FACET_SCREENS.register(
    "debt",
    doc="Keep a facet in the pool with probability exp(-debt / temperature); children "
        "inherit their parent's debt.",
)
class DebtScreen(FacetScreen):
    r"""Screen out explored-and-discarded facets; children inherit their parents' debts.

    A compression throws vertices away.  Each discarded vertex ``v`` lies
    beyond the facets of the compressed hull it can see; the cap it cut off,
    ``conv(P ∪ {v}) \ P``, is the union of the cones ``conv(F, v)`` over those
    facets.  Re-expanding such a facet probes directions whose vertices were
    already pruned for low marginal volume, so the facet is assigned a
    *debt* - by default (``weighting="count"``) the number of discarded
    vertices that lie beyond it; ``"cone"`` charges the full unit only to the
    facet each vertex sits over most and ``cone(F, v) / max cone`` to the
    others (see :func:`facet_debts`).

    Step 1 keeps a facet in the pool with probability

    .. math::  \pi(F) = \max\big(e^{-\mathrm{debt}(F)/\tau},\; \pi_{\min}\big),

    so one discarded vertex straight above a facet (debt 1, ``tau = 0.5``)
    leaves it a 13.5 % chance per iteration, and it is never excluded for
    good.  When a facet with debt ``D`` is broken anyway, each of its children
    starts with ``inheritance * D``, propagating the penalty to subdivisions
    of previously pruned caps.  Debt-free facets are always kept and
    cost no random draw.

    Parameters
    ----------
    temperature:
        ``tau`` above; smaller screens harder.
    inheritance:
        Share of a parent's debt each child inherits (1 = full inheritance).
    min_keep:
        ``pi_min`` above - the small chance a facet always keeps.
    weighting:
        How a discarded vertex charges the facets it sees; see above.
    """

    name = "debt"

    def __init__(self, *, temperature: float = 0.5, inheritance: float = 0.8,
                 min_keep: float = 0.01, weighting: str = "count") -> None:
        if weighting not in ("count", "cone"):
            raise ValueError("weighting must be 'count' or 'cone'")
        self.weighting = weighting
        if (
            isinstance(temperature, (bool, np.bool_))
            or not isinstance(temperature, (int, float, np.number))
            or not (np.isfinite(temperature) and temperature > 0.0)
        ):
            raise ValueError("temperature must be finite and positive")
        if (
            isinstance(inheritance, (bool, np.bool_))
            or not isinstance(inheritance, (int, float, np.number))
            or not 0.0 <= inheritance <= 1.0
        ):
            raise ValueError("inheritance must lie in [0, 1]")
        if (
            isinstance(min_keep, (bool, np.bool_))
            or not isinstance(min_keep, (int, float, np.number))
            or not 0.0 < min_keep <= 1.0
        ):
            raise ValueError("min_keep must lie in (0, 1]")
        self.temperature = float(temperature)
        self.inheritance = float(inheritance)
        self.min_keep = float(min_keep)
        self.reset()

    def reset(self) -> None:
        self.n_screened = 0
        self.n_excluded = 0

    def keep_probability(self, debts: np.ndarray) -> np.ndarray:
        raw = np.asarray(debts)
        if raw.dtype == bool:
            raise ValueError("debts must be numeric, not boolean")
        return np.maximum(np.exp(-np.asarray(raw, dtype=float) / self.temperature),
                          self.min_keep)

    def pool(self, debts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raw = np.asarray(debts)
        if raw.dtype == bool or raw.ndim != 1:
            raise ValueError("debts must be a 1-D numeric array")
        debts = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(debts)):
            raise ValueError("debts must contain only finite values")
        keep = debts <= 0.0
        indebted = np.flatnonzero(~keep)
        if len(indebted):
            keep[indebted] = rng.random(len(indebted)) < self.keep_probability(
                debts[indebted])
            self.n_screened += len(indebted)
            self.n_excluded += int(len(indebted) - keep[indebted].sum())
        return keep

    def inherit(self, debt: float) -> float:
        return self.inheritance * float(debt)

    def report(self) -> dict:
        return {"screen": self.name, "temperature": self.temperature,
                "inheritance": self.inheritance, "min_keep": self.min_keep,
                "weighting": self.weighting,
                "screened": self.n_screened, "excluded": self.n_excluded}


def facet_debts(live, points: np.ndarray, *, weighting: str = "count") -> np.ndarray:
    """Debt of every facet of ``live`` owed to the discarded ``points``.

    ``weighting="count"``: a facet owes one unit to every discarded point that
    lies beyond it (can see it) - its debt is the number of discarded vertices
    that belong to it.  ``"cone"``: the unit goes in full only to the facet the
    point sits over most (largest cone ``conv(F, v)``), and in proportion
    ``cone / max cone`` to the other facets it sees.

    Rows follow ``live.facet_positions()``.  A point inside the hull (grown
    over since it was discarded) sees no facet and charges nothing.
    """
    from OpenRC.geometry.live_hull import cone_volumes

    if weighting not in ("count", "cone"):
        raise ValueError("weighting must be 'count' or 'cone'")
    if not all(hasattr(live, attr) for attr in ("points", "simplices", "equations", "n_facets")):
        raise TypeError(f"live must be a LiveHull; got {type(live).__name__}")
    hull_points, simplices = live.points, live.simplices
    debts = np.zeros(live.n_facets, dtype=float)
    eq = live.equations
    dim = eq.shape[1] - 1
    raw_in = np.asarray(points)
    if raw_in.dtype == bool:
        raise ValueError("points must be numeric, not boolean")
    raw = np.asarray(raw_in, dtype=float)
    if raw.size == 0:
        if raw.ndim == 2 and raw.shape[1] not in (0, dim):
            raise ValueError(f"points must have dimension {dim}; got {raw.shape}")
        return debts
    pts = np.atleast_2d(raw)
    if pts.ndim != 2 or pts.shape[1] != dim:
        raise ValueError(f"points must have shape (n, {dim}); got {raw.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")
    coord = max(
        float(np.abs(hull_points).max(initial=0.0)),
        float(np.abs(pts).max(initial=0.0)),
        1.0,
    )
    tol = 1e2 * np.finfo(float).eps * coord
    if weighting == "count":
        normals = eq[:, :-1]
        offsets = eq[:, -1:]
        for start in range(0, len(pts), 256):
            chunk = pts[start:start + 256]
            debts += np.sum(normals @ chunk.T + offsets > tol, axis=1)
        return debts
    for v in pts:
        visible = np.flatnonzero(eq[:, :-1] @ v + eq[:, -1] > tol)
        if len(visible) == 0:
            continue
        cones = cone_volumes(hull_points, simplices[visible], v)
        top = float(cones.max())
        if top > 0.0:
            debts[visible] += cones / top
    return debts


def get_facet_screen(name: str, **kwargs) -> FacetScreen:
    return FACET_SCREENS.get(name)(**kwargs)


def list_facet_screens() -> List[tuple]:
    return FACET_SCREENS.describe()


def resolve_facet_screen(screen, **kwargs) -> Optional[FacetScreen]:
    """A name, an instance, or ``None`` (no screening)."""
    if isinstance(screen, str) and screen != "none":
        return get_facet_screen(screen, **kwargs)
    if kwargs:
        raise TypeError("screen keyword arguments require a screen name")
    if screen is None or screen == "none":
        return None
    if isinstance(screen, FacetScreen):
        return screen
    raise TypeError(f"screen must be None, a registered name or a FacetScreen; got {screen!r}")
