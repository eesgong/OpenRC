"""Random and randomised-greedy baseline compressors for benchmarking.

Neither is meant for production use.  They are reference points against which
the local-search and hybrid methods can be measured.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from OpenRC.compression.base import (
    COMPRESSORS,
    Compressor,
    moved_positions,
    trace_row,
)
from OpenRC.geometry.live_hull import LiveHull


@COMPRESSORS.register("random", doc="Remove vertices uniformly at random (lower baseline).")
class RandomCompressor(Compressor):
    """Uniformly random vertex removal."""

    name = "random"

    def __init__(self, *, seed: int = 0) -> None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)

    def _run_live(self, live: LiveHull, *, target_vertices: int,
                  max_volume_loss_pct: Optional[float]):
        rng = random.Random(self.seed)
        original_volume = live.volume
        trace = []
        while self._more(live, target_vertices):
            candidates = list(range(live.n_vertices))
            kept = None
            while candidates:
                index = rng.choice(candidates)
                kept = self._remove(live, index, original_volume, max_volume_loss_pct)
                if kept is not None:
                    break
                candidates.remove(index)
            if kept is None:
                break
            trace.append(trace_row(live, original_volume))
        return trace, {"compressor": self.name, "seed": self.seed}


@COMPRESSORS.register(
    "roulette",
    doc="Sample among the k cheapest vertices, weighted - a randomised greedy.",
)
class RouletteCompressor(Compressor):
    """Randomised greedy: pick among the ``k`` cheapest vertices by roulette.

    Softening the greedy choice occasionally escapes the myopia that costs pure
    local search volume in the tail.  It is cheaper than backward recycling and
    weaker; having both makes it possible to say how much of the hybrid method's
    advantage comes from randomisation and how much from recycling.
    """

    name = "roulette"

    def __init__(self, *, k: int = 3, seed: int = 0) -> None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.k = int(k)
        self.seed = int(seed)

    def _run_live(self, live: LiveHull, *, target_vertices: int,
                  max_volume_loss_pct: Optional[float]):
        rng = random.Random(self.seed)
        original_volume = live.volume
        trace: list = []
        if not self._more(live, target_vertices):
            return trace, {"compressor": self.name, "k": self.k, "seed": self.seed}

        losses = live.marginal_losses()
        while self._more(live, target_vertices):
            k = min(self.k, live.n_vertices)
            cheapest = np.argsort(losses)[:k]
            # Weight by the volume that would survive, so cheaper removals are
            # preferred without being certain.  Zero-weight vertices (e.g. a
            # pyramid apex whose deletion leaves the base flat) are excluded.
            pool = [int(i) for i in cheapest]
            kept = None
            neighbours = np.empty(0, dtype=np.intp)
            while pool:
                sub_weights = np.clip(live.volume - losses[pool], 0.0, None)
                positive = [idx for idx, w in zip(pool, sub_weights) if w > 0.0]
                if not positive:
                    index = pool[0]
                else:
                    pos_weights = np.clip(live.volume - losses[positive], 0.0, None)
                    total = float(pos_weights.sum())
                    draw = rng.random() * total
                    cumulative = 0.0
                    index = positive[-1]
                    for idx, w in zip(positive, pos_weights):
                        cumulative += float(w)
                        if draw <= cumulative:
                            index = idx
                            break
                neighbours = live.neighbours(index)
                kept = self._remove(live, index, original_volume, max_volume_loss_pct)
                if kept is not None:
                    break
                pool.remove(index)
            if kept is None:
                break
            losses = losses[kept]
            refresh = moved_positions(neighbours, kept)
            if len(refresh):
                losses[refresh] = live.marginal_losses(refresh)
            trace.append(trace_row(live, original_volume))

        return trace, {"compressor": self.name, "k": self.k, "seed": self.seed}
