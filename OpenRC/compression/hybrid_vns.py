"""Hybrid compression: greedy local search plus backward recycling with VNS.

Pure greedy pruning is myopic: a vertex with small marginal loss early in
compression can become a good geometric substitute later once adjacent vertices
have also been removed.

Backward recycling addresses this by storing pruned vertices in per-vertex
**pools** on their surviving hull neighbours, merging pools into remaining
neighbours on subsequent deletions so discarded candidates stay associated with
their boundary region.

Before deleting the lowest-loss candidate ``v_o``, a two-phase variable
neighbourhood search (VNS) checks whether swapping ``v_o`` for a pooled point
strictly increases hull volume: first evaluating ``v_o``'s own pool to find the
best local replacement ``v_n``, and then evaluating the ``neighbourhood_size``
pooled points closest to ``v_n`` across all pools.

Every swap is scored without constructing a hull per candidate: with the hull
``H`` of the vertices other than ``v_o`` built once, swapping in ``c`` yields
volume ``vol(H) + gain_H(c)`` (:meth:`~OpenRC.geometry.live_hull.LiveHull.gain`).

Because backward swap evaluation is costlier than greedy pruning, it activates
only once cumulative hypervolume loss reaches ``threshold_pct``.
"""

from __future__ import annotations

from numbers import Real
from typing import List, Optional

import numpy as np

from OpenRC.compression.base import (
    COMPRESSORS,
    Compressor,
    moved_positions,
    trace_row,
)
from OpenRC.geometry.live_hull import LiveHull
from OpenRC.geometry.ops import GeometryNumericsError


@COMPRESSORS.register(
    "hybrid-vns",
    doc="Greedy pruning that switches to backward recycling (VNS) in the tail.",
)
class HybridVnsCompressor(Compressor):
    """Local search, then local search + backward recycling.

    Parameters
    ----------
    threshold_pct:
        Cumulative hypervolume loss (%) at which backward search switches on.
        ``0.0`` uses it from the start; a large value never does, which
        reproduces pure :class:`LocalSearchCompressor`.
    neighbourhood_size:
        Cap on the pool scanned per phase - the ``|N(v)|`` of the algorithm.
    max_recycles:
        Safety cap on the re-selection loop after a successful swap.
    seed:
        Kept for reproducible records; the method itself is deterministic.
    """

    name = "hybrid-vns"

    def __init__(self, *, threshold_pct: float = 1.0, neighbourhood_size: int = 50,
                 max_recycles: int = 50, seed: int = 0) -> None:
        if (isinstance(threshold_pct, (bool, np.bool_))
                or not isinstance(threshold_pct, (int, float, Real))
                or not np.isfinite(float(threshold_pct)) or float(threshold_pct) < 0.0):
            raise ValueError("threshold_pct must be a finite non-negative number")
        if (isinstance(neighbourhood_size, (bool, np.bool_))
                or not isinstance(neighbourhood_size, (int, np.integer))
                or neighbourhood_size < 1):
            raise ValueError("neighbourhood_size must be a positive integer")
        if (isinstance(max_recycles, (bool, np.bool_))
                or not isinstance(max_recycles, (int, np.integer))
                or max_recycles < 0):
            raise ValueError("max_recycles must be a non-negative integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.threshold_pct = float(threshold_pct)
        self.neighbourhood_size = int(neighbourhood_size)
        self.max_recycles = int(max_recycles)
        self.seed = int(seed)

    def _run_live(self, live: LiveHull, *, target_vertices: int,
                  max_volume_loss_pct: Optional[float]):
        original_volume = live.volume
        trace: List[dict] = []
        n_recycled = 0
        if not self._more(live, target_vertices):
            return trace, {
                "compressor": self.name,
                "threshold_pct": self.threshold_pct,
                "neighbourhood_size": self.neighbourhood_size,
                "n_recycled": n_recycled,
            }
        # Each vertex starts with only itself as a recycling candidate.
        pools: List[np.ndarray] = [v.reshape(1, -1).copy() for v in live.vertices]
        losses = live.marginal_losses()
        consecutive_recycles = 0

        while self._more(live, target_vertices):
            loss_pct = (100.0 * (1.0 - live.volume / original_volume)
                        if original_volume > 0 else 0.0)
            use_backward = loss_pct >= self.threshold_pct
            index = int(np.argmin(losses))

            # Execute either a recycle swap or a vertex removal per iteration
            # so a newly recycled vertex is not immediately pruned.
            if use_backward and consecutive_recycles < self.max_recycles:
                point = self._vns(live, pools, index)
                if point is not None:
                    vertices = live.vertices
                    retired = vertices[index].copy()
                    vertices[index] = point
                    # The vertex swapped out becomes a candidate in its turn.
                    extra = [pools[index], retired[None, :]]
                    try:
                        survivors = live.reset(vertices)
                    except GeometryNumericsError:
                        point = None
                if point is not None:
                    surviving_set = set(int(i) for i in survivors)
                    for j in range(len(vertices)):
                        if j not in surviving_set and j != index:
                            extra.extend([pools[j], vertices[j][None, :]])
                    host = (index if index in surviving_set
                            else int(survivors[int(np.argmin(
                                np.linalg.norm(vertices[survivors] - point[None, :], axis=1)
                            ))]))
                    if host != index:
                        extra.insert(0, pools[host])
                    pools[host] = _nearest(
                        np.unique(np.vstack(extra), axis=0),
                        vertices[host], self.neighbourhood_size)
                    pools = [pools[i] for i in survivors]
                    losses = live.marginal_losses()
                    n_recycled += 1
                    consecutive_recycles += 1
                    trace.append(trace_row(live, original_volume, action="recycle"))
                    continue            # do not prune this round
            consecutive_recycles = 0

            # The pruned vertex's pool is handed to its neighbours.
            vertices = live.vertices
            neighbours = live.neighbours(index)
            kept = self._remove(live, index, original_volume, max_volume_loss_pct)
            if kept is None:
                break
            removed_pool = pools[index]
            for neighbour in neighbours:
                merged = np.unique(np.vstack([pools[neighbour], removed_pool]), axis=0)
                pools[int(neighbour)] = _nearest(merged, vertices[neighbour],
                                                 self.neighbourhood_size)
            pools = [pools[i] for i in kept]
            losses = losses[kept]
            refresh = moved_positions(neighbours, kept)
            if len(refresh):
                losses[refresh] = live.marginal_losses(refresh)
            trace.append(trace_row(live, original_volume, action="prune",
                                   backward=bool(use_backward)))

        return trace, {
            "compressor": self.name,
            "threshold_pct": self.threshold_pct,
            "neighbourhood_size": self.neighbourhood_size,
            "n_recycled": n_recycled,
        }

    # ------------------------------------------------------------------ VNS
    def _vns(self, live: LiveHull, pools: List[np.ndarray], index: int) -> Optional[np.ndarray]:
        """Best replacement for vertex ``index``, or ``None`` if none gains volume.

        Phase 1 scans the vertex's own pool; phase 2 scans the pruned points
        nearest to the phase-1 winner, from every pool.  Candidates that are
        current vertices are skipped, so a swap never duplicates a vertex.
        """
        vertices = live.vertices
        try:
            rest = LiveHull(np.delete(vertices, index, axis=0))
        except (ValueError, GeometryNumericsError):
            return None           # the others alone are flat: nothing to score against
        try:
            base = rest.volume
            span = float(np.ptp(vertices, axis=0).max())
            dup_tol = 1e-12 * (span if span > 0.0 else 1.0)

            def best_of(candidates: np.ndarray, floor: float):
                best_volume, best_point = floor, None
                for candidate in candidates:
                    if np.any(np.all(np.abs(vertices - candidate) <= dup_tol, axis=1)):
                        continue
                    volume = base + rest.gain(candidate)
                    if volume > best_volume * (1.0 + 1e-12):
                        best_volume, best_point = volume, candidate
                return best_volume, best_point

            volume_1, point_1 = best_of(pools[index], live.volume)
            if point_1 is None:
                return None
            neighbourhood = _nearest(np.unique(np.vstack(pools), axis=0), point_1,
                                     self.neighbourhood_size)
            _, point_2 = best_of(neighbourhood, volume_1)
            return (point_2 if point_2 is not None else point_1).copy()
        finally:
            rest.release()


def _nearest(points: np.ndarray, centre: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` pooled points closest to ``centre``.

    Retains the ``k`` nearest candidates in Euclidean distance so bounded
    recycling pools preserve local geometric replacements.
    """
    if len(points) <= k:
        return points
    distances = np.linalg.norm(points - centre[None, :], axis=1)
    return points[np.argsort(distances)[:k]]
