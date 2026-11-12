"""Greedy local-search compression - remove the cheapest vertex, repeat.

At every step the vertex whose removal costs the least hypervolume is dropped.
It is the natural greedy heuristic and in practice removes most of a grown
region's vertices for a negligible loss: the low-quality vertices a
facet-breaking search produces are almost free to discard.

Cost
----
Everything runs on one :class:`~OpenRC.geometry.live_hull.LiveHull`.  A
vertex's marginal loss is computed from its neighbourhood (the lemma in that
module), so in 3-D it costs two hulls of about six points; after a removal only
the neighbours of the removed vertex are refreshed (``incremental=True``), and
the hull itself is rebuilt once per removal, keeping per-step cost low enough
for in-loop memory-aware compression.

High dimension
--------------
From about 8-D on, almost every pair of vertices shares a facet: each removal
then refreshes every loss, and each loss is a full hull.
:class:`StarScreenedCompressor` (``"star-screened"``) keeps the cost at a few
hulls per removal by ranking vertices with their *star volume* - an upper bound
on the marginal loss read off the live hull in one pass - and computing exact
losses only for the ``screen`` most promising ones.

Limitations
-----------
Greedy removal is myopic when multiple adjacent vertices are pruned in
sequence; see :mod:`OpenRC.compression.hybrid_vns` for backward VNS recycling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from OpenRC.compression.base import (
    COMPRESSORS,
    Compressor,
    moved_positions,
    trace_row,
)
from OpenRC.geometry.live_hull import LiveHull


@COMPRESSORS.register(
    "local-search",
    doc="Greedy: drop the vertex with the smallest marginal hypervolume loss.",
)
class LocalSearchCompressor(Compressor):
    """Greedy vertex pruning, with an optional incremental update.

    Parameters
    ----------
    incremental:
        Refresh only the neighbours of the removed vertex instead of the whole
        marginal-loss vector, yielding identical losses at lower cost.
    """

    name = "local-search"

    def __init__(self, *, incremental: bool = True) -> None:
        if not isinstance(incremental, (bool, np.bool_)):
            raise ValueError("incremental must be a boolean")
        self.incremental = bool(incremental)

    def _run_live(self, live: LiveHull, *, target_vertices: int,
                  max_volume_loss_pct: Optional[float]):
        original_volume = live.volume
        trace: list = []
        if not self._more(live, target_vertices):
            return trace, {"compressor": self.name, "incremental": self.incremental}
        losses = live.marginal_losses()
        while self._more(live, target_vertices):
            index = int(np.argmin(losses))
            neighbours = live.neighbours(index) if self.incremental else None
            kept = self._remove(live, index, original_volume, max_volume_loss_pct)
            if kept is None:
                break   # this removal would breach the quality budget
            if self.incremental:
                losses = losses[kept]
                refresh = moved_positions(neighbours, kept)
                if len(refresh):
                    losses[refresh] = live.marginal_losses(refresh)
            else:
                losses = live.marginal_losses()
            trace.append(trace_row(live, original_volume))
        return trace, {"compressor": self.name, "incremental": self.incremental}


@COMPRESSORS.register(
    "star-screened",
    doc="Greedy, screened by star volume: a few hulls per removal (for high dimension).",
)
class StarScreenedCompressor(Compressor):
    """Greedy pruning that computes exact losses only for a short list.

    Each step reads every vertex's *star volume* off the live hull (the cones
    from the vertex centroid over the facets containing it - one pass for all
    vertices), computes the exact marginal loss of the ``screen`` vertices with
    the smallest stars, and removes the cheapest of those.  Cost per removal:
    ``screen`` hulls plus one rebuild, independent of the vertex count, where
    :class:`LocalSearchCompressor` needs a hull per neighbour - in high
    dimension, per vertex.

    The star bounds the loss from above whenever the centroid stays inside the
    hull of the other vertices, so a small star is a guaranteed-cheap vertex; a
    large star may still hide a cheap one, which is why ``screen`` is more than
    one.  ``screen >= n_vertices`` reproduces :class:`LocalSearchCompressor`
    without the incremental update.
    """

    name = "star-screened"

    def __init__(self, *, screen: int = 4) -> None:
        if (isinstance(screen, (bool, np.bool_))
                or not isinstance(screen, (int, np.integer)) or screen < 1):
            raise ValueError("screen must be a positive integer")
        self.screen = int(screen)

    def _run_live(self, live: LiveHull, *, target_vertices: int,
                  max_volume_loss_pct: Optional[float]):
        original_volume = live.volume
        trace = []
        n_exact = 0
        while self._more(live, target_vertices):
            stars = live.star_volumes()
            shortlist = np.argsort(stars, kind="stable")[:min(self.screen, live.n_vertices)]
            losses = live.marginal_losses(shortlist)
            n_exact += len(shortlist)
            index = int(shortlist[int(np.argmin(losses))])
            if self._remove(live, index, original_volume, max_volume_loss_pct) is None:
                break
            trace.append(trace_row(live, original_volume))
        return trace, {"compressor": self.name, "screen": self.screen,
                       "exact_losses": n_exact}
