r"""Base classes and live-hull utilities for polytope vertex compression.

A polytope produced by facet-breaking growth is mostly made of *low-quality*
vertices - points that were explored in the course of finding the shape but
contribute almost nothing to the volume.  Removing them shrinks the stored
region by an order of magnitude at a cost of a fraction of a percent of volume,
which reduces storage of the vertex table.  When compression also reduces the
half-space row count, optimisation models that embed those inequalities get
smaller.  Vertex count alone does not determine facet count or LP solve time.

The problem
-----------
Given a vertex set ``V`` whose convex hull is the region, find ``S`` with

.. math::  S = \\arg\\max_{|S| = n,\; S \\subset V} H(\\mathrm{conv}(S))

or, in dual form, the smallest ``S`` whose hull keeps at least a given share of
the original hypervolume.  Both are NP-hard, so every method here is a
heuristic, each with its own trade-off between quality and cost.

Requiring ``S ⊂ V`` is not a restriction: an optimal *inner* compression can
always be taken to be a subset of the original vertices, so compression and
vertex pruning are the same problem.

Implementing a new compressor
-----------------------------
Subclass :class:`Compressor` and register it.  There are two hooks; implement
either one::

    @COMPRESSORS.register("my-method", doc="...")
    class MyCompressor(Compressor):
        # simplest: work on a Polytope, return a CompressionResult
        def _run(self, polytope, *, target_vertices, max_volume_loss_pct): ...

        # or: work in place on the live hull (no hull rebuilds of your own)
        def _run_live(self, live, *, target_vertices, max_volume_loss_pct):
            ...                      # live.marginal_losses(), live.remove([i]) ...
            return trace, info

A ``_run`` compressor works everywhere, including inside a memory-aware build:
the base class hands it the current vertices and reloads the hull from its
answer.  ``_run_live`` avoids that round trip, which is what the built-in
methods do.  Inside ``_run_live``, loop ``while self._more(live, target)``
and remove with ``self._remove(...)``: those two honour every stopping rule of
:meth:`Compressor.compress_live`, including the memory goal and the per-vertex
loss cap a compression schedule may set.

The live hull
-------------
Built-in compressors operate directly on :class:`~OpenRC.geometry.live_hull.LiveHull`:
marginal vertex losses are evaluated from local vertex neighbourhoods, requiring
one hull update per accepted removal rather than one full hull per candidate.
:meth:`Compressor.compress_live` mutates the live hull in place during region
growth.
"""

from __future__ import annotations

import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from OpenRC.geometry.live_hull import LiveHull
from OpenRC.geometry.ops import GeometryNumericsError
from OpenRC.geometry.polytope import Polytope
from OpenRC.registry import Registry
from OpenRC.utils import LOGGER

#: Registry of compressor classes.
COMPRESSORS: Registry = Registry("compressor")
_GOAL_STATE = threading.local()


@dataclass
class CompressionResult:
    """A compressed region plus the trajectory that produced it."""

    polytope: Polytope
    original_volume: float
    original_n_vertices: int
    trace: List[dict] = field(default_factory=list)
    seconds: float = 0.0
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.polytope, Polytope):
            raise TypeError(f"polytope must be a Polytope, got {type(self.polytope).__name__}")
        if (isinstance(self.original_volume, (bool, np.bool_))
                or not isinstance(self.original_volume, (int, float, np.number))
                or not np.isfinite(self.original_volume) or float(self.original_volume) < 0.0):
            raise ValueError("original_volume must be a finite non-negative number")
        if (isinstance(self.original_n_vertices, (bool, np.bool_))
                or not isinstance(self.original_n_vertices, (int, np.integer))
                or int(self.original_n_vertices) < 0):
            raise ValueError("original_n_vertices must be a non-negative integer")
        self.original_volume = float(self.original_volume)
        self.original_n_vertices = int(self.original_n_vertices)

    @property
    def volume_loss_pct(self) -> float:
        """Hypervolume given up, as a percentage of the original."""
        if self.original_volume <= 0:
            return 0.0
        return 100.0 * (1.0 - self.polytope.volume / self.original_volume)

    @property
    def size_reduction_pct(self) -> float:
        """Vertices removed, as a percentage of the original count."""
        if self.original_n_vertices <= 0:
            return 0.0
        return 100.0 * (1.0 - self.polytope.n_vertices / self.original_n_vertices)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.trace)

    def __repr__(self) -> str:
        comp = self.info.get("compressor", "compression")
        return (
            f"CompressionResult(compressor={comp!r}, "
            f"vertices={self.original_n_vertices}->{self.polytope.n_vertices}, "
            f"volume_loss_pct={self.volume_loss_pct:.3f}, seconds={self.seconds:.2f})"
        )

    def summary(self) -> str:
        return (
            f"{self.info.get('compressor', 'compression')}: "
            f"{self.original_n_vertices} -> {self.polytope.n_vertices} vertices "
            f"({self.size_reduction_pct:.1f}% smaller), "
            f"{self.volume_loss_pct:.3f}% hypervolume lost, {self.seconds:.2f} s"
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the compressed polytope (CSV + JSON) and pruning trace side by side."""
        stem = Path(path)
        if stem.suffix.lower() in (".csv", ".json"):
            stem = stem.with_suffix("")
        saved = self.polytope.save(stem)
        if self.trace:
            self.to_frame().to_csv(stem.with_name(stem.name + "_compression_trace.csv"),
                                   index=False)
        return saved


class Compressor(ABC):
    """Reduces a polytope's vertex count under a quality constraint."""

    name: str = "compressor"

    def compress(
        self,
        polytope: Polytope,
        *,
        target_vertices: Optional[int] = None,
        max_volume_loss_pct: Optional[float] = None,
    ) -> CompressionResult:
        """Compress until a vertex target or a volume-loss budget is reached.

        Parameters
        ----------
        target_vertices:
            Stop once this many vertices remain (problem form 1).
        max_volume_loss_pct:
            Stop before cumulative loss would exceed this percentage
            (problem form 2).  May be combined with ``target_vertices``, in
            which case whichever binds first wins.

        Raises
        ------
        ValueError
            If neither stopping rule is given, or the target is unreachable.
        """
        if not isinstance(polytope, Polytope):
            raise TypeError("polytope must be a Polytope")
        self._check_rules(polytope.dim, target_vertices, max_volume_loss_pct)
        live = LiveHull(polytope.vertices)
        try:
            result = self.compress_live(live, target_vertices=target_vertices,
                                        max_volume_loss_pct=max_volume_loss_pct,
                                        axes=polytope.axes)
        finally:
            live.release()
        result.polytope.metadata = dict(polytope.metadata) | result.polytope.metadata
        return result

    def compress_live(
        self,
        live: LiveHull,
        *,
        target_vertices: Optional[int] = None,
        max_volume_loss_pct: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        max_vertex_loss: Optional[float] = None,
        axes=None,
    ) -> CompressionResult:
        """Compress a live hull **in place**; it stays alive for further growth.

        Besides the two rules of :meth:`compress`, a live compression can stop

        * once modelled hull memory falls to ``max_memory_mb`` (evaluated
          from the updated hull after each removal); and
        * before any single removal whose volume loss exceeds
          ``max_vertex_loss`` (in absolute volume units).

        Compression stops as soon as any active rule binds.  The returned
        :class:`Polytope` stores the surviving vertices with ``volume``
        pre-populated from ``live.volume``.
        """
        if not isinstance(live, LiveHull):
            raise TypeError("live must be a LiveHull")
        self._check_rules(live.dim, target_vertices, max_volume_loss_pct,
                          max_memory_mb, max_vertex_loss)
        live.compact()
        original_volume, original_n = live.volume, live.n_vertices
        start = time.perf_counter()
        self._push_goal((max_memory_mb, max_vertex_loss))
        try:
            trace, info = self._run_live(
                live,
                target_vertices=target_vertices if target_vertices is not None else live.dim + 1,
                max_volume_loss_pct=max_volume_loss_pct,
            )
        finally:
            self._pop_goal()
        result = CompressionResult(
            polytope=polytope_from_live(live, axes=axes),
            original_volume=original_volume,
            original_n_vertices=original_n,
            trace=list(trace),
            seconds=time.perf_counter() - start,
            info=dict(info),
        )
        result.info.setdefault("compressor", self.name)
        result.polytope.metadata = {"compression": result.info | {
            "volume_loss_pct": result.volume_loss_pct,
            "size_reduction_pct": result.size_reduction_pct,
        }}
        return result

    @staticmethod
    def _check_rules(dim: int, target_vertices: Optional[int],
                     max_volume_loss_pct: Optional[float],
                     max_memory_mb: Optional[float] = None,
                     max_vertex_loss: Optional[float] = None) -> None:
        if (target_vertices is None and max_volume_loss_pct is None
                and max_memory_mb is None and max_vertex_loss is None):
            raise ValueError(
                "give target_vertices, max_volume_loss_pct, or both - otherwise there "
                "is no stopping rule"
            )
        for label, value in (("max_memory_mb", max_memory_mb),
                             ("max_vertex_loss", max_vertex_loss)):
            if value is not None and (isinstance(value, (bool, np.bool_))
                                      or not isinstance(value, (int, float, np.number))
                                      or not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{label} must be a finite non-negative number")
        # A hull needs at least dim+1 vertices to keep a non-zero volume.
        floor = dim + 1
        if target_vertices is not None:
            if (isinstance(target_vertices, (bool, np.bool_))
                    or not isinstance(target_vertices, (int, np.integer))):
                raise ValueError(f"target_vertices must be an integer, got {target_vertices!r}")
            if target_vertices < floor:
                raise ValueError(
                    f"target_vertices={target_vertices} is below the {floor} vertices a "
                    f"{dim}-D body needs"
                )
        if max_volume_loss_pct is not None and (
                isinstance(max_volume_loss_pct, (bool, np.bool_))
                or not isinstance(max_volume_loss_pct, (int, float, np.number))
                or not np.isfinite(max_volume_loss_pct) or max_volume_loss_pct < 0.0):
            raise ValueError("max_volume_loss_pct must be a finite non-negative number")

    # --------------------------------------------- stopping rules for hooks
    @staticmethod
    def _push_goal(goal: tuple) -> None:
        stack = getattr(_GOAL_STATE, "stack", None)
        if stack is None:
            stack = _GOAL_STATE.stack = []
        stack.append(goal)

    @staticmethod
    def _pop_goal() -> None:
        _GOAL_STATE.stack.pop()

    @property
    def _goal(self) -> tuple:
        stack = getattr(_GOAL_STATE, "stack", None)
        return stack[-1] if stack else (None, None)

    def _more(self, live: LiveHull, target_vertices: int) -> bool:
        """Whether another removal is wanted: above the vertex target, and -
        when a memory goal is set - still above it."""
        if live.n_vertices <= target_vertices:
            return False
        max_memory_mb = self._goal[0]
        return max_memory_mb is None or live.memory_mb() > max_memory_mb

    def _remove(self, live: LiveHull, position: int, original_volume: float,
                max_volume_loss_pct: Optional[float]) -> Optional[np.ndarray]:
        """Remove one vertex unless a quality rule forbids it.

        Checks the cumulative budget and the per-vertex loss on the rebuilt
        hull, and puts the vertex back (one incremental insertion) if either
        is breached.  Returns the surviving old positions, or ``None`` if
        nothing was removed - the compressor should then stop.  That includes
        the rare point set without that vertex that Qhull cannot hull in any
        of its modes, or a pyramid apex whose deletion leaves the rest flat:
        the hull is then left exactly as it was.
        """
        before = live.volume
        point = live.vertices[position].copy()
        try:
            kept = live.remove([position])
        except GeometryNumericsError as exc:
            LOGGER.warning("compression stopped at %d vertices: %s", live.n_vertices,
                           str(exc)[:200])
            return None
        except ValueError as exc:
            if not str(exc).startswith("the points are degenerate"):
                raise
            return None
        max_vertex_loss = self._goal[1]
        if (loss_breaches(original_volume, live.volume, max_volume_loss_pct)
                or (max_vertex_loss is not None and before - live.volume > max_vertex_loss)):
            live.add(point)
            return None
        return kept

    # ------------------------------------------------------------------ hooks
    def _run_live(self, live: LiveHull, *, target_vertices: int,
                  max_volume_loss_pct: Optional[float]):
        """Compress ``live`` in place; return ``(trace, info)``.

        The default delegates to :meth:`_run` on the current vertices and
        reloads the hull from its answer, so a compressor written against
        polytopes works inside a live computation unchanged.  ``_run`` knows
        only the vertex target and the loss budget: when a memory goal or a
        per-removal price is set, it is called for one vertex at a time and
        every rule is checked after each step, which is undone if it breaches
        the loss budget or the price.
        """
        if type(self)._run is Compressor._run:
            raise NotImplementedError(
                f"{type(self).__name__} implements neither _run_live nor _run")
        max_memory_mb, max_vertex_loss = self._goal
        if max_memory_mb is None and max_vertex_loss is None:
            result = self._run(Polytope(live.vertices), target_vertices=target_vertices,
                               max_volume_loss_pct=max_volume_loss_pct)
            live.reset(result.polytope.vertices)
            return result.trace, result.info
        original_volume = live.volume
        trace: List[dict] = []
        info: dict = {}
        while self._more(live, target_vertices):
            before_vertices, before_volume = live.vertices.copy(), live.volume
            result = self._run(Polytope(before_vertices), target_vertices=live.n_vertices - 1,
                               max_volume_loss_pct=max_volume_loss_pct)
            if len(result.polytope.vertices) >= len(before_vertices):
                break                                   # the compressor declined
            live.reset(result.polytope.vertices)
            if (loss_breaches(original_volume, live.volume, max_volume_loss_pct)
                    or (max_vertex_loss is not None
                        and before_volume - live.volume > max_vertex_loss)):
                live.reset(before_vertices)
                break
            trace.append(trace_row(live, original_volume))
            info = dict(result.info)
        return trace, info

    def _run(self, polytope: Polytope, *, target_vertices: int,
             max_volume_loss_pct: Optional[float]) -> CompressionResult:
        """Polytope hook; ``polytope`` has interior points removed."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"compressor": self.name}


def auto_compressor_name(dim: int) -> str:
    """The compressor that ``"auto"`` stands for in ``dim`` dimensions.

    ``"local-search"`` up to 6-D.  Above, every vertex neighbours almost every
    other, exact greedy costs a full hull per vertex per removal, and
    ``"star-screened"`` is 3.2-3.9x faster in 7- and 8-D at volume ratios
    0.93-1.12.
    """
    if (isinstance(dim, (bool, np.bool_))
            or not isinstance(dim, (int, np.integer))
            or int(dim) < 1):
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    return "local-search" if int(dim) <= 6 else "star-screened"


def get_compressor(name: str, **kwargs) -> Compressor:
    return COMPRESSORS.get(name)(**kwargs)


def list_compressors() -> List[tuple]:
    return COMPRESSORS.describe()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def polytope_from_live(live: LiveHull, *, axes=None) -> Polytope:
    """The live hull's vertex set as a :class:`Polytope`, volume pre-filled.

    Only the vertices are kept; the hull itself is working state and is not
    copied.  Pre-populates ``volume`` from ``live.volume``; facet equations and
    other hull-derived attributes are computed lazily on demand.
    """
    if not isinstance(live, LiveHull):
        raise TypeError("live must be a LiveHull")
    return Polytope.with_known_volume(live.vertices, live.volume, axes=axes)


def loss_breaches(original_volume: float, volume_after: float,
                  max_volume_loss_pct: Optional[float]) -> bool:
    """Whether ending at ``volume_after`` would exceed the loss budget."""
    for label, val in (("original_volume", original_volume), ("volume_after", volume_after)):
        if (isinstance(val, (bool, np.bool_))
                or not isinstance(val, (int, float, np.number))
                or not np.isfinite(val)):
            raise ValueError(f"{label} must be a finite number")
    if max_volume_loss_pct is not None and (
        isinstance(max_volume_loss_pct, (bool, np.bool_))
        or not isinstance(max_volume_loss_pct, (int, float, np.number))
        or not np.isfinite(max_volume_loss_pct)
    ):
        raise ValueError("max_volume_loss_pct must be a finite number or None")
    if max_volume_loss_pct is None or original_volume <= 0:
        return False
    return 100.0 * (1.0 - volume_after / original_volume) > max_volume_loss_pct


def trace_row(live: LiveHull, original_volume: float, **extra) -> dict:
    """One compression-trace row for the current state of ``live``."""
    if not isinstance(live, LiveHull):
        raise TypeError(f"live must be a LiveHull, got {type(live).__name__}")
    if (isinstance(original_volume, (bool, np.bool_))
            or not isinstance(original_volume, (int, float, np.number))
            or not np.isfinite(original_volume)):
        raise ValueError("original_volume must be a finite number")
    return {
        "n_vertices": live.n_vertices,
        "volume": live.volume,
        "volume_loss_pct": 100.0 * (1.0 - live.volume / original_volume)
        if original_volume > 0 else 0.0,
        **extra,
    }


def moved_positions(old_positions, kept: np.ndarray) -> np.ndarray:
    """New positions of those ``old_positions`` that survived a removal.

    ``kept`` is what :meth:`LiveHull.remove` returns: the old positions still
    present, in (ascending) new order.
    """
    raw_old = np.asarray(old_positions)
    raw_kept = np.asarray(kept)
    if raw_old.dtype == bool or raw_kept.dtype == bool:
        raise ValueError("positions must be integer arrays, not boolean")
    if raw_old.ndim != 1 or raw_kept.ndim != 1:
        raise ValueError("positions must be 1-D integer arrays")
    old = np.asarray(raw_old, dtype=np.intp)
    kept_arr = np.asarray(raw_kept, dtype=np.intp)
    if len(kept_arr) == 0 or len(old) == 0:
        return np.empty(0, dtype=np.intp)
    where = np.searchsorted(kept_arr, old)
    alive = where < len(kept_arr)
    alive[alive] = kept_arr[where[alive]] == old[alive]
    return where[alive]


def marginal_losses(vertices: np.ndarray, base_volume: Optional[float] = None) -> np.ndarray:
    """Hypervolume each vertex would cost if removed (standalone helper).

    Builds one live hull and asks it for every vertex's exact loss, which uses
    the vertex's neighbourhood instead of a full rebuild where that is smaller.
    ``base_volume`` is accepted for compatibility and ignored: the loss is
    always measured against the hull of ``vertices`` itself.
    """
    raw = np.asarray(vertices)
    if raw.dtype == bool or raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError("vertices must be a 2-D numeric array with d >= 1")
    live = LiveHull(vertices)
    try:
        if live.n_vertices != len(vertices):
            # Interior points cost nothing; each hull vertex's loss is measured
            # against the full input array (which may contain duplicate or
            # near-boundary interior points that cushion its deletion).
            from OpenRC.geometry.live_hull import PARALLEL_FROM_DIM, _volume, parallel_map

            pts = np.asarray(raw, dtype=float)
            base_vol = live.volume
            v_ids = [int(i) for i in live._vertex_ids]

            def _loss_at(idx: int) -> float:
                rest = _volume(np.delete(pts, idx, axis=0))
                if rest is None:
                    return base_vol
                return max(base_vol - rest, 0.0)

            out = np.zeros(len(pts), dtype=float)
            losses = (parallel_map(_loss_at, v_ids)
                      if pts.shape[1] >= PARALLEL_FROM_DIM and len(v_ids) > 1
                      else [_loss_at(i) for i in v_ids])
            out[v_ids] = losses
            return out
        return live.marginal_losses()
    finally:
        live.release()


def hull_adjacency(vertices: np.ndarray) -> dict:
    """``{vertex index: [neighbour indices]}`` from the convex hull's facets.

    Two vertices are neighbours when they share a facet.  Removing a vertex only
    changes the marginal loss of its neighbours, which is the basis of the
    incremental update.
    """
    from scipy.spatial import ConvexHull, QhullError

    from OpenRC.geometry.live_hull import _is_flat
    from OpenRC.geometry.polytope import _hull_geometry_consistent

    raw = np.asarray(vertices)
    if raw.dtype == bool:
        raise ValueError("vertices must be a 2-D numeric array, not boolean")
    pts = np.asarray(raw, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[1] == 0:
        raise ValueError("vertices must be a 2-D (n, d) array with d >= 1")
    if not np.all(np.isfinite(pts)):
        raise ValueError("vertices must contain only finite numbers")
    if pts.shape[0] <= pts.shape[1]:
        return {}
    if pts.shape[1] == 1:
        col = pts[:, 0]
        i_lo = int(np.argmin(col))
        i_hi = int(np.argmax(col))
        if not col[i_hi] > col[i_lo]:
            return {}
        return {i_lo: [i_hi], i_hi: [i_lo]}
    hull = None
    for qhull_options in (None, "Qx", "Qt", "Qx Q12", "Q14"):
        try:
            candidate = ConvexHull(pts, qhull_options=qhull_options)
        except QhullError as exc:
            if _is_flat(exc):
                return {}
            continue
        except ValueError:
            continue
        if qhull_options is not None and not _hull_geometry_consistent(candidate, pts):
            continue
        hull = candidate
        break
    if hull is None:
        return {}
    simplices = np.asarray(hull.simplices)
    dim = simplices.shape[1]
    # All ordered pairs within each facet, deduplicated in one pass.
    a = np.repeat(simplices, dim, axis=1).ravel()
    b = np.tile(simplices, (1, dim)).ravel()
    pairs = np.unique(np.stack([a[a != b], b[a != b]], axis=1), axis=0)
    out: dict = {}
    for i, j in pairs:
        out.setdefault(int(i), []).append(int(j))
    return out
