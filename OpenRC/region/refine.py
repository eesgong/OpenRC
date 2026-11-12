r"""Refinement at a fixed vertex count: move vertices along the boundary.

Whereas expansion adds vertices and compression removes them, a high-dimensional
memory cap limits the hull to a small fixed vertex count (e.g. ~36 vertices at
80 MiB in 16-D; :mod:`OpenRC.region.memory_aware`) where further pruning loses
more volume than new rays recover.  Refinement relocates existing vertices along
the boundary to maximise the volume of the ``N``-vertex inscribed polytope.

A refiner improves a live hull at a fixed vertex count.  The registered one,
``"relocation"``, is first-order local search on that problem.

The volume gradient
-------------------
With the hull triangulated into cones ``conv(c, F)`` from an interior point
``c`` (the volume does not depend on ``c``), each cone has volume
``V_F = |det D_F| / d!`` with the rows of ``D_F`` the vertices of ``F`` minus
``c``, and by Jacobi's formula

.. math::  \frac{\partial V_F}{\partial v_j} = V_F\,(D_F^{-\top})_j ,

so ``grad_i = sum_{F containing i} V_F (D_F^{-T})_{row of i}`` for every vertex
at once, from the live hull.  A vertex must stay on the region's boundary, so
only the component of the gradient tangent to the sphere of directions at
``v_i`` can be followed; a polytope all of whose tangential gradients vanish
satisfies the first-order form of *Property Z* (no vertex can be moved within
a neighbourhood to increase the volume), the local-optimality criterion for
maximum-volume inscribed polytopes.

A move
------
Pick the vertex whose tangential gradient promises the most (gradient norm x
distance x current step), rotate its direction by the step angle towards the
gradient, locate the boundary point along that ray, and accept the replacement
only when hull volume strictly increases.  The volume therefore never falls
and the vertex count never rises.  Each vertex keeps its own step angle,
adapted by Rechenberg's 1/5 success rule (grow on success, shrink by the
fourth root on failure), and the search stops once every step is below
``min_step`` - the resolution at which Property Z is reached.

The moves are those of the ``"relocation"`` ray strategy
(:mod:`OpenRC.region.ray_growth`); this refiner runs that strategy at the
hull's current vertex count.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from OpenRC.geometry.live_hull import _CHUNK, LiveHull
from OpenRC.registry import Registry

#: Registry of refiners (fixed-vertex-count improvement of a live hull).
REFINERS: Registry = Registry("refiner")


def volume_gradient(live: LiveHull) -> np.ndarray:
    """``d vol / d v_i`` for every hull vertex, ``(n_vertices, dim)``.

    Exact derivative of the hull volume with respect to each vertex position,
    holding the combinatorial structure fixed (Jacobi's formula over the cone
    triangulation, see the module docstring).
    """
    if not isinstance(live, LiveHull):
        raise TypeError(f"live must be a LiveHull, got {type(live).__name__}")
    points = live.points
    simplices = live.simplices
    rows = live.facet_positions()
    vertices = live.vertices
    c = vertices.mean(axis=0)
    dim = live.dim
    grad = np.zeros_like(vertices)
    scale = 1.0 / math.factorial(dim)
    for start in range(0, len(simplices), _CHUNK):
        D = points[simplices[start:start + _CHUNK]] - c[None, None, :]
        det = np.linalg.det(D)
        ok = np.abs(det) > 0.0
        if not ok.any():
            continue
        D, det = D[ok], det[ok]
        inv_t = np.transpose(np.linalg.inv(D), (0, 2, 1))       # (m, d, d)
        contrib = (np.abs(det) * scale)[:, None, None] * inv_t   # row j -> vertex j
        np.add.at(grad, rows[start:start + _CHUNK][ok].ravel(),
                  contrib.reshape(-1, dim))
    return grad


class Refiner(ABC):
    """Improves a live hull without adding vertices.

    Implement :meth:`refine`: work on ``live`` in place, never let its volume
    fall, and return an ``info`` dict (its ``"trace"`` entry is a list of
    rows).  Registered with ``@REFINERS.register(name)``, a refiner can be
    used by name in :class:`~OpenRC.region.memory_aware.MemoryAwareBuilder`;
    an instance works without registration.
    """

    name: str = "refiner"

    def reset(self) -> None:
        """Clear per-run counters."""

    @abstractmethod
    def refine(self, live: LiveHull, oracle, axes, *, max_moves: Optional[int] = None,
               max_hull_mb: Optional[float] = None, ray_cache: Optional[dict] = None,
               max_seconds: Optional[float] = None,
               max_oracle_calls: Optional[int] = None,
               ray_kwargs: Optional[dict] = None) -> dict:
        """Improve ``live`` in place; return ``{"trace": [...], ...}``."""

    def refine_polytope(self, polytope, oracle, axes=None, **kwargs):
        """Refine a stored region; returns ``(polytope, info)``.

        The polytope's vertices become a live hull, which is refined and then
        released; the result carries the same number of vertices or fewer.
        """
        from OpenRC.compression.base import polytope_from_live
        from OpenRC.geometry.polytope import Polytope

        if not isinstance(polytope, Polytope):
            raise TypeError(f"polytope must be a Polytope, got {type(polytope).__name__}")
        axes = axes if axes is not None else polytope.axes
        if axes is None or len(axes) != polytope.dim:
            raise ValueError(
                f"axes must be provided (on the polytope or as an argument) and have length {polytope.dim}"
            )
        polytope.validate_axes(axes)
        live = LiveHull(polytope.vertices)
        try:
            info = self.refine(live, oracle, axes, **kwargs)
            refined = polytope_from_live(live, axes=axes)
            summary = {key: value for key, value in info.items() if key != "trace"}
            describe = getattr(oracle, "describe", None)
            if callable(describe):
                summary["oracle"] = describe()
            refined.metadata = dict(polytope.metadata) | {"refinement": summary}
            return refined, info
        finally:
            live.release()

    def report(self) -> dict:
        return {"refiner": self.name}


@REFINERS.register(
    "relocation",
    doc="Move vertices along the boundary up the volume gradient; accept only gains.",
)
class GradientRelocation(Refiner):
    """First-order local search for the best ``N``-vertex inscribed polytope.

    Parameters
    ----------
    step:
        Initial step angle (radians) of every vertex.
    min_step:
        A vertex whose step has shrunk below this is considered locally optimal;
        the search stops when every vertex is.
    grow:
        Step multiplier on success; failure divides by ``grow ** 0.25`` (the
        1/5 success rule).
    moves_per_vertex:
        Default move budget per call, as a multiple of the vertex count.
    ray_kwargs:
        Passed to :func:`~OpenRC.geometry.bisection.ray_boundary`; ``hint`` and
        ``check_origin`` are set by the engine and refused here.
    first_order_screen:
        Opt-in speed heuristic that can miss improvements after the hull's
        facets change.  It is disabled by default so every replacement is
        decided from its exact volume.
    batch:
        Moves proposed per engine step, measured in parallel threads, the best
        applied (:func:`~OpenRC.region.ray_growth.run_rays`); ``1`` by default.
    """

    name = "relocation"

    def __init__(self, *, step: float = 0.2, min_step: float = 0.01, grow: float = 1.5,
                 moves_per_vertex: float = 2.0, ray_kwargs: Optional[dict] = None,
                 first_order_screen: bool = False, batch: int = 1) -> None:
        if (isinstance(batch, (bool, np.bool_))
                or not isinstance(batch, (int, float, np.number))
                or not np.isfinite(batch)
                or int(batch) != batch or batch < 1):
            raise ValueError("batch must be a positive integer")
        self.batch = int(batch)
        if (isinstance(min_step, (bool, np.bool_)) or isinstance(step, (bool, np.bool_))
                or not isinstance(min_step, (int, float, np.number))
                or not isinstance(step, (int, float, np.number))
                or not (math.isfinite(min_step) and math.isfinite(step)
                        and 0.0 < min_step < step < math.pi / 2)):
            raise ValueError("need 0 < min_step < step < pi/2")
        if (isinstance(grow, (bool, np.bool_)) or isinstance(moves_per_vertex, (bool, np.bool_))
                or not isinstance(grow, (int, float, np.number))
                or not isinstance(moves_per_vertex, (int, float, np.number))
                or not (math.isfinite(grow) and math.isfinite(moves_per_vertex)
                        and grow > 1.0 and moves_per_vertex > 0)):
            raise ValueError("grow must exceed 1 and moves_per_vertex be positive")
        self.step = float(step)
        self.min_step = float(min_step)
        self.grow = float(grow)
        self.moves_per_vertex = float(moves_per_vertex)
        from OpenRC.region.ray_growth import _check_ray_kwargs

        if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
            raise TypeError("ray_kwargs must be a dict")
        self.ray_kwargs = dict(ray_kwargs or {})
        _check_ray_kwargs(self.ray_kwargs)
        if not isinstance(first_order_screen, (bool, np.bool_)):
            raise ValueError("first_order_screen must be a boolean")
        self.first_order_screen = bool(first_order_screen)
        self.n_moves = 0
        self.n_accepted = 0

    def reset(self) -> None:
        self.n_moves = 0
        self.n_accepted = 0

    def refine(self, live: LiveHull, oracle, axes, *, max_moves: Optional[int] = None,
               max_hull_mb: Optional[float] = None, ray_cache: Optional[dict] = None,
               max_seconds: Optional[float] = None,
               max_oracle_calls: Optional[int] = None,
               ray_kwargs: Optional[dict] = None) -> dict:
        from OpenRC.region.ray_growth import RelocationRays, _check_ray_kwargs, run_rays

        if not isinstance(live, LiveHull):
            raise TypeError(f"live must be a LiveHull, got {type(live).__name__}")
        if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
            raise TypeError("ray_kwargs must be a dict")
        effective_ray_kwargs = (self.ray_kwargs if not ray_kwargs
                                else (self.ray_kwargs | dict(ray_kwargs)))
        _check_ray_kwargs(effective_ray_kwargs)
        start = time.perf_counter()
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
        if max_hull_mb is not None and (
            isinstance(max_hull_mb, (bool, np.bool_))
            or not isinstance(max_hull_mb, (int, float, np.number))
            or not np.isfinite(max_hull_mb)
            or max_hull_mb <= 0.0
        ):
            raise ValueError("max_hull_mb must be finite and positive when provided")
        budget = (int(max_moves) if max_moves is not None
                  else int(math.ceil(self.moves_per_vertex * live.n_vertices)))
        volume_before = live.volume
        calls_before = oracle.stats.n_calls
        strategy = RelocationRays(step=self.step, min_step=self.min_step, grow=self.grow)
        run = run_rays(live, oracle, axes, strategy, max_moves=budget,
                       max_vertices=live.n_vertices, max_hull_mb=max_hull_mb,
                       max_seconds=max_seconds, max_oracle_calls=max_oracle_calls,
                       ray_kwargs=effective_ray_kwargs,
                       ray_cache=ray_cache, first_order_screen=self.first_order_screen,
                       batch=self.batch)
        self.n_moves += run.counts["moves"]
        self.n_accepted += run.counts["accepted"]
        return {
            "refiner": self.name, "trace": run.trace, "stopped_because": run.stopped_because,
            "moves": run.counts["moves"], "accepted": run.counts["accepted"],
            "volume_before": volume_before, "volume_after": live.volume,
            "oracle_calls": oracle.stats.n_calls - calls_before,
            "seconds": time.perf_counter() - start,
        }

    def report(self) -> dict:
        return {"refiner": self.name, "step": self.step, "min_step": self.min_step,
                "first_order_screen": self.first_order_screen,
                "moves": self.n_moves, "accepted": self.n_accepted}


def get_refiner(name: str, **kwargs) -> Refiner:
    return REFINERS.get(name)(**kwargs)


def list_refiners() -> List[tuple]:
    return REFINERS.describe()


def resolve_refiner(refiner, **kwargs) -> Optional[Refiner]:
    """A name, an instance, or ``None`` (no refinement)."""
    if isinstance(refiner, str) and refiner != "none":
        return get_refiner(refiner, **kwargs)
    if kwargs:
        raise TypeError("refiner keyword arguments require a refiner name")
    if refiner is None or refiner == "none":
        return None
    if isinstance(refiner, Refiner):
        return refiner
    raise TypeError(f"refiner must be None, a registered name or a Refiner; got {refiner!r}")
