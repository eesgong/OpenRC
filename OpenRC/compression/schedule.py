r"""Compression schedules controlling when and how far to compress during growth.

Whereas a :class:`~OpenRC.compression.base.Compressor` selects which vertices
to prune, a schedule governs two online decisions during expansion:

* **whether** to pause the expansion and compress now, and
* **how much** - a vertex target, a hypervolume-loss budget, or both.

Schedules are a registry, independent of the compressor, so a new policy is
one class and works with every compressor and every builder that accepts a
schedule (currently :class:`~OpenRC.region.memory_aware.MemoryAwareBuilder`)::

    @SCHEDULES.register("my-rule", doc="...")
    class MyRule(CompressionSchedule):
        def decide(self, state):
            if state.n_facets > 50_000:
                return CompressionPlan(max_volume_loss_pct=0.5, reason="facets")
            return None

``schedule=None`` switches compression off entirely.

What a schedule sees
--------------------
:class:`GrowthState` is a snapshot taken once per expansion iteration: the
live hull's size (vertices, facets, modelled memory), the process RSS, the
volume, the oracle calls and time spent, and the growth trace since the last
compression.  All fields are read directly from the active
:class:`~OpenRC.geometry.live_hull.LiveHull` without extra hull constructions.

Registered schedules
--------------------
``vertex-budget``   the published rule: grow to ``M`` vertices, compress to
                    ``M_hat``.  Dimension-blind; kept for reproducibility.
``memory-budget``   watch the live hull's modelled memory.  At the high-water
                    mark, compress until the memory - measured after every
                    removal - is under the low-water mark.
``adaptive``        ``memory-budget``, plus two decisions taken from the
                    exploration itself: a vertex is only removed if it costs
                    less than a new vertex currently brings, and the run stops
                    once compressing no longer pays.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Real
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from OpenRC.registry import Registry

#: Registry of compression schedules.
SCHEDULES: Registry = Registry("compression schedule")


# --------------------------------------------------------------------------
# What a schedule sees, and what it returns
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GrowthState:
    """A snapshot of a growing region, taken once per expansion iteration.

    Attributes
    ----------
    dim, iteration:
        Region dimension; iterations since the current expansion started.
    n_points:
        Points the builder holds (what its ``max_vertices`` counts).
    n_vertices, n_facets, volume:
        Live-hull counts and volume.  Above ~4-D, memory and update time scale
        primarily with ``n_facets`` rather than ``n_vertices``.
    hull_mb:
        Modelled resident memory of the live hull
        (:func:`~OpenRC.geometry.live_hull.live_hull_memory_mb`).
    rss_mb:
        Process resident set size (0 when ``psutil`` is missing).
    memory_limit_mb:
        The RSS budget the run was given, if any.
    oracle_calls, seconds:
        Cumulative over the whole run.
    max_vertices:
        The builder's hard cap on ``n_points``.
    at_cap:
        ``True`` when the expansion stopped because it reached ``max_vertices``
        and the schedule is asked what to do about it.
    trace:
        Growth rows since the last compression, oldest first: ``n_vertices``
        (points held), ``n_hull_vertices``, ``n_facets``, ``hull_mb``,
        ``volume``, ``added`` (exact volume the step added), ``oracle_calls``,
        ``seconds`` ...
    round:
        Index of the expansion round (0 before the first compression).
    """

    dim: int
    iteration: int
    n_points: int
    n_vertices: int
    n_facets: int
    volume: float
    hull_mb: float
    rss_mb: float
    memory_limit_mb: Optional[float]
    oracle_calls: int
    seconds: float
    max_vertices: int
    at_cap: bool = False
    trace: Sequence[dict] = ()
    round: int = 0


@dataclass
class CompressionPlan:
    """What to do at a compression point.

    Each field is a stopping rule of
    :meth:`~OpenRC.compression.base.Compressor.compress_live`; whichever binds
    first ends the compression, and at least one must be set:

    ``target_vertices``      stop at this many vertices;
    ``max_volume_loss_pct``  cumulative loss budget, % of the volume;
    ``max_memory_mb``        stop once the live hull's memory is this low;
    ``max_vertex_loss``      never remove a vertex costing more than this
                             (absolute volume).
    """

    target_vertices: Optional[int] = None
    max_volume_loss_pct: Optional[float] = None
    reason: str = ""
    max_memory_mb: Optional[float] = None
    max_vertex_loss: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if all(v is None for v in (self.target_vertices, self.max_volume_loss_pct,
                                   self.max_memory_mb, self.max_vertex_loss)):
            raise ValueError("a plan needs at least one stopping rule")
        if self.target_vertices is not None:
            self.target_vertices = _positive_int(
                self.target_vertices, "target_vertices must be a positive integer"
            )
        for label, value in (("max_volume_loss_pct", self.max_volume_loss_pct),
                             ("max_memory_mb", self.max_memory_mb),
                             ("max_vertex_loss", self.max_vertex_loss)):
            if value is not None:
                if (isinstance(value, (bool, np.bool_))
                        or not isinstance(value, Real)
                        or not (math.isfinite(float(value)) and float(value) >= 0.0)):
                    raise ValueError(f"{label} must be a finite non-negative number")
                setattr(self, label, float(value))

    def rules(self) -> dict:
        """The stopping rules as keyword arguments of ``compress_live``."""
        return {"target_vertices": self.target_vertices,
                "max_volume_loss_pct": self.max_volume_loss_pct,
                "max_memory_mb": self.max_memory_mb,
                "max_vertex_loss": self.max_vertex_loss}


class CompressionSchedule(ABC):
    """Decides when a growing region is compressed, and how far.

    Implement :meth:`decide`.  The other hooks have sensible defaults:
    :meth:`at_cap` handles an expansion that hit the builder's vertex cap,
    :meth:`observe` sees every compression result, :meth:`finished` can end the
    run early.
    """

    name: str = "schedule"

    def reset(self) -> None:
        """Clear learned state so a schedule object can be reused across runs."""

    @abstractmethod
    def decide(self, state: GrowthState) -> Optional[CompressionPlan]:
        """Called once per expansion iteration: ``None`` keeps expanding."""

    def at_cap(self, state: GrowthState) -> Optional[CompressionPlan]:
        """The expansion reached the vertex cap; default: halve the vertex set."""
        return CompressionPlan(target_vertices=max(state.dim + 1, state.n_vertices // 2),
                               reason="vertex cap reached")

    def observe(self, state: GrowthState, plan: CompressionPlan, result) -> None:
        """Feedback after each compression (``result`` is a CompressionResult)."""

    def finished(self, cycles: Sequence[dict]) -> Optional[str]:
        """Return a reason to stop the whole run, or ``None`` to continue."""
        return None

    def memory_cap(self, memory_limit_mb: Optional[float]) -> Optional[float]:
        """A hard cap on the live hull's memory, enforced before each insertion.

        ``None`` (default) disables the cap.  Memory schedules return their
        hull budget so expansions check post-insertion memory before inserting.
        """
        return None

    def report(self) -> dict:
        return {"schedule": self.name}


def stalled(cycles: Sequence[dict], patience: int, tol: float) -> Optional[str]:
    r"""The potential rule: stop once the best volume stops rising.

    ``Phi_k``, the best volume after expansion round ``k``, never decreases and
    is bounded by the volume ``V*`` of the true region.  Requiring it to grow
    by a factor ``1 + tol`` at least once every ``patience`` rounds therefore
    ends any run within

    .. math::  \mathrm{patience} \cdot \Big(\Big\lceil
               \frac{\ln(V^*/\Phi_0)}{\ln(1+\mathrm{tol})}\Big\rceil + 1\Big)

    rounds, whatever the compressor, the policy or the random streams do - a
    run cannot cycle forever, because a cycle would have to hold ``Phi``
    constant.
    """
    if isinstance(cycles, (str, bytes, bytearray)):
        raise TypeError("cycles must be a sequence of cycle records, not str or bytes")
    if (isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, float, Real))
            or not np.isfinite(float(patience))
            or int(float(patience)) != float(patience) or float(patience) < 1):
        raise ValueError("patience must be a positive integer")
    if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, Real))
            or not np.isfinite(float(tol)) or float(tol) < 0.0):
        raise ValueError("tol must be a finite non-negative number")
    patience = int(patience)
    tol = float(tol)
    best = [c["best_volume"] for c in cycles if c["phase"] == "expand"]
    if len(best) > patience and best[-1] <= best[-1 - patience] * (1.0 + tol):
        return (f"best volume rose by less than {tol:g} (relative) over the last "
                f"{patience} rounds")
    return None


def get_schedule(name: str, **kwargs) -> CompressionSchedule:
    return SCHEDULES.get(name)(**kwargs)


def list_schedules() -> List[tuple]:
    return SCHEDULES.describe()


def resolve_schedule(schedule, **kwargs) -> Optional[CompressionSchedule]:
    """A name, an instance, or ``None``/``"none"`` (compression switched off)."""
    if isinstance(schedule, str) and schedule != "none":
        return get_schedule(schedule, **kwargs)
    if kwargs:
        raise TypeError("schedule options apply only when the schedule is given by name")
    if schedule is None or schedule == "none":
        return None
    if isinstance(schedule, CompressionSchedule):
        return schedule
    raise TypeError(
        f"schedule must be None, a registered name or a CompressionSchedule; got {schedule!r}"
    )


def _positive_int(value, message: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(numeric) or numeric < 1.0 or int(numeric) != numeric:
        raise ValueError(message)
    return int(numeric)


# --------------------------------------------------------------------------
# The published rule
# --------------------------------------------------------------------------
@SCHEDULES.register(
    "vertex-budget",
    doc="Grow to M vertices, compress to M_hat - the published rule (dimension-blind).",
)
class VertexBudgetSchedule(CompressionSchedule):
    """Compress to ``target_vertices`` whenever the builder reaches its cap.

    This is the rule of the memory-aware paper: the builder's ``max_vertices``
    is ``M`` and ``target_vertices`` is ``M_hat``.  It never interrupts an
    expansion, so a run follows the published algorithm step for step.
    Its weakness is that it ignores dimension: the same step that costs well
    under 1 % of a 3-D region's volume costs a large share of a
    high-dimensional one, whose vertices are each worth much more.
    """

    name = "vertex-budget"

    def __init__(self, *, target_vertices: int) -> None:
        self.target_vertices = _positive_int(
            target_vertices, "target_vertices must be a positive integer"
        )

    def decide(self, state: GrowthState) -> Optional[CompressionPlan]:
        return None

    def at_cap(self, state: GrowthState) -> Optional[CompressionPlan]:
        return CompressionPlan(target_vertices=max(state.dim + 1, self.target_vertices),
                               reason=f"reached M={state.max_vertices} points")

    def report(self) -> dict:
        return {"schedule": self.name, "target_vertices": self.target_vertices}


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
@SCHEDULES.register(
    "memory-budget",
    doc="Compress when the live hull's memory reaches a high-water mark, until it is "
        "under a low-water mark.",
)
class MemoryBudgetSchedule(CompressionSchedule):
    """Keep the live hull within a memory budget.

    Parameters
    ----------
    budget_mb:
        Memory the hull may occupy.  ``None`` takes ``hull_share`` of the
        run's ``memory_limit_mb``; with neither, only the vertex cap binds.
    high_water, low_water:
        Compress once the hull reaches ``high_water * budget``; compress until
        it drops to ``low_water * budget``.  Hull memory is measured directly
        after each removal because compressed hulls retain more facets per
        vertex than growing hulls (e.g., in 8-D, pruning 108 -> 92 vertices
        freed 10 % of hull memory versus 45 % predicted by a growth-curve fit).
    rss_high_water:
        Also compress when the process RSS reaches this share of
        ``memory_limit_mb`` - the safety net for memory the model does not see.
    hull_share:
        Share of ``memory_limit_mb`` given to the hull when ``budget_mb`` is
        not set.
    max_loss_pct:
        Optional cap on the hypervolume one compression may cost.  The memory
        goal is a hard requirement, so leave it ``None`` unless an incomplete
        compression is acceptable.
    patience, round_tol:
        The potential rule (:func:`stalled`): stop once the best volume has
        not risen by ``round_tol`` in ``patience`` rounds.
    """

    name = "memory-budget"

    def __init__(self, *, budget_mb: Optional[float] = None, high_water: float = 0.9,
                 low_water: float = 0.5, rss_high_water: float = 0.9,
                 hull_share: float = 0.5, max_loss_pct: Optional[float] = None,
                 patience: int = 2, round_tol: float = 1e-3) -> None:
        if budget_mb is not None and (
                isinstance(budget_mb, (bool, np.bool_))
                or not isinstance(budget_mb, (int, float, Real))
                or not (math.isfinite(float(budget_mb)) and float(budget_mb) > 0)):
            raise ValueError("budget_mb must be finite and positive")
        if (isinstance(low_water, (bool, np.bool_)) or isinstance(high_water, (bool, np.bool_))
                or not isinstance(low_water, (int, float, Real))
                or not isinstance(high_water, (int, float, Real))
                or not 0.0 < float(low_water) < float(high_water) <= 1.0):
            raise ValueError("need 0 < low_water < high_water <= 1")
        if (isinstance(rss_high_water, (bool, np.bool_)) or isinstance(hull_share, (bool, np.bool_))
                or not isinstance(rss_high_water, (int, float, Real))
                or not isinstance(hull_share, (int, float, Real))
                or not 0.0 < float(rss_high_water) <= 1.0 or not 0.0 < float(hull_share) <= 1.0):
            raise ValueError("rss_high_water and hull_share must lie in (0, 1]")
        if max_loss_pct is not None and (
                isinstance(max_loss_pct, (bool, np.bool_))
                or not isinstance(max_loss_pct, (int, float, Real))
                or not (math.isfinite(float(max_loss_pct)) and float(max_loss_pct) >= 0.0)):
            raise ValueError("max_loss_pct must be None or a finite non-negative number")
        self.budget_mb = None if budget_mb is None else float(budget_mb)
        self.high_water = float(high_water)
        self.low_water = float(low_water)
        self.rss_high_water = float(rss_high_water)
        self.hull_share = float(hull_share)
        self.max_loss_pct = None if max_loss_pct is None else float(max_loss_pct)
        self.patience = _positive_int(patience, "patience must be >= 1 and round_tol >= 0")
        if isinstance(round_tol, (bool, np.bool_)):
            raise ValueError("patience must be >= 1 and round_tol >= 0")
        try:
            tol = float(round_tol)
        except (TypeError, ValueError):
            raise ValueError("patience must be >= 1 and round_tol >= 0") from None
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError("patience must be >= 1 and round_tol >= 0")
        self.round_tol = tol
        self.n_triggers = {"hull": 0, "rss": 0, "cap": 0}

    def reset(self) -> None:
        self.n_triggers = {"hull": 0, "rss": 0, "cap": 0}

    def budget(self, state: GrowthState) -> Optional[float]:
        return self.memory_cap(state.memory_limit_mb)

    def memory_cap(self, memory_limit_mb: Optional[float]) -> Optional[float]:
        if self.budget_mb is not None:
            return float(self.budget_mb)
        if memory_limit_mb:
            return self.hull_share * float(memory_limit_mb)
        return None

    def _trigger(self, state: GrowthState) -> Optional[str]:
        budget = self.budget(state)
        if budget is not None and state.hull_mb >= self.high_water * budget:
            self.n_triggers["hull"] += 1
            return (f"live hull {state.hull_mb:.1f} MiB >= {self.high_water:.0%} of "
                    f"{budget:.1f} MiB ({state.n_facets} facets)")
        if (state.memory_limit_mb and state.rss_mb
                and state.rss_mb >= self.rss_high_water * state.memory_limit_mb):
            self.n_triggers["rss"] += 1
            return (f"process RSS {state.rss_mb:.0f} MiB >= {self.rss_high_water:.0%} "
                    f"of {state.memory_limit_mb:.0f} MiB")
        return None

    def memory_goal(self, state: GrowthState) -> Optional[float]:
        """The live-hull memory a compression should get down to."""
        budget = self.budget(state)
        goal = None if budget is None else self.low_water * budget
        if state.hull_mb > 0.0 and (goal is None or goal >= state.hull_mb):
            return (self.low_water / self.high_water) * state.hull_mb
        return goal

    def decide(self, state: GrowthState) -> Optional[CompressionPlan]:
        reason = self._trigger(state)
        if reason is None:
            return None
        return CompressionPlan(target_vertices=state.dim + 1,
                               max_memory_mb=self.memory_goal(state),
                               max_volume_loss_pct=self.max_loss_pct, reason=reason)

    def at_cap(self, state: GrowthState) -> Optional[CompressionPlan]:
        self.n_triggers["cap"] += 1
        ratio = math.sqrt(self.low_water / self.high_water)
        target = max(state.dim + 1, min(state.n_vertices - 1, int(state.n_vertices * ratio)))
        return CompressionPlan(target_vertices=target, max_volume_loss_pct=self.max_loss_pct,
                               reason=f"reached the cap of {state.max_vertices} points")

    def finished(self, cycles: Sequence[dict]) -> Optional[str]:
        return stalled(cycles, self.patience, self.round_tol)

    def report(self) -> dict:
        return {"schedule": self.name, "budget_mb": self.budget_mb,
                "high_water": self.high_water, "low_water": self.low_water,
                "triggers": dict(self.n_triggers)}


# --------------------------------------------------------------------------
# Memory plus exploration progress
# --------------------------------------------------------------------------
@SCHEDULES.register(
    "adaptive",
    doc="Memory-triggered, with each compression's loss budget set by what the recent "
        "expansion can win back, and a stop once compressing no longer pays.",
)
class AdaptiveSchedule(MemoryBudgetSchedule):
    r"""Memory-triggered schedule with EWMA marginal-gain pricing and stopping.

    A compression is triggered, and aims at its memory goal, exactly as in
    ``memory-budget``.  What the exploration adds is a price for each vertex:
    removing a vertex only pays if the vertex that will take its place in the
    next expansion adds more volume than the removal cost.  With ``g`` the
    current value of a new vertex (volume it adds), each removal must cost at
    most

    .. math::  \ell_{\max} = \mathrm{payback} \cdot g,

    so every single removal is profitable in expectation.  The greedy
    compressors remove the cheapest vertex first, so this is the marginal rule
    "keep swapping while the swap pays": it frees as many vertices as are
    worth freeing, independently of the dimension.

    Estimating ``g``
    ----------------
    The per-vertex gains ``y_t`` (the ``added`` column of the growth trace)
    are noisy observations of a slowly drifting mean ``g_t`` - returns
    diminish as the region fills - i.e. the *local level* model
    ``g_t = g_{t-1} + w_t``, ``y_t = g_t + e_t``.  Its Kalman filter converges
    to the exponentially weighted average ``g_t = g_{t-1} + eta (y_t - g_{t-1})``;
    ``eta = 2 / (window + 1)`` uses ``window`` as the EWMA span, and
    the average is bias-corrected for its zero start.  The filter runs
    **across rounds**, so an expansion that only managed one step before the
    next trigger still has an estimate built from every earlier step.

    Stopping
    --------
    The run stops after a compression that could not free ``min_free``
    vertices at a profitable price, or by the potential rule of
    :func:`stalled`, which bounds the number of rounds of any run.

    Parameters
    ----------
    payback:
        Share of a new vertex's expected value a removal may cost; below 1
        keeps each swap profitable in expectation.
    window:
        EWMA span, in accepted vertices, of the estimate of ``g``.
    min_free:
        See *Stopping*.
    (plus every :class:`MemoryBudgetSchedule` parameter, including
    ``patience`` and ``round_tol``)
    """

    name = "adaptive"

    def __init__(self, *, payback: float = 0.5, window: int = 10, min_free: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if (isinstance(payback, (bool, np.bool_))
                or not isinstance(payback, (int, float, Real))
                or not 0.0 < float(payback) <= 1.0):
            raise ValueError("payback must lie in (0, 1]")
        self.payback = float(payback)
        self.window = _positive_int(window, "window and min_free must be >= 1")
        self.min_free = _positive_int(min_free, "window and min_free must be >= 1")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._stalled = False
        self._level = 0.0          # un-corrected filter state
        self._weight = 0.0         # 1 - (1 - eta)^t, the bias correction
        self._n_obs = 0
        self._seen: tuple = (-1, -1)
        self.history: List[dict] = []

    # --------------------------------------------------------------- g
    def _absorb(self, state: GrowthState) -> None:
        """Feed accepted vertex-addition rows of ``state.trace`` not seen yet."""
        eta = 2.0 / (self.window + 1.0)
        for pos, row in enumerate(state.trace):
            step_id = int(row["iteration"]) if "iteration" in row else pos
            key = (state.round, step_id)
            if key <= self._seen:
                continue
            self._seen = key
            if not row.get("accepted", True) or row.get("action") == "replace":
                continue
            if "added" in row:
                y = max(float(row["added"]), 0.0)
            elif "gain" in row:
                y = max(float(row["gain"]), 0.0)
            else:
                continue
            self._level += eta * (y - self._level)
            self._weight += eta * (1.0 - self._weight)
            self._n_obs += 1

    def vertex_value(self, state: GrowthState) -> float:
        """Current estimate of the volume one more vertex adds (``g``)."""
        self._absorb(state)
        return self._level / self._weight if self._weight > 0 else 0.0

    # ------------------------------------------------------------ plans
    def _plan(self, state: GrowthState, reason: str, *, target: Optional[int] = None,
              memory_goal: Optional[float] = None) -> CompressionPlan:
        g = self.vertex_value(state)
        if self._n_obs == 0:
            return CompressionPlan(
                target_vertices=target if target is not None else state.dim + 1,
                max_memory_mb=memory_goal,
                max_vertex_loss=None,
                max_volume_loss_pct=self.max_loss_pct,
                reason=f"{reason}; no expansion gains observed yet",
            )
        price = self.payback * g
        return CompressionPlan(
            target_vertices=target if target is not None else state.dim + 1,
            max_memory_mb=memory_goal, max_vertex_loss=price,
            max_volume_loss_pct=self.max_loss_pct,
            reason=f"{reason}; a removal may cost at most {price:.4g} "
                   f"(= {self.payback:g} x {g:.4g}, the value of a new vertex)")

    def decide(self, state: GrowthState) -> Optional[CompressionPlan]:
        self._absorb(state)
        reason = self._trigger(state)
        if reason is None:
            return None
        return self._plan(state, reason, memory_goal=self.memory_goal(state))

    def at_cap(self, state: GrowthState) -> Optional[CompressionPlan]:
        self.n_triggers["cap"] += 1
        ratio = math.sqrt(self.low_water / self.high_water)
        target = max(state.dim + 1, min(state.n_vertices - 1, int(state.n_vertices * ratio)))
        return self._plan(state, f"reached the cap of {state.max_vertices} points",
                          target=target)

    def observe(self, state: GrowthState, plan: CompressionPlan, result) -> None:
        freed = result.original_n_vertices - result.polytope.n_vertices
        if plan.max_vertex_loss is not None:
            self._stalled = freed < self.min_free
        self.history.append({"freed": int(freed), "loss_pct": result.volume_loss_pct,
                             "price": plan.max_vertex_loss,
                             "memory_goal_mb": plan.max_memory_mb,
                             "vertex_value": self._level / self._weight
                             if self._weight > 0 else 0.0})

    def finished(self, cycles: Sequence[dict]) -> Optional[str]:
        if self._stalled:
            return (f"compression could not free {self.min_free} vertices within "
                    "the marginal volume-loss cap")
        return super().finished(cycles)

    def report(self) -> dict:
        info = super().report()
        info.update(payback=self.payback, window=self.window, observations=self._n_obs,
                    compressions=list(self.history))
        return info


__all__ = [
    "SCHEDULES",
    "AdaptiveSchedule",
    "CompressionPlan",
    "CompressionSchedule",
    "GrowthState",
    "MemoryBudgetSchedule",
    "VertexBudgetSchedule",
    "get_schedule",
    "list_schedules",
    "resolve_schedule",
    "stalled",
]
