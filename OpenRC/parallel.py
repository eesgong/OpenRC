"""Parallel evaluation of feasibility oracles.

Batch feasibility queries during seeding, region expansion, and validation are
independent and can be evaluated concurrently across worker processes or threads.

Why processes rather than threads
----------------------------------
A Pyomo model and the solver object behind it are stateful, and neither MOSEK's
nor Gurobi's Python bindings promise thread safety for concurrent solves on one
model.  OpenRC therefore uses **one oracle per worker process**, built inside
that process, so nothing is shared.  Threads are offered as a backend when the
bottleneck sits in a solver that releases the GIL, while processes remain the
default.

What crosses the process boundary
----------------------------------
Only plain data: the *names* of a case and an oracle, a keyword dict, the axis
labels, and an array of points.  Workers reconstruct the case, axes, and oracle
from these registry keys and serialized keyword arguments without pickling live
solver handles or ``pandapower`` networks.

Determinism
-----------
Results are returned in input order regardless of worker count, so a run with
``n_workers=8`` gives the same verdicts as one with ``n_workers=1``.
"""

from __future__ import annotations

import os
import pickle
import threading
from collections import OrderedDict
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

import logging

if TYPE_CHECKING:
    from OpenRC.powerflow.base import OracleStats

LOGGER = logging.getLogger("OpenRC")

#: Per-thread cache of built oracles.  Processes naturally get their own copy;
#: threads must not share a stateful Pyomo model or solver handle.
_WORKER_LOCAL = threading.local()

# A long-lived worker may see many cases or parameter recipes.  Never retain
# an unbounded number of native solver models in its thread-local cache.
_MAX_CACHED_CONTEXTS = 4


def spawn_is_safe() -> bool:
    """Can ``multiprocessing`` re-import ``__main__`` in a spawned child?

    The "spawn" start method restarts the interpreter and re-imports the entry
    module, so it only works when ``__main__`` came from a real file *and* that
    file guards its top-level work with ``if __name__ == "__main__":``.  From a
    REPL, a notebook, or ``python - <<EOF`` there is no such file and every
    worker dies on start-up.

    When ``spawn`` is unsafe, :class:`FeasibilityPool` logs a warning and falls
    back to a thread pool instead of failing with ``BrokenProcessPool``.  A
    script without the guard would otherwise be re-run from the top by every
    worker, and a worker process must not spawn a nested pool.
    """
    import __main__
    import multiprocessing

    if multiprocessing.parent_process() is not None:
        return False
    path = getattr(__main__, "__file__", None)
    if not (isinstance(path, (str, bytes, os.PathLike)) and os.path.isfile(path)):
        return False
    return _has_main_guard(os.fsdecode(path))


def _has_main_guard(path: str) -> bool:
    """Does the module at ``path`` contain a top-level ``if __name__ == "__main__":``?"""
    import ast

    try:
        with open(path, "rb") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, ValueError):
        return True           # cannot tell; keep the previous behaviour
    for node in tree.body:
        test = getattr(node, "test", None) if isinstance(node, ast.If) else None
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)):
            sides = [test.left, *test.comparators]
            if (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
                    and any(isinstance(s, ast.Constant) and s.value == "__main__"
                            for s in sides)):
                return True
    return False


#: Assumed cost of starting a spawned worker pool before anything is measured.
#: A fresh interpreter imports pandapower, pyomo, and solver bindings and
#: rebuilds the case (~1 s per worker on larger networks).  Uses a conservative
#: prior before session timing so small batches avoid process start-up overhead.
POOL_STARTUP_SECONDS = 6.0

#: Cached wall-clock start-up times recorded by :func:`_make_pool`, keyed by
#: ``(case, oracle, workers)`` because each worker builds its own case and
#: oracle instance.
_MEASURED_STARTUP: Dict[Tuple[str, str, int], float] = {}


def measured_startup_seconds(default: float = POOL_STARTUP_SECONDS,
                             key: Optional[Tuple[str, str, int]] = None) -> float:
    """Pool start-up cost observed in this session.

    If ``key`` ``(case, oracle, workers)`` is given, return the measured
    start-up time for that configuration or ``default`` if unmeasured; otherwise
    return the maximum start-up time recorded in this session (or ``default``).
    """
    if (isinstance(default, (bool, np.bool_))
            or not isinstance(default, (int, float, Real))
            or not np.isfinite(float(default)) or float(default) < 0.0):
        raise ValueError("default must be a non-negative finite number")
    if key is not None:
        if isinstance(key, (str, bytes, bytearray)):
            raise TypeError("key must be a (case, oracle, workers) tuple, not str or bytes")
        try:
            k_case, k_oracle, k_workers = key
        except (TypeError, ValueError) as exc:
            raise ValueError("key must be a 3-tuple (case, oracle, workers)") from exc
        if (not isinstance(k_case, str) or not isinstance(k_oracle, str)
                or isinstance(k_workers, (bool, np.bool_))
                or not isinstance(k_workers, Integral) or int(k_workers) < 1):
            raise ValueError("key must be (case: str, oracle: str, workers: positive int)")
        return _MEASURED_STARTUP.get((k_case, k_oracle, int(k_workers)), float(default))
    values = list(_MEASURED_STARTUP.values())
    return max(values) if values else float(default)


def resolve_workers(n_workers) -> int:
    """Turn ``None`` / ``-1`` / ``0`` / ``"auto"`` into a concrete worker count.

    ``None`` and ``1`` mean sequential, ``-1`` and ``"all"`` mean every core, and
    anything larger than the core count is clamped to it.  ``"auto"`` resolves
    to the core count here; the decision whether to *use* it is made by
    :func:`plan_workers`, which needs to know how much work there is.
    """
    cores = os.cpu_count() or 1
    if isinstance(n_workers, str):
        if n_workers in ("auto", "all"):
            return cores
        raise ValueError(f"n_workers must be an int, None, 'auto' or 'all', got {n_workers!r}")
    if n_workers is None:
        return 1
    if isinstance(n_workers, (bool, np.bool_)) or not isinstance(n_workers, Integral):
        raise ValueError("n_workers must be an integer, None, 'auto' or 'all'")
    if n_workers in (0, 1):
        return 1
    if n_workers == -1:
        return cores
    if n_workers < -1:
        raise ValueError("negative n_workers values other than -1 are invalid")
    return min(int(n_workers), cores)


#: Required ratio of estimated parallel time savings to pool start-up cost
#: before ``n_workers="auto"`` launches a process pool.
PARALLEL_MARGIN = 1.5


def plan_workers(n_items: int, seconds_per_item: float, n_workers,
                 startup_seconds: Optional[float] = None,
                 margin: float = PARALLEL_MARGIN,
                 key: Optional[Tuple[str, str]] = None) -> int:
    r"""Select the worker count for ``n_items`` tasks of ``seconds_per_item`` duration.

    For ``n_workers="auto"``, uses multiple workers only when the estimated
    time saving exceeds ``margin * startup_seconds``:

    .. math:: n \cdot t \cdot (1 - 1/W) > \text{margin} \cdot \text{startup}

    Explicit integer ``n_workers`` values bypass this heuristic via
    :func:`resolve_workers`.  ``key`` specifies ``(case, oracle)`` to look up
    the corresponding measured pool start-up time.
    """
    if isinstance(n_items, (bool, np.bool_)) or not isinstance(n_items, Integral) or n_items < 0:
        raise ValueError("n_items must be a non-negative integer")
    if (isinstance(seconds_per_item, (bool, np.bool_))
            or not isinstance(seconds_per_item, (int, float, Real))
            or not np.isfinite(float(seconds_per_item)) or float(seconds_per_item) < 0.0):
        raise ValueError("seconds_per_item must be a non-negative finite number")
    if startup_seconds is not None and (
            isinstance(startup_seconds, (bool, np.bool_))
            or not isinstance(startup_seconds, (int, float, Real))
            or not np.isfinite(float(startup_seconds)) or float(startup_seconds) < 0.0):
        raise ValueError("startup_seconds must be a non-negative finite number")
    if (isinstance(margin, (bool, np.bool_))
            or not isinstance(margin, (int, float, Real))
            or not np.isfinite(float(margin)) or float(margin) < 0.0):
        raise ValueError("margin must be a non-negative finite number")
    if key is not None:
        if isinstance(key, (str, bytes, bytearray)):
            raise TypeError("key must be a (case, oracle) tuple, not str or bytes")
        try:
            k_case, k_oracle = key
        except (TypeError, ValueError) as exc:
            raise ValueError("key must be a 2-tuple (case, oracle)") from exc
        if not isinstance(k_case, str) or not isinstance(k_oracle, str):
            raise TypeError("key elements must be strings (case, oracle)")
    if not isinstance(n_workers, str) or n_workers not in ("auto",):
        return resolve_workers(n_workers)

    workers = resolve_workers("auto")
    if startup_seconds is None:
        startup_seconds = measured_startup_seconds(
            key=None if key is None else (key[0], key[1], workers))
    if workers <= 1 or n_items <= 1:
        return 1
    sequential = int(n_items) * float(seconds_per_item)
    saving = sequential * (1.0 - 1.0 / workers)
    if saving <= margin * startup_seconds:
        LOGGER.info(
            "auto: staying sequential (%d items x %.1f ms = %.1fs; a %d-worker pool "
            "would save %.1fs, short of the %.1fs needed to be clearly worth "
            "~%.1fs of start-up)",
            n_items, 1000 * seconds_per_item, sequential, workers, saving,
            margin * startup_seconds, startup_seconds,
        )
        return 1
    LOGGER.info(
        "auto: using %d workers (%d items x %.1f ms = %.1fs sequential)",
        workers, n_items, 1000 * seconds_per_item, sequential,
    )
    return workers


#: Case fields a worker must reproduce; the rest describe or index the case.
_CASE_FIELDS = ("name", "base_p_mw", "base_q_mvar", "v_nom_kv", "vm_min_pu", "vm_max_pu",
                "load_scale", "slack_bus_override", "slack_vm_pu", "line_charging")
#: Net entries a power flow writes: results (``res_*``), internals (``_*``) and
#: these flags.  Running pandapower on a case does not change its network.
_NET_STATE = ("converged", "OPF_converged")
_SERIAL_WARNINGS: set = set()


def _recipe_mismatch(case) -> Optional[str]:
    """Compare ``case`` against ``reference_case(case.registry_name, case.factory_kwargs)``.

    Detects post-construction or ``dataclasses.replace`` modifications that
    worker processes would not reproduce from the registry recipe.
    """
    import pandas as pd

    from OpenRC.cases.base import GridCase, validate_case_configuration
    from OpenRC.cases.builtin import reference_case

    validate_case_configuration(case)
    try:
        reference = reference_case(case.registry_name, getattr(case, "factory_kwargs", None))
    except Exception as exc:  # the recipe itself no longer builds
        return f"its recipe does not rebuild ({type(exc).__name__}: {exc})"
    for name in _CASE_FIELDS:
        mine, theirs = getattr(case, name, None), getattr(reference, name, None)
        if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
            a, b = np.asarray(mine, dtype=float), np.asarray(theirs, dtype=float)
            same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        else:
            same = mine == theirs
        if not same:
            return name
    net, ref_net = case.net, reference.net
    keys = {key for table in (net, ref_net) for key in table.keys()
            if not key.startswith(("res_", "_")) and key not in _NET_STATE}
    for key in sorted(keys):
        mine, theirs = net.get(key, None), ref_net.get(key, None)
        if isinstance(mine, pd.DataFrame) or isinstance(theirs, pd.DataFrame):
            if not (isinstance(mine, pd.DataFrame) and isinstance(theirs, pd.DataFrame)
                    and mine.equals(theirs)):
                return f"net.{key}"
        elif (isinstance(mine, (bool, int, float, str, type(None))) and mine != theirs
              and not (mine != mine and theirs != theirs)):        # NaN equals NaN here
            return f"net.{key}"
    if (isinstance(case, GridCase) and isinstance(reference, GridCase)
            and case.configuration_id != reference.configuration_id):
        # Unlike solver working state, _openrc_* modelling markers and
        # physical dataclass extension fields must survive reconstruction.
        return "OpenRC modelling markers or extended configuration inputs"
    return None


@dataclass(frozen=True)
class OracleSpec:
    """Serializable specification for reconstructing a case and oracle in a worker.

    Contains only registry names, axis labels, and pickled option bytes, avoiding
    live solver handles or ``pandapower`` objects.
    """

    case: str
    oracle: str
    axes: Tuple[str, ...]
    case_kwargs: bytes = b""
    oracle_kwargs: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.case, str) or not self.case.strip():
            raise ValueError("OracleSpec.case must be a non-empty string")
        if not isinstance(self.oracle, str) or not self.oracle.strip():
            raise ValueError("OracleSpec.oracle must be a non-empty string")
        if isinstance(self.axes, (str, bytes, bytearray)):
            raise TypeError("OracleSpec.axes must be a tuple of axis label strings, not str or bytes")
        try:
            axes_tuple = tuple(self.axes)
        except TypeError as exc:
            raise TypeError("OracleSpec.axes must be an iterable of axis label strings") from exc
        if not axes_tuple or any(not isinstance(a, str) or not a.strip() for a in axes_tuple):
            raise ValueError("OracleSpec.axes must be a non-empty tuple of non-empty strings")
        object.__setattr__(self, "axes", axes_tuple)
        if not isinstance(self.case_kwargs, (bytes, bytearray)):
            raise TypeError("OracleSpec.case_kwargs must be bytes")
        if not isinstance(self.oracle_kwargs, (bytes, bytearray)):
            raise TypeError("OracleSpec.oracle_kwargs must be bytes")

    def case_options(self) -> dict:
        """Fresh case-factory options encoded in this worker recipe."""
        return pickle.loads(self.case_kwargs) if self.case_kwargs else {}

    def oracle_options(self) -> dict:
        """Fresh oracle-factory options encoded in this worker recipe."""
        return pickle.loads(self.oracle_kwargs) if self.oracle_kwargs else {}

    @classmethod
    def from_oracle(cls, oracle, axes) -> Optional["OracleSpec"]:
        """Spec for a live oracle, or ``None`` when it cannot be rebuilt.

        An oracle constructed directly rather than through
        :func:`OpenRC.powerflow.get_oracle` carries no registry name, so there is
        no recipe to send to a worker; callers fall back to sequential work.
        """
        name = getattr(oracle, "registry_name", None)
        if not name or not getattr(oracle.case, "name", None):
            return None
        return _spec_or_none(oracle.case, axes, name, getattr(oracle, "registry_kwargs", {}))

    @classmethod
    def build(cls, case, axes, oracle_name: str, oracle_kwargs: Optional[dict] = None
              ) -> "OracleSpec":
        """Construct a spec from live objects, validating that it is reproducible."""
        if not isinstance(oracle_name, str) or not oracle_name.strip():
            raise ValueError("oracle_name must be a non-empty string")
        from OpenRC.powerflow.base import _ORACLE_ALIASES

        oracle_name = _ORACLE_ALIASES.get(oracle_name, oracle_name)
        if oracle_kwargs is not None and not isinstance(oracle_kwargs, dict):
            raise TypeError("oracle_kwargs must be a dict")
        case_name = getattr(case, "name", None)
        if not case_name or getattr(case, "registry_name", None) != case_name:
            raise ValueError(
                "parallel evaluation needs a case returned by OpenRC.get_case so "
                "workers can reproduce its complete construction recipe; pass "
                "n_workers=1 for an ad-hoc or subsequently modified case"
            )
        changed = _recipe_mismatch(case)
        if changed is not None:
            raise ValueError(
                f"case {case_name!r} differs from what get_case rebuilds in a worker "
                f"({changed}), so workers would evaluate the original network; pass the "
                "change to get_case(...) as a factory option, or use n_workers=1"
            )
        try:
            # Bytes remain cheap, hashable and process-safe even when a case
            # option contains a dict/list (for example ``axis_presets``).
            case_bytes = pickle.dumps(getattr(case, "factory_kwargs", {}) or {},
                                      protocol=pickle.HIGHEST_PROTOCOL)
            oracle_bytes = pickle.dumps(oracle_kwargs or {}, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            # get_oracle keeps a recipe whenever its options deep-copy, and a
            # lambda or a live handle can deep-copy without pickling.
            raise ValueError(
                f"the options of oracle {oracle_name!r} on case {case_name!r} cannot be "
                f"sent to a worker ({type(exc).__name__}: {exc}); use n_workers=1"
            ) from None
        return cls(case=case_name, oracle=oracle_name, axes=tuple(axes.labels),
                   case_kwargs=case_bytes, oracle_kwargs=oracle_bytes)


def _spec_or_none(case, axes, oracle_name: str,
                  oracle_kwargs: Optional[dict] = None) -> Optional[OracleSpec]:
    """A worker recipe, or ``None`` (with a warning) when workers cannot rebuild
    this case; every caller then evaluates the live case in this process."""
    try:
        return OracleSpec.build(case, axes, oracle_name, oracle_kwargs)
    except ValueError as exc:
        # Asked once per batch; one warning per distinct reason is enough.
        message = str(exc)
        if message not in _SERIAL_WARNINGS and len(_SERIAL_WARNINGS) < 100:
            _SERIAL_WARNINGS.add(message)
            LOGGER.warning("running serially: %s", message)
        return None


def _build_context(spec: OracleSpec):
    """Rebuild an independent oracle from its complete worker recipe."""
    from OpenRC.cases import get_case
    from OpenRC.powerflow import get_oracle

    case = get_case(spec.case, **spec.case_options())
    axes = case.resolve_axes(list(spec.axes))
    oracle = get_oracle(spec.oracle, case, **spec.oracle_options())
    return case, axes, oracle


def clear_worker_cache() -> None:
    """Release cached oracles in the calling thread (not other workers).

    Synchronous pool shutdown closes its owned thread caches. This explicit hook
    is useful for applications that call worker helpers in a long-lived thread.
    """
    cache = getattr(_WORKER_LOCAL, "cache", None)
    if cache is not None:
        _close_context_caches([cache])


def _close_context_caches(caches) -> None:
    """Close all cached oracles via :class:`ExitStack` so an error in one does not skip others."""
    with ExitStack() as cleanup:
        for cache in caches:
            while cache:
                _, (_, _, oracle) = cache.popitem(last=False)
                cleanup.callback(oracle.close)


class _WorkerThreadPool(ThreadPoolExecutor):
    """Keep worker-local reuse but explicitly close caches after a joined exit.

    Only public executor hooks are used. Waiting for workers before cleanup
    ensures no solver is closed while a query is still using it. Internal
    OpenRC owners always use ``shutdown(wait=True)`` or a context manager.
    """

    def __init__(self, max_workers: int) -> None:
        self._context_caches: List[OrderedDict] = []
        self._cache_lock = threading.Lock()
        super().__init__(max_workers=max_workers, initializer=self._initialise_worker)

    def _initialise_worker(self) -> None:
        cache: OrderedDict = OrderedDict()
        _WORKER_LOCAL.cache = cache
        with self._cache_lock:
            self._context_caches.append(cache)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait:
            with self._cache_lock:
                caches, self._context_caches = self._context_caches, []
            _close_context_caches(caches)


def _worker_context(spec: OracleSpec):
    """Reuse one of a bounded number of oracles in this worker thread."""
    key = (spec.case, spec.oracle, spec.axes, spec.case_kwargs, spec.oracle_kwargs)
    cache = getattr(_WORKER_LOCAL, "cache", None)
    if cache is None:
        cache = OrderedDict()
        _WORKER_LOCAL.cache = cache
    if key not in cache:
        cache[key] = _build_context(spec)
        if len(cache) > _MAX_CACHED_CONTEXTS:
            _, (_, _, evicted) = cache.popitem(last=False)
            evicted.close()
    cache.move_to_end(key)
    return cache[key]


def _worker_batch(args) -> List[bool]:
    """Answer a batch of points inside a worker process."""
    spec, points = args
    _, axes, oracle = _worker_context(spec)
    return oracle.feasible_mask(axes, points).tolist()


def _worker_batch_timed(args):
    """Boolean batch plus query time, excluding worker construction/start-up."""
    spec, points = args
    _, axes, oracle = _worker_context(spec)
    before = oracle.stats.seconds
    values = oracle.feasible_mask(axes, points)
    return values.tolist(), oracle.stats.seconds - before


def _worker_batch_verdicts(args) -> List[str]:
    """Three-valued version of :func:`_worker_batch`, for validation sweeps."""
    spec, points = args
    _, axes, oracle = _worker_context(spec)
    return [oracle.verdict_at(axes, point).value for point in points]


def _worker_batch_verdicts_timed(args):
    """Verdict strings plus query time, excluding worker construction/start-up."""
    spec, points = args
    _, axes, oracle = _worker_context(spec)
    before = oracle.stats.seconds
    values = [oracle.verdict_at(axes, point).value for point in points]
    return values, oracle.stats.seconds - before


def _worker_ray(args) -> List[float]:
    """Run one boundary search inside a worker and return the boundary point.

    Each ray is a sequential ~30-call bisection, but rays are independent of one
    another, so a seeding sweep over 2d axis directions parallelises across them.
    """
    spec, direction, origin, kwargs = args
    _, axes, oracle = _worker_context(spec)
    from OpenRC.geometry.bisection import ray_boundary

    feasible_before, seconds_before = oracle.stats.n_feasible, oracle.stats.seconds
    result = ray_boundary(oracle, axes, np.asarray(direction, dtype=float),
                          origin=None if origin is None else np.asarray(origin, float),
                          **kwargs)
    return [float(v) for v in result.boundary] + [
        float(result.n_oracle_calls), float(oracle.stats.n_feasible - feasible_before),
        oracle.stats.seconds - seconds_before,
    ]


def map_rays(case, axes, oracle_name: str, directions, *, origin=None,
             n_workers: Optional[int] = 1, backend: str = "process",
             oracle_kwargs: Optional[dict] = None, ray_kwargs: Optional[dict] = None,
             stats: Optional["OracleStats"] = None):
    """Boundary points for many directions, optionally in parallel.

    Returns ``(boundary_points, total_oracle_calls)``.  Falls back to a
    sequential sweep when ``n_workers`` resolves to 1, or with a warning when
    workers could not rebuild ``case`` (see :func:`_spec_or_none`).  If
    supplied, ``stats`` receives the actual query count, feasible count and
    solver time, excluding pool start-up probes; the return value is the same
    pair either way.
    """
    if axes is None or isinstance(axes, (str, bytes, bytearray)):
        raise TypeError("axes must be a RegionAxes or sequence of axes, not None or str")
    if not isinstance(oracle_name, str) or not oracle_name.strip():
        raise ValueError("oracle_name must be a non-empty string")
    if oracle_kwargs is not None and not isinstance(oracle_kwargs, dict):
        raise TypeError("oracle_kwargs must be a dict")
    if ray_kwargs is not None and not isinstance(ray_kwargs, dict):
        raise TypeError("ray_kwargs must be a dict")
    raw_dir = np.asarray(directions)
    if raw_dir.dtype == bool:
        raise ValueError("directions must be a numeric array, not boolean")
    raw = np.asarray(raw_dir, dtype=float)
    ray_kwargs = dict(ray_kwargs or {})
    workers = resolve_workers(n_workers)
    if backend not in ("process", "thread"):
        raise ValueError("backend must be 'process' or 'thread'")
    if origin is not None:
        raw_orig = np.asarray(origin)
        if raw_orig.dtype == bool:
            raise ValueError("origin must be a numeric vector, not boolean")
        orig = np.asarray(raw_orig, dtype=float)
        if orig.shape != (len(axes),) or not np.all(np.isfinite(orig)):
            raise ValueError(f"origin must be a finite 1-D vector of length {len(axes)}")
    if raw.size == 0:
        if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[0] == 0
                             and raw.shape[1] in (0, len(axes))):
            return np.empty((0, len(axes)), dtype=float), 0
        raise ValueError("directions must be a finite (n, len(axes)) matrix")
    directions = np.atleast_2d(raw)
    if (directions.ndim != 2 or directions.shape[1] != len(axes)
            or not np.all(np.isfinite(directions))):
        raise ValueError("directions must be a finite (n, len(axes)) matrix")

    spec = _spec_or_none(case, axes, oracle_name, oracle_kwargs) if workers > 1 else None
    if spec is None:
        from OpenRC.geometry.bisection import ray_boundary
        from OpenRC.powerflow import get_oracle

        with get_oracle(oracle_name, case, **(oracle_kwargs or {})) as oracle:
            points, calls = [], 0
            for direction in directions:
                result = ray_boundary(oracle, axes, direction, origin=origin, **ray_kwargs)
                points.append(result.boundary)
                calls += result.n_oracle_calls
            if stats is not None and stats is not oracle.stats:
                stats.n_calls += calls
                stats.n_feasible += oracle.stats.n_feasible
                stats.seconds += oracle.stats.seconds
        return np.array(points), calls

    origin_list = None if origin is None else [float(v) for v in origin]
    tasks = [(spec, [float(v) for v in d], origin_list, ray_kwargs) for d in directions]
    with _make_pool(workers, backend, spec=spec) as pool:
        rows = list(pool.map(_worker_ray, tasks))
    boundary = np.array([r[:-3] for r in rows], dtype=float)
    calls = int(sum(r[-3] for r in rows))
    if stats is not None:
        stats.n_calls += calls
        stats.n_feasible += int(sum(r[-2] for r in rows))
        stats.seconds += sum(r[-1] for r in rows)
    return boundary, calls


@dataclass
class FeasibilityPool:
    """Evaluates many points against one oracle, optionally across processes.

    Use it as a context manager so the pool is always shut down::

        with FeasibilityPool(spec, n_workers=8) as pool:
            mask = pool.feasible(points)

    Parameters
    ----------
    spec:
        What each worker should build.
    n_workers:
        See :func:`resolve_workers`.  ``1`` runs in this process and builds the
        oracle once, which is the right choice for small batches where process
        start-up would dominate.
    backend:
        ``"process"`` (default) or ``"thread"``.
    chunk_size:
        Points per task.  Larger chunks amortise the hand-off; smaller ones
        balance the load better when solve times vary.  Chosen automatically
        when omitted.
    """

    spec: OracleSpec
    n_workers: int = 1
    backend: str = "process"
    chunk_size: Optional[int] = None
    _pool: Optional[Executor] = field(default=None, repr=False, init=False)
    _local: Optional[tuple] = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, OracleSpec):
            raise TypeError("spec must be an OracleSpec")
        self.n_workers = resolve_workers(self.n_workers)
        if self.chunk_size is not None:
            if (isinstance(self.chunk_size, (bool, np.bool_))
                    or not isinstance(self.chunk_size, Integral) or self.chunk_size <= 0):
                raise ValueError("chunk_size must be a positive integer when provided")
        if self.backend not in ("process", "thread"):
            raise ValueError(f"backend must be 'process' or 'thread', got {self.backend!r}")
        if self.n_workers > 1 and self.backend == "process" and not spawn_is_safe():
            LOGGER.warning(
                "process workers need an importable __main__ (a script file whose "
                "top-level work sits under `if __name__ == \"__main__\":`). This is a "
                "REPL, notebook, piped stdin, an unguarded script or a worker process, "
                "so OpenRC is falling back to threads; expect little speed-up, because "
                "the solver holds the GIL for most of each call."
            )
            self.backend = "thread"

    # ------------------------------------------------------------- lifecycle
    def __enter__(self) -> "FeasibilityPool":
        if self._pool is not None:
            raise RuntimeError("this FeasibilityPool is already active")
        if self.n_workers > 1:
            self._pool = _make_pool(self.n_workers, self.backend, spec=self.spec)
            LOGGER.info("feasibility pool: %d %s workers", self.n_workers, self.backend)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        pool, self._pool = self._pool, None
        local, self._local = self._local, None
        with ExitStack() as cleanup:
            if local is not None:
                cleanup.callback(local[2].close)
            if pool is not None:
                cleanup.callback(pool.shutdown, wait=True)

    # ------------------------------------------------------------ evaluation
    def _sequential_context(self):
        if self._local is None:
            # Build a dedicated oracle per sequential FeasibilityPool so closing
            # one pool does not close another pool's solver.
            self._local = _build_context(self.spec)
        return self._local

    def _chunks(self, points: np.ndarray) -> List[np.ndarray]:
        n = len(points)
        size = self.chunk_size
        if size is None:
            # Aim for a few chunks per worker so a slow point cannot stall a core.
            size = max(1, n // (self.n_workers * 4)) if self.n_workers > 1 else n
        return [points[i:i + size] for i in range(0, n, size)]

    def _validate_points(self, points) -> np.ndarray:
        raw_in = np.asarray(points)
        if raw_in.dtype == bool:
            raise ValueError("points must be a numeric array, not boolean")
        raw = np.asarray(raw_in, dtype=float)
        dim = len(self.spec.axes)
        if raw.size == 0:
            # Accept empty 1-D arrays or 2-D arrays with 0 rows and 0 or dim columns;
            # reject non-empty row counts with 0 columns (e.g. shape (3, 0)).
            if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[0] == 0
                                 and raw.shape[1] in (0, dim)):
                return np.empty((0, dim), dtype=float)
            raise ValueError(f"points must be a finite (n, {dim}) matrix, got shape {raw.shape}")
        pts = np.atleast_2d(raw)
        if pts.ndim != 2 or pts.shape[1] != dim or not np.all(np.isfinite(pts)):
            raise ValueError(f"points must be a finite (n, {dim}) matrix, got shape {raw.shape}")
        return pts

    def feasible(self, points: np.ndarray, *, stats: Optional["OracleStats"] = None) -> np.ndarray:
        """Boolean verdicts in input order, optionally charging query statistics.

        ``stats`` receives one event per row and summed worker query time,
        excluding executor start-up.  It may implement custom logging hooks.
        """
        points = self._validate_points(points)
        if len(points) == 0:
            return np.zeros(0, dtype=bool)

        if self.n_workers == 1 or self._pool is None:
            _, axes, oracle = self._sequential_context()
            before = oracle.stats.seconds
            values = oracle.feasible_mask(axes, points)
            if stats is not None and stats is not oracle.stats:
                stats.record_many(values, oracle.stats.seconds - before)
            return values

        tasks = [(self.spec, chunk) for chunk in self._chunks(points)]
        out: List[bool] = []
        if stats is None:
            for result in self._pool.map(_worker_batch, tasks):
                out.extend(result)
        else:
            for result, seconds in self._pool.map(_worker_batch_timed, tasks):
                out.extend(result)
                stats.record_many(result, seconds)
        return np.array(out, dtype=bool)

    def verdicts(self, points: np.ndarray, *,
                 stats: Optional["OracleStats"] = None) -> List[str]:
        """Three-valued verdict strings per row, in input order.

        ``stats`` receives one event per row (feasible when the verdict is) and
        summed worker query time, as in :meth:`feasible`.
        """
        points = self._validate_points(points)
        if len(points) == 0:
            return []

        if self.n_workers == 1 or self._pool is None:
            _, axes, oracle = self._sequential_context()
            before = oracle.stats.seconds
            values = [oracle.verdict_at(axes, p).value for p in points]
            if stats is not None and stats is not oracle.stats:
                stats.record_many([v == "feasible" for v in values],
                                  oracle.stats.seconds - before)
            return values

        tasks = [(self.spec, chunk) for chunk in self._chunks(points)]
        out: List[str] = []
        if stats is None:
            for result in self._pool.map(_worker_batch_verdicts, tasks):
                out.extend(result)
        else:
            for result, seconds in self._pool.map(_worker_batch_verdicts_timed, tasks):
                out.extend(result)
                stats.record_many([v == "feasible" for v in result], seconds)
        return out


def _init_process_worker(spec: Optional[OracleSpec]) -> None:
    """Pre-build the worker's case and oracle on process start-up without solving."""
    if spec is not None:
        _worker_context(spec)


def _worker_noop(_index: int) -> int:
    return _index


def _make_pool(n_workers: int, backend: str, *, spec: Optional[OracleSpec] = None):
    """Build the executor, using "spawn" so solver libraries start clean.

    A forked child inherits the parent's MOSEK/Gurobi environment, including any
    licence handle it has already checked out, which is a well-known source of
    hangs.  "spawn" starts each worker from a fresh interpreter instead.

    On the first call for a given ``(case, oracle, n_workers)`` key, initializes
    worker contexts and records the elapsed start-up time in
    :data:`_MEASURED_STARTUP`.
    """
    import time as _time
    from concurrent.futures import ProcessPoolExecutor

    if backend == "thread" or not spawn_is_safe():
        return _WorkerThreadPool(max_workers=n_workers)
    import multiprocessing as mp

    start = _time.perf_counter()
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_process_worker,
        initargs=(spec,),
    )
    key = None if spec is None else (spec.case, spec.oracle, int(n_workers))
    if spec is not None and key is not None and key not in _MEASURED_STARTUP:
        try:
            list(pool.map(_worker_noop, range(n_workers)))
            _MEASURED_STARTUP[key] = _time.perf_counter() - start
            LOGGER.info("pool start-up for %s on %r with %d workers measured at %.1f s",
                        key[1], key[0], key[2], _MEASURED_STARTUP[key])
        except Exception:  # the probe is best-effort only
            pass
    return pool


def evaluate_points(case, axes, oracle_name: str, points: np.ndarray, *,
                    n_workers=1, backend: str = "process",
                    oracle_kwargs: Optional[dict] = None,
                    chunk_size: Optional[int] = None,
                    calibration: int = 12) -> np.ndarray:
    """Feasibility of many points, sequentially or across workers.

    ``n_workers="auto"`` times a short calibration batch in this process, then
    selects a worker count via :func:`plan_workers` while retaining the
    calibration verdicts.  Cases not constructed via
    :func:`~OpenRC.cases.get_case` or modified afterwards are evaluated
    sequentially in the current process with a warning.
    """
    from OpenRC.powerflow import get_oracle
    from OpenRC.powerflow.base import _ORACLE_ALIASES

    if axes is None or isinstance(axes, (str, bytes, bytearray)):
        raise TypeError("axes must be a RegionAxes or sequence of axes, not None or str")
    if not isinstance(oracle_name, str) or not oracle_name.strip():
        raise ValueError("oracle_name must be a non-empty string")
    oracle_name = _ORACLE_ALIASES.get(oracle_name, oracle_name)
    if oracle_kwargs is not None and not isinstance(oracle_kwargs, dict):
        raise TypeError("oracle_kwargs must be a dict")
    if (isinstance(calibration, (bool, np.bool_))
            or not isinstance(calibration, Integral) or calibration < 1):
        raise ValueError("calibration must be a positive integer")
    if backend not in ("process", "thread"):
        raise ValueError(f"backend must be 'process' or 'thread', got {backend!r}")
    if chunk_size is not None and (
            isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, Integral) or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer when provided")
    resolved_workers = None if n_workers == "auto" else resolve_workers(n_workers)
    raw_in = np.asarray(points)
    if raw_in.dtype == bool:
        raise ValueError("points must be a numeric array, not boolean")
    raw = np.asarray(raw_in, dtype=float)
    dim = len(axes)
    if raw.size == 0:
        if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[0] == 0
                             and raw.shape[1] in (0, dim)):
            return np.zeros(0, dtype=bool)
        raise ValueError(f"points must be a finite (n, {dim}) matrix, got shape {raw.shape}")
    pts = np.atleast_2d(raw)
    if pts.ndim != 2 or pts.shape[1] != dim or not np.all(np.isfinite(pts)):
        raise ValueError(f"points must be a finite (n, {dim}) matrix, got shape {raw.shape}")

    if n_workers == "auto":
        import time as _time

        with get_oracle(oracle_name, case, **(oracle_kwargs or {})) as oracle:
            head = pts[:max(1, min(int(calibration), len(pts)))]

            # Evaluate head[0] untimed so one-time Pyomo model construction and
            # solver initialization do not bias the per-query runtime estimate.
            first = bool(oracle.feasible_at(axes, head[0]))
            timed = head[1:]
            if len(timed):
                start = _time.perf_counter()
                timed_mask = oracle.feasible_mask(axes, timed)
                per_item = (_time.perf_counter() - start) / len(timed)
                head_arr = np.concatenate([np.array([first], dtype=bool), timed_mask])
            else:
                per_item = 0.0
                head_arr = np.array([first], dtype=bool)

            rest = pts[len(head):]
            if len(rest) == 0:
                return head_arr
            workers = plan_workers(len(rest), per_item, "auto",
                                   key=(getattr(case, "name", ""), oracle_name))
            spec = _spec_or_none(case, axes, oracle_name, oracle_kwargs) if workers > 1 else None
            if spec is None:
                tail_values = oracle.feasible_mask(axes, rest)
                return np.concatenate([head_arr, tail_values])
        with FeasibilityPool(spec, n_workers=workers, backend=backend,
                             chunk_size=chunk_size) as pool:
            tail_array = pool.feasible(rest)
        return np.concatenate([head_arr, tail_array])

    workers = resolved_workers if resolved_workers is not None else resolve_workers(n_workers)
    spec = _spec_or_none(case, axes, oracle_name, oracle_kwargs) if workers > 1 else None
    if spec is None:
        with get_oracle(oracle_name, case, **(oracle_kwargs or {})) as oracle:
            return oracle.feasible_mask(axes, pts)

    with FeasibilityPool(spec, n_workers=workers, backend=backend,
                         chunk_size=chunk_size) as pool:
        return pool.feasible(pts)
