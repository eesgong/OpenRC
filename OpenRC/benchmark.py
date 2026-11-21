"""Paired multi-instance benchmarking and confidence-interval estimation.

* An :class:`Instance` specifies a network, region axes, an oracle, and a seed.
  :func:`random_instances` samples ``dim`` non-slack bus-quantity axes per
  ``(dim, seed)`` pair so each seed corresponds to a distinct region projection.
* A *method* is any callable
  ``method(case, axes, oracle, *, max_vertices, seed) -> RegionResult``.
  :data:`BENCHMARK_METHODS` holds a few examples; any function of that form can
  be passed instead.
* :func:`run_benchmark` runs every method on every instance.  Every method sees
  the same instances, so the comparisons are paired.  It returns one row per
  run.
* :func:`summarize` reports, per network, dimension and method, the mean with a
  Student-t interval.  Against a baseline method it also reports the
  geometric-mean ratio over the paired instances, with its interval and the win
  rate.

Example::

    from OpenRC.benchmark import random_instances, run_benchmark, summarize

    instances = random_instances("9bus", dims=[3, 5], seeds=range(8))
    runs = run_benchmark(["expansive", "grow-and-polish"], instances,
                         vertices_per_dim=6)
    print(summarize(runs, baseline="expansive"))
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from OpenRC.registry import Registry

logger = logging.getLogger(__name__)

#: Registry of example methods: ``method(case, axes, oracle, *, max_vertices, seed)``.
BENCHMARK_METHODS: Registry = Registry("benchmark method")

_RUN_COLUMNS = [
    "case", "dim", "seed", "axes", "oracle", "method", "max_vertices",
    "instance_id", "volume", "vertices", "stopped", "error", "oracle_calls", "seconds",
]


def _count(value, name: str, *, minimum: int = 1) -> int:
    """Validate a count before starting any expensive work."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if not math.isfinite(number) or number < minimum or int(number) != number:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _confidence(value: float) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, Real))
            or not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0):
        raise ValueError("confidence must be finite and lie strictly between 0 and 1")
    return float(value)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ("__dict__", tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ("__list__", tuple(_freeze_value(v) for v in value))
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__dict__":
        return {k: _thaw_value(v) for k, v in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__list__":
        return [_thaw_value(v) for v in value[1]]
    if isinstance(value, tuple):
        return tuple(_thaw_value(v) for v in value)
    return value


def _freeze_kwargs(kwargs: Union[Mapping[str, Any], Iterable[tuple[Any, Any]]]) -> tuple:
    items: Iterable[tuple[Any, Any]] = (
        kwargs.items() if isinstance(kwargs, Mapping) else kwargs
    )
    return tuple(sorted((str(k), _freeze_value(v)) for k, v in items))


def _thaw_kwargs(pairs: Union[Mapping[str, Any], Iterable[tuple[Any, Any]]]) -> Dict[str, Any]:
    items: Iterable[tuple[Any, Any]] = (
        pairs.items() if isinstance(pairs, Mapping) else pairs
    )
    return {str(k): _thaw_value(v) for k, v in items}


@dataclass(frozen=True)
class Instance:
    """One region to grow: a network, the axes, the oracle, and a label seed."""

    case: str
    axes: tuple
    oracle: str = "lindistflow"
    seed: int = 0
    case_kwargs: tuple = ()           # sorted ``(key, value)`` pairs, hashable
    oracle_kwargs: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.case, str) or not self.case.strip():
            raise ValueError("case must be a non-empty string")
        if not isinstance(self.oracle, str) or not self.oracle.strip():
            raise ValueError("oracle must be a non-empty string")
        if isinstance(self.axes, (str, bytes, bytearray)):
            raise ValueError("axes must be a sequence of axis labels, not a single string")
        axes_tuple = tuple(self.axes)
        if not axes_tuple or any(not isinstance(a, str) or not a.strip() for a in axes_tuple):
            raise ValueError("axes must be a non-empty sequence of non-empty axis labels")
        object.__setattr__(self, "axes", axes_tuple)
        object.__setattr__(self, "seed", _count(self.seed, "seed", minimum=0))
        object.__setattr__(self, "case_kwargs", _freeze_kwargs(self.case_kwargs))
        object.__setattr__(self, "oracle_kwargs", _freeze_kwargs(self.oracle_kwargs))

    @property
    def dim(self) -> int:
        return len(self.axes)

    def build(self):
        """The case, the resolved axes and a fresh oracle (fresh call counters)."""
        import OpenRC as orc

        case = orc.get_case(self.case, **_thaw_kwargs(self.case_kwargs))
        axes = case.resolve_axes(list(self.axes))
        return case, axes, orc.get_oracle(
            self.oracle, case, **_thaw_kwargs(self.oracle_kwargs)
        )


def random_instances(case: str, dims: Iterable[int], seeds: Iterable[int], *,
                     oracle: str = "lindistflow", quantities: Sequence[str] = ("p", "q"),
                     case_kwargs: Optional[dict] = None,
                     oracle_kwargs: Optional[dict] = None,
                     require_feasible_origin: bool = True,
                     max_draws: int = 50) -> List[Instance]:
    """``dim`` random bus quantities of ``case`` per (dimension, seed).

    With ``require_feasible_origin`` (default) a draw whose base point is
    infeasible is replaced by the next one from the same seed's stream; a
    region can only be grown from a feasible point.  Draws are reproducible:
    the same arguments give the same instances.
    """
    import OpenRC as orc

    if not isinstance(require_feasible_origin, (bool, np.bool_)):
        raise ValueError("require_feasible_origin must be a boolean")
    if isinstance(dims, (str, bytes, bytearray)):
        raise TypeError("dims must be an iterable of integers, not a string")
    if isinstance(seeds, (str, bytes, bytearray)):
        raise TypeError("seeds must be an iterable of integers, not a string")
    if isinstance(quantities, (str, bytes, bytearray)):
        raise TypeError("quantities must be a sequence of 'p'/'q' labels, not a single string")
    if case_kwargs is not None and not isinstance(case_kwargs, Mapping):
        raise TypeError("case_kwargs must be a dict")
    if oracle_kwargs is not None and not isinstance(oracle_kwargs, Mapping):
        raise TypeError("oracle_kwargs must be a dict")
    dims = tuple(_count(dim, "dim") for dim in dims)
    # Iterables may be generators: consume once, then reuse for every dimension.
    seeds = tuple(_count(seed, "seed", minimum=0) for seed in seeds)
    max_draws = _count(max_draws, "max_draws")
    quantities = tuple(quantities)
    if not quantities or any(q not in ("p", "q") for q in quantities) \
            or len(set(quantities)) != len(quantities):
        raise ValueError("quantities must be a non-empty sequence of unique 'p'/'q' labels")
    case_kwargs = dict(case_kwargs or {})
    oracle_kwargs = dict(oracle_kwargs or {})
    grid = orc.get_case(case, **case_kwargs)
    pool = [f"{bus}{q}" for bus in grid.non_slack_buses for q in quantities]
    for dim in dims:
        if not 1 <= dim <= len(pool):
            raise ValueError(f"{case!r} offers {len(pool)} axes; cannot draw {dim}")
    instances: List[Instance] = []
    if not dims or not seeds:
        return instances
    if require_feasible_origin:
        with orc.get_oracle(oracle, grid, **oracle_kwargs) as probe:
            probe_axes = grid.resolve_axes([pool[0]])
            if not probe.feasible_at(probe_axes, np.zeros(1)):
                raise RuntimeError(
                    f"base operating point of {case!r} is infeasible under {oracle!r}"
                )
    for dim in dims:
        for seed in seeds:
            rng = np.random.default_rng([int(seed), int(dim)])
            labels = tuple(sorted(rng.choice(pool, size=dim, replace=False).tolist(),
                                  key=lambda s: (int(s[:-1]), s[-1])))
            instances.append(
                Instance(case, labels, oracle, int(seed),
                         _freeze_kwargs(case_kwargs),
                         _freeze_kwargs(oracle_kwargs))
            )
    return instances


Method = Callable[..., Any]


def _resolve(method: Union[str, Method]) -> Method:
    if isinstance(method, str):
        return BENCHMARK_METHODS.get(method)
    if not callable(method):
        raise TypeError(f"benchmark method must be a registered name or callable; got {method!r}")
    return method


def run_benchmark(methods: Union[Sequence[Union[str, Method]], Mapping[str, Method]],
                  instances: Sequence[Instance], *,
                  vertices_per_dim: Optional[float] = None,
                  max_vertices: Optional[int] = None,
                  workers: int = 1,
                  progress: Optional[Callable[[str], None]] = None):
    """Run every method on every instance; one ``pandas`` row per run.

    Give the vertex budget either as ``vertices_per_dim`` (``N = ceil(k d)``,
    so the budget per dimension is the same in every dimension) or as a fixed
    ``max_vertices``.  Each row holds the instance, the method, the volume, the
    vertex count, the oracle calls, the wall time, and the stop reason.
    Exceptions raised by a method are captured in the ``error`` column with
    ``volume=NaN`` so remaining runs continue.

    ``workers > 1`` runs the (instance, method) pairs in that many processes
    (at most one per CPU),
    started with "spawn" as everywhere in OpenRC (:mod:`OpenRC.parallel`).
    Methods then travel by registered name, or must be picklable (defined at
    module level).  Rows come back in the same order either way.  Where spawn
    cannot work (a REPL, a notebook, a script read from stdin) the runs are
    sequential, with a warning.
    """
    import pandas as pd

    if (vertices_per_dim is None) == (max_vertices is None):
        raise ValueError("give exactly one of vertices_per_dim and max_vertices")
    # Cap process count at os.cpu_count() to avoid CPU oversubscription.
    workers = min(_count(workers, "workers"), os.cpu_count() or 1)
    if max_vertices is not None:
        max_vertices = _count(max_vertices, "max_vertices")
    elif vertices_per_dim is not None:
        if (isinstance(vertices_per_dim, (bool, np.bool_))
                or not isinstance(vertices_per_dim, (int, float, Real))
                or not math.isfinite(float(vertices_per_dim)) or float(vertices_per_dim) <= 0.0):
            raise ValueError("vertices_per_dim must be finite and strictly positive")
    if isinstance(methods, (str, bytes, bytearray)):
        raise TypeError("methods must be a sequence or mapping of methods, not a single string")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be a callable or None")
    if isinstance(instances, (str, bytes, bytearray)):
        raise TypeError("instances must be a sequence of Instance objects")
    instances = list(instances)
    if not all(isinstance(inst, Instance) for inst in instances):
        raise TypeError("instances must be a sequence of Instance objects")
    named: Dict[str, Union[str, Method]]
    if isinstance(methods, Mapping):
        named = {str(k): v for k, v in methods.items()}
    else:
        named = {(m if isinstance(m, str) else getattr(m, "__name__", repr(m))): m
                 for m in methods}
    for method in named.values():
        _resolve(method)                         # unknown names fail before any run
    jobs = []
    for instance_id, instance in enumerate(instances):
        budget = (max_vertices if max_vertices is not None
                  else int(math.ceil(float(vertices_per_dim) * instance.dim)))  # type: ignore[arg-type]
        jobs += [(instance, name, method, budget, instance_id) for name, method in named.items()]

    def report(row: Dict[str, Any]) -> None:
        if progress is not None:
            progress(f"{row['case']} {row['dim']}-D seed {row['seed']} {row['method']}: "
                     + (f"volume {row['volume']:.6g}" if not row["error"] else row["error"]))

    from OpenRC.parallel import spawn_is_safe

    if workers > 1 and not spawn_is_safe():
        logger.warning("run_benchmark: worker processes need __main__ to be a real file; "
                       "running sequentially")
        workers = 1
    rows: List[Dict[str, Any]] = []
    if workers == 1 or not jobs:
        for job in jobs:
            rows.append(_run_one(*job))
            report(rows[-1])
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # Each process gets its share of the cores for threaded geometry.
        share = max(1, (os.cpu_count() or 1) // int(workers))
        with ProcessPoolExecutor(max_workers=int(workers),
                                 mp_context=mp.get_context("spawn"),
                                 initializer=_share_cores, initargs=(share,)) as pool:
            for row in pool.map(_run_one, *zip(*jobs)):
                rows.append(row)
                report(row)
    return pd.DataFrame(rows, columns=_RUN_COLUMNS)


def _share_cores(threads: int) -> None:
    """Worker-process initialiser: cap the geometry threads of this process."""
    from OpenRC.geometry.live_hull import set_geometry_workers

    set_geometry_workers(threads)


def _run_one(instance: Instance, name: str, method: Union[str, Method],
             budget: int, instance_id: int = 0) -> Dict[str, Any]:
    """One (instance, method) run as a row; a module-level function, so it pickles."""
    # Append non-empty case_kwargs to the case label so summarize() and CSV
    # outputs keep distinct network configurations separate.
    label = instance.case + ("" if not instance.case_kwargs else ":" + ",".join(
        f"{k}={v}" for k, v in _thaw_kwargs(instance.case_kwargs).items()))
    row: Dict[str, Any] = dict(case=label, dim=instance.dim, seed=instance.seed,
                               axes=" ".join(instance.axes), oracle=instance.oracle,
                               method=name, max_vertices=budget, instance_id=instance_id)
    oracle = None
    reported_calls = 0
    start = time.perf_counter()
    try:
        case, axes, oracle = instance.build()
        result = _resolve(method)(case, axes, oracle, max_vertices=budget,
                                  seed=instance.seed)
        reported_calls = int(result.oracle_stats.get("n_calls", 0) or 0)
        row.update(volume=result.volume, vertices=result.n_vertices,
                   stopped=result.stopped_because, error="")
    except Exception as exc:          # recorded, see run_benchmark
        row.update(volume=math.nan, vertices=0, stopped="",
                   error=f"{type(exc).__name__}: {exc}")
    finally:
        if oracle is not None:
            try:
                oracle.close()
            except Exception as exc:
                if not row.get("error"):
                    row.update(volume=math.nan, vertices=0, stopped="",
                               error=f"{type(exc).__name__}: {exc}")
    calls = 0 if oracle is None else (oracle.stats.n_calls or reported_calls)
    row.update(oracle_calls=calls,
               seconds=time.perf_counter() - start)
    return row


@dataclass
class Interval:
    """A mean (or geometric mean) with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    n: int
    extra: Dict[str, float] = field(default_factory=dict)


def mean_interval(values: Union[Sequence[float], np.ndarray],
                  confidence: float = 0.95) -> Interval:
    """Mean and Student-t interval; a single value gets an infinite interval."""
    from scipy import stats

    confidence = _confidence(confidence)
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("values must be a sequence of numbers, not a string")
    if (np.asarray(values).dtype == bool
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, Real))
                   for v in values)):
        raise ValueError("values must be numeric, not boolean")
    x = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    n = len(x)
    if n == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    scale = float(np.max(np.abs(x)))
    scaled = x / scale if scale else x
    mean = float(scaled.mean()) * scale
    if n == 1:
        return Interval(mean, -math.inf, math.inf, 1)
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1)
                 * scaled.std(ddof=1) / math.sqrt(n)) * scale
    return Interval(mean, mean - half, mean + half, n)


def ratio_interval(ratios: Union[Sequence[float], np.ndarray],
                   confidence: float = 0.95) -> Interval:
    """Geometric-mean ratio with a t-interval on the log scale, and the win rate.

    The mean and interval need a finite positive ratio.  The win rate counts
    every decided instance: a ratio of 0 (the method found no volume) is a
    loss and an infinite one (the baseline found none) a win; leaving them
    out would report ``[2, 2, 0, 0]`` as a win rate of 1.  ``extra`` also
    gives how many decided ratios the mean leaves out (``n_excluded``).
    """
    confidence = _confidence(confidence)
    if isinstance(ratios, (str, bytes, bytearray)):
        raise TypeError("ratios must be a sequence of numbers, not a string")
    if (np.asarray(ratios).dtype == bool
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, Real))
                   for v in ratios)):
        raise ValueError("ratios must be numeric, not boolean")
    decided = np.asarray([float(v) for v in ratios if not np.isnan(float(v)) and float(v) >= 0], dtype=float)
    r = decided[np.isfinite(decided) & (decided > 0)]
    extra = {"win_rate": float(np.mean(decided > 1.0)) if len(decided) else math.nan,
             "tie_rate": float(np.mean(decided == 1.0)) if len(decided) else math.nan,
             "n_excluded": float(len(decided) - len(r))}
    if len(r) == 0:
        return Interval(math.nan, math.nan, math.nan, 0, extra=extra)
    log = mean_interval(np.log(r), confidence)

    def exponent(value: float) -> float:
        return math.inf if value > math.log(np.finfo(float).max) else math.exp(value)

    return Interval(exponent(log.estimate), exponent(log.low), exponent(log.high), log.n,
                    extra=extra)


def summarize(runs, *, metric: str = "volume", baseline: Optional[str] = None,
              confidence: float = 0.95, higher_is_better: Optional[bool] = None):
    """Per (network, dimension, method): mean and interval of ``metric``.

    With ``baseline``, also the paired geometric-mean ratio of ``metric`` to
    the baseline's on the same instance, with its interval and the fraction of
    instances where the method beats the baseline.  A ratio interval that
    excludes 1 is a difference the instances support at that confidence.
    ``higher_is_better`` (default: false for ``seconds``/``oracle_calls``,
    true otherwise) decides which side of 1 counts as a win.
    """
    if higher_is_better is None:
        higher_is_better = metric not in ("seconds", "oracle_calls")
    import pandas as pd

    _confidence(confidence)
    if metric not in runs.columns:
        raise KeyError(f"unknown metric column {metric!r}")
    if baseline is not None and len(runs) > 0 and baseline not in set(runs["method"]):
        raise KeyError(f"baseline method {baseline!r} is not present in runs")
    # Normalize NaN error entries produced when reading benchmark CSVs via pandas.
    ok = runs[runs["error"].fillna("").astype(str) == ""]
    multi_oracle = "oracle" in ok.columns and ok["oracle"].nunique() > 1
    group_cols = ["case", "oracle", "dim", "method"] if multi_oracle else ["case", "dim", "method"]
    rows = []
    for group_key, group in ok.groupby(group_cols, sort=False):
        if multi_oracle:
            case, oracle_val, dim, method = group_key
        else:
            case, dim, method = group_key
            oracle_val = None
        stat = mean_interval(group[metric].to_numpy(), confidence)
        row: Dict[str, Any] = dict(case=case, dim=dim, method=method, n=stat.n,
                                   mean=stat.estimate, ci_low=stat.low, ci_high=stat.high,
                                   calls=float(group["oracle_calls"].mean()),
                                   seconds=float(group["seconds"].mean()))
        if multi_oracle:
            row["oracle"] = oracle_val
        if baseline is not None:
            base_mask = (ok["case"] == case) & (ok["dim"] == dim) & (ok["method"] == baseline)
            if multi_oracle:
                base_mask = base_mask & (ok["oracle"] == oracle_val)
            base = ok[base_mask]
            keys = (["instance_id"] if "instance_id" in ok
                    else (["seed", "axes", "oracle"] if "oracle" in ok else ["seed", "axes"]))
            paired = group.merge(base, on=keys, suffixes=("", "_base"), validate="one_to_one")
            num = paired[metric].to_numpy(dtype=float)
            den = paired[f"{metric}_base"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where((num == 0.0) & (den == 0.0), 1.0, num / den)
            ratio = ratio_interval(ratios, confidence)
            if not higher_is_better:
                with np.errstate(divide="ignore"):
                    ratio.extra["win_rate"] = ratio_interval(1.0 / ratios).extra["win_rate"]
            row.update(ratio=ratio.estimate, ratio_low=ratio.low, ratio_high=ratio.high,
                       win_rate=ratio.extra.get("win_rate", math.nan),
                       tie_rate=ratio.extra.get("tie_rate", math.nan),
                       ratio_excluded=int(ratio.extra.get("n_excluded", 0)))
        rows.append(row)
    columns = (["case", "oracle", "dim", "method"] if multi_oracle
               else ["case", "dim", "method"]) + [
        "n", "mean", "ci_low", "ci_high", "calls", "seconds"
    ]
    if baseline is not None:
        columns += ["ratio", "ratio_low", "ratio_high", "win_rate", "tie_rate", "ratio_excluded"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Example methods
# ---------------------------------------------------------------------------
def _expansive(seed_kind: str):
    def method(case, axes, oracle, *, max_vertices, seed):
        from OpenRC.region.expansive import ExpansiveBuilder

        return ExpansiveBuilder(policy="adaptive-ema", seed=seed_kind, seed_rng=seed).build(
            case, axes, oracle, max_vertices=max_vertices)
    return method


def _rays(strategy: str):
    def method(case, axes, oracle, *, max_vertices, seed):
        from OpenRC.region.ray_growth import RayGrowthBuilder

        return RayGrowthBuilder(strategy=strategy, seed_rng=seed).build(
            case, axes, oracle, max_vertices=max_vertices)
    return method


def _compress_and_regrow(screen: Optional[str], weighting: str = "count"):
    def method(case, axes, oracle, *, max_vertices, seed):
        from OpenRC.region.expansive import ExpansiveBuilder
        from OpenRC.region.memory_aware import MemoryAwareBuilder

        return MemoryAwareBuilder(
            inner_builder=ExpansiveBuilder(policy="adaptive-ema", seed_rng=seed),
            max_vertices=max_vertices,
            target_vertices=max(len(axes) + 1, int(0.75 * max_vertices)),
            rounds=4, facet_screen=screen,
            screen_kwargs={"weighting": weighting} if screen else None,
        ).build(case, axes, oracle)
    return method


BENCHMARK_METHODS.add("expansive", _expansive("auto"),
                      doc="adaptive-ema facet growth from the default seed")
BENCHMARK_METHODS.add("expansive-simplex", _expansive("simplex"),
                      doc="adaptive-ema facet growth from the simplex seed")
BENCHMARK_METHODS.add("expansive-axis", _expansive("axis"),
                      doc="adaptive-ema facet growth from the axis (cross-polytope) seed")
BENCHMARK_METHODS.add("facet-queue", _rays("facet-queue"),
                      doc="ray growth, one ray per facet in queue order")
BENCHMARK_METHODS.add("grow-and-polish", _rays("grow-and-polish"),
                      doc="ray growth, then gradient relocation at the vertex budget")
BENCHMARK_METHODS.add("compress-regrow", _compress_and_regrow(None),
                      doc="published vertex budget: grow to N, compress to 3N/4, regrow")
BENCHMARK_METHODS.add("compress-regrow+debt", _compress_and_regrow("debt"),
                      doc="grow/compress/regrow with facet-debt screening (count weighting)")
BENCHMARK_METHODS.add("compress-regrow+debt-cone", _compress_and_regrow("debt", "cone"),
                      doc="grow/compress/regrow with facet-debt screening (cone weighting)")


def list_benchmark_methods():
    return BENCHMARK_METHODS.describe()
