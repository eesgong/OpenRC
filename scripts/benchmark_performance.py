"""Cross-network performance: oracle queries, ray searches, builds, certificates.

All comparisons are paired on identical points, rays, or random instances and
reported with a 95 % Student-t confidence interval (on the log scale for
ratios, see :mod:`OpenRC.benchmark`).  Besides the timings, each section
checks that the faster setting gives the same answer: verdicts, boundaries,
volumes or certificate statuses.

Sections (``--sections``, default all):

``oracle``    time per query of every oracle configuration on points drawn
              around the LinDistFlow boundary (a mix of feasible and
              infeasible points, as a builder sees them).
``rays``      ray searches, bisection against margin-guided ITP.
``builds``    AC region builds: bisection with a fresh model per query, against
              the defaults (margin search, recycled model).
``geometry``  8-D compress-and-regrow with 1 geometry thread against the default.
``polish``    8-D grow-and-polish with ``batch=1`` against ``batch=8``.
``certify``   the inner-region certificate: relaxation screen, and workers.

Usage::

    python scripts/benchmark_performance.py                  # everything
    python scripts/benchmark_performance.py --sections oracle rays --points 40
    python scripts/benchmark_performance.py --out perf.json  # raw measurements

Solver-backed configurations need Gurobi or MOSEK and are skipped (with a note)
when a solver is not available.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = ["9bus", "pp:case33bw", "pp:mv_oberrhein:feeder=1",
         "pp:create_cigre_network_mv:load_scale=0.8"]
SECTIONS = ["oracle", "rays", "builds", "geometry", "polish", "certify"]

#: (family, label, oracle name, keyword arguments); the first entry of a family
#: is the reference its verdicts are compared with.
ORACLE_CONFIGS: List[Tuple[str, str, str, Dict[str, Any]]] = [
    ("LinDistFlow", "analytic", "lindistflow", {}),
    ("LinDistFlow", "solver, pyomo+mosek", "lindistflow", {"backend": "solver"}),
    ("LinDistFlow", "solver, mosek", "lindistflow", {"backend": "solver", "modeling": "mosek"}),
    ("LinDistFlow", "solver, gurobi", "lindistflow", {"backend": "solver", "modeling": "gurobi"}),
    ("SOCP", "pyomo+mosek", "socp", {}),
    ("SOCP", "mosek", "socp", {"modeling": "mosek"}),
    ("SOCP", "gurobi", "socp", {"modeling": "gurobi"}),
    ("SOCP", "gurobi + sweep", "socp", {"modeling": "gurobi", "candidate_solver": "sweep"}),
    ("exact", "pyomo+gurobi", "exact", {}),
    ("exact", "gurobi", "exact", {"modeling": "gurobi"}),
    ("exact", "gurobi + sweep (float)", "exact",
     {"modeling": "gurobi", "candidate_solver": "sweep",
      "candidate_options": {"certificate": "float"}}),
    ("exact", "gurobi + sweep (interval)", "exact",
     {"modeling": "gurobi", "candidate_solver": "sweep"}),
    ("AC", "fresh model", "acpf", {"recycle": False}),
    ("AC", "recycled model", "acpf", {}),
]


def _parse_scalar(raw: str):
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _parse_case(spec: str):
    """``name[:key=value[,key=value]]`` -> (name, kwargs), as in benchmark.py."""
    name, sep, options = spec.rpartition(":")
    if not sep or "=" not in options:
        if not spec.strip():
            raise ValueError(f"case specification must be non-empty, got {spec!r}")
        return spec, {}
    if not name.strip():
        raise ValueError(f"case specification must include a case name, got {spec!r}")
    kwargs: Dict[str, Any] = {}
    for item in options.split(","):
        if "=" not in item:
            raise ValueError(f"expected key=value in case specification {spec!r}, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"expected key=value in case specification {spec!r}, got {item!r}")
        kwargs[key] = _parse_scalar(value)
    return name, kwargs


def _short(spec: str) -> str:
    return {"pp:create_cigre_network_mv:load_scale=0.8": "CIGRE MV (0.8)"}.get(spec, spec)


def _seeds(n: int) -> str:
    return "seed 0" if n == 1 else f"seeds 0-{n - 1}"


def _instance(spec: str, dim: int, seed: int):
    from OpenRC.benchmark import random_instances

    name, kwargs = _parse_case(spec)
    return random_instances(name, [dim], [seed], case_kwargs=kwargs)[0]


def _ci(values: Sequence[float]) -> str:
    from OpenRC.benchmark import mean_interval

    stat = mean_interval(values)
    if stat.n < 2:
        return f"{stat.estimate:.3g}"
    return f"{stat.estimate:.3g} [{stat.low:.3g}, {stat.high:.3g}]"


def _ratio(values: Sequence[float]) -> Tuple[str, float]:
    """Geometric-mean ratio with its interval, and the estimate itself."""
    from OpenRC.benchmark import ratio_interval

    stat = ratio_interval(values)
    if stat.n < 2:
        return f"{stat.estimate:.2f}", stat.estimate
    return f"{stat.estimate:.2f} [{stat.low:.2f}, {stat.high:.2f}]", stat.estimate


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print(flush=True)


# ---------------------------------------------------------------------------
# oracle queries
# ---------------------------------------------------------------------------
def _query_points(case, axes, n: int, rng) -> np.ndarray:
    """Points at 0.2-1.4 times the LinDistFlow boundary along random rays."""
    import OpenRC as orc
    from OpenRC.geometry.bisection import ray_boundary

    with orc.get_oracle("lindistflow", case) as reference:
        points: List[np.ndarray] = []
        while len(points) < n:
            direction = rng.normal(size=len(axes))
            ray = ray_boundary(reference, axes, direction, tol=1e-3, require_bracket=False)
            unit = direction / np.linalg.norm(direction)
            points.append(rng.uniform(0.2, 1.4) * ray.ratio * unit)
    return np.array(points)


def oracle_section(cases, args, record) -> None:
    import OpenRC as orc
    from OpenRC.powerflow.base import Verdict

    print("## Oracle queries\n")
    print(f"4-D random axes (seed 0), {args.points} points at 0.2-1.4 times the "
          f"LinDistFlow boundary, {args.repeats} passes; ms per query, mean with 95 % "
          f"interval over all queries.  'differs' counts verdicts that differ from the "
          f"first configuration of the family.\n")
    rows: List[List[Any]] = []
    for spec in cases:
        instance = _instance(spec, 4, 0)
        case, axes, _inst_oracle = instance.build()
        _inst_oracle.close()
        points = _query_points(case, axes, args.points, np.random.default_rng(1))
        reference: Dict[str, List[Verdict]] = {}
        for family, label, name, kwargs in ORACLE_CONFIGS:
            oracle = None
            try:
                start = time.perf_counter()
                oracle = orc.get_oracle(name, case, **kwargs)
                # Trigger model build and verify that the solver decides the
                # base point (skips unlicensed or non-functional solver setups).
                if oracle.verdict_at(axes, np.zeros(len(axes))) is Verdict.INCONCLUSIVE:
                    raise RuntimeError("base point undecided: solver unavailable?")
                setup = time.perf_counter() - start
            except Exception as exc:              # a solver that is not installed
                if oracle is not None:
                    oracle.close()
                rows.append([_short(spec), family, label, f"skipped ({type(exc).__name__})",
                             "", "", "", "", ""])
                continue
            try:
                times: List[float] = []
                verdicts: List[Verdict] = []
                for _ in range(args.repeats):
                    verdicts = []
                    for x in points:
                        t0 = time.perf_counter()
                        verdicts.append(oracle.verdict_at(axes, x))
                        times.append(1e3 * (time.perf_counter() - t0))
                base = reference.setdefault(family, verdicts)
                differs = sum(a is not b for a, b in zip(base, verdicts))
                feasible = sum(v is Verdict.FEASIBLE for v in verdicts)
                stats = getattr(oracle, "candidate_stats", None)
                accepted = (f"{stats.accepted}/{stats.attempts}"
                            if stats is not None and stats.attempts else "")
                rows.append([_short(spec), family, label, _ci(times), f"{np.median(times):.3g}",
                             f"{feasible}/{len(points)}", differs, accepted,
                             f"{1e3 * setup:.0f}"])
                record.setdefault("oracle", []).append(dict(
                    case=spec, family=family, config=label, ms=times, setup_ms=1e3 * setup,
                    feasible=feasible, points=len(points), differs=differs,
                    candidate_accepted=getattr(stats, "accepted", None)))
            except orc.SolverUnavailableError as exc:
                # A pre-solver can decide the base point; the missing solver
                # then shows only at the first query it has to decide.
                rows.append([_short(spec), family, label, f"skipped ({type(exc).__name__})",
                             "", "", "", "", ""])
            finally:
                oracle.close()
    _table(["network", "model", "configuration", "ms per query", "median", "feasible",
            "differs", "sweep accepted", "setup ms"], rows)


# ---------------------------------------------------------------------------
# ray searches
# ---------------------------------------------------------------------------
def rays_section(cases, args, record) -> None:
    import OpenRC as orc
    from OpenRC.geometry.bisection import ray_boundary

    print("## Ray searches: bisection against margin (ITP)\n")
    print(f"4-D random axes (seed 0), {args.rays} random rays, tol 1e-4.  Ratios are "
          f"bisection / margin (> 1: margin is faster), geometric mean with 95 % "
          f"interval.\n")
    rows: List[List[Any]] = []
    for spec in cases:
        case, axes, _inst_oracle = _instance(spec, 4, 0).build()
        _inst_oracle.close()
        directions = np.random.default_rng(2).normal(size=(args.rays, len(axes)))
        for name in ("lindistflow", "acpf"):
            with orc.get_oracle(name, case) as oracle:
                runs: Dict[str, List[Tuple[int, float, float]]] = {"bisect": [], "margin": []}
                for d in directions:
                    for search in runs:
                        t0 = time.perf_counter()
                        ray = ray_boundary(oracle, axes, d, search=search)
                        runs[search].append((ray.n_oracle_calls, time.perf_counter() - t0,
                                             ray.ratio))
            calls = [b[0] / m[0] for b, m in zip(runs["bisect"], runs["margin"])]
            secs = [b[1] / m[1] for b, m in zip(runs["bisect"], runs["margin"])]
            gap = max(abs(b[2] - m[2]) for b, m in zip(runs["bisect"], runs["margin"]))
            rows.append([_short(spec), name,
                         f"{np.mean([r[0] for r in runs['bisect']]):.1f}",
                         f"{np.mean([r[0] for r in runs['margin']]):.1f}",
                         _ratio(calls)[0], _ratio(secs)[0], f"{gap:.1e}"])
            record.setdefault("rays", []).append(dict(case=spec, oracle=name, **{
                k: [list(map(float, r)) for r in v] for k, v in runs.items()}))
    _table(["network", "oracle", "calls/ray bisect", "calls/ray margin", "calls ratio",
            "time ratio", "max boundary gap (MW)"], rows)


# ---------------------------------------------------------------------------
# region builds
# ---------------------------------------------------------------------------
def _timed_build(builder, case, axes, oracle, **kwargs):
    with oracle:
        start = time.perf_counter()
        result = builder.build(case, axes, oracle, **kwargs)
        return result, time.perf_counter() - start, oracle.stats.n_calls


def builds_section(cases, args, record) -> None:
    import OpenRC as orc
    from OpenRC.region.expansive import ExpansiveBuilder

    print("## AC region builds\n")
    print(f"3-D random axes, {_seeds(args.seeds)}, 18 vertices, policy 'generation'.  "
          f"Three settings: bisection with a fresh pandapower model per query; margin "
          f"search with a fresh model; margin search with the recycled model (the "
          f"defaults).  Speed-ups are time ratios, geometric mean with 95 % interval.\n")
    settings = [("bisect, fresh", {"search": "bisect"}, {"recycle": False}),
                ("margin, fresh", {}, {"recycle": False}),
                ("margin, recycled", {}, {})]
    rows: List[List[Any]] = []
    for spec in cases:
        seconds: List[List[float]] = []
        calls: List[List[int]] = []
        volumes: List[List[float]] = []
        for seed in range(args.seeds):
            case, axes, _inst_oracle = _instance(spec, 3, seed).build()
            _inst_oracle.close()
            out = [_timed_build(ExpansiveBuilder(policy="generation", seed_rng=seed,
                                                 ray_kwargs=ray),
                                case, axes, orc.get_oracle("acpf", case, **oracle),
                                max_vertices=18)
                   for _, ray, oracle in settings]
            seconds.append([o[1] for o in out])
            calls.append([o[2] for o in out])
            volumes.append([o[0].volume for o in out])
            record.setdefault("builds", []).append(dict(
                case=spec, seed=seed, settings=[x[0] for x in settings],
                seconds=seconds[-1], calls=calls[-1], volume=volumes[-1]))
        t = np.array(seconds)
        gap = max(abs(v[k] / v[0] - 1.0) for v in volumes for k in (1, 2))
        rows.append([_short(spec), _ratio(t[:, 0] / t[:, 2])[0], _ratio(t[:, 0] / t[:, 1])[0],
                     _ratio(t[:, 1] / t[:, 2])[0],
                     " -> ".join(f"{np.mean([c[k] for c in calls]):.0f}" for k in range(3)),
                     f"{100 * gap:.3f} %"])
    _table(["network", "total speed-up", "margin search", "recycling", "oracle calls",
            "max volume change"], rows)


def geometry_section(cases, args, record) -> None:
    from OpenRC.benchmark import BENCHMARK_METHODS
    from OpenRC.geometry.live_hull import geometry_workers, set_geometry_workers

    print("## Geometry threads: 8-D compress-and-regrow\n")
    set_geometry_workers(None)
    threads = geometry_workers()
    print(f"8-D random axes, {_seeds(args.seeds)}, 48 vertices, LinDistFlow.  1 thread "
          f"against {threads}; time ratio 1 / {threads}.\n")
    method = BENCHMARK_METHODS.get("compress-regrow")
    rows: List[List[Any]] = []
    for spec in cases:
        speed, same = [], True
        for seed in range(args.seeds):
            instance = _instance(spec, 8, seed)
            out = []
            for workers in (1, None):
                set_geometry_workers(workers)
                case, axes, oracle = instance.build()
                with oracle:
                    start = time.perf_counter()
                    result = method(case, axes, oracle, max_vertices=48, seed=seed)
                    out.append((time.perf_counter() - start, result.volume, oracle.stats.n_calls))
            set_geometry_workers(None)
            speed.append(out[0][0] / out[1][0])
            same &= out[0][1:] == out[1][1:]
            record.setdefault("geometry", []).append(dict(case=spec, seed=seed, runs=out))
        rows.append([_short(spec), _ratio(speed)[0], "yes" if same else "NO"])
    _table(["network", "speed-up", "identical volume and calls"], rows)


def polish_section(cases, args, record) -> None:
    from OpenRC.region.ray_growth import RayGrowthBuilder

    print("## Batched relocation: 8-D grow-and-polish\n")
    print(f"8-D random axes, {_seeds(args.seeds)}, 48 vertices, LinDistFlow.  Time "
          f"ratio batch 1 / batch 8; volume ratio batch 8 / batch 1.\n")
    rows: List[List[Any]] = []
    for spec in cases:
        speed, volume, calls = [], [], []
        for seed in range(args.seeds):
            instance = _instance(spec, 8, seed)
            out = []
            for batch in (1, 8):
                case, axes, oracle = instance.build()
                result, seconds, n = _timed_build(
                    RayGrowthBuilder(strategy="grow-and-polish", seed_rng=seed, batch=batch),
                    case, axes, oracle, max_vertices=48)
                out.append((seconds, result.volume, n))
            speed.append(out[0][0] / out[1][0])
            volume.append(out[1][1] / out[0][1])
            calls.append((out[0][2], out[1][2]))
            record.setdefault("polish", []).append(dict(case=spec, seed=seed, runs=out))
        rows.append([_short(spec), _ratio(speed)[0], _ratio(volume)[0],
                     f"{min(volume) - 1:+.1%} to {max(volume) - 1:+.1%}",
                     f"{np.mean([c[0] for c in calls]):.0f} / "
                     f"{np.mean([c[1] for c in calls]):.0f}"])
    _table(["network", "speed-up", "volume ratio", "volume range", "oracle calls 1 / 8"], rows)


def certify_section(cases, args, record) -> None:
    import OpenRC as orc
    from OpenRC.region.expansive import ExpansiveBuilder

    print("## Inner-region certificate\n")
    print("The LinDistFlow (Model-I) region over 4-D random axes (seed 0), 40 vertices "
          "- a region any network has, usually refuted somewhere, so the certificate "
          "must settle every limit and confirm witnesses.  Limits thermal, v_min, v_max; "
          "no surface sample.  'global only' disables the relaxation screen (9-bus "
          "network only: on the larger ones single global solves reach the 60 s "
          "time-out).\n")
    rows: List[List[Any]] = []
    for spec in cases:
        case, axes, oracle = _instance(spec, 4, 0).build()
        with oracle:
            region = ExpansiveBuilder(policy="adaptive-ema", seed_rng=0).build(
                case, axes, oracle, max_vertices=40).polytope
        settings: List[Tuple[str, Dict[str, Any]]] = [
            ("screened, 1 worker", {}), ("screened, 8 workers", {"workers": 8})]
        if case.n_bus <= 12:
            settings.append(("global only, 1 worker", {"relaxation_rounds": None}))
        statuses = None
        for label, kwargs in settings:
            try:
                cert = orc.certify_inner_region(region, case, axes, surface_points=0, **kwargs)
            except orc.SolverUnavailableError as exc:
                rows.append([_short(spec), label, f"skipped ({type(exc).__name__})",
                             "", "", "", "", ""])
                continue
            now = [(c.kind, c.element, c.status) for c in cert.checks]
            statuses = statuses or now
            rows.append([_short(spec), label, f"{cert.runtime:.1f}", len(cert.checks),
                         len(cert.violations), len(cert.undecided), cert.status,
                         "yes" if now == statuses else "NO"])
            record.setdefault("certify", []).append(dict(
                case=spec, setting=label, seconds=cert.runtime, limits=len(cert.checks),
                violations=len(cert.violations), status=cert.status))
    _table(["network", "setting", "s", "limits", "violated", "undecided", "status",
            "same statuses"], rows)


RUNNERS = {"oracle": oracle_section, "rays": rays_section, "builds": builds_section,
           "geometry": geometry_section, "polish": polish_section,
           "certify": certify_section}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    parser.add_argument("--points", type=int, default=60, help="query points per network")
    parser.add_argument("--repeats", type=int, default=3, help="passes over the points")
    parser.add_argument("--rays", type=int, default=20, help="rays per network and oracle")
    parser.add_argument("--seeds", type=int, default=3, help="instances per network")
    parser.add_argument("--geometry-cases", nargs="+", default=["9bus", "pp:mv_oberrhein:feeder=1"],
                        help="networks of the 8-D sections")
    parser.add_argument("--certify-cases", nargs="+",
                        default=["9bus", "pp:mv_oberrhein:feeder=1",
                                 "pp:create_cigre_network_mv:load_scale=0.8"])
    parser.add_argument("--out", help="JSON file for the raw measurements")
    args = parser.parse_args(argv)
    logging.getLogger("OpenRC").setLevel(logging.ERROR)
    logging.getLogger("pandapower").setLevel(logging.ERROR)

    import os
    import platform

    print(f"# OpenRC performance ({time.strftime('%Y-%m-%d %H:%M')}, "
          f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores, "
          f"Python {platform.python_version()})\n")
    record: Dict[str, Any] = {}
    total = time.perf_counter()
    for section in args.sections:
        cases = {"geometry": args.geometry_cases, "polish": args.geometry_cases,
                 "certify": args.certify_cases}.get(section, args.cases)
        start = time.perf_counter()
        try:
            RUNNERS[section](cases, args, record)
        except Exception as exc:        # one failed section must not lose the others
            print(f"**{section} failed:** {type(exc).__name__}: {exc}\n")
            record.setdefault("failed", []).append(dict(section=section, error=repr(exc)))
        print(f"_{section}: {time.perf_counter() - start:.0f} s_\n", flush=True)
    print(f"_total: {time.perf_counter() - total:.0f} s_", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(record, handle, default=lambda o: o.name if hasattr(o, "name")
                      else float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
                      indent=1)
    return record


if __name__ == "__main__":
    main()
