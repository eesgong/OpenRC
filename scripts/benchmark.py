"""Benchmark growth methods over networks x dimensions x seeds, with intervals.

Each seed draws a different set of axes (``OpenRC.benchmark.random_instances``),
every method grows every instance with the same vertex budget per dimension,
and the summary reports each method's paired volume ratio to a baseline: the
geometric mean over instances, its confidence interval, and the win rate.

Usage::

    python scripts/benchmark.py --cases 9bus pp:mv_oberrhein:feeder=1 --dims 3 5 8 --seeds 8 \\
        --methods expansive facet-queue grow-and-polish --baseline expansive
    python scripts/benchmark.py --list

A case may carry keyword arguments as ``name:key=value``, for example
``pp:create_cigre_network_mv:load_scale=0.8``.  Results go to stdout as
Markdown tables and, with ``--out``, every run to a CSV file.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make the package importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    """``name[:key=value[,key=value]]`` -> (name, kwargs)."""
    name, sep, options = spec.rpartition(":")
    if not sep or "=" not in options:        # no options (``pp:`` is part of the name)
        if not spec.strip():
            raise ValueError(f"case specification must be non-empty, got {spec!r}")
        return spec, {}
    if not name.strip():
        raise ValueError(f"case specification must include a case name, got {spec!r}")
    kwargs: dict = {}
    for item in options.split(","):
        if "=" not in item:
            raise ValueError(f"expected key=value in case specification {spec!r}, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"expected key=value in case specification {spec!r}, got {item!r}")
        kwargs[key] = _parse_scalar(value)
    return name, kwargs


def main(argv=None):
    from OpenRC.benchmark import (list_benchmark_methods, random_instances, run_benchmark,
                                  summarize)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=["9bus"])
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--seeds", type=int, default=5, help="instances per case and dimension")
    parser.add_argument("--methods", nargs="+", default=["expansive", "grow-and-polish"])
    parser.add_argument("--baseline", default=None,
                        help="method the ratios are taken against (default: the first)")
    parser.add_argument("--vertices-per-dim", type=float, default=6.0)
    parser.add_argument("--oracle", default="lindistflow")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=1, help="processes running the grid")
    parser.add_argument("--out", help="CSV file for every run")
    parser.add_argument("--list", action="store_true", help="list the example methods")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for name, doc in list_benchmark_methods():
            print(f"{name:28s} {doc}")
        return None
    logging.getLogger("OpenRC").setLevel(logging.WARNING)
    if args.baseline is not None and args.baseline not in args.methods:
        parser.error(f"--baseline {args.baseline!r} must be one of --methods")

    instances = []
    for spec in args.cases:
        name, kwargs = _parse_case(spec)
        instances += random_instances(name, args.dims, range(args.seeds), oracle=args.oracle,
                                      case_kwargs=kwargs)
    start = time.perf_counter()
    runs = run_benchmark(args.methods, instances, vertices_per_dim=args.vertices_per_dim,
                         workers=args.workers,
                         progress=None if args.quiet else
                         (lambda msg: print("  " + msg, file=sys.stderr, flush=True)))
    if args.out:
        runs.to_csv(args.out, index=False)
    baseline = args.baseline or args.methods[0]
    summary = summarize(runs, baseline=baseline, confidence=args.confidence)
    level = f"{100 * args.confidence:g} %"
    print(f"\n{len(instances)} instances x {len(args.methods)} methods, "
          f"{args.vertices_per_dim:g} vertices per dimension, oracle {args.oracle}, "
          f"{time.perf_counter() - start:.0f} s\n")
    print(f"| case | d | method | volume / {baseline} (geo. mean) | {level} interval "
          f"| wins | ties | excluded | oracle calls | s |")
    print("|---|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for row in summary.itertuples():
        print(f"| {row.case} | {row.dim} | {row.method} | {row.ratio:.3f} | "
              f"[{row.ratio_low:.3f}, {row.ratio_high:.3f}] | {row.win_rate:.0%} | "
              f"{row.tie_rate:.0%} | {row.ratio_excluded} | "
              f"{row.calls:.0f} | {row.seconds:.1f} |")
    failed = runs[runs["error"] != ""]
    if len(failed):
        print(f"\n{len(failed)} run(s) failed:")
        for row in failed.itertuples():
            print(f"  {row.case} {row.dim}-D seed {row.seed} {row.method}: {row.error}")
    return summary


if __name__ == "__main__":
    main()
