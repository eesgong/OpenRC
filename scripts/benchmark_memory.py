"""Benchmark: growing a region under a memory budget, with and without compression.

Every configuration grows the same region with the same oracle, facet policy
and live-hull memory budget, and differs only in what happens at the budget:

``plain``           stop there (no compression)
``vertex-budget``   the published rule, compress M -> M_hat whenever M is hit
``adaptive``        the adaptive schedule without facet screening
``adaptive+debt``   the adaptive schedule, resumed expansions screened by facet
                    debt (count weighting) - an optional heuristic
``adaptive+debt-cone``  the same with cone weighting
``adaptive+relocation``  ``adaptive+debt`` plus gradient relocation of the
                    vertices after every expansion (fixed vertex count)
``ray-growth``      single-pass expansion without compression: ``grow-and-polish``
                    grows with the facet queue until the memory cap refuses
                    another vertex, then relocates vertices at that size

The expansions use the builder's default seed (``"auto"``: the axis seed from
4-D up).

Usage::

    python scripts/benchmark_memory.py --case 9bus --axes all8 --budget 8
    python scripts/benchmark_memory.py --case 9bus --axes pq16 --budget 80 --oracle acpf

``--axes pq16`` means P and Q at every non-slack bus.  Results go to stdout as
a Markdown table and, with ``--out``, to a CSV file.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make the package importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import OpenRC as orc
from OpenRC.region.expansive import ExpansiveBuilder
from OpenRC.region.memory_aware import MemoryAwareBuilder
from OpenRC.region.ray_growth import RayGrowthBuilder


def configurations(budget: float, max_vertices: int, target_vertices: int):
    adaptive = dict(schedule="adaptive", schedule_kwargs={"budget_mb": budget},
                    max_vertices=100_000, target_vertices=10, rounds=40)
    return {
        "plain": dict(schedule="memory-budget", schedule_kwargs={"budget_mb": budget},
                      max_vertices=100_000, target_vertices=10, rounds=0),
        "vertex-budget": dict(max_vertices=max_vertices, target_vertices=target_vertices,
                              rounds=5),
        "adaptive": dict(adaptive, facet_screen=None),
        "adaptive+debt": dict(adaptive, facet_screen="debt"),
        "adaptive+debt-cone": dict(adaptive, facet_screen="debt",
                                   screen_kwargs={"weighting": "cone"}),
        "adaptive+relocation": dict(adaptive, facet_screen="debt", refiner="relocation"),
        "ray-growth": {"ray_growth": True},
    }


def main(argv=None) -> list:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="9bus")
    parser.add_argument("--axes", default="all8")
    parser.add_argument("--oracle", default="acpf")
    parser.add_argument("--policy", default="adaptive-ema")
    parser.add_argument("--budget", type=float, default=8.0, help="live-hull MiB")
    parser.add_argument("--max-vertices", type=int, default=60,
                        help="M of the vertex-budget rule")
    parser.add_argument("--target-vertices", type=int, default=45,
                        help="M_hat of the vertex-budget rule")
    parser.add_argument("--compressor", default="auto")
    parser.add_argument("--only", nargs="*", help="run only these configurations")
    parser.add_argument("--ray-moves", type=int, default=400,
                        help="move budget of the ray-growth configuration")
    parser.add_argument("--out", help="CSV file for the results")
    args = parser.parse_args(argv)
    logging.getLogger("OpenRC").setLevel(logging.WARNING)

    case = orc.get_case(args.case)
    if args.axes == "pq16":
        buses = case.non_slack_buses
        axes = case.resolve_axes([f"{b}p" for b in buses] + [f"{b}q" for b in buses])
    else:
        axes = case.resolve_axes(args.axes)

    rows = []
    for name, options in configurations(args.budget, args.max_vertices,
                                        args.target_vertices).items():
        if args.only and name not in args.only:
            continue
        with orc.get_oracle(args.oracle, case) as oracle:
            start = time.perf_counter()
            if options.get("ray_growth"):
                ray = RayGrowthBuilder(strategy="grow-and-polish",
                                       strategy_kwargs={"policy": args.policy}).build(
                    case, axes, oracle, max_vertices=100_000, max_hull_mb=args.budget,
                    max_moves=args.ray_moves)
                if ray.trace:
                    peak_hull_mb = max(r["hull_mb"] for r in ray.trace)
                else:
                    # A zero-move run still builds its seed. Measure that geometry
                    # only on this path; ordinary runs keep their recorded peak.
                    from OpenRC.resources import measure_polytope_memory

                    peak_hull_mb = measure_polytope_memory(ray.polytope)["total_mb"]
                rows.append({
                    "configuration": name, "volume": ray.volume,
                    "vertices": ray.n_vertices, "rounds": 1, "best_round": 0,
                    "oracle_calls": oracle.stats.n_calls,
                    "peak_hull_mb": peak_hull_mb,
                    "seconds": time.perf_counter() - start, "excluded": 0,
                    "stopped": ray.stopped_because,
                })
                print(f"  done: {name}", flush=True)
                continue
            builder = MemoryAwareBuilder(inner_builder=ExpansiveBuilder(policy=args.policy),
                                         compressor=args.compressor, **options)
            result = builder.build(case, axes, oracle)
            info = result.builder_info
            cycles = info["cycles"]
            rows.append({
                "configuration": name,
                "volume": result.volume,
                "vertices": result.n_vertices,
                "rounds": sum(c["phase"] == "expand" for c in cycles),
                "best_round": info["best_round"],
                "oracle_calls": oracle.stats.n_calls,
                "peak_hull_mb": max(c["hull_mb"] for c in cycles),
                "seconds": time.perf_counter() - start,
                "excluded": (info.get("facet_screen") or {}).get("excluded", 0),
                "stopped": result.stopped_because,
            })
            print(f"  done: {name}", flush=True)

    base = next((r["volume"] for r in rows if r["configuration"] == "plain"), None)
    print(f"\n{args.case}, {len(axes)}-D ({args.axes}), oracle {args.oracle}, policy "
          f"{args.policy}, live-hull budget {args.budget:g} MiB\n")
    print("| configuration | volume | vs plain | vertices | rounds | oracle calls | "
          "peak hull MiB | s | excluded |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rel = f"{100 * (r['volume'] / base - 1):+.1f}%" if base else "-"
        print(f"| {r['configuration']} | {r['volume']:.6g} | {rel} | {r['vertices']} | "
              f"{r['rounds']} | {r['oracle_calls']} | {r['peak_hull_mb']:.1f} | "
              f"{r['seconds']:.0f} | {r['excluded']} |")
    for r in rows:
        print(f"  {r['configuration']}: {r['stopped']}")
    if args.out:
        import pandas as pd

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    return rows


if __name__ == "__main__":
    main()
