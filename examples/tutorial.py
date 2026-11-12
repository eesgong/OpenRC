#!/usr/bin/env python3
"""Executable walkthrough of OpenRC workflows.

    python examples/tutorial.py            # quick start: first region + equation candidate
    python examples/tutorial.py --all      # complete tutorial (several minutes)
    python examples/tutorial.py 2 4 6      # selected, self-contained steps
    python examples/tutorial.py --list     # list the steps without running them

Each step is a self-contained function that prints what it does and why; the
quick start is also a fast check that an installation works.

Sign and unit convention
------------------------
Powers are **MW / Mvar, consumption positive**.  A security-region coordinate is
therefore the *extra* load a point of connection may draw, and a negative
coordinate means export.

Why the ``if __name__ == "__main__"`` guard at the bottom matters
-----------------------------------------------------------------
OpenRC parallelises feasibility sweeps with spawned worker processes, and
``spawn`` re-imports the entry module in every child.  Without the guard, each
worker would re-run the whole tutorial.  Any script of your own that uses
``n_workers`` needs the same guard; when ``__main__`` is not import-safe for
``spawn``, OpenRC logs a warning and falls back to thread workers.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

# Make the package importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import OpenRC as orc  # noqa: E402


# --------------------------------------------------------------------------
def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def step_1_discover() -> None:
    """List registered cases, oracles, builders, policies, compressors, and samplers.

    All pluggable components in OpenRC are exposed through registries; custom
    implementations registered at runtime appear automatically in discovery and CLI
    commands.
    """
    rule("1. Component registries")
    for family, entries in orc.registries().items():
        names = ", ".join(name for name, _ in entries)
        print(f"  {family:<16} {names}")

    print("\n  Ask any registry for details:")
    for name, doc in orc.list_oracles()[:3]:
        print(f"    {name:<20} {doc}")


def step_2_case() -> None:
    """Load a network and choose what the region's axes mean.

    A *case* wraps a network, its base loading, conversion provenance and axis
    presets.  Radial cases support the DistFlow family; meshed or multi-infeed
    cases remain available to AC/DC/OPF.  The *axes* say which injections span
    the region: ``"247"`` is a named preset (the paper's POC
    2, 4, 7), ``[1, 3, 6]`` names buses directly, and ``["1p", "1q"]`` would
    build a mixed active/reactive region at one bus.
    """
    rule("2. A network, and the axes a region is built over")
    case = orc.get_case("9bus")
    print(case.summary())

    axes = case.resolve_axes("247")
    print(f"\n  axes             {axes.labels}  (buses {axes.buses})")
    print(f"  region dimension {axes.dim}")

    topology = case.topology
    print("\n  Electrical distance and branch depth from the slack bus:")
    for axis in axes:
        print(f"    bus {axis.bus}: {topology.electrical_distance(axis.bus):.3f} ohm, "
              f"{topology.depth(axis.bus)} branches from the slack")


def step_3_oracles() -> None:
    """Ask the power-flow models whether an operating point is feasible.

    Every region builder queries power-flow feasibility through the common
    :class:`~OpenRC.powerflow.base.FeasibilityOracle` interface, allowing linear,
    conic, exact, and AC models to be interchanged directly.
    """
    rule("3. Feasibility oracles and three-valued verdicts")
    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")

    points = {"base point": [0, 0, 0], "+1 MW each": [1, 1, 1],
              "+3 MW each": [3, 3, 3], "far outside": [500, 500, 500]}
    names = ["lindistflow", "lindistflow-vdrop", "socp", "acpf"]

    # Reuse each model across points and release every oracle on exit.
    with ExitStack() as stack:
        oracles = [stack.enter_context(orc.get_oracle(name, case)) for name in names]
        print(f"  {'point':<14}" + "".join(f"{n:>20}" for n in names))
        for label, delta in points.items():
            row = ""
            for oracle in oracles:
                try:
                    value = oracle.verdict_at(axes, delta).value
                except orc.SolverUnavailableError:      # socp needs MOSEK/Gurobi
                    value = "no solver"
                row += f"{value:>20}"
            print(f"  {label:<14}{row}")

    print("\n  'inconclusive' distinguishes solver non-convergence from proven")
    print("  infeasibility; step 6 shows how disputed points are escalated.")


def step_4_build() -> None:
    """Grow a region, and see what the seeding strategy is worth.

    The region is grown by repeatedly breaking a facet of the current polytope
    and firing a ray through it.  Two things are pluggable: which facet to pick
    (the policy) and what to start from (the seed).
    """
    rule("4. Building a region")
    from OpenRC.region.expansive import ExpansiveBuilder

    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")

    print(f"  {'policy':<18} {'seed':<9} {'vertices':>9} {'volume':>10} "
          f"{'oracle calls':>13} {'s':>6}")
    results = {}
    for policy, seed in [("volume-gain", "simplex"), ("generation", "simplex"),
                         ("adaptive-ema", "simplex"), ("adaptive-ema", "random")]:
        with orc.get_oracle("lindistflow", case) as oracle:
            kwargs: dict[str, Any] = {"seed_kwargs": {"n_samples": 150}} if seed == "random" else {}
            result = ExpansiveBuilder(policy=policy, seed=seed, seed_rng=0,
                                      **kwargs).build(case, axes, oracle, max_vertices=120)
            results[f"{policy}/{seed}"] = result
            print(f"  {policy:<18} {seed:<9} {result.n_vertices:>9} "
                  f"{result.volume:>10.2f} {oracle.stats.n_calls:>13} {result.seconds:>6.1f}")

    print("\n  'volume-gain' evaluates candidate gains before choosing a facet;")
    print("  compare its volume and query count above.  The 'random' seed tests a")
    print("  batch of points before growing, which costs one oracle call each")
    print("  instead of the ~30 a ray search needs.")


def step_5_compress() -> None:
    """Reduce the vertex count while measuring the volume loss.

    Removing vertices trades region volume for a smaller representation.
    Compare compression methods before choosing a vertex or loss budget for
    downstream optimisation.
    """
    rule("5. Compression")
    from OpenRC.region.expansive import ExpansiveBuilder

    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")
    with orc.get_oracle("lindistflow", case) as oracle:
        region = ExpansiveBuilder(policy="adaptive-ema").build(
            case, axes, oracle, max_vertices=150).polytope
    print(f"  source region: {region.n_vertices} vertices, {region.n_facets} facets, "
          f"volume {region.volume:.2f}")

    targets = [50, 20, 12]
    print(f"\n  {'method':<14}" + "".join(f"{'n=' + str(t):>16}" for t in targets))
    for name, _ in orc.list_compressors():
        cells = ""
        for target in targets:
            result = orc.get_compressor(name).compress(region, target_vertices=target)
            cells += f"{result.volume_loss_pct:>10.3f}% lost"
        print(f"  {name:<14}{cells}")

    print("\n  Or compress to a quality budget instead of a vertex count:")
    result = orc.get_compressor("local-search").compress(
        region, target_vertices=4, max_volume_loss_pct=1.0)
    print(f"    at most 1% loss -> {result.polytope.n_vertices} vertices, "
          f"{result.volume_loss_pct:.3f}% lost")


def step_6_validate() -> None:
    """How much feasible space did the region lose, and does it admit infeasible points?

    The error rate is the share of
    in-region samples that are *proven* infeasible; a conservative region must
    score zero.  The loss rate is the feasible space given up.
    """
    rule("6. Validation - loss rate and error rate")
    from OpenRC.region.expansive import ExpansiveBuilder

    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")
    with orc.get_oracle("lindistflow", case) as oracle:
        region = ExpansiveBuilder(policy="adaptive-ema").build(
            case, axes, oracle, max_vertices=120).polytope

    quality = orc.evaluate_region(region, case, axes, reference="acpf",
                                  n_samples=400, rng=1)
    print(quality.summary())

    print("\n  Only unresolved points are excluded.  A point the AC solver")
    print("  could not converge on is escalated to a global solve of the exact")
    print("  equations; whatever survives that is reported as unresolved and left")
    print("  out of the error rate, because neither outcome was demonstrated.")


def step_7_conservative() -> None:
    """Compare the two conservative constructions and Method 1 scopes.

    Both intersect two convex regions.  Method 1 shifts by the losses of the
    complete path to the slack (its default scope); the direct-feeder scope
    charges only the branch feeding each POC and carries no general guarantee.
    Method 2 intersects the published complementary tight models directly.
    """
    rule("7. Conservative construction - two methods")
    from OpenRC.region.conservative import CONSERVATIVE_METHODS, ConservativeBuilder
    from OpenRC.region.expansive import ExpansiveBuilder

    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")
    inner = ExpansiveBuilder(policy="adaptive-ema")

    print("  registered methods:")
    for method, (a, b, shift) in CONSERVATIVE_METHODS.items():
        print(f"    {method:<18} {a} + {b}"
              f"{'  (then shift)' if shift else '  (direct intersection)'}")

    with orc.get_oracle("lindistflow", case) as oracle:
        base = inner.build(case, axes, oracle, max_vertices=120).polytope
    regions = {"model1 (no guarantee)": base}
    recipes: list[tuple[str, dict[str, Any]]] = [
        ("M1 intersect-shift", dict(method="intersect-shift")),
        ("M1 direct-feeder scope",
         dict(method="intersect-shift", shift_scope="feeding_branch")),
        ("M2 tight-intersect", dict(method="tight-intersect")),
    ]
    for label, kwargs in recipes:
        regions[label] = ConservativeBuilder(inner_builder=inner, **kwargs).build(
            case, axes, max_vertices=120).polytope

    # One common sampling box keeps the loss rates strictly comparable.
    lower, upper = base.bounding_box(2.0)
    sampler = orc.get_sampler("uniform-box", lower=lower, upper=upper)

    print(f"\n  {'construction':<24} {'volume':>9} {'error %':>9} "
          f"{'95% CI (error %)':>18} {'loss %':>8}  sample evidence")
    for label, region in regions.items():
        quality = orc.evaluate_region(region, case, axes, reference="acpf",
                                      n_samples=1200, sampler=sampler, rng=1)
        # A finite sample can refute a region, but cannot certify the whole body.
        if quality.n_confirmed_violations:
            verdict = "refuted by samples"
        elif quality.n_unresolved_in_region or not quality.n_decided_in_region:
            verdict = "undecided"
        else:
            verdict = "no sampled violations"
        low, high = quality.error_rate_ci
        interval = f"[{100 * low:.2f}, {100 * high:.2f}]"
        print(f"  {label:<24} {region.volume:>9.2f} {100 * quality.error_rate:>9.3f} "
              f"{interval:>18} {100 * quality.loss_rate:>8.3f}  {verdict}")

    print("\n  When few samples fall inside a small region, the point estimate has")
    print("  high variance; inspect the 95% Wilson confidence interval alongside it.")
    print("  No sampled violations is not an inner-region certificate; see step 13.")


def step_8_apply() -> None:
    """Use the region: a grid-aware problem with no power flow in it.

    Once the feasible set is ``A x <= b``, capacity allocation is a compact
    linear program with one row per facet (no exponential corner enumeration).
    Compression therefore reduces both storage and every downstream solve.
    """
    rule("8. Applications - the region replaces the power flow")
    from OpenRC.region.expansive import ExpansiveBuilder

    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")
    with orc.get_oracle("lindistflow", case) as oracle:
        region = ExpansiveBuilder(policy="adaptive-ema").build(
            case, axes, oracle, max_vertices=150).polytope

    rng = np.random.default_rng(0)
    loading = rng.uniform(0.8, 1.0, size=(300, 3))     # per-scenario loading ratios
    shifts = rng.uniform(-8, 8, size=(300, 3))         # congested operating points

    print(f"  {'region':<18} {'vertices':>9} {'facets':>7} "
          f"{'capacity obj':>13} {'ms':>6} {'storage obj':>12} {'ms':>6}")
    variants = [("original", region)]
    for target in (50, 15):
        variants.append((f"compressed n={target}",
                         orc.get_compressor("local-search").compress(
                             region, target_vertices=target).polytope))
    for label, body in variants:
        cap = orc.get_application("capacity-allocation").solve_batch(body, loading)
        sto = orc.get_application("storage-sizing").solve_batch(body, shifts)
        print(f"  {label:<18} {body.n_vertices:>9} {body.n_facets:>7} "
              f"{cap.mean_objective:>13.4f} {cap.mean_ms:>6.2f} "
              f"{sto.mean_objective:>12.4f} {sto.mean_ms:>6.2f}")

    print("\n  Fewer facets reduce the number of linear constraints; compare the")
    print("  solve times and objective changes before choosing a compression target.")


def step_9_resources() -> None:
    """Inspect hull memory scaling and worker-pool calibration.

    Two resources decide whether a region computation finishes: memory, because
    a high-dimensional hull outgrows RAM long before it is accurate, and time,
    because power-flow queries and repeated hull updates can be expensive.
    """
    rule("9. Memory and parallelism")
    from OpenRC.parallel import plan_workers, resolve_workers, spawn_is_safe
    from OpenRC.resources import (
        hull_memory_mb,
        measure_polytope_memory,
        memory_report,
    )

    print("  memory right now:", {k: (round(v, 1) if isinstance(v, float) else v)
                                  for k, v in memory_report().items()})

    print("\n  In 4+ dimensions, hull memory scales with facet count rather than")
    print("  vertex count, so OpenRC measures live facet storage directly:")
    rng = np.random.default_rng(0)
    for dim in (3, 8):
        for n in (40, 80):
            points = rng.normal(size=(n, dim))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            measured = measure_polytope_memory(orc.Polytope(points))
            print(f"    {dim}-D, {n:3d} vertices -> {measured['n_facets']:7d} facets, "
                  f"{measured['total_mb']:7.3f} MiB "
                  f"(model {hull_memory_mb(n, dim, measured['n_facets']):.3f})")

    print(f"\n  cores available: {resolve_workers('auto')}   "
          f"spawn usable here: {spawn_is_safe()}")
    print("  'auto' starts a worker pool only when total query cost exceeds start-up:")
    for n, ms in [(400, 1.0), (400, 20.0), (5000, 20.0)]:
        chosen = plan_workers(n, ms / 1000.0, "auto", startup_seconds=5.0)
        print(f"    {n:>5} points x {ms:>4.0f} ms -> "
              f"{'sequential' if chosen == 1 else f'{chosen} workers'}")

    print("\n  Pass n_workers to any batched entry point:")
    print("    orc.evaluate_region(..., n_workers='auto')")
    print("    ExpansiveBuilder(seed='random', n_workers='auto')")
    print("  Results are identical to a sequential run, whatever the worker count.")


def step_10_memory_aware() -> None:
    """Compute a high-dimensional region that would not otherwise fit.

    Expansion and compression alternate on one live hull.  The published rule
    compresses at a fixed vertex count; the adaptive schedule is given the
    memory the hull may use and decides when to compress, how far, and when to
    stop.
    """
    rule("10. Memory-aware computation of an 8-D region")
    from OpenRC.region.expansive import ExpansiveBuilder
    from OpenRC.region.memory_aware import MemoryAwareBuilder

    case = orc.get_case("9bus-scaled")
    axes = case.resolve_axes("all8")
    runs: dict[str, dict[str, Any]] = {
        "vertex budget 50 -> 25": dict(max_vertices=50, target_vertices=25, rounds=3),
        "adaptive, 2 MiB hull": dict(schedule="adaptive",
                                      schedule_kwargs={"budget_mb": 2.0},
                                      max_vertices=10_000, rounds=20),
    }
    for label, options in runs.items():
        with orc.get_oracle("lindistflow", case) as oracle:
            result = MemoryAwareBuilder(
                inner_builder=ExpansiveBuilder(policy="adaptive-ema"), compressor="auto",
                **options).build(case, axes, oracle, memory_limit_mb=2000)
            print(f"\n  {label}: best volume {result.volume:.6g} in round "
                  f"{result.builder_info['best_round']}, {oracle.stats.n_calls} oracle calls")
        print(f"  {'round':>6} {'phase':<10} {'vertices':>9} {'hull MiB':>9} {'volume':>12}")
        for cycle in result.builder_info["cycles"]:
            print(f"  {cycle['round']:>6} {cycle['phase']:<10} {cycle['n_vertices']:>9} "
                  f"{cycle['hull_mb']:>9.2f} {cycle['volume']:>12.6g}")
        print(f"  stopped: {result.stopped_because}")
    print("\n  Only the vertex set is returned; the hull that carried the whole run is")
    print("  released.  See scripts/benchmark_memory.py for the 8-D and 16-D comparison.")


def step_11_extend() -> None:
    """Add your own method through a registry."""
    rule("11. Extending OpenRC")
    from OpenRC.compression.base import COMPRESSORS, CompressionResult, Compressor

    @COMPRESSORS.register("tutorial-drop-last",
                          doc="Toy: drop whichever vertex comes last in the array.")
    class DropLastCompressor(Compressor):
        """Minimal example compressor retaining the first ``target_vertices`` rows."""

        name = "tutorial-drop-last"

        def _run(self, polytope, *, target_vertices, max_volume_loss_pct):
            vertices = polytope.vertices[:target_vertices]
            return CompressionResult(
                polytope=orc.Polytope(vertices, axes=polytope.axes),
                original_volume=polytope.volume,
                original_n_vertices=polytope.n_vertices,
            )

    print("  registered compressor 'tutorial-drop-last'; active compressors:")
    print(f"    {[n for n, _ in orc.list_compressors()]}")

    rng = np.random.default_rng(0)
    points = rng.normal(size=(80, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    region = orc.Polytope(points).reduce_to_hull()
    for name in ("local-search", "tutorial-drop-last"):
        result = orc.get_compressor(name).compress(region, target_vertices=15)
        print(f"    {name:<22} {result.volume_loss_pct:6.2f}% volume lost")
    print("\n  The same pattern applies to CASES, ORACLES, BUILDERS, SEEDS,")
    print("  FACET_POLICIES, SAMPLERS and APPLICATIONS.")


def step_12_pandapower() -> None:
    """Use any pandapower network - your own model, or any of the built-in ones.

    Everything so far ran on OpenRC's curated cases.  The same pipeline takes any
    pandapowerNet: the adapter works out the infeed, applies switch states,
    collapses a substation transformer so the DistFlow models apply, and records
    every modelling decision it made in ``case.metadata["conversion"]``.
    """
    rule("12. Any pandapower network")
    import pandapower as pp

    from OpenRC.cases.pp_networks import _oracle_families
    from OpenRC.powerflow.opf import opf_ready
    from OpenRC.region.expansive import ExpansiveBuilder

    # --- 1. the whole pandapower library is available by name ---------------
    library = orc.list_pandapower_networks()
    print(f"  {len(library)} networks from pandapower.networks, as 'pp:<name>', e.g.")
    for name in ("pp:case33bw", "pp:create_kerber_dorfnetz", "pp:case118"):
        case = orc.get_case(name)
        family = _oracle_families(case.is_radial, opf_ready(case.net))
        print(f"    {name:<28} {case.n_bus:>4} buses  -> {family}")

    # --- 2. the IEEE 33-bus feeder, straight from pandapower ----------------
    case = orc.get_case("pp:case33bw")
    axes = case.resolve_axes([17, 24, 32])
    with orc.get_oracle("lindistflow", case) as oracle:
        region = ExpansiveBuilder(policy="generation", seed_rng=0).build(
            case, axes, oracle, max_vertices=30).polytope
    print(f"\n  IEEE 33 (Baran-Wu): {region.n_vertices} vertices, "
          f"volume {region.volume:.2f} MW^3")
    print(f"    note: {case.metadata['conversion']['n_unrated']} lines carry MATPOWER's "
          "placeholder rating; Model-II and Method 2 need")
    print("    real ones:  orc.get_case('pp:case33bw', default_rating_ka=0.4)")

    # --- 3. a network you built yourself, with your own bus IDs -------------
    net = pp.create_empty_network()
    for bus_id, kv in ((100, 20.0), (200, 0.4), (300, 0.4), (400, 0.4), (500, 0.4)):
        pp.create_bus(net, vn_kv=kv, index=bus_id)
    pp.create_ext_grid(net, bus=100)
    pp.create_transformer(net, hv_bus=100, lv_bus=200, std_type="0.4 MVA 20/0.4 kV")
    for u, v in ((200, 300), (300, 400), (300, 500)):
        pp.create_line(net, from_bus=u, to_bus=v, length_km=0.2, std_type="NAYY 4x150 SE")
    for b in (300, 400, 500):
        pp.create_load(net, bus=b, p_mw=0.02, q_mvar=0.005)

    mine = orc.case_from_pandapower(net, name="my-feeder",
                                    axis_presets={"poc": [300, 400, 500]})
    conversion = mine.metadata["conversion"]
    print(f"\n  your own network: {conversion['n_trafo_collapsed']} transformer "
          f"collapsed to {conversion['reference_kv']:g} kV, root found by "
          f"'{mine.slack_source}'")
    print(f"    your bus IDs [300, 400, 500] -> OpenRC buses {mine.axis_presets['poc']}")
    with orc.get_oracle("lindistflow", mine) as oracle:
        region = ExpansiveBuilder(policy="generation").build(
            mine, mine.resolve_axes("poc"), oracle, max_vertices=20).polytope
    print(f"    region: {region.n_vertices} vertices, volume {region.volume:.5f} MW^3")

    # --- 4. a meshed network is still computable, just not with DistFlow ----
    meshed = orc.get_case("pp:case9")
    print(f"\n  pp:case9 is meshed: {meshed.radiality.n_loops} loop(s). DistFlow refuses")
    print("  it; the OPF oracles compute its region with no radiality requirement:")
    with orc.get_oracle("dcopf", meshed) as oracle:
        region = ExpansiveBuilder(policy="generation", seed="axis").build(
            meshed, meshed.resolve_axes([4, 6, 8]), oracle, max_vertices=15).polytope
    print(f"    dcopf region: {region.n_vertices} vertices, volume {region.volume:.4g} MW^3")


def step_13_certify() -> None:
    """Bound every limit globally, then prove or sample existence.

    Whereas finite sampling tests individual draws, global certification solves,
    for every thermal and voltage limit, the worst case over the *whole* region
    with the exact branch-flow equations (Gurobi spatial branch-and-bound).  Existence of a solution is proven at every point by a
    contraction condition where it holds; otherwise it is checked at points
    spread over the region's surface, and the conclusion assumes a continuous
    high-voltage solution with no fold in the voltage window.  Monte Carlo in a
    large cube is the cheaper, statistical alternative.  Both decide each point
    by escalating: AC -> exact (Gurobi) -> the method's sub-models (MOSEK) ->
    worst overrun, recorded as computationally infeasible.
    """
    rule("13. Verification - certificate, or Monte Carlo")
    from OpenRC.region.conservative import ConservativeBuilder
    from OpenRC.region.expansive import ExpansiveBuilder

    case = orc.get_case("9bus")
    axes = case.resolve_axes([2, 4, 7, 0])             # a 4-D region
    inner = ExpansiveBuilder(policy="adaptive-ema", seed_rng=0)
    with orc.get_oracle("lindistflow", case) as oracle:
        linear = inner.build(case, axes, oracle, max_vertices=80).polytope
    regions = {
        "Model-I (linear)": linear,
        "M1 intersect-shift": ConservativeBuilder(inner_builder=inner).build(
            case, axes, max_vertices=80).polytope,
        "M2 tight-intersect": ConservativeBuilder(
            method="tight-intersect", inner_builder=inner).build(
            case, axes, max_vertices=80).polytope,
    }

    print(f"  {'construction':<20} {'Monte Carlo':<12} {'certificate':<12} "
          f"{'worst loading':>14}")
    for label, region in regions.items():
        mc = orc.verify_inner_region(region, case, axes, mode="monte-carlo",
                                     n_points=5000, min_inside=100)
        try:
            surface_report = orc.verify_inner_region(
                region, case, axes, mode="surface", n_points=8
            )
        except orc.SolverUnavailableError:              # the certificate needs Gurobi
            print(f"  {label:<20} {mc.status:<12} {'no solver':<12}")
            continue
        assert surface_report.certificate is not None
        worst = surface_report.certificate.worst_loading
        worst_pct = 100.0 * worst.bound if worst is not None else float("nan")
        print(f"  {label:<20} {mc.status:<12} {surface_report.status:<12} "
              f"{worst_pct:>13.3f}%")
        for check in surface_report.certificate.violations[:1]:
            where = (np.round(check.witness, 2).tolist()
                     if check.witness is not None else "?")
            print(f"      witness: {check.name} at {100 * check.incumbent:.2f}% for "
                  f"{where} ({check.note})")

    print("\n  Unlike finite sampling, the certificate computes global worst-case")
    print("  limit bounds and hands back a violating witness when one exists;")
    print("  existence is proven by contraction where that condition holds, and rests")
    print("  on the no-fold assumption elsewhere.  A small contraction can make the")
    print("  linear region pass it:")
    try:
        shrunk, cert, lam = orc.shrink_to_certified(regions["Model-I (linear)"], case, axes,
                                                    surface_points=8)
    except orc.SolverUnavailableError as exc:
        print(f"    skipped: {exc}")
        return
    shrunk_worst = 100.0 * cert.worst_loading.bound if cert.worst_loading is not None else float("nan")
    print(f"    lam = {lam:.4f}  ->  {cert.status}, worst loading "
          f"{shrunk_worst:.3f}%")


def step_14_equations() -> None:
    """Try the optional equation layer: a local candidate, the solver as fallback.

    A local method may prove feasibility by returning a candidate that passes
    every constraint.  Failure proves nothing: a solver-backed oracle then
    falls back to its optimiser.  DistFlow's default analytic backend is faster
    still, because these tree equations have a closed form.
    """
    rule("14. Equation candidates - fast proposal, strict acceptance")
    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")

    from OpenRC.powerflow.distflow import DistFlowOracle

    # The base point is accepted by the candidate, so the fallback optimiser
    # is never constructed and this example needs no solver licence.
    with cast(
        DistFlowOracle,
        orc.get_oracle("lindistflow", case, backend="solver", candidate_solver="newton"),
    ) as oracle:
        verdict = oracle.verdict_at(axes, np.zeros(3))
        print(f"  verified base-point candidate: {verdict.value}")
        print(f"  candidate counters: {oracle.candidate_stats.as_dict()}")

        # The public problem follows SciPy's fun/jac/x0/bounds convention.  OpenRC
        # separately checks bounds and cones after the equations converge.
        p_mw, q_mvar = case.injection_vector(axes, np.zeros(3))
        problem = oracle.equation_problem(p_mw, q_mvar)
        result = orc.solve_equations(problem, method="gauss-seidel")
        maximum = result.residuals.maximum if result.residuals is not None else float("inf")
        print(f"  Gauss-Seidel: accepted={result.accepted}, "
              f"iterations={result.iterations}, max violation="
              f"{maximum:.2e}")
        prepared = orc.PreparedEquationSolver(oracle.conic_form(), method="newton")

    # The exact radial model has a structured alternative.  Its public
    # certificate exposes every number in the local Banach proof; it can prove
    # feasibility, but a failed certificate still has to fall back.
    sweep = orc.RadialSweep(case)
    swept = sweep.solve(p_mw, q_mvar)
    certificate = sweep.last_certificate
    assert certificate is not None
    print(f"  certified sweep: accepted={swept.accepted}, "
          f"L={certificate.lipschitz:.3g}, "
          f"error<={certificate.error_bound:.2e} pu")

    outside_p, outside_q = case.injection_vector(axes, np.full(3, 100.0))
    rejected = prepared.solve(outside_p, outside_q)
    print(f"  far point: equations converged={rejected.converged}, "
          f"accepted={rejected.accepted}, fallback required={rejected.needs_fallback}")
    print("  A rejected root is never called infeasible; the configured optimiser")
    print("  must decide it.  See README 'Optional equation candidates'.")


STEPS = [
    step_1_discover, step_2_case, step_3_oracles, step_4_build, step_5_compress,
    step_6_validate, step_7_conservative, step_8_apply, step_9_resources,
    step_10_memory_aware, step_11_extend, step_12_pandapower, step_13_certify,
    step_14_equations,
]

# The quick start builds a real region without the heavier validation,
# conservative, application and 8-D steps; step 14 is short and shows the
# equation layer.
QUICK_STEPS = (1, 2, 3, 4, 14)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", type=int, help="step numbers to run")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    parser.add_argument("--all", action="store_true",
                        help="run all steps (the default runs the quick-start subset)")
    parser.add_argument("--verbose", action="store_true", help="show OpenRC's log")
    args = parser.parse_args()

    if args.list:
        for i, step in enumerate(STEPS, 1):
            summary = (step.__doc__ or "").strip().split("\n")[0]
            print(f"  {i:>2}. {summary}")
        return

    if args.all and args.steps:
        parser.error("choose either explicit step numbers or --all")

    orc.configure_logging(logging.INFO if args.verbose else logging.WARNING)
    chosen = (range(1, len(STEPS) + 1) if args.all else
              (args.steps if args.steps else QUICK_STEPS))
    mode = "complete tutorial" if args.all else ("selected steps" if args.steps else "quick start")
    print(f"OpenRC tutorial: {mode}; running steps {list(chosen)}")

    started = time.perf_counter()
    for number in chosen:
        if not 1 <= number <= len(STEPS):
            print(f"no step {number}; there are {len(STEPS)}")
            continue
        STEPS[number - 1]()
    print(f"\n{'=' * 78}\nfinished in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
