#!/usr/bin/env python3
"""OpenRC command-line driver.

Command-line interface for case inspection, region construction, compression,
validation, certification, and downstream applications.  Example usage::

    python scripts/demo.py list
    python scripts/demo.py case --case 9bus

    # build a region and save it
    python scripts/demo.py region --case 9bus --axes 247 --max-vertices 200 \\
                                  --policy adaptive-ema --seed random --out out/

    # the two conservative constructions, side by side
    python scripts/demo.py conservative --case 9bus --axes 247 --validate 800

    # benchmark compression methods across target vertex counts
    python scripts/demo.py bench-compress --region out/9bus_b1p-b3p-b6p.csv \\
                                          --targets 50,15

    # compare facet-selection policies
    python scripts/demo.py bench-policy --case 9bus --axes 247 --max-vertices 200

    # an 8-D region under a memory budget
    python scripts/demo.py memory --case 9bus-scaled --axes all8 \\
                                  --max-vertices 150 --target-vertices 75 --rounds 4

    # a guided end-to-end tour (a few minutes)
    python scripts/demo.py tour --case 9bus --axes 247

Axes accept either a preset name (``247``) or an explicit list (``1,3,6``, or
``1p,1q,3p`` for a mixed active/reactive region).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Make the package importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import OpenRC as orc  # noqa: E402
from OpenRC.region.conservative import (  # noqa: E402
    ConservativeBuilder,
    EmptyComponentRegionError,
    InfeasibleComponentOriginError,
)
from OpenRC.region.expansive import ExpansiveBuilder  # noqa: E402
from OpenRC.region.memory_aware import MemoryAwareBuilder  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 8)}")


def _parse_workers(value):
    """Argparse converter accepting a positive integer, ``auto`` or ``all``."""
    if isinstance(value, str) and value.strip() in ("auto", "all"):
        return value.strip()
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer, auto or all") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer, auto or all")
    return number


def _positive_int(value):
    """Argparse converter for a count that must be at least one."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _parse_slack_vm_pu(value: str):
    """Argparse converter for --slack-vm-pu: 'setpoint' or a positive finite float."""
    if value.strip().lower() == "setpoint":
        return "setpoint"
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "slack-vm-pu must be 'setpoint' or a positive number"
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError(
            "slack-vm-pu must be 'setpoint' or a positive number"
        )
    return number


def _case_overrides(args) -> dict:
    """Keyword overrides for the case factory, from the optional CLI flags."""
    out = {}
    for flag, key in (("feeder", "feeder"), ("default_rating_ka", "default_rating_ka"),
                      ("load_scale", "load_scale"),
                      ("transformer_handling", "transformer_handling"),
                      ("slack_vm_pu", "slack_vm_pu"),
                      ("line_charging", "line_charging")):
        value = getattr(args, flag, None)
        if value is not None:
            out[key] = value
    return out


def _load_case(args):
    """A registered case, or any pandapower network saved as JSON.

    ``--case`` accepts a registered case name (e.g. ``9bus`` or ``pp:case33bw``),
    and ``--case-file`` loads a network serialized with ``pandapower.to_json``.
    """
    overrides = _case_overrides(args)
    path = getattr(args, "case_file", None)
    if path:
        import pandapower as pp

        net = pp.from_json(path)
        return orc.case_from_pandapower(net, name=Path(path).stem, **overrides)
    return orc.get_case(args.case, **overrides)


def _load_case_axes(args):
    case = _load_case(args)
    axes = case.resolve_axes(args.axes)
    return case, axes


def _oracle_options(args) -> dict:
    """Parse repeatable ``--oracle-option KEY=VALUE`` settings."""
    options = {}
    for item in getattr(args, "oracle_option", None) or []:
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"--oracle-option expects KEY=VALUE, got {item!r}")
        value: object = text
        for convert in (int, float):
            try:
                value = convert(text)
                break
            except ValueError:
                pass
        # Parse case-insensitive boolean and null literals ("true", "false", "none").
        words = {"true": True, "false": False, "none": None}
        low = text.strip().lower()
        if low in words:
            value = words[low]
        options[key] = value
    return options


def _oracle(args, case):
    """``--oracle`` with its ``--oracle-option KEY=VALUE`` settings."""
    return orc.get_oracle(args.oracle, case, **_oracle_options(args))


def _certify_options(args) -> dict:
    """``--certify-workers`` and ``--relaxation-rounds`` for the certificate."""
    rounds = args.relaxation_rounds
    is_none = rounds is None or str(rounds).strip().lower() == "none"
    return {"workers": args.certify_workers,
            "relaxation_rounds": None if is_none else int(rounds)}


def _case_provenance(case, args) -> dict:
    """What a saved region needs to be reproduced: the network and its options."""
    return {"case": case.name, "case_options": dict(getattr(case, "factory_kwargs", None) or {}),
            "case_file": getattr(args, "case_file", None),
            "oracle": getattr(args, "oracle", None),
            "oracle_options": _oracle_options(args)}


def _default_stem(case, axes, tag: str = "") -> str:
    # Factory options (--load-scale, --feeder, ...) change the network, so they
    # are part of the name: two variants must not overwrite one another.
    options = "".join(f"_{k}={v}" for k, v in
                      sorted((getattr(case, "factory_kwargs", None) or {}).items()))
    stem = f"{case.name}{options}_{axes.label()}{('_' + tag) if tag else ''}"
    # Portable file names: no ':' (pp:<name>, invalid on Windows), and no '.'
    # or other characters a shell would need quoted ("1.5" becomes "1-5").
    return re.sub(r"[^A-Za-z0-9_=+-]+", "-", stem).strip("-")


def _build_expansive(args, case, axes):
    with _oracle(args, case) as oracle:
        if getattr(args, "strategy", None):
            from OpenRC.region.ray_growth import RayGrowthBuilder

            seed_kwargs = ({"n_samples": args.seed_samples, "batch_size": args.seed_batch_size}
                           if args.seed == "random" else {})
            strategy_kwargs = ({"policy": args.policy}
                               if args.strategy in ("facet-queue", "grow-and-polish") else {})
            return RayGrowthBuilder(strategy=args.strategy, strategy_kwargs=strategy_kwargs,
                                    seed=args.seed, seed_kwargs=seed_kwargs,
                                    seed_rng=args.rng).build(
                case, axes, oracle,
                max_vertices=args.max_vertices,
                max_moves=args.max_moves,
                max_seconds=args.max_seconds,
                max_oracle_calls=args.max_oracle_calls,
                memory_limit_mb=args.memory_limit_mb,
                max_hull_mb=args.max_hull_mb,
                progress_every=args.progress,
            )
        seed_kwargs = ({"n_samples": args.seed_samples, "batch_size": args.seed_batch_size}
                       if args.seed == "random" else {})
        builder = ExpansiveBuilder(
            policy=args.policy,
            seed=args.seed,
            seed_kwargs=seed_kwargs,
            seed_rng=args.rng,
            n_workers=args.workers,
            backend=args.backend,
        )
        return builder.build(
            case, axes, oracle,
            max_vertices=args.max_vertices,
            max_seconds=args.max_seconds,
            max_oracle_calls=args.max_oracle_calls,
            memory_limit_mb=args.memory_limit_mb,
            max_hull_mb=args.max_hull_mb,
            progress_every=args.progress,
        )


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_list(args) -> None:
    """List registered components across all OpenRC registries."""
    for family, entries in orc.registries().items():
        if family == "cases":
            curated = [(n, d) for n, d in entries if not n.startswith(orc.PP_PREFIX)]
            library = [(n, d) for n, d in entries if n.startswith(orc.PP_PREFIX)]
            _rule("CASES")
            width = max((len(n) for n, _ in curated), default=0)
            for name, doc in curated:
                print(f"  {name:<{width}}  {doc}")
            print(f"\n  + {len(library)} networks from pandapower.networks, as "
                  f"'{orc.PP_PREFIX}<name>'.")
            print("    `python scripts/demo.py pp list` names them; "
                  "`pp classify` says which oracles apply.")
            continue
        _rule(family.upper())
        width = max((len(n) for n, _ in entries), default=0)
        for name, doc in entries:
            print(f"  {name:<{width}}  {doc}")


def cmd_case(args) -> None:
    """Describe one network."""
    case = _load_case(args)
    _rule(f"case {case.name}")
    print(case.summary())
    conversion = case.metadata.get("conversion")
    if conversion:
        print(f"    imported from pandapower: {conversion['n_trafo_collapsed']} "
              f"transformer(s) collapsed, {conversion['n_unrated']} unrated line(s), "
              f"switches {conversion['switches_applied'] or 'none'}")
        for message in conversion["warnings"]:
            print(f"      note: {message}")
    unrated = conversion.get("n_unrated", 0) - conversion.get("n_rating_filled", 0) \
        if conversion else 0
    from OpenRC.powerflow.opf import opf_ready

    # The OPF oracles need generator costs and limits; the others do not.
    ready = opf_ready(case.net)
    power_flow = "AC/DC/OPF" if ready else "AC/DC (OPF needs generator costs and limits)"
    if not case.is_radial:
        applicable = (f"{'AC/DC power flow and OPF' if ready else 'AC/DC power flow'} only "
                      "(DistFlow needs a radial network)"
                      + ("" if ready else "; OPF needs generator costs and limits"))
    elif unrated:
        applicable = (f"lindistflow, socp, exact + {power_flow}; Model-II and Method 2 "
                      "also need ratings (--default-rating-ka)")
    else:
        applicable = f"the whole DistFlow family + {power_flow}"
    print(f"    oracles that apply: {applicable}")
    try:
        topo = case.topology
        print(f"    radial tree: {len(topo)} branches, depth "
              f"{max(topo.depth(b) for b in topo.buses)}")
        for name, buses in sorted(case.axis_presets.items()):
            distances = ", ".join(
                f"bus {b}: {topo.electrical_distance(b):.3f} ohm" for b in buses
            )
            print(f"    preset {name}: {distances}")
    except ValueError as exc:
        print(f"    (not radial: {exc})")


def cmd_pp(args) -> None:
    """The pandapower network library: list it, classify it, split a network."""
    if args.action == "list":
        entries = orc.list_pandapower_networks()
        _rule(f"{len(entries)} networks from pandapower.networks")
        width = max(len(n) for n, _ in entries)
        for name, doc in entries:
            print(f"  {name:<{width}}  {doc}")
        return

    if args.action == "classify":
        names = None
        if args.only:
            names = [n if n.startswith(orc.PP_PREFIX) else f"{orc.PP_PREFIX}{n}"
                     for n in args.only.split(",")]
        _rule("building and classifying" + (" (subset)" if names else " every network"))
        table = orc.classify_pandapower_networks(names)
        columns = ["case", "buses", "branches", "loops", "infeeds",
                   "trafo_collapsed", "radial", "oracles", "error"]
        import pandas as pd

        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(table[columns].to_string(index=False))
        ok = table.error == ""
        print(f"\n  imported {int(ok.sum())}/{len(table)}; "
              f"{int((table.radial == True).sum())} radial (DistFlow applies), "  # noqa: E712
              f"{int((table.radial == False).sum())} for AC/DC/OPF")  # noqa: E712
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"  saved -> {args.out}")
        return

    if args.action == "feeders":
        import pandapower as pp

        if args.case_file:
            net = pp.from_json(args.case_file)
        else:
            name = args.case[len(orc.PP_PREFIX):] if args.case.startswith(orc.PP_PREFIX) \
                else args.case
            import pandapower.networks as pn

            net = getattr(pn, name)()
        table = orc.list_feeders(net)
        _rule(f"independently fed parts ({len(table)})")
        print(table.to_string(index=False))
        print("\n  pass --feeder N to any command to work on one of them.")


def cmd_region(args) -> None:
    """Build one region and optionally save it."""
    case, axes = _load_case_axes(args)
    _rule(f"region {case.name} / {axes.label()}")
    result = _build_expansive(args, case, axes)
    print(result.summary())
    print(result.polytope.summary())
    if args.out:
        path = Path(args.out) / _default_stem(case, axes)
        result.builder_info["provenance"] = _case_provenance(case, args)
        result.save(path)
        print(f"    saved -> {path.with_suffix('.csv')}")


def cmd_conservative(args) -> None:
    """Build the optimistic region and both inner approximations, then compare."""
    case, axes = _load_case_axes(args)
    _rule(f"conservative constructions on {case.name} / {axes.label()}")

    seed_kwargs = ({"n_samples": args.seed_samples, "batch_size": args.seed_batch_size}
                   if args.seed == "random" else {})
    inner = ExpansiveBuilder(
        policy=args.policy, seed=args.seed, seed_kwargs=seed_kwargs, seed_rng=args.rng,
        n_workers=args.workers, backend=args.backend,
    )
    regions = {}
    oracle_options = _oracle_options(args)

    with orc.get_oracle("lindistflow", case, **oracle_options) as base_oracle:
        base = inner.build(case, axes, base_oracle,
                           max_vertices=args.max_vertices,
                           max_oracle_calls=args.max_oracle_calls,
                           memory_limit_mb=args.memory_limit_mb)

    all_recipes: list[tuple[str, str, dict[str, Any]]] = [
        ("m1", "M1 intersect-shift (path)", dict(method="intersect-shift")),
        ("m1-direct", "M1 direct-feeder scope", dict(method="intersect-shift",
                                                     shift_scope="feeding_branch")),
        ("m2", "M2 tight-intersect", dict(method="tight-intersect")),
        ("m2-subtree", "M2 subtree (recommended)", dict(method="tight-intersect",
                                                        tighten_scope="subtree")),
    ]
    recipes = [(label, kwargs) for key, label, kwargs in all_recipes
               if args.method in ("all", key)]
    runs = {}
    details = {}
    for label, kwargs in recipes:
        try:
            run = ConservativeBuilder(
                inner_builder=inner, oracle_kwargs=oracle_options, **kwargs
            ).build(
                case, axes, max_vertices=args.max_vertices,
                max_oracle_calls=args.max_oracle_calls,
                memory_limit_mb=args.memory_limit_mb,
            )
        except InfeasibleComponentOriginError as exc:
            print(f"  {label}: CANNOT START - {exc}")
            continue
        except EmptyComponentRegionError as exc:
            print(f"  {label}: NO COMPONENT BUILT - {exc}")
            continue
        if run.polytope.volume <= 0:
            print(f"  {label}: EMPTY region on this network")
            continue
        runs[label] = run
        regions[label] = run.polytope
        details[label] = run.builder_info

    print(f"\n  {'construction':<26} {'models intersected':<34} {'shift':<6} "
          f"{'theorem':<8} {'vertices':>9} {'volume':>11}")
    print(f"  {'model1 (no guarantee)':<26} {'lindistflow':<34} {'-':<6} "
          f"{'no':<8} {base.polytope.n_vertices:>9} {base.polytope.volume:>11.3f}")
    for label, region in regions.items():
        info = details[label]
        pair = f"{info['model_a']} + {info['model_b']}"
        shift = "yes" if info["shift_applied"] else "no"
        theorem = "yes" if info.get("theorem_backed") else "no"
        print(f"  {label:<26} {pair:<34} {shift:<6} "
              f"{theorem:<8} {region.n_vertices:>9} {region.volume:>11.3f}")

    runs = {"model1 (no guarantee)": base, **runs}
    regions = {"model1 (no guarantee)": base.polytope, **regions}

    if args.certify:
        _rule("global inner-region certificate (Gurobi, exact branch flow)")
        print(f"  {'construction':<30} {'status':<10} {'worst loading':>14} "
              f"{'V min':>8} {'V max':>8}")
        for label, region in regions.items():
            cert = orc.certify_inner_region(region, case, axes,
                                            surface_points=args.certify_points,
                                            **_certify_options(args))
            loading, low, high = cert.worst_loading, cert.worst_v_min, cert.worst_v_max
            print(f"  {label:<30} {cert.status:<10} "
                  f"{(100 * loading.bound if loading else float('nan')):>13.3f}% "
                  f"{(low.bound if low else float('nan')):>8.4f} "
                  f"{(high.bound if high else float('nan')):>8.4f}")
            for check in cert.violations[:3]:
                where = (np.round(check.witness, 3).tolist()
                         if check.witness is not None else "?")
                print(f"      {check.name}: {check.incumbent:.5f} at "
                      f"{where} [{check.note}]")

    if args.validate:
        # One common sampling box keeps the loss rates strictly comparable.
        lower, upper = base.polytope.bounding_box(args.margin)
        sampler = orc.get_sampler("uniform-box", lower=lower, upper=upper)
        _rule(f"validation against {args.reference} ({args.validate} samples)")
        print(f"  {'construction':<30} {'error %':>9} {'loss %':>9} "
              f"{'unresolved':>11}  verdict")
        for label, region in regions.items():
            quality = orc.evaluate_region(
                region, case, axes, reference=args.reference,
                n_samples=args.validate, sampler=sampler, rng=args.rng,
                n_workers=args.workers, backend=args.backend,
            )
            if quality.n_confirmed_violations:
                verdict = "NOT conservative"
            elif not quality.n_decided_in_region:     # error rate is nan
                verdict = "undecided (no decided in-region sample)"
            else:
                verdict = "no sampled violation"
            err_text = (f"{100 * quality.error_rate:9.3f}"
                        if quality.n_decided_in_region else f"{'n/a':>9}")
            loss_text = (f"{'n/a':>9}" if np.isnan(quality.loss_rate)
                         else f"{100 * quality.loss_rate:9.3f}")
            print(f"  {label:<30} {err_text} {loss_text} "
                  f"{quality.n_unresolved_total:>11}  {verdict}")
            if quality.n_unresolved_total:
                print("      NOTE: unresolved points need manual confirmation "
                      "(excluded from the error rate)")

    if args.out:
        for label, run_obj in runs.items():
            run_obj.builder_info["provenance"] = _case_provenance(case, args)
            run_obj.save(Path(args.out) / _default_stem(case, axes, label))
        print(f"    saved -> {args.out}")


def cmd_compress(args) -> None:
    """Compress a stored region."""
    region = orc.Polytope.load(Path(args.region))
    _rule(f"compressing {Path(args.region).name}")
    print(region.summary())
    compressor = orc.get_compressor(args.method)
    result = compressor.compress(
        region,
        target_vertices=args.target,
        max_volume_loss_pct=args.max_loss,
    )
    print("  " + result.summary())
    if args.out:
        path = Path(args.out) / (Path(args.region).stem + f"_{args.method}_{args.target}")
        saved = result.save(path)
        print(f"    saved -> {saved}")


def cmd_bench_compress(args) -> None:
    """Compare every compressor at several vertex targets."""
    region = orc.Polytope.load(Path(args.region))
    targets = [int(t) for t in args.targets.split(",")]
    _rule(f"compression benchmark on {Path(args.region).name} "
          f"({region.n_vertices} vertices, volume {region.volume:.4g})")
    print(f"  {'method':<14} " + " ".join(f"{'n=' + str(t):>18}" for t in targets))
    for name, _ in orc.list_compressors():
        cells = []
        for target in targets:
            result = orc.get_compressor(name).compress(region, target_vertices=target)
            cells.append(f"{result.volume_loss_pct:7.3f}% / {result.seconds:5.2f}s")
        print(f"  {name:<14} " + " ".join(f"{c:>18}" for c in cells))
    print("\n  cells show hypervolume lost / wall-clock time")


def cmd_bench_policy(args) -> None:
    """Compare facet-selection policies at an equal vertex budget."""
    case, axes = _load_case_axes(args)
    _rule(f"facet-policy benchmark on {case.name} / {axes.label()} "
          f"(budget {args.max_vertices} vertices)")
    print(f"  {'policy':<20} {'seed':<8} {'vertices':>9} {'volume':>11} "
          f"{'oracle calls':>13} {'seconds':>9}")
    if args.strategy is not None or args.max_moves is not None:
        raise SystemExit("bench-policy compares the facet-breaking builder's policies; "
                         "--strategy and --max-moves belong to `region --strategy`")
    policies = (args.policies.split(",") if args.policies
                else [n for n, _ in orc.list_facet_policies()])
    seeds = (args.seeds if args.seeds is not None
             else "simplex" if args.seed == "auto" else args.seed)
    for policy in policies:
        for seed in seeds.split(","):
            with _oracle(args, case) as oracle:
                seed_kwargs = ({"n_samples": args.seed_samples,
                                "batch_size": args.seed_batch_size}
                               if seed == "random" else {})
                builder = ExpansiveBuilder(
                    policy=policy, seed=seed, seed_kwargs=seed_kwargs, seed_rng=args.rng,
                    n_workers=args.workers, backend=args.backend,
                )
                result = builder.build(
                    case, axes, oracle, max_vertices=args.max_vertices,
                    max_seconds=args.max_seconds, max_oracle_calls=args.max_oracle_calls,
                    memory_limit_mb=args.memory_limit_mb, max_hull_mb=args.max_hull_mb,
                    progress_every=args.progress,
                )
                print(f"  {policy:<20} {seed:<8} {result.n_vertices:>9} "
                      f"{result.volume:>11.3f} {oracle.stats.n_calls:>13} "
                      f"{result.seconds:>9.2f}")


def cmd_memory(args) -> None:
    """Build a high-dimensional region under a vertex/memory budget."""
    if args.budget_mb is not None and args.schedule not in ("memory-budget", "adaptive"):
        raise SystemExit(
            f"--budget-mb applies only to --schedule memory-budget or adaptive, "
            f"got {args.schedule!r}"
        )
    case, axes = _load_case_axes(args)
    _rule(f"memory-aware build on {case.name} / {axes.label()} ({len(axes)}-D)")
    with _oracle(args, case) as oracle:
        builder = MemoryAwareBuilder(
            inner_builder=ExpansiveBuilder(
                policy=args.policy, seed=args.seed, seed_rng=args.rng,
                n_workers=args.workers, backend=args.backend,
            ),
            compressor=args.method,
            schedule=None if args.schedule == "none" else args.schedule,
            schedule_kwargs=({"budget_mb": args.budget_mb}
                             if args.budget_mb is not None and args.schedule in
                             ("memory-budget", "adaptive") else None),
            refiner=args.refiner,
            max_vertices=args.max_vertices,
            target_vertices=args.target_vertices,
            rounds=args.rounds,
        )
        result = builder.build(case, axes, oracle, memory_limit_mb=args.memory_limit)
    print(result.summary())
    print(result.polytope.summary())
    _rule("cycles")
    for cycle in result.builder_info["cycles"]:
        print(f"  round {cycle['round']:>2} {cycle['phase']:<9} "
              f"{cycle['n_vertices']:>4} vertices  volume {cycle['volume']:.6g}")
    if args.out:
        path = Path(args.out) / _default_stem(case, axes, f"{len(axes)}d")
        result.builder_info["provenance"] = _case_provenance(case, args)
        result.save(path)
        print(f"    saved -> {path.with_suffix('.csv')}")


def cmd_validate(args) -> None:
    """Estimate the loss and error rate of a stored region."""
    case, axes = _load_case_axes(args)
    region = orc.Polytope.load(Path(args.region), axes=axes)
    _rule(f"validating {Path(args.region).name} against {args.reference}")
    quality = orc.evaluate_region(
        region, case, axes,
        reference=args.reference,
        adjudicate=not args.no_adjudicate,
        n_samples=args.samples,
        sampler=args.sampler,
        sampler_kwargs={"margin": args.margin} if "box" in args.sampler else None,
        rng=args.rng,
        n_workers=args.workers,
        backend=args.backend,
        progress_every=args.progress,
    )
    print(quality.summary())
    if len(quality.violating_points):
        from OpenRC.validation import diagnose_violations

        _rule("violation diagnostics")
        print("  " + json.dumps(diagnose_violations(case, axes, quality.violating_points),
                                indent=2).replace("\n", "\n  "))
    if len(quality.unresolved_points):
        _rule("unresolved points (confirm these manually)")
        for point in quality.unresolved_points[:10]:
            print("  ", np.round(point, 4))


def cmd_verify(args) -> None:
    """Check that a stored region is inner: global certificate or Monte Carlo."""
    case, axes = _load_case_axes(args)
    region = orc.Polytope.load(Path(args.region), axes=axes)
    _rule(f"verifying {Path(args.region).name} ({args.mode})")
    kwargs = dict(mode=args.mode, n_points=args.points, primary=args.primary,
                  method=None if args.method == "none" else args.method,
                  exact_time_limit=args.time_limit, rng=args.rng)
    if args.mode == "surface":
        kwargs.update(_certify_options(args))
    if args.mode == "monte-carlo":
        kwargs.update(cube_scale=args.cube_scale, min_inside=args.min_inside,
                      n_workers=args.workers, backend=args.backend)
    report = orc.verify_inner_region(region, case, axes, **kwargs)
    print(report.summary())
    if args.shrink and report.status not in ("certified", "passed"):
        _rule("contracting until certified")
        from OpenRC.validation.certify import default_point_verifier

        # The same surface points, first stage and theory sub-models as above.
        verifier_options = dict(primary=args.primary, exact_time_limit=args.time_limit)
        if args.method != "auto":
            verifier_options["method"] = None if args.method == "none" else args.method
        shrink_options = dict(_certify_options(args))
        if args.mode == "surface" and args.points is not None:
            shrink_options["surface_points"] = args.points
        shrink_options["rng"] = args.rng          # the same surface points as above
        with default_point_verifier(region, case, **verifier_options) as verifier:
            shrunk, cert, lam = orc.shrink_to_certified(region, case, axes,
                                                         time_limit=args.time_limit,
                                                         point_verifier=verifier,
                                                         **shrink_options)
        print(cert.summary())
        source = Path(args.region)
        if source.suffix.lower() in (".csv", ".json"):
            source = source.with_suffix("")
        out = source.as_posix() + "_certified"
        shrunk.save(Path(out))
        print(f"    lam = {lam:.4f}; saved -> {out}.csv")


def cmd_apply(args) -> None:
    """Run an application over a region, at several compression levels."""
    case, axes = _load_case_axes(args)
    region = orc.Polytope.load(Path(args.region), axes=axes)
    generator = np.random.default_rng(args.rng)
    dim = region.dim

    if args.problem == "storage-sizing":
        lower, upper = region.bounding_box(args.margin)
        scenarios = generator.uniform(lower, upper, size=(args.scenarios, dim))
    else:
        scenarios = generator.uniform(0.8, 1.0, size=(args.scenarios, dim))

    application = orc.get_application(args.problem)
    _rule(f"{args.problem} on {Path(args.region).name} ({args.scenarios} scenarios)")
    print(f"  {'region':<16} {'vertices':>9} {'facets':>8} {'objective':>11} "
          f"{'ms/solve':>10}")
    variants = [("original", region)]
    for target in [int(t) for t in args.targets.split(",") if t]:
        compressed = orc.get_compressor(args.method).compress(
            region, target_vertices=target).polytope
        variants.append((f"compressed n={target}", compressed))
    for label, body in variants:
        batch = application.solve_batch(body, scenarios)
        print(f"  {label:<16} {body.n_vertices:>9} {body.n_facets:>8} "
              f"{batch.mean_objective:>11.4f} {batch.mean_ms:>10.3f}")


def cmd_tour(args) -> None:
    """A short guided run through the whole workflow."""
    case, axes = _load_case_axes(args)

    _rule("1. the network")
    print(case.summary())

    _rule("2. oracles agree on the base point")
    for name in ("lindistflow", "lindistflow-vdrop", "socp", "acpf"):
        with orc.get_oracle(name, case) as oracle:
            try:
                value = oracle.verdict_at(axes, np.zeros(len(axes))).value
            except orc.SolverUnavailableError as exc:
                value = f"skipped ({exc})"
        print(f"  {name:<20} {value}")

    _rule("3. build the region (random-accelerated seed)")
    with orc.get_oracle("lindistflow", case) as oracle:
        result = ExpansiveBuilder(policy="adaptive-ema", seed="random",
                                  seed_kwargs={"n_samples": 150}).build(
            case, axes, oracle, max_vertices=args.max_vertices)
    print("  " + result.summary().replace("\n", "\n  "))
    print(f"  seed: {result.builder_info['seed']}")
    region = result.polytope

    _rule("4. compress it")
    for name in ("local-search", "hybrid-vns"):
        compressed = orc.get_compressor(name).compress(region, target_vertices=15)
        print("  " + compressed.summary())

    _rule("5. how good is it?")
    quality = orc.evaluate_region(region, case, axes, n_samples=args.validate,
                                 rng=args.rng)
    print("  " + quality.summary().replace("\n", "\n  "))

    _rule("6. use it")
    generator = np.random.default_rng(args.rng)
    scenarios = generator.uniform(0.8, 1.0, size=(100, len(axes)))
    small = orc.get_compressor("local-search").compress(region, target_vertices=15).polytope
    for label, body in [("original", region), ("compressed", small)]:
        batch = orc.get_application("capacity-allocation").solve_batch(body, scenarios)
        print(f"  capacity allocation, {label:<10} "
              f"{body.n_facets:>4} constraints  objective {batch.mean_objective:7.3f}  "
              f"{batch.mean_ms:6.2f} ms/solve")


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, *, axes: bool = True):
        p.add_argument("--case", default="9bus",
                       help="registered case: a curated name, or pp:<name> for any "
                            "network in pandapower.networks")
        p.add_argument("--case-file", default=None,
                       help="any pandapower network saved with pandapower.to_json")
        p.add_argument("--feeder", type=int, default=None,
                       help="keep only this independently fed part (see `pp feeders`)")
        p.add_argument("--default-rating-ka", type=float, default=None,
                       help="rating for lines that carry only a placeholder")
        p.add_argument("--load-scale", type=float, default=None,
                       help="scale the recorded base load")
        p.add_argument("--transformer-handling", default=None,
                       choices=("auto", "collapse", "keep"))
        p.add_argument("--slack-vm-pu", type=_parse_slack_vm_pu, default=None,
                       help="slack bus voltage (p.u.) for branch-flow models, or 'setpoint'")
        p.add_argument("--line-charging", default=None,
                       choices=("omit", "constant"),
                       help="how branch-flow models treat line shunt capacitance")
        if axes:
            p.add_argument("--axes", default="247",
                           help="preset name, or '1,3,6', or '1p,1q,3p'")
        p.add_argument("--rng", type=int, default=0, help="random seed")

    def add_parallel(p):
        p.add_argument("--workers", type=_parse_workers, default=1,
                       help="worker count, auto, or all (batched operations only)")
        p.add_argument("--backend", choices=("process", "thread"), default="process")

    def add_build(p):
        p.add_argument("--oracle", default="lindistflow")
        p.add_argument("--oracle-option", action="append", metavar="KEY=VALUE",
                       help="oracle keyword, repeatable: e.g. backend=solver "
                            "modeling=gurobi; numbers, true/false and none are "
                            "converted (any case)")
        p.add_argument("--policy", default="adaptive-ema")
        p.add_argument("--seed", default="auto",
                       choices=["auto"] + [n for n, _ in orc.list_seeds()],
                       help="auto: axis from 4-D up, simplex below")
        p.add_argument("--seed-samples", type=int, default=150,
                       help="random points for --seed random")
        p.add_argument("--seed-batch-size", type=int, default=10_000,
                       help="maximum random seed points held at once")
        p.add_argument("--max-vertices", type=int, default=200)
        p.add_argument("--max-seconds", type=float, default=None)
        p.add_argument("--max-oracle-calls", type=int, default=None)
        p.add_argument("--memory-limit-mb", type=float, default=None)
        p.add_argument("--progress", type=int, default=0)
        p.add_argument("--strategy", default=None,
                       choices=[n for n, _ in orc.list_ray_strategies()],
                       help="grow with this ray strategy (ray-growth builder) instead of "
                            "the facet-breaking builder")
        p.add_argument("--max-moves", type=int, default=None,
                       help="ray-growth: move budget (default 4 x --max-vertices)")
        p.add_argument("--max-hull-mb", type=float, default=None,
                       help="ray-growth: hard cap on the live hull's memory")
        add_parallel(p)

    p = sub.add_parser("list", help="show every registry")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("pp", help="the pandapower network library")
    p.add_argument("action", choices=("list", "classify", "feeders"))
    p.add_argument("--only", default=None, help="classify: comma-separated subset")
    p.add_argument("--out", default=None, help="classify: save the table as CSV")
    p.add_argument("--case", default="mv_oberrhein", help="feeders: network name")
    p.add_argument("--case-file", default=None, help="feeders: a pandapower JSON file")
    p.set_defaults(func=cmd_pp)

    p = sub.add_parser("case", help="describe a network")
    add_common(p, axes=False)
    p.set_defaults(func=cmd_case)

    p = sub.add_parser("region", help="build a region")
    add_common(p)
    add_build(p)
    p.add_argument("--out", default=None, help="directory to save into")
    p.set_defaults(func=cmd_region)

    p = sub.add_parser("conservative", help="compare the inner-approximation methods")
    add_common(p)
    p.add_argument("--method", default="all",
                   choices=("all", "m2-subtree", "m2", "m1", "m1-direct"),
                   help="run every conservative construction or one selected recipe")
    p.add_argument("--policy", default="adaptive-ema")
    p.add_argument("--seed", default="simplex")
    p.add_argument("--seed-samples", type=int, default=150)
    p.add_argument("--seed-batch-size", type=int, default=10_000)
    p.add_argument("--max-vertices", type=int, default=150)
    p.add_argument("--max-oracle-calls", type=int, default=None)
    p.add_argument("--memory-limit-mb", type=float, default=None)
    p.add_argument("--oracle-option", action="append", metavar="KEY=VALUE",
                   help="option for both component oracles, repeatable: e.g. "
                        "backend=solver candidate_solver=newton")
    add_parallel(p)
    p.add_argument("--validate", type=int, default=0,
                   help="number of validation samples (0 = skip)")
    p.add_argument("--reference", default="acpf")
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--certify", action="store_true",
                   help="prove or refute each region globally (needs Gurobi)")
    p.add_argument("--certify-points", type=int, default=16,
                   help="surface points checked by the certificate")
    p.add_argument("--certify-workers", type=int, default=1,
                   help="global solves of the certificate in parallel threads")
    p.add_argument("--relaxation-rounds", default="3",
                   help="bound-tightening rounds of the relaxation screen, or 'none'")
    p.add_argument("--out", default=None, help="directory to save into")
    p.set_defaults(func=cmd_conservative)

    p = sub.add_parser("compress", help="compress a stored region")
    p.add_argument("--region", required=True)
    p.add_argument("--method", default="local-search",
                   choices=[n for n, _ in orc.list_compressors()])
    p.add_argument("--target", type=int, default=15)
    p.add_argument("--max-loss", type=float, default=None,
                   help="stop before exceeding this %% of hypervolume")
    p.add_argument("--out", default=None, help="directory to save into")
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("bench-compress", help="compare compressors")
    p.add_argument("--region", required=True)
    p.add_argument("--targets", default="50,15")
    p.set_defaults(func=cmd_bench_compress)

    p = sub.add_parser("bench-policy", help="compare facet-selection policies")
    add_common(p)
    add_build(p)
    p.add_argument("--policies", default=None, help="comma-separated subset")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed strategies (default: --seed, or simplex)")
    p.set_defaults(func=cmd_bench_policy)

    p = sub.add_parser("memory", help="memory-aware high-dimensional build")
    add_common(p)
    p.add_argument("--oracle", default="lindistflow")
    p.add_argument("--oracle-option", action="append", metavar="KEY=VALUE",
                   help="oracle keyword, repeatable: e.g. backend=solver "
                        "modeling=gurobi")
    p.add_argument("--policy", default="adaptive-ema")
    p.add_argument("--seed", default="auto")
    p.add_argument("--method", default="auto", help="compressor (auto: by dimension)")
    p.add_argument("--schedule", default="vertex-budget",
                   choices=["none"] + [n for n, _ in orc.list_schedules()])
    p.add_argument("--budget-mb", type=float, default=None,
                   help="live-hull memory budget of the memory/adaptive schedules")
    p.add_argument("--refiner", default=None,
                   choices=[n for n, _ in orc.list_refiners()],
                   help="relocate vertices after every expansion")
    p.add_argument("--max-vertices", type=int, default=150)
    p.add_argument("--target-vertices", type=int, default=75)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--memory-limit", "--memory-limit-mb", dest="memory_limit",
                   type=float, default=None, help="MiB")
    add_parallel(p)
    p.add_argument("--out", default=None, help="directory to save into")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("validate", help="loss rate and error rate of a stored region")
    add_common(p)
    p.add_argument("--region", required=True)
    p.add_argument("--reference", default="acpf")
    p.add_argument("--no-adjudicate", action="store_true",
                   help="skip solver adjudication of candidate violations")
    p.add_argument("--samples", type=_positive_int, default=1000)
    p.add_argument("--sampler", default="uniform-box",
                   choices=[n for n, _ in orc.list_samplers() if n != "scenarios"])
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--progress", type=int, default=0)
    add_parallel(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("verify", help="prove (surface) or sample (monte-carlo) that a "
                                      "stored region is inner")
    add_common(p)
    p.add_argument("--region", required=True)
    p.add_argument("--mode", default="surface", choices=list(orc.validation.VERIFY_MODES))
    p.add_argument("--points", type=int, default=None,
                   help="surface points (default 32) or Monte Carlo draws (default 5000)")
    p.add_argument("--primary", default="acpf", choices=["acpf", "acopf"])
    p.add_argument("--method", default="auto",
                   choices=["auto", "tight-intersect", "intersect-shift", "none"],
                   help="whose sub-models form the theory stage")
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--cube-scale", type=float, default=1.5)
    p.add_argument("--min-inside", type=int, default=None)
    p.add_argument("--shrink", action="store_true",
                   help="if not certified, contract the region until it is and save it")
    p.add_argument("--certify-workers", type=int, default=1,
                   help="global solves of the certificate in parallel threads")
    p.add_argument("--relaxation-rounds", default="3",
                   help="bound-tightening rounds of the relaxation screen, or 'none'")
    add_parallel(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("apply", help="run an application over a region")
    add_common(p)
    p.add_argument("--region", required=True)
    p.add_argument("--problem", default="capacity-allocation",
                   choices=[n for n, _ in orc.list_applications()])
    p.add_argument("--scenarios", type=_positive_int, default=200)
    p.add_argument("--targets", default="50,15")
    p.add_argument("--method", default="local-search")
    p.add_argument("--margin", type=float, default=2.0)
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("tour", help="guided end-to-end walkthrough")
    add_common(p)
    p.add_argument("--max-vertices", type=int, default=150)
    p.add_argument("--validate", type=int, default=500)
    p.set_defaults(func=cmd_tour)

    return parser


def main() -> None:
    orc.configure_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
