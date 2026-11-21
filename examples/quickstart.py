"""Solver-free first run. Install OpenRC, then: python examples/quickstart.py.

This script is also distributed unchanged by the documentation website.
Timings are observations, never pass/fail thresholds. The theorem flag concerns
the supported radial branch-flow model and its assumptions, not arbitrary grids.
"""

import numpy as np
import OpenRC as orc


def main():
    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")  # paper POCs 2,4,7 -> buses 1,3,6
    run = orc.get_builder(
        "conservative", method="tight-intersect", tighten_scope="subtree",
    ).build(case, axes, max_vertices=24)
    small = orc.get_compressor("local-search").compress(
        run.polytope, target_vertices=14,
    )
    scenarios = np.random.default_rng(0).uniform(0.8, 1.0, size=(300, len(axes)))
    batch = orc.get_application("capacity-allocation").solve_batch(
        small.polytope, scenarios,
    )
    # Draw inside the region to audit violations; this does not measure lost volume.
    audit = orc.evaluate_region(
        small.polytope, case, axes, reference="acpf", adjudicate=False,
        sampler="in-region", n_samples=400, rng=0,
    )
    print(run.summary(), small.summary(), batch.summary(), audit.summary(), sep="\n")
    checks = {
        "theorem conditions": run.builder_info["theorem_backed"],
        "positive finite volume": np.isfinite(small.polytope.volume) and small.polytope.volume > 0,
        "compression stays inside source": bool(np.all(run.polytope.contains(small.polytope.vertices))),
        "300 LPs solved": batch.n_solved == 300,
        "400 decided audit samples": audit.n_decided_in_region == 400,
        "no sampled violations": audit.n_confirmed_violations == 0,
    }
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    if not all(checks.values()):
        raise RuntimeError("Quickstart acceptance failed; inspect the summary above.")
    print("PASS: quickstart; sampling is not a global AC certificate. Timings may vary.")
    return run, small, batch, audit


if __name__ == "__main__":
    main()
