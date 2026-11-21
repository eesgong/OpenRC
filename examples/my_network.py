"""Radial pandapower example: python examples/my_network.py.

Replace the example network and explicitly choose its bus IDs. load_scale=0.5
is part of this experiment, not an automatic repair for an infeasible network.
Writes out/my_feeder.csv and its provenance sidecar in the working directory.
"""

import pandapower.networks as pn
import OpenRC as orc


def main():
    case = orc.case_from_pandapower(
        pn.mv_oberrhein(), feeder=1, slack_vm_pu="setpoint", load_scale=0.5,
    )
    axes = case.resolve_axes(["30p", "60p", "90p"])  # 0-based bus IDs
    run = orc.get_builder(
        "conservative", method="tight-intersect", tighten_scope="subtree",
    ).build(case, axes, max_vertices=24)
    audit = orc.evaluate_region(
        run.polytope, case, axes, reference="acpf", adjudicate=False,
        sampler="in-region", n_samples=200, rng=0,
    )
    print(run.summary(), audit.summary(), sep="\n")
    if not run.builder_info["theorem_backed"]:
        raise RuntimeError("The selected configuration does not satisfy the theorem conditions.")
    if audit.n_decided_in_region != 200 or audit.n_confirmed_violations:
        raise RuntimeError("Audit acceptance failed; inspect unresolved points and violations.")
    run.save("out/my_feeder")
    print("PASS: saved out/my_feeder.csv + .json; sample audit is not global certification.")
    return run, audit


if __name__ == "__main__":
    main()
