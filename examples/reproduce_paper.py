"""Reproduce the archived Model-I volume within 50 ppm, without commercial solvers.

This regression concerns the linearised paper model, not an AC inner certificate.
Run: python examples/reproduce_paper.py
"""

import math
import OpenRC as orc


def main():
    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")
    with orc.get_oracle("lindistflow", case) as oracle:
        run = orc.get_builder(
            "expansive", policy="volume-gain", refresh_gains=False,
            ray_kwargs={"tol": 1e-5},
        ).build(case, axes, oracle, max_vertices=14)
    ppm = (run.volume / 301.000642981 - 1) * 1e6
    print(f"{run.volume:.9f} MW^3 ({ppm:+.1f} ppm)")
    if not math.isfinite(ppm) or abs(ppm) >= 50:
        raise RuntimeError(f"Paper regression outside the 50 ppm acceptance band: {ppm}")
    print("PASS: paper volume regression; linearised Model-I, not an AC certificate.")
    return run


if __name__ == "__main__":
    main()
