"""Tests for parallel/benchmark/resources/viz workflows and scripts/demo.py CLI."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import OpenRC as orc
from OpenRC import viz
from OpenRC.benchmark import BENCHMARK_METHODS, random_instances, summarize
from OpenRC.geometry.live_hull import live_hull_memory_mb
from OpenRC.parallel import evaluate_points
from OpenRC.resources import extrapolate_vertex_budget


REPO_ROOT = Path(__file__).resolve().parent.parent


class WorkflowsAndCliTests(unittest.TestCase):
    def test_oracle_alias_distflow_and_auto_workers(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes(["1p", "3p"])
        pts = np.zeros((12, 2))
        mask_alias = evaluate_points(case, axes, "distflow", pts, n_workers=1)
        mask_auto = evaluate_points(case, axes, "distflow", pts, n_workers="auto")
        self.assertTrue(np.all(mask_alias))
        np.testing.assert_array_equal(mask_alias, mask_auto)

    def test_benchmark_random_instances_and_summarize_edge_cases(self):
        instances = random_instances("9bus", [2, 3], seeds=[0, 1], oracle="lindistflow")
        self.assertEqual(len(instances), 4)

        # Infeasible base point raises immediately without looping over seeds.
        with self.assertRaises(RuntimeError):
            random_instances(
                "9bus", [2, 3], seeds=[0, 1, 2],
                oracle="lindistflow", case_kwargs={"load_scale": 50.0},
            )

        # Paired 0.0 / 0.0 tie and multi-oracle grouping in summarize().
        runs = pd.DataFrame(
            [
                dict(case="9bus", oracle="lindistflow", dim=2, seed=0, axes="b1p,b2p",
                     instance_id="i0", method="a", volume=0.0, oracle_calls=10,
                     seconds=0.1, error=""),
                dict(case="9bus", oracle="lindistflow", dim=2, seed=0, axes="b1p,b2p",
                     instance_id="i0", method="b", volume=0.0, oracle_calls=12,
                     seconds=0.1, error=""),
                dict(case="9bus", oracle="acpf", dim=2, seed=0, axes="b1p,b2p",
                     instance_id="i1", method="a", volume=1.0, oracle_calls=10,
                     seconds=0.2, error=""),
                dict(case="9bus", oracle="acpf", dim=2, seed=0, axes="b1p,b2p",
                     instance_id="i1", method="b", volume=2.0, oracle_calls=14,
                     seconds=0.2, error=""),
            ]
        )
        summary = summarize(runs, baseline="a")
        self.assertIn("oracle", summary.columns)
        self.assertEqual(len(summary), 4)
        tie_row = summary[(summary["oracle"] == "lindistflow") & (summary["method"] == "b")].iloc[0]
        self.assertAlmostEqual(float(tie_row["ratio"]), 1.0)
        self.assertAlmostEqual(float(tie_row["tie_rate"]), 1.0)
        self.assertEqual(int(tie_row["ratio_excluded"]), 0)

        # Conservative benchmark method records non-zero oracle_calls via result.oracle_stats.
        @BENCHMARK_METHODS.register("_test_conservative_calls")
        def _cons_method(case, axes, oracle, *, max_vertices, seed):
            return orc.get_builder(
                "conservative", method="tight-intersect", tighten_scope="subtree"
            ).build(case, axes, max_vertices=max_vertices)

        try:
            df = orc.benchmark.run_benchmark(
                ["_test_conservative_calls"], instances[:1], vertices_per_dim=5.0
            )
            self.assertGreater(int(df["oracle_calls"].iloc[0]), 0)
        finally:
            BENCHMARK_METHODS._items.pop("_test_conservative_calls", None)
            BENCHMARK_METHODS._docs.pop("_test_conservative_calls", None)

    def test_extrapolate_vertex_budget_matches_live_hull_memory_mb(self):
        # Exact power law F(N) = 2 * N^2 observed at N = 10 and N = 20.
        observed = [(10, 200), (20, 800)]
        for dim in (3, 6, 8):
            budget_mb = 4.0
            n = extrapolate_vertex_budget(observed, dim, budget_mb)
            self.assertLessEqual(live_hull_memory_mb(n, dim, 2 * n * n), budget_mb)
            self.assertGreater(live_hull_memory_mb(n + 1, dim, 2 * (n + 1) ** 2), budget_mb)

    def test_viz_plot_region_2d_and_3d_projection_and_slice(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes(["1p", "3p", "6p", "7p"])
        with orc.get_oracle("lindistflow", case) as oracle:
            region = orc.get_builder("expansive").build(
                case, axes, oracle, max_vertices=20
            ).polytope

        fig2_proj, ax2_proj = plt.subplots()
        viz.plot_region_2d(region, ax=ax2_proj, dims=("1p", "3p"), label="proj")
        self.assertEqual(ax2_proj.get_xlabel(), "P @ bus 1 (MW)")

        fig2_slice, ax2_slice = plt.subplots()
        viz.plot_region_2d(
            region, ax=ax2_slice, dims=("1p", "3p"), fixed={"6p": 0.0, "7p": 0.0}
        )
        self.assertEqual(ax2_slice.get_ylabel(), "P @ bus 3 (MW)")

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection="3d")
        viz.plot_region_3d(region, ax=ax3, dims=("1p", "3p", "6p"), fixed={"7p": 0.0})
        self.assertEqual(ax3.get_zlabel(), "P @ bus 6 (MW)")
        plt.close("all")

    def _run_demo(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "demo.py"), *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )

    def test_demo_cli_new_flags_and_artifact_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "out"

            # 1. pp classify with nested non-existent --out parent directory
            pp_csv = out_dir / "pp_table.csv"
            res_pp = self._run_demo("pp", "classify", "--only", "case33bw", "--out", str(pp_csv))
            self.assertEqual(res_pp.returncode, 0, msg=res_pp.stderr)
            self.assertTrue(pp_csv.exists())

            # 2. conservative --method m2-subtree --validate 40 --out
            cons_dir = out_dir / "cons"
            res_cons = self._run_demo(
                "conservative", "--case", "9bus", "--axes", "247",
                "--method", "m2-subtree", "--max-vertices", "16",
                "--validate", "40", "--out", str(cons_dir),
            )
            self.assertEqual(res_cons.returncode, 0, msg=res_cons.stderr)
            self.assertIn("M2 subtree (recommended)", res_cons.stdout)
            self.assertIn("yes", res_cons.stdout)
            self.assertNotIn("M1 intersect-shift", res_cons.stdout)
            saved_csvs = sorted(p for p in cons_dir.glob("*.csv") if not p.name.endswith("_trace.csv"))
            self.assertEqual(len(saved_csvs), 2)  # model1 + M2 subtree
            self.assertTrue(any(cons_dir.glob("*_trace.csv")))

            # 3. compress --out writes compression trace CSV alongside polytope CSV/JSON
            comp_dir = out_dir / "comp"
            res_comp = self._run_demo(
                "compress", "--region", str(saved_csvs[-1]),
                "--target", "10", "--out", str(comp_dir),
            )
            self.assertEqual(res_comp.returncode, 0, msg=res_comp.stderr)
            self.assertTrue(any(comp_dir.glob("*_compression_trace.csv")))

            # 4. validate --no-adjudicate
            res_val = self._run_demo(
                "validate", "--case", "9bus", "--axes", "247",
                "--region", str(saved_csvs[-1]), "--samples", "30", "--no-adjudicate",
            )
            self.assertEqual(res_val.returncode, 0, msg=res_val.stderr)
            self.assertIn("samples", res_val.stdout)

            # 5. memory rejects --budget-mb when schedule is vertex-budget, and accepts --memory-limit-mb
            res_bad_mem = self._run_demo(
                "memory", "--case", "9bus", "--axes", "247",
                "--schedule", "vertex-budget", "--budget-mb", "4.0",
            )
            self.assertNotEqual(res_bad_mem.returncode, 0)
            self.assertIn("--budget-mb applies only", res_bad_mem.stderr)

            res_ok_mem = self._run_demo(
                "memory", "--case", "9bus", "--axes", "247",
                "--max-vertices", "16", "--target-vertices", "10", "--rounds", "1",
                "--memory-limit-mb", "4096",
            )
            self.assertEqual(res_ok_mem.returncode, 0, msg=res_ok_mem.stderr)


if __name__ == "__main__":
    unittest.main()
