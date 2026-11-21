"""Regression tests for the distinction between geometry, evidence and proof."""
import unittest
import warnings

import numpy as np

import OpenRC as orc
from OpenRC.geometry import Polytope
from OpenRC.region.base import RegionResult
from OpenRC.validation.quality import RegionQuality


class EvidenceReportingTests(unittest.TestCase):
    def test_candidate_guarantee_is_visible_without_inspecting_metadata(self):
        run = RegionResult(
            Polytope(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
            builder_info={"theorem_backed": False, "guarantee": "candidate only"},
        )
        self.assertIn("theorem_backed: False", run.summary())
        self.assertIn("candidate only", run.summary())

    def test_error_denominator_excludes_outside_and_unresolved_samples(self):
        q = RegionQuality(200, 15, 190, 13, n_unresolved_in_region=2)
        self.assertEqual(q.n_decided_in_region, 13)
        self.assertAlmostEqual(q.error_rate_ci[1], 1.96 ** 2 / (13 + 1.96 ** 2))
        self.assertIn("13 decided in-region samples", q.summary())
        self.assertIn("2 in-region point(s) could not be decided", q.summary())

    def test_zero_decided_points_is_not_success(self):
        q = RegionQuality(10, 0, 0, 0)
        self.assertFalse(q.is_conservative)
        self.assertIn("not estimated: no decided", q.summary())
        self.assertNotIn("nan%", q.summary())

    def test_dependent_sampling_does_not_claim_binomial_interval_or_loss(self):
        q = RegionQuality(
            20, 20, 20, 20, info={"iid_sampling": False, "loss_measured": False}
        )
        self.assertIn("binomial CI unavailable", q.summary())
        self.assertIn("not measured", q.summary())

    def test_single_oracle_builders_always_report_guarantee_and_theorem_flag(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes(["1p", "3p"])

        with orc.get_oracle("lindistflow", case) as df_oracle:
            exp_probe = orc.get_builder("expansive").build(
                case, axes, df_oracle, max_vertices=10
            )
            self.assertIs(exp_probe.builder_info["theorem_backed"], False)
            self.assertIn("boundary probes", exp_probe.builder_info["guarantee"])
            self.assertIn("theorem_backed: False", exp_probe.summary())

            mem = orc.get_builder(
                "memory-aware", max_vertices=12, target_vertices=8, rounds=1
            ).build(case, axes, df_oracle)
            self.assertIs(mem.builder_info["theorem_backed"], False)
            self.assertIn("boundary probes", mem.builder_info["guarantee"])
            self.assertEqual(
                mem.polytope.metadata["builder"]["theorem_backed"], False
            )

            ray = orc.get_builder("ray-growth").build(
                case, axes, df_oracle, max_vertices=10
            )
            self.assertIs(ray.builder_info["theorem_backed"], False)
            self.assertIn("boundary probes", ray.builder_info["guarantee"])

        with orc.get_oracle("socp", case) as socp_oracle:
            self.assertIn(
                "outer-relaxation",
                orc.region.base.single_oracle_guarantee(socp_oracle),
            )

    def test_conservative_tighten_scope_and_ignored_oracle_warning(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes("247")

        m2_path = orc.get_builder("conservative", method="tight-intersect").build(
            case, axes, max_vertices=12
        )
        self.assertIs(m2_path.builder_info["theorem_backed"], False)

        m2_sub = orc.get_builder(
            "conservative", method="tight-intersect", tighten_scope="subtree"
        ).build(case, axes, max_vertices=12)
        self.assertIs(m2_sub.builder_info["theorem_backed"], True)
        self.assertEqual(m2_sub.builder_info["tighten_scope"], "subtree")

        with self.assertRaises(ValueError):
            orc.get_builder(
                "conservative", method="intersect-shift", tighten_scope="subtree"
            )
        with self.assertRaises(ValueError):
            orc.get_builder(
                "conservative",
                method="tight-intersect",
                tighten_scope="subtree",
                oracle_kwargs={"tighten_scope": "path"},
            )

        with orc.get_oracle("lindistflow", case) as ignored_oracle:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                orc.get_builder(
                    "conservative", method="tight-intersect", tighten_scope="subtree"
                ).build(case, axes, oracle=ignored_oracle, max_vertices=10)
            self.assertTrue(
                any("ignores the supplied oracle" in str(w.message) for w in caught)
            )

    def test_result_repr_and_as_dict_are_concise_and_complete(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes(["1p", "3p"])
        run = orc.get_builder(
            "conservative", method="tight-intersect", tighten_scope="subtree"
        ).build(case, axes, max_vertices=12)
        small = orc.get_compressor("local-search").compress(
            run.polytope, target_vertices=8
        )
        scenarios = np.random.default_rng(0).uniform(0.8, 1.0, size=(20, 2))
        batch = orc.get_application("capacity-allocation").solve_batch(
            small.polytope, scenarios
        )
        quality = RegionQuality(50, 20, 40, 20)

        for obj in (run, small, batch, quality):
            text = repr(obj)
            self.assertLess(len(text), 200)
            self.assertNotIn("\n", text)

        payload = run.as_dict()
        self.assertEqual(payload["n_vertices"], run.n_vertices)
        self.assertIs(payload["builder_info"]["theorem_backed"], True)


if __name__ == "__main__":
    unittest.main()
