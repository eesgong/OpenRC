"""Tests for provenance persistence/stripping, Polytope geometry, and certification."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import OpenRC as orc
from OpenRC.geometry import Polytope
from OpenRC.geometry.ops import EmptyIntersectionError
from OpenRC.region.refine import get_refiner


class ProvenanceAndGeometryTests(unittest.TestCase):
    def test_provenance_survives_compression_and_disk_roundtrip(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes("247")
        run = orc.get_builder(
            "conservative", method="tight-intersect", tighten_scope="subtree"
        ).build(case, axes, max_vertices=16)

        self.assertIn("builder", run.polytope.metadata)
        self.assertIs(run.polytope.metadata["builder"]["theorem_backed"], True)

        small = orc.get_compressor("local-search").compress(
            run.polytope, target_vertices=10
        )
        self.assertIn("builder", small.polytope.metadata)
        self.assertIs(small.polytope.metadata["builder"]["theorem_backed"], True)
        self.assertEqual(small.polytope.metadata["compression"]["compressor"], "local-search")

        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "compressed_region"
            csv_path = small.save(stem)
            self.assertTrue(csv_path.exists())
            self.assertTrue(stem.with_suffix(".json").exists())
            self.assertTrue((Path(tmp) / "compressed_region_compression_trace.csv").exists())

            loaded = Polytope.load(csv_path)
            self.assertEqual(loaded.axes, axes)
            self.assertIs(loaded.metadata["builder"]["theorem_backed"], True)
            self.assertEqual(loaded.metadata["compression"]["compressor"], "local-search")

    def test_non_subset_transforms_strip_region_and_certificate_claims(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes("247")
        run = orc.get_builder(
            "conservative", method="tight-intersect", tighten_scope="subtree"
        ).build(case, axes, max_vertices=16)
        annotated = run.polytope.replace(
            certified=True,
            certified_contraction=1.0,
            certificate={"status": "certified"},
            custom_tag="keep-me",
        )

        # Zero translation preserves claims; non-zero translation strips them.
        zero_shifted = annotated.translate([0.0, 0.0, 0.0])
        self.assertIn("builder", zero_shifted.metadata)
        self.assertTrue(zero_shifted.metadata.get("certified"))

        shifted = annotated.translate([0.1, 0.0, 0.0])
        projected = annotated.subspace(["1p", "6p"])
        sliced = annotated.slice({"3p": 0.0})

        with orc.get_oracle("lindistflow", case) as oracle:
            refined, _ = get_refiner("relocation").refine_polytope(
                annotated, oracle, axes, max_moves=1
            )

        for transformed in (shifted, projected, sliced, refined):
            self.assertEqual(transformed.metadata.get("custom_tag"), "keep-me")
            for claim in (
                "builder",
                "oracle_stats",
                "theorem_backed",
                "guarantee",
                "certified",
                "certified_contraction",
                "certificate",
            ):
                self.assertNotIn(claim, transformed.metadata)

    def test_subspace_and_slice_accept_indices_and_axis_selectors(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes("247")  # buses 1p, 3p, 6p
        corners = np.array(
            [[x, y, z] for x in (-1.0, 2.0) for y in (-0.5, 1.5) for z in (-2.0, 3.0)]
        )
        box = Polytope(corners, axes=axes)

        p_idx = box.subspace([0, 2])
        p_sel = box.subspace(["1p", "b6p"])
        self.assertEqual(p_idx.axes, p_sel.axes)
        self.assertAlmostEqual(p_sel.volume, 3.0 * 5.0, places=9)

        cross = box.slice({"3p": 0.25})
        self.assertEqual(cross.dim, 2)
        self.assertEqual(cross.axes.labels, ["b1p", "b6p"])
        self.assertAlmostEqual(cross.volume, 3.0 * 5.0, places=9)

        with self.assertRaises(EmptyIntersectionError):
            box.slice({"3p": 5.0})  # outside [-0.5, 1.5]

        with self.assertRaises(ValueError):
            Polytope(corners).subspace(["1p", "6p"])  # no axes attached

        # Cross-section of a curved 3-D power-flow polytope must stay inside P.
        run = orc.get_builder(
            "conservative", method="tight-intersect", tighten_scope="subtree"
        ).build(case, axes, max_vertices=20)
        section = run.polytope.slice({"6p": 0.05})
        lifted = np.column_stack(
            [section.vertices[:, 0], section.vertices[:, 1], np.full(section.n_vertices, 0.05)]
        )
        self.assertTrue(np.all(run.polytope.contains(lifted, tol=1e-7)))

    def test_contains_with_positive_tol_is_invariant_under_reduce_to_hull(self):
        square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        skew_interior = np.repeat([[0.01, 0.01]], repeats=50, axis=0)
        with_redundant = Polytope(np.vstack([square, skew_interior]))
        reduced = with_redundant.reduce_to_hull()

        # Probe points straddling the boundary within the tolerance band.
        probes = np.array(
            [
                [2.0 + 1e-4, 1.0],
                [2.0 + 2e-4, 1.0],
                [-1e-4, 0.5],
                [-2e-4, 0.5],
                [1.0, 2.0 + 1.4e-4],
            ]
        )
        np.testing.assert_array_equal(
            with_redundant.contains(probes, tol=1e-4),
            reduced.contains(probes, tol=1e-4),
        )

    def test_load_bare_csv_axis_inference_and_high_dim_repr_laziness(self):
        pts_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            named_csv = Path(tmp) / "named.csv"
            named_csv.write_text("b1p,b3q\n0,0\n1,0\n0,1\n", encoding="utf-8")
            loaded_named = Polytope.load(named_csv)
            self.assertIsNotNone(loaded_named.axes)
            self.assertEqual(loaded_named.axes.labels, ["b1p", "b3q"])

            generic_csv = Path(tmp) / "generic.csv"
            np.savetxt(generic_csv, pts_2d, delimiter=",", header="0,1", comments="")
            loaded_generic = Polytope.load(generic_csv)
            self.assertIsNone(loaded_generic.axes)

            coerced = Polytope.load(generic_csv, axes=["1p", "3p"])
            self.assertEqual(coerced.axes.labels, ["b1p", "b3p"])

        simplex_6d = Polytope(np.vstack([np.zeros((1, 6)), np.eye(6)]))
        text = repr(simplex_6d)
        self.assertIn("dim=6", text)
        self.assertNotIn("hull", simplex_6d.__dict__)
        self.assertNotIn("volume", simplex_6d.__dict__)

    def test_shrink_to_certified_attaches_certificate_metadata(self):
        case = orc.get_case("9bus")
        axes = case.resolve_axes(["1p", "3p"])
        inner = orc.get_builder(
            "conservative", method="tight-intersect", tighten_scope="subtree"
        ).build(case, axes, max_vertices=12).polytope
        try:
            shrunk_ok, cert_ok, lam_ok = orc.shrink_to_certified(
                inner, case, axes, surface_points=4
            )
        except orc.SolverUnavailableError:
            self.skipTest("Global certification solver unavailable")
        self.assertEqual(lam_ok, 1.0)
        self.assertTrue(cert_ok.certified)
        self.assertIs(shrunk_ok.metadata.get("certified"), True)
        self.assertEqual(shrunk_ok.metadata.get("certified_contraction"), 1.0)
        self.assertEqual(shrunk_ok.metadata["certificate"]["status"], "certified")

        with orc.get_oracle("lindistflow", case) as outer_oracle:
            outer = orc.get_builder("expansive").build(
                case, axes, outer_oracle, max_vertices=14
            ).polytope
        shrunk_lt1, cert_lt1, lam_lt1 = orc.shrink_to_certified(
            outer, case, axes, surface_points=4, rel_tol=0.05, max_probes=8
        )
        self.assertLess(lam_lt1, 1.0)
        self.assertTrue(cert_lt1.certified)
        self.assertIs(shrunk_lt1.metadata.get("certified"), True)
        self.assertAlmostEqual(shrunk_lt1.metadata["certified_contraction"], lam_lt1)
        self.assertEqual(shrunk_lt1.metadata["certificate"]["status"], "certified")


if __name__ == "__main__":
    unittest.main()
