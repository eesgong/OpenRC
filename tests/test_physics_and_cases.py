"""Tests for power-flow physics, pandapower adapter consistency, OPF, and case rebuilds."""
import math
import threading
import unittest

import numpy as np
import pandapower as pp
import pandapower.networks as pn

import OpenRC as orc
from OpenRC.parallel import _recipe_mismatch, evaluate_points
from OpenRC.powerflow.base import Verdict, _model_compatibility
from OpenRC.powerflow.topology import (
    RadialTopology,
    effective_line_current_ka,
    unrated_lines,
)


class PhysicsAndCasesTests(unittest.TestCase):
    def _small_radial_net(self) -> pp.pandapowerNet:
        net = pp.create_empty_network(sn_mva=10.0)
        b0 = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.92, max_vm_pu=1.08)
        b1 = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.92, max_vm_pu=1.08)
        b2 = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.92, max_vm_pu=1.08)
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
        pp.create_line_from_parameters(
            net, from_bus=b0, to_bus=b1, length_km=2.0,
            r_ohm_per_km=0.25, x_ohm_per_km=0.35, c_nf_per_km=0.0, max_i_ka=0.4,
        )
        pp.create_line_from_parameters(
            net, from_bus=b1, to_bus=b2, length_km=2.0,
            r_ohm_per_km=0.25, x_ohm_per_km=0.35, c_nf_per_km=0.0, max_i_ka=0.4,
        )
        pp.create_load(net, bus=b1, p_mw=1.0, q_mvar=0.2)
        pp.create_load(net, bus=b2, p_mw=0.8, q_mvar=0.1)
        return net

    def test_unrated_nan_and_inf_lines_and_max_loading_percent(self):
        net = self._small_radial_net()
        net.line.loc[0, "max_i_ka"] = float("nan")
        net.line.loc[1, "max_loading_percent"] = 80.0

        self.assertEqual(list(unrated_lines(net)), [0])
        self.assertAlmostEqual(effective_line_current_ka(net.line.loc[1]), 0.32)

        case = orc.case_from_pandapower(net)
        topo = RadialTopology.from_case(case)
        by_child = {br.child: br for br in topo.branches}
        br_unrated = by_child[1]
        br_derated = by_child[2]
        self.assertFalse(br_unrated.rated)
        self.assertTrue(math.isinf(br_unrated.i_max_a))
        self.assertTrue(br_derated.rated)
        axes = case.resolve_axes(["1p", "2p"])
        origin = np.zeros(2)
        with orc.get_oracle("lindistflow", case) as df, orc.get_oracle("acpf", case) as ac:
            self.assertTrue(_model_compatibility(df, ac)[0])
            self.assertIs(df.verdict_at(axes, origin), Verdict.FEASIBLE)
            self.assertIs(ac.verdict_at(axes, origin), Verdict.FEASIBLE)

        # Also check explicit +inf rating.
        net.line.loc[0, "max_i_ka"] = float("inf")
        case_inf = orc.case_from_pandapower(net)
        self.assertEqual(list(unrated_lines(net)), [0])
        with orc.get_oracle("lindistflow", case_inf) as df_inf, orc.get_oracle("acpf", case_inf) as ac_inf:
            self.assertTrue(_model_compatibility(df_inf, ac_inf)[0])
            self.assertIs(df_inf.verdict_at(axes, origin), Verdict.FEASIBLE)

    def test_sgen_limit_clipping_matches_between_case_and_acpf(self):
        net = self._small_radial_net()
        # Reactive injection exceeds max_q_mvar; user_pf_options clips it in pandapower >= 3.
        pp.create_sgen(
            net, bus=2, p_mw=1.5, q_mvar=4.0,
            min_p_mw=0.0, max_p_mw=1.0, min_q_mvar=-0.5, max_q_mvar=0.5,
        )
        pp.set_user_pf_options(net, enforce_p_lims=True, enforce_q_lims=True)
        case = orc.case_from_pandapower(net)
        # Base load at bus 2 is (0.8, 0.1); clipped sgen is (1.0, 0.5) -> net (-0.2, -0.4).
        self.assertAlmostEqual(float(case.effective_base_p_mw[2]), -0.2, places=9)
        self.assertAlmostEqual(float(case.effective_base_q_mvar[2]), -0.4, places=9)

        axes = case.resolve_axes(["1p", "2p"])
        with orc.get_oracle("acpf", case) as ac, orc.get_oracle("lindistflow", case) as df:
            # At x = 0, acpf must evaluate the exact clipped operating point, matching net.
            self.assertIs(ac.verdict_at(axes, np.zeros(2)), Verdict.FEASIBLE)
            self.assertIs(df.verdict_at(axes, np.zeros(2)), Verdict.FEASIBLE)
            # Check actual solved AC bus injection at x = [0.3, -0.1] (+delta = consumption).
            ac.verdict_at(axes, np.array([0.3, -0.1]))
            net_p_bus2 = float(ac._net.res_bus.loc[2, "p_mw"])
            self.assertAlmostEqual(net_p_bus2, -0.3, places=6)

    def test_explicit_vm_limits_override_pandapower_bus_columns(self):
        case_default = orc.case_from_pandapower(pn.case33bw())
        self.assertAlmostEqual(case_default.vm_min_pu, 0.9)
        self.assertAlmostEqual(case_default.vm_max_pu, 1.1)

        case_tight = orc.get_case("pp:case33bw", vm_min_pu=0.96, vm_max_pu=1.04)
        self.assertAlmostEqual(case_tight.vm_min_pu, 0.96)
        self.assertAlmostEqual(case_tight.vm_max_pu, 1.04)
        with orc.get_oracle("lindistflow", case_tight) as df, orc.get_oracle("acpf", case_tight) as ac:
            lo, hi = ac._bus_vm_limits(case_tight.net)
            self.assertAlmostEqual(float(lo[1]), 0.96)
            self.assertAlmostEqual(float(hi[1]), 1.04)
            self.assertTrue(_model_compatibility(df, ac)[0])

    def test_opf_preserves_user_controllable_load_and_thread_safe_logging(self):
        net = self._small_radial_net()
        net.ext_grid["min_p_mw"] = -20.0
        net.ext_grid["max_p_mw"] = 20.0
        net.ext_grid["min_q_mvar"] = -20.0
        net.ext_grid["max_q_mvar"] = 20.0
        pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=20.0)

        # Add a curtailable flexible load on bus 2 that makes heavy injection feasible.
        flex_idx = pp.create_load(
            net, bus=2, p_mw=0.5, q_mvar=0.0,
            min_p_mw=0.0, max_p_mw=6.0, min_q_mvar=0.0, max_q_mvar=0.0,
            controllable=True,
        )
        pp.create_poly_cost(net, flex_idx, "load", cp1_eur_per_mw=-5.0)

        case = orc.case_from_pandapower(net)
        axes = case.resolve_axes(["1p"])
        with orc.get_oracle("dcopf", case) as dcopf:
            dcopf._prepare_net()
            self.assertIs(bool(dcopf._net.load.loc[flex_idx, "controllable"]), True)
            for ctrl_idx in dcopf._control_rows:
                self.assertIs(bool(dcopf._net.load.loc[int(ctrl_idx), "controllable"]), False)
            self.assertIs(dcopf.verdict_at(axes, np.zeros(1)), Verdict.FEASIBLE)
            self.assertGreater(len(dcopf._net.res_bus), 0)
            dcopf.reset_warm_state()
            self.assertEqual(len(dcopf._net.res_bus), 0)

            # Concurrent _solver_logging blocks must restore logger level deterministically.
            import logging

            pp_logger = logging.getLogger("pandapower")
            orig_level = pp_logger.level
            pp_logger.setLevel(logging.INFO)
            barrier = threading.Barrier(4)

            def worker():
                barrier.wait()
                with dcopf._solver_logging():
                    barrier.wait()
                    self.assertEqual(pp_logger.level, logging.ERROR)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(pp_logger.level, logging.INFO)
            pp_logger.setLevel(orig_level)

    def test_case_rebuild_preserves_multiprocess_recipe(self):
        base = orc.get_case("9bus")
        lighter = base.rebuild(load_scale=0.4)
        self.assertIsNone(_recipe_mismatch(lighter))
        axes = lighter.resolve_axes(["1p", "3p"])
        pts = np.array([[0.0, 0.0], [0.2, 0.2], [-0.2, 0.1], [0.1, -0.2]])
        mask_1 = evaluate_points(lighter, axes, "lindistflow", pts, n_workers=1)
        mask_2 = evaluate_points(lighter, axes, "lindistflow", pts, n_workers=2)
        np.testing.assert_array_equal(mask_1, mask_2)

    def test_multi_infeed_message_mentions_feeder_and_list_feeders(self):
        net = self._small_radial_net()
        b3 = pp.create_bus(net, vn_kv=20.0)
        pp.create_ext_grid(net, bus=b3, vm_pu=1.0)
        case = orc.case_from_pandapower(net)
        report = case.radiality
        self.assertFalse(report.is_radial)
        self.assertIn("feeder=", report.message)
        self.assertIn("list_feeders", report.message)


if __name__ == "__main__":
    unittest.main()
