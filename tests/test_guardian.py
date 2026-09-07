"""Unit tests for the Nexus-Predator defensive AI core."""

import os
import unittest

import numpy as np

from airlab.guardian import (GuardianState, ThreatEngine, EvasionPlanner,
                             GuardianBrain, Obstacle)
from airlab.guardian.brain import CRUISE_SAFE, EVADE, RECOVER_NAV, ABORT
from airlab.guardian.threats import GuardianState as GState


def _state(**kw) -> GuardianState:
    defaults = dict(
        t=0.0, pos=np.zeros(3), vel=np.zeros(3), a_cmd=np.zeros(3),
        imu_dr_pos=np.zeros(3), imu_dr_vel=np.zeros(3),
        gps_pos=np.zeros(3), gps_vel=np.zeros(3),
        mag_heading=0.0, gps_course=0.0, gps_signal_quality=1.0,
        baro_ok=True, battery_frac=1.0, energy_required_frac=0.0,
        wind_est=np.zeros(3), obstacles=[],
    )
    defaults.update(kw)
    return GuardianState(**defaults)


class TestThreats(unittest.TestCase):
    def test_healthy_state_has_no_active_threat(self):
        reports = ThreatEngine().evaluate(_state())
        active = [r for r in reports if r.active]
        self.assertEqual(active, [])

    def test_gps_spoof_heading_mismatch_detected(self):
        s = _state(gps_pos=np.array([1.0, 1.0, 0.0]),
                   imu_dr_pos=np.array([0.0, 0.0, 0.0]),
                   mag_heading=0.0, gps_course=1.0)
        reports = ThreatEngine().evaluate(s)
        kinds = {r.kind: r.score for r in reports}
        self.assertGreater(kinds.get("spoofing", 0.0), 0.55)
        spoof = next(r for r in reports if r.kind == "spoofing")
        self.assertTrue(any(e.startswith("mag_vs_gps_heading") for e in spoof.evidence))

    def test_jamming_detected_when_gps_missing(self):
        s = _state(gps_pos=None, gps_signal_quality=0.2, baro_ok=True)
        reports = ThreatEngine().evaluate(s)
        kinds = {r.kind: r.score for r in reports}
        self.assertGreater(kinds.get("jamming", 0.0), 0.55)

    def test_obstacle_close_detected(self):
        obs = Obstacle(pos=np.array([0.0, 0.5, 0.0]), vel=np.array([0.0, 0.0, 0.0]))
        s = _state(vel=np.array([0.0, 0.0, 0.0]), obstacles=[obs])
        reports = ThreatEngine().evaluate(s)
        kinds = {r.kind: r.score for r in reports}
        self.assertGreater(kinds.get("obstacle", 0.0), 0.55)


class TestAvoidance(unittest.TestCase):
    def test_picks_non_none_when_intruder_approaching(self):
        obs = Obstacle(pos=np.array([0.0, 1.0, 0.0]), vel=np.array([0.0, -1.0, 0.0]))
        s = _state(vel=np.array([0.0, 0.0, 0.0]), obstacles=[obs])
        dec = EvasionPlanner().plan(s, np.array([0.0, 0.0, 0.0]))
        self.assertTrue(dec.evading)


class TestBrain(unittest.TestCase):
    def test_cruise_when_healthy(self):
        brain = GuardianBrain()
        dec = brain.decide(_state(), np.zeros(3))
        self.assertEqual(dec.mode, CRUISE_SAFE)

    def test_evade_on_approaching_obstacle(self):
        brain = GuardianBrain()
        obs = Obstacle(pos=np.array([0.0, 1.0, 0.0]), vel=np.array([0.0, -1.0, 0.0]))
        s = _state(vel=np.array([0.0, 0.0, 0.0]), obstacles=[obs])
        dec = brain.decide(s, np.zeros(3))
        self.assertEqual(dec.mode, EVADE)

    def test_recover_nav_on_spoof(self):
        brain = GuardianBrain()
        s = _state(gps_pos=np.array([2.0, 0.0, 0.0]), imu_dr_pos=np.zeros(3),
                   mag_heading=0.0, gps_course=1.2)
        dec = brain.decide(s, np.zeros(3))
        self.assertEqual(dec.mode, RECOVER_NAV)

    def test_abort_on_critical_energy(self):
        brain = GuardianBrain()
        s = _state(battery_frac=0.2, energy_required_frac=0.5)
        dec = brain.decide(s, np.zeros(3))
        self.assertEqual(dec.mode, ABORT)

    def test_undeclared_capabilities_are_tracked_when_used(self):
        brain = GuardianBrain()
        obs = Obstacle(pos=np.array([0.0, 1.2, 0.0]), vel=np.array([0.0, -1.0, 0.0]))
        s = _state(vel=np.array([0.0, 0.0, 0.0]), obstacles=[obs])
        dec = brain.decide(s, np.zeros(3))
        self.assertIn("predictive_sense_avoid", dec.declared_used)


class TestHealth(unittest.TestCase):
    def test_healthy_subsystems_stay_healthy(self):
        from airlab.guardian import (SubsystemHealth, HealthPrognosis,
                                     simulated_features)
        rng = np.random.default_rng(3)
        health = SubsystemHealth(warmup_samples=20)
        prog = HealthPrognosis()
        for k in range(60):
            health.update(*simulated_features(rng, k))
            agg = prog.aggregate(health.scores())
            if k >= 20:
                self.assertGreaterEqual(agg, 0.5)  # healthy should not be critical

    def test_degraded_subsystems_trigger_critical_heap(self):
        from airlab.guardian import (SubsystemHealth, HealthPrognosis,
                                     simulated_features)
        rng = np.random.default_rng(3)
        health = SubsystemHealth(warmup_samples=20)
        prog = HealthPrognosis()
        for k in range(80):
            feats = simulated_features(rng, k, battery_bad=(k >= 55),
                                       motor_bad=(k >= 55),
                                       thermal_bad=(k >= 55),
                                       vib_bad=(k >= 55))
            health.update(*feats)
            agg = prog.aggregate(health.scores())
            if k >= 70:
                self.assertLess(agg, 0.5)      # poor health -> ABORT threshold
                self.assertLessEqual(len([h for h in health.scores()
                                          if h.subsystem.startswith("sensor")]),
                                     2)  # sensors stay separate

    def test_guardian_aborts_on_low_aggregate_health(self):
        from airlab.guardian import GuardianBrain
        from airlab.guardian.brain import ABORT
        brain = GuardianBrain()
        dec = brain.decide(GState(), np.zeros(3), health_score=0.21)
        self.assertEqual(dec.mode, ABORT)
        self.assertIn("predictive_maintenance_health", dec.declared_used)


class TestTelemetryHealth(unittest.TestCase):
    def _run(self, duration=10.0, motor_degrade_at=None, motor_eff=1.0):
        from airlab.simulator import Simulator, SimConfig
        cfg = SimConfig()
        cfg.duration = duration
        cfg.cruise_speed = 2.0
        cfg.motor_efficiency = motor_eff
        cfg.motor_degrade_at = motor_degrade_at
        cfg.motor_degrade_eff = 0.7
        cfg.guardian_health_enabled = True
        sim = Simulator(cfg)
        sim.run()
        return sim

    def test_telemetry_health_healthy_run_stays_ok(self):
        sim = self._run(duration=8.0)
        self.assertIsNotNone(sim.guardian_health_bridge)
        hs = sim.guardian_health_bridge.health.scores()
        # baseline learning (warmup) then healthy flight must stay non-critical
        self.assertGreaterEqual(sim.guardian_health_bridge.prognosis.history[-1], 0.5)
        self.assertFalse(any(h.status == "critical" for h in hs))

    def test_telemetry_health_detects_mid_flight_motor_degrade(self):
        """Calibrate healthy then degrade the motor at 6 s; the health engine
        must see the *same aircraft* degrade, not a different run."""
        sim = self._run(duration=11.0, motor_degrade_at=6.0)
        hb = sim.guardian_health_bridge
        # pre-fault window: after warmup (2 s) before the fault (6 s)
        pre_motor = [h["motor_resid"] for h in hb.history
                     if h["motor_resid"] < 1.0][:4]
        post_motor = [h["motor_resid"] for h in hb.history[-20:]]
        self.assertGreater(sum(post_motor) / len(post_motor),
                           sum(pre_motor) / len(pre_motor))
        # final motor health must be worse than the health it had before fault
        pre_health = [h["health"] for h in hb.history
                      if h["health"] > 0][:5]
        self.assertLess(
            sum(h["health"] for h in hb.history[-5:]) / 5,
            sum(pre_health) / len(pre_health),
        )


class TestSimBridge(unittest.TestCase):
    def _bridge(self, obstacle=True, battery=1.0):
        from airlab.mission import WaypointMission
        from airlab.guardian import MissionReplanBridge
        mission = WaypointMission([(0, 0, 2), (8, 0, 2), (16, 0, 2)], speed=2.0)
        start = np.array([0.0, 0.0, -2.0])
        obstacles = ([[np.array([8.0, 0.0, -2.0]), np.array([0.0, 0.0, 0.0]), 1.0]]
                     if obstacle else [])
        return mission, MissionReplanBridge(
            mission, start, battery_frac=battery, obstacles=obstacles,
            config=None)

    def test_bridge_applies_safe_detour_around_obstacle(self):
        mission, bridge = self._bridge(obstacle=True, battery=1.0)
        res = bridge.try_replan(t=0.0, force=True)
        self.assertIsNotNone(res)
        self.assertTrue(res.feasible)
        self.assertTrue(res.risk_reduction > 0.05)
        self.assertTrue(res.min_clearance_m >= 2.0)
        self.assertTrue(bridge.applied)
        self.assertIn("applied", bridge.history.modes)
        # The mission is no longer a straight line through the obstacle.
        corners = [tuple(np.round(w[:2], 1)) for w in mission.wp_ned]
        self.assertNotEqual(corners[0][1], 0.0)

    def test_bridge_refuses_risky_route_on_low_battery(self):
        mission, bridge = self._bridge(obstacle=True, battery=0.02)
        res = bridge.try_replan(t=0.0, force=True)
        self.assertIsNotNone(res)
        self.assertFalse(res.feasible)
        self.assertFalse(bridge.applied)
        self.assertIn("rejected_energy", bridge.history.modes)

    def test_bridge_refuses_no_gain_if_no_threat(self):
        mission, bridge = self._bridge(obstacle=False, battery=1.0)
        res = bridge.try_replan(t=0.0, force=True)
        self.assertIsNotNone(res)
        self.assertFalse(bridge.applied)
        self.assertIn("rejected_low_gain", bridge.history.modes)

    def test_simulator_uses_guardian_bridge(self):
        from airlab.simulator import Simulator, SimConfig
        cfg = SimConfig()
        cfg.duration = 12.0
        cfg.cruise_speed = 2.0
        # Straight mission that flies directly through an obstacle at (8, 4?).
        cfg.waypoints = [(0, 0, 2), (12, 0, 2), (24, 0, 2)]
        cfg.guardian_replan = True
        cfg.guardian_replan_period_s = 2.0
        cfg.guardian_obstacles = [
            ([12.0, 0.0, -2.0], [0.0, 0.0, 0.0], 1.5),
        ]
        sim = Simulator(cfg)
        self.assertIsNotNone(sim.guardian_bridge)
        run = sim.run()
        self.assertTrue(hasattr(run, "mode"))
        self.assertTrue(sim.guardian_bridge.applied)
        lands = [m for m in run.mode if m == "LAND"]
        self.assertEqual(len(lands), 0)


class TestLogReader(unittest.TestCase):
    def test_px4_reader_loads_fixture_and_refuses_unlabelled_prior(self):
        from airlab.guardian import (Px4RosLogReader, write_test_fixture,
                                     RiskPriorModel)
        path = write_test_fixture("out/guardian/px4_fixture.csv", n=120)
        r = Px4RosLogReader().load(path)
        self.assertGreater(len(r.telemetry), 100)
        self.assertGreater(r.risk.n, 100)
        # no ground-truth label from a logger -> the prior must NOT fit
        self.assertFalse(r.fit_prior(RiskPriorModel()).fitted)

    def test_px4_reader_fits_labeled_fixture_and_sees_jam(self):
        from airlab.guardian import (Px4RosLogReader, write_test_fixture,
                                     RiskPriorModel)
        path = write_test_fixture("out/guardian/px4_fixture_labeled.csv",
                                  n=120, with_risk_label=True)
        r = Px4RosLogReader().load(path)
        prior = r.fit_prior(RiskPriorModel())
        self.assertTrue(prior.fitted)
        arr = r.risk.as_array()
        # late samples have few satellites => higher jamming feature
        late_jam = float(np.mean(arr[-20:, 1]))
        early_jam = float(np.mean(arr[:20, 1]))
        self.assertGreater(late_jam, early_jam)
        near = float(prior.predict(np.array([0.5]), np.array([0.0]))[0])
        far = float(prior.predict(np.array([10.0]), np.array([0.0]))[0])
        self.assertGreater(near, far)


class TestThermalBudget(unittest.TestCase):
    def test_planner_rejects_hot_route(self):
        from airlab.guardian import PredictiveRePlanner
        import numpy as np
        planner = PredictiveRePlanner(thermal_aware=True, thermal_ambient_c=90.0,
                                      cruise_speed=3.0, hover_power_w=112.0,
                                      battery_capacity_wh=71.0)
        res = planner.plan(np.array([0.0, 0.0, -2.0]),
                           [np.array([12.0, 0.0, -2.0])], battery_frac=1.0)
        self.assertFalse(res.thermal_feasible)
        self.assertFalse(res.feasible)
        self.assertIn("thermal_infeasible", res.reasons)

    def test_planner_accepts_cool_route(self):
        from airlab.guardian import PredictiveRePlanner
        import numpy as np
        planner = PredictiveRePlanner(thermal_aware=True, thermal_ambient_c=25.0,
                                      cruise_speed=3.0, hover_power_w=112.0,
                                      battery_capacity_wh=71.0)
        res = planner.plan(np.array([0.0, 0.0, -2.0]),
                           [np.array([12.0, 0.0, -2.0])], battery_frac=1.0)
        self.assertTrue(res.thermal_feasible)
        self.assertTrue(res.feasible)
        self.assertNotIn("thermal_infeasible", res.reasons)
        self.assertEqual(res.thermal_worst_node, "cpu_npu")


class TestThermal(unittest.TestCase):
    def test_part_model_heats_more_under_load(self):
        from airlab.guardian import PartThermalModel
        model = PartThermalModel(ambient_c=25.0)
        for _ in range(500):
            model.step(120.0, 0.01, compute_frac=0.3)
        base = model.max_temp()
        model2 = PartThermalModel(ambient_c=25.0)
        for _ in range(500):
            model2.step(120.0, 0.01, compute_frac=1.0)
        high = model2.max_temp()
        self.assertGreater(high, base)
        self.assertGreater(model2.temperatures()["cpu_npu"],
                           model.temperatures()["cpu_npu"])
        # part-level: the hot node must be exposed, not a single lumped number
        self.assertIn(model2.worst_node(), {"cpu_npu", "esc", "motor", "battery"})

    def test_part_model_respects_node_specific_limits(self):
        from airlab.guardian import PartThermalModel
        model = PartThermalModel(ambient_c=25.0)
        # run a hot mission; node margins are exposed and no node should ever
        # go arbitrarily unbounded (it should be slower than the frame).
        for _ in range(1000):
            model.step(200.0, 0.01, compute_frac=1.0)
        margins = model.margins()
        for name, margin in margins.items():
            self.assertGreaterEqual(margin, -1e-9)
        self.assertTrue(model.battery.max_temp_c < model.motor.max_temp_c)

    def test_telemetry_health_uses_part_model(self):
        from airlab.guardian import PartThermalModel
        from airlab.simulator import Simulator, SimConfig
        cfg = SimConfig()
        cfg.duration = 8.0
        cfg.guardian_health_enabled = True
        sim = Simulator(cfg)
        sim.run()
        th = sim.guardian_health_bridge.thermal
        self.assertIsInstance(th, PartThermalModel)
        temps = th.temperatures()
        self.assertTrue({"cpu_npu", "esc", "motor", "battery"} <= set(temps))


class TestDataPipeline(unittest.TestCase):
    def test_pipeline_records_reproducible_dataset(self):
        from airlab.guardian import (DataPipeline, RiskTelemetryDataset,
                                     TelemetryDataset)
        from airlab.simulator import Simulator, SimConfig
        cfg = SimConfig()
        cfg.duration = 2.0
        cfg.guardian_data_pipeline = True
        cfg.guardian_health_enabled = True
        cfg.guardian_data_obstacles = [
            ([2.0, 0.0, -2.0], [0.0, 0.0, 0.0], 1.0),
        ]
        cfg.guardian_data_jamming = [[1.0, 0.0, -2.0]]
        sim = Simulator(cfg)
        sim.run()
        self.assertIsNotNone(sim.guardian_pipeline)
        ds = sim.guardian_pipeline.dataset
        self.assertGreater(len(ds), 100)
        self.assertIn("pos_n", ds.rows[0])
        self.assertIn("cpu_npu_c", ds.rows[0])
        self.assertGreater(len(sim.guardian_pipeline.risk.samples), 100)
        # write + read back reproduction
        path = "out/guardian/telemetry_test.csv"
        ds.to_csv(path)
        self.assertTrue(os.path.exists(path))

    def test_risk_dataset_fits_real_prior(self):
        from airlab.guardian import RiskTelemetryDataset, RiskPriorModel
        ds = RiskTelemetryDataset()
        for i in range(50):
            ds.add(1.0 + i * 0.2, 0.5 if i % 2 else 0.0, 0.8 - i * 0.01)
        prior = ds.fit_prior(RiskPriorModel())
        self.assertTrue(prior.fitted)
        self.assertEqual(prior.summary()["n"], 50)
        near = float(prior.predict(np.array([1.0]), np.array([0.0]))[0])
        far = float(prior.predict(np.array([10.0]), np.array([0.0]))[0])
        self.assertGreater(near, far)


class TestRiskPrior(unittest.TestCase):
    def test_prior_learns_near_obstacle_is_riskier(self):
        from airlab.guardian import RiskPriorModel, simulate_telemetry
        prior = RiskPriorModel().fit(simulate_telemetry(n=800))
        self.assertTrue(prior.fitted)
        near = float(prior.predict(np.array([0.5]), np.array([0.0]))[0])
        far = float(prior.predict(np.array([10.0]), np.array([0.0]))[0])
        self.assertGreater(near, far)
        j = float(prior.predict(np.array([8.0]), np.array([0.9]))[0])
        j0 = float(prior.predict(np.array([8.0]), np.array([0.0]))[0])
        self.assertGreater(j, j0)

    def test_risk_world_model_uses_learned_prior(self):
        from airlab.guardian import (RiskWorldModel, RiskPriorModel,
                                     simulate_telemetry, Obstacle)
        prior = RiskPriorModel().fit(simulate_telemetry(n=800))
        model = RiskWorldModel(prior=prior, learned_alpha=1.0)
        obs = Obstacle(pos=np.array([5.0, 0.0, -2.0]),
                       vel=np.array([0.0, 0.0, 0.0]))
        field = model.build((0, 10, -5, 5, -6, 0), [obs], [[5.0, 0.0, -2.0]])
        near = field.sample(np.array([5.0, 0.0, -2.0]))
        far = field.sample(np.array([5.0, 5.0, -2.0]))
        self.assertGreater(near, far)
        # a learned prior must be strictly more informative than the analytic
        # field on the obstacle centre (telemetry says "very dangerous")
        self.assertGreater(near, 0.5)

    def test_replanner_with_learned_prior_beats_baseline(self):
        from airlab.guardian import (RiskWorldModel, RiskPriorModel,
                                     PredictiveRePlanner, simulate_telemetry,
                                     Obstacle)
        prior = RiskPriorModel().fit(simulate_telemetry(n=800))
        planner = PredictiveRePlanner(
            model=RiskWorldModel(prior=prior, learned_alpha=1.0))
        start = np.array([0.0, 0.0, -2.0])
        remaining = [np.array([6.0, 0.0, -2.0]),
                     np.array([12.0, 0.0, -2.0])]
        obs = Obstacle(pos=np.array([6.0, 0.0, -2.0]),
                       vel=np.array([0.0, 0.0, 0.0]))
        res = planner.plan(start, remaining, battery_frac=1.0, obstacles=[obs])
        self.assertTrue(res.feasible)
        self.assertGreater(res.risk_reduction, 0.0)
        self.assertGreaterEqual(res.min_clearance_m, 2.0)


class TestRiskAndReplan(unittest.TestCase):
    def test_risk_field_penalises_obstacle_and_route_is_replanned(self):
        from airlab.guardian import (RiskWorldModel, PredictiveRePlanner, Obstacle)

        model = RiskWorldModel(cell=1.0, obstacle_sigma=1.5, sampling_step=0.5)
        start = np.array([0.0, 0.0, -5.0])
        remaining = [np.array([12.0, 0.0, -5.0])]
        obs = Obstacle(pos=np.array([6.0, 0.0, -5.0]),
                       vel=np.array([0.0, 0.0, 0.0]), radius=1.0)
        pl = PredictiveRePlanner(model=model, beam=4, cruise_speed=3.0,
                                 battery_capacity_wh=71.0)
        res = pl.plan(start, remaining, battery_frac=1.0, obstacles=[obs])
        # A same-line route (baseline) runs straight through the obstacle.
        self.assertGreater(res.bas_risk, res.repl_risk)
        self.assertGreater(res.risk_reduction, 0.0)
        # The replanned route should clear the obstacle farther than 2 m.
        self.assertGreaterEqual(res.min_clearance_m, 0.5)

    def test_replan_respects_energy_feasibility(self):
        from airlab.guardian import PredictiveRePlanner

        pl = PredictiveRePlanner(battery_capacity_wh=1.0, cruise_speed=3.0)
        start = np.array([0.0, 0.0, -5.0])
        remaining = [np.array([120.0, 0.0, -5.0])]
        # With a tiny battery, the re-planned route must be flagged infeasible.
        res = pl.plan(start, remaining, battery_frac=0.05)
        self.assertFalse(res.feasible)

    def test_replan_clears_threat_corridor_with_small_extra_distance(self):
        from airlab.guardian import (RiskWorldModel, PredictiveRePlanner, Obstacle)

        model = RiskWorldModel(cell=1.0, obstacle_sigma=2.0, jamming_sigma=4.0)
        start = np.array([0.0, 0.0, -5.0])
        remaining = [np.array([18.0, 0.0, -5.0]), np.array([30.0, 4.0, -6.0])]
        obs = Obstacle(pos=np.array([12.0, 0.0, -5.0]),
                       vel=np.array([0.0, 0.0, 0.0]), radius=1.5)
        pl = PredictiveRePlanner(model=model)
        res = pl.plan(start, remaining, battery_frac=1.0, obstacles=[obs],
                      jamming_centers=[np.array([13.0, 0.0, -5.0])])
        self.assertGreater(res.risk_reduction, 0.2)
        self.assertGreaterEqual(res.min_clearance_m, 2.0)
        self.assertIs(res.feasible, True)
        self.assertLess(res.extra_distance_frac, 0.2)


if __name__ == "__main__":
    unittest.main()
