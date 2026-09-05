"""Unit tests for the Nexus-Predator defensive AI core."""

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
