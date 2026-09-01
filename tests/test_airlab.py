"""Unit/smoke tests for the AIR Lab stack.

Run with::

    .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.math_utils import (euler_to_R, quat_from_euler, euler_from_quat,
                               quat_multiply, R_from_quat, wrap_pi)
from airlab.dynamics import Quadrotor
from airlab.ahrs import accel_to_level
from airlab.mission import WaypointMission
from airlab.control import FlightController
from airlab.simulator import Simulator, SimConfig
from airlab.metrics import evaluate_mission, safety_metrics
from airlab.scenarios import random_scenario, run_scenario
from airlab.energy import compute_energy
from airlab.landmarks import LandmarkField, LandmarkConsistency
from airlab.factorgraph import SlidingFactorGraph, build_keyframe

G = 9.80665


class TestMath(unittest.TestCase):
    def test_euler_quat_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            rpy = rng.uniform(-0.9, 0.9, 3)
            q = quat_from_euler(rpy)
            rpy2 = euler_from_quat(q)
            if abs(rpy2[2] - rpy[2]) > np.pi:
                rpy2[2] += 2 * np.pi * (1 if rpy[2] > 0 else -1)
            self.assertTrue(np.allclose(rpy, rpy2, atol=1e-6))

    def test_quat_matrix_agreement(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            rpy = rng.uniform(-1, 1, 3)
            self.assertTrue(np.allclose(euler_to_R(rpy), R_from_quat(quat_from_euler(rpy)), atol=1e-6))

    def test_quat_multiply_normalization(self):
        q1 = quat_from_euler(np.array([0.1, -0.2, 0.3]))
        q2 = quat_from_euler(np.array([-0.4, 0.5, 0.2]))
        q = quat_multiply(q1, q2)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)

    def test_wrap_pi(self):
        self.assertAlmostEqual(wrap_pi(np.pi - 0.1), np.pi - 0.1)
        self.assertAlmostEqual(wrap_pi(-np.pi - 0.1), np.pi - 0.1)


class TestDynamics(unittest.TestCase):
    def test_hover_does_not_collapse(self):
        v = Quadrotor(init_pos=[0, 0, -3])
        for _ in range(300):
            v.step(np.array([G, 0.0, 0.0, 0.0]), 0.01)
        self.assertAlmostEqual(v.pos[0], 0.0, places=3)
        self.assertAlmostEqual(v.pos[1], 0.0, places=3)
        self.assertAlmostEqual(v.altitude, 3.0, delta=0.15)
        self.assertEqual(v.altitude, -v.pos[2])

    def test_horizontal_force_direction(self):
        # Negative pitch should produce a positive (north) acceleration.
        v = Quadrotor(init_pos=[0, 0, -3])
        v.step(np.array([G, 0.0, -1.0, 0.0]), 0.01)
        self.assertGreater(v.a_ned[0], 0.0)
        # Positive roll should produce a positive (east) acceleration.
        v = Quadrotor(init_pos=[0, 0, -3])
        v.step(np.array([G, 1.0, 0.0, 0.0]), 0.01)
        self.assertGreater(v.a_ned[1], 0.0)

    def test_ground_friction_stops_skid(self):
        # A landed aircraft with horizontal momentum must decelerate on contact
        # (otherwise "safe landing" is a lie: it would skid forever).
        v = Quadrotor(init_pos=[0.0, 0.0, 0.0], init_vel=[5.0, 3.0, 0.0],
                      ground=0.0, ground_friction=5.0)
        for _ in range(100):  # 1 s
            v.step(np.array([G, 0.0, 0.0, 0.0]), 0.01)
        self.assertLess(np.linalg.norm(v.vel[:2]), 0.2)
        self.assertEqual(v.pos[2], 0.0)

    def test_ground_friction_zero_if_disabled(self):
        v = Quadrotor(init_pos=[0.0, 0.0, 0.0], init_vel=[5.0, 0.0, 0.0],
                      ground=0.0, ground_friction=0.0)
        v.step(np.array([G, 0.0, 0.0, 0.0]), 0.01)
        # No friction => horizontal velocity only decays by air drag (tiny).
        self.assertGreater(v.vel[0], 4.9)


class TestAHRS(unittest.TestCase):
    def test_leveling_sign(self):
        roll, pitch = accel_to_level(np.array([0.0, 0.0, -G]))
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)

        # Positive roll (east acceleration) => positive measured fy.
        roll, pitch = accel_to_level(np.array([0.0, 0.05, -G]))
        self.assertGreater(roll, 0.0)

        # Positive measured fx corresponds to negative pitch (north accel).
        roll, pitch = accel_to_level(np.array([-0.05, 0.0, -G]))
        self.assertGreater(pitch, 0.0)


class TestMission(unittest.TestCase):
    def test_velocity_decelerates_near_waypoint(self):
        m = WaypointMission([(0, 0, 2), (10, 0, 2)], speed=2.0, reach_radius=0.6)
        # far from the endpoint
        _, vfar, _ = m.desired(np.array([0.0, 0.0, -2.0]))
        # just before the endpoint
        _, vnear, _ = m.desired(np.array([9.7, 0.0, -2.0]))
        self.assertGreater(np.linalg.norm(vfar), np.linalg.norm(vnear))

    def test_hold_at_waypoint(self):
        m = WaypointMission([(0, 0, 2), (10, 0, 2)], speed=2.0)
        for _ in range(10):
            _, v, _ = m.desired(np.array([10.2, 0.0, -2.0]))
        self.assertLess(np.linalg.norm(v), 1e-9)


class TestController(unittest.TestCase):
    def test_north_command_requires_negative_pitch(self):
        c = FlightController()
        u = c.compute(np.zeros(3), np.zeros(3), np.zeros(3),
                      np.array([3.0, 0.0, -2.0]), yaw_ref=0.0)
        self.assertLess(u[2], 0.0)  # q_rate cmd < 0 => pitch down toward north


class TestE2E(unittest.TestCase):
    def _run(self, outage=None, duration=30.0):
        cfg = SimConfig()
        cfg.duration = duration
        cfg.gps_outage = outage
        sim = Simulator(cfg)
        return sim.run(record=True), cfg

    def test_baseline_stays_in_bounds(self):
        r, cfg = self._run(None)
        m = evaluate_mission(r.true_pos, r.est_pos, r.ref_pos,
                             r.ref_vel, r.est_vel, r.gps_available, cfg.dt)
        self.assertGreater(m["in_bounds_frac"], 0.95)
        self.assertLess(m["pos_rmse"], 3.0)
        self.assertEqual(m["ground_collision_frac"], 0.0)

    def test_gnss_outage_with_flow_is_stable(self):
        r, cfg = self._run((10.0, 20.0), duration=30.0)
        m = evaluate_mission(r.true_pos, r.est_pos, r.ref_pos,
                             r.ref_vel, r.est_vel, r.gps_available, cfg.dt)
        self.assertGreater(m["in_bounds_frac"], 0.9)
        self.assertEqual(m["ground_collision_frac"], 0.0)
        self.assertLess(m["pos_rmse"], 5.0)


class TestScenario(unittest.TestCase):
    def test_random_scenario_is_usable(self):
        rng = np.random.default_rng(99)
        s = random_scenario(rng, duration=8.0, index=7)
        self.assertGreaterEqual(len(s.waypoints), 3)
        self.assertEqual(s.name, "scen_0007")
        cfg = s.to_config()
        self.assertEqual(cfg.duration, 8.0)

    def test_run_scenario_returns_metrics(self):
        rng = np.random.default_rng(11)
        s = random_scenario(rng, duration=8.0, index=0)
        m, _ = run_scenario(s, record=False)
        self.assertIn("pos_rmse", m)
        self.assertIn("in_bounds_frac", m)
        self.assertIsInstance(float(m["pos_rmse"]), float)

    def test_safety_lands_on_corrupt_velocity_aiding(self):
        # Goal: when GPS drops AND the velocity-aiding source becomes corrupt,
        # the safety layer must react (hold/land) instead of blindly trusting
        # the wrong measurement until the vehicle crashes.
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.flow_bias_ramp = 0.25
        cfg.safety_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertGreater(m["safety_fraction"], 0.1)
        self.assertEqual(m["crash"], 0.0)
        self.assertIn(m["safety_outcome"], ("reactive_hold", "landed_safely"))

    def test_healthy_mission_does_not_trigger_needless_hold(self):
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.safety_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertEqual(m["safety_outcome"], "completed")
        self.assertEqual(m["safety_fraction"], 0.0)

    def test_landmark_consistency_detects_position_drift(self):
        field = LandmarkField()
        cons = LandmarkConsistency(field, residual_scale=0.12)
        pos = np.array([0.0, 0.0, -5.0])
        rpy = np.zeros(3)
        # ground-truth camera looking at the field from the true pose
        ids, dirs = field.observe(type("V", (), {
            "pos": pos, "R_nb": np.eye(3)})())
        # consistent: should score close to 1
        score_ok = cons.evaluate(ids, dirs, pos, rpy)
        # EKF position drifted 5 m: predicted/observed inter-landmark angles
        # should now disagree, so the score must drop.
        score_bad = cons.evaluate(ids, dirs, pos + np.array([5.0, 5.0, 0.0]), rpy)
        self.assertGreater(score_ok, 0.8)
        self.assertLess(score_bad, score_ok * 0.9)
        self.assertGreater(cons.residual, 0.0)

    def test_healthy_mission_with_landmark_detector_stays_completed(self):
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.safety_enabled = True
        cfg.landmark_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertEqual(m["safety_outcome"], "completed")
        self.assertEqual(m["safety_fraction"], 0.0)
        # And the landmark detector should stay reasonably healthy throughout.
        self.assertTrue(np.min(np.asarray(r.landmark_score)) > 0.8)

    def test_calibrated_factor_graph_no_false_alarm_healthy(self):
        # Live factor graph with startup calibration should not trigger
        # safety on a healthy mission.
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.flow_bias_ramp = 0.0
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = False
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertEqual(m["safety_outcome"], "completed")
        self.assertEqual(m["safety_fraction"], 0.0)
        self.assertGreater(np.mean(np.asarray(r.factorgraph_health)), 0.7)

    def test_independent_detector_lands_even_with_gps(self):
        # A corrupt velocity source can crash the aircraft even while an
        # absolute GNSS fix is still present, because the controller uses the
        # (wrong) estimated velocity.  An independent detector must therefore
        # be allowed to force a landing regardless of GPS availability.  This
        # was the exact gap that kept bias.25 crashing in the batch study.
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = None          # GPS stays available the whole time
        cfg.flow_bias_ramp = 0.25
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = False
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertIn(m["safety_outcome"], ("landed_safely", "reactive_hold"))
        self.assertEqual(m["crash"], 0.0)

    def test_calibrated_factor_graph_detects_large_bias(self):
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.flow_bias_ramp = 0.25
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = False
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        # With a large bias the calibrated factor graph must at least react
        # (reactive_hold / landed), and must reduce unintended ground contact
        # to near zero (it is honest to allow a tiny transient, unlike the
        # ~0.08 crash fraction when the detector is off entirely).
        self.assertGreater(m["safety_fraction"], 0.0)
        self.assertLess(m["crash"], 0.01)
        self.assertIn(m["safety_outcome"], ("reactive_hold", "landed_safely"))

    def test_consensus_policy_combine(self):
        cfg = SimConfig()
        sim = Simulator(cfg)
        parts = [0.2, 0.9]
        cfg.detector_consensus = "min"
        self.assertAlmostEqual(sim._combine_detectors(parts), 0.2)
        cfg.detector_consensus = "max"
        self.assertAlmostEqual(sim._combine_detectors(parts), 0.9)
        cfg.detector_consensus = "geom"
        self.assertAlmostEqual(sim._combine_detectors(parts), np.sqrt(0.18))

    def test_adaptive_consensus_escalates_only_when_degraded(self):
        cfg = SimConfig()
        sim = Simulator(cfg)
        cfg.detector_consensus = "adaptive"
        # Soft consensus still vouches for the sensors => return soft (no
        # needless aggressive landing on a mild, survivable fault).
        high = sim._combine_detectors([0.8, 0.9])
        self.assertGreater(high, 0.8)
        # A deep fault makes even the soft consensus fall below the warn line,
        # so we escalate to the worst-of opinion (decisive fail).
        deep = sim._combine_detectors([0.2, 0.9])
        self.assertLess(deep, 0.25)
        self.assertAlmostEqual(deep, 0.2)

    def test_rtl_requires_gps(self):
        # Return-to-base must NOT be attempted when there is no absolute fix:
        # without GPS the aircraft does not know where base is in world frame,
        # so RTL would be a guess.  It should land immediately instead.
        cfg = SimConfig()
        cfg.duration = 30.0
        cfg.gps_outage = (2.0, 30.0)
        cfg.flow_bias_ramp = 0.25
        cfg.mission_aware = True
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        self.assertNotIn("RTL", set(r.mode))
        m = safety_metrics(r, cfg.dt)
        self.assertIn(m["safety_outcome"], ("reactive_hold", "landed_safely"))
        self.assertEqual(m["crash"], 0.0)

    def test_factor_graph_detects_large_flow_bias(self):
        """A big ramping bias must drive the factor-graph flow residual up."""
        field = LandmarkField()
        pos = np.array([0.0, 0.0, -3.0])
        vel = np.array([1.5, 0.0, 0.0])
        rpy = np.zeros(3)

        def run(bias):
            graph = SlidingFactorGraph(field.positions, window=6, dt_keyframe=0.5)
            p = pos.copy()
            v = vel.copy()
            for _ in range(12):
                flow = v + np.array([bias, 0.0, 0.0])
                ids, dirs = field.observe(type("V", (), {
                    "pos": p, "R_nb": np.eye(3)})())
                kf = build_keyframe(p, v, np.zeros(3), rpy, 0.5,
                                    flow, p.copy(), v.copy(), ids, dirs)
                graph.push(kf)
                p = p + v * 0.5
            return graph.optimize()

        healthy = run(0.0)
        biased = run(0.8)
        self.assertGreater(biased["flow_residual"], healthy["flow_residual"])
        self.assertLess(biased["health"], healthy["health"])

    def test_energy_increases_with_thrust(self):
        from airlab.energy import PowerModel
        p = PowerModel()
        dt = 0.01
        # 1 s of control history
        thrust = np.full(100, 9.80665)
        e1 = p.energy(thrust, dt)
        thrust2 = np.full(100, 15.0)
        e2 = p.energy(thrust2, dt)
        self.assertGreater(e2, e1)
        self.assertGreater(e1, 0.0)


if __name__ == "__main__":
    unittest.main()
