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

        # Availability-weighted soft consensus: when the landmark detector has
        # no usable data it is down-weighted, so its (possibly misleading) low
        # score should not drag the consensus down as far as an unweighted geom.
        cfg.detector_consensus = "geom"
        sim._landmark_obs_count = 0        # no landmarks available
        sim._fg_flow_components = 0        # no factor-graph data either (full under-determination)
        self.assertGreater(sim._combine_detectors([0.9, 0.2]), 0.0)
        sim._landmark_obs_count = 3        # landmark detector has good data now
        sim._fg_flow_components = 0        # but the graph is thin
        weighted_low_info = sim._combine_detectors([0.9, 0.2])
        # A weighted consensus should be >= the unweighted geometric mean when
        # one detector is under-informed, so a thin detector cannot force a
        # hard reaction on weak evidence.
        self.assertGreater(weighted_low_info, np.sqrt(0.18))

    def test_weighted_consensus_ignores_thin_detector_when_healthy(self):
        # Guardrail: with the landmark detector feature-poor (outage) and a
        # healthy factor graph, the availability-weighted consensus must NOT be
        # dragged down by the stale/unknown landmark opinion.  The thin
        # detector should carry a weak voice, not force a hard reaction.
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.landmark_outage = (10.0, 35.0)
        cfg.detector_consensus = "adaptive_weighted"
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertEqual(m["safety_outcome"], "completed")
        self.assertEqual(m["safety_fraction"], 0.0)

    def test_frame_trust_discriminates_spread_vs_clustered(self):
        # Two cameras both report 3 landmarks, but one sees a wide angular
        # spread and the other a tight cluster.  The trust model must give the
        # spread one a much stronger voice (geometric leverage, not raw count).
        cfg = SimConfig()
        sim = Simulator(cfg)
        # spread: unit dirs in well-separated directions
        spread = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -0.2]])
        spread = spread / np.linalg.norm(spread, axis=1, keepdims=True)
        sim._landmark_dirs = spread
        sim._landmark_obs_count = 3
        trust_spread = sim._detector_trust("landmark")
        # clustered: near-identical directions
        clustered = np.array([[1.0, 0.0, -0.2], [1.0, 0.02, -0.2], [1.0, -0.02, -0.2]])
        clustered = clustered / np.linalg.norm(clustered, axis=1, keepdims=True)
        sim._landmark_dirs = clustered
        sim._landmark_obs_count = 3
        trust_clustered = sim._detector_trust("landmark")
        self.assertGreater(trust_spread, 0.5)
        self.assertLess(trust_clustered, trust_spread * 0.5)

    def test_landmark_cluster_keeps_count_but_collapses_trust(self):
        # Degenerate-parallax window: rank count stays high (many landmarks)
        # while angular-diversity trust collapses, so a camera with many data
        # points in a tight cone is treated as a thin voice by the trust model.
        from airlab.simulator import _cone_bearing
        cfg = SimConfig()
        sim = Simulator(cfg)
        ids = np.arange(4)
        ax = np.array([0.0, 0.0, 1.0])
        rng = np.random.default_rng(5)
        pert = rng.uniform(-0.04, 0.04, size=(len(ids), 2))
        dirs = np.array([_cone_bearing(ax, p[0], p[1]) for p in pert])
        sim._landmark_dirs = dirs
        sim._landmark_obs_count = len(ids)
        trust_cluster = sim._detector_trust("landmark")
        self.assertGreaterEqual(sim._detector_weight("landmark"), 0.5)  # count says credible
        self.assertLess(trust_cluster, 0.35)                            # geometry says thin
        # Same count, spread directions -> high trust (sanity).
        spread = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -0.2],
                           [0.3, -0.4, 0.85]])
        spread = spread / np.linalg.norm(spread, axis=1, keepdims=True)
        sim._landmark_dirs = spread
        sim._landmark_obs_count = 4
        self.assertGreater(sim._detector_trust("landmark"), 0.7)

    def test_frame_trust_learner_calibrates_from_startup_and_detects_cluster(self):
        # Feed a healthy startup distribution (well-spread ~1.0 rad) then score
        # a degenerate frame (large count, tight cone ~0.04 rad).  The learned
        # model must give the healthy frame high trust and the cluster low.
        from airlab.trust import FrameTrustLearner
        lm = FrameTrustLearner("landmark")
        rng = np.random.default_rng(7)
        for _ in range(20):
            lm.calibrate(float(rng.normal(1.0, 0.05)), 5)
        self.assertTrue(lm.calibrated)
        self.assertGreater(lm.trust(1.05, 5), 0.7)      # healthy frame
        self.assertLess(lm.trust(0.04, 6), 0.35)        # degenerate but many
        # count reference should be learned, not hard-coded /3.
        self.assertAlmostEqual(lm.count_ref, 5.0, places=1)

    def test_sparse_factorgraph_is_clean_healthy_and_detects_bias(self):
        # A sparse/under-determined factor graph (no flow factors for the FG
        # only, EKF flow aiding unchanged) must not false-alarm a healthy
        # mission, and must still allow detection of a real fault once the
        # graph recovers.
        from dataclasses import replace
        from airlab.scenarios import random_scenario

        rng = np.random.default_rng(909)
        random_scenario(rng, duration=40.0, index=0)
        s = random_scenario(rng, duration=40.0, index=1)

        # healthy + sparse graph
        s_h = replace(s, factorgraph_flow_outage=(8.0, 28.0),
                      detector_consensus="adaptive_veto_trust")
        m, _ = run_scenario(s_h, record=True)
        self.assertEqual(m["safety_outcome"], "completed")
        self.assertEqual(m["crash"], 0.0)
        self.assertEqual(m["landed"], 0.0)

        # persistent bias.25 + sparse graph: still detected after recovery
        s_f = replace(s, flow_bias_ramp=0.25, factorgraph_flow_outage=(8.0, 28.0),
                      detector_consensus="adaptive_veto_trust")
        m_f, _ = run_scenario(s_f, record=True)
        self.assertEqual(m_f["crash"], 0.0)
        self.assertGreater(m_f["landed"], 0.4)

    def test_transient_in_sparse_fg_does_not_crash_under_trust(self):
        # A transient velocity-bias fault fully contained inside a sparse-FG
        # outage (the FG has no flow factors while the fault exists) is invisible
        # to the factor graph.  A landmark-only "worst-of" arm reacts to this
        # self-recovering fault by rejecting the recovering flow source and can
        # crash; adaptive_veto_trust is conservative (does not false-reject a
        # recovering source) and survives.
        from dataclasses import replace
        from airlab.scenarios import random_scenario

        rng = np.random.default_rng(909)
        base = [random_scenario(rng, duration=40.0, index=k) for k in range(3)]
        s = base[0]

        fault = dict(flow_bias_shift=3.0, flow_bias_window=(8.0, 18.0),
                     factorgraph_flow_outage=(5.0, 28.0), safety_enabled=True)

        # worst-of landmark-only: reacts to the transient & crashes scen0
        s_lm = replace(s, **fault, landmark_enabled=True, factorgraph_enabled=False,
                       detector_consensus="min")
        m_lm, _ = run_scenario(s_lm, record=True)
        # trust arm: conservative, does not over-react to a recovering source
        s_tr = replace(s, **fault, landmark_enabled=True, factorgraph_enabled=True,
                       detector_consensus="adaptive_veto_trust")
        m_tr, _ = run_scenario(s_tr, record=True)
        self.assertGreater(m_lm["crash"], 0.0)
        self.assertEqual(m_tr["crash"], 0.0)

    def test_flow_reject_persist_gate_is_a_tradeoff_not_a_win(self):
        # The persistence gate is exposed as a tunable knob.  On the transient
        # fault the immediate default (0.0) rejects early and can crash a
        # landmark-only run; gating it to 2.0 avoids that crash (the warn only
        # lasted ~0.2 s).  This asserts the knob works and is honest that it is
        # a tradeoff, not a strict win.
        from dataclasses import replace
        from airlab.scenarios import random_scenario

        rng = np.random.default_rng(909)
        base = [random_scenario(rng, duration=40.0, index=k) for k in range(3)]
        s = base[0]

        def run_transient(persist):
            kw = dict(flow_bias_shift=3.0, flow_bias_window=(8.0, 18.0),
                      factorgraph_flow_outage=(5.0, 28.0),
                      landmark_enabled=True, factorgraph_enabled=False,
                      detector_consensus="min", safety_enabled=True,
                      flow_reject_persist_s=persist)
            sc = replace(s, **kw)
            m, _ = run_scenario(sc, record=True)
            return m["crash"], m["landed"]

        c_imm, _ = run_transient(0.0)
        c_gate, _ = run_transient(2.0)
        self.assertGreater(c_imm, 0.0)   # immediate rejects the transient too early
        self.assertEqual(c_gate, 0.0)    # gate lets the ~0.2 s warn recover

    def test_trust_veto_does_not_false_land_on_degenerate_parallax(self):
        # A camera that reports *many* features in a tiny angular cone (close
        # wall / low parallax) has high count but low angular-diversity trust.
        # The count-based veto treats it as credible and false-lands a healthy
        # mission; the trust veto recognises the geometry is degenerate and
        # does not.  This is the discriminating case research-brief-16.
        from dataclasses import replace
        import numpy as np
        from airlab.scenarios import random_scenario

        rng = np.random.default_rng(909)
        # consume the first two random scenarios, use the second which has a
        # GPS outage that exposes the velocity-aiding signal path
        random_scenario(rng, duration=40.0, index=0)
        s = random_scenario(rng, duration=40.0, index=1)

        for policy, expect_landed in (("adaptive_veto", 1.0),
                                      ("adaptive_veto_trust", 0.0)):
            s2 = replace(s, detector_consensus=policy,
                         landmark_cluster=(6.0, 22.0))
            m, _ = run_scenario(s2, record=True)
            self.assertAlmostEqual(m["crash"], 0.0, places=3)
            if expect_landed > 0:
                self.assertGreater(m["landed"], 0.4)     # false landing exposed
            else:
                self.assertEqual(m["landed"], 0.0)       # trust veto stays quiet

    def test_adaptive_veto_keeps_healthy_veto_during_outage(self):
        # A feature-poor camera keeps its last healthy score.  A symmetric
        # availability weight would down-weight it and let a noisier factor
        # graph false-land a healthy mission.  adaptive_veto must NOT do that:
        # thin data may not trigger, but the healthy (stale) detector keeps
        # its veto.
        cfg = SimConfig()
        sim = Simulator(cfg)
        cfg.detector_consensus = "adaptive_veto"
        sim._landmark_obs_count = 0        # landmark feature-poor
        sim._fg_flow_components = 4        # factor graph well-determined
        sim._fg_converged = True
        # landmark stale-high, factor graph dipped below warn but not fail.
        consensus = sim._combine_detectors([0.95, 0.50])
        # The healthy landmark must veto; do not escalate to warn-level hold.
        self.assertGreater(consensus, 0.65)
        # But a credible (well-data) low factor graph DOES escalate.
        low = sim._combine_detectors([0.95, 0.20])
        self.assertLess(low, 0.30)

    def test_adaptive_veto_clean_on_healthy_landmark_outage(self):
        # The whole reason adaptive_veto exists: under a feature-poor camera
        # following a healthy opinion, the detector must NOT let a modest
        # factor-graph dip escalate into a false landing.
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.landmark_outage = (10.0, 35.0)
        cfg.detector_consensus = "adaptive_veto"
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertEqual(m["safety_outcome"], "completed")
        self.assertEqual(m["safety_fraction"], 0.0)

    def test_adaptive_veto_still_detects_under_outage(self):
        # And it must NOT lose detection on a real fault during the outage.
        cfg = SimConfig()
        cfg.duration = 40.0
        cfg.gps_outage = (12.0, 24.0)
        cfg.landmark_outage = (10.0, 35.0)
        cfg.flow_bias_ramp = 0.25
        cfg.detector_consensus = "adaptive_veto"
        cfg.factorgraph_enabled = True
        cfg.landmark_enabled = True
        sim = Simulator(cfg)
        r = sim.run(record=True)
        m = safety_metrics(r, cfg.dt)
        self.assertIn(m["safety_outcome"], ("reactive_hold", "landed_safely"))
        self.assertEqual(m["crash"], 0.0)

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
