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


if __name__ == "__main__":
    unittest.main()
