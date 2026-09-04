#!/usr/bin/env python3
"""Nexus-Predator defensive AI behaviour lab.

Runs the GuardianBrain across five kinematic scenarios:
  healthy         - no threat, CRUISE_SAFE expected
  intruder        - a fast non-cooperative obstacle, EVADE expected
  spoof           - GPS position/course inconsistent with IMU/mag, RECOVER_NAV
  jamming         - GPS gone, IMU/baro healthy, RF-SILENT defensive mode
  jamming_obstacle- GPS gone + obstacle, EVADE + silent_rf + cloak

Defensive only: no weapons, no targeting.  This is a behaviour twin, not full
rigid-body dynamics (that lives in simulator.py).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

from airlab.guardian import (GuardianState, ThreatEngine, EvasionPlanner,
                             GuardianBrain, NexusAirV2, Obstacle)
_DESIRE = np.array([1.0, 0.0, 0.0])


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _initial() -> GuardianState:
    return GuardianState(
        pos=np.array([0.0, 0.0, -5.0]), vel=np.array([1.0, 0.0, 0.0]),
        a_cmd=np.zeros(3), gps_pos=np.array([0.05, 0.0, -5.0]),
        gps_vel=np.array([1.0, 0.0, 0.0]), imu_dr_pos=np.array([0.0, 0.0, -5.0]),
        imu_dr_vel=np.array([1.0, 0.0, 0.0]), mag_heading=0.0, gps_course=0.0,
        gps_signal_quality=1.0, baro_ok=True, battery_frac=1.0,
        energy_required_frac=0.2, wind_est=np.zeros(3), obstacles=[],
    )


def _obstacle_scenario(speed=1.0, start_x=5.0, radius=0.8):
    def fn(k, pos):
        t = k * 0.05
        return Obstacle(pos=np.array([start_x - speed * t, 0.0, -5.0]),
                        vel=np.array([-speed, 0.0, 0.0]), radius=radius)
    return fn


def _spoof_mod(k, st: GuardianState) -> GuardianState:
    # GPS is progressively pulled off course; IMU DR + mag stay healthy.
    st.gps_pos = st.pos + np.array([0.0, 0.015 * k, 0.0])
    st.gps_course = float(k * 0.05)
    st.mag_heading = 0.0
    st.gps_signal_quality = 0.9
    return st


def _jam_mod(k, st: GuardianState) -> GuardianState:
    st.gps_pos = None
    st.gps_vel = None
    st.gps_signal_quality = 0.05
    st.baro_ok = True
    return st


def _row_replan(name: str, res) -> dict:
    return {
        "scenario": name,
        "bas_risk": round(res.bas_risk, 3),
        "repl_risk": round(res.repl_risk, 3),
        "risk_reduction": round(res.risk_reduction, 3),
        "bas_length_m": round(res.bas_length, 2),
        "repl_length_m": round(res.repl_length, 2),
        "extra_distance_frac": round(res.extra_distance_frac, 3),
        "energy_required": round(res.energy_heavy_required, 3),
        "feasible": int(res.feasible),
        "min_clearance_m": round(res.min_clearance_m, 2),
        "reasons": str(res.reasons),
    }


def _row(name: str, m) -> dict:
    return {
        "scenario": name, "mode_counts": str(m.mode_histogram),
        "max_threat": round(m.max_threat_reached, 3),
        "final_clearance_m": round(m.final_clearance, 3),
        "crashed": int(m.crashed),
        "declared_used": str(sorted(m.declared_used)),
        "undeclared_used": str(sorted(m.undeclared_used)),
        "mean_compute_ms": round(m.mean_compute_ms, 2),
        "mean_energy_mw": round(m.mean_energy_mw, 1),
    }


def main() -> int:
    os.makedirs("out/guardian", exist_ok=True)
    air = NexusAirV2(GuardianBrain(ThreatEngine(), EvasionPlanner()))

    scenarios = [
        ("healthy", air.simulate(_initial(), _DESIRE, n_steps=20)),
        ("intruder", air.simulate(_initial(), _DESIRE, n_steps=140,
                                  obstacle_fn=_obstacle_scenario())),
        ("spoof", air.simulate(_initial(), _DESIRE, n_steps=40,
                               state_mod=_spoof_mod)),
        ("jamming", air.simulate(_initial(), _DESIRE, n_steps=40,
                                 state_mod=_jam_mod)),
        ("jamming_obstacle", air.simulate(_initial(), _DESIRE, n_steps=140,
                                          obstacle_fn=_obstacle_scenario(),
                                          state_mod=_jam_mod)),
    ]
    rows = []
    for name, m in scenarios:
        rows.append(_row(name, m))
        print(f"[guardian] {name:18s} modes={m.mode_histogram} "
              f"max_threat={m.max_threat_reached:.2f} "
              f"clearance={m.final_clearance:.2f}m crash={int(m.crashed)} "
              f"declared={sorted(m.declared_used)} "
              f"undeclared={sorted(m.undeclared_used)}")
    _write("out/guardian/summary.csv", rows)
    print(f"[guardian] wrote out/guardian/summary.csv")

    # Oracle risk world model + predictive re-planning demonstration.
    st = _initial()
    wp0 = np.array([18.0, 0.0, -5.0])
    wp1 = np.array([30.0, 4.0, -6.0])
    remaining = [wp0, wp1]
    obs_a = Obstacle(pos=np.array([12.0, 0.0, -5.0]), vel=np.array([0.0, 0.0, 0.0]),
                     radius=1.5)
    jam_center = np.array([13.0, 0.0, -5.0])
    res = air.replan_route(st, remaining, obstacles=[obs_a],
                           jamming_centers=[jam_center])
    repl_rows = [_row_replan("oracle_replan", res)]
    _write("out/guardian/replan.csv", repl_rows)
    print(f"[guardian][replan] bas_risk={res.bas_risk:.3f} -> repl_risk={res.repl_risk:.3f} "
          f"(reduction={res.risk_reduction:.3f}, extra_frac={res.extra_distance_frac:.3f}, "
          f"clearance={res.min_clearance_m:.2f}m, energy_req={res.energy_heavy_required:.3f}, "
          f"feasible={int(res.feasible)})")
    print(f"[guardian] wrote out/guardian/replan.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
