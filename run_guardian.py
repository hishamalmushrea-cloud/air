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
                             GuardianBrain, NexusAirV2, Obstacle,
                             SubsystemHealth, HealthPrognosis, simulated_features)
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

    # Predictive maintenance (master prompt §35): subsystem health scores.
    rng = np.random.default_rng(7)
    health = SubsystemHealth(warmup_samples=20)
    prognosis = HealthPrognosis()
    # healthy for 60 steps, then force every subsystem to degrade.
    rows_health = []
    for k in range(80):
        feats = simulated_features(rng, k,
                                   battery_bad=(k >= 60),
                                   motor_bad=(k >= 60),
                                   thermal_bad=(k >= 60),
                                   vib_bad=(k >= 60))
        health.update(*feats)
        agg = prognosis.aggregate(health.scores())
        if k in (19, 59, 69, 79):
            for hs in health.scores():
                rows_health.append({"step": k, "subsystem": hs.subsystem,
                                    "score": round(hs.score, 3),
                                    "status": hs.status, "evidence": hs.evidence})
    _write("out/guardian/health.csv", rows_health)
    # final aggregate + whether guardian would abort
    final_scores = health.scores()
    final_agg = prognosis.aggregate(final_scores)
    final_dec = air.brain.decide(_initial(), np.array([1.0, 0.0, 0.0]),
                                 health_score=final_agg)
    print(f"[guardian][health] final_aggregate={final_agg:.3f} "
          f"trend={prognosis.trend:.4f}/s critical={[h.subsystem for h in final_scores if h.degraded]} "
          f"guardian_mode={final_dec.mode}")
    print(f"[guardian] wrote out/guardian/health.csv")

    # Guardian oracle inside the real mission controller (brief-25).
    # We build a real Simulator, put an obstacle on the straight mission path,
    # and let the PredictiveRePlanner swap the remaining waypoints for a
    # lower-risk corridor mid-flight.
    from airlab.simulator import Simulator, SimConfig
    cfg = SimConfig()
    cfg.duration = 14.0
    cfg.cruise_speed = 2.0
    cfg.waypoints = [(0, 0, 2), (12, 0, 2), (24, 0, 2)]
    cfg.guardian_replan = True
    cfg.guardian_replan_period_s = 2.0
    cfg.guardian_obstacles = [
        ([12.0, 0.0, -2.0], [0.0, 0.0, 0.0], 1.5),
    ]
    sim = Simulator(cfg)
    run = sim.run()
    bridge = sim.guardian_bridge
    events = getattr(bridge, "history", None).events if bridge else []
    modes = getattr(bridge, "history", None).modes if bridge else []
    _write("out/guardian/sim_bridge.csv", [{
        "applied": int(bool(bridge and bridge.applied)),
        "risk_reduction": round(float(events[-1]["risk_reduction"]) if events else 0.0, 4),
        "bas_risk": round(float(events[-1]["bas_risk"]) if events else 0.0, 4),
        "repl_risk": round(float(events[-1]["repl_risk"]) if events else 0.0, 4),
        "extra_frac": round(float(events[-1]["extra_frac"]) if events else 0.0, 4),
        "clearance_m": round(float(events[-1]["clearance_m"]) if events else 0.0, 3),
        "n_events": len(events),
        "n_land": int(sum(1 for m in run.mode if m == "LAND")),
    }])
    print(f"[guardian][sim_bridge] applied={bridge and bridge.applied} "
          f"events={len(events)} modes={modes} n_land={sum(1 for m in run.mode if m == 'LAND')}")
    if events:
        e = events[-1]
        print(f"[guardian][sim_bridge] routes around obstacle: risk "
              f"{e['bas_risk']:.3f}->{e['repl_risk']:.3f} (red {e['risk_reduction']:.3f}), "
              f"clearance {e['clearance_m']:.2f}m, extra {e['extra_frac']:.2%}")
    print(f"[guardian] wrote out/guardian/sim_bridge.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
