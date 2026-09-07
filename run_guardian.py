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

    # Learned risk prior (priority #3): fit a transparent kernel-regression
    # risk map on *labelled telemetry*, then show it calibrates the severity of
    # the same obstacle/jam corridor (C -> B: analytic surrogate -> learned).
    from airlab.guardian import (RiskPriorModel, RiskWorldModel,
                                 PredictiveRePlanner, simulate_telemetry,
                                 Obstacle as PriorObstacle)
    prior = RiskPriorModel().fit(simulate_telemetry(n=1200))
    learned_model = RiskWorldModel(prior=prior, learned_alpha=1.0)
    prior_obs = PriorObstacle(pos=np.array([12.0, 0.0, -5.0]),
                              vel=np.array([0.0, 0.0, 0.0]))
    prior_center = np.array([13.0, 0.0, -5.0])
    prior_field = learned_model.build((0, 24, -8, 8, -8, -2),
                                      [prior_obs], [prior_center])
    at_obs = prior_field.sample(np.array([12.0, 0.0, -5.0]))
    side = prior_field.sample(np.array([12.0, 6.0, -5.0]))
    base_model = RiskWorldModel()
    base_field = base_model.build((0, 24, -8, 8, -8, -2),
                                  [prior_obs], [prior_center])
    base_at = base_field.sample(np.array([12.0, 0.0, -5.0]))
    learned_res = PredictiveRePlanner(model=learned_model).plan(
        np.array([0.0, 0.0, -5.0]),
        [np.array([12.0, 0.0, -5.0]), np.array([24.0, 0.0, -5.0])],
        battery_frac=1.0, obstacles=[prior_obs],
        jamming_centers=[prior_center])
    prior_rows = [{
        "field": "learned_at_obstacle", "risk": round(float(at_obs), 4),
        "note": f"prior={prior.summary().get('n', 0)} samples",
    }, {
        "field": "learned_side", "risk": round(float(side), 4),
    }, {
        "field": "analytic_at_obstacle", "risk": round(float(base_at), 4),
    }, {
        "field": "replan_risk", "risk": round(float(learned_res.repl_risk), 4),
        "note": f"reduction={learned_res.risk_reduction:.3f} "
                f"clearance={learned_res.min_clearance_m:.2f}m "
                f"extra={learned_res.extra_distance_frac:.2f}",
    }]
    _write("out/guardian/risk_prior.csv", prior_rows)
    print(f"[guardian][risk_prior] prior={prior.summary()}")
    print(f"[guardian][risk_prior] learned at obstacle={at_obs:.3f} "
          f"(analytic={base_at:.3f}), side={side:.3f}")
    print(f"[guardian][risk_prior] replan learned risk "
          f"{learned_res.bas_risk:.3f}->{learned_res.repl_risk:.3f} "
          f"reduction={learned_res.risk_reduction:.3f} "
          f"clearance={learned_res.min_clearance_m:.2f}m")
    print(f"[guardian] wrote out/guardian/risk_prior.csv")

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

    # Health bridge on real fused telemetry (priority #2).  We fly the full
    # stack twice: once healthy, once with a degrading motor (physical
    # max-thrust shortfall), and show the health engine separates them from
    # real throttle/accel/battery/thermal/vib/consensus telemetry.
    def _run_health(motor_degrade_at=None):
        cfg = SimConfig()
        cfg.duration = 14.0
        cfg.cruise_speed = 2.0
        cfg.motor_degrade_at = motor_degrade_at
        cfg.motor_degrade_eff = 0.7
        cfg.guardian_health_enabled = True
        sim = Simulator(cfg)
        sim.run()
        return sim

    healthy_sim = _run_health()
    degraded_sim = _run_health(motor_degrade_at=7.0)
    hb = degraded_sim.guardian_health_bridge
    hs = {h.subsystem: (h.score, h.status) for h in hb.health.scores()}
    hh = {h.subsystem: (h.score, h.status)
          for h in healthy_sim.guardian_health_bridge.health.scores()}
    rows_h2 = []
    for name in hh:
        rows_h2.append({"run": "healthy", "subsystem": name,
                        "score": round(float(hh[name][0]), 3),
                        "status": hh[name][1]})
        rows_h2.append({"run": "degraded", "subsystem": name,
                        "score": round(float(hs[name][0]), 3),
                        "status": hs[name][1]})
    _write("out/guardian/telemetry_health.csv", rows_h2)
    print("[guardian][telemetry_health] healthy = "
          + ", ".join(f"{k}={v[0]:.2f}({v[1]})" for k, v in hh.items()))
    print("[guardian][telemetry_health] degraded = "
          + ", ".join(f"{k}={v[0]:.2f}({v[1]})" for k, v in hs.items()))
    print(f"[guardian][telemetry_health] final_agg healthy="
          f"{healthy_sim.guardian_health_bridge.prognosis.history[-1]:.3f} "
          f"degraded={hb.prognosis.history[-1]:.3f}")
    print(f"[guardian] wrote out/guardian/telemetry_health.csv")

    # Part-level low-watt thermal model (priority #4).  Fly the same stack
    # twice: baseline edge load (0.30) vs full edge/NPU load (1.0), and show
    # which *part* heats first (cpu_npu, esc, motor, battery) rather than a
    # single lumped temperature.
    def _run_thermal(compute_frac, ambient_c=25.0):
        cfg = SimConfig()
        cfg.duration = 120.0
        cfg.cruise_speed = 2.0
        cfg.compute_frac = compute_frac
        cfg.thermal_ambient_c = ambient_c
        cfg.guardian_health_enabled = True
        sim = Simulator(cfg)
        sim.run()
        return sim

    low = _run_thermal(0.30, ambient_c=25.0)
    high = _run_thermal(1.0, ambient_c=45.0)
    lo_t = low.guardian_health_bridge.thermal
    hi_t = high.guardian_health_bridge.thermal
    rows_thermal = []
    for name in lo_t.temperatures():
        rows_thermal.append({
            "run": "baseline", "node": name,
            "temp_c": round(float(lo_t.temperatures()[name]), 2),
            "status": lo_t.status()[name],
        })
        rows_thermal.append({
            "run": "full_load", "node": name,
            "temp_c": round(float(hi_t.temperatures()[name]), 2),
            "status": hi_t.status()[name],
        })
    _write("out/guardian/thermal.csv", rows_thermal)
    print(f"[guardian][thermal] baseline = {lo_t.summary()}")
    print(f"[guardian][thermal] full_load = {hi_t.summary()}")
    print(f"[guardian] wrote out/guardian/thermal.csv")

    # Telemetry data pipeline (priority #5): record a real flight into a
    # reproducible dataset, then fit the learned risk prior on it.
    from airlab.guardian import RiskPriorModel
    cfg = SimConfig()
    cfg.duration = 5.0
    cfg.cruise_speed = 2.0
    cfg.guardian_data_pipeline = True
    cfg.guardian_health_enabled = True
    cfg.guardian_data_obstacles = [
        ([4.0, 0.0, -2.0], [0.0, 0.0, 0.0], 1.5),
    ]
    cfg.guardian_data_jamming = [[1.0, 0.0, -2.0]]
    sim = Simulator(cfg)
    sim.run()
    pipe = sim.guardian_pipeline
    pipe.dataset.to_csv("out/guardian/telemetry.csv")
    prior = pipe.risk.fit_prior(RiskPriorModel())
    arr = pipe.risk.as_array()
    near = float(prior.predict(np.array([0.5]), np.array([0.0]))[0])
    far = float(prior.predict(np.array([10.0]), np.array([0.0]))[0])
    # query a point inside the recorded jamming corridor that is also near the
    # obstacle (dist ~3 m from the obstacle, jamming ~0.9)
    jammed = float(prior.predict(np.array([3.0]), np.array([0.9]))[0])
    jam_train = float(np.max(arr[:, 1])) if len(arr) else 0.0
    _write("out/guardian/data_pipeline.csv", [{
        "dataset_rows": len(pipe.dataset),
        "risk_samples": len(pipe.risk),
        "prior_n": int(prior.summary().get("n", 0)),
        "prior_near": round(near, 4),
        "prior_far": round(far, 4),
        "prior_jammed": round(jammed, 4),
        "train_jam_max": round(jam_train, 4),
    }])
    print(f"[guardian][pipeline] recorded {len(pipe.dataset)} rows, "
          f"{len(pipe.risk)} risk samples -> out/guardian/telemetry.csv")
    print(f"[guardian][pipeline] prior fitted n={prior.summary().get('n', 0)} "
          f"near={near:.3f} far={far:.3f} jammed={jammed:.3f} "
          f"(train_jam_max={jam_train:.3f})")
    print(f"[guardian] wrote out/guardian/data_pipeline.csv")

    # Thermal-aware mission budget (brief-30): the same re-planner now checks
    # that a route stays inside the part-level thermal envelope, too.  Cool
    # ambient accepts; desert-hot ambient rejects a route that would cook the
    # edge/NPU / battery before it lands.
    from airlab.guardian import PredictiveRePlanner as ThermPlanner
    start_pt = np.array([0.0, 0.0, -2.0])
    rem_pt = [np.array([12.0, 0.0, -2.0])]
    cool = ThermPlanner(thermal_aware=True, thermal_ambient_c=25.0,
                        cruise_speed=3.0, hover_power_w=112.0,
                        battery_capacity_wh=71.0).plan(
        start_pt, rem_pt, battery_frac=1.0)
    hot = ThermPlanner(thermal_aware=True, thermal_ambient_c=90.0,
                       cruise_speed=3.0, hover_power_w=112.0,
                       battery_capacity_wh=71.0).plan(
        start_pt, rem_pt, battery_frac=1.0)
    _write("out/guardian/thermal_budget.csv", [{
        "case": "cool_25C", "thermal_feasible": int(cool.thermal_feasible),
        "feasible": int(cool.feasible), "worst_node": cool.thermal_worst_node,
        "max_temp_c": round(float(cool.thermal_max_c), 2),
        "margin_c": round(float(cool.thermal_margin_c), 2),
    }, {
        "case": "hot_90C", "thermal_feasible": int(hot.thermal_feasible),
        "feasible": int(hot.feasible), "worst_node": hot.thermal_worst_node,
        "max_temp_c": round(float(hot.thermal_max_c), 2),
        "margin_c": round(float(hot.thermal_margin_c), 2),
    }])
    print(f"[guardian][thermal_budget] cool=feasible={cool.feasible} "
          f"worst={cool.thermal_worst_node} max={cool.thermal_max_c:.1f}C "
          f"margin={cool.thermal_margin_c:.1f}C")
    print(f"[guardian][thermal_budget] hot=feasible={hot.feasible} "
          f"worst={hot.thermal_worst_node} max={hot.thermal_max_c:.1f}C "
          f"margin={hot.thermal_margin_c:.1f}C")
    print(f"[guardian] wrote out/guardian/thermal_budget.csv")

    # Real PX4/ROS log reader (priority #6).  No real .ulg/.bag flight log is
    # shipped in the repo, so this demo reads a *simulated* PX4-schema fixture
    # and honestly shows: (a) an unlabelled log cannot fit the risk prior, and
    # (b) a labelled fixture (still simulated) can.
    from airlab.guardian import (Px4RosLogReader, write_test_fixture,
                                 RiskPriorModel as LogPriorModel,
                                 FIXTURE_LABEL)
    plain = write_test_fixture("out/guardian/px4_fixture.csv", n=120)
    labelled = write_test_fixture("out/guardian/px4_fixture_labeled.csv",
                                  n=120, with_risk_label=True)
    r_plain = Px4RosLogReader().load(plain)
    r_labelled = Px4RosLogReader().load(labelled)
    prior_plain = r_plain.fit_prior(LogPriorModel())
    prior_labelled = r_labelled.fit_prior(LogPriorModel())
    near = float(prior_labelled.predict(np.array([0.5]), np.array([0.0]))[0])
    far = float(prior_labelled.predict(np.array([10.0]), np.array([0.0]))[0])
    late_jam = float(np.mean(r_labelled.risk.as_array()[-20:, 1]))
    early_jam = float(np.mean(r_labelled.risk.as_array()[:20, 1]))
    _write("out/guardian/log_reader.csv", [{
        "source": FIXTURE_LABEL,
        "telemetry_rows": len(r_plain.telemetry),
        "risk_samples": r_labelled.risk.n,
        "unlabelled_prior_fitted": int(prior_plain.fitted),
        "labelled_prior_fitted": int(prior_labelled.fitted),
        "near_risk": round(near, 4),
        "far_risk": round(far, 4),
        "early_jam": round(early_jam, 4),
        "late_jam": round(late_jam, 4),
    }])
    print(f"[guardian][log_reader] source={FIXTURE_LABEL}")
    print(f"[guardian][log_reader] read {len(r_plain.telemetry)} telemetry "
          f"rows, {r_labelled.risk.n} risk samples")
    print(f"[guardian][log_reader] unlabelled_prior_fitted="
          f"{prior_plain.fitted} labelled_prior_fitted={prior_labelled.fitted}")
    print(f"[guardian][log_reader] jam late={late_jam:.3f} > early="
          f"{early_jam:.3f}; prior near={near:.3f} far={far:.3f}")
    print(f"[guardian] wrote out/guardian/log_reader.csv")

    # Low-watt edge perception path (priority #7).  A transparent spiking-style
    # front-end converts a sparse depth-return cloud into obstacles, then feeds
    # them to the *real* mission bridge (sensed, not scripted).  This is a
    # **simulated/estimated** stand-in for SNN/NeuViT silicon, with declared
    # efficiency numbers exposed.
    from airlab.guardian import (SpikeVision, PerceptionToGuardian,
                                 PerceptionConfig, MissionReplanBridge)
    from airlab.mission import WaypointMission as PercepMission
    cfg2 = PerceptionConfig(sensor_range_m=9.0, spike_voltage_gate_m=0.5,
                            min_points=3)
    vision = SpikeVision(cfg2)
    # depth-return cloud: two objects ahead of the aircraft
    pts = np.array([
        # object A near the flight path (will be sensed + avoided)
        [3.0, 0.0, -2.0], [3.2, 0.0, -2.0], [3.1, 0.2, -2.0],
        # object B far enough to be ignored (too sparse)
        [7.0, 5.0, -2.0], [7.1, 5.0, -2.0],
    ])
    percep = vision.process(pts, dt=0.1)
    sensed = percep.obstacles
    # real mission (3 waypoints) with the *sensed* obstacles
    p_mission = PercepMission([(0, 0, 2), (6, 0, 2), (12, 0, 2)], speed=2.0)
    from airlab.guardian import BridgeConfig as PercepBridgeConfig
    p_bridge = MissionReplanBridge(
        p_mission, np.array([0.0, 0.0, -2.0]), obstacles=sensed,
        config=PercepBridgeConfig(max_extra_distance_frac=0.80,
                                  min_clearance_m=2.0))
    p_res = p_bridge.try_replan(0.0, force=True)
    _write("out/guardian/perception.csv", [{
        "spike_count": percep.spike_count,
        "cluster_count": percep.cluster_count,
        "detected_obs": len(sensed),
        "edge_power_w": round(float(percep.energy_w), 3),
        "gops_per_w": round(float(percep.gops_per_w), 1),
        "intelligence_per_watt": round(float(percep.intelligence_per_watt), 3),
        "bridge_applied": int(bool(p_bridge.applied)),
        "risk_reduction": round(float(p_res.risk_reduction) if p_res else 0.0, 4),
        "clearance_m": round(float(p_res.min_clearance_m) if p_res else 0.0, 3),
    }])
    print(f"[guardian][perception] detected={len(sensed)} clusters "
          f"spikes={percep.spike_count} power={percep.energy_w:.2f}W "
          f"iperf={percep.intelligence_per_watt:.2f} Hz/W "
          f"gops/W={percep.gops_per_w:.0f}")
    print(f"[guardian][perception] bridge applied={p_bridge.applied} "
          f"risk_reduction={p_res.risk_reduction if p_res else 0.0:.3f} "
          f"clearance={p_res.min_clearance_m if p_res else 0.0:.2f}m")
    print(f"[guardian] wrote out/guardian/perception.csv")

    # Event-camera / RGB motion path (priority #9).  A transparent event-
    # camera surrogate accumulates sparse (x,y,polarity)+ events into a low-res
    # motion occupancy histogram, thresholds, and converts active cells to
    # obstacles.  It is a compliant, lighter path than depth and shares the
    # exact MissionReplanBridge interface.
    from airlab.guardian import (EventVision, EventConfig, MultiSensorGuardian)
    ev_cfg = EventConfig(occupancy_min=5, sensor_range_m=8.0)
    ev_vision = EventVision(ev_cfg, image_h=16, image_w=16)
    event_points = np.array(
        [[0.0, 0.0, 0.0, 1.0] for _ in range(120)] +
        [[0.0, -0.2, 0.1, -1.0] for _ in range(40)],
        dtype=float)
    ev_res = ev_vision.process(event_points)
    ev_sensed = ev_res.obstacles
    print(f"[guardian][event] detected={len(ev_sensed)} clusters "
          f"spikes={ev_res.spike_count} power={ev_res.energy_w:.2f}W "
          f"(depth was {percep.energy_w:.2f}W) gops/W={ev_res.gops_per_w:.0f}")
    fusion = MultiSensorGuardian(depth=PerceptionToGuardian(),
                                 events=EventVision(ev_cfg,
                                                    image_h=16, image_w=16))
    fused = fusion.fuse(np.array([[4.0, 0.0, -2.0], [4.1, 0.0, -2.0],
                                  [4.05, 0.2, -2.0]]), event_points)
    _write("out/guardian/event_perception.csv", [{
        "detected": len(ev_sensed),
        "spikes": ev_res.spike_count,
        "power_w": round(float(ev_res.energy_w), 3),
        "depth_power_w": round(float(percep.energy_w), 3),
        "gops_per_w": round(float(ev_res.gops_per_w), 1),
        "fused_count": len(fused),
    }])
    print(f"[guardian][event] fusion kept={len(fused)} objects "
          f"(depth + event consensus), wrote out/guardian/event_perception.csv")

    # Edge-vs-Ground compute split (master prompt section 18, priority #10).
    # Transparent allocator keeps safety-critical tasks onboard and offloads
    # latency-tolerant, low-privacy analytics to GCS/cloud by declared-estimate
    # link/reliability numbers.
    from airlab.guardian import EdgeGroundSplit, LinkEstimate
    split = EdgeGroundSplit(LinkEstimate(link_bandwidth_mbps=20.0,
                                         ground_rtt_ms=50.0,
                                         cloud_rtt_ms=200.0))
    sres = split.place()
    rows_split = []
    for p in sres.placements:
        rows_split.append({
            "task": p.task, "location": p.location, "onboard": int(p.onboard),
            "reason": p.reason, "latency_ms": round(p.latency_ms, 2),
            "power_w": round(p.power_w, 2),
            "bandwidth_mbps": round(p.bandwidth_mbps, 2),
            "privacy_risk": p.privacy_risk,
        })
    _write("out/guardian/edge_split.csv", rows_split)
    ss = sres.summary()
    print(f"[guardian][edge_split] safety tasks all onboard="
          f"{sres.score['safety_full_onboard']} "
          f"onboard_topps={ss['onboard_topps']} "
          f"onboard_power={ss['onboard_power_w']}W "
          f"edge_power_frac={sres.score['edge_power_frac']} "
          f"edge_topps_frac={sres.score['edge_topps_frac']}")
    print(f"[guardian][edge_split] offload_frac="
          f"{sres.score['ground_offload_fraction']} "
          f"_data_mbps={ss['offloadable_data_mbps']} "
          f"ground_latency={ss['ground_latency_ms']}ms")
    print(f"[guardian] wrote out/guardian/edge_split.csv")

    # Dynamic thermal state in re-planning (priority #8).  Instead of always
    # restarting the thermal model at ambient, feed the model the *live*
    # node temperatures.  Demo: a flight that has already heated the edge/NPU,
    # then ask the planner whether the remaining route is thermally feasible.
    live_state = {"cpu_npu": 52.0, "esc": 30.0, "motor": 31.0, "battery": 25.9}
    from airlab.guardian import PredictiveRePlanner as LiveThermPlanner
    cold = LiveThermPlanner(thermal_aware=True, thermal_ambient_c=25.0,
                            cruise_speed=3.0, hover_power_w=112.0,
                            battery_capacity_wh=71.0).plan(
        np.array([0, 0, -2.0]), [np.array([12, 0, -2.0])], battery_frac=1.0)
    hot_edge = LiveThermPlanner(thermal_aware=True, thermal_ambient_c=25.0,
                                thermal_initial_temps=live_state,
                                cruise_speed=3.0, hover_power_w=112.0,
                                battery_capacity_wh=71.0).plan(
        np.array([0, 0, -2.0]), [np.array([12, 0, -2.0])], battery_frac=1.0)
    # A battery-only-hot live state: the edge is cool, so the (over-limit)
    # battery is the worst node and the same route must be rejected.
    live_battery = {"cpu_npu": 30.0, "esc": 30.0, "motor": 31.0,
                    "battery": 46.0}
    hot_batt = LiveThermPlanner(thermal_aware=True, thermal_ambient_c=25.0,
                                thermal_initial_temps=live_battery,
                                cruise_speed=3.0, hover_power_w=112.0,
                                battery_capacity_wh=71.0).plan(
        np.array([0, 0, -2.0]), [np.array([12, 0, -2.0])], battery_frac=1.0)
    _write("out/guardian/dynamic_thermal.csv", [
        {"case": "cold_start", "thermal_feasible": int(cold.thermal_feasible),
         "max_temp_c": round(float(cold.thermal_max_c), 2),
         "worst_node": cold.thermal_worst_node,
         "margin_c": round(float(cold.thermal_margin_c), 2),
         "feasible": int(cold.feasible)},
        {"case": "live_hot_edge", "thermal_feasible": int(hot_edge.thermal_feasible),
         "max_temp_c": round(float(hot_edge.thermal_max_c), 2),
         "worst_node": hot_edge.thermal_worst_node,
         "margin_c": round(float(hot_edge.thermal_margin_c), 2),
         "feasible": int(hot_edge.feasible)},
        {"case": "live_hot_battery", "thermal_feasible": int(hot_batt.thermal_feasible),
         "max_temp_c": round(float(hot_batt.thermal_max_c), 2),
         "worst_node": hot_batt.thermal_worst_node,
         "margin_c": round(float(hot_batt.thermal_margin_c), 2),
         "feasible": int(hot_batt.feasible)},
    ])
    print(f"[guardian][dynamic_thermal] cold_start feasible="
          f"{cold.feasible} worst={cold.thermal_worst_node} "
          f"max={cold.thermal_max_c:.1f}C")
    print(f"[guardian][dynamic_thermal] live_hot_edge feasible="
          f"{hot_edge.feasible} worst={hot_edge.thermal_worst_node} "
          f"max={hot_edge.thermal_max_c:.1f}C "
          f"margin={hot_edge.thermal_margin_c:.1f}C")
    print(f"[guardian][dynamic_thermal] live_hot_battery feasible="
          f"{hot_batt.feasible} worst={hot_batt.thermal_worst_node} "
          f"max={hot_batt.thermal_max_c:.1f}C "
          f"margin={hot_batt.thermal_margin_c:.1f}C")
    print(f"[guardian] wrote out/guardian/dynamic_thermal.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
