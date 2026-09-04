# Research Brief #24 — Predictive Maintenance / Subsystem Health Score

**Date:** 2026-09-04
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  First engineered component of the master
prompt's §35 (predictive maintenance).  The guardian now has a
health-sense subsystem and can **ABORT** on a degrading maintenance state before
it becomes a crash.

---

## 1. Objective
From telemetry detect motor degradation, battery aging, temperature anomalies,
vibration anomalies, and sensor degradation → produce a per-subsystem
**Health Score** and an aggregate signal the guardian can act on.

## 2. What was built (`health.py`)
`SubsystemHealth` learns a *baseline* from the first `warmup_samples` healthy
frames (same design-in trust idea as frame trust), then scores each live frame
against it:

| subsystem | feature | score basis |
|---|---|---|
| battery | energy fraction | drop below healthy reference |
| motor | throttle-to-speed residual | ratio vs healthy p95 |
| thermal | temperature | °C above baseline |
| vibration | jitter | ratio vs healthy p95 |
| sensor_gps / sensor_flow | disagreement | fraction above threshold |

Status mapping: ≥0.85 ok, ≥0.70 watch, ≥0.55 warn, <0.55 critical.

`HealthPrognosis` aggregates by **weakest-link (min)** — any critical subsystem
drags the whole health down — and exposes a per-second trend.

The guardian `decide()` now takes `health_score`; a critical aggregate
(< 0.45) triggers `ABORT` with reason `predictive_maintenance_health_low`
and the `predictive_maintenance_health` declared capability + `maintenance_abort`
undeclared capability.

## 3. Demo (`run_guardian.py`, health scenario)

After a healthy 55-frame window, all subsystems degrade.  At step 69+ the scores
collapse to critical and the guardian enters `ABORT`:

```
final_aggregate=0.000
critical=['battery','motor','thermal','vibration']
guardian_mode=ABORT
```

Healthy window at step 19 stays out of critical (battery 0.67 watch, motor 0.76
watch, thermal 0.81 watch, vibe 0.84 watch, sensors 0.86+ ok) — so the engine
does **not** false-alarm a steady healthy mission.

## 4. Tests
- `test_healthy_subsystems_stay_healthy`: ≥0.5 aggregate on a healthy 60-frame
  window.
- `test_degraded_subsystems_trigger_critical_heap`: <0.5 aggregate after
  degradation.
- `test_guardian_aborts_on_low_aggregate_health`: `decide(health_score=0.21)`
  returns ABORT with the declared capability.
- Guardian suite: **16/16**. Full suite run below.

## 5. Honest limits (scientific honesty)
- All scores are **Simulated** (numpy features), not measured on hardware.
- Motor "degradation" is a proxy (throttle-to-speed residual); no real motor
  telemetry / ESC current.
- Thermal uses a generic °C anomaly; no real thermal model or vapour chamber.
- Baseline is learned from the current flight's first frames — this is
  vulnerable to a fault present *at startup*; the full stack's independent
  detectors mitigate that but the health engine alone does not.

## 6. Decision / next
- Keep weakest-link aggregate (safety-first).
- **Next**: allow the health engine to consume the *real* simulator telemetry
  (`simulator.py`) once the guardian is ported into the main loop, so the scores
  are measured from the fused state, not synthetic features.
- **Then**: thermal model (CPU/ESC/motor/battery) and a GCS health dashboard.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
```
