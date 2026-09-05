# Research Brief #26 — Health Engine Fed by Real Fused Telemetry

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  **Program priority #2** (per master-prompt
prioritisation).

---

## 1. Objective
`SubsystemHealth` (brief-24) was fed synthetic numpy features.  Now that the
oracle bridge is inside the real mission controller (brief-25), the next step is
to let the predictive-maintenance engine score the **actual fused telemetry** the
flight stack produces — so the health scores reflect what the aircraft truly
experiences, not hand-picked test vectors.

## 2. What was built

### `airlab.guardian.telemetry_health.TelemetryHealthBridge`
Reads the live `Simulator` every control step and maps it to the health
engine's feature tuple:

| health feature | real telemetry source |
|---|---|
| battery | `_battery_frac()` from the online power model (actual throttle → energy) |
| motor | throttle-efficiency observable `|1 - (g - a_z_ned)/cmd|` from the actual IMU/command |
| thermal | `ThermalState` (lumped first-order model driven by real power draw) |
| vibration | EMA of vehicle acceleration jitter |
| sensor_gps | GNSS availability + landmark / factor-graph residuals |
| sensor_flow | `_last_flow_mismatch` from the velocity-integrity monitor |

`ThermalState` is a documented model (not a measurement): 1200 J/K lumped mass,
2.2 W/K to ambient, 6 W idle baseline — tuned qualitatively for a small edge
avionics stack.

### Physical motor degradation in the plant
`Quadrotor` now takes `motor_efficiency` and multiplies it into the delivered
thrust. `SimConfig` adds `motor_degrade_at` / `motor_degrade_eff` so a fault can
appear **mid-flight** (calibrate healthy → degrade → observe).  This is the
honest way to exercise the engine; a fault present at startup would be absorbed
into the warmed-up baseline (documented limit in brief-24/26).

## 3. Demo (`run_guardian.py`, telemetry_health section)

Same aircraft, healthy for 7 s then motor degrades to 70 % efficiency:

```
healthy   battery=0.77 watch, motor=0.87 ok,   thermal=0.81 watch,
          vibration=0.83 watch, gps=0.87 ok, flow=0.88 ok
degraded  battery=0.76 watch, motor=0.33 critical, thermal=0.81 watch,
          vibration=0.87 ok,   gps=0.87 ok, flow=0.88 ok

final_agg healthy=0.767  degraded=0.325
```

The airborne health aggregate drops from 0.767 to 0.325, and the **motor**
subsystem goes from ok to critical — exactly what predictive maintenance is
supposed to catch before it becomes a crash.

## 4. Tests
- `test_telemetry_health_healthy_run_stays_ok`: full stack 8 s, engine calibrated,
  no critical subsystem.
- `test_telemetry_health_detects_mid_flight_motor_degrade`: motor degrades at
  6 s; post-fault motor residual is bigger than the pre-fault window and the
  aggregate health falls.
- Guardian suite: **22/22**; full suite below.

## 5. Honest limits
- Thermal is a **modelled** temperature, not a measurement, and uses a generic
  lumped mass — no real CPU/ESC/motor part data yet.
- The motor "degradation" is a scalar efficiency loss; a real degrading motor
  also changes current, temperature, and acoustic signature.
- A **startup** fault (present before warm-up) is absorbed into the baseline;
  the independent consensus detectors (brief-06/09) remain the backstop for
  that.
- The bridge is diagnostic-only: it never commands the aircraft.  Decision
  authority stays in `GuardianBrain`.

## 6. Next autonomous tasks (priority #3)
1. **Learned risk prior** from recorded near-miss / jam / obstacle telemetry:
   upgrade `RiskWorldModel` from hand-set Gaussian amplitudes/σ to a model
   fitted on actual flight history.
2. **Low-watt thermal model** with part-level data (CPU/NPU/ESC/motor/battery)
   so the health engine can predict heat before it exceeds a budget.
3. **Data pipeline** that writes real simulator telemetry to a dataset so the
   learned prior has reproducible inputs.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestTelemetryHealth -v
```
