# Research Brief #05 — Calibrated Factor-Graph as a Live Safety Signal

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (20/20), and honestly evaluated.

---

## 1. Goal

Brief #04 showed the factor-graph detector works as a *research instrument* but
had a high healthy baseline, so it could not be a live safety signal.  This
stage makes the changes needed to close that gap:

1. **Proper IMU preintegration** (`ImuPreintegrator` in `factorgraph.py`) to
   seed the graph from a clean relative pose instead of a drifted absolute
   dead-reckon.
2. **Startup baseline calibration** so a healthy mission maps to health ≈ 1.0,
   and a real fault (residual far above the calibrated floor) still maps
   sharply down.
3. **A live ablation study** (`run_factorgraph_live.py`) that answers "now that
   it is calibrated, is it a reliable live detector?" rather than merely
   reporting the residual.

## 2. What changed

- `ImuPreintegrator`: integrates bias-corrected body acceleration into NED using
  the AHRS attitude and produces a relative position/velocity prediction.  It is
  re-anchored to GPS and to the graph's optimized last keyframe, so the graph is
  seeded near the true geometry.
- `SlidingFactorGraph.baseline_residual`: a calibrated value is subtracted
  before the residual → health mapping.  The baseline is collected only while
  GNSS (a trusted absolute reference) is available, using the 90th percentile of
  the first samples so the healthy motion floor is not underestimated (the mean
  caused false HOLDs during turns).
- `residual_scale` raised to 0.20 m/s (a real fault has residual of several m/s,
  so this remains sharply separable).
- `SimConfig.factorgraph_enabled` is still **False by default** (the proven
  landmark detector remains the live safety signal); the factor graph is now
  evaluated as a candidate live signal via `run_factorgraph_live.py`.

## 3. Isolation result — the calibrated detector now works

Single-mission check (40 s, GNSS outage 12–24 s):

| Case | Factorgraph live | FG health min | FG residual max | Safety outcome |
|---|---|---:|---:|---|
| healthy | ON | 0.719 | 0.138 | **completed** (no false alarm) |
| bias.25 | ON | 0.048 | 4.223 | **reactive_hold**, no crash |

Compare to the previous state (before calibration): a healthy run dipped to
health ~0.11–0.34 and triggered false HOLD/LAND.  The calibration is what made
the detector usable.

## 4. Batch ablation (n=3, safety layer ON, landmark detector OFF)

| Fault | FG live | In-bounds | Unintended crash | Landed safely | Outcomes |
|---|---|---:|---:|---:|---|
| none | OFF | 1.00 | 0.00 | 0.00 | completed |
| none | ON | 1.00 | 0.00 | 0.00 | completed |
| scale1.5 | OFF | 1.00 | 0.00 | 0.00 | completed |
| scale1.5 | ON | 1.00 | 0.00 | 0.00 | completed |
| bias.1 | OFF | 0.30 | 0.00 | 0.00 | completed |
| bias.1 | ON | 0.31 | 0.00 | 0.00 | completed;reactive_hold;completed |
| bias.25 | OFF | 0.15 | 0.042 | 0.00 | completed;crash;completed |
| bias.25 | ON | 0.15 | 0.056 | 0.00 | completed;crash;completed |

## 5. Honest reading

### What improved
- **No false alarm on healthy/scale1.5** — the calibrated detector is now
  safe to use without degrading a nominal mission.
- **bias.1 (smaller fault) now produces a reactive_hold in one of three runs**
  which the FG-off baseline did not — evidence it is *detecting* a subtle fault
  that the flow self-report cannot see.

### What is still not solved
- **bias.25 still crashes at 0.056 with FG live**, essentially the same as
  without it (0.042).  Two of three missions "completed" even though they left
  the 4 m boundary (in-bounds ~0.15) — meaning the FG did *not* drive the
  safety layer hard enough in those seeds.
- **Root cause (from code): the factor-graph IMU factor still subtracts the
  EKF's accel-bias estimate.**  A corrupt flow source can contaminate that EKF
  bias, which then contaminates the "independent" graph.  The landmark detector
  does not have this dependency because it only uses world geometry.
- **The 90th-percentile baseline is a two-edged sword.**  It fixes false alarms
  on healthy flights, but if a fault is already ramping during the early
  calibration window, the baseline can rise and mask a modest fault.

## 6. Conclusion of this stage

The factor graph is now a **valid, calibrated detector in isolation** (healthy →
1.0, obvious fault → low), but it is **not yet robust enough to replace the
landmark detector as the live safety signal**.  The single remaining blocker is
the graph's dependence on the EKF bias estimate; the landmark detector is
structurally immune to that.

## 7. Next steps

1. **Make the IMU factor fully independent**: estimate accel/gyro bias *inside*
   the graph (or use the graph's own preintegration) instead of borrowing the
   flow-fed EKF bias.  This is the single highest-value change and should make
   bias.25 reliably detected.
2. Then re-run `run_factorgraph_live.py`; if crash → 0 and bias.1 →
   reactive_hold increases, flip `factorgraph_enabled=True` by default.
3. Keep both detectors: run the landmark detector as the primary and the factor
   graph as a cross-check (consensus of two structurally different monitors).

## 8. Run

```bash
.venv/bin/python run_factorgraph_live.py --n 3 --out out/factorgraph_live.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv
```
