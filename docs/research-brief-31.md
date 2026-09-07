# Research Brief #31 — Real PX4 / ROS Telemetry Log Reader

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Data-pipeline priority #6.

---

## 1. Objective
The data pipeline (brief-29) recorded a live `Simulator`.  To be able to
calibrate the learned risk prior and thermal constants from **real** flight
history, the pipeline needs a source adapter that reads actual PX4 / ROS logs
— not a bespoke sim-specific format.

**Honest note:** there is **no real .ulg / .bag / .db3 flight log in this
repository**.  So this brief implements the *adapter* and tests it against a
**simulated PX4-schema fixture** explicitly labelled
`simulated PX4-schema fixture (not a real flight)`.  Swapping in a real
`ulog2csv` directory is exactly what the adapter is designed for.

## 2. What was built

### `airlab.guardian.log_reader`
- `Px4RosLogReader` — reads:
  - a **directory** of topic-per-file CSVs produced by PX4's `ulog2csv`, or
  - a **single merged CSV** (ROS-bag CSV / simple log).
- `Px4LogResult` — normalised `TelemetryDataset` + `RiskTelemetryDataset` +
  metadata.
- Column alias mapping for standard PX4 topics:
  - `vehicle_local_position` (`x/y/z`, `vx/vy/vz`) → pos/vel
  - `battery_status.remaining` → battery
  - `vehicle_global_position.satellites_used` / `eph` → GNSS jam feature
  - `distance_sensor` / `obstacle_distance` → `dist_m`
  - `system_usage` / `cpu_load` → `compute_frac`
  - `board_temperature` / `cpu_temperature` → thermal (if present)
- **Scientific honesty by design:** a real logger has **no ground-truth risk
  label**, so unlabelled logs carry a NaN label and `fit_prior()` **refuses** to
  fit (they are an input distribution, not a silent calibration).  A fixture
  with `with_risk_label=True` exercises the supervised path (still simulated).

## 3. Demo (`out/guardian/log_reader.csv`)
```
source=simulated PX4-schema fixture (not a real flight)
read 120 telemetry rows, 120 risk samples
unlabelled_prior_fitted=False labelled_prior_fitted=True
jam late=0.750 > early=0.067; prior near=0.844 far=0.044
```

This demonstrates the ethical + correct behaviour:
- unlabelled log cannot silently train the prior,
- labelled fixture trains it and correctly sees the late-mission GNSS jam
  (0.75 vs 0.07) and near-obstacle risk (0.844 vs 0.044).

## 4. Tests (`TestLogReader`, 2 tests)
- loads a fixture and refuses an unlabelled prior fit.
- loads a labelled fixture, detects late-mission jamming, and fits a prior with
  near > far.
- Guardian suite: **34/34**; full suite below.

## 5. Honest limits
- No real PX4/ROS log is shipped; the reader is validated on a simulated
  fixture.
- Only `ulog2csv`-style CSV input (no binary `.ulg`/`.db3` parser); a real log
  must first be converted with `ulog2csv` (PX4's own tool).
- No ground-truth risk label from a logger, so the prior can only become truly
  learned when a labelled obstacle/jam dataset is supplied (or a log includes
  an obstacle-distance channel plus an independent danger label).

## 6. Next autonomous tasks
1. **Perception path (SNN/NeuViT):** sensed obstacles → `guardian_obstacles`
   (replace scripted obstacles in the mission bridge).
2. **Dynamic thermal state in re-planning** (feed live node temps into the
   planner instead of restarting at ambient).
3. **Thermal calibration from a real PX4 log** (when a board-temp channel is
   available) so the declared thermal constants become measured.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestLogReader -v
```
