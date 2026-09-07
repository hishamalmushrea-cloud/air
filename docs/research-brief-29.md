# Research Brief #29 — Telemetry Data Pipeline (Edge → Storage → Analytics)

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  **Program priority #5** (per master-prompt
prioritisation).  This closes the loop that the earlier *prior* and *thermal*
models relied on **simulated / modelled** data: now a real flight can be
recorded into a reproducible dataset and the learned prior fitted from it.

---

## 1. Objective
`RiskPriorModel` (brief-27) and `PartThermalModel` (brief-28) were calibrated on
injected simulated labels / declared constants.  The master prompt (§34) asks
for a real **Sensor → Edge → Storage → Processing → Analytics → Visualisation**
pipeline.  This brief adds the storage/analytics link for the guardian
telemetry, so the learned oracle and thermal calibration can be retrained from
*recorded* flights instead of fabricated ones.

## 2. What was built

### `airlab.guardian.pipeline`
- `TelemetryDataset` — canonical schema (t, fused pos/vel, throttle, `power_w`,
  `battery_frac`, per-node temps, motor/vib/GPS/flow residuals, landmark &
  factor-graph residuals, mode/reason).  Includes `append()`, `to_csv()`,
  `as_arrays()`.
- `RiskTelemetryDataset` — **tabular risk samples** `(dist_m, jam, label)` for
  the learned prior:
  - `dist_m` = distance to nearest projected obstacle,
  - `jam` = actual GNSS degradation (then max with jamming-corridor proximity),
  - `label` = live risk-field sample at the aircraft position (what the oracle
    actually saw, not the analytic formula).
  - `fit_prior(prior)` — trains a `RiskPriorModel` directly from recorded data.
- `DataPipeline` — couples a live `Simulator` to both; `record(dt, obstacles,
  jamming_centers, risk_field)` appends one telemetry row + one risk sample.

### `SimConfig` integration
`guardian_data_pipeline`, `guardian_data_obstacles`, `guardian_data_jamming`
let a `Simulator` record its own mission into the dataset.  `run()` calls the
pipeline every control step.

## 3. Demo (`out/guardian/telemetry.csv`, `out/guardian/data_pipeline.csv`)
A 5 s flight through an obstacle at `(4,0)` + jamming corridor at `(1,0)`:

```
recorded 500 rows, 500 risk samples -> out/guardian/telemetry.csv
prior fitted n=500  near=0.387  far=0.000  jammed=0.692  (train_jam_max=1.000)
```

The learned prior — now fitted on **recorded** telemetry rather than synthetic
labels — says: risk is high near the obstacle (0.387), zero far away (0.000),
and rises inside the recorded jam corridor (0.692).  The dataset itself is
reproducible (CSV with timestamps + integrity metadata).

## 4. Tests (`TestDataPipeline`, 2 tests)
- records >100 rows of reproducible telemetry (pos + per-node temps) and writes
  a CSV.
- fits a `RiskPriorModel` from a `RiskTelemetryDataset` (n=50, near > far).
- Guardian suite: **30/30**; full suite below.

## 5. Honest limits
- The recorded flight is still a *simulation*; the pipeline is real, but the
  data source is `Simulator`, not a physical aircraft.  Swapping the source to
  a real log reader is the remaining engineering step.
- Only one frame-sampled CSV schema; no parquet/indexing yet.  Fine for the
  guardian's feature set, not yet a mission-scale store.
- `jam` feature blends declared jamming corridors + GNSS degradation; it does
  not come from a spectrum/antenna receiver.

## 6. Next autonomous tasks
1. **Thermal-aware mission budget**: use `PartThermalModel` + energy to keep
   the platform inside both budgets while planning.
2. **Perception path (SNN/NeuViT)**: replace scripted `guardian_obstacles` with
   sensed obstacles.
3. **Real log reader**: ingest actual PX4/ROS bag telemetry into
   `RiskTelemetryDataset` so the prior is trained on real flight history.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestDataPipeline -v
```
