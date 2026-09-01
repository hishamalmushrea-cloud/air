# Research Brief #06 — True IMU Independence via In-Graph Bias Estimation

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (22/22), and honestly evaluated.

---

## 1. Goal

Brief #05 identified the single remaining blocker preventing the factor graph
from replacing the landmark detector as the live safety signal:

> The factor-graph IMU factor still borrowed the **flow-fed EKF's accel-bias
> estimate** (`accel - ekf.accel_bias`).  A corrupt flow source can contaminate
> that EKF bias, which then contaminates the "independent" graph.

This stage removes that dependency by estimating the IMU acceleration bias
**inside the factor graph itself**, using the independent landmark/GPS geometry
as the constraint.

## 2. What changed

### State vector now includes a shared IMU bias
```
X = [p0,v0, p1,v1, ..., p_{n-1},v_{n-1}, ba]
```
where `ba` is a 3-vector NED acceleration bias shared across the sliding window.

### Raw IMU is fed to the graph
The simulator now passes `self.last_imu.accel` (raw) instead of
`accel - ekf.accel_bias`.  The graph subtracts its **own** estimated bias inside
the IMU motion factor.

### IMU residual now depends on the bias state
`rv = v_{k+1} - v_k - (a_raw - ba) * dt`, and the analytic Jacobian includes
`+dt` on the `ba` columns.  A bias-prior residual `ba - 0` is appended and
weighted by `bias_reg` (default now `1.0`).

## 3. Important finding — weak bias prior breaks the detector

We first tried `bias_reg = 0.05`.  The results were **worse**, not better:

| bias_reg | Healthy | bias.25 | Why |
|---|---:|---:|---|
| 0.05 | false HOLD (health 0.64) | no reaction, crash 0.044 | graph *absorbs the flow fault into its bias*, so flow residual stays low |
| **0.20** | completed (health 0.72) | reactive_hold, crash 0.0 | bias prior strong enough that graph cannot "explain away" the flow fault |
| **1.00** | completed (health 0.72) | reactive_hold, crash 0.0 | same, bias stays ~0.02 |
| **5.00** | completed (health 0.71) | reactive_hold, crash 0.0 | same |

The reason is subtle but important: an **under-regularised** graph bias is
another degree of freedom that lets the optimizer *self-confirm* the corrupt
flow measurement (it changes the IMU prediction to match the wrong flow).  This
is the multi-hypothesis equivalent of "the outlier is absorbed by a parameter
rather than rejected by the robust kernel."  A strong prior on the bias is what
forces the graph to *reject* the flow outlier instead of fitting it.

## 4. Isolation results (single mission, GNSS outage 12–24 s, bias_reg=1.0)

| Case | FG health min | FG residual max | Safety outcome |
|---|---:|---:|---|
| healthy | 0.721 | 0.143 | **completed** (no false alarm) |
| bias.25 | 0.044 | 4.63 | **reactive_hold**, crash 0.0 |

## 5. Batch ablation (n=3, safety ON, landmark detector OFF)

| Fault | FG live | In-bounds | Unintended crash | Landed | Outcomes |
|---|---|---:|---:|---:|---|
| none | OFF/ON | 1.00/1.00 | 0/0 | 0/0 | completed |
| scale1.5 | OFF/ON | 1.00/1.00 | 0/0 | 0/0 | completed |
| bias.1 | OFF | 0.30 | 0 | 0 | completed |
| bias.1 | ON | 0.31 | 0 | 0 | completed;reactive_hold;completed |
| bias.25 | OFF | 0.15 | 0.042 | 0 | completed;crash;completed |
| bias.25 | ON | 0.15 | 0.056 | 0 | completed;crash;completed |

## 6. Honest conclusion

* **We fixed the dependency** the brief-05 analysis identified: the IMU factor
  no longer borrows the EKF bias.  The graph now owns its bias estimate and
  infers it from independent geometry.
* **In isolation this works:** healthy → no false alarm; a large bias → clear
  residual and safe hold.
* **Across random seeds at the hardest fault**, the batch still shows a crash
  (0.056 with FG vs 0.042 without — not a decisive improvement).  The factor
  graph now *detects* the fault (bias.1 gets a reactive_hold it didn't before),
  but its reaction is not always fast/aggressive enough to beat the seed
  variance at bias.25.
* **Decision: keep `factorgraph_enabled=False` (opt-in).**  The landmark
  detector remains the live safety signal because it is structurally immune to
  the flow-fed-EKF-bias problem and gives cleaner monotone separation on medium
  faults.  The factor graph is now a properly-calibrated *second* detector we
  can cross-check against.

## 7. Next steps

1. **Run a bigger live study** (n≥10 per fault) to get a statistically reliable
   crash-vs-detector curve before promoting the FG to live.
2. **Make the FG reaction faster** (reduce safety grace or couple the FG health
   directly into the LAND trigger) so the detected fault converts to a *land*,
   not just a transient HOLD.
3. **Consensus** between landmark detector + FG for the deepest safety margin.

## 8. Run

```bash
.venv/bin/python run_factorgraph_live.py --n 3 --out out/factorgraph_live.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv
```
