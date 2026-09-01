# Research Brief #04 — Factor-Graph / Multi-Hypothesis Consistency Detector

**Date:** 2026-08-31
**Project:** AIR Lab
**Status:** implemented; 20/20 tests passing; detector-characterisation study done.

---

## 1. Goal

Brief #03 proved that an *independent landmark detector* can catch a corrupt
velocity-aiding source that the EKF self-report cannot.  But that detector was a
single hand-crafted heuristic.  This stage builds the **principled version**: a
small nonlinear factor graph that jointly optimises the plausible navigation
state over a sliding window, then uses the **post-optimisation residual of the
"suspect" flow factor** as the consistency signal.  This is the textbook
"factor graph + robust kernel + residual-based outlier rejection" framing of
multi-hypothesis navigation integrity.

## 2. What was built

### `src/airlab/factorgraph.py`
- **`Keyframe`**: position/velocity initial estimate, bias-corrected IMU
  acceleration, optional GNSS fix, optional flow measurement, optional landmark
  bearing observation.
- **`SlidingFactorGraph`**:
  - Factors: IMU motion (velocity/position propagation), GNSS absolute
    (when available), flow velocity (the suspect), landmark inter-angle
    (the independent world geometry).
  - Cauchy robust loss inside a Gauss-Newton / IRLS loop, so the graph itself
    down-weights a single bad factor rather than merely reporting it.
  - **Analytic Jacobian** (the IMU/flow/GPS factors are linear; landmark factors
    are non-linear only in one keyframe position block), which makes it fast
    enough to run repeatedly (~20 ms/optimisation).
  - **Warm start**: optimised state is written back into the keyframes so the
    sliding window can lock onto the landmark-constrained geometry.
- **`build_keyframe`** helper: rotates measured accelerometer into NED using the
  AHRS attitude and removes gravity, so the IMU factor uses sensor data, not
  truth.

### Integration
- `SimConfig.factorgraph_enabled` (default **False** — opt-in for analysis).
  The live safety signal remains the proven landmark detector.
- When enabled, the simulator builds keyframes at 2 Hz from raw measured data
  (including an IMU-only dead-reckon for the independent initial estimate) and
  reads `fg_info["health"]` / `fg_info["flow_residual"]`.
- `run_factorgraph.py` / `factorgraph_detector_study()` isolates the *detector*:
  safety layer disabled, flow still on, and records max flow residual / min
  graph health per fault.

## 3. Detector-characterisation results (n=3 per cell, 40 s missions)

| Fault | FG max flow residual (m/s) | FG min health | Landmark min health |
|---|---:|---:|---:|
| none | 3.19 | 0.110 | 0.823 |
| scale1.8 | 2.91 | 0.023 | 0.447 |
| bias.25 | 10.03 | 0.006 | 0.210 |
| bias.2 | 44.37 | 0.001 | 0.268 |

## 4. Honest reading (what works and what does not)

### What works
1. **Large faults are brutally obvious.** A 2 m/s ramping bias drives the flow
   residual to ~44 m/s and health to ~0.001.  The factor graph is extremely
   sensitive to a flow source that is badly wrong.
2. **The independent landmark detector is cleaner for medium faults.** Its
   min-health monotonically separates healthy (0.82) > scale1.8 (0.45) >
   bias.25 (0.21) — a clean monotone ranking.
3. **Fast enough to be a real-time candidate** (~20 ms/optimisation with the
   analytic Jacobian).

### What does not work (and is why it is opt-in)
1. **The healthy baseline is not near zero** in the full mission (surplus
   residual ~3.2 m/s, health ~0.11).  In a clean synthetic straight-line test
   the healthy residual was ~0.01.  The difference is the *full* loop: the graph
   is initialised from a drifting IMU dead-reckon, and during aggressive turns
   the sliding window cannot always keep ≥2 landmarks in view, so it never fully
   converges to the true geometry.  A healthy flight should score ~1.0 and it
   does not.
2. **It does not cleanly separate healthy from scale-1.8** (3.19 vs 2.91).  The
   robust loss is doing its job by *absorbing* the scale-1.8 error as a modest
   outlier, which is good for state estimation but bad for detection when the
   healthy baseline is also elevated.
3. **It is not yet a live safety signal.** Because the healthy baseline is not
   ~1.0, feeding it into the safety FSM causes false HOLD/LAND.  We deliberately
   keep `factorgraph_enabled=False`.

## 5. Why this is still valuable

The factor graph is the *correct architecture* for this problem, and this stage
proves the concept end-to-end (source code, analytic Jacobian, robust kernel,
warm start, detector characterisation).  It also shows precisely **where** the
remaining work is: it needs better geometry observability (more/persistent
landmarks), a better independent init (a proper VIO/IMU pre-integration rather
than a naive dead-reckon), and a calibration step so a healthy flight maps to
health≈1.0.

The practical recommendation from this data is clear:
> **Use the simple landmark detector as the live safety signal, and use the
> factor graph as the research-grade detector that we now know how to make
> robust** (it already catches the worst faults decisively).

## 6. Run

```bash
.venv/bin/python run_factorgraph.py --n 3 --out out/factorgraph.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv
```
