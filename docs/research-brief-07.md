# Research Brief #07 — Live Factor Graph: GNSS-Independent Detector Path

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (24/24), factor graph promoted to live (default).

---

## 1. The gap that kept the FG from being live

Brief #06 made the factor graph a *valid calibrated detector in isolation*:
healthy → no false alarm, obvious fault → health ~0.05.  Yet in the n=3 batch it
still crashed at bias.25 (0.056 with FG vs 0.042 without).  The batch study
could not see *why* until we reproduced an individual crashing mission.

**Root cause:** at the crash instant the FG health was already **0.068**, but
the safety layer was still in **`CRUISE`**.  The safety FSM only reacted to a low
velocity-health signal when **GNSS was unavailable**:

```python
elif health < flow_health_fail and gps_available:
    fail_reason = None   # "we have a little more room"
```

That assumption is wrong for a *convincing-but-wrong* velocity source.  Even
with an absolute GNSS fix present, the EKF's estimated velocity is corrupted by
the bad flow measurement and the **controller uses that velocity**, so the
aircraft can still be driven into the ground.  In the reproduced mission the
vehicle climbed to ~24 m and then descended to the ground while still in
`CRUISE`.  GNSS presence did not protect it.

## 2. Fix: a dedicated independent-detector safety path

The safety layer now receives a separate `detector_health` signal (min of the
enabled independent monitors: landmark geometry + calibrated factor graph).  It
is **allowed to force a landing even when GPS is still up** because it is the
*reason we no longer trust the velocity source at all*.

### SafetyConfig additions
```
detector_health_warn = 0.55   # wide margin: healthy fast turns dip to ~0.58
detector_health_fail = 0.25   # deep margin: real faults sit at ~0.04
grace_detector_warn_s = 1.50
grace_detector_fail_s = 1.00
```

The warn margin is deliberately set *below* the healthy fast-turn dip (0.58), so
a healthy mission never false-holds; the fail margin is *far* below a real fault
(0.04), so detection stays decisive.

### Factor-graph is *not* folded into instantaneous flow health
The FG is 2 Hz and its healthy residual can dip briefly during fast turns.  It
must not be min'd into `_effective_flow_health` (which drives the flow-warn/fail
path), or it will false-alarm a healthy turn during a GNSS outage.  It is fed
only to the dedicated detector path.  Landmark remains in both paths (it is
fast and reliable).

### Factor-graph health is gated on calibration
Until the startup GNSS-based baseline is collected, the FG residual maps to a
low, meaningless health.  The detector path uses `factorgraph_health_trusted`,
so an uncalibrated graph can never trigger.

## 3. Results — live ablation n=10 (safety ON, landmark detector OFF)

| Fault | FG live | In-bounds | Unintended crash | Landed | Outcomes |
|---|---|---:|---:|---:|---|
| none | OFF | 1.000 | 0.000 | 0.000 | 10× completed |
| none | ON | 1.000 | 0.000 | 0.000 | 10× completed |
| scale1.5 | OFF | 0.939 | 0.000 | 0.000 | 10× completed |
| scale1.5 | ON | 0.932 | 0.000 | 0.000 | 8× completed, 2× reactive_hold |
| bias.1 | OFF | 0.288 | 0.002 | 0.000 | 10× completed |
| bias.1 | ON | 0.480 | 0.002 | 0.538 | 9× landed_safely, 1× reactive_hold |
| bias.25 | OFF | 0.149 | **0.024** | 0.000 | 2× crash, 8× completed |
| bias.25 | ON | 0.214 | **0.000** | 0.655 | **10× landed_safely** |

## 4. Interpretation

* **Healthy stays clean.** FG ON = 10/10 completed, no false hold.  The wide
  warn margin and the separation of FG from instantaneous flow health fixed the
  1/10 false alarm seen in the prototype run.
* **bias.25 is now fully contained.** Crashes dropped from 0.024 → **0.000**;
  every run either landed safely (6.5/10 landing fraction) or held.  This is the
  acceptance criterion from brief #05.
* **Even small faults are now visible.** bias.1 (a mild ramping bias) is
  detected and lands safely with the FG; without it the mission completes while
  carrying a corrupt velocity source.  scale1.5 gets a light warning (2/10
  reactive_hold), which is a genuinely conservative reaction to a real scale
  error.
* **Consensus default.** Both detectors are now ON by default.  The landmark
  detector remains the fast primary; the FG is the slow, principled
  cross-check.  They are structurally different (world-geometry angle vs
  joint IMU/flow/GPS/landmark least-squares), so their agreement is a stronger
  safety claim than either alone.

## 5. Remaining honest caveats

* **Trade-off: survivability vs mission completion.**  The FG now *lands* on
  faults it earlier ignored (bias.1, sometimes scale1.5).  That is the correct
  safety behaviour but it means the safety layer is intentionally
  conservative — a downstream mission planner would need to weigh "land now"
  vs "fly on degraded but bounded".
* **Scale1.5 is not a hard separation.**  FG warns (2/10 hold) but does not land
  — consistent with the brief-06 isolated finding that medium scale errors are
  harder to separate than ramp biases.
* **One run's residual can still sit near the warn boundary** on aggressive
  turns.  We used a long grace (1.5 s) to convert those into no-ops, but a
  wider grid is still the right next experiment.

## 6. Next steps

1. **Wider grid** (n≥30, more fault depths and seeds) to fit a proper
   crash/detection ROC and choose the final consensus policy.
2. **Explicit consensus policy** (AND / OR / weighted) between landmark + FG.
3. **Mission-aware response** — return-to-base vs land-now trade-off using the
   energy model.

## 7. Run

```bash
.venv/bin/python run_factorgraph_live.py --n 10 --seed 77 --out out/factorgraph_live.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv
```
