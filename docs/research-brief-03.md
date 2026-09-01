# Research Brief #03 — Independent Landmark-Based Consistency Detection

**Date:** 2026-08-31
**Project:** AIR Lab
**Status:** implemented; 19/19 tests passing; landmark ablation study done.

---

## 1. Problem being solved

Brief #02 showed the biggest open safety gap:

> A **"convincing but wrong"** velocity-aiding source (scale error, slowly
> growing bias) is invisible to a single-stream EKF.  The filter accepts the
> wrong measurement, the *estimate* follows it (so the EKF covariance stays
> small), and the uncertainty-based safety layer never realises anything is
> wrong.  The aircraft drifts away, then crashes.

We also tried (and deliberately left disabled) a parallel IMU dead-reckon.  It
is **not** a valid second opinion because it is another integral of the same
IMU with the same biases — it drifts even in a perfectly healthy mission and
generates false landings.

## 2. Approach

Add a **structurally independent** consistency signal: a small, fixed landmark
field plus a simplified body-mounted camera.

* The camera measures the *true* direction to known landmarks (real world).
* The EKF predicts what those directions should be from its own state.
* Any disagreement is evidence that the EKF has drifted into a state that the
  corrupt flow/VIO source hid from it.

### The key design choice: compare inter-landmark angles, not absolute bearings

The first attempt compared each landmark's absolute body-direction to the
prediction.  It was too sensitive to normal EKF attitude error — a healthy
mission that had only ~0.05 rad of attitude noise was being flagged as
inconsistent at ~0.4–0.6 score.

The fix: compare the **angles between pairs of landmarks**.  These are
*invariant to camera/attitude rotation* but *highly sensitive to position*, so
a wrong position is detected without being confused by normal attitude noise.

### The key modeling choice: the VIO self-report must NOT see accuracy faults

In the earlier model, the flow module's "health" signal was a function of the
actual scale/bias error.  That is physically wrong: a real optical-flow/VIO
module reports "features tracked, high confidence" even when its scale or bias
is wrong — it simply does not know.  We now make the self-report reflect
**availability** (dropout / feature-loss) only, and leave **accuracy** faults
to the independent landmark detector.

## 3. Results from the ablation study (n=3 per cell)

Bias/scale faults, GNSS outage windows, safety layer always ON, landmark
detector ON vs OFF:

| Fault | Outage | Detector | Unintended crash | Landed safely |
|---|---|---:|---:|---:|
| none | 0/15/30 | on/off | 0.00 | 0.00 |
| scale1.8 | 0/15/30 | on | 0.00 | 0.00 |
| scale1.8 | 0/15/30 | off | 0.00 | 0.00 |
| bias.25 | 0 | on | 0.00 | 0.00 |
| bias.25 | 0 | off | 0.00 | 0.00 |
| bias.25 | 15 | on | 0.00 | **0.268** |
| bias.25 | 15 | off | 0.00 | 0.00 |
| bias.25 | 30 | on | 0.00 | **0.435** |
| bias.25 | 30 | off | 0.00 | 0.00 |

Interpretation:
* **No crashes in any cell** — the safety layer already removes the crashes.
* **The landmark detector materially improves the outcome for the hardest
  fault (bias.25) during GNSS loss**: it increases controlled landings from
  0.00 → 0.27 (15 s) and 0.00 → 0.44 (30 s).  It is the *only* thing that sees
  a VIO bias fault when the VIO self-report cannot.
* **Scale-1.8 is still largely benign here** because the controller/estimator
  are robust enough at that magnitude; the detector does no harm (0 crash).
* **Healthy missions stay completed** — the detector remains >0.8 score
  throughout when nothing is wrong, so it is not a false-alarm generator.

The honest caveat: a 0.44 controlled-landing rate at 30 s means the detector
fails or lands late about half the time.  This is because the landmark field is
small, only 5 Hz, and does not always have ≥2 landmarks in view during the
whole drift.  Improving that is a clear next step (see §5).

## 4. What I learned / corrected again

1. **"Second opinion" must be structurally independent, not just another module
   running the same math.**  A dead-reckon is a re-derivation of the same
   flawed IMU; landmarks are observation of *world geometry*, so they are
   genuinely independent.
2. **Attitude invariance matters.** Comparing absolute bearings conflates a
   harmless attitude error with a real position problem.  Inter-landmark angles
   separate the two.
3. **Don't make a sensor "confess" to its own accuracy fault.** Sensor
   self-diagnostics report feature availability, not absolute truth.  Give the
   independent monitor the job of detecting accuracy faults, and model the
   sensor honestly.

## 5. Next experiments

1. **Better landmark geometry / data association.** Larger field, more
   landmarks, higher rate, and persistence (track the same landmarks across
   frames) so the detector can watch the *trend* of the residual and not just
   single-frame snapshots.
2. **Full factor-graph backend.** Replace the fixed landmark set + heuristic
   score with a batch/manifold factor graph where the *flow factor* and the
   *landmark factor* are jointly optimised.  A single outlier/biased factor will
   then show up directly in the residual, which is the textbook way to do
   multi-hypothesis rejection.
3. **Learned residual model.** A small CNN/MLP on the landmark patch that
   predicts "trust this frame" to handle occlusions / moving landmarks.
4. **Event-camera surrogate.** High-rate, low-power landmark edges to close the
   same gap with much lower energy — the "intelligence per watt" direction.

## 6. Run

```bash
.venv/bin/python run_landmark.py --durations 0,15,30 --n 3 --out out/landmark.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv
```
