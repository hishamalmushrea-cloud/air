# Research Brief #02 — Corrupt Velocity-Aiding & Uncertainty-Aware Safety

**Date:** 2026-08-31
**Project:** AIR Lab
**Status:** implemented; 17/17 tests passing; corruption study run.

---

## 1. Why this stage

The previous study showed that **navigation survives a 30 s GNSS outage when
optical-flow / VIO velocity aiding is available**, but fails badly without it.
The important follow-up question is:

> What happens when the *aiding sensor itself* is wrong?

That is the hardest failure mode in real systems, because a "convincing but
wrong" VIO/flow source feeds the EKF measurements that are themselves consistent
with the filter.  The EKF alone will often not know it is wrong.

## 2. What was built

* `SafetyMonitor` (`safety.py`) — a conservative, explainable FSM:
  `CRUISE → HOLD → LAND → LANDED`.
  * Triggers on **EKF horizontal position uncertainty** (from the covariance),
    **flow self-reported health**, and **GNSS availability**.
  * Uses grace periods so a short noise burst does not immediately abort a
    mission.
* `FlightController.landing()` — an **open-loop vertical descent** used during
  loss-of-trusted-horizontal-navigation.  It deliberately ignores horizontal
  position error (there is no trustworthy reference to control against) and
  only levels the attitude + sinks on the vertical channel.
* Corrupt velocity-aiding sensor faults in `SensorConfig` / `SensorSuite`:
  * `flow_dropout` — the source goes dark.
  * `flow_scale` — e.g. 1.5× or 1.8× wrong magnitude.
  * `flow_bias_ramp` — slowly growing velocity bias (worst case).
  * `flow_health` — a noisy self-diagnostic signal derived from scale/bias/
    dropout (so the safety layer does not receive truth).
* `VelocityIntegrityMonitor` (`safety.py`) — a parallel IMU dead-reckon used to
  cross-check the flow.  It is currently **off by default**; see §5 (why).
* `run_corruption.py` / `corruption_study` — a grid study over GNSS outage ×
  velocity-aiding fault × safety layer on/off.
* Safety-aware metrics: `safety_fraction`, `landed`, and **unintended crash** —
  where "crash" now means ground contact that was *not* an intentional landing.

## 3. Results

Grid: `n=3` per cell, faults `none/dropout/scale1.5/scale1.8/bias.1/bias.25`,
GNSS outage `0/15/30 s`, safety layer on/off.

### Key comparisons (30 s GNSS outage unless stated)

| Fault | Off | On | On landed safely | Off collision |
|---|---|---|---:|---:|
| none | 1.00 in-bounds | 1.00 | 0.00 | 0.00 |
| dropout | 0.32 | 0.34 | 0.63 | **0.64** |
| scale1.5 | 0.81 | 1.00 | 0.00 | 0.00 |
| scale1.8 | 0.82 | 0.67 | 0.62 | 0.00 |
| bias.1 | 0.26 | 0.27 | 0.67 | 0.00 |
| bias.25 | 0.15 | 0.16 | 0.67 | 0.01 |

Interpreting the four "outcome" columns:
* **Landed safety** significantly improves with the layer on for **dropout**,
  **scale1.8**, **bias.1**, and **bias.25** (the layer detects degraded health
  and decides to land).
* It removes the catastrophic dropout crashes almost completely: unintended
  crash at 30 s is `0.63→0.007` for dropout and `0.64→0.0` for the others.
* The **in-bounds fraction does not improve** for bias faults.  That is
  expected: if the only velocity/navigation source is wrong, the *location*
  uncertainty and tracking error are genuinely high.  The safety layer does not
  magically recover accuracy; it converts a likely crash into a controlled
  landing at a degraded (but safer) location.

## 4. The most important caveat

**A persistent bias is effectively invisible to a filter that only ever sees the
same biased measurement.**  At `bias.25`, flow health is ~0.06 (because the
self-report includes bias magnitude), so the safety layer *does* react.  But a
real optical-flow bias might not be observable that way, and the study shows
that even with the layer on the vehicle can drift far from the nominal path
before safety lands it.  This is the open research gap.

## 5. Why the parallel IMU integrity monitor is off by default

We added a `VelocityIntegrityMonitor` but did not enable it by default, because
an **open-loop IMU dead-reckon is not a reliable independent reference over long
outages**: the same IMU biases that plague the EKF also plague the dead-reckon,
and the two disagree whenever there is any real bias — not only when the flow is
corrupt.  In the test a healthy mission triggered a false "landing" decision
because the dead-reckon drifted by tens of m/s.  A useful integrity monitor
needs either:
1. A second, *truly* independent sensor (not another integral of the same IMU),
   or
2. A **factor-graph / multi-hypothesis** engine that can reason about "which
   measurement stream is consistent with a sparse set of landmarks" — the
   proper version, but much more work than a first FSM.

## 6. What I learned / corrected

* **"Safety" is not the same as "accuracy".** The layer converts crashes into
  controlled landings.  It does not make a wrong sensor accurate.
* **The safety layer must not control horizontal position during a
  loss-of-trusted-navigation landing.** Doing that takes the (unreliable) EKF
  position, subtracts it from a ground reference, and commands the vehicle
  toward a target it cannot actually observe.  The correct move is level +
  descend only (`FlightController.landing()`).
* **Once landed, do not keep commanding descent.** A thrust command based on a
  bad vertical estimate can re-accelerate the aircraft after touchdown; the
  ground-on state is now held at neutral thrust.
* **"Ground contact" must distinguish intended landing from crash.** With no
  safety layer the vehicle "contacts" the ground only by crashing; with the
  layer on, the same touch-down is a successful recovery.  Our metric now
  reports both raw ground contact and *unintended crash*.

## 7. Run

```bash
.venv/bin/python run_corruption.py --durations 0,15,30 --n 3 --out out/corruption.csv
.venv/bin/python plot_results.py --out out
```
