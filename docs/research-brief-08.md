# Research Brief #08 — Detector Consensus + Mission-Aware Response

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (27/27), consensus policy recommended; mission-aware RTL documented as unsafe for this fault class.

---

## 1. Goals

Two remaining items from brief #07:

1. **Pick a consensus policy** between the landmark detector and the calibrated
   factor graph (they are structurally different: world-geometry angles vs joint
   IMU/flow/GPS/landmark least-squares).
2. **Mission-aware response** — instead of always landing where the fault is
   found, return-to-base (RTL) if battery/geometry safely allow; otherwise land
   now.

## 2. Consensus study (n=6 per fault/policy, duration 45 s, safety ON)

The same 6 random missions were run under every policy.  Cells:

| Fault | none | lm_only | fg_only | **min (OR)** | max (AND) | geom |
|---|---|---:|---:|---:|---:|---:|
| none crash / landed | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 |
| scale1.5 crash / landed | 0 / 0.00 | 0 / 0.00 | 0 / **0.79** | 0 / **0.79** | 0 / 0.00 | 0 / 0.32 |
| bias.1 crash / landed | 0.002 / 0.00 | 0 / 0.26 | 0 / **0.81** | 0 / **0.81** | 0 / 0.25 | 0 / 0.57 |
| bias.25 crash / landed | **0.056** / 0.00 | 0 / 0.28 | 0 / **0.85** | 0 / **0.85** | 0 / 0.28 | 0 / 0.76 |

### Findings

* **`none` is the only policy that crashes** (bias.25 → 0.056).  Every
  independent-detector policy gets crash to zero.
* **`fg_only` and `min` are identical** on these faults: the factor graph is the
  more *sensitive* detector for ramp-bias and scale faults, so its health
  dominates worst-of.  The landmark detector adds value on faults the FG is less
  sensitive to (and provides a faster, geometry-based opinion), but in this grid
  it never goes below the FG on a fault the FG already sees.
* **`max` (AND) is the *least* protective of the two-monitor policies.**  It
  requires both detectors to agree, which re-introduces the landmark detector's
  conservative-but-less-sensitive limit (bias.1/25 → about 0.25–0.28 landed).
  It never crashes, but it tolerates a "minor or moderate" corrupt source that
  the FG alone would have removed.
* **`geom` is the middle ground** — full crash avoidance, high landing rate
  (0.76–0.80 on the hard fault) but with fewer aggressive landings than `min`
  (0.32 vs 0.79 on scale1.5).

### Recommendation

> **Keep `detector_consensus = "min"` (worst-of / OR) as the default.**  It is
> the only two-monitor policy that both (a) flags a fault as soon as *any*
> independent detector is confident and (b) still has zero false alarms on a
> healthy mission.  `geom` is the recommended tunable setting when an operator
> wants to trade a little landing-aggressiveness for mission completion
> (`--mission-aware`-style operating modes).

## 3. Mission-aware RTL — honest result

We implemented a **return-to-base (RTL)** mode:

* Battery-aware feasibility (`can_rtl`) requires enough energy for the cruise
  plus a reserve, **and** an absolute fix (`_can_rtl()` returns False without
  GNSS because the aircraft then does not know where base is in world frame).
* Detector-triggered **source rejection**: as soon as the independent detectors
  warn, the corrupt flow source is removed from the EKF and the
  position/velocity state is hard-reset from GPS when available.
* **Ground-contact friction** added to the dynamics model (a landed aircraft
  now stops instead of skidding forever — previously a "safe landing" could
  keep sliding).

### Result

| Scenario | Outcome | crash | final horizontal position |
|---|---:|---:|---|
| GPS up, bias.25, land-now | landed_safely | 0 | ~60 m from divergence (friction=5) |
| GPS up, bias.25, land-now, friction=20 | landed_safely | 0 | ~15 m |
| GPS up, bias.25, RTL | landed_safely | 0 | ~27 m from base (not at base) |
| GPS outage, bias.25, RTL | landed_safely | 0 | lands immediately (no RTL) |

### Why RTL does not beat immediate landing here

Under a **ramping corrupt velocity source**, the detector fires only after the
corrupt estimate has already begun driving the aircraft off course.  By the
time we reject the source and clean the velocity, the vehicle is no longer
trustworthy enough to navigate back to a specific base.  RTL is therefore a
**guess at best**; immediate landing with a level-and-hold controller plus
ground friction is the safe, predictable response for this fault class.

### Deliverable decision

1. `mission_aware = False` (default) — immediate land remains the safe response
   for a "convincing but wrong" velocity source.
2. **Early source rejection** is now part of the design (fires at the *warn*
   level, not only at fail), and **ground friction** makes the landing metric
   honest.  Both are enabled implicitly and apply to immediate land too.
3. RTL stays as an opt-in research feature (`--mission-aware`) and is gated so
   it can never fire without an absolute fix.

## 4. Next steps

1. **Wider grid (n≥30)** across fault depth × policy to fit a proper crash /
   detection ROC and a cost function for landing-aggressiveness.
2. **Fast rejection** — shorten detector latency (higher FG rate, warn-level
   rejection) so that RTL *can* become safe for faults that only corrupt
   velocity, not absolute position.
3. **RTL for a different fault class** (e.g. battery/actuator degradation where
   navigation is intact) — that is where RTL is genuinely appropriate.

## 5. Run

```bash
.venv/bin/python run_consensus.py --n 6 --duration 45 --seed 191 --out out/consensus.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv out/consensus.csv
```
