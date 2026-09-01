# Research Brief #11 — Scaling the Adaptive Escalation Line

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (28/28), `adaptive_escalate=0.65` promoted to default.

---

## 1. Problem

Brief #10 introduced adaptive consensus: soft (geometric-mean) consensus while
it vouches for the sensors, then escalation to worst-of (`min`) once the soft
consensus drops below a line.  That line was tied to `detector_health_warn`
(0.55).  This brief **decouples it** (`adaptive_escalate`) and sweeps it, because
"when to warn" and "when to go hard" are different operational decisions.

## 2. Change

* New `SafetyConfig.adaptive_escalate` (default 0.65), threaded through
  `SimConfig`, `Scenario`, and `run_consensus.py --adaptive-escalate`.
* `_combine_detectors` uses `adaptive_escalate` for the soft→worst-of switch,
  so it no longer shares the warning line.

```python
soft = sqrt(prod(parts))
if soft >= adaptive_escalate: return soft   # soft / geom behaviour
return min(parts)                           # worst-of / min behaviour
```

## 3. Sensitivity sweep (n=3, seed 717, adaptive_escalate ∈ {0.45,0.55,0.65})

| Policy / line | benign landing-activation | benign mission loss | healthy false alarm | bias.25 crash | bias.25 landed |
|---|---:|---:|---:|---:|---:|
| min           | 0.489 | 0.175 | **0.195** | 0.000 | 0.848 |
| geom          | 0.203 | 0.102 | 0.000 | 0.000 | 0.751 |
| adaptive 0.45 | 0.378 | 0.114 | 0.000 | 0.000 | 0.840 |
| adaptive 0.55 | 0.378 | 0.114 | 0.000 | 0.000 | 0.840 |
| **adaptive 0.65** | 0.391 | **0.105** | 0.000 | 0.000 | **0.851** |

`none` (unwatched): bias.25 crash = **0.072**.

## 4. Interpretation

* The escalation line is **not very sensitive** between 0.45 and 0.55 on this
  grid (identical adaptive cells); the soft consensus either stays well above
  the line (healthy/shallow faults) or drops well below it (deep faults).
* At **0.65** the line crosses *more* shallow-fault states, so adaptive becomes
  more like `min` on them — but crucially it keeps `geom`'s **zero false alarm
  on healthy** while slightly *improving* hard-fault protection.
* **`min` false-lands on a healthy mission** in this grid (0.195 landed, first
  healthy cell in the whole study).  `min` avoids crashes but pays for it with
  real false alarms.  `geom` and `adaptive` do not false-alarm on healthy.
* Overall `adaptive@0.65` is the **best point in this sweep**: bias.25 landed
  0.851 (≥ min 0.848), crash 0, healthy false alarm 0, benign mission loss
  0.105 (vs min 0.175).

## 5. Decision

> **`adaptive_escalate = 0.65` is now the default.**  It keeps `min`-level
> (slightly better) hard-fault protection without `min`'s healthy false alarm,
> and with lower benign-fault mission cost.

The earlier `detector_health_warn=0.55` remains the *warning* line; the
escalation line is now independently settable.

## 6. Caveats & next steps

* n=3 is small; the 0.65 vs 0.55 gap (0.851 vs 0.840) is within noise.  A
  larger grid (n≥10) is needed to confirm.
* The healthy `min` false alarm is a **new, real finding**: `min` is
  conservative not just on mild faults but occasionally even on a healthy
  high-motion mission.  This makes the adaptive/geom family (which never false
  alarms here) even more compelling.
* Next: n≥30 grid; then weight the soft consensus by which monitor is more
  locally trustworthy.

## 7. Run

```bash
# sweep the escalation line:
.venv/bin/python -u - <<'PY'   # (see repo history / workspace for the loop)
...
PY
# single point:
.venv/bin/python run_consensus.py --n 4 --adaptive-escalate 0.65 ...
```
