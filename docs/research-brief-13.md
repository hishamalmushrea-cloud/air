# Research Brief #13 — Detector-Poor Flight & the Asymmetric Veto Guardrail

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (30/30).  `adaptive_veto` validated as the correct
asymmetric weighting; symmetric weighting (`adaptive_weighted`) shown to be a
net regression under camera outage.

---

## 1. Goal

Brief #12 added an **availability-weighted** soft consensus.  On the dense-field
grid it was identical to `adaptive` (both weights ≈1), so we could not tell
whether it was a guardrail or a liability.  This brief **injects a feature-poor
camera** (`landmark_outage` window) to actually exercise it — and finds an
important negative result.

## 2. The negative result (why symmetric weighting is wrong)

Under a landmark outage the camera holds its **last healthy score** (unknown is
not bad), but its availability weight drops toward 0.  With a *symmetric*
weighted geometric mean that down-weighting **removes the healthy veto**: the
noisier factor graph becomes the only voice, so on a healthy mission a modest
FG dip can now escalate and **false-land** the aircraft.

Poor-landmark grid (n=4, seed 505, landmark outage 10–35 s):

| Fault | none | adaptive | adaptive_weighted | **adaptive_veto** |
|---|---|---|---|---|
| none | 0.000 | 0.000 | **0.108** | **0.000** |
| scale1.5 | 0.000 | 0.697 | 0.710 | 0.697 |
| bias.1 | 0.000 | 0.786 | 0.786 | 0.786 |
| bias.25 | **0.049** | 0.846 | 0.846 | 0.846 |

*(landed-safely fraction; none crash shown on bias.25)*

`adaptive_weighted` protects the real faults just as well, but **false-lands on
a healthy mission** (0.108) under camera outage — precisely the scenario the
guardrail was meant to protect.

## 3. The asymmetric fix

The lesson is that **availability must not reduce a detector's ability to veto a
false alarm — only its ability to trigger one**.

New policy `adaptive_veto`:

```python
soft = unweighted_geom(a)              # healthy voices keep FULL weight
if soft >= escalate: return soft
credible_low = [ a_i where a_i < warn AND availability_i >= 0.5 ]
if not credible_low:                   # only thin detectors were low
    return max(healthy_voice) or warn # unknown, not bad -> do not react
return min(min(credible_low), soft)    # a credible low detector CAN escalate
```

So a thin detector cannot *cause* a landing on its own, but a healthy (even if
stale) detector can always *prevent* one.

## 4. Validation

| Grid | healthy (outage) | bias.25 landed | crash |
|---|---|---|---|
| adaptive | 0.000 | 0.846 | 0 |
| adaptive_weighted | 0.108 | 0.846 | 0 |
| **adaptive_veto** | **0.000** | 0.846 | 0 |

`adaptive_veto` keeps every real-fault protection of `adaptive` while removing
the false alarm that symmetric weighting introduced.  It also passes the
unit-level guardrail test (healthy stale landmark + dip FG → healthy veto wins;
credible low FG → escalates).

## 5. Decision

* Keep `adaptive` as the default (zero regression, zero false alarm).
* `adaptive_veto` is the recommended mode **when feature-poor flight is
  expected** (indoor corridors, low light, obstacle-close cameras): it gives
  the same protection as `adaptive` *without* the symmetric-weighting liability.
* `adaptive_weighted` / `weighted` are **not recommended as built**: the sweep
  shows symmetric availability weighting can trade a healthy mission for a
  false landing during a camera outage.

## 6. Why the unit test and the grid agree

The design is now asymmetric on purpose: the availability weight gates only the
*trigger* half of the decision (whether a low score is credible enough to act
on).  The *veto* half always keeps the highest healthy voice, regardless of how
many landmarks the camera happened to see.

## 7. Next steps

1. **Per-frame trust learning** — replace the hand-set availability floor (0.5)
   with a learned confidence that knows when a camera frame is informative.
2. **Wider feature-poor grid (n≥10)** and other outage types (sparse FG).
3. **n≥30 statistical confirmation.**

## 8. Run

```bash
.venv/bin/python -u - <<'PY'   # inject landmark outage + compare policies
...
PY
# (see repo history for the exact loop; CSV: out/consensus_veto.csv)
```
