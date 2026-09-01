# Research Brief #12 — Availability-Weighted Detector Consensus

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (28/28); design-in protection, inactive under the current dense field.

---

## 1. Goal

The adaptive consensus gives each independent detector an **equal** voice in the
soft (geometric-mean) opinion.  That is wrong when one detector has almost no
local data: a camera in a feature-poor area, or a factor graph with too few
factors, is an *under-determined* voice and should not be able to force (or
resist) a safety reaction on thin evidence.

We want the soft consensus to weight each detector by **how much usable local
data it actually has**, not by how loud its (possibly wrong) verdict is.

## 2. Design — availability weight, not agreement weight

The crucial design rule:

> **A faulty detector must not be able to down-weight itself and hide the fault.**

So the weight comes only from **measurement availability**, never from the
detector's own verdict:

* **landmark**: `clip(observed_landmarks / 3, 0, 1)` — how many landmarks were
  actually seen in the current frame (a camera pointing at empty space is a
  weak voice).
* **factor graph**: `clip(flow_components / min_keyframes, 0, 1) * converged` —
  whether the graph had enough factors to be well-determined, and whether the
  optimisation actually converged.  An uncalibrated graph gets weight 0.

The soft consensus becomes the **weighted geometric mean**:

```python
soft_w = exp(sum_i w_i * log(a_i))        # w normalized to sum 1
```

Policies added: `"weighted"` (soft only) and `"adaptive_weighted"` (weighted
soft with the worst-of escalation line).

## 3. Result (n=4, fault-depth grid, seed 909)

`adaptive` vs `adaptive_weighted`:

| Fault | none (unwatched) | adaptive | adaptive_weighted |
|---|---|---|---|
| none | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| scale1.2 | 1.00 / 0.00 | 0.97 / 0.13 | 0.97 / 0.13 |
| scale1.5 | 0.97 / 0.00 | 0.90 / 0.75 | 0.90 / 0.75 |
| scale1.8 | 0.97 / 0.00 | 0.96 / 0.84 | 0.96 / 0.84 |
| bias.05 | 0.44 / 0.00 | 0.93 / 0.70 | 0.93 / 0.70 |
| bias.1 | 0.26 / 0.003 | 0.78 / 0.82 | 0.78 / 0.82 |
| bias.25 | 0.13 / 0.019 | 0.88 / 0.86 | 0.88 / 0.86 |

*(in-bounds / landed-safely)*

## 4. Honest reading

`adaptive` and `adaptive_weighted` are **identical** on this grid.  The reason
is clear and reassuring: the current landmark field is dense (≥3 landmarks
visible through the whole flight) and the factor graph always has ≥
`min_keyframes` factors, so **both weights are ~1** and the weighted mean
reduces to the unweighted mean.

This is the correct outcome for the *design*:
* It adds **no cost** in a well-observed environment — the stack behaves exactly
  as before.
* It is a **guardrail** for feature-poor or over-AUAV rare conditions (indoor
  corridor, low-height forest, camera-blink during a turn), where a thin
  detector would otherwise carry the same authority as a well-informed one.

To *exercise* the weighting we would need to inject a detector-poor scenario
(e.g. a landmark-outage window or a graph with few keyframes); that is the
natural next experiment rather than a change to the current results.

## 5. Decision

* Keep `adaptive` as the default (it is the characterised safe point).
* Expose `adaptive_weighted` / `weighted` as **operating modes** for
  feature-poor environments (e.g. low-light or obstacle-close flight), where
  availability weighting genuinely matters.
* The weighting is a **design-in safety property**, not a parameter to tune
  for these results.

## 6. Next steps

1. **Inject detector-poor scenarios** (landmark outage window, sparse factor
   graph) to produce a head-to-head where `adaptive_weighted` can actually be
   exercised.
2. **Learn the "trust this frame" model** — replace the hand-set availability
   weight with a learned per-frame confidence that knows when a camera frame is
   informative.
3. **n≥30** statistical grid.

## 7. Run

```bash
.venv/bin/python run_consensus.py --n 4 --duration 45 --seed 909 \
  --faults "none,scale1.2,scale1.5,scale1.8,bias.05,bias.1,bias.25" \
  --policies "none,adaptive,adaptive_weighted" --out out/consensus_weighted.csv
```
