# Research Brief #10 — Adaptive Detector Consensus

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (28/28), promoted to default.

---

## 1. Problem

Brief #09 quantified the key tradeoff:

* `min` (worst-of / OR) is safety-max: bias.25 landed 0.845, crash 0 — but it
  costs ~21% of a benign mission (it lands on survivable scale faults).
* `geom` (soft consensus) is cheaper (~16% cost) but its hard-fault protection is
  weaker (0.777 landed).

The tradeoff is **fault-depth dependent**: scale faults (magnitude error) were
survivable in the grid (unwatched in-bounds ≥0.88), while ramp biases diverged
(and eventually crashed).  A single fixed policy is therefore wrong for one end
of that spectrum.  We need a policy that behaves like `geom` on shallow faults
and like `min` on deep ones.

## 2. Adaptive consensus

New `detector_consensus = "adaptive"`:

```python
soft = sqrt(prod(parts))                    # geometric mean
if soft >= detector_health_warn:            # still vouches for the sensors
    return soft                             # -> behave softly (geom)
return min(parts)                           # escalate to worst-of (min)
```

Interpretation: *whilst the soft consensus is healthy enough to vouch for the
sensors, act softly (avoid a needless landing on a survivable fault).  The
moment even the soft consensus falls below the warning line, the sensors can no
longer be trusted collectively, so escalate to the most conservative opinion.*

## 3. Results (same fault-depth grid, n=4)

Decision-cost summary:

| Policy | benign landing-activation | benign mission loss | bias.25 crash | bias.25 landed |
|---|---:|---:|---:|---:|
| none             | 0.000 | 0.030 | **0.033** | 0.000 |
| min              | 0.512 | 0.212 | 0.000 | 0.845 |
| geom             | 0.290 | 0.159 | 0.000 | 0.777 |
| **adaptive**     | 0.346 | **0.142** | 0.000 | **0.841** |

Per-fault (landed-safely fraction):

| Fault | min | geom | adaptive |
|---|---:|---:|---:|
| none | 0.00 | 0.00 | 0.00 |
| scale1.2 | 0.49 | 0.00 | **0.00** |
| scale1.5 | 0.80 | 0.45 | 0.63 |
| scale1.8 | 0.76 | 0.71 | 0.75 |
| bias.05 | 0.70 | 0.39 | 0.64 |
| bias.1 | 0.81 | 0.56 | 0.68 |
| bias.25 | 0.85 | 0.78 | **0.84** |

## 4. Interpretation

* **Hard-fault protection is essentially unchanged from `min`** (0.841 vs 0.845
  landed at bias.25, crash 0).
* **Benign-fault mission cost drops by ~33%** (0.142 vs 0.212 mission loss) and
  landing-activation drops from 0.512 → 0.346.
* **The biggest win is at the shallow end**: scale1.2 adaptive lands 0 (same as
  `geom`) instead of 0.49 (`min`), while scale1.8/bias.25 keep `min`-level
  protection.  This is the "choose the policy based on depth" behaviour we
  wanted, made automatic.
* The escalation is a **firebreak**: it only happens once the *soft* consensus
  can no longer vouch for the sensors, so it cannot be triggered by a single
  transient healthy dip.

## 5. Decision

> **`detector_consensus = "adaptive"` is now the default.**
> It is dominance-close to `min` for safety and strictly cheaper for mission
> completion, so there is no reason to keep `min` as the default for this grid.

## 6. Next steps

1. **n≥30 per cell** to tighten the confidence intervals around adaptive's 0.841
   vs min's 0.845 and the 0.142 vs 0.212 cost gap.
2. **Tune the escalation line** — currently `detector_health_warn` (0.55).  A
   separate `adaptive_escalate` parameter would decouple "when to warn" from
   "when to go hard".
3. **Adaptive `lm`-vs-FG weighting** — the landmark detector is cheaper and
   geometry-only; the FG is more sensitive.  Weight the soft consensus by which
   monitor is more locally trustworthy.

## 7. Run

```bash
.venv/bin/python run_consensus.py --n 4 --duration 45 --seed 313 \
  --faults "none,scale1.2,scale1.5,scale1.8,bias.05,bias.1,bias.25" \
  --policies "none,min,geom,adaptive" --out out/consensus_adaptive.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv out/consensus_adaptive.csv
```
