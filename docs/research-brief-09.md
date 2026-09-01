# Research Brief #09 — Fault-Depth Sweep & Decision-Cost ROC

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (27/27), decision-cost tradeoff quantified.

---

## 1. Goal

Brief #07/#08 established that detector *consensus* is live.  The remaining
question is the **operational cost**: independent monitors are *conservative* —
they will land the aircraft on a fault that would not necessarily crash it.  A
purely "crash = bad" metric hides this.  This brief sweeps fault *depth* and
produces a decision-cost ROC: **how much mission do we sacrifice to prevent a
crash?**

## 2. Method

* Same random missions across every cell (paired comparison, n=4).
* Fault depths: none, scale1.2 / 1.5 / 1.8 (velocity magnitude error),
  bias.05 / 0.1 / 0.25 (ramping velocity bias).
* Policies: none, lm_only, fg_only, min (worst-of), geom (soft consensus).
* Metrics per cell: unintended crash, landed-safely fraction, in-bounds.

## 3. Results by fault depth (n=4)

| Fault | none (unwatched) | lm_only | fg_only / min | geom |
|---|---|---|---|---|
| none | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| scale1.2 | 1.00 / 0.00 | 1.00 / 0.00 | 0.73 / 0.49 | 1.00 / 0.00 |
| scale1.5 | 1.00 / 0.00 | 0.92 / 0.10 | 0.75 / 0.80 | 0.76 / 0.45 |
| scale1.8 | 0.88 / 0.00 | 0.96 / 0.06 | 0.68 / 0.76 | 0.60 / 0.71 |
| bias.05 | 0.41 / 0.00 | 0.53 / 0.25 | 0.73 / 0.70 | 0.72 / 0.39 |
| bias.1 | 0.25 / **0.016** | 0.50 / 0.26 | 0.77 / 0.81 | 0.71 / 0.56 |
| bias.25 | 0.13 / **0.033** | 0.56 / 0.28 | 0.75 / 0.85 | 0.73 / 0.78 |

*(shown as in-bounds / landed-safely fraction)*

## 4. Decision-cost summary (from the sweep CSV)

We define:

* **benign = none, scale1.2, scale1.5, scale1.8** — faults where the *unwatched*
  vehicle stays in-bounds (≥0.88) for the whole mission, so acting on them is a
  pure mission-interruption cost.
* **bias.05 is NOT benign**: it never crashes but already leaves bounds
  (unwatched in-bounds 0.41), so a landing is a justified safety response.

| Policy | benign landing-activation | benign mission loss | bias.25 crash | bias.25 landed |
|---|---:|---:|---:|---:|
| none             | 0.000 | 0.030 | **0.033** | 0.000 |
| lm_only          | 0.041 | 0.030 | 0.000 | 0.282 |
| fg_only / min    | 0.512 | 0.212 | 0.000 | 0.845 |
| geom             | 0.290 | 0.159 | 0.000 | 0.777 |

## 5. Reading the ROC (lower-left is better)

* **`none`** is the only policy that *crashes* (bias.25 → 3.3%).
* **`lm_only`** is the **minimum-interruption** safe point: it prevents all
  crashes in this grid while only landing on ~4% of benign missions (mission
  loss ~3%).  Its protection is *weaker*, though — on the hardest fault it only
  lands 28% of the time, so it tolerates more degraded-but-not-crashed flight.
* **`fg_only` / `min`** is the **safety-max** point: full protection (85% landed
  on bias.25, crash 0) at a real cost (~51% of benign scale-fault missions get
  landed and ~21% of mission time is lost).
* **`geom`** is the **balanced** point: crash 0, 78% landed on bias.25, with
  only ~29% benign-activation and ~16% mission loss.

## 6. Recommendations

1. **Keep `min` as the simulator default** for a safety-critical autonomy
   stack (maximum crash protection, no false alarm on a healthy mission).
2. **Expose `geom` as the "mission-completion-aware" mode**: 70% of `min`'s
   mission cost for ~92% of its hard-fault protection.  This is the policy an
   operator would choose for a search/rescue mission where a slightly degraded
   flight is preferable to a needless landing.
3. **`lm_only` is the minimum-obstruction option** but should only be used when
   the mission is more important than protecting the aircraft from a
   degraded-but-not-yet-fatal source; it already eliminates crashes in this
   grid.

## 7. Next steps

1. **n≥30 per cell** to tighten the ROC confidence intervals.
2. **Adaptive consensus** — switch `geom`→`min` as fault depth grows, since the
   sweep shows the *correct* policy depends on whether the fault is a
   scale error (survivable) or a ramp bias (diverges).
3. **Fault-depth cost function** for a mission planner: `land_now` vs `continue`
   using the in-bounds trajectory and the energy model.

## 8. Run

```bash
.venv/bin/python run_consensus.py --n 4 --duration 45 --seed 313 \
  --faults "none,scale1.2,scale1.5,scale1.8,bias.05,bias.1,bias.25" \
  --policies "none,lm_only,fg_only,min,geom" --out out/consensus_sweep.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv out/consensus_sweep.csv
```
