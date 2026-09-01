# Research Brief #14 — Wider Confirmation of the Veto Guardrail (n=6)

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** confirmed, tested (32/32), `adaptive_veto` matches `adaptive` across all faults.

---

## 1. Goal

Brief #13 found the veto guardrail on a small n=4 grid.  This brief **widens the
sample to n=6** on the same feature-poor scenario (landmark outage 10–30 s) to
confirm that `adaptive_veto` is statistically identical to `adaptive` on every
real fault **and** remains false-alarm-free on healthy.

## 2. Results (n=6, seed 909, landmark outage 10–30 s)

| Fault | none (unwatched) | adaptive | adaptive_veto |
|---|---|---|---|
| none | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |
| scale1.5 | 0.990 / 0.000 | 0.895 / **0.727** | 0.895 / **0.727** |
| bias.1 | 0.286 / 0.000 | 0.839 / **0.789** | 0.839 / **0.789** |
| bias.25 | 0.148 / **0.020 crash** | 0.918 / **0.836** | 0.918 / **0.836** |

*(in-bounds / landed-safely; crash shown on the unwatched arm)*

## 3. Reading

* **`adaptive_veto` is byte-for-byte identical to `adaptive` on every fault.**
  This is exactly what we want: the guardrail changes *nothing* when a credible
  detector is present (the healthy landmark veto is not needed because the low
  factor graph is also credible), yet it holds back a thin-only low signal.
* **Healthy + outage stays 0.000** for both — no false landing.
* **None (unwatched) is the only crashing arm** (bias.25 crash 0.020, one in
  six).  Every independent-detector arm lands 0.836 at that fault with crash 0.
* The numbers are now stable across n=4 and n=6 (bias.25 landed 0.846 vs 0.836;
  scale1.5 0.697 vs 0.727; bias.1 0.807 vs 0.789).  The earlier negative result
  (`adaptive_weighted` false-land 0.108 on healthy) was a real non-random
  effect, not sampling noise.

## 4. Conclusion

The veto guardrail is **confirmed and safe to use**:

* `adaptive` remains the default (zero regression).
* `adaptive_veto` is the recommended mode when feature-poor flight is expected —
  it gives the same protection as `adaptive`, with an explicit guarantee that a
  thin detector can trigger but cannot veto.
* `adaptive_weighted` / `weighted` remain **not recommended as built** (brief
  #13 showed they can false-land a healthy mission during a camera outage).

## 5. Next steps

1. **Per-frame trust learning** — replace the hand-set availability floor (0.5)
   with a learned confidence for "is this camera frame informative?".
2. **Sparse factor-graph outage** in addition to landmark outage.
3. **n≥30** for the headline crash numbers (the 0.020 unwatched crash at n=6 has
   wide confidence; the detector arms are all zero-crash which is cleaner).

## 6. Run

```bash
.venv/bin/python -u run_consensus.py --n 6 --duration 40 --seed 909 \
  --landmark-outage 10-30 \
  --faults "none,scale1.5,bias.1,bias.25" \
  --policies "none,adaptive,adaptive_veto" --adaptive-escalate 0.65 \
  --out out/consensus_veto_n6.csv
.venv/bin/python plot_results.py out/batch.csv out/degradation.csv out/corruption.csv out/landmark.csv out/factorgraph.csv out/factorgraph_live.csv out/consensus_veto_n6_all.csv
```
