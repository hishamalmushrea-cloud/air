# Research Brief #15 — Self-Measured Frame Trust (angular diversity, not count)

**Date:** 2026-09-01
**Project:** AIR Lab
**Status:** implemented, tested (33/33).  `adaptive_veto_trust` is a more
principled veto guardrail; identical to `adaptive_veto` in the current world
(only differentiates under a clustered-landmark world we do not yet simulate).

---

## 1. Goal

Brief #13/#14 fixed the false-landing liability with an **asymmetric veto**: a
thin detector may trigger but may not veto.  The "thin" test used a binary
availability floor (raw landmark count / 3, factor count / min_keyframes).  That
is coarse: a camera can report **many** landmarks that all lie in a *tiny
angular cluster* — high count, but almost no geometric leverage.  This brief
replaces the binary floor with a **continuous, self-measured frame trust**.

## 2. The trust model (transparent, no black box)

**landmark trust** = `count/3 × angular_diversity`, where

```
angular_diversity = clamp( RMS(pairwise angles between observed bearings) / 1.2 rad )
```

Full spread → ~1.0; a tight cluster of points in one direction → low diversity
→ weak voice.  This is the standard geometric-dilution idea in a camera setting.

**factor-graph trust** = `flow_factors / min_keyframes × converged` (an
under-determined or unconverged graph is a weak opinion even if it has many
residuals).  Both are **measurement-availability only** — never the detector's
own verdict, so a faulty detector cannot down-weight itself.

New policy `adaptive_veto_trust`: like `adaptive_veto` but a "credible low" is
any detector that is (a) below warn and (b) has trust ≥ 0.45, and the healthy
veto voice is the highest trust-enabled detector rather than a stale thin one.

## 3. Unit validation

`test_frame_trust_discriminates_spread_vs_clustered`:
- 3 landmarks in well-separated directions → trust > 0.5.
- 3 landmarks in a tight cluster → trust < half of the spread case.

So the model does **not** confuse "many" with "informative".

## 4. End-to-end grid (n=3, seed 909, landmark outage 10–30 s)

| Fault | none | adaptive_veto | adaptive_veto_trust |
|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 |
| scale1.5 | 0.000 | 0.765 | 0.765 |
| bias.25 | 0.025 crash | 0.844 | 0.844 |

`adaptive_veto_trust` is **identical to `adaptive_veto`** on this grid.

## 5. Honest reading

* **Why identical?**  In our synthetic world, when landmarks are visible they
  are always well spread (the field is deliberately dense and geometrically
  rich), so `angular_diversity ≈ 1` and trust ≈ count fraction — exactly the old
  availability test.  During the outage the count is 0 so both policies treat it
  as thin and (correctly) do not let it veto.
* **What it buys:** the trust model now has the *capacity* to down-weight a
  cluster-of-points camera even when it reports many features.  This world does
  not yet produce such a frame, so the two policies coincide on every measured
  cell.
* **Why keep it:** it is a strictly more informative guardrail with no regression
  and no false alarm.  It is honest to record that its distinguishing behaviour
  is **untested end-to-end** rather than to claim a benefit we did not measure.

## 6. Decision

* Keep `adaptive` as the default (zero regression).
* Keep `adaptive_veto` as the recommended feature-poor mode (confirmed at n=6 in
  brief #14).
* Expose `adaptive_veto_trust` as the **next refinement** for a real camera where
  geometry matters; we cannot justify promoting it to default until we can
  simulate a clustered/feature-degenerate frame.

## 7. Next step (what would exercise trust)

1. **Inject a clustered-degenerate camera** (many landmarks within a small
   angular cone) as a first-class failure mode, then compare `adaptive_veto` vs
   `adaptive_veto_trust` there.
2. **Learned per-frame confidence** on top of the analytic trust (the current
   model is a hand-set geometry heuristic).

## 8. Run

```bash
.venv/bin/python -u run_consensus.py --n 3 --duration 40 --seed 909 \
  --landmark-outage 10-30 \
  --faults "none,scale1.5,bias.25" \
  --policies "none,adaptive_veto,adaptive_veto_trust" --adaptive-escalate 0.65 \
  --out out/consensus_veto_trust.csv
```
