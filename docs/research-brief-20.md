# Research Brief #20 — Transient Fault Fully Inside a Sparse-FG Outage

**Date:** 2026-09-03
**Project:** AIR Lab
**Status:** implemented, tested.  This is the worst case we had deliberately
left unmeasured in brief #18: a *transient* velocity-aiding fault that appears
and disappears while the factor graph has **no flow factors**, so the graph
cannot see it at all.  The result is subtle and important.

---

## 1. Goal

Brief #18 used only persistent faults.  A persistent fault is still present
when the sparse-FG window closes, so the graph can detect it by the flow
residual it leaves behind.  A **transient** fault that fully lives and dies
inside the sparse window leaves no residual for the graph to catch.  This brief
builds that fault and measures which consensus arms survive.

## 2. Implementation

- New **transient** velocity-bias fault: `flow_bias_shift` (m/s) applied only
  inside `flow_bias_window` and reset to 0 outside it.  Unlike the persistent
  `flow_bias_ramp`, it does **not** accumulate, so the source genuinely recovers
  the instant the window closes.
- Wired through `SensorConfig`, `SimConfig`, `Scenario`, `FAULT_PRESETS`
  (`transient.3` = 3 m/s bias, 8–18 s) and `--transient-window` CLI override.
- Combined with `--fg-flow-outage 5-28` so the sparse-FG window (5–28 s) fully
  contains the fault (8–18 s).

## 3. Mechanism found (the important part)

On scen0, `lm_only` (worst-of, landmark only) crashes (`crash` time-fraction
0.258) even though the safety state stays `CRUISE` for the whole run.  The
explanation is **not** a bad landing:

```
flow_rejected = True
reason        = independent_detector_warn
final safety  = CRUISE / nominal
```

The landmark detector dipped below warn during the transient, which triggered
**early flow-source rejection**.  Because the fault is *transient*, the source
recovers, but we have already cut it off — navigation is left on GPS/baro only
during the recovery and the aircraft crashes.  This is the **false-rejection
penalty**: acting on a transient warn can be worse than not acting, if the fault
would have healed.

## 4. Results (n=6, seed 909, transient 8–18 s inside sparse-FG 5–28 s)

| policy | crash (time-frac) | crash events | landed | in-bounds | outcome |
|---|---|---|---|---|---|
| none | 0.000 | 0/6 | 0.000 | 0.679 | all completed |
| lm_only (worst-of) | **0.097** | **2/6** | 0.173 | 0.619 | 2 crash |
| fg_only | 0.005 | 0/6 | 0.000 | 0.742 | all completed |
| adaptive | 0.016 | 0/6 | 0.244 | 0.798 | 3 landed |
| adaptive_veto_trust | **0.000** | **0/6** | 0.000 | 0.681 | all completed |

Reading:
- **`fg_only` is blind** (0 landed): as expected, a fault hidden inside the
  sparse window never reaches the graph.
- **`lm_only` is harmful**: reacting to a *transient* warn with worst-of causes
  **2/6 crashes** by rejecting a recovering source.
- **`adaptive` is a middle path**: it keeps ~0.80 in-bounds and converts the
  worst cases into controlled landings (0.244) with one small scrape (time-frac
  0.016), no crash event.
- **`adaptive_veto_trust` is the safest here**: it does **not** false-react to
  the transient, so it neither rejects the recovering source nor crashes.  It
  also detects *persistent* faults (brief #19: bias.25 0.817 landed, 0 crash),
  so it is not "blind" — it is selectively conservative.

## 5. Honest reading / design gap

This is a genuine **reaction policy** problem, not a detector-detection problem:

* Transient fault + `min`/`lm_only` → early rejection converts a would-recover
  event into a crash.
* Persistent fault + `min`/`lm_only` → early rejection is correct and save lives.
* The safety layer currently **cannot tell the two apart** before acting.

`adaptive_veto_trust` accidentally wins on the transient case because it is
conservative about *credibility* (thin/low-but-uncertain evidence).  But that is
not the same as a principled **"is this fault persisting?"** decision.  The
principled next step is a **persistence gate on flow-source rejection**: only
reject a velocity source after the independent detectors have been *below warn
for a sustained window*, not at the first tick.  This would let a transient
fault heal without stripping the source and would preserve immediate rejection
for genuine persistent faults.

## 6. Tests

- `test_transient_in_sparse_fg_does_not_crash_under_trust`: scen0 reproduces the
  exact mechanism — `lm_only` crashes (crash > 0), `adaptive_veto_trust` does
  not (crash == 0).
- Full suite: **38/38 OK** (after this brief).

## 7. Decision

- Keep `adaptive` as the default (balanced: stops up the persistent-fault
  crashes, reduces transient reaction compared with `min`).
- `adaptive_veto_trust` remains the recommended mode for feature-degenerate /
  transient-heavy flight.
- **Do not fix the false-rejection penalty in this brief** — it is a real design
  decision (persistence-gated rejection) that deserves its own experiment.

## 8. Run

```bash
.venv/bin/python -u run_consensus.py --n 6 --duration 40 --seed 909 \
  --fg-flow-outage 5-28 \
  --faults transient.3 \
  --policies none,lm_only,fg_only,adaptive,adaptive_veto_trust \
  --adaptive-escalate 0.65 \
  --out out/consensus_transient_fgsparse_n6.csv
```
