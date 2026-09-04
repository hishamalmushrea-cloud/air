# Research Brief #18 — Sparse (Under-Determined) Factor Graph as a First-Class Failure

**Date:** 2026-09-03
**Project:** AIR Lab
**Status:** implemented, tested.  A sparse/under-determined factor graph is now
both injectable and its *trusted-weakness* behaviour is measured.

---

## 1. Goal

Brief #17 made landmark *frame trust* learned.  The factor-graph side was kept
analytic because its reference value is 1.0 by construction.  This brief adds
the missing first-class failure for the FG: a **sparse graph** — the graph keeps
running but receives **no flow factors** (and therefore has almost nothing to
measure the flow fault against).  The EKF flow aiding is deliberately left
intact, so the experiment isolates the *detector's weakness*, not the
navigation.

## 2. Implementation

- New `SimConfig.factorgraph_flow_outage: Optional[(start,end)]` and
  `Scenario.factorgraph_flow_outage`.
- New CLI `--fg-flow-outage start-end` (applied to all faults).
- In the FG loop, when the window is active the graph's `flow_here` is `None`
  (no flow factors); the EKF still receives flow as normal.

## 3. Structural check (instrumented, scen1)

During the sparse window (8–28 s):

```
t=5.00  comp=6  health=0.965  trust=1.000   (before)
t=14.00 comp=0  health=1.000  trust=0.000   (inside sparse window)
t=35.00 comp=6  health=0.900  trust=1.000   (after recovery)
```

So the graph is **under-determined (comp=0, trust=0) but HEALTHY (health≈1.0)**.
The trust/availability guard correctly treats it as a thin voice — it may not
veto — while a healthy landmark keeps its full veto.

## 4. Results (n=3, seed 909, sparse FG 8–28 s)

### 4a. Healthy mission

Every policy (none, lm_only, fg_only, adaptive, adaptive_veto,
adaptive_veto_trust) completes: `in_bounds=1.000`, crash 0.000, landed 0.000.
**No false alarm from the under-determined graph.**

### 4b. Real faults

| fault | lm_only | fg_only | adaptive | adaptive_veto | adaptive_veto_trust |
|---|---|---|---|---|---|
| scale1.5 | 0.000 | 0.529 | 0.529 | 0.529 | 0.529 |
| bias.25 | 0.170 | 0.844 | 0.844 | 0.844 | 0.844 |

- The **FG still detects** scale1.5 (0.529) and bias.25 (0.844) — because the
  fault is persistent; once the graph recovers (after 28 s) the flow residual
  appears and the FG triggers.
- **`adaptive_veto_trust` is identical to the other detector arms** here (as
  expected): no false alarm, no lost detection, and on this grid the landmark
  arm is weaker than the FG arm for bias.25 (0.170 vs 0.844), which is why FG
  remains important.

## 5. Honest reading / design gap

A **transient** flow fault fully contained inside the sparse window would be
invisible to the FG during that window (no flow factors → no flow residual).
The trust guard's job is not to make the FG detect during its own outage, but to
ensure the FG is treated as **thin** (may trigger, may not veto) so it cannot
suppress a healthy landmark.  We do not currently have a transient fault preset
to measure that worst case end-to-end; the persistent-fault result above does
not exercise it.  This is recorded as a limitation rather than a claim.

## 6. Tests

- `test_sparse_factorgraph_is_clean_healthy_and_detects_bias` — end-to-end:
  healthy+sparse completes (no false land), persistent bias.25+sparse still
  lands (≥0.4) with crash 0.
- Full suite: **37/37 OK** (after this brief).

## 7. Decision

- `adaptive_veto_trust` is unchanged and remains the recommended
  feature-degeneracy mode.
- A sparse/under-determined FG is confirmed **non-false-alarming** and still
  detects persistent faults after recovery.
- Transient-fault-inside-sparse-window is the next honest experiment (needs a
  transient fault preset).

## 8. Run

```bash
.venv/bin/python -u run_consensus.py --n 3 --duration 40 --seed 909 \
  --fg-flow-outage 8-28 \
  --faults "none,scale1.5,bias.25" \
  --policies "none,lm_only,fg_only,adaptive,adaptive_veto,adaptive_veto_trust" \
  --adaptive-escalate 0.65 \
  --out out/consensus_fgsparse_faults.csv
```
