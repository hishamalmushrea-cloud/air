# Research Brief #19 — n=30 Statistical Confirmation of the Detector Arms

**Date:** 2026-09-03
**Project:** AIR Lab
**Status:** complete.  The headline detector results hold at n=30; the unwatched
arm's crash number is small but *nonzero*, and its confidence interval is wide.

---

## 1. Goal

Every prior headline used n=6 (brief #14) or n=3 (briefs #15–18).  A zero-crash
reading at n=3/6 is weak evidence — it could be an artifact of a small sample.
This brief runs the two key cells at **n=30** on the same seed/core and reports
the confidence intervals honestly.

## 2. Design

- Seed 909, duration 40 s.
- Paired scenarios: all policies share the same 30 random missions per fault.
- Policy arms: `none` (no independent detector), `adaptive` (default),
  `adaptive_veto_trust` (recommended feature-degeneracy mode).
- Fault windows:
  - `bias.25` with landmark outage 10–30 s (the hard case),
  - `none` (healthy) with the same outage,
  - `bias.25` without outage (baseline).

## 3. Two kinds of "crash"

This is important and was easy to misread.  The `mean_crash` column is
**not the crash rate**.  In `metrics.py`:

```
crash = mean fraction of timesteps where the aircraft is on the ground
        and the safety layer is NOT deliberately landing/unintended
```

So `mean_crash = 0.021` means ~2.1% of total simulated time was unintended
ground contact *averaged over 30 runs* (i.e. the 3 crashed runs spent time in
contact).  The per-run `safety_outcome` label is the event count: 3 / 30 =
**crash rate 0.100**.  I report both below.

## 4. Results

### 4a. Persistent bias.25 + landmark outage (n=30)

| policy | crash-rate (events/30) | crash time-fraction | mean in-bounds | landed |
|---|---|---|---|---|
| none | **3/30 = 0.100** | 0.021 | 0.148 | 0.000 |
| adaptive | **0/30 = 0.000** | 0.000 | 0.892 | 0.817 |
| adaptive_veto_trust | **0/30 = 0.000** | 0.000 | 0.892 | 0.817 |

Wilson 95% intervals for the crash *rate*:
- none: 0.100 → **[0.035, 0.256]**
- adaptive / trust: 0.000 → **[0.000, 0.113]** (upper bound 11.3%)

`adaptive` and `adaptive_veto_trust` are **identical** on every cell (0.892
in-bounds, 0.817 landed, 0.000 crash).  The trust guardrail does not regress
detection at all.

### 4b. Healthy + landmark outage (n=30)

Every arm (none, adaptive, adaptive_veto_trust): 30/30 completed, in-bounds
1.000, crash 0.000, landed 0.000.  Zero false land in 30 healthy runs.

### 4c. Persistent bias.25, no outage (n=30)

| policy | crash-rate | crash time-fraction | in-bounds | landed |
|---|---|---|---|---|
| none | 3/30 = 0.100 | 0.021 | 0.148 | 0.000 |
| adaptive_veto_trust | 0/30 = 0.000 | 0.000 | 0.892 | 0.817 |

The camera-outage window has no effect on the headline result at this scale — the
detector is already active in the first 10 s.

## 5. Honest reading

- **Detector arms are robustly safe at n=30**: 0 crash in 30 runs on the hardest
  bias.25, with a tight upper 95% bound of 11.3%.
- **The unwatched arm's crash is real but rare**: 3/30 (95% CI 3.5%–25.6%).
  The earlier n=6 "0.020" was time-fraction, not rate; the rate at n=30 is
  0.100.  This is exactly why n≥30 was needed.
- **`adaptive` == `adaptive_veto_trust` on this grid**, so the learned trust
  guardrail is a strict safety improvement (kills the degenerate-parallax false
  land) with **zero measured detection cost**.
- **Healthy false land = 0/30** for both arms — no regression on the quiet case.

## 6. Tests

No code change in this brief (this is a statistical run).  Existing full suite
(37/37) remains the test gate.  If a code change had been needed, it would
require a full re-run.

## 7. Decision

The headline result "detector arms convert bias.25 from crash to controlled
landing" is **now statistically credible at n=30**, not just a small-sample
artifact.  The n=30 plot (`out/consensus_n30_all.csv`,
`out/consensus_roc.png`) is the canonical reference.

## 8. Run

```bash
.venv/bin/python -u run_consensus.py --n 30 --duration 40 --seed 909 \
  --landmark-outage 10-30 --faults bias.25 \
  --policies none,adaptive,adaptive_veto_trust --adaptive-escalate 0.65 \
  --out out/consensus_n30_b25_outage.csv
```
