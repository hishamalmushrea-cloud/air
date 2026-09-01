# Research Brief #17 — Learned Per-Frame Trust (replaces hand-set count/3 and rms/1.2)

**Date:** 2026-09-02
**Project:** AIR Lab
**Status:** implemented, tested.  The landmark frame-trust is now *learned from
the run's own healthy startup* instead of hand-set constants.  It keeps the
degenerate-parallax advantage from brief #16 and adds adaptivity to whatever the
camera actually sees.

---

## 1. Goal

Brief #15/#16 used a hand-set analytic trust: `count/3 × rms/1.2`.  Those
constants assume a fixed "3 landmarks is enough" and "~69 deg is informative".
A real camera may see 6 or 12 landmarks, and the informative angular spread may
be scene-dependent.  This brief **learns** the per-frame reference distribution
on a clean startup, then scores every later frame against it.

## 2. Model (`src/airlab/trust.py`)

`FrameTrustLearner`:
- During the first `trust_calibrate_s = 6.0` s (default), when the camera is
  healthy and outside any outage/cluster window, it collects the RMS pairwise
  angular separation and the observed landmark count.
- Calibrates on the 25th (low) and 90th (high) percentiles:
  - `ref` = low percentile,
  - `band` = (high − low)/2,
  - `offset` chosen so a frame at `ref` maps to trust ≈ 0.75 and a frame at the
    high percentile to ≈ 0.95,
  - `count_ref` = median observed count (replaces the hand-set `/3`).
- Every later frame: `trust = sigmoid((rms − ref)/band + offset) × clip(count/count_ref)`.

Properties:
- **Self-calibrated, transparent, numpy-only** — no external learner, no black
  box; all parameters are inspectable.
- **Never uses the detector's own verdict** — it measures *could this frame
  inform a geometric check?*, so a faulty detector cannot down-weight itself.
- **Structural continuity:** factor-graph trust stays analytic because its
  reference value is 1.0 by construction (the graph either has its minimum
  factors or it does not), and learning from startup samples (mostly `base=0`
  while the graph warms up) would weaken the sparse-graph penalty.

## 3. Calibration observed

On the exact degenerate-parallax scenario (seed 909, scen1):

```
landmark: ref=1.646 rad, band=0.152 rad, n=8, count_ref=6.00
```

The learner adapts to a camera that sees ~6 landmarks and ~94° spread, rather
than assuming 3 landmarks / 69°.

## 4. Results (unchanged from brief #16 — the learned model keeps the guardrail)

### 4a. Healthy + degenerate-parallax (n=3, seed 909, 6–22 s)

| policy | in_bounds | landed | crash |
|---|---|---|---|
| adaptive_veto (count) | 0.957 | **0.748 false-land** | 0.000 |
| adaptive_veto_trust (learned) | **1.000** | **0.000** | 0.000 |

### 4b. Real faults + same window (n=3)

| fault | adaptive_veto | adaptive_veto_trust |
|---|---|---|
| scale1.5 | 0.783 | 0.773 |
| bias.25 | 0.844, crash 0 | **0.844, crash 0** |

### 4c. Outage regression (n=3, 10–30 s)

`adaptive_veto_trust == adaptive_veto` on every fault (none 0.000, scale1.5
0.765, bias.25 0.844), and both clean on healthy.

## 5. Tests

- `test_frame_trust_learner_calibrates_from_startup_and_detects_cluster`:
  feeds a healthy ~1.0 rad / 5-landmark startup, then verifies the model scores
  a healthy frame > 0.7, a degenerate high-count frame < 0.35, and `count_ref ≈
  5` (learned, not `/3`).
- `test_trust_veto_does_not_false_land_on_degenerate_parallax` (existing e2e):
  still passes — the learned model, not the analytic fallback, keeps the healthy
  mission clean.
- Full suite: **36/36 OK**.

## 6. Decision

- Default poly remains `adaptive` (zero regression).
- `adaptive_veto_trust` is now **learned, not analytic**.  Still recommended for
  feature-degenerate / low-parallax flight, and it adapts its reference to the
  camera rather than to two hard-coded constants.
- The learner is a conservative upgrade: it is derived from the run's own
  healthy startup, so it cannot be gamed by the fault (which starts after 6 s in
  every evaluated scenario).

## 7. Next step

- **Sparse factor-graph outage** experiment (a first-class under-determined
  graph), now that landmark trust is solid.
- **n≥30** statistical grid for the headline crash numbers (the unwatched bias.25
  crash 0.025 at n=3 has a wide CI; detector arms are all zero-crash).

## 8. Run

```bash
.venv/bin/python -u run_consensus.py --n 3 --duration 40 --seed 909 \
  --landmark-cluster 6-22 \
  --faults "none" \
  --policies "none,adaptive_veto,adaptive_veto_trust" --adaptive-escalate 0.65 \
  --out out/consensus_cluster_learned.csv
```
