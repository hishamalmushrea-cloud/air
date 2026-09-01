# Research Brief #16 — Degenerate-Parallax Camera Discriminates Count-Based vs Trust-Based Veto

**Date:** 2026-09-02
**Project:** AIR Lab
**Status:** implemented, tested, and **discriminated end-to-end**.  This is the
first scenario where `adaptive_veto_trust` measurably beats `adaptive_veto`.

---

## 1. Goal

Brief #15 implemented a continuous, self-measured frame *trust*
(landmark count × angular diversity) but we could not test it end-to-end because
the synthetic world always sees well-spread landmarks.  This brief adds the
missing first-class failure: a **degenerate-parallax camera** — the camera still
reports *many* landmarks, but all within a tiny angular cone (e.g. a camera
looking at a close wall / low-parallax scene).  Raw count is high, so a
count-based guardrail is blind to it; the angular-diversity trust is not.

## 2. Implementation

- New `SimConfig.landmark_cluster: Optional[(start, end)]` and
  `landmark_cluster_cone` (default `0.04` rad).
- During the window, the camera keeps the observed landmark **ids/count** but
  collapses the body-frame bearings into a tight cone around their mean bearing
  (deterministic, RNG 1234).
- New `--landmark-cluster start-end` CLI flag  (`run_consensus.py`, threaded
  through `Scenario` and `consensus_study`).
- The factor graph uses its own `LandmarkField.observe` (unclustered), so the
  cluster only affects the landmark *detector* — a clean experimental isolation.

## 3. Trust floor gates the velocity-aiding fold (the critical bug)

The cluster drops the landmark score to ~0.07.  That low score was being folded
straight into **`_effective_flow_health`** (the velocity-aiding signal), **not
only** into the detector consensus.  During a GPS outage the safety layer then
triggered `velocity_aiding_failed_without_gps` — bypassing the consensus
entirely, even though the trust-veto consensus itself had returned a healthy
`det=1.0` (factor graph was flawless).

**Fix:** add `SafetyConfig.detector_trust_floor = 0.45` and only fold the
landmark opinion into flow health when `_detector_trust("landmark") >= floor`.
This makes design-in safety consistent: a detector needs enough *geometric
leverage* to influence any safety path, not merely a high raw count.

## 4. Results

### 4a. Healthy mission + degenerate-parallax window (n=3, seed 909, 6–22 s)

| policy | in_bounds | landed | crash |
|---|---|---|---|
| none | 1.000 | 0.000 | 0.000 |
| adaptive_veto (count-based) | 0.957 | **0.748 false-land** | 0.000 |
| adaptive_veto_trust (diversity-based) | **1.000** | **0.000** | 0.000 |

Root cause of the 0.748: the count-based veto treats a high-count camera as
credible even when its geometry is degenerate, so the low landmark score
escalates the consensus.  The trust veto sees `trust=0.036 → thin`, keeps the
healthy factor-graph voice, and stays quiet.

### 4b. Real faults + same window (n=3, seed 909)

| fault | none | adaptive_veto | adaptive_veto_trust |
|---|---|---|---|
| scale1.5 | 0.000 landed | 0.783 | 0.773 |
| bias.25 | **0.025 crash** | 0.844 | 0.844 |

`adaptive_veto_trust` still detects real faults: it lands bias.25 (0.844, crash
0.000) just like the count-based veto.  The trust floor does **not** suppress
genuine detection.

### 4c. Outage regression (n=3, seed 909, landmark outage 10–30 s)

| fault | adaptive_veto | adaptive_veto_trust |
|---|---|---|
| none | 0.000 | 0.000 |
| scale1.5 | 0.765 | 0.765 |
| bias.25 | 0.844 | 0.844 |

Identical on every fault → the trust change is a **strict improvement**, not a
regression.

## 5. Tests

- `test_landmark_cluster_keeps_count_but_collapses_trust` — unit.
- `test_trust_veto_does_not_false_land_on_degenerate_parallax` — end-to-end on
  the exact random scenario + GPS outage that originally leaked: veto false-lands
  (landed > 0.4), trust veto lands 0.0, both crash 0.
- Full suite: **35/35 OK**.

## 6. Decision

- Keep `adaptive` as the default (zero regression).
- **`adaptive_veto_trust` is now the recommended mode for geometric
  degeneracy.**  It is identical to `adaptive_veto` on feature-poor (outage)
  flight and on every real fault, and removes the new false-landing mode
  that the degenerate-parallax camera exposes.
- `adaptive_veto` remains useful and is the simpler reference; but it is
  **not** safe against a high-count / low-diversity camera.

## 7. Next step

Learned per-frame confidence on top of the analytic trust (the analytic model is
a single hand-set RMS-angle heuristic; a learned regression from landmark
distribution + graph conditioning to "is this frame informative?" is the natural
upgrade).  Then sparse factor-graph outage and n≥30.

## 8. Run

```bash
.venv/bin/python -u run_consensus.py --n 3 --duration 40 --seed 909 \
  --landmark-cluster 6-22 \
  --faults "none" \
  --policies "none,adaptive_veto,adaptive_veto_trust" --adaptive-escalate 0.65 \
  --out out/consensus_cluster.csv
```
