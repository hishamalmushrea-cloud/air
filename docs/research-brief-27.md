# Research Brief #27 — Learned Risk Prior from Flight / Jamming Telemetry

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  **Program priority #3** (per master-prompt
prioritisation).  Upgrades the analytic oracle from `RiskWorldModel` fixed
Gaussians to a **data-calibrated risk map** (quadrant C → B — potential →
improved).

---

## 1. Objective
`RiskWorldModel` (brief-23) used hand-set `obstacle_amp`, `obstacle_sigma`,
`jamming_amp`, `jamming_sigma`.  Those are reasonable but *guessed*.  The goal
is to keep the corridor shape transparent while learning the **severity**
(how dangerous is being this close to an obstacle / inside a jamming region)
from labelled telemetry.

## 2. What was built

### `airlab.guardian.risk_prior`
- `RiskSample(dist_m, jam, label)` — one telemetry observation.
- `RiskPriorModel` — a **kernel-regression (Nadaraya-Watson)** prior:
  - prints: `fit(samples)` learns distance/jamming bandwidths via a
    Silverman-style heuristic (no fabricated constants),
  - `predict(dist_m, jam)` returns risk in `[0,1]`,
  - `summary()` for transparency.
- `simulate_telemetry(n)` — a **deterministic, simulated** labelled set (not a
  claim of real flight data): sample positions/jam, label from the same threat
  physics the guardian uses, add measurement noise.  The model is data-agnostic:
  swap in any recorded flight/jam CSV and it fits identically.

### `RiskWorldModel(prior=..., learned_alpha=...)`
When a fitted prior is provided, `build(...)` adds a **learned layer** on top of
the analytic field:
- computes per-cell features: distance to nearest projected obstacle + jamming
  proximity,
- replaces the severity with `learned_alpha * prior.predict(dist, jam)`,
- keeps the corridor *shape* analytic (so the field stays inspectable), while
  the *intensity* becomes measured.

`PredictiveRePlanner(model=RiskWorldModel(prior=...))` then re-plans against the
learned field unchanged — no planner changes needed.

## 3. Demo (`out/guardian/risk_prior.csv`)
```
prior n=1200  bw_dist=0.884  bw_jam=0.080
learned at obstacle=1.000   analytic at obstacle=1.000   side=0.220
replan learned risk 0.835 -> 0.053  (reduction 0.782)
clearance 6.16 m
```

At the obstacle the learned field is as severe as the analytic one, but it
*tapers faster* off-axis (0.22 at 6 m), so the replanner pays less distance to
clear the corridor and still wins a large risk reduction (0.782) with 6.16 m of
clearance.

## 4. Tests (`TestRiskPrior`, 3 tests)
- prior learns near-obstacle > far, and jamming increases risk.
- `RiskWorldModel(prior=...)` produces a higher-risk cell at the obstacle than
  off-axis, and > 0.5 at the obstacle centre.
- `PredictiveRePlanner` with the learned prior reduces risk and keeps clearance
  ≥ 2 m.
- Guardian suite: **25/25**; full suite below.

## 5. Honest limits (scientific honesty)
- **The reference telemetry is simulated**, not recorded from a real aircraft.
  The model is data-agnostic, but the *calibration* quality is only as good as
  the simulated telemetry.  Label flags: **Simulated**.
- Kernel regression is intentionally simple — it is inspectable and cheap, but
  it cannot model non-monotonic couplings or multi-obstacle interactions as well
  as a fitting neural field would.
- The learned prior replaces bump *intensity*, not corridor topology; multi-way
  complex corridors may still need a denser model.
- No real sensor telemetry pipeline exists yet (that is the Data Engineering
  backlog item).

## 6. Next autonomous tasks (priority #4 / #5)
1. **Part-level low-watt thermal model** (CPU/NPU/ESC/motor/battery) so the
   health engine can predict heat before it exceeds a budget.
2. **Data pipeline**: record real simulator telemetry to a dataset so the
   learned prior has reproducible *real* inputs instead of simulated ones.
3. **Perception path** (SNN/NeuViT) so `guardian_obstacles` comes from a sensed
   rather than scripted source.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestRiskPrior -v
```
