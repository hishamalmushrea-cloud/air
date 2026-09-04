# Research Brief #23 — Oracle Risk World Model + Predictive Re-Planning

**Date:** 2026-09-04
**Project:** AIR Lab
**Status:** implemented, tested.  This is the best-next-experiment from brief #22:
the guardian now predicts and avoids the **whole threat corridor** over the
remaining mission, instead of only dodging the next step.

---

## 1. Goal

Brief #22's `EvasionPlanner` reacts to a *current* obstacle.  That handles the
last second but not the corridor: by the time a collision risk is severe the
aircraft may already be committed.  This brief adds a **risk world model** over
the remaining mission and a **predictive re-planner** that reorders/offsets the
remaining waypoints before the threat is reached.

## 2. What was built

### `risk.py` — RiskWorldModel
- Coarse 3-D risk field over the remaining mission area.
- Sources:
  - **obstacle projections** (static + dynamic, projected over a 3 s horizon)
    as bounded Gaussians,
  - **jamming / spoofing corridors** (areas where GPS/RF degradation was
    observed) as wider Gaussians,
  - **energy floor** enforced by the replanner (not the field).
- Fully **transparent / inspectable**: every contribution is a documented
  analytic field.  We call it an *oracle* only in the sense that it is the
  planner's internal world model over the remaining mission — it is **not** a
  black-box or superhuman predictor.

### `replan.py` — PredictiveRePlanner
- **Beam search** (K=6) over bounded lateral (±8 m) and vertical (±2 m) offsets
  of each remaining waypoint.
- Each candidate polyline is scored by `mean_risk + length_penalty`.
- Returns risk reduction, extra distance, required energy fraction, clearance,
  and feasibility against the platform's battery/reserve.
- Uses the `NexusSpec` energy envelope (112 W hover, 71 Wh, 15% reserve).

## 3. Demonstrated result (`run_guardian.py`, oracle_replan scenario)

The mission ends with a 2-leg route straight through a static obstacle
(radius 1.5 m) and a jam centre.  The re-planner offsets the remaining
waypoints:

| metric | value |
|---|---|
| baseline mean risk / m | 0.525 |
| re-planned mean risk / m | **0.180** |
| risk reduction | **0.345 (66%)** |
| min clearance from obstacle | **3.50 m** |
| extra distance fraction | 5.7 % |
| required energy fraction | 0.155 |
| feasible | yes |

So the aircraft takes a modest detour (5.7 % longer) and cuts predicted risk by
two-thirds while clearing the obstacle by 3.5 m instead of flying through it.

## 4. Energy / compute (design-in)
- Beam search cost scales with `waypoints × beam × offsets`.  For a short
  remaining mission it is a few ms in numpy; the cost estimate in `NexusSpec`
  (0.35 TOPS class, spiking-friendly target) and NeuEdge (~847 GOp/s/W) would
  put it inside a 10 W edge envelope.  These are research targets, not bench
  measurements.
- The re-planner itself is small and reproducible; no external model or weights.

## 5. Honest limits
- The risk field is **analytic**, not learned from real data yet.  It is a
  surrogate that understands "obstacle near path" and "jammed area near path"
  but has no learned risk prior from field telemetry.
- The beam search is **coarse** (bounded offsets).  It does not search arbitrary
  routes, so it can locally fail to find a full detour around a complex corridor.
- We label the "oracle" as **C (potential)** in the A/B/C/D quadrant scheme: the
  deterministic field is implemented (A), but a genuinely *learned* risk prior
  over the mission (from past flights / shared fleet telemetry) is C.

## 6. Tests
- `TestRiskAndReplan`:
  - risk field penalises an obstacle and replanning increases clearance,
  - replanning respects energy feasibility (tiny battery → infeasible),
  - `test_replan_clears_threat_corridor_with_small_extra_distance` verifies
    `risk_reduction > 0.2`, `clearance ≥ 2.0 m`, feasible, and
    `extra_distance_frac < 0.2`.
- Guardian suite: **13/13**.  Full suite: **52/52** (see run).

## 7. Decision / next
- Keep the deterministic RiskWorldModel as the default world model for the
  guardian (transparent, zero external deps).
- **Next**: learn a risk prior from flight telemetry (past obstacle near-misses,
  jam history) to replace the hand-set Gaussian sigma/amp — this upgrades the
  model from analytic surrogate to learned oracle (quadrant C → B).
- **Then**: bring the re-planned route into the real mission controller in
  `simulator.py` (currently a behaviour-lab demonstration only).

## 8. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
```
