# Research Brief #25 — Guardian Oracle Routed Into the Real Mission Controller

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  **Program priority #1** (per master-prompt
prioritisation: port the predictive re-planner out of the behaviour lab and into
`simulator.py`).

---

## 1. Objective
Before this brief the `PredictiveRePlanner` only lived in `run_guardian.py`, a
point-mass *behaviour twin*.  It never actually steered a real mission.  Goal:
let the oracle rewrite the **remaining waypoints** of the live
`WaypointMission` inside the full flight stack (dynamics + sensors + EKF +
cascaded controller), only when it is strictly safer.

## 2. What was built

### `airlab.guardian.sim_bridge.MissionReplanBridge`
A *reactive route editor* wrapped around a `WaypointMission`:

- Every `replan_period_s` it runs `PredictiveRePlanner.plan(...)` against the
  current position + remaining waypoints, obstacles, jamming centres, and
  battery fraction.
- It only swaps the route when **all** gates pass:
  - `feasible` (energy envelope),
  - `risk_reduction >= min_risk_reduction`,
  - `min_clearance_m >= min_clearance_m`,
  - `extra_distance_frac <= max_extra_distance_frac`.
- It never overrides the safety FSM, never flies in `HOLD`/`LAND`, and never
  changes the first point (it swaps the *future* polyline only), so it cannot
  command the aircraft backwards.

Gates are configurable; the detour cap is `0.50` (only mission-changing detours
rejected) because a corridor that cuts predicted risk by >0.5 can justify a
large lateral swing.

### `src/airlab/simulator.py` integration
`SimConfig` gains:
- `guardian_replan: bool`
- `guardian_obstacles: [(pos_ned, vel_ned, radius), ...]`
- `guardian_jamming_centers: [pos_ned, ...]`
- `guardian_replan_period_s`, `guardian_min_risk_reduction`,
  `guardian_min_clearance_m`, `guardian_max_extra_frac`,
  `guardian_replan_kwargs`.

When enabled, `Simulator` constructs one bridge and calls it in the main loop
*before* `mission.desired(...)`, gated on `dec.mode in (CRUISE, HOLD)`.

`WaypointMission` gained `remaining_ned()` and `set_route_ned(...)` so the
controller can be retargeted without reconstructing the mission and without
losing its safe-state semantics.

## 3. Demo result (`out/guardian/sim_bridge.csv`)

Straight mission `(0,0,2) → (12,0,2) → (24,0,2)` with a 1.5 m-radius obstacle at
`(12,0,2)` dead on the path.  The oracle swaps the route mid-flight:

```
applied=1
risc 0.4591 -> 0.0012   (reduction 0.4579)
clearance=5.16m
extra distance=20.2%
n_land=0
modes=['evaluated', 'applied']
```

The aircraft no longer flies through the obstacle; it takes an efficient
~20 % longer corridor and still completes without landing for safety.

## 4. Tests
`TestSimBridge` (4 tests):
- applies a safe detour (risk reduced > 0.05, clearance ≥ 2 m, mission no longer
  straight).
- refuses on low battery (`rejected_energy`).
- refuses when there is no gain (`rejected_low_gain`).
- full `Simulator` smoke test with `guardian_replan=True`: bridge runs, mission
  detoured, `0` LAND entries.
- Guardian suite: **20/20**; full suite run in §5.

## 5. Honest limits
- Obstacles are **supplied by the scenario**, not yet detected by a real
  perception path (the perception/SNN item is still on the backlog).
- The obstacle model is a point + radius in the risk field, not a detailed 3-D
  mesh.
- The bridge only replans while the safety FSM is in `CRUISE`/`HOLD`; it does
  not yet feed *its* detour into the energy/thermal budget of the low-watt path.
- `set_route_ned` re-bases the remaining mission; repeated replans at the same
  location are prevented by `replan_period_s`.

## 6. Next autonomous tasks (now that the oracle is in the real loop)
1. **Feed the health engine real simulator telemetry** (priority #2) so
   `SubsystemHealth` scores real fused state, not synthetic numpy features.
2. **Terminal/thermal model** on the low-watt path.
3. **Learned risk prior** from recorded near-miss/jam/obstacle telemetry
   (upgrade analytic surrogate → learned oracle, C→B).
4. **Perception** (obstacle detection → `guardian_obstacles`) so the bridge uses
   sensed threats, not only scripted ones.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestSimBridge -v
```
