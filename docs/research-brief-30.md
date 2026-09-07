# Research Brief #30 — Thermal-Aware Mission Budget

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Next increment after the part-level thermal
model (brief-28) and data pipeline (brief-29).

---

## 1. Objective
Energy was already a hard constraint in `PredictiveRePlanner` (`feasible`).
But a route can be *energy-feasible* and still thermally infeasible: a hot
ambient + long cruise means the edge/NPU or battery crosses its temperature
limit before the aircraft lands.  This brief makes the re-planner check **both**
budgets, so it refuses a route that would cook a part even if the battery would
survive it.

## 2. What was built

### `PredictiveRePlanner` (replan.py)
- `thermal_aware: bool` + `thermal_ambient_c` constructor flags (off by
  default, so existing callers are unchanged).
- `ReplanResult` gains `thermal_feasible`, `thermal_max_c`,
  `thermal_worst_node`, `thermal_margin_c`.
- `plan()` now runs `_thermal_feasibility(route, energy_req)`:
  - builds a **fresh** `PartThermalModel` (so repeated plans never carry heat
    between calls),
  - simulates the route's cruise time at `hover_power_w` / 0.3 compute,
  - if any node reaches its limit, adds `thermal_infeasible` and sets
    `feasible=False`.
- `feasible` is now `energy_feasible && thermal_feasible`.

### `MissionReplanBridge` / `SimConfig`
- `BridgeConfig.thermal_aware` / `thermal_ambient_c`; passed to the planner.
- `SimConfig.guardian_replan_thermal_aware` /
  `guardian_replan_thermal_ambient_c` wire it through from a real mission.
- Bridge events record `thermal_feasible` + `thermal_max_c`.

## 3. Demo (`out/guardian/thermal_budget.csv`)

Same short route, two ambients:

| case | feasible | worst node | max temp | margin |
|---|---|---|---|---|
| cool 25 °C | **yes** | cpu_npu | 25.4 °C | +29.6 °C |
| hot 90 °C | **no** | cpu_npu | 90.4 °C | −35.4 °C |

The re-planner now refuses the same route under desert-hot ambient because the
edge/NPU would exceed its limit before landing — even though the battery energy
fraction was fine.

## 4. Tests (`TestThermalBudget`, 2 tests)
- hot ambient rejects the route (`thermal_infeasible`, `feasible=False`).
- cool ambient accepts it (`thermal_feasible=True`, worst node=cpu_npu).
- Guardian suite: **32/32**; full suite below.

## 5. Honest limits
- The thermal feasibility model uses a **declared** `PartThermalModel` (no real
  board thermal constants) and assumes steady cruise at hover power — it does
  not model throttle transient peaks, prop wash, or wind cooling.
- The model starts at ambient; a route that starts hot would already exceed
  limits (correct conservative behaviour, but it does not include a cooling
  recovery).
- Thermal margin uses the node's `max_temp_c` (declared); the 0.85 `warn_frac`
  is not yet used in the planner.

## 6. Next autonomous tasks
1. **Real log reader** (PX4/ROS → `RiskTelemetryDataset`) so the prior and
   thermal constants are calibrated on real flight history.
2. **Perception path (SNN/NeuViT)**: sensed obstacles → `guardian_obstacles`.
3. **Mission re-plan with dynamic thermal state** (feed current thermal node
   temps into the planner instead of restarting at ambient).

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestThermalBudget -v
```
