# Research Brief #36 — Thermal-Aware Trajectory (Cool, Not Just Reject)

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Program priority #11.

---

## 1. Objective
Priority #8/#9 showed that a hot node can make a route thermally infeasible and
the guard rejects it.  That is only half of the answer: many hot routes can be
made feasible by *modulating the execution profile* so the stack runs cooler —
lower compute load on the edge/NPU and/or a lighter throttle/power profile
(which also produces less ESC/motor heat, over a longer exposure).  A
thermally-aware trajectory should choose that mitigation when it exists and
only reject when no in-range profile is safe.

## 2. What was built

### `ReplanResult` new fields
- `thermal_mitigated`, `thermal_compute_frac`, `thermal_power_frac`,
  `thermal_power_w`.

### `PredictiveRePlanner._sim_thermal(route, compute_frac, power_frac)`
A fresh `PartThermalModel` simulation of the route at a given profile:
- `power_frac` scales the propulsive power *and* the airspeed (a lighter
  throttle means a longer exposure — no silent free lunch).
- `compute_frac` scales the edge/NPU heat.
- Feasibility is still per-node against its **own** limit.

### `PredictiveRePlanner._thermal_feasibility`
- Run the **nominal** profile first (`cf=0.30, pf=1.0`).  Feasible → no
  mitigation.
- Otherwise search a small grid
  `pf ∈ {1.0…0.4}`, `cf ∈ {0.30…0.05}` and keep the *least disruptive* profile
  that keeps every node under its own limit.
- If none is feasible, return the honest rejection with the nominal worst node.

## 3. Demo (`out/guardian/thermal_trajectory.csv`)
```
hot_edge_59C  feasible=True  mitigated=True  cf=0.30  pf=0.90  pw=100.8W
              max=54.9C  worst=cpu_npu   (was over/at the 55 C limit at nominal)
hot_edge_68C  feasible=False mitigated=False max=63.1C worst=cpu_npu
              reasons=['thermal_infeasible']
```

A hot edge at **59 °C** is cooled into feasibility by flying at 90 % throttle
(100.8 W vs 112 W).  An **extreme** 68 °C edge has no safe profile in range and
is correctly rejected — mitigation is not a magic trick.

## 4. Tests (`TestDynamicThermal` +2)
- `thermal_mitigation_cools_hot_edge_route`: nominal over-limit hot edge at a
  short route becomes feasible (`thermal_mitigated=True`, reduced power).
- `too_hot_route_still_rejects`: extreme hot edge stays infeasible/rejected.
- Guardian suite: **49/49**; full suite below.

## 5. Honest limits
- Power profile is modelled, not measured.  The `power_frac↔airspeed` coupling
  is a declared assumption.
- The mitigation grid is coarse and greedy (least-disruption heuristic), not a
  continuous optimum.
- It mitigates the *remaining route* only; it does not yet change the
  onboard/ground split or the compute-placement in real time.
- No real thermal sensor or flight profile validation.

## 6. Next autonomous tasks
1. **Dynamic link-aware rebalancing** (drop/queue analytics when the RF link
   degrades; keep the safety loop untouched).
2. **Real GCS integration** (MAVLink telemetry + a supervisor board).
3. **Constraint-aware trajectory** (take the thermal plan as a mission-level
   constraint set, not a result-only check).

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestDynamicThermal -v
```
