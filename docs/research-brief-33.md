# Research Brief #33 — Dynamic Thermal State in Re-Planning

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Priority #8.

---

## 1. Objective
The thermal-aware budget (brief-30) started the `PartThermalModel` **from
ambient** on every plan.  It therefore ignored what the aircraft was **already
feeling**: a long, hot prior mission can leave the edge/NPU or battery hot
before the next replan.  This brief feeds the planner the **live** node
temperatures so it predicts from where the aircraft is, not from its default.

## 2. What was built

### `PartThermalModel(initial_temps=...)`
The thermal network now accepts a dict of starting node temps:
```
cpu_npu / esc / motor / battery / frame
```
so a planner can seed it with the current fused thermal state.

### `PredictiveRePlanner(thermal_initial_temps=...)`
- new constructor flag; passed to the fresh `PartThermalModel` each call.
- **Fixed a latent bug**: `_thermal_feasibility()` previously compared the max
  temp to the **cpu limit**; it now compares it to the **worst node's own
  limit** (a battery has 45 °C, an edge 55 °C, motor 90 °C — very different).

### Bridge / Simulator
- `BridgeConfig.thermal_initial_temps` is carried to the planner.
- `SimConfig.guardian_replan_use_live_thermal` (off by default): when on, the
  Simulator pulls the health bridge's live node temps into the bridge config
  before each replan.

## 3. Demo (`out/guardian/dynamic_thermal.csv`)
Same short route, three initial states:

| case | feasible | worst node | max temp | margin |
|---|---|---|---|---|
| cold start | yes | cpu_npu | 25.4 °C | — |
| live hot edge (52→49.1 °C) | yes | cpu_npu | 49.1 °C | +5.9 °C |
| live hot battery (46 °C) | **no** | battery | 45.9 °C | −0.9 °C |

Physics is honest: a hot edge **cools** back under normal power and stays
feasible; a **battery already over its limit stays over** and rejects the
route.

## 4. Tests (`TestDynamicThermal`, 2 tests)
- live battery-over-limit rejects a route that a cool start accepts.
- bridge carries configured live initial temps into the planner.
- plus fixed `TestThermalBudget` still passes.
- Guardian suite: **40/40**; full suite below.

## 5. Honest limits
- The live state is injected as a dict; the physical source is the health
  bridge's `PartThermalModel` (itself modelled, not measured).
- The model still assumes nominal cruise power and no transient/wind cooling.
- An over-limit edge can cool over a short route (honest, physically correct);
  the safety layer must still act on the *current* breach, which is why the
  thermal model is advisory for planning while `SubsystemHealth` remains the
  abort authority.

## 6. Next autonomous tasks
1. **Event-camera / RGB perception** as a second sensing path (closer to
   NeuViT) integrated with the same occupancy map.
2. **Onboard-vs-ground split** (master prompt §18): decide which perception
   runs on the 0.35 TOPS edge and which is offloaded.
3. **Thermal-aware trajectory already**: optionally modulate `compute_frac` or
   cruise power in the plan so a hot route can cool (not just reject).

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestDynamicThermal -v
```
