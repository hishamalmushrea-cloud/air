# Research Brief #28 — Part-Level Low-Watt Thermal Model

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  **Program priority #4** (per master-prompt
prioritisation).

---

## 1. Objective
The thermal signal in `TelemetryHealthBridge` (brief-26) was a *single lumped
mass*.  The master prompt's thermal requirement is part-level: an operator needs
to know **which** component (edge/NPU, ESC, motor, battery) is heading toward
its limit, so an overheat can be predicted **before** it is exceeded.  This
brief replaces the lumped model with a transparent thermal network.

## 2. What was built

### `airlab.guardian.thermal`
- `ThermalNode(name, capacity_jpk, conductance_wpk, max_temp_c)` — one part.
  Exposes `temp_c`, `margin_c`, `ok`.
- `PartThermalModel` — a tiny thermal network:
  ```
  nodes: battery, motor, esc, cpu_npu (heat generators)
         + frame (passive heat sink)
  node  C*dT/dt = P_in - k_amb*(T-T_amb) - k_frame*(T-T_frame)
  frame C*dT/dt = sum(k_frame*(T_i-T_frame)) - k_amb_frame*(T_frame-T_amb)
  ```
- Heat inputs derived from real flight telemetry:
  - `motor`: 12% of propulsive power (copper/iron loss)
  - `esc`: 5% of propulsive power (switching loss)
  - `battery`: 6% of total electrical power (internal-resistance heat)
  - `cpu_npu`: idle 2 W + `compute_frac * 8 W` (the actual edge/NPU load)
- Exposes per-node `temperatures()`, `margins()`, `worst_node()`, `status()`,
  `summary()`.

### Integration
`TelemetryHealthBridge` now owns a `PartThermalModel` and feeds the **worst
node temperature** into `SubsystemHealth`'s thermal subsystem.  It also records
per-node temps and `compute_frac` in its history.  `SimConfig.compute_frac`
drives the edge load (0.30 baseline, 1.0 full inference).

## 3. Demo (`out/guardian/thermal.csv`)

120 s flight, two conditions:

| condition | cpu_npu | esc | motor | battery | worst | status |
|---|---|---|---|---|---|---|
| baseline (25 °C, 0.30 load) | 28.6 | 28.1 | 31.1 | 25.9 | motor | all ok |
| full load (45 °C, 1.0 load) | 52.9 | 48.2 | 51.2 | 45.9 | **cpu_npu** | **cpu warn + battery critical** |

The model answers the part-level question: at cool/baseline the *motor* is the
hottest but fine; at hot + full edge inference the **cpu_npu** approaches its
55 °C derating limit (warn) and the **battery** crosses its 45 °C limit
(critical).  This is exactly the "predict heat before it exceeds budget"
capability.

## 4. Tests(`TestThermal`, 3 tests)
- high compute load heats `cpu_npu` more than baseline; exposes a specific hot
  node.
- per-node margins are bounded and node-specific limits are respected
  (battery max < motor max).
- `TelemetryHealthBridge` uses a `PartThermalModel` with cpu_npu/esc/motor/
  battery nodes.
- Guardian suite: **28/28**; full suite below.

## 5. Honest limits (scientific honesty)
- All thermal constants are **modelled/estimated**, not measured on a real
  board.  The network is the right *shape*, but numbers should be tuned to a
  specific CPU/NPU + ESC + motor + battery after thermal test data.
- No AC/radiator model (no prop wash, no heatsink fins yet) — it is a first-
  order conductance model.
- Heat is computed from `PowerModel` power (the controlled thrust) plus a
  `compute_frac`; a real aircraft would feed ESC/motor current sensors and a
  board temperature sensor.
- Status thresholds are declared (`warn_frac=0.85`); an operator can retune
  them.

## 6. Next autonomous tasks (priority #5)
1. **Data pipeline** (edge → storage → analytics) that writes real simulator
   telemetry to a dataset, so the learned risk prior and the thermal constants
   can be calibrated from *recorded* data, not modelled guesses.
2. **Thermal-aware mission** (throttle/compute budget) so the oracle can keep
   the platform inside both energy and thermal budgets.
3. **Perception path** (SNN/NeuViT) so obstacles are sensed, not scripted.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestThermal -v
```
