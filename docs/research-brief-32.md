# Research Brief #32 — Low-Watt Edge Perception Path (Spiking / NeuViT-style)

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Program priority #7.

---

## 1. Objective
Up to now the mission bridge only ever received *scripted* obstacles
(`guardian_obstacles`).  This brief introduces the first **sensing path**: a
transparent, low-watt edge front-end that converts a sparse depth/range stream
into obstacle detections and feeds them to the guardian oracle, so the route
planner can avoid what the aircraft **actually sees** rather than what a
scenario script told it.

## 2. What was built

### `airlab.guardian.perception`
- `PerceptionConfig` — declared edge/NPU constants: `edge_topps=0.35`,
  `edge_power_w=8.0`, `neuromorphic_gops_per_w=847.0`, range gate, cluster
  separation, frame rate, sensor range, FOV.
- `EdgePerception` — detected `Obstacle`s + spike/cluster counts + energy + the
  `intelligence_per_watt` (frames-per-second per watt) and declared
  `gops_per_w`.
- `SpikeVision` — a **transparent spiking-style front-end** (numpy-only):
  - threshold the range-return points (spike gate),
  - one-shot binary spike (no continuous firing),
  - **edge-side clustering** with a `min_points`-per-cluster gate,
  - frame-to-frame cluster tracking → velocity,
  - returns `Obstacle(pos, vel, radius)`.
- `PerceptionToGuardian` — thin wrapper that converts a depth/range stream into
  an obstacle list for `MissionReplanBridge`.

### Integration
The `MissionReplanBridge` already accepts any `Obstacle` objects, so the
perception output drops straight into the oracle — no bridge change needed.

## 3. Honest classification
This is **not** a claim of running on a real NeuEdge/NeuViT chip:
- The spiking front-end is **simulated** (a transparent surrogate).
- `gops_per_w = 847` is a **declared research figure**, not a measurement.
- `edge_power_w = 8.0` is **Estimated**; real figures need a board power probe.
- Label: **D / Simulated / Estimated** (master prompt §4/§27).

## 4. Demo (`out/guardian/perception.csv`)
A depth-return cloud with one dense object near the path and one sparse group
far away:

```
detected=1 clusters  spikes=5
power=8.00 W  intelligence_per_watt=1.25 Hz/W  gops/W=847
bridge applied=True   risk_reduction=0.637   clearance=2.11 m
```

The sparse group is correctly rejected (too few points), the dense object is
detected, and the oracle changes the route to avoid it with a 2.11 m clearance
and a large risk reduction — all from a *sensed* obstacle, not a scripted one.

## 5. Tests (`TestPerception`, 4 tests)
- detects a dense cluster and rejects sparse points.
- returns no obstacle on empty input.
- reports non-zero energy + declared `gops_per_w`.
- feeds sensed obstacles into a `MissionReplanBridge` → feasible, risk
  reduction > 0, applied.
- Guardian suite: **38/38**; full suite below.

## 6. Honest limits
- No real depth/LiDAR/NeuViT hardware; the sensor cloud is synthetic.
- Gaussian/nearest-centroid clustering is a surrogate — a real SNN/NeuViT
  detector would run on spiking patterns on an actual edge/NPU.
- The energy/intelligence-per-watt numbers are declared estimates; they should
  be replaced by board power measurements.
- The detector uses range points only (no RGB/event camera, no learned
  classifier yet).

## 7. Next autonomous tasks
1. **Dynamic thermal state in re-planning** (feed live node temps into the
   planner instead of restarting at ambient).
2. **Event-camera / RGB perception** as a second sensing path (closer to
   NeuViT), integrated with the same occupancy map.
3. **Onboard-vs-Ground split** (master prompt §18): decide which perception
   runs on the 0.35 TOPS edge and which is offloaded.

## 8. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestPerception -v
```
