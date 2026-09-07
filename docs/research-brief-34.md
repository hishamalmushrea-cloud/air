# Research Brief #34 — Event-Camera / RGB Motion Perception Path

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Program priority #9.

---

## 1. Objective
Depth/range perception (brief-32) is a valid first sensing path, but it cannot
see passively in all scenes and it costs ~8 W on the declared edge budget.  A
neuromorphic event camera is a much lower-power second path (few W) and, in
real systems, can densely encode scene motion with sparse asynchronous events.
This brief adds that **second sensing path** so the guardian has independent
sense evidence, and a small cross-sensor fusion that is conservative for the
defensive/safety role.

## 2. What was built

### `airlab.guardian.perception.EventVision`
- `EventConfig` — declared event/RGB front-end constants: `event_power_w=2.5`,
  `rgb_power_w=1.5`, accumulator window, grid rate, occupancy threshold, cluster
  gate, sensor range, max clusters.
- `EventVision` — transparent event-camera / RGB-motion surrogate:
  1. consume sparse `(dt, x_norm, y_norm, polarity)` events,
  2. accumulate into a low-res **motion-occupancy histogram** (leaky, decays
     0.98/frame),
  3. threshold by event density,
  4. cluster active cells,
  5. project to body-forward NED and track velocity frame-to-frame,
  6. return the identical `Obstacle` list for `MissionReplanBridge`.

### `MultiSensorGuardian`
A conservative cross-sensor fusion for the defensive/safety role:
- keep depth detections when the depth path reports ≥ 2 clusters (dense range
  cloud is a strong signal),
- otherwise preserve a confirmed event-camera detection,
- single weak sources are not allowed to create a false detour alone.

## 3. Honest classification
Same rule as brief-32: D / Simulated / Estimated.  The event path is a
transparent surrogate for a real neuromorphic event camera; power figures are
declared estimates and `gops_per_w=847` is a declared research figure, not a
measurement.

## 4. Demo (`out/guardian/event_perception.csv`)
```
event detected=1 clusters  spikes=160
event power=4.00W   (depth was 8.00W)   gops/W=847
fusion kept=1 objects (depth + event consensus)
```

The event path detects the motion-occupancy object at **half the declared power
of the depth path**, and the fusion selects a single confirmed obstacle.

## 5. Tests (TestPerception now 7)
- event occupancy detects an object; event power < depth power.
- depth + event fusion keeps the confirmed obstacle.
- no-events → empty.
- plus the original range/depth tests.
- Guardian suite: **43/43**; full suite below.

## 6. Honest limits
- The event stream is synthetic; no real event camera (DAVIS/Prophesee) or RGB
  sensor is wired in.
- Occupancy clustering is a simple histogram surrogate — a NeuViT-style spiking
  classifier would operate on true event patterns.
- Power is declared/estimated; real camera + NPU power needs a measurement.
- Fusion is deliberately biased toward depth when it sees multiple objects
  (conservative), so an event-only object is only accepted when the depth path
  is quiet.

## 7. Next autonomous tasks
1. **Onboard-vs-ground compute split** (master prompt §18): decide which
   perception runs on the 0.35 TOPS edge and which is offloaded.
2. **Thermal-aware trajectory**: modulate `compute_frac`/cruise power in the
   plan so a hot route can cool (not only reject).
3. **Real event/RGB integration** when a sensor + driver becomes available.

## 8. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestPerception -v
```
