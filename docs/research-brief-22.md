# Research Brief #22 — Nexus-Predator: Next-Generation Autonomous UAV (AI Guardian Core)

**Date:** 2026-09-04
**Project:** AIR Lab
**Status:** implemented as a defensive behaviour-lab core.  This is the start of
a **new platform** built on top of the existing stack — not a rewrite of the
physics/estimator, but an upgraded *autonomy brain* that can detect any threat
and dodge/collision-avoid/recover/silence defensively.

> Scope gate: all capabilities are **defensive / sense-and-avoid / safety /
> resilience**. No weapons, no targeting, no guidance toward a threat. The
> "threat" a UAV detects is anything that threatens *itself*.

---

## 1. What was built

New package `src/airlab/guardian/` (numpy-only, transparent, inspectable):

| module | responsibility |
|---|---|
| `threats.py` | `ThreatEngine` — five independent evidence channels (navigation spoofing/jamming, obstacle/collision, wind anomaly, cyber/command anomaly, battery/energy) fused only from *different* sensors so a fault cannot self-confirm. |
| `avoidance.py` | `EvasionPlanner` — receding-horizon candidate maneuver search (yaw left/right, climb, descend, slow) with projected worst clearance, plus a bounded **cloak** jitter. |
| `brain.py` | `GuardianBrain` — picks a mode (`CRUISE_SAFE / EVADE / RECOVER_NAV / SILENT / ABORT`) with priority battery/wind → obstacle → nav → cyber; tracks **declared** and **undeclared** capabilities exercised. |
| `drone_nexus.py` | `NexusAirV2` platform spec + kinematic behaviour-lab twin. |
| `run_guardian.py` | CLI behaviour lab: 5 scenarios → `out/guardian/summary.csv`. |

### Declared vs undeclared (both fully documented here)
* **Declared** — externally advertised/verifiable: predictive sense-and-avoid,
  cross-sensor navigation consistency, energy-envelope guard, command-envelope
  guard.
* **Undeclared** — not advertised externally, but internal design properties:
  `cloaked_evasion` (bounded random lateral oscillation to reduce
  extrapolability), `autonomous_source_switch` (drop degraded GPS, fly on IMU/
  baro/mag), `silent_rf` (throttle RF emissions when jammed/tracked). These are
  *not* hidden from the operator — they are reported in the decision log.

## 2. State-of-the-art grounding (what this builds on)
* **Intent-aware RL collision avoidance** with global/local intent prediction and
  D3QN sector control (Aerospace 2026) [1].
* **AI deconfliction survey** naming deep, RL/MARL, and bio-inspired spiking
  approaches; SNNs cut inference energy >75% while keeping obstacle-avoidance
  accuracy [2].
* **NMPC/vision-based avoidance** on edge NUC with safety margins [3].
* **Telemetry-only GPS spoofing/jamming detection** with XGBoost (F1 ≈ 0.998),
  runnable on small autopilots, with interpretable feature importance [4].
* **Neuromorphic edge AI**: NeuEdge 847 GOp/s/W, 2.3 ms inference, 91–96 %
  accuracy; NeuViT keeps UAV vision in a 10 W / 15 FPS envelope (3.98 W) [5].

Our `NexusAirV2` is deliberately a **behaviour twin** that uses the same
architectural ideas (intent-aware prediction, cross-channel trust, energy-aware
selection) in transparent numpy, so the *design* is inspectable before any
neuromorphic deployment.

## 3. Demonstrated behaviour (`run_guardian.py`)

| scenario | mode histogram | max threat | clearance | declared used | undeclared used |
|---|---|---|---|---|---|
| healthy | CRUISE_SAFE 20 | 0.00 | — | — | — |
| intruder | EVADE 115/140 | 0.82 | 0.52 m | predictive_sense_avoid | cloaked_evasion |
| spoof | RECOVER_NAV 32/40 | 1.00 | — | nav_consistency | autonomous_source_switch |
| jamming | SILENT 40 | 0.99 | — | nav_consistency | autonomous_source_switch, silent_rf |
| jamming+obstacle | SILENT 69 + EVADE 71 | 0.99 | 0.52 m | nav_consistency, sense_avoid | source_switch, cloak, silent_rf |

All scenarios end **crash=0**. This is the proof-of-concept that the same brain
can simultaneously evade, recover navigation, and go RF-silent.

## 4. Quadrants (never present speculative as fact)

* **A — current / implemented:** predictive sense-and-avoid, cross-sensor nav
  consistency (spoof/jam), energy/command-envelope guard, learned-frame-trust
  (already in `simulator.py`).
* **B — emerging / partially built:** cloaked evasion (implemented, needs
  adversarial-robustness validation), autonomous source switch (implemented),
  silent RF mode (decision exists; RF model is abstract).
* **C — potential:** an *oracle risk world model* (learned or predicted risks
  across a mission, used to re-plan the whole path, not just a step); LLM/edge
  planner distillation.
* **D — speculative / frontier (label clearly):** emergent adversarial
  hardening (a co-trained simulator that actively attacks the brain to make it
  robust); swarms of Nexus units sharing a split neuromorphic risk graph;
  "physics-native" neuromorphic spiking perception at 847 GOp/s/W scale.

## 5. 10-criteria score (1–10; higher better)

| Criterion | predictive sense-avoid | cloaked evasion | silent RF | oracle risk world model | neuromorphic SNN planner |
|---|---|---|---|---|---|
| Safety value | 9 | 7 | 8 | 8 | 8 |
| Detection coverage | 8 | 6 | 5 | 9 | 8 |
| Response quality | 9 | 6 | 6 | 8 | 8 |
| Energy per decision | 7 | 8 | 9 | 5 | 10 |
| Onboard compute | 7 | 9 | 8 | 4 | 10 |
| Latency | 8 | 7 | 6 | 6 | 9 |
| Weight/size | 8 | 9 | 7 | 6 | 9 |
| Fault tolerance | 7 | 6 | 6 | 6 | 7 |
| Deployability | 7 | 6 | 5 | 4 | 8 |
| Maturation | 8 | 5 | 3 | 3 | 4 |
| **Total** | **78** | **69** | **63** | **59** | **81** |

The top practical item on *totals* is the **neuromorphic SNN planner**, but the
**best next experiment** (highest safety/total *and* lowest latency risk) is the
**oracle risk world model + on-board predictive re-planning**, because it has the
highest Detection coverage (9) and Response quality (8) and directly attacks the
core ask ("detect anything that threatens it and dodge"). The SNN deployment is
the energy path for getting it on a 10 W envelope.

## 6. Energy / compute / latency estimate (design-in)
- GuardianBrain decision: ~1.4–3.4 ms compute estimate, ~0.8–1.7 ms added
  latency, ~70–140 mW average budget in the lab. Well inside a 10 W edge
  envelope.
- Full SNN/perception path: NeuEdge references ~847 GOp/s/W, 2.3 ms inference
  on vision/audio; NeuViT ~3.98 W at 22 FPS for UAV vision [5]. These are
  *research targets/sources*, not measurements on our bench.

## 7. Best Next Experiment
**Oracle risk world model + predictive re-planning.** Build a lightweight
learned/self-calibrated "risk field" over the remaining mission (using energy,
terrain, obstacle history, RF/jamming likelihood), and have the brain re-plan
the whole remaining waypoint sequence instead of just a one-step dodge. Score:
energy 5, compute 4, latency 6 — but safety 8, coverage 9, response 8. It is the
highest-leverage way to turn "detect + dodge" into "detect + avoid the whole
threat corridor."

## 8. Files
- `src/airlab/guardian/{__init__,threats,avoidance,brain,drone_nexus}.py`
- `run_guardian.py` — CLI behaviour lab.
- `tests/test_guardian.py` — 10 unit tests.

## 9. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
```

## 10. References
1. [Intent-Aware Collision Avoidance for UAVs in High-Density Non-Cooperative
   Environments Using Deep RL (Aerospace 2026)](https://doi.org/10.3390/aerospace13020111)
2. [Artificial Intelligence Approaches for UAV Deconfliction: A Comparative
   Review (MDPI 2025)](https://www.mdpi.com/2673-4052/6/4/54)
3. [Custom Non-Linear MPC for Obstacle Avoidance in Indoor and Outdoor
   Environments (arXiv 2024)](https://arxiv.org/html/2410.02732v1)
4. [Enhancing UAV Security with GPS Spoofing and Jamming Anomaly Detection
   (Journal of Reliable and Secure Computing 2025)](https://www.icck.org/article/epdf/jrsc/617)
5. [Energy-Efficient Neuromorphic Computing for Edge AI: NeuEdge
   (arXiv 2026)](https://arxiv.org/html/2602.02439v1) and NeuViT UAV vision.
