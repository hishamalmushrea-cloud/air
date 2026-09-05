# Nexus-Predator Master Program

Operating under `docs/ULTRA_MASTER_PROMPT.md`. This document is the live
engineering record for the next-generation autonomous UAV platform.

**Framework:** ULTRA MASTER PROMPT (adopted).
**Platform:** Nexus-Predator (defensive AI guardian core + full stack).
**Immutable scope:** non-weaponised, safe/non-offensive civil UAV platform.

---

## 1. Objective
Build a self-contained, research-grade autonomous UAV platform that detects any
threat to itself (obstacle, spoof/jam, wind, cyber-command, energy), dodge/
evade/recover/silence defensively, and predicts and avoids whole threat corridors
over the remaining mission — all inside a realistic energy / compute / weight
envelope, with evidence (simulated + tested) at every step.

## 2. Current State
- Full simulation stack: dynamics, sensors (IMU/GNSS/baro/mag/flow), fusion EKF,
  AHRS, cascaded controller, mission, energy, safety FSM, fault injection.
- Safety/consensus research line (Stages 1–22): independent landmark + factor-graph
  detectors, adaptive consensus, learned frame trust, degenerate-parallax &
  sparse-FG faults, n=30 statistical confirmation, transient-fault &
  persistence-gate findings. All committed and pushed.
- **New platform initiative (Stage 23+):** `guardian/` package — ThreatEngine,
  EvasionPlanner, GuardianBrain, NexusAirV2, RiskWorldModel, PredictiveRePlanner.
  `run_guardian.py` behaviour lab passes healthy/intruder/spoof/jamming/
  jamming+obstacle/replan, all crash-free.

## 3. Phases status (per master §42)

| Phase | Status |
|---|---|
| 1 Mission Definition | ✔ (Nexus-Predator: safe multirole + SAR/AG/inspection) |
| 2 System Requirements | ✔ partial (see §4) |
| 3 Trade Study | ✔ partial (10-criteria rubric in brief-22) |
| 4 Airframe Architecture | ◐ proposed (quad; VTOL hybrid on roadmap) |
| 5 Aerodynamics | ◐ proposed; not yet numerically simulated |
| 6 Propulsion | ◐ PowerModel only; motor/ESC/prop not modelled as system |
| 7 Power | ◐ PowerModel + energy reserve; thermal not modelled |
| 8 Avionics | ◐ abstract (edge compute estimate) |
| 9 Sensors | ✔ family defined; per-sensor health model in progress |
| 10 Navigation | ✔ robust EKF + GNSS-independent detectors |
| 11 Flight Control | ✔ cascaded PID + safety FSM; guard-rail not in main loop |
| 12 AI / Vision | ◐ analytics + guardian; no learned perception model yet |
| 13 Software | ✔ modular + tests; Android/GCS not built |
| 14 Communications | ◐ abstract RF silence / jam detection only |
| 15 Ground Station | ◐ not built |
| 16 Simulation | ✔ strong; SITL/HIL not yet |
| 17 Digital Twin | ◐ behaviour twin only |
| 18 Testing | ✔ 52 tests + statutory grids; no real rig |
| 19 Manufacturing | ◐ not started |
| 20 Maintenance | ◐ Predictive maintenance proposed → next executable task |
| 21 Documentation | ✔ briefs + README |
| 22 Optimization | ◐ partly (risk route, energy) |
| 23 Next-Gen Research | ◐ neuromorphic SNN planner, learned oracle, swarm |

## 4. System requirements (draft, confidence Medium)
- Payload: ≤ 350 g (multi-sensor + optional economy camera).
- Mass ≤ 1200 g all-up (twin), hover power ~112 W, endurance ~38 min.
- Onboard edge compute ~0.35 TOPS class, within a 10 W envelope.
- Independent safety path must disagree with a single sensor fault
  (already validated: landmark FG consensus).
- Must survive: obstacle intrusion, GPS spoof/jam, excessive wind, cyber-command
  anomaly, battery shortfall — defensively.
- Must log and report declared + undeclared defensive capabilities used.

## 5. Architecture current
```
sensors → fusion EKF → controller → dynamics
                 │
                 └── safety FSM (CRUISE/HOLD/LAND/LANDED)
                 └── guardian (ThreatEngine → GuardianBrain)
                          ├── EvasionPlanner (one-step dodge + cloak)
                          └── RiskWorldModel + PredictiveRePlanner (corridor)
```

## 6. Backlog (ranked: Impact × Feasibility × Risk × Cost × Innovation)

| Priority | Item | Status |
|---|---|---|
| Critical | Port predictive re-planning into the real mission controller (`simulator.py`) | **✅ done (brief-25)** |
| High | **Predictive maintenance / subsystem Health Score** | **✅ done (brief-24)** |
| High | Feed health engine real simulator telemetry | **✅ done (brief-26)** |
| High | Learned risk prior from flight/jam telemetry | **NEXT → priority #3** |
| High | GCS telemetry + health dashboard | near future |
| Medium | Thermal model (CPU/ESC/motor/battery) | NOW |
| Medium | Data pipeline (edge → storage → analytics) | NOW |
| Medium | Motor/ESC/prop system model (thrust/power/temp) | RESEARCH |
| Experimental | Neuromorphic SNN planner @ ~847 GOp/s/W class | LONG TERM |
| Experimental | Swarm split-risk graph | LONG TERM |
| Long-term | Cross-domain morphing/adaptive airframe | THEORETICAL |

**Ranking rationale (Impact / Feasibility / Risk↓ / Innovation / Cost / Test):**
- #1 Port oracle→real controller **44** (high impact, closes the behaviour-lab gap).
- #2 Health engine from real telemetry **40** (cheap, converts estimate→measured).
- #3 Learned risk prior **38** (high methodology value, needs data pipeline).
- #4 Thermal model **36**
- #5 Neuromorphic SNN / NeuViT perception **33** (highest innovation, long-term).

## 7. Generation log (G0 → G5)
- **G0**: baseline stack (dynamics/EKF/controller/safety FSM).
- **G1**: independent landmark + factor-graph detectors; bias.25 crash → 0.
- **G2**: learned frame trust + adaptive_veto_trust; degenerate-parallax &
  sparse-FG & transient-fault characterised; persistence-gate tradeoff known.
- **G3**: Guardian brain + oracle risk re-planning (this program start).
- **G4**: predictive maintenance / health score + guardian oracle ported into
  the real mission controller (`MissionReplanBridge` in `simulator.py`).
- **G5**: health engine consumes real fused simulator telemetry
  (`TelemetryHealthBridge`, physical mid-flight motor degradation).
- **G6 (next)**: learned risk prior from recorded flight/jam telemetry; then
  part-level low-watt thermal model.

## 8. Engineering Decision Log (current entries)

| Decision | Reason | Alternatives | Data | Expected | Actual | Change | Version |
|---|---|---|---|---|---|---|---|
| Guardian is defensive-only | master scope + ethics | offensive (rejected) | master prompt | safe platform | validated | — | G3 |
| RiskWorldModel analytic | transparent, no external deps | learned neural field | brief-23 | corridor avoid | risk 0.525→0.180 | — | G3 |
| Predictive RePlanner beam | simple, bounded, no black box | RL / MPC | brief-23 | small detour, big risk cut | confirmed | — | G3 |
| Route-edit bridge gates (feasible/risk/clearance/detour) | safety-in-design | unconditional replan (rejected) | brief-25 | no unsafe rewrite | risk .459→.001, clr 5.16 m, 0 land | — | G4 |
| Detour cap 0.50 | large risk cut can justify big lateral swing | 0.25 (rejected too-strict) | brief-25 | allow 20 % detour | first real detour 20.2 % applied | — | G4 |

## 9. Confidence
- Current architecture: **High** for safety/consensus line; **Medium** for the new
  guardian (analytic, not yet learned); **Low** for any mass/endurance figure
  (not yet calculated/measured).

## 10. Next autonomous tasks
1. **Predictive maintenance / subsystem Health Score engine** (High) → implement
   + test now.
2. Export a health score into the guardian so a degrading motor/battery/sensor
   raises an `ABORT` / maintenance alert before it becomes a crash.
3. Port oracle re-planning into the real mission controller.
4. Learned risk prior from telemetry.
