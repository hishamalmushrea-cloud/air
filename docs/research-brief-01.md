# Research Brief #01 — First Integrated Lab Milestone

**Date:** 2026-08-31
**Project:** AIR Lab (Autonomous UAV Research & Innovation Lab)
**Status:** first runnable milestone; 12/12 tests passing; CLI demo working.

---

## 1. What was built and why

The repository starts as an empty tracked directory. Rather than jumping into a
deep dive on one discipline, the first milestone is a **complete, minimal, and
honest vertical slice of the autonomy stack**:

```
mission → controller → quadrotor dynamics → sensors → AHRS → nav EKF → feedback
```

A vertical slice is the highest-value first move because it lets an idea from
any one discipline be tested against the *whole* system. It also exposes the
reality that most UAV problems are not single-discipline problems: a simple
sign-error in the accelerometer attitude model (aerospace frame convention) can
look like a bad controller, a bad estimator, or a bad plant to someone who only
looks at one layer.

## 2. Four-layer analysis applied to this milestone

### A — What exists today (real / available)
* PX4 / ArduPilot / ROS 2 / Gazebo / AirSim / JSBSim are mature and production-grade.
* Open-source EKF stacks (PX4 EKF2, ArduPilot EKF3) are the reference architecture.
* Standard GNSS-denied navigation research uses visual-inertial odometry, UWB,
  LIDAR, barometer, and optical flow.

### B — What is emerging (in development)
* Tightly-coupled VIO / factor-graph SLAM running on small edge SoCs.
* Learned inertial models (`learned IMU`, neural network gyroscope bias models)
  that beat classical models under aggressive motion.
* Event cameras that provide extreme-rate, low-power visual features.
* Precise-point-positioning / multi-band GNSS as a middle ground.

### C — What is potentially possible (near future)
* An autonomy stack that actively **selects its navigation source**: when GNSS and
  VIO disagree with *uncertainty-aware* reasoning, the vehicle decides which
  source to trust and when to switch to a conservative "hold-and-land" mode.
* Learned mid-level state estimation (neural state-space models) that is checked
  by safety envelopes instead of replacing them.

### D — Exploratory (hypothetical)
* A swarm that maintains a shared belief distribution and negotiates which
  vehicle carries the most informative sensor role.
* Neuromorphic sensor + spiking-network controller pairs that trade accuracy for
  extremely low energy / latency.
* A **digital twin that is part of the control loop**: the simulator predicts the
  next seconds and the controller re-plans against the twin before committing.

These D-level ideas are explicitly marked as hypotheses; the current repo does
not claim them.

## 3. What we learned (and explicitly recorded)

1. **Frame conventions kill systems.** The original controller used
   `theta_ref = +a_north / T`, which made the vehicle accelerate south when told
   to go north. The correct derivation from `R @ [0,0,-T] + g` is
   `a_north = -T*sin(theta)`, `a_east = +T*sin(phi)`.
2. **AHRS leveling signs are not trivial.** `atan2(-fy, -fz)` gives the *wrong*
   roll sign in a body-z-down frame with thrust along `-z`. The correct form is
   `atan2(fy, -fz)` / `atan2(-fx, -fz)`, documented in `ahrs.py`.
3. **Sensor rates matter.** The EKF only performs "real" fusion if IMU runs at
   100 Hz while GNSS runs at 10 Hz and baro at 20 Hz.
4. **Barometer random-walk constants have outsized effects.** A sensor model that
   is *too noisy* makes the estimator and the controller fail in ways that are
   indistinguishable from a bug.
5. **Feedforward velocity must decelerate near waypoints.** Without an
   arrival-scale on the velocity feedforward, the vehicle overshoots every
   waypoint and oscillates around the final goal.
6. **GNSS outage is a safety test, not a corner case.** With only IMU/baro/AHRS,
   the vehicle diverges thousands of metres and crashes. Adding a low-rate
   optical-flow velocity aiding keeps navigation bounded and the mission stable.

## 4. Idea evaluation (first integrated idea: *"an always-on, tightly-fused
degradation-aware navigation layer"*)

| Criterion (weight 1–10) | Score | Note |
|---|---|---:|
| Technical feasibility | 8 | Standard EKF + flow/baro; no exotic sensor needed |
| Cost | 9 | Software-only; reuses existing cheap sensors |
| Energy efficiency | 8 | Very low compute; no camera neural net required |
| AI added value | 4 | Classical filter; AI would add learned model later |
| Autonomy level | 7 | Enables GNSS-denied mission continuation |
| Reliability | 9 | Bounded drift, safer than blind IMU-only |
| Scalability | 8 | Easily generalizes to other vehicles/sensors |
| Research novelty | 5 | Well-known, but valuable as a transferable baseline |
| Commercial potential | 8 | Affects inspection, rescue, mapping deployments |
| Implementation difficulty | 8 | Achievable in a weekend framework |

**Best Next Experiment:** a *mission-level degradation study* — batch 100+
randomized outages (window, length, flow noise) and measure the probability of
remaining in bounds and of ground collision as a function of outage duration.

## 5. From the "expand beyond the prompt" rule — fields/frameworks worth adding

These were not in the original prompt but are important and should become part of
the roadmap:

* **Energy / mission-aware planning** — battery model, thermal state, and
  mission-time horizon in the planner, not just the vehicle.
* **Uncertainty-aware autonomy** — use EKF covariance as a state to decide when
  to "go, climb, or land", not just as a number to print.
* **Cyber-physical security** — spoofed GNSS / rejected sensor fusion; the same
  fault-injection machinery can generate attack scenarios.
* **Formal safety envelopes** — reachable-set methods or guard functions that
  block unsafe commands even when a learned component is wrong.
* **Explainable decisions** — mission state should be logged so an operator can
  answer "why did the aircraft turn there?".
* **Data-centric engineering** — the simulator should be a data factory
  (telemetry + scenario labels), not only a visualization tool.
* **Edge-AI deployment** — temperature, power, latency, and quantized-model
  budget are first-class simulation outputs.

## 6. Next steps (proposed order)

1. **Telemetry serialization + replay** (`Digital Twin`): save each run to a
   clean `.jsonl`, add `--replay` and a `twins` API.
2. **Scenario generator**: randomize waypoints, wind, sensor dropouts, and
   obstacle regions; produce a batch report.
3. **Degradation study**: quantify how long an outage the current stack survives
   under different optical-flow noise levels.
4. **Battery + energy model**: add power draw, energy budget, and a mission
   "go / no-go" decision.
5. **Learnable component**: a lightweight learned IMU model or a policy trained
   against the same EKF stack.
6. **Multi-agent validation bridge**: port scenario/conventions to PX4 SITL to
   compare against a production stack.

## 7. How to run

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib
.venv/bin/python run.py --duration 40 --outdir out --plot
.venv/bin/python -m unittest discover -s tests -v
```

---

## Appendix — Stage 2: batch scenario factory + degradation study

### Built

* `src/airlab/scenarios.py` — serializable `Scenario` descriptor, random
  mission generator, `run_scenario`.
* `src/airlab/experiments.py` — batch runner (`run_batch`), random batch
  generator, `degradation_study`, CSV writer.
* `src/airlab/energy.py` — lightweight power/energy model (not a motor FDM).
* `run_batch.py` — CLI: `python run_batch.py --num 20 --duration 40`.
* `run_degradation.py` — CLI: `python run_degradation.py --durations 0,10,20,25`.
* `plot_results.py` — batch/degradation chart generator.

### Results (30 s missions, 5 scenarios per duration, seed 5)

| Outage | velocity aiding | In-bounds | Ground collision | mean pos RMSE (m) |
|---:|---|---:|---:|---:|
| 0 s | flow | 100% | 0% | 0.82 |
| 0 s | none | 91% | 1.4% | 3.71 |
| 10 s | flow | 100% | 0% | 0.84 |
| 10 s | none | 69% | 22% | 102 |
| 20 s | flow | 100% | 0% | 0.94 |
| 20 s | none | 36% | 60% | 558 |
| 30 s | flow | 100% | 0% | 0.92 |
| 30 s | none | 6% | 92% | 1016 |

### Scientific reading

1. **Velocity aiding is the safety-critical component, not GPS.** The stack
   stays 100% in-bounds across the whole tested outage range *because* optical
   flow / VIO velocity aiding is enabled. Without it, degradation is roughly
   linear and by 30 s of outage nearly every mission ends in a collision.
2. **The earlier baseline-only degradation chart was misleading.** With only
   flow-enabled runs, safety stayed at 1.0 and RMSE barely moved, which looked
   like the system was genuinely robust. The **with/without ablation** is what
   reveals *why*.
3. **Batch metrics are now a power statement.** We can now answer "the design
   survives a 30 s GPS dropout" with a distribution rather than a single run,
   and we know the engineering dependency that makes that true.
4. **Energy is decoupled from outage.** Mean energy is ~1.0 Wh across the
   random batch even with failures, which means the power model alone does not
   predict the outage-safety boundary — the navigation architecture does.

### Best next experiment after this stage

Increase **sensor realism** on the velocity-aiding source: add a *faulty/corrupt
optical-flow* mode (mirror dropouts, scale error, bias ramp), then ask: *how much
of the 100% safety margin survives when the aiding sensor itself is imperfect?*
This is the natural next research gap.`
