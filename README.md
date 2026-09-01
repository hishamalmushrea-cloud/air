# AIR Lab — Autonomous UAV Research & Innovation Lab

A growing, readable reference implementation of a **full autonomy stack** for a
quadrotor, built to research and test ideas rather than to reproduce PX4/Gazebo.
The repo is deliberately small and self-contained (only `numpy`), so every piece
of the stack is inspectable and replaceable.

> Design intent: **together, not separately**. Flight dynamics, sensing,
> estimation, control, mission logic, fault injection, and evaluation all live
> side by side, so an idea about one component can be tested against the whole
> system.

---

## What is implemented

```
 Mission/waypoints
        │
        ▼
 ┌─────────────────────────────┐
 │  Flight Controller          │  position → attitude → rate (cascaded)
 │  (pos→att→rate)             │
 └──────────────┬──────────────┘
                ▼
 ┌─────────────────────────────┐
 │  Quadrotor dynamics (NED)   │  rigid body, rate inner loop, wind
 └──────────────┬──────────────┘
                ▼
 ┌─────────────────────────────┐
 │  Sensor suite               │  IMU 100Hz, AHRS 50Hz, baro 20Hz,
 │                             │  magnetometer 50Hz, optical flow 25Hz,
 │                             │  GNSS 10Hz (with fault injection)
 └──────────────┬──────────────┘
                ▼
 ┌─────────────────────────────┐
 │  Complementary AHRS         │  gyro + accel leveling + mag heading
 └──────────────┬──────────────┘
                ▼
 ┌─────────────────────────────┐
 │  Navigation EKF (15-state)  │  loosely-coupled INS/GPS/baro/AHRS/flow
 └─────────────────────────────┘
```

### Files

| File | Purpose |
|---|---|
| `src/airlab/math_utils.py` | NED, quaternion, Euler utilities |
| `src/airlab/dynamics.py` | quadrotor rigid-body dynamics |
| `src/airlab/sensors.py` | IMU/GNSS/baro/mag/AHRS/optical-flow sensor models |
| `src/airlab/ahrs.py` | complementary-filter attitude reference |
| `src/airlab/fusion.py` | 15-state navigation EKF |
| `src/airlab/control.py` | cascaded position/attitude/rate controller |
| `src/airlab/mission.py` | waypoint trajectory + feedforward/deceleration |
| `src/airlab/energy.py` | lightweight mission power/energy model |
| `src/airlab/safety.py` | uncertainty-aware safety FSM (`CRUISE/HOLD/LAND/LANDED`) + integrity monitor |
| `src/airlab/landmarks.py` | independent landmark-field consistency detector |
| `src/airlab/factorgraph.py` | sliding-window nonlinear factor graph + IMU preintegrator + robust flow-residual detector |
| `src/airlab/simulator.py` | orchestrator, subsystem rates, fault timing |
| `src/airlab/metrics.py` | navigation and safety metrics |
| `src/airlab/scenarios.py` | serializable scenario descriptors + random mission generator |
| `src/airlab/experiments.py` | batch runner, degradation, corruption, landmark ablation, CSV output |
| `src/airlab/main.py` | CLI demo: baseline vs GNSS outage |
| `run.py` | convenient `python run.py` wrapper |
| `run_batch.py` | random scenario batch CLI |
| `run_degradation.py` | GNSS-outage degradation study CLI |
| `run_corruption.py` | corrupt velocity-aiding + safety-layer study CLI |
| `run_landmark.py` | independent landmark detector ablation CLI |
| `run_factorgraph.py` | factor-graph detector characterisation CLI |
| `run_factorgraph_live.py` | calibrated factor-graph live-detector ablation CLI |
| `plot_results.py` | batch / degradation / corruption / landmark / factor-graph charts |

---

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install numpy
.venv/bin/python run.py --duration 40 --outdir out
```

This runs two missions:

* **baseline** — full sensor suite, no faults.
* **GNSS outage** — 12–24 s of GPS dropout; optical flow + baro keep the
  navigation bounded.

Both save `out/summary.csv` and print per-metric comparisons.

---

## Current baseline results (40 s waypoint mission)

| Metric | Baseline | GNSS outage (12–24 s) |
|---|---:|---:|
| position RMSE (m) | 0.60 | 0.64 |
| horizontal RMSE (m) | 0.53 | 0.57 |
| vertical RMSE (m) | 0.29 | 0.30 |
| max horizontal error (m) | 1.15 | 1.08 |
| estimator horizontal RMSE (m) | 0.12 | 0.20 |
| in-bounds fraction | 1.00 | 1.00 |
| ground collision fraction | 0.00 | 0.00 |

Without the optical-flow velocity aiding, the same GNSS outage makes the
navigation diverge and the vehicle crash — that is the **safety gap** the
sensor-fusion layer is designed to close.

---

## Methodology / decision notes

* **Coordinates.** NED airframe convention; missions are written in
  `(north, east, height)` and converted to NED internally.
* **Subsystem rates.** Every sensor runs at a realistic rate (IMU 100 Hz,
  GNSS 10 Hz, etc.), so the EKF is doing actual fusion, not reading truth.
* **Estimator.** 15-state EKF with numerical Jacobians. Numeric Jacobians make
  it easy to add a new measurement model without deriving a large sparse matrix
  by hand.
* **Attitude reference.** The AHRS uses a complementary filter. Its accel
  leveling sign convention is explicitly documented and validated against the
  rotation matrix (this was a real bug that made the first build fly backwards).
* **Controller.** Cascaded position → attitude → rate, with a horizontal
  attitude reference derived from `a = R @ [0,0,-T] + g`, plus tilt
  compensation on the vertical thrust command.
* **Ablation.** `SimConfig.truth_estimate = True` feeds the controller perfect
  state, isolating control bugs from estimation issues.
* **Safety.** A GNSS-outage scenario is a first-class simulation, and the
  metrics include in-bounds fraction and ground-collision fraction, not just
  RMSE.

---

## Roadmap (current thinking)

The repo is at **Stage 6** of the longer research program. The next stages in
order of value:

1. **Make the factor-graph IMU factor fully independent** — estimate IMU bias
   inside the graph (or use graph-own preintegration) instead of borrowing the
   flow-fed EKF bias.  This is the single highest-value change; it should make
   the calibrated factor graph replace the landmark detector as the live signal
   (see `docs/research-brief-05.md`).
2. **Consensus of two monitors** — run the landmark detector as primary and the
   calibrated factor graph as a cross-check (two structurally different
   opinions).
3. **Richer landmark geometry** — higher rate, landmark persistence and a
   learned "trust this frame" model.
3. **Physics fidelity v4** — motor/spin dynamics, propeller model, thermal,
   ground effect.
4. **Digital twin** — save a run as telemetry and provide replay/re-command.
5. **Battery-aware mission decision** — use the power model to decide "return
   now" vs "continue", and add a thermal budget.
6. **Learnable components** — learned surrogate for the EKF motion model and a
   policy/neural controller trained in this environment.
7. **Multi-agent swarm** — extend to several vehicles with distributed
   waypoint assignment.
8. **Validation bridge** — port the same mission to PX4 SITL or JSBSim to
   compare the lightweight model against a real EOM.

---

## Safety layer (Stage 3)

An explainable `CRUISE → HOLD → LAND → LANDED` decision layer uses:
* EKF horizontal position uncertainty (std-dev from covariance),
* velocity-aiding health self-report,
* GNSS availability,
* a fixed vertical descent when navigation is not trustworthy.

The point is to **trade accuracy for survivability**: a wrong navigation source
cannot be made accurate by a safety layer, but the layer can convert a likely
crash into a *controlled landing* at a degraded location.  See
`docs/research-brief-02.md` for the results and the important caveat that a
persistent uninstrumented bias fault is effectively invisible to a single-stream
filter.

To close that gap, the repo now includes an **independent landmark-based
consistency detector** (`landmarks.py`): it compares camera measurements of a
known landmark field against the EKF-predicted geometry using **inter-landmark
angles** (invariant to attitude error).  Because it observes real world geometry
rather than re-integrating the same IMU, it is the only thing that can catch a
"convincing but wrong" VIO bias fault that the VIO self-report cannot see.  The
ablation study (`docs/research-brief-03.md`) shows it raises controlled landings
for the hardest fault (bias.25) from 0.00 → 0.44 at 30 s GNSS loss, while
healthy missions still finish without false alarms.

For the research-grade version, `factorgraph.py` builds a small sliding-window
factor graph (IMU + flow + GPS + landmarks) with a Cauchy robust kernel, an IMU
preintegrator, and a startup baseline calibration.  It now behaves as a **valid
calibrated detector in isolation** (healthy → health ~1.0, an obvious fault →
health ~0.05) and is tested via `run_factorgraph_live.py`.  It is still **opt-in**
because the graph's IMU factor depends on the EKF's accel-bias estimate, so a
corrupt flow can contaminate it; the landmark detector remains the live signal
(see `docs/research-brief-05.md`).

---

## Ethical / scope boundary

This project is for **research, education, simulation, civil applications**
(search & rescue, inspection, agriculture, mapping, science), and defensive
engineering (safety, fault tolerance, cyber-hardening). It does not and will
not include weapons targeting, guidance of munitions, or autonomous lethal
systems. Any sensitive topic is redirected into simulation / defensive
engineering / prevention.

---

## License

MIT — see `LICENSE`.
