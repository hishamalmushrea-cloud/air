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
| `run_consensus.py` | consensus policy comparison between the two independent detectors |
| `plot_results.py` | batch / degradation / corruption / landmark / factor-graph / consensus charts |

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

The repo is at **Stage 19** of the longer research program. The next stages in
order of value:

1. **Sparse factor-graph outage is a first-class failure (implemented).**
   `--fg-flow-outage start-end` removes flow factors from the factor graph only
   (EKF flow aiding unchanged), producing an **under-determined but healthy**
   graph (`comp=0, trust=0, health≈1.0`).  Healthy missions complete with
   **0.000 false land** under every policy; persistent faults are still detected
   after the graph recovers (scale1.5 0.529, bias.25 0.844, crash 0), and
   `adaptive_veto_trust` matches the other arms.  The trust guard correctly
   treats the sparse graph as thin (may trigger, may not veto)
   (`docs/research-brief-18.md`).
2. **A transient flow fault fully contained inside the sparse window** is the
   next honest experiment (needs a transient fault preset); the persistent-fault
   result does not exercise the worst case.
3. **n≥30 statistical grid** for the headline crash numbers (the unwatched
   bias.25 crash at n=3 has a wide CI; detector arms are all zero-crash).
2. **Mission-aware RTL is opt-in and honestly limited.**  A ramping corrupt
   velocity source corrupts navigation before detection, so RTL is a guess
   there.  The useful design-in pieces (early source rejection, GPS-only state
   reset, ground-contact friction) apply to immediate land too; RTL becomes
   attractive for a different fault class (e.g. battery/actuator degradation
   with intact navigation).
3. **Richer landmark geometry** — higher rate, landmark persistence and a
   learned "trust this frame" model.
5. **Physics fidelity v4** — motor/spin dynamics, propeller model, thermal,
   ground effect.
6. **Digital twin** — save a run as telemetry and provide replay/re-command.
7. **Battery-aware mission decision** — use the power model to decide "return
   now" vs "continue", and add a thermal budget.
8. **Learnable components** — learned surrogate for the EKF motion model and a
   policy/neural controller trained in this environment.
9. **Multi-agent swarm** — extend to several vehicles with distributed
   waypoint assignment.
10. **Validation bridge** — port the same mission to PX4 SITL or JSBSim to
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
preintegrator, a startup baseline calibration, and its **own in-graph IMU
accel-bias estimate** (so it no longer borrows the flow-fed EKF bias).  It is
now **on by default as a second independent detector** alongside the landmark
monitor.

Two changes made it live-worthy:

1. **Strong bias prior** (`bias_reg` ≥ 0.2).  A weak prior lets the graph
   *absorb the flow fault into its own bias* and then self-confirm the corrupt
   measurement — it must be forced to reject the outlier instead of fitting it.
2. **Dedicated independent-detector safety path.**  The previous safety layer
   ignored a low velocity-health signal while GNSS was still available ("we have
   a little more room").  That was the exact gap that let a corrupt flow source
   drive the aircraft into the ground during a GNSS-aided phase.  The safety
   layer now has a separate `detector_health` input that can force `LAND` even
   with GPS up, with a wide margin (warn < 0.55, fail < 0.25) and long grace to
   stay silent on healthy fast turns.

The n=10 live ablation (`docs/research-brief-07.md`) shows the FG alone
converts bias.25 crashes from 0.024 → **0.000** (10/10 `landed_safely`) and
detects even bias.1, while healthy remains 10/10 `completed`.

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
