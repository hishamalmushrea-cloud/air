"""Orchestrator: connect vehicle + sensors + estimator + controller + mission.

Run at a base simulation step of 10 ms. Each subsystem samples at its own rate
(IMU 100 Hz, AHRS 50 Hz, baro 20 Hz, GPS 10 Hz), exactly like a real flight
stack, so that the fusion has to interpolate and *fuse* rather than being
handed perfectly co-located measurements.
"""

from __future__ import annotations

import numpy as np

from .dynamics import Quadrotor
from .sensors import SensorSuite, SensorConfig
from .ahrs import ComplementaryAHRS
from .fusion import (NavEKF, GPS, BARO, AHRS, FLOW,
                     S_IDX, V_IDX, R_IDX, BG_IDX, BA_IDX, _wrap_state)
from .control import FlightController
from .mission import WaypointMission
from .metrics import evaluate_mission
from .safety import (SafetyMonitor, SafetyConfig, CRUISE, HOLD, LAND, LANDED,
                     VelocityIntegrityMonitor)
from .landmarks import LandmarkField, LandmarkConsistency
from .factorgraph import SlidingFactorGraph, build_keyframe


class SimConfig:
    def __init__(self) -> None:
        self.dt = 0.01
        self.duration = 40.0
        self.wind_ned = np.array([0.0, 0.0, 0.0])
        # Fault scenario: GNSS outage window (seconds) or None.
        self.gps_outage = None  # (start, end)
        self.gps_outage_scale = 1.0
        self.seed = 7

        # Gate optional velocity-aiding sensor (optical flow / VIO surrogate).
        self.flow_enabled = True
        # Velocity-aiding faults (windows / scales / bias ramp).
        self.flow_outage = None              # (start, end)
        self.flow_scale = 1.0
        self.flow_bias_ramp = 0.0

        # Uncertainty-aware safety layer.
        self.safety_enabled = True
        self.safety_kwargs = {}
        # Optional parallel IMU dead-reckon / integrity check.  Currently off
        # by default: an open-loop IMU dead-reckon is not reliable enough as an
        # independent monitor over long outages (see research-brief-02).
        self.integrity_monitor_enabled = False

        # Landmark-based independent consistency check.  This is the preferred
        # "second opinion": it observes true world geometry, so it can catch a
        # corrupt velocity-aiding source that the EKF would otherwise trust.
        self.landmark_enabled = True
        self.landmark_hz = 5.0

        # Principled factor-graph consistency monitor (IMU+flow+GPS+landmarks,
        # jointly optimised with a robust kernel).  Currently opt-in for
        # analysis: the live safety signal stays on the proven landmark
        # detector while we characterise when the graph residual is reliable.
        self.factorgraph_enabled = False
        self.factorgraph_hz = 2.0
        self.factorgraph_kwargs = {}

        self.vehicle_kwargs = {}
        self.sensor_kwargs = {}
        self.ekf_kwargs = {"r_scale": 1.0, "q_scale": 1.0}
        self.controller_kwargs = {}
        # Ablation flag: feed the controller perfect state (for debugging /
        # separating control performance from estimation performance).
        self.truth_estimate = False

        # Waypoints are (north, east, height) with height positive-up.
        self.waypoints = [
            (0.0, 0.0, 2.0),
            (8.0, 0.0, 3.0),
            (8.0, 8.0, 4.0),
            (-2.0, 8.0, 5.0),
            (-2.0, -2.0, 5.0),
        ]
        self.cruise_speed = 2.0


class SimRun:
    """A single simulation run; keeps every recorded signal."""

    def __init__(self) -> None:
        self.t = []
        self.true_pos = []
        self.true_vel = []
        self.true_rpy = []
        self.est_pos = []
        self.est_vel = []
        self.est_rpy = []
        self.ref_pos = []
        self.ref_vel = []
        self.ref_yaw = []
        self.gps_available = []
        self.imu = []
        self.control = []
        self.cost = []
        self.mode = []
        self.reason = []
        self.unc_horiz_std = []
        self.flow_health = []
        self.flow_mismatch = []
        self.flow_dr_vel = []
        self.landmark_score = []
        self.landmark_residual = []
        self.factorgraph_health = []
        self.factorgraph_residual = []
        self.landed = []

    def record(self, sim, pos_ref, vel_ref, yaw_ref) -> None:
        self.t.append(sim.time)
        self.true_pos.append(sim.vehicle.pos.copy())
        self.true_vel.append(sim.vehicle.vel.copy())
        self.true_rpy.append(sim.vehicle.rpy.copy())
        self.est_pos.append(sim.ekf.pos)
        self.est_vel.append(sim.ekf.vel)
        self.est_rpy.append(sim.ekf.rpy)
        self.ref_pos.append(pos_ref)
        self.ref_vel.append(vel_ref)
        self.ref_yaw.append(yaw_ref)
        self.gps_available.append(1.0 if sim.sensors.cfg.gps_dropout <= 0.0 else 0.0)
        self.imu.append([sim.last_imu.accel.copy(), sim.last_imu.gyro.copy()])
        self.control.append(sim.last_control.copy())
        self.cost.append(float(np.linalg.norm(sim.vehicle.pos - sim.ekf.pos)))
        self.mode.append(sim.safety.mode)
        self.reason.append(sim.safety.reason)
        self.unc_horiz_std.append(sim._last_unc_std)
        self.flow_health.append(sim._effective_flow_health)
        self.flow_mismatch.append(sim._last_flow_mismatch)
        self.flow_dr_vel.append(sim.flow_integrity.vel.copy())
        self.landmark_score.append(sim._landmark_score)
        self.landmark_residual.append(sim._landmark_residual)
        self.factorgraph_health.append(sim._factorgraph_health)
        self.factorgraph_residual.append(sim._factorgraph_residual)
        self.landed.append(1.0 if sim.safety.mode == LANDED else 0.0)

    def as_arrays(self) -> dict:
        d = dict(self.__dict__)
        for k in ("t", "true_pos", "true_vel", "true_rpy", "est_pos", "est_vel",
                  "est_rpy", "ref_pos", "ref_vel", "control", "cost",
                  "unc_horiz_std", "flow_health", "flow_mismatch",
                  "flow_dr_vel", "landmark_score", "landmark_residual",
                  "factorgraph_health", "factorgraph_residual", "landed"):
            d[k] = np.asarray(d[k], dtype=float)
        d["ref_yaw"] = np.asarray(d["ref_yaw"], dtype=float)
        d["gps_available"] = np.asarray(d["gps_available"], dtype=float)
        d["mode"] = d["mode"]
        d["reason"] = d["reason"]
        d["imu"] = d["imu"]
        return d


class Simulator:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.cfg = config if config is not None else SimConfig()
        self.vehicle = self._make_vehicle(self.cfg)
        self.sensors = self._make_sensors(self.cfg)
        self.ahrs = ComplementaryAHRS(rpy=self.vehicle.rpy)
        self.ekf = NavEKF(self.vehicle.pos, self.vehicle.vel, self.vehicle.rpy,
                          backend="numeric", **self.cfg.ekf_kwargs)
        self.controller = FlightController(**self.cfg.controller_kwargs)
        self.mission = WaypointMission(self.cfg.waypoints, speed=self.cfg.cruise_speed)
        self.safety = SafetyMonitor(SafetyConfig(**self.cfg.safety_kwargs))
        self.safety.cfg.enabled = self.cfg.safety_enabled
        self.flow_integrity = VelocityIntegrityMonitor()
        self.flow_integrity.reset(self.vehicle.vel, self.vehicle.rpy)
        # IMU-only dead-reckon position, independent of the flow-fed EKF.
        self._dr_pos = self.vehicle.pos.copy()
        self._dr_vel = self.vehicle.vel.copy()
        self.landmark_field = LandmarkField()
        self.landmark_consistency = LandmarkConsistency(self.landmark_field)
        self.factorgraph = SlidingFactorGraph(self.landmark_field.positions,
                                              **self.cfg.factorgraph_kwargs)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.time = 0.0
        self.last_imu = None
        self.last_control = np.zeros(4)
        self._last_unc_std = 0.0
        self._effective_flow_health = 1.0
        self._last_flow_mismatch = 0.0
        self._landmark_score = 1.0
        self._landmark_residual = 0.0
        self._factorgraph_health = 1.0
        self._factorgraph_residual = 0.0

    def _make_vehicle(self, cfg: SimConfig) -> Quadrotor:
        return Quadrotor(**cfg.vehicle_kwargs)

    def _make_sensors(self, cfg: SimConfig) -> SensorSuite:
        return SensorSuite(**cfg.sensor_kwargs)

    # ---------------- helpers ----------------
    def _sync_truth_estimate(self) -> None:
        """Debug / ablation helper: force the EKF state to ground truth."""
        x = self.ekf.x
        x[S_IDX] = self.vehicle.pos
        x[V_IDX] = self.vehicle.vel
        x[R_IDX] = self.vehicle.rpy
        x[BG_IDX] = 0.0
        x[BA_IDX] = 0.0
        self.ekf.x = _wrap_state(x)

    def _apply_faults(self) -> None:
        if self.cfg.gps_outage is not None:
            start, end = self.cfg.gps_outage
            if start <= self.time < end:
                self.sensors.cfg.gps_dropout = max(self.sensors.cfg.gps_dropout,
                                                   end - self.time)
            else:
                self.sensors.cfg.gps_dropout = 0.0
        self.sensors.cfg.gps_noise_scale = self.cfg.gps_outage_scale

        # velocity-aiding corruption windows / scales
        if self.cfg.flow_outage is not None:
            start, end = self.cfg.flow_outage
            if start <= self.time < end:
                self.sensors.cfg.flow_dropout = max(self.sensors.cfg.flow_dropout,
                                                    end - self.time)
            else:
                self.sensors.cfg.flow_dropout = 0.0
        self.sensors.cfg.flow_scale = self.cfg.flow_scale
        self.sensors.cfg.flow_bias_ramp = self.cfg.flow_bias_ramp

    def _gps_available(self) -> bool:
        return self.sensors.cfg.gps_dropout <= 0.0

    def _safety_override(self, pos_ref: np.ndarray, vel_ref: np.ndarray,
                         yaw_ref: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Modify the commanded trajectory based on the safety mode."""
        if mode == CRUISE:
            return pos_ref, vel_ref
        est = self.ekf.pos
        if mode == HOLD:
            return est.copy(), np.zeros(3)
        if mode == LAND:
            # Sit on the ground as the goal, descend at a fixed rate.
            ground = float(self.vehicle.ground)
            target = np.array([est[0], est[1], ground])
            v = np.array([0.0, 0.0, -self.safety.cfg.land_speed])
            return target, v
        if mode == LANDED:
            # Hold on the ground; don't keep commanding descent (which can
            # re-accelerate the vehicle after touchdown).
            ground = float(self.vehicle.ground)
            target = np.array([est[0], est[1], ground])
            return target, np.zeros(3)
        return pos_ref, vel_ref

    def run(self, duration: float | None = None, record: bool = True) -> SimRun:
        duration = duration if duration is not None else self.cfg.duration
        out = SimRun()
        dt = self.cfg.dt
        steps = int(round(duration / dt))

        imu_period = 1.0 / max(self.sensors.cfg.imu_hz, 1.0)
        ahrs_period = 1.0 / max(self.sensors.cfg.ahrs_hz, 1.0)
        baro_period = 1.0 / max(self.sensors.cfg.baro_hz, 1.0)
        gps_period = 1.0 / max(self.sensors.cfg.gps_hz, 1.0)
        mag_period = 1.0 / max(self.sensors.cfg.mag_hz, 1.0)
        flow_period = 1.0 / max(self.sensors.cfg.flow_hz, 1.0)
        landmark_period = 1.0 / max(self.cfg.landmark_hz, 1.0)
        factorgraph_period = 1.0 / max(self.cfg.factorgraph_hz, 1.0)

        t_acc = 0.0
        ah_acc, baro_acc, gps_acc, mag_acc, flow_acc, lm_acc, fg_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(steps):
            self.time = i * dt
            self._apply_faults()
            if self.cfg.truth_estimate:
                self._sync_truth_estimate()

            # ---- uncertainty-aware safety decision (from last EKF state) ----
            gps_ok = self._gps_available()
            dec = self.safety.update(self.ekf, self.sensors,
                                     self.vehicle.altitude, dt, gps_ok,
                                     flow_health=self._effective_flow_health)
            self._last_unc_std = dec.uncertainty_std

            # ---- mission desired state using *estimated* position ----
            pos_ref, vel_ref, yaw_ref = self.mission.desired(self.ekf.pos, dt)
            pos_ref, vel_ref = self._safety_override(pos_ref, vel_ref, yaw_ref, dec.mode)

            # ---- controller ----
            control = self.controller.compute(
                self.ekf.pos, self.ekf.vel, self.ekf.rpy,
                pos_ref, vel_ref, yaw_ref,
            )
            # In a true loss-of-trusted-navigation landing, ignore the
            # (unreliable) horizontal position entirely and just level + sink.
            if dec.mode == LAND:
                control = self.controller.landing(
                    self.ekf.rpy, self.ekf.vel,
                    land_speed=self.safety.cfg.land_speed,
                )
            # Once genuinely on the ground, behave like a disarmed-but-on
            # vehicle: hold neutral thrust so a bad vertical estimate cannot
            # command the aircraft back into the air.
            if dec.mode == LANDED and self.vehicle.altitude <= 0.25:
                control = np.array([9.80665, 0.0, 0.0, 0.0])
            self.last_control = control

            # ---- plant ----
            self.vehicle.step(control, dt, wind_ned=self.cfg.wind_ned)

            # ---- sensor housekeeping + sampling based on schedules ----
            self.sensors.step(dt)
            self.last_imu = self.sensors.sample_imu(self.vehicle, dt)

            t_acc += dt
            ah_acc += dt
            baro_acc += dt
            gps_acc += dt
            mag_acc += dt
            flow_acc += dt
            lm_acc += dt
            fg_acc += dt

            # EKF predict at IMU rate
            self.ekf.predict(self.last_imu.accel, self.last_imu.gyro, dt)

            # Independent IMU dead-reckon for velocity-integrity checking.
            # Remove the *estimated* biases so the drift is tolerable.
            acc_corr = self.last_imu.accel - self.ekf.accel_bias
            gyro_corr = self.last_imu.gyro - self.ekf.gyro_bias
            self.flow_integrity.predict(acc_corr, gyro_corr, dt)
            self._dr_pos += self.flow_integrity.vel * dt

            if t_acc >= imu_period - 1e-9:
                t_acc = 0.0
            if ah_acc >= ahrs_period - 1e-9:
                self.ahrs.update(self.last_imu.gyro, self.last_imu.accel,
                                 self.sensors.sample_magnetometer(self.vehicle), dt)
                # Treat the AHRS estimate as a low-rate attitude reference.
                self.ekf.update(self.ahrs.rpy, AHRS)
                ah_acc = 0.0
            if baro_acc >= baro_period - 1e-9:
                z = np.array([self.sensors.sample_baro(self.vehicle)])
                self.ekf.update(z, BARO)
                baro_acc = 0.0
            if mag_acc >= mag_period - 1e-9:
                # (magnetometer already used inside AHRS; no separate update)
                mag_acc = 0.0
            if self.cfg.flow_enabled and flow_acc >= flow_period - 1e-9:
                flow_reading = self.sensors.sample_flow(self.vehicle)
                if flow_reading is not None:
                    self.ekf.update(flow_reading, FLOW)
                    # Health starts from the sensor self-diagnostic (model of
                    # optical-flow confidence / feature quality, noisy).
                    health = float(self.sensors.flow_health)
                    # Independent landmark consistency is the *primary* second
                    # opinion; the factor graph is the principled version of
                    # the same idea and replaces it when enabled.
                    if self.cfg.factorgraph_enabled:
                        health = float(min(health, self._factorgraph_health))
                    if self.cfg.landmark_enabled:
                        health = float(min(health, self._landmark_score))
                    if self.cfg.integrity_monitor_enabled:
                        integrity = self.flow_integrity.evaluate(flow_reading)
                        self._last_flow_mismatch = self.flow_integrity.mismatch
                        health = float(min(health, integrity))
                    self._effective_flow_health = health
                flow_acc = 0.0

            # Independent landmark camera / geometry check.  This runs even
            # when flow is disabled; it only needs the camera + known world.
            if lm_acc >= landmark_period - 1e-9:
                ids, dirs = self.landmark_field.observe(self.vehicle)
                self._landmark_score = self.landmark_consistency.evaluate(
                    ids, dirs, self.ekf.pos, self.ekf.rpy)
                self._landmark_residual = self.landmark_consistency.residual
                lm_acc = 0.0

            # Factor-graph consistency monitor.  Every keyframe we push raw
            # measured data: IMU plus AHRS attitude (for accel->NED), a GNSS
            # fix if available, a flow velocity sample if the sensor is alive,
            # and the camera landmark observation.  The graph then jointly
            # optimises all of these; the post-optimisation flow residual is
            # the independent signal.  If a corrupt flow is consistent with
            # itself but inconsistent with landmarks/IMU/GNSS, that residual
            # will be large and the health will drop.
            if fg_acc >= factorgraph_period - 1e-9:
                if self.cfg.factorgraph_enabled:
                    gps_here = self.sensors.sample_gps(self.vehicle)
                    gps_pos, gps_vel = (gps_here if gps_here is not None
                                        else (None, None))
                    flow_here = None
                    if self.cfg.flow_enabled:
                        flow_here = self.sensors.sample_flow(self.vehicle)
                    ids_here, dirs_here = self.landmark_field.observe(self.vehicle)
                    # Use the *bias-corrected* acceleration in the IMU factor
                    # so the graph's propagation matches the real (estimated)
                    # specific force rather than the biased raw measurement.
                    # The keyframe's initial estimate comes from an *IMU-only
                    # dead-reckon* (not the flow-fed EKF) so the factor graph
                    # is independent of the estimator it is watching.
                    kf = build_keyframe(
                        self._dr_pos, self.flow_integrity.vel,
                        self.last_imu.accel - self.ekf.accel_bias,
                        self.ahrs.rpy,
                        factorgraph_period,
                        flow_here, gps_pos, gps_vel,
                        ids_here, dirs_here,
                    )
                    self.factorgraph.push(kf)
                    fg_info = self.factorgraph.optimize()
                    self._factorgraph_health = fg_info["health"]
                    self._factorgraph_residual = fg_info["flow_residual"]
                fg_acc = 0.0

            if gps_acc >= gps_period - 1e-9:
                gps_reading = self.sensors.sample_gps(self.vehicle)
                if gps_reading is not None:
                    pos_g, vel_g = gps_reading
                    self.ekf.update(np.concatenate([pos_g, vel_g]), GPS)
                    # Re-anchor the independent dead-reckon to the corrected
                    # state whenever a trusted absolute fix is available.
                    self.flow_integrity.reset(self.ekf.vel, self.ekf.rpy)
                    self._dr_pos = self.ekf.pos.copy()
                    self._dr_vel = self.ekf.vel.copy()
                gps_acc = 0.0

            if record:
                out.record(self, pos_ref, vel_ref, yaw_ref)

        return out
