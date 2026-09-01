"""Uncertainty-aware safety layer: monitor navigation health and decide.

The core idea of this module is that **EKF covariance is a real state**, not a
debug printout.  When GNSS and velocity aiding become unreliable the position
uncertainty grows and the monitor should react — ideally by holding and landing
before the vehicle drifts out of a safe region.

This is deliberately a simple finite-state machine (FSM).  It is conservative
and explainable: an operator can always answer *why* the aircraft changed mode.
Later this can be upgraded to formal reachable-set guards or a learned policy
without changing the rest of the stack.
"""

from __future__ import annotations

import numpy as np

from .math_utils import GRAVITY_NED, euler_to_R, wrap_pi

# Modes
CRUISE = "CRUISE"
HOLD = "HOLD"
LAND = "LAND"
LANDED = "LANDED"


class SafetyConfig:
    def __init__(self) -> None:
        self.enabled = True

        # Horizontal position standard deviation threshold (m).
        self.pos_std_hold = 1.5
        self.pos_std_land = 3.0

        # Velocity-aiding health thresholds (0..1)
        self.flow_health_warn = 0.70
        self.flow_health_fail = 0.40

        # Independent *detector* health (landmark factor-graph / geometry
        # monitor). This is a different kind of signal: it does NOT rely on the
        # EKF's own state, so even when GPS is present a confident detector
        # fault means the velocity-aiding source is *convincing but wrong* and
        # must be refused. Land/hold even with GPS up.
        # 0.55 is deliberately conservative: a healthy FR (fast turn) can dip
        # to ~0.58 for a fraction of a second, while a real fault lands at
        # ~0.04 for seconds.  Wide margin = no false hold; deep margin = still
        # decisive on the genuine fault.
        self.detector_health_warn = 0.55
        self.detector_health_fail = 0.25

        # Time the condition must persist before reacting (s).
        self.grace_flow_warn_s = 1.0
        self.grace_flow_fail_s = 2.0
        self.grace_detector_warn_s = 1.5
        self.grace_detector_fail_s = 1.0

        # Landing behaviour
        self.land_speed = 0.7            # descent rate (m/s)
        self.land_altitude = 1.5         # start final descent below this alt
        self.land_trigger_is_latched = True


class SafetyDecision:
    __slots__ = ("mode", "uncertainty_std", "flow_health", "reason",
                 "gps_available", "landed")

    def __init__(self, mode: str = CRUISE) -> None:
        self.mode = mode
        self.uncertainty_std = 0.0
        self.flow_health = 1.0
        self.reason = "nominal"
        self.gps_available = True
        self.landed = False


class VelocityIntegrityMonitor:
    """Parallel IMU dead-reckon used to *check* a velocity-aiding sensor.

    This is the key to detecting a "convincing but wrong" flow/VIO source.  The
    EKF alone cannot see the problem because the corrupted measurement is
    *consistent with its own state*.  A dead-reckoned velocity that is never
    touched by that sensor provides a second, independent opinion: when the two
    disagree, the integrity score drops.

    This is intentionally a small, interpretable monitor, not a full consistency
    filter.
    """

    def __init__(self, corruption_scale: float = 1.0) -> None:
        self.vel = np.zeros(3)
        self.rpy = np.zeros(3)
        self.initialized = False
        self.mismatch = 0.0
        self.corruption_scale = corruption_scale

    def reset(self, vel: np.ndarray, rpy: np.ndarray) -> None:
        self.vel = np.asarray(vel, dtype=float).reshape(3).copy()
        self.rpy = np.asarray(rpy, dtype=float).reshape(3).copy()
        self.initialized = True
        self.mismatch = 0.0

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        if not self.initialized:
            return
        R = euler_to_R(self.rpy)
        a_ned = R @ accel + GRAVITY_NED
        self.vel += a_ned * dt

        phi, theta, _ = self.rpy
        p, q, r = float(gyro[0]), float(gyro[1]), float(gyro[2])
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        roll_dot = p + sp * st / ct * q + cp * st / ct * r
        pitch_dot = cp * q - sp * r
        yaw_dot = sp / ct * q + cp / ct * r
        self.rpy = self.rpy + np.array([roll_dot, pitch_dot, yaw_dot]) * dt
        for i in range(3):
            self.rpy[i] = wrap_pi(self.rpy[i])

    def evaluate(self, flow_meas: np.ndarray) -> float:
        """Return an integrity score in [0,1] for a velocity measurement."""
        if not self.initialized:
            self.mismatch = 0.0
            return 1.0
        flow_meas = np.asarray(flow_meas, dtype=float).reshape(3)
        # Only the horizontal components are meaningful here (vertical is
        # usually baro/INS fed).
        diff = flow_meas[:2] - self.vel[:2]
        self.mismatch = float(np.linalg.norm(diff))
        score = 1.0 / (1.0 + self.mismatch / self.corruption_scale)
        return float(np.clip(score, 0.0, 1.0))


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else SafetyConfig()
        self.mode = CRUISE
        self.reason = "nominal"
        self._warn_time = 0.0
        self._fail_time = 0.0
        self._det_warn_time = 0.0
        self._det_fail_time = 0.0
        self._unc_smooth = 0.0

    def update(
        self,
        ekf,
        sensors,
        altitude: float,
        dt: float,
        gps_available: bool,
        flow_health: float | None = None,
        detector_health: float | None = None,
    ) -> SafetyDecision:
        # Horizontal uncertainty from the EKF covariance (position 0..1).
        P_h = ekf.P[0, 0] + ekf.P[1, 1]
        unc = float(np.sqrt(max(P_h, 0.0)))
        self._unc_smooth = 0.9 * self._unc_smooth + 0.1 * unc if dt > 0 else unc
        health = float(sensors.flow_health if flow_health is None else flow_health)
        det_health = (float(np.clip(detector_health, 0.0, 1.0))
                      if detector_health is not None else 1.0)

        dec = SafetyDecision()
        dec.uncertainty_std = self._unc_smooth
        dec.flow_health = health
        dec.gps_available = gps_available

        if not self.cfg.enabled:
            dec.mode = CRUISE
            dec.reason = "safety_disabled"
            self.mode = dec.mode
            return dec

        if self.mode == LANDED:
            dec.mode = LANDED
            dec.reason = self.reason
            dec.landed = True
            return dec

        state = self.mode

        # ---- failure conditions ----
        fail_reason = None
        if self._unc_smooth > self.cfg.pos_std_land:
            fail_reason = "position_uncertainty_high"
        elif health < self.cfg.flow_health_fail and not gps_available:
            fail_reason = "velocity_aiding_failed_without_gps"
        elif health < self.cfg.flow_health_fail and gps_available:
            # Flow self-report is low but GPS is up: do NOT land on the
            # self-report alone (it can be noisy).  Only an *independent*
            # detector can safely override a GPS-aided state.
            fail_reason = None

        det_fail_reason = None
        if det_health < self.cfg.detector_health_fail:
            det_fail_reason = "independent_detector_fault"

        warn_reason = None
        if self._unc_smooth > self.cfg.pos_std_hold:
            warn_reason = "position_uncertainty_elevated"
        elif health < self.cfg.flow_health_warn and not gps_available:
            warn_reason = "velocity_aiding_degraded_without_gps"

        det_warn_reason = None
        if det_health < self.cfg.detector_health_warn:
            det_warn_reason = "independent_detector_degraded"

        # ---- counting grace periods ----
        if fail_reason:
            self._fail_time += dt
        else:
            self._fail_time = 0.0
        if warn_reason and not fail_reason:
            self._warn_time += dt
        else:
            self._warn_time = 0.0
        if det_fail_reason:
            self._det_fail_time += dt
        else:
            self._det_fail_time = 0.0
        if det_warn_reason and not det_fail_reason:
            self._det_warn_time += dt
        else:
            self._det_warn_time = 0.0

        # ---- state machine ----
        # An independent-detector fault is a hard signal: land even if GPS is
        # still healthy, because the corrupt velocity-aiding source is driving
        # the controller and can crash the vehicle anyway.
        det_land = self._det_fail_time >= self.cfg.grace_detector_fail_s
        det_hold = self._det_warn_time >= self.cfg.grace_detector_warn_s

        if state == CRUISE:
            # Once uncertainty is already too high, land regardless of health.
            if self._fail_time >= self.cfg.grace_flow_fail_s or \
               self._unc_smooth > self.cfg.pos_std_land or det_land:
                state = LAND
                self.reason = det_fail_reason or fail_reason or "uncertainty_critical"
            elif self._warn_time >= self.cfg.grace_flow_warn_s or \
                 (health < self.cfg.flow_health_warn and not gps_available) or det_hold:
                state = HOLD
                self.reason = det_warn_reason or warn_reason or "degraded_navigation"
        elif state == HOLD:
            # A hold buys time but should not become an indefinite drift.
            if self._fail_time >= self.cfg.grace_flow_fail_s or \
               self._unc_smooth > self.cfg.pos_std_land or det_land:
                state = LAND
                self.reason = det_fail_reason or fail_reason or "hold_uncertainty_critical"
            elif not det_hold and gps_available and health >= self.cfg.flow_health_warn \
                 and self._unc_smooth < self.cfg.pos_std_hold and det_health >= self.cfg.detector_health_warn:
                # recovered
                state = CRUISE
                self.reason = "recovered"
            elif altitude < self.cfg.land_altitude:
                state = LAND
                self.reason = "hold_low_altitude"

        # once latched, stay LAND until touchdown
        if state == LAND and self.cfg.land_trigger_is_latched:
            state = LAND
            if altitude <= 0.15:
                state = LANDED
                self.reason = "safely_landed"

        self.mode = state
        dec.mode = state
        dec.reason = self.reason
        dec.landed = (state == LANDED)
        return dec

    @property
    def landing(self) -> bool:
        return self.mode in (LAND, LANDED)
