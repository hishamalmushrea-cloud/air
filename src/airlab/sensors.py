"""Configurable sensor model for the simulated UAV.

The sensor suite deliberately keeps rates realistic relative to the flight
controller (IMU fast, GNSS slow) so that the estimator has to *do* something
meaningful rather than being handed clean truth.
"""

from __future__ import annotations

import numpy as np

from .math_utils import GRAVITY_NED, euler_to_R

# Local magnetic field in NED (north, east, down).  A small declination keeps
# things honest without dragging in a full WMM model.
MAGNETIC_FIELD_NED = np.array([24.0e-6, 0.0, 38.0e-6])


class SensorConfig:
    def __init__(self) -> None:
        self.imu_hz = 100.0
        self.mag_hz = 50.0
        self.gps_hz = 10.0
        self.baro_hz = 20.0
        self.ahrs_hz = 50.0
        self.flow_hz = 25.0

        # magnitude of white noise (standard deviations)
        self.gyro_sigma = 0.003         # rad/s
        self.accel_sigma = 0.10         # m/s^2
        self.mag_sigma = 0.5e-6         # Tesla
        self.gps_pos_sigma = 0.35       # m
        self.gps_vel_sigma = 0.10       # m/s
        self.baro_sigma = 0.25          # m
        self.ahrs_sigma = 0.01          # rad
        self.flow_sigma = 0.05          # m/s (horizontal velocity aiding)

        # slowly-varying biases
        self.gyro_bias_init = np.array([0.012, -0.010, 0.004])   # rad/s
        self.accel_bias_init = np.array([0.06, 0.04, -0.05])     # m/s^2
        self.gyro_bias_walk = 1e-4      # rad/s^2->rad/s per s (approx)
        self.accel_bias_walk = 5e-4
        self.baro_drift = 0.01          # m/s (pressure-like drift)
        self.flow_bias_init = np.array([0.02, -0.015, 0.0])   # m/s
        self.flow_bias_walk = 0.002
        # Transient additive bias (m/s) applied only while a fault window is
        # active; reset to 0.0 outside it.  Distinct from flow_bias_ramp, which
        # accumulates permanently.  Enables a fault that is fully contained in a
        # time window (used to test a fault hidden inside a sparse-FG outage).
        self.flow_bias_shift = 0.0

        # faults (0 => none).  These mutate over time.
        self.gps_dropout = 0.0          # seconds remaining of GPS outage
        self.gps_noise_scale = 1.0
        self.imu_bias_ramp = 0.0        # extra rad/(s*s) on z-gyro
        self.flow_dropout = 0.0         # seconds remaining of velocity-aiding outage
        self.flow_scale = 1.0           # scale error (1.0 = healthy)
        self.flow_bias_ramp = 0.0       # m/s^2 extra bias on velocity aiding
        self.flow_health_noise = 0.04   # noise on the reported health signal


class IMUReading:
    __slots__ = ("accel", "gyro")

    def __init__(self, accel: np.ndarray, gyro: np.ndarray) -> None:
        self.accel = accel
        self.gyro = gyro


class SensorSuite:
    def __init__(self, config: SensorConfig | None = None) -> None:
        self.cfg = config if config is not None else SensorConfig()
        self.rng = np.random.default_rng(12345)
        self.t = 0.0

        self._gyro_bias = self.cfg.gyro_bias_init.copy()
        self._accel_bias = self.cfg.accel_bias_init.copy()
        self._baro_err = 0.0
        self._flow_bias = self.cfg.flow_bias_init.copy()
        self._flow_bias_accum = np.zeros(3)
        self._flow_health = 1.0

    def _miss(self, dt: float) -> bool:
        # Poisson-ish sampling: sample a measurement each fixed interval
        # handled by the caller; this helper is just for clarity.
        return False

    def step(self, dt: float) -> None:
        self.t += dt
        self._gyro_bias += self.cfg.gyro_bias_walk * dt * self.rng.standard_normal(3)
        self._accel_bias += self.cfg.accel_bias_walk * dt * self.rng.standard_normal(3)
        self._gyro_bias[2] += self.cfg.imu_bias_ramp * dt
        self._baro_err += self.cfg.baro_drift * dt + 0.0015 * self.rng.standard_normal()
        self._flow_bias += self.cfg.flow_bias_walk * dt * self.rng.standard_normal(3)
        self._flow_bias_accum += self.cfg.flow_bias_ramp * dt * self.rng.standard_normal(3)
        self._flow_bias += self.cfg.flow_bias_ramp * dt

        if self.cfg.flow_dropout > 0.0:
            self.cfg.flow_dropout = max(0.0, self.cfg.flow_dropout - dt)

        if self.cfg.gps_dropout > 0.0:
            self.cfg.gps_dropout = max(0.0, self.cfg.gps_dropout - dt)

        # Report a noisy "self-diagnostic" *availability* health signal.  We
        # deliberately do NOT penalise scale/bias here: a real VIO/optical-flow
        # module can report "features tracked, high confidence" even when its
        # scale or bias is wrong.  The *accuracy* faults are the job of the
        # structurally independent landmark detector (landmarks.py).
        dropout_penalty = 0.35 if self.cfg.flow_dropout > 0.0 else 0.0
        raw = max(0.0, 1.0 - dropout_penalty)
        self._flow_health = float(np.clip(raw + self.cfg.flow_health_noise * self.rng.standard_normal(), 0.0, 1.0))
        # (a small excursion is fine; the estimator/safety still treat it as a
        #  noisy confidence report, not ground truth)

    def sample_imu(self, veh, dt: float) -> IMUReading:
        """Accelerometer and gyro at IMU rate."""
        f_meas = veh.R_nb.T @ (veh.a_ned - GRAVITY_NED) + self._accel_bias
        accel_noise = self.cfg.accel_sigma * self.rng.standard_normal(3)
        accel = f_meas + accel_noise

        gyro = veh.omega + self._gyro_bias
        gyro_noise = self.cfg.gyro_sigma * self.rng.standard_normal(3)
        gyro = gyro + gyro_noise
        return IMUReading(accel, gyro)

    def sample_magnetometer(self, veh) -> np.ndarray:
        m_body = veh.R_nb.T @ MAGNETIC_FIELD_NED
        return m_body + self.cfg.mag_sigma * self.rng.standard_normal(3)

    @property
    def flow_health(self) -> float:
        """Reported self-diagnostic health of the velocity-aiding source."""
        return self._flow_health

    def sample_flow(self, veh) -> np.ndarray | None:
        """Visual-odometry / optical-flow velocity estimate (NED).

        Returns ``None`` during a dropout, so the estimator correctly gets no
        update.  An active scale error or a ramping bias corrupts the measured
        velocity without changing the reported health, which is the kind of
        "not obviously dead but wrong" failure that matters most.
        """
        if self.cfg.flow_dropout > 0.0:
            return None
        true_vel = veh.vel
        scaled = true_vel * self.cfg.flow_scale
        return (scaled + self._flow_bias + self.cfg.flow_bias_shift
                + self.cfg.flow_sigma * self.rng.standard_normal(3))

    def sample_gps(self, veh) -> tuple[np.ndarray, np.ndarray] | None:
        if self.cfg.gps_dropout > 0.0:
            return None
        pos = veh.pos + self.cfg.gps_pos_sigma * self.cfg.gps_noise_scale * self.rng.standard_normal(3)
        vel = veh.vel + self.cfg.gps_vel_sigma * self.cfg.gps_noise_scale * self.rng.standard_normal(3)
        return pos, vel

    def sample_baro(self, veh) -> float:
        return veh.altitude + self._baro_err + self.cfg.baro_sigma * self.rng.standard_normal()

    def sample_ahrs(self, veh) -> np.ndarray:
        """Synthetic AHRS reference: true attitude plus a modest sensing error.

        In a real system this would come from a gyro+accel/mag complementary
        filter.  We inject a *realistic* error model (plus a tiny lag-inducing
        bias) so the nav EKF treats it as uncertain, not perfect.
        """
        return veh.rpy + self.cfg.ahrs_sigma * self.rng.standard_normal(3)
