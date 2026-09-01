"""A tiny complementary-filter AHRS (gyro + accelerometer + magnetometer).

This is intentionally simple but *not* a toy: it demonstrates how to combine a
high-rate gyroscope with low-frequency heading/leveling measurements.  The
output is a noisy, slightly-biased attitude reference used by the navigation
EKF — mirroring how many real INS stacks separate an AHRS from a filter.
"""

from __future__ import annotations

import numpy as np

from .math_utils import wrap_pi


def accel_to_level(f_acc: np.ndarray) -> tuple[float, float]:
    """Roll/pitch from a specific-force (accelerometer) measurement.

    Convention: body z points down, so a perfect level accelerometer reads
    ``[0, 0, -g]``.  With thrust along ``-z`` in the body frame and the
    rotation matrix ``T = euler_to_R(rpy)``, the measured specific force is
    ``T.T @ (a_ned - g_ned)``.  This yields:
        fy ~= +g * sin(phi),   fx ~= -g * sin(theta)
    hence the sign conventions below.
    """
    fx, fy, fz = float(f_acc[0]), float(f_acc[1]), float(f_acc[2])
    roll = np.arctan2(fy, -fz)
    pitch = np.arctan2(-fx, np.sqrt(fy * fy + fz * fz))
    return roll, pitch


def mag_to_yaw(m_body: np.ndarray) -> float:
    """Yaw from body-frame magnetometer (north/down field model)."""
    mx, my = float(m_body[0]), float(m_body[1])
    return wrap_pi(np.arctan2(-my, mx))


class ComplementaryAHRS:
    def __init__(
        self,
        rpy: np.ndarray | None = None,
        accel_gain: float = 0.02,
        mag_gain: float = 0.01,
    ) -> None:
        self.rpy = np.zeros(3) if rpy is None else np.asarray(rpy, dtype=float).copy()
        self.accel_gain = accel_gain
        self.mag_gain = mag_gain

    def predict(self, gyro: np.ndarray, dt: float) -> None:
        """Advance attitude using gyro (small-angle Euler update)."""
        # A compact approximation valid away from singularities.
        p, q, r = float(gyro[0]), float(gyro[1]), float(gyro[2])
        phi, theta, _ = self.rpy
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        roll_dot = p + sp * st / ct * q + cp * st / ct * r
        pitch_dot = cp * q - sp * r
        yaw_dot = sp / ct * q + cp / ct * r
        self.rpy = self.rpy + np.array([roll_dot, pitch_dot, yaw_dot]) * dt
        self.rpy[0] = wrap_pi(self.rpy[0])
        self.rpy[1] = wrap_pi(self.rpy[1])
        self.rpy[2] = wrap_pi(self.rpy[2])

    def correct_accel(self, f_acc: np.ndarray) -> None:
        roll, pitch = accel_to_level(f_acc)
        self.rpy[0] = wrap_pi(self.rpy[0] + self.accel_gain * wrap_pi(roll - self.rpy[0]))
        self.rpy[1] = wrap_pi(self.rpy[1] + self.accel_gain * wrap_pi(pitch - self.rpy[1]))

    def correct_mag(self, m_body: np.ndarray) -> None:
        yaw = mag_to_yaw(m_body)
        self.rpy[2] = wrap_pi(self.rpy[2] + self.mag_gain * wrap_pi(yaw - self.rpy[2]))

    def update(self, gyro: np.ndarray, f_acc: np.ndarray, m_body: np.ndarray, dt: float) -> None:
        self.predict(gyro, dt)
        self.correct_accel(f_acc)
        self.correct_mag(m_body)

    def reset(self, rpy: np.ndarray) -> None:
        self.rpy = np.asarray(rpy, dtype=float).copy()
