"""Navigation EKF: loosely-coupled INS/GPS/baro/AHRS fusion.

State (15):
    [pn, pe, pd, vn, ve, vd, phi, theta, psi, bgx, bgy, bgz, bax, bay, baz]

The prediction step uses the IMU (gyro + accelerometer) and the update steps
ingest GNSS position/velocity, barometric altitude and an AHRS attitude
reference.  The Jacobians are computed numerically, which makes the filter
much easier to extend (and to reason about) than hand-derived sparse Jacobians
for a first reference implementation.
"""

from __future__ import annotations

import numpy as np

from .math_utils import GRAVITY_NED, euler_to_R, wrap_pi
from .ahrs import accel_to_level, mag_to_yaw

# measurement types
GPS = "gps"
BARO = "baro"
AHRS = "ahrs"
FLOW = "flow"

S_IDX = slice(0, 3)          # position
V_IDX = slice(3, 6)          # velocity
R_IDX = slice(6, 9)          # euler
BG_IDX = slice(9, 12)        # gyro bias
BA_IDX = slice(12, 15)       # accel bias


def _wrap_state(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    for i in range(6, 9):
        x[i] = wrap_pi(x[i])
    return x


class NavEKF:
    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rpy: np.ndarray,
        gyro_bias: np.ndarray | None = None,
        accel_bias: np.ndarray | None = None,
        q_scale: float = 1.0,
        r_scale: float = 1.0,
        backend: str = "numeric",
        **kwargs,
    ) -> None:
        self.x = np.zeros(15)
        self.x[S_IDX] = pos
        self.x[V_IDX] = vel
        self.x[R_IDX] = rpy
        self.x[BG_IDX] = gyro_bias if gyro_bias is not None else np.zeros(3)
        self.x[BA_IDX] = accel_bias if accel_bias is not None else np.zeros(3)

        # Initial uncertainty
        self.P = np.diag([
            1.0, 1.0, 1.0,          # position
            0.5, 0.5, 0.5,          # velocity
            (0.05) ** 2, (0.05) ** 2, (0.05) ** 2,  # attitude
            (0.02) ** 2, (0.02) ** 2, (0.02) ** 2,  # gyro bias
            (0.05) ** 2, (0.05) ** 2, (0.05) ** 2,  # accel bias
        ]) * q_scale

        # Process noise (per-second units, scaled by dt in predict)
        self.q = np.diag([
            0.01, 0.01, 0.01,       # pos
            0.10, 0.10, 0.10,       # vel
            0.005, 0.005, 0.005,    # attitude
            1e-5, 1e-5, 1e-5,       # gyro bias walk
            1e-4, 1e-4, 1e-4,       # accel bias walk
        ])

        # Measurement noise
        self.r_gps = np.diag([0.35 ** 2, 0.35 ** 2, 0.35 ** 2,
                              0.10 ** 2, 0.10 ** 2, 0.10 ** 2]) * r_scale
        self.r_baro = np.array([[0.25 ** 2]]) * r_scale
        self.r_ahrs = np.diag([0.02 ** 2, 0.02 ** 2, 0.03 ** 2]) * r_scale
        self.r_flow = np.diag([0.05 ** 2, 0.05 ** 2, 0.05 ** 2]) * r_scale

        self.last_dt = 0.0
        self._reset_reference_altitude()
        self._fuse_dtype = float

    def _reset_reference_altitude(self) -> None:
        self.baro_ref = 0.0

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _dynamics(self, x: np.ndarray, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> np.ndarray:
        x = _wrap_state(x)
        p, v, rpy = x[S_IDX].copy(), x[V_IDX].copy(), x[R_IDX].copy()
        bg, ba = x[BG_IDX].copy(), x[BA_IDX].copy()

        gyro = gyro_meas - bg
        accel = accel_meas - ba

        R = euler_to_R(rpy)
        a_ned = R @ accel + GRAVITY_NED
        v_new = v + a_ned * dt
        p_new = p + v_new * dt

        # Euler rate update (compact, valid away from singularities)
        phi, theta, _ = rpy
        p_rate, q_rate, r_rate = float(gyro[0]), float(gyro[1]), float(gyro[2])
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        roll_dot = p_rate + sp * st / ct * q_rate + cp * st / ct * r_rate
        pitch_dot = cp * q_rate - sp * r_rate
        yaw_dot = sp / ct * q_rate + cp / ct * r_rate
        rpy_new = rpy + np.array([roll_dot, pitch_dot, yaw_dot]) * dt

        x_new = x.copy()
        x_new[S_IDX] = p_new
        x_new[V_IDX] = v_new
        x_new[R_IDX] = rpy_new
        return _wrap_state(x_new)

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> None:
        self.last_dt = dt
        f = lambda x: self._dynamics(x, accel_meas, gyro_meas, dt)
        F = self._jac(f, self.x)

        self.x = _wrap_state(f(self.x))
        self.P = F @ self.P @ F.T + self.q * dt

    # ------------------------------------------------------------------ #
    # Measurement update
    # ------------------------------------------------------------------ #
    def _meas_model(self, x: np.ndarray, kind: str) -> np.ndarray:
        if kind == GPS:
            return np.concatenate([x[S_IDX], x[V_IDX]])
        if kind == BARO:
            return np.array([-x[2] + self.baro_ref])
        if kind == AHRS:
            return x[R_IDX].copy()
        if kind == FLOW:
            return x[V_IDX].copy()
        raise ValueError(f"unknown measurement kind {kind}")

    def update(self, z: np.ndarray, kind: str, r_cov: np.ndarray | None = None) -> None:
        h = lambda x: self._meas_model(x, kind)
        H = self._jac(h, self.x)
        zp = h(self.x)
        y = np.asarray(z, dtype=float) - zp
        if kind == AHRS:
            y[0] = wrap_pi(y[0])
            y[1] = wrap_pi(y[1])
            y[2] = wrap_pi(y[2])

        R = self._noise_cov(kind) if r_cov is None else r_cov
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = _wrap_state(self.x + K @ y)
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def _noise_cov(self, kind: str) -> np.ndarray:
        if kind == GPS:
            return self.r_gps
        if kind == BARO:
            return self.r_baro
        if kind == AHRS:
            return self.r_ahrs
        if kind == FLOW:
            return self.r_flow
        raise ValueError(kind)

    # ------------------------------------------------------------------ #
    # Numeric Jacobian (central difference)
    # ------------------------------------------------------------------ #
    def _jac(self, f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        f0 = f(x)
        n = len(x)
        m = len(f0)
        J = np.zeros((m, n))
        for j in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = _wrap_state(f(xp)) if len(f(xp)) == 15 else f(xp)
            fm = _wrap_state(f(xm)) if len(f(xm)) == 15 else f(xm)
            J[:, j] = (fp - fm) / (2.0 * eps)
        return J

    # convenience accessors
    @property
    def pos(self) -> np.ndarray:
        return self.x[S_IDX].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[V_IDX].copy()

    @property
    def rpy(self) -> np.ndarray:
        return self.x[R_IDX].copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.x[BG_IDX].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[BA_IDX].copy()
