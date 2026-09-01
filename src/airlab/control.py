"""Cascaded flight controller: position -> attitude -> rate.

The estimator only ever sees *noisy sensor data*; the controller therefore
consumes the *estimated* state (state estimate + estimated velocity/attitude),
exactly like a real flight controller.  This gives a clean place to study
how estimation error propagates into tracking performance.
"""

from __future__ import annotations

import numpy as np

from .math_utils import GRAVITY_NED, wrap_pi


class FlightController:
    def __init__(
        self,
        pos_gains: np.ndarray | None = None,
        vel_gains: np.ndarray | None = None,
        att_gain: float = 3.0,
        max_tilt: float = 0.55,
        max_rate: float = 2.5,
    ) -> None:
        self.kp_pos = pos_gains if pos_gains is not None else np.array([1.0, 1.0, 1.5])
        self.kd_vel = vel_gains if vel_gains is not None else np.array([2.5, 2.5, 3.0])
        self.k_att = att_gain
        self.max_tilt = max_tilt
        self.max_rate = max_rate

    def compute(
        self,
        pos_est: np.ndarray,
        vel_est: np.ndarray,
        rpy_est: np.ndarray,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray | None = None,
        yaw_ref: float = 0.0,
    ) -> np.ndarray:
        """Return control ``[thrust_specific, p_rate_cmd, q_rate_cmd, r_rate_cmd]``."""
        if vel_ref is None:
            vel_ref = np.zeros(3)

        g = GRAVITY_NED[2]

        # ---- outer loop: desired acceleration ----
        a_des = self.kp_pos * (pos_ref - pos_est) + self.kd_vel * (vel_ref - vel_est)

        # ---- vertical (down positive, with tilt compensation) ----
        phi_est, theta_est, psi_est = rpy_est
        cos_tilt = float(np.clip(np.cos(phi_est) * np.cos(theta_est), 0.35, 1.0))
        thrust = float(np.clip((g - a_des[2]) / cos_tilt, 1.0, 22.0))

        # ---- attitude reference from horizontal acceleration ----
        # Derived from R @ [0,0,-T] + g:
        #   a_north ~= -T * sin(theta),  a_east ~= +T * sin(phi)
        # so  theta_ref = -a_north/T,  phi_ref = +a_east/T
        phi_ref = a_des[1] / thrust
        theta_ref = -a_des[0] / thrust
        phi_ref = float(np.clip(phi_ref, -self.max_tilt, self.max_tilt))
        theta_ref = float(np.clip(theta_ref, -self.max_tilt, self.max_tilt))

        # ---- inner rate loop ----
        p_cmd = self.k_att * (phi_ref - phi_est)
        q_cmd = self.k_att * (theta_ref - theta_est)
        r_cmd = self.k_att * 0.8 * wrap_pi(yaw_ref - psi_est)
        p_cmd = float(np.clip(p_cmd, -self.max_rate, self.max_rate))
        q_cmd = float(np.clip(q_cmd, -self.max_rate, self.max_rate))
        r_cmd = float(np.clip(r_cmd, -self.max_rate, self.max_rate))

        return np.array([thrust, p_cmd, q_cmd, r_cmd])

    def landing(self, rpy_est: np.ndarray, vel_est: np.ndarray,
                land_speed: float = 0.7) -> np.ndarray:
        """Open-loop landing control that ignores horizontal position error.

        Used after loss of trusted navigation.  It levels the attitude and
        commands a fixed vertical descent rate using the baro/vertical channel;
        horizontal position is deliberately *not* controlled because there is no
        trustworthy reference to control it against.  The aircraft lands at
        whatever horizontal position it happens to be in.
        """
        phi_est, theta_est, _ = rpy_est
        p_cmd = -self.k_att * float(phi_est)
        q_cmd = -self.k_att * float(theta_est)
        r_cmd = 0.0
        p_cmd = float(np.clip(p_cmd, -self.max_rate, self.max_rate))
        q_cmd = float(np.clip(q_cmd, -self.max_rate, self.max_rate))

        v_ref_z = -abs(float(land_speed))
        v_err = float(v_ref_z - vel_est[2])
        g = GRAVITY_NED[2]
        thrust = float(np.clip(g + 2.0 * v_err, 1.0, 22.0))
        return np.array([thrust, p_cmd, q_cmd, r_cmd])
