"""Quadrotor rigid-body dynamics (NED, body z down, thrust along -z body).

This is a *first principles but deliberately compact* model: it is physically
plausible enough for estimator/controller research without pretending to be a
high-fidelity aircraft FDM.  It is intentionally small so the code is readable
and can later be replaced by a Gazebo/JSBSim or a learned surrogate model.

Units
-----
- Position: metres (NED)
- Velocity: m/s
- Attitude: rad (ZYX)
- Angular rates: rad/s
- Thrust control: N/kg (specific force, i.e. acceleration-equivalent)
"""

from __future__ import annotations

import numpy as np

from .math_utils import (GRAVITY_NED, euler_to_R, quat_derivative,
                         quat_from_euler, normalize_quat, euler_from_quat)


def _vec_len3(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(3)


class Quadrotor:
    """Simple rigid-body quadrotor with rate-controlled inner loop.

    Control input (4):
        [0] thrust_specific  : N/kg (== m/s^2 of body- -z thrust)
        [1] roll_rate_cmd    : rad/s
        [2] pitch_rate_cmd   : rad/s
        [3] yaw_rate_cmd     : rad/s
    """

    def __init__(
        self,
        mass: float = 1.3,               # kg
        inertia: np.ndarray | None = None,  # Ixx, Iyy, Izz
        drag_linear: np.ndarray | None = None,  # NED drag coefficient
        rate_tau: float = 0.06,          # rate inner-loop time constant (s)
        max_roll: float = 0.6,           # rad
        max_pitch: float = 0.6,          # rad
        max_thrust: float = 22.0,        # N/kg
        init_pos: np.ndarray | None = None,
        init_vel: np.ndarray | None = None,
        init_rpy: np.ndarray | None = None,
        ground: float = 0.0,             # NED z of ground (down positive)
        ground_friction: float = 5.0,    # 1/s horizontal damping on contact
    ) -> None:
        self.mass = mass
        self.inertia = np.diag(inertia if inertia is not None else [0.012, 0.012, 0.022])
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.drag = _vec_len3(drag_linear if drag_linear is not None else [0.20, 0.20, 0.15])
        self.rate_tau = rate_tau
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.max_thrust = max_thrust
        self.ground = ground
        self.ground_friction = ground_friction

        self.pos = _vec_len3(init_pos if init_pos is not None else [0.0, 0.0, -2.0])
        self.vel = _vec_len3(init_vel if init_vel is not None else [0.0, 0.0, 0.0])
        self.rpy = _vec_len3(init_rpy if init_rpy is not None else [0.0, 0.0, 0.0])
        self.q = quat_from_euler(self.rpy)
        self.omega = np.zeros(3)         # body angular rates
        self.time = 0.0

        # exposure for physics bookkeeping
        self.thrust_specific = 0.0
        self.a_ned = np.zeros(3)

    @property
    def altitude(self) -> float:
        """Height above ground (positive up)."""
        return -self.pos[2] + self.ground

    @property
    def R_nb(self) -> np.ndarray:
        """Body->NED rotation matrix."""
        return euler_to_R(self.rpy)

    def set_state(self, pos: np.ndarray, vel: np.ndarray, rpy: np.ndarray) -> None:
        self.pos = _vec_len3(pos)
        self.vel = _vec_len3(vel)
        self.rpy = _vec_len3(rpy)
        self.q = quat_from_euler(self.rpy)
        self.omega = np.zeros(3)

    def step(
        self,
        control: np.ndarray,
        dt: float,
        wind_ned: np.ndarray | None = None,
        add_disturbance: np.ndarray | None = None,
    ) -> None:
        """Advance the model by ``dt`` seconds.

        ``control`` is the 4-vector described in the class docstring.
        ``wind_ned`` adds a constant/current-like acceleration (e.g. gust).
        ``add_disturbance`` is a raw acceleration added to the state (for
        experiments / deliberate perturbation).
        """
        control = np.asarray(control, dtype=float).reshape(4)
        wind = _vec_len3(wind_ned if wind_ned is not None else [0.0, 0.0, 0.0])
        dist = _vec_len3(add_disturbance if add_disturbance is not None else [0.0, 0.0, 0.0])

        thrust = float(np.clip(control[0], 0.0, self.max_thrust))
        rate_cmds = np.asarray(control[1:4], dtype=float).reshape(3)

        # ---- inner-loop rate response (1st order, bounded) ----
        self.omega += (rate_cmds - self.omega) * (dt / max(self.rate_tau, 1e-4))

        # ---- attitude integration ----
        qdot = quat_derivative(self.q, self.omega)
        self.q = normalize_quat(self.q + dt * qdot)
        self.rpy = euler_from_quat(self.q)

        # ---- forces ----
        R = self.R_nb
        f_thrust_body = np.array([0.0, 0.0, -thrust])
        a_thrust_ned = R @ f_thrust_body

        # linear drag (simple), including apparent wind
        drag_ned = -self.drag * self.vel
        a_ned = a_thrust_ned + GRAVITY_NED + drag_ned + wind + dist

        # simple ground collision: clamp, kill vertical rate, and apply contact
        # friction to the horizontal velocity.  Without friction a landed
        # aircraft keeps the momentum it arrived with (a skid), which makes a
        # "safe" landing metric misleading.
        new_pos = self.pos + self.vel * dt + 0.5 * a_ned * dt * dt
        new_vel = self.vel + a_ned * dt
        if self.ground is not None and new_pos[2] >= self.ground:
            new_pos[2] = self.ground
            if new_vel[2] > 0.0:
                new_vel[2] = 0.0
            # Coulomb-like scrub: exponential horizontal damping while in
            # contact.  A larger coefficient models a firmer brake / skid.
            mu = max(float(self.ground_friction), 0.0)
            decay = float(np.exp(-mu * dt))
            new_vel[0] *= decay
            new_vel[1] *= decay

        self.pos = new_pos
        self.vel = new_vel
        self.a_ned = (self.vel - (self.vel - a_ned * dt)) / dt  # (kept simple)
        self.a_ned = a_ned
        self.thrust_specific = thrust
        self.time += dt
