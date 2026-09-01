"""3D rotation / quaternion utilities (NED, ZYX aerospace convention).

Convention
----------
- Frame: NED (North, East, Down). ``+z`` points **down**.
- Body frame: x forward, y right, z down (aircraft convention).
- Euler order used for the rotation matrix is ZYX: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
- ``R`` maps a vector expressed in the *body* frame to the *NED* frame.
"""

from __future__ import annotations

import numpy as np

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])


def euler_to_R(rpy: np.ndarray) -> np.ndarray:
    """Return the body->NED rotation matrix from roll/pitch/yaw."""
    phi, theta, psi = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cp, -sp],
                   [0.0, sp, cp]])
    Ry = np.array([[ct, 0.0, st],
                   [0.0, 1.0, 0.0],
                   [-st, 0.0, ct]])
    Rz = np.array([[cpsi, -spsi, 0.0],
                   [spsi, cpsi, 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def quat_from_euler(rpy: np.ndarray) -> np.ndarray:
    """Euler ZYX -> unit quaternion [w, x, y, z]."""
    phi, theta, psi = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(phi / 2.0), np.sin(phi / 2.0)
    cp, sp = np.cos(theta / 2.0), np.sin(theta / 2.0)
    cy, sy = np.cos(psi / 2.0), np.sin(psi / 2.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return normalize_quat(np.array([w, x, y, z]))


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> Euler ZYX (roll, pitch, yaw)."""
    q = normalize_quat(q)
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    # Clamp to avoid sqrt of a slightly negative number at |pitch| ~ 90 deg.
    sin_theta = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, theta, yaw], dtype=float)


def R_from_quat(q: np.ndarray) -> np.ndarray:
    """Body->NED rotation from a unit quaternion."""
    q = normalize_quat(q)
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product ``a * b`` (vectors are [w,x,y,z])."""
    wa, xa, ya, za = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    wb, xb, yb, zb = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ], dtype=float)


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Quaternion derivative from body angular rates (rad/s)."""
    omega = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion (avoid division by zero)."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def wrap_pi(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi
