"""Landmark-based consistency monitor — a *structurally independent* sanity
check for the velocity-aiding source.

The problem (from research-brief-02): a corrupt optical-flow / VIO source can be
"convincing but wrong".  The EKF trusts it and drifts, and because the wrong
measurement is *self-consistent* with the filter, the EKF covariance does not
grow — so an uncertainty-based safety layer never notices.

Solution: keep a small set of fixed, known landmarks in the world and a simple
camera.  The camera measures the *true* direction to each landmark.  The EKF
state predicts what that direction *should* be.  If the two disagree, the EKF
position/attitude has drifted in a way that the flow/VIO source hid from us.
This gives us a second, structurally independent opinion without needing a
second physical sensor.

This is intentionally simplified (known landmarks, no data association / SLAM
complexity) to establish the *concept* before we grow it into a full
multi-hypothesis / factor-graph backend.
"""

from __future__ import annotations

import numpy as np

from .math_utils import euler_to_R, wrap_pi


class LandmarkField:
    """A fixed set of world-frame landmarks observed by a body-mounted camera."""

    def __init__(
        self,
        positions_ned: np.ndarray | None = None,
        range_m: float = 40.0,
        fov_cos: float = 0.15,          # camera z-axis (down) field-of-view limit
        bearing_noise: float = 0.01,    # rad
        max_landmarks: int = 6,
        seed: int = 3,
    ) -> None:
        if positions_ned is None:
            # A "synthetic world" covering the default mission area densely
            # enough that the camera almost always sees >=2 landmarks.  The
            # down component is the negative of the intuitive height, so these
            # are near/below the vehicle (camera looks mostly downward).
            positions_ned = np.array([
                [12.0, 10.0, -0.5], [-10.0, 12.0, -0.5],
                [12.0, -12.0, -0.5], [-12.0, -14.0, -0.5],
                [0.0, 18.0, -2.0], [18.0, 0.0, -2.0],
                [0.0, -10.0, -1.0], [-12.0, 4.0, -1.0],
                [14.0, 4.0, -1.0], [4.0, 16.0, -1.5],
                [-4.0, -16.0, -1.5], [16.0, -16.0, -1.5],
                [-16.0, 16.0, -1.5], [8.0, 12.0, -1.0],
                [-8.0, -8.0, -1.0], [20.0, 8.0, -2.0],
            ])
        self.positions = np.asarray(positions_ned, dtype=float).reshape(-1, 3)
        self.range_m = range_m
        self.fov_cos = fov_cos
        self.bearing_noise = bearing_noise
        self.max_landmarks = max_landmarks
        self.rng = np.random.default_rng(seed)

    def observe(self, veh) -> tuple[np.ndarray, np.ndarray]:
        """Return (landmark_ids, unit_bearing_vectors_body) for visible landmarks.

        The bearing vectors are in the *body* frame, as a real camera would
        report them.  They are produced from the true vehicle pose (the physics
        world), not from the EKF estimate, which is exactly why a drifted EKF
        will "disagree" with them.
        """
        ids = []
        dirs = []
        R_nb = veh.R_nb                                   # body -> NED
        for i, lm in enumerate(self.positions):
            r = lm - veh.pos                              # NED vector to landmark
            dist = float(np.linalg.norm(r))
            if dist > self.range_m:
                continue
            if dist < 1e-3:
                continue
            dir_ned = r / dist
            dir_body = R_nb.T @ dir_ned                   # NED -> body
            # camera looks along body -z (straight down/world-down-ish); keep
            # landmarks that are in the front/underside hemisphere.
            if dir_body[2] < -self.fov_cos:
                continue
            if len(ids) >= self.max_landmarks:
                # keep nearest ones (the loop order is arbitrary; just stop)
                break
            # raw sensor noise (the camera is real, so its measurement is
            # slightly noisy, but it is *not* driven by the flow/VIO state).
            dir_noisy = dir_body + self.bearing_noise * self.rng.standard_normal(3)
            dir_noisy /= np.linalg.norm(dir_noisy)
            ids.append(i)
            dirs.append(dir_noisy)
        if not ids:
            return np.array([], dtype=int), np.zeros((0, 3))
        return np.array(ids, dtype=int), np.array(dirs, dtype=float)


def _body_bearing(pos_ned: np.ndarray, rpy: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Predicted body-frame unit vector to a landmark from an estimated pose."""
    r = lm - pos_ned
    n = np.linalg.norm(r)
    if n < 1e-6:
        return np.zeros(3)
    dir_ned = r / n
    R_nb = euler_to_R(rpy)
    dir_body = R_nb.T @ dir_ned
    n2 = np.linalg.norm(dir_body)
    if n2 < 1e-6:
        return np.zeros(3)
    return dir_body / n2


class LandmarkConsistency:
    """Compare *inter-landmark angles* against the EKF-predicted position.

    Using the angle between two visible landmarks (instead of their absolute
    body direction) makes the check **invariant to attitude error**.  A camera
    that is pointed slightly wrong still measures the same relative geometry;
    only a wrong *position* changes the angles between landmarks.  This is
    exactly the failure mode we care about: a corrupt flow/VIO source causes a
    position drift that a single-stream EKF cannot see.

    Produces a health score in [0,1] (1 == consistent, 0 == badly inconsistent)
    plus the mean angular residual in radians.
    """

    def __init__(
        self,
        field: LandmarkField,
        residual_scale: float = 0.12,   # rad at which score ~= 0.5
        smooth: float = 0.8,
    ) -> None:
        self.field = field
        self.residual_scale = residual_scale
        self.smooth = smooth
        self.score = 1.0
        self.residual = 0.0

    def _dir(self, pos: np.ndarray, lm: np.ndarray) -> np.ndarray:
        r = lm - pos
        n = np.linalg.norm(r)
        return r / n if n > 1e-6 else np.zeros(3)

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.arccos(dot))

    def evaluate(
        self,
        observed_ids: np.ndarray,
        observed_dirs: np.ndarray,
        est_pos: np.ndarray,
        est_rpy: np.ndarray,
    ) -> float:
        """Update and return the current consistency score."""
        n = len(observed_ids)
        if n < 2:
            # Need at least two landmarks to estimate an attitude-invariant
            # angular signature.  With fewer, retain the last score (unknown
            # rather than bad).
            return float(self.score)

        residuals = []
        for i in range(n):
            for j in range(i + 1, n):
                lm_i = self.field.positions[int(observed_ids[i])]
                lm_j = self.field.positions[int(observed_ids[j])]
                pred_angle = self._angle(self._dir(est_pos, lm_i),
                                         self._dir(est_pos, lm_j))
                obs_angle = self._angle(observed_dirs[i], observed_dirs[j])
                residuals.append(abs(pred_angle - obs_angle))

        resid = float(np.mean(residuals))
        raw = 1.0 / (1.0 + resid / self.residual_scale)
        raw = float(np.clip(raw, 0.0, 1.0))
        self.residual = resid
        self.score = self.smooth * self.score + (1.0 - self.smooth) * raw
        return float(self.score)
