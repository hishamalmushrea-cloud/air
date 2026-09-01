"""Sliding-window nonlinear factor graph for navigation-source consistency.

This is the *principled* replacement for the heuristic landmark score from
`landmarks.py`.  Instead of a single hand-crafted score, it builds a small
least-squares graph over a sliding window of keyframes and jointly optimises:

    IMU motion factors    (velocity/position propagation from accelerometer)
    GNSS absolute factors (when available)
    flow/VIO velocity factors   (the *suspect* source)
    landmark angle factors (the *independent* world-geometry source)

After optimisation, each factor has a residual.  A corrupt flow/VIO source is
"convincing but wrong": it is consistent with the filter that trusts it, but it
is *not* consistent with the landmark/IMU/GNSS factors that constrain the true
geometry.  Therefore its post-optimisation residual is large.  That residual,
mapped to [0,1], is the multi-hypothesis consistency signal we feed the safety
layer.

Optimisation uses Gauss-Newton with a Cauchy robust loss, which makes the graph
itself reject a single bad factor (outlier rejection) instead of just reporting
it.  This is the textbook "factor graph + robust kernel" approach, implemented
from scratch with only numpy so the whole idea is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .math_utils import euler_to_R, GRAVITY_NED


class ImuPreintegrator:
    """Preintegrate the IMU specific force over a sliding relative interval.

    The naive approach in the first version seeded the factor-graph state with
    a *full-pose dead-reckon that had already drifted since the last absolute
    reference*.  That made the healthy baseline high.  A proper IMU
    preintegrator instead produces a **relative** velocity/position change over
    a short window and is re-anchored to the latest trusted state, so the graph
    starts much closer to the true geometry and the flow residual is then
    dominated by genuine inconsistency rather than dead-reckon drift.

    This is a minimal, transparent implementation: bias-corrected acceleration
    rotated by the AHRS attitude, integrated once per IMU step.
    """

    def __init__(self) -> None:
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.n_steps = 0
        self.start_vel = None
        self.start_pos = None
        self.total_dt = 0.0

    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.start_pos = np.asarray(pos, dtype=float).reshape(3).copy()
        self.start_vel = np.asarray(vel, dtype=float).reshape(3).copy()
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.n_steps = 0
        self.total_dt = 0.0

    def feed(self, acc_ned: np.ndarray, dt: float, rpy: np.ndarray) -> np.ndarray:
        """Feed one bias-corrected accelerometer sample (rotated to NED).

        Returns the current dead-reckoned velocity estimate (absolute, based
        on ``start_vel`` plus accumulated delta-v).  The dead-reckoned position
        is available via ``predict_pos``.
        """
        self.delta_v += np.asarray(acc_ned, dtype=float).reshape(3) * dt
        self.delta_p += self.delta_v * dt
        self.n_steps += 1
        self.total_dt += dt
        return self.start_vel + self.delta_v

    def predict_pos(self) -> np.ndarray:
        return self.start_pos + self.start_vel * self.total_dt + self.delta_p

    def predict_vel(self) -> np.ndarray:
        return self.start_vel + self.delta_v


@dataclass
class Keyframe:
    p0: np.ndarray             # initial position estimate (NED)
    v0: np.ndarray             # initial velocity estimate (NED)
    acc_ned: np.ndarray        # mean body-frame-NED specific force (m/s^2)
    dt_to_next: float          # seconds to next keyframe (0 for last)
    flow_meas: Optional[np.ndarray] = None
    gps_pos: Optional[np.ndarray] = None
    gps_vel: Optional[np.ndarray] = None
    lm_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    lm_dirs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


class SlidingFactorGraph:
    def __init__(
        self,
        landmark_positions: np.ndarray,
        window: int = 6,
        dt_keyframe: float = 0.5,
        cauchy_scale: float = 0.10,
        residual_scale: float = 0.20,     # m/s mapping residual -> health
        max_iter: int = 12,
        min_keyframes: int = 4,
        baseline_residual: float = 0.0,   # subtracted before health mapping
        bias_reg: float = 1.0,            # weight on bias-vs-zero residual
    ) -> None:
        self.landmarks = np.asarray(landmark_positions, dtype=float).reshape(-1, 3)
        self.window = window
        self.dt_keyframe = dt_keyframe
        self.cauchy_scale = cauchy_scale
        self.residual_scale = residual_scale
        self.max_iter = max_iter
        self.min_keyframes = min_keyframes
        self.baseline_residual = baseline_residual
        self.bias_reg = bias_reg

        self.keyframes: list[Keyframe] = []
        self.flow_residual = 0.0
        self.lm_residual = 0.0
        self.total_residual = 0.0
        self.health = 1.0
        self._last_opt_time = -1.0
        self._last_flow_residuals = np.zeros(0)
        self.aux_positions = {}   # keep for optional diagnostics
        self.aux_velocities = {}
        # Shared accel-bias estimate (NED), estimated inside the graph.  It is
        # deliberately NOT borrowed from the flow-fed EKF, so a corrupt flow
        # cannot contaminate the "independent" IMU motion model.
        self.bias = np.zeros(3)

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #
    def push(self, kf: Keyframe) -> None:
        self.keyframes.append(kf)
        if len(self.keyframes) > self.window:
            self.keyframes.pop(0)

    def clear(self) -> None:
        self.keyframes.clear()

    def _n_pose_vars(self) -> int:
        return len(self.keyframes)

    def _n_vars(self) -> int:
        # 6 vars (p,v) per keyframe + shared accel bias (3)
        return 6 * len(self.keyframes) + 3

    def _bias_start(self) -> int:
        return 6 * len(self.keyframes)

    def _x0(self) -> np.ndarray:
        x = []
        for k in self.keyframes:
            x.extend([k.p0[0], k.p0[1], k.p0[2], k.v0[0], k.v0[1], k.v0[2]])
        x.extend([self.bias[0], self.bias[1], self.bias[2]])
        return np.asarray(x, dtype=float)

    def _block_from_index(self, k: int) -> np.ndarray:
        return np.asarray(self.keyframes[k], dtype=object)

    def _pos(self, X: np.ndarray, k: int) -> np.ndarray:
        return X[6 * k:6 * k + 3]

    def _vel(self, X: np.ndarray, k: int) -> np.ndarray:
        return X[6 * k + 3:6 * k + 6]

    def _bias(self, X: np.ndarray) -> np.ndarray:
        return X[self._bias_start():self._bias_start() + 3]

    # ------------------------------------------------------------------ #
    # residual model.  Returns (r_vec, split, names).
    # split is a list of slices per factor contribution.
    # ------------------------------------------------------------------ #
    def _residuals(self, X: np.ndarray):
        n = self._n_pose_vars()
        r = []
        names = []
        slices = []
        flow_slice = {}

        # ---- IMU factors (between consecutive keyframes) ----
        # We subtract the graph's *own* estimated bias, so a corrupt flow cannot
        # contaminate the IMU model through the EKF's bias estimate.
        bias_est = self._bias(X)
        for k in range(n - 1):
            dt = self.keyframes[k].dt_to_next
            if dt <= 1e-6:
                continue
            a = self.keyframes[k].acc_ned - bias_est
            p_k, p_k1 = self._pos(X, k), self._pos(X, k + 1)
            v_k, v_k1 = self._vel(X, k), self._vel(X, k + 1)
            # trapezoidal position + velocity update
            rp = p_k1 - p_k - 0.5 * (v_k + v_k1) * dt
            rv = v_k1 - v_k - a * dt
            for val in (rp, rv):
                start = len(r)
                r.extend([val[0], val[1], val[2]])
                slices.append((start, start + 3))
                names.append("imu")
        # slice index for flow per keyframe
        for k in range(n):
            kf = self.keyframes[k]
            if kf.flow_meas is not None:
                v_k = self._vel(X, k)
                flow_res = v_k - kf.flow_meas
                start = len(r)
                r.extend([flow_res[0], flow_res[1], flow_res[2]])
                slices.append((start, start + 3))
                names.append("flow")
                flow_slice[k] = (start, start + 3)

        # ---- GNSS absolute factors ----
        for k in range(n):
            kf = self.keyframes[k]
            if kf.gps_pos is not None:
                p_k, v_k = self._pos(X, k), self._vel(X, k)
                rp = p_k - kf.gps_pos
                rv = v_k - kf.gps_vel
                for val in (rp, rv):
                    start = len(r)
                    r.extend([val[0], val[1], val[2]])
                    slices.append((start, start + 3))
                    names.append("gps")

        # ---- landmark angle factors (position-only, attitude-invariant) ----
        lm_slices = []
        for k in range(n):
            kf = self.keyframes[k]
            if len(kf.lm_ids) < 2:
                continue
            p_k = self._pos(X, k)
            for i in range(len(kf.lm_ids)):
                for j in range(i + 1, len(kf.lm_ids)):
                    lm_i = self.landmarks[int(kf.lm_ids[i])]
                    lm_j = self.landmarks[int(kf.lm_ids[j])]
                    a_obs = _angle(kf.lm_dirs[i], kf.lm_dirs[j])
                    a_pred = _angle(_unit(lm_i - p_k), _unit(lm_j - p_k))
                    resid = a_pred - a_obs
                    start = len(r)
                    r.append(resid)
                    slices.append((start, start + 1))
                    names.append("lm")
                    lm_slices.append((start, start + 1))

        # ---- bias prior: keep the estimated IMU bias small ----
        bias_res = bias_est
        start = len(r)
        r.extend([bias_res[0], bias_res[1], bias_res[2]])
        slices.append((start, start + 3))
        names.append("bias_prior")

        return np.asarray(r, dtype=float), slices, names, flow_slice, lm_slices

    # ------------------------------------------------------------------ #
    # numeric Jacobian (central diff) for the concatenated residual.
    # ------------------------------------------------------------------ #
    def _jac(self, f, X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        f0, _, _, _, _ = f(X)
        n = len(X)
        m = len(f0)
        J = np.zeros((m, n))
        for j in range(n):
            Xp = X.copy()
            Xm = X.copy()
            Xp[j] += eps
            Xm[j] -= eps
            fp, _, _, _, _ = f(Xp)
            fm, _, _, _, _ = f(Xm)
            J[:, j] = (fp - fm) / (2.0 * eps)
        return J

    def _analytic_jacobian(self, X: np.ndarray):
        """Build the residual Jacobian directly from the factor model.

        IMU / flow / GPS factors are linear in the state (so their derivatives
        are constants). Only the landmark angle factors are nonlinear in one
        keyframe's position block, and those are small and computed in closed
        form.  This keeps the whole optimizer fast enough to run inline.
        """
        n = self._n_pose_vars()
        ncols = 6 * n + 3
        if n == 0:
            return np.zeros((0, ncols)), [], [], {}, []

        # First pass: dimensions / slices (same layout as _residuals).
        r0, slices, names, flow_slice, lm_slice = self._residuals(X)
        m = len(r0)
        J = np.zeros((m, ncols))
        bstart = self._bias_start()

        # ---- IMU (linear between consecutive keyframes) ----
        row = 0
        for k in range(n - 1):
            dt = self.keyframes[k].dt_to_next
            if dt <= 1e-6:
                continue
            base = 6 * k
            nxt = 6 * (k + 1)
            # rp = p_{k+1}-p_k - .5(v_k+v_{k+1})dt
            for c in range(3):
                J[row, base + c] = -1.0
                J[row, nxt + c] = 1.0
                J[row, base + 3 + c] = -0.5 * dt
                J[row, nxt + 3 + c] = -0.5 * dt
                row += 1
            # rv = v_{k+1} - v_k - (a_raw - b) dt
            # d(rv)/db = +dt * I
            for c in range(3):
                J[row, base + 3 + c] = -1.0
                J[row, nxt + 3 + c] = 1.0
                J[row, bstart + c] = dt
                row += 1

        # ---- flow (linear: v_k - flow) ----
        for k in range(n):
            if k in flow_slice:
                base = 6 * k
                for c in range(3):
                    J[row, base + 3 + c] = 1.0
                    row += 1

        # ---- GPS (linear: p_k - gps, v_k - gps_vel) ----
        for k in range(n):
            kf = self.keyframes[k]
            if kf.gps_pos is not None:
                base = 6 * k
                for c in range(3):
                    J[row, base + c] = 1.0
                    row += 1
                for c in range(3):
                    J[row, base + 3 + c] = 1.0
                    row += 1

        # ---- landmark angle (nonlinear, only in that keyframe position) ----
        for k in range(n):
            kf = self.keyframes[k]
            if len(kf.lm_ids) < 2:
                continue
            p_k = self._pos(X, k)
            for i in range(len(kf.lm_ids)):
                for j in range(i + 1, len(kf.lm_ids)):
                    lm_i = self.landmarks[int(kf.lm_ids[i])]
                    lm_j = self.landmarks[int(kf.lm_ids[j])]
                    gi = _unit(lm_i - p_k)
                    gj = _unit(lm_j - p_k)
                    cos = float(np.clip(np.dot(gi, gj), -1.0, 1.0))
                    sin = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
                    # dtheta/dp
                    di = -(lm_i - p_k) / max(np.linalg.norm(lm_i - p_k), 1e-9)
                    dj = -(lm_j - p_k) / max(np.linalg.norm(lm_j - p_k), 1e-9)
                    if sin > 1e-9:
                        dtheta_dui = -(gj - cos * gi) / sin
                        dtheta_duj = -(gi - cos * gj) / sin
                        dtheta_dp = dtheta_dui @ di + dtheta_duj @ dj
                    else:
                        dtheta_dp = np.zeros(3)
                    base = 6 * k
                    J[row, base:base + 3] = dtheta_dp
                    row += 1

        # ---- bias prior row: r_b = b  => d(r_b)/db = I ----
        for c in range(3):
            J[row, bstart + c] = 1.0
            row += 1

        return J, slices, names, flow_slice, lm_slice

    # ------------------------------------------------------------------ #
    # optimisation
    # ------------------------------------------------------------------ #
    def _cauchy_weights(self, r: np.ndarray) -> np.ndarray:
        c2 = self.cauchy_scale ** 2
        return 1.0 / (1.0 + (r * r) / c2)

    def optimize(self, verbose: bool = False) -> dict:
        n = self._n_pose_vars()
        if n < self.min_keyframes:
            return {"iterations": 0, "converged": False,
                    "flow_residual": 0.0, "lm_residual": 0.0,
                    "total_residual": 0.0, "health": 1.0,
                    "bias_est": self.bias.copy()}
        X = self._x0()

        def residual_fn(Xx):
            return self._residuals(Xx)

        r, _, _, flow_slice, lm_slice = residual_fn(X)
        lam = 1e-3
        last_cost = float(np.sum(self._cauchy_weights(r) * r ** 2))
        for it in range(self.max_iter):
            F, _, _, flow_slice, lm_slice = self._analytic_jacobian(X)
            r, _, _, flow_slice, lm_slice = residual_fn(X)
            w = self._cauchy_weights(r)
            # Bias-prior residual is not an outlier; it is a regulariser, so it
            # keeps a fixed (unrobust) weight.
            n_bias_first = len(r) - 3
            if n_bias_first >= 0:
                w[n_bias_first:n_bias_first + 3] = self.bias_reg
            W = np.sqrt(w)
            # Normal equations with robust weight (IRLS) + LM damping.
            A = (W[:, None] * F).T @ (W[:, None] * F)
            b = (W[:, None] * F).T @ (W * r)
            A += lam * np.eye(len(X))
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dx)):
                break
            X_new = X - dx
            r_new, _, _, _, _ = residual_fn(X_new)
            cost_new = float(np.sum(self._cauchy_weights(r_new) * r_new ** 2))
            if cost_new < last_cost:
                X = X_new
                last_cost = cost_new
                lam = max(lam * 0.3, 1e-6)
            else:
                lam *= 3.0
                if lam > 1e6:
                    break

        r, _, _, flow_slice, lm_slice = residual_fn(X)
        flow_resids = []
        for k in range(n):
            if k in flow_slice:
                s, e = flow_slice[k]
                flow_resids.append(float(np.linalg.norm(r[s:e])))
        mean_flow = float(np.mean(flow_resids)) if flow_resids else 0.0
        lm_resids = [float(abs(r[s:e][0])) for s, e in lm_slice]
        mean_lm = float(np.mean(lm_resids)) if lm_resids else 0.0

        # health: 1.0 when residual ~ baseline (healthy), -> ~0 when residual
        # is an outlier relative to the calibrated healthy baseline.  This is
        # the calibration step that makes a healthy mission map to health~1.0.
        excess = max(0.0, mean_flow - self.baseline_residual)
        health = 1.0 / (1.0 + excess / self.residual_scale)
        health = float(np.clip(health, 0.0, 1.0))

        self.flow_residual = mean_flow
        self.lm_residual = mean_lm
        self.total_residual = float(np.mean(r ** 2)) if r.size else 0.0
        self.health = health
        self._last_flow_residuals = np.asarray(flow_resids)

        # Warm start: write the optimized state back into the keyframes so the
        # next call does not start from the (flow-fed / far) initial guess.
        # The estimated IMU bias is kept on the graph itself, not the keyframes.
        for k in range(n):
            self.keyframes[k].p0 = self._pos(X, k).copy()
            self.keyframes[k].v0 = self._vel(X, k).copy()
            self.aux_positions[k] = self._pos(X, k).copy()
            self.aux_velocities[k] = self._vel(X, k).copy()
        self.bias = self._bias(X).copy()

        return {
            "iterations": it + 1,
            "converged": True,
            "flow_residual": mean_flow,
            "lm_residual": mean_lm,
            "total_residual": self.total_residual,
            "health": health,
            "flow_components": len(flow_resids),
            "bias_est": self.bias.copy(),
        }


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def build_keyframe(
    p: np.ndarray,
    v: np.ndarray,
    acc_meas: np.ndarray,
    rpy: np.ndarray,
    dt_to_next: float,
    flow_meas: np.ndarray | None,
    gps_pos: np.ndarray | None,
    gps_vel: np.ndarray | None,
    lm_ids: np.ndarray,
    lm_dirs: np.ndarray,
) -> Keyframe:
    """Build a keyframe from raw sensor data.

    ``acc_meas`` is the raw body-frame accelerometer.  ``rpy`` is the AHRS
    attitude.  The gravity is removed and rotated to NED so the IMU factor uses
    a *measured* acceleration, not truth.
    """
    R_nb = euler_to_R(np.asarray(rpy, dtype=float))
    acc_ned = R_nb @ np.asarray(acc_meas, dtype=float) + GRAVITY_NED
    return Keyframe(
        p0=np.asarray(p, dtype=float).copy(),
        v0=np.asarray(v, dtype=float).copy(),
        acc_ned=acc_ned,
        dt_to_next=float(dt_to_next),
        flow_meas=None if flow_meas is None else np.asarray(flow_meas, dtype=float).copy(),
        gps_pos=None if gps_pos is None else np.asarray(gps_pos, dtype=float).copy(),
        gps_vel=None if gps_vel is None else np.asarray(gps_vel, dtype=float).copy(),
        lm_ids=np.asarray(lm_ids, dtype=int).copy(),
        lm_dirs=np.asarray(lm_dirs, dtype=float).copy(),
    )
