"""Evaluation metrics for autonomous navigation experiments.

These are deliberately *domain-appropriate* metrics (not generic RMSE-only):
we care about where the errors come from and whether the system is becoming
*safe* over the mission, in addition to raw tracking quality.
"""

from __future__ import annotations

import numpy as np

from .safety import CRUISE, LAND, LANDED


def evaluate_mission(
    true_pos: np.ndarray,
    est_pos: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray | None = None,
    est_vel: np.ndarray | None = None,
    gps_available: np.ndarray | None = None,
    dt: float = 0.01,
) -> dict:
    """Metrics over an N x 3 set of recorded positions.

    ``true_pos`` is the simulator ground truth (NED).
    ``est_pos`` is the EKF position estimate (NED).
    ``ref_pos`` is the mission desired position (NED).
    ``gps_available`` is optional 1-D boolean masking valid GNSS.
    """
    true_pos = np.asarray(true_pos, dtype=float)
    est_pos = np.asarray(est_pos, dtype=float)
    ref_pos = np.asarray(ref_pos, dtype=float)
    n = len(true_pos)
    if n == 0:
        return {}

    tracking_err = ref_pos - true_pos
    est_err = est_pos - true_pos

    horizontal_rmse = float(np.sqrt(np.mean(tracking_err[:, 0] ** 2 + tracking_err[:, 1] ** 2)))
    vertical_rmse = float(np.sqrt(np.mean(tracking_err[:, 2] ** 2)))
    pos_rmse = float(np.sqrt(np.mean(np.sum(tracking_err ** 2, axis=1))))
    max_horizontal = float(np.max(np.linalg.norm(tracking_err[:, :2], axis=1)))
    max_vertical = float(np.max(np.abs(tracking_err[:, 2])))

    est_h_rmse = float(np.sqrt(np.mean(est_err[:, 0] ** 2 + est_err[:, 1] ** 2)))
    est_v_rmse = float(np.sqrt(np.mean(est_err[:, 2] ** 2)))

    # Velocity alignment (useless if no refs)
    vel_align = None
    if ref_vel is not None and est_vel is not None and len(ref_vel) == n:
        ref_vel = np.asarray(ref_vel)
        est_vel = np.asarray(est_vel)
        den = np.linalg.norm(ref_vel, axis=1)
        cos = np.sum(ref_vel * est_vel, axis=1) / np.maximum(den, 1e-9)
        vel_align = float(np.mean(cos))

    # fraction of time in a collision / ground state (true pos down >=0)
    collision = float(np.mean(true_pos[:, 2] >= -1e-6)) if n else 0.0

    # fraction of samples with acceptable tracking error (horizontal + vertical)
    ok = (np.linalg.norm(tracking_err[:, :2], axis=1) < 4.0) & (np.abs(tracking_err[:, 2]) < 2.0)
    in_bounds = float(np.mean(ok))

    # estimator availability metric (e.g. percentage of time with GNSS)
    gps_frac = float(np.mean(gps_available)) if gps_available is not None and len(gps_available) == n else None

    return {
        "pos_rmse": pos_rmse,
        "horizontal_rmse": horizontal_rmse,
        "vertical_rmse": vertical_rmse,
        "max_horizontal_err": max_horizontal,
        "max_vertical_err": max_vertical,
        "est_horizontal_rmse": est_h_rmse,
        "est_vertical_rmse": est_v_rmse,
        "velocity_alignment": vel_align,
        "ground_collision_frac": collision,
        "in_bounds_frac": in_bounds,
        "gps_available_frac": gps_frac,
        "samples": n,
    }


def safety_metrics(run, dt: float = 0.01) -> dict:
    """Safety/decision metrics from a SimRun with a SafetyMonitor active.

    These complement the tracking metrics with *outcome* metrics: did the
    system recognise danger, later activate hold/land, and actually finish on
    the ground instead of crashing somewhere else?
    """
    n = len(run.mode)
    if n == 0:
        return {
            "safety_outcome": "none",
            "safety_fraction": 0.0,
            "landed": 0.0,
            "crash": 0.0,
            "mean_unc_horiz_std": float("nan"),
            "mean_flow_health": float("nan"),
        }

    # LANDED means altitude reached ground under a deliberate land command.
    landed_frac = float(np.mean(np.asarray(run.landed, dtype=float)))
    # Time spent outside CRUISE (i.e. reacting to danger).
    safety_frac = float(np.mean([1.0 if m != CRUISE else 0.0 for m in run.mode]))

    true_pos = np.asarray(run.true_pos, dtype=float)
    on_ground = true_pos[:, 2] >= -1e-6
    # An "unintended crash" is ground contact that happened while the safety
    # layer was *not* deliberately landing / had not yet landed.
    intentional = [(m in (LAND, LANDED)) for m in run.mode]
    unintended = float(np.mean([1.0 if og and not inten else 0.0
                                for og, inten in zip(on_ground, intentional)]))

    mean_unc = float(np.mean(np.asarray(run.unc_horiz_std, dtype=float))) if run.unc_horiz_std else float("nan")
    mean_health = float(np.mean(np.asarray(run.flow_health, dtype=float))) if run.flow_health else float("nan")

    outcome = "completed"
    if landed_frac >= 0.3 and unintended < 0.05:
        outcome = "landed_safely"
    elif unintended > 0.05:
        outcome = "crash"
    elif safety_frac < 0.05:
        outcome = "completed"
    else:
        outcome = "reactive_hold"

    return {
        "safety_outcome": outcome,
        "safety_fraction": safety_frac,
        "landed": landed_frac,
        "crash": unintended,
        "mean_unc_horiz_std": mean_unc,
        "mean_flow_health": mean_health,
    }


def average_of_metrics(metric_list: list[dict], keys: list[str] | None = None) -> dict:
    """Average a list of metric dicts (ignores None values)."""
    if not metric_list:
        return {}
    if keys is None:
        keys = sorted(set().union(*[m.keys() for m in metric_list]))
    out = {}
    for k in keys:
        values = [m[k] for m in metric_list if k in m and m[k] is not None]
        if values:
            out[k] = float(np.mean(values))
    return out
