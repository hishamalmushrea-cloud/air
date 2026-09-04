"""Bridge the guardian oracle re-planner into the real mission controller.

``src/airlab/simulator.py`` is the high-fidelity flight stack (dynamics +
sensors + EKF + cascaded controller + waypoint mission).  The guardian's
``PredictiveRePlanner`` was previously only demonstrated in the behaviour lab
(``run_guardian.py``).  This bridge makes it usable *inside* a real mission:

  * runs the oracle against the remaining mission waypoints,
  * accounts for known obstacle / jamming corridors supplied to the mission,
  * checks energy feasibility against the platform battery,
  * only swaps the route when it is strictly better (risk reduction, clearance,
    bounded detour) so that a noisy risk field can never degrade a safe mission.

It is deliberately a *reactive route editor*: it never overrides the safety
FSM, never turns a hold/land into a climb, and never raises risk.  That keeps
safety-in-design (master prompt §30) rather than bolt-on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .replan import PredictiveRePlanner, ReplanResult


@dataclass
class _BridgeObstacle:
    pos: np.ndarray
    vel: np.ndarray
    radius: float = 0.5


def _norm_obs(o) -> _BridgeObstacle:
    if hasattr(o, "pos") and hasattr(o, "vel"):
        return _BridgeObstacle(np.asarray(o.pos, dtype=float).reshape(3),
                               np.asarray(o.vel, dtype=float).reshape(3),
                               float(getattr(o, "radius", 0.5)))
    p, v, *rest = o
    return _BridgeObstacle(np.asarray(p, dtype=float).reshape(3),
                           np.asarray(v, dtype=float).reshape(3),
                           float(rest[0]) if rest else 0.5)


@dataclass
class BridgeConfig:
    replan_period_s: float = 2.0
    min_risk_reduction: float = 0.02       # must beat baseline by this much
    min_clearance_m: float = 2.0
    # A corridor that cuts predicted risk by >0.5 may justify a large detour
    # (e.g. routing around a single obstacle on an otherwise straight leg).
    # 0.50 rejects only mission-changing detours while still bounding cost.
    max_extra_distance_frac: float = 0.50
    battery_capacity_wh: float = 71.0
    energy_reserve_frac: float = 0.15
    hover_power_w: float = 112.0
    cruise_speed: float = 3.0


@dataclass
class BridgeHistory:
    steps: list[float] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class MissionReplanBridge:
    """Apply the guardian oracle to a running ``WaypointMission``."""

    def __init__(self, mission, pos_ned: np.ndarray,
                 battery_frac: float = 1.0,
                 obstacles=None, jamming_centers=None,
                 planner: PredictiveRePlanner | None = None,
                 config: BridgeConfig | None = None) -> None:
        self.mission = mission
        self.config = config or BridgeConfig()
        self.planner = planner or PredictiveRePlanner(
            cruise_speed=self.config.cruise_speed,
            hover_power_w=self.config.hover_power_w,
            battery_capacity_wh=self.config.battery_capacity_wh,
            energy_reserve_frac=self.config.energy_reserve_frac,
            min_clearance=self.config.min_clearance_m,
        )
        self.obstacles = [_norm_obs(o) for o in (obstacles or [])]
        self.jamming_centers = [np.asarray(c, dtype=float).reshape(3)
                                for c in (jamming_centers or [])]
        self.pos_ned = np.asarray(pos_ned, dtype=float).reshape(3)
        self.battery_frac = float(battery_frac)
        self.history = BridgeHistory()
        self._last_planner_t = -float("inf")
        self.last_result: ReplanResult | None = None
        self.applied = False

    def update_telem(self, pos_ned: np.ndarray, battery_frac: float,
                     t: float) -> None:
        self.pos_ned = np.asarray(pos_ned, dtype=float).reshape(3)
        self.battery_frac = float(battery_frac)
        self.history.steps.append(float(t))

    def try_replan(self, t: float, force: bool = False) -> ReplanResult | None:
        """Run the oracle if enough time has elapsed and the route is open."""
        if (not force and self.config.replan_period_s > 0.0 and
                t - self._last_planner_t < self.config.replan_period_s):
            return None
        if self.mission.completed:
            return None
        remaining = self.mission.remaining_ned()
        if len(remaining) < 2:
            return None
        self._last_planner_t = t
        res = self.planner.plan(self.pos_ned, remaining,
                                self.battery_frac,
                                obstacles=self.obstacles,
                                jamming_centers=self.jamming_centers)
        self.last_result = res
        self.history.modes.append("evaluated")
        self._maybe_apply(res, t)
        return res

    def _maybe_apply(self, res: ReplanResult, t: float) -> None:
        if not res.feasible:
            self.history.modes.append("rejected_energy")
            return
        if res.risk_reduction < self.config.min_risk_reduction:
            self.history.modes.append("rejected_low_gain")
            return
        if res.extra_distance_frac > self.config.max_extra_distance_frac:
            self.history.modes.append("rejected_detour")
            return
        if res.min_clearance_m < self.config.min_clearance_m:
            self.history.modes.append("rejected_clearance")
            return
        # The oracle's route begins at the current position; swap only the
        # future polyline so we never command the aircraft back to where it is.
        future = [w.copy() for w in res.route[1:]]
        if len(future) < 2:
            self.history.modes.append("rejected_short_route")
            return
        self.mission.set_route_ned(future, reset=True)
        self.applied = True
        self.history.modes.append("applied")
        self.history.events.append({
            "t_s": float(t),
            "risk_reduction": float(res.risk_reduction),
            "bas_risk": float(res.bas_risk),
            "repl_risk": float(res.repl_risk),
            "extra_frac": float(res.extra_distance_frac),
            "clearance_m": float(res.min_clearance_m),
        })
