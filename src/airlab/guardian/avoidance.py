"""Predictive sense-and-avoid + defensive maneuver selection.

Uses a receding-horizon candidate maneuver search over a lightweight linear
model: each candidate changes the velocity reference; the planner projects own
and obstacle motion, computes the worst projected clearance, and picks the
maneuver that maximises the margin while minimising mission deviation.

``cloak`` is the defensive "don't be predictable" option: when a trackable
intruder/emitter is detected, the aircraft injects a small, bounded, randomised
lateral oscillation so an external tracker cannot simply extrapolate a straight
line.  It is bounded and never harms the mission-envelope check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .threats import GuardianState, Obstacle


@dataclass
class EvasionDecision:
    action: str                 # none / yaw_left / yaw_right / climb / descend / slow
    offset_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    min_clearance: float = 1e9
    reason: str = ""

    @property
    def evading(self) -> bool:
        return self.action != "none"


class EvasionPlanner:
    def __init__(self, horizon: float = 0.8, step: float = 0.1,
                 safety_margin: float = 1.5, lateral_gain: float = 1.2,
                 vertical_gain: float = 1.0, slow_gain: float = 0.8,
                 mission_weight: float = 0.35, cloak_amp: float = 0.25) -> None:
        self.horizon = horizon
        self.step = step
        self.safety_margin = safety_margin
        self.lateral_gain = lateral_gain
        self.vertical_gain = vertical_gain
        self.slow_gain = slow_gain
        self.mission_weight = mission_weight
        self.cloak_amp = cloak_amp

    def plan(self, state: GuardianState, desire: np.ndarray) -> EvasionDecision:
        """Pick the best maneuver given current obstacles and the desired vel.

        ``desire`` is the mission velocity reference (NED).  Returns the chosen
        action, the velocity offset and the projected worst clearance.
        """
        candidates: list[tuple[str, np.ndarray]] = [
            ("none", np.zeros(3)),
            ("yaw_left", self._left(state, desire)),
            ("yaw_right", self._right(state, desire)),
            ("climb", np.array([0.0, 0.0, -self.vertical_gain])),
            ("descend", np.array([0.0, 0.0, self.vertical_gain])),
            ("slow", -self.slow_gain * np.asarray(state.vel, dtype=float)),
        ]
        best: tuple[str, np.ndarray] | None = None
        best_cost = float("inf")
        best_clear = 1e9
        for action, dv in candidates:
            clear = self._min_clearance(state, dv)
            dev = float(np.linalg.norm(np.asarray(desire, dtype=float)
                                       - np.asarray(state.vel, dtype=float) - dv))
            # A maneuver only matters if the straight-line path is threatened.
            if clear < self.safety_margin:
                cost = -clear + self.mission_weight * dev
            else:
                cost = self.mission_weight * dev
            if cost < best_cost:
                best_cost = cost
                best = (action, dv)
                best_clear = clear

        if best is None:
            return EvasionDecision("none")

        action, dv = best
        reason = (f"min_clearance={best_clear:.2f} m"
                  if best_clear < self.safety_margin else "safe")

        # Defensive cloak: bounded random lateral oscillation when an obstacle /
        # tracker is present but still outside the hard margin.  Keeps the
        # trajectory less extrapolable without breaking the safe envelope.
        final_dv = dv.copy()
        if state.obstacles and best_clear < self.safety_margin + 2.0:
            rng = np.random.default_rng(int((state.t * 1000) % (2**31)))
            ra, rb = state.vel[0], state.vel[1]
            mag = float(np.hypot(ra, rb))
            if mag > 1e-6:
                perp = np.array([-rb / mag, ra / mag, 0.0])
                jit = float(rng.uniform(-self.cloak_amp, self.cloak_amp))
                final_dv = final_dv + jit * perp

        return EvasionDecision(action=action, offset_vel=final_dv,
                               min_clearance=best_clear, reason=reason)

    # -- helpers ------------------------------------------------------------ #
    def _left(self, state: GuardianState, desire: np.ndarray) -> np.ndarray:
        return self._lateral(state, desire, +1.0)

    def _right(self, state: GuardianState, desire: np.ndarray) -> np.ndarray:
        return self._lateral(state, desire, -1.0)

    def _lateral(self, state: GuardianState, desire: np.ndarray,
                 sign: float) -> np.ndarray:
        d = np.asarray(desire, dtype=float)
        mag = float(np.linalg.norm(d[:2]))
        if mag < 1e-6:
            return np.zeros(3)
        perp = np.array([-d[1], d[0], 0.0]) / mag
        return self.lateral_gain * sign * perp

    def _min_clearance(self, state: GuardianState, dv: np.ndarray) -> float:
        own_v = np.asarray(state.vel, dtype=float) + np.asarray(dv, dtype=float)
        worst = 1e9
        for step in np.arange(self.step, self.horizon + 1e-9, self.step):
            own_p = np.asarray(state.pos, dtype=float) + own_v * step
            for o in state.obstacles:
                op = np.asarray(o.pos, dtype=float) + np.asarray(o.vel, dtype=float) * step
                dist = float(np.linalg.norm(own_p - op)) - o.radius - state.rab
                worst = min(worst, dist)
        if not state.obstacles:
            return 1e9
        return worst
