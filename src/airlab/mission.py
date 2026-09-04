"""Mission manager: waypoint trajectory generation.

Coordinates
-----------
Input mission waypoints use the intuitive ``(north, east, height)`` convention
(height in metres above ground, positive up).  Internally they are mapped to
NED ``(north, east, down)`` with ``down = -height``.

The trajectory generator is deliberately non-trivial: it projects the current
position onto the active segment and advances the *desired position* by a
lookahead, giving the controller a smooth feedforward velocity reference.
"""

from __future__ import annotations

import numpy as np

from .math_utils import wrap_pi


def _to_ned(wp: np.ndarray) -> np.ndarray:
    return np.array([float(wp[0]), float(wp[1]), -float(wp[2])])


class WaypointMission:
    def __init__(
        self,
        waypoints: list[tuple[float, float, float]],
        speed: float = 2.0,
        yaw: float = 0.0,
        reach_radius: float = 0.6,
        lookahead: float = 0.8,
        ground: float = 0.0,
    ) -> None:
        """waypoints: [(north, east, height)].

        ``speed`` is the desired cruise speed along each segment (m/s).
        ``yaw`` is a constant heading command (can be replaced later with
        per-waypoint headings).
        """
        if len(waypoints) < 2:
            raise ValueError("mission needs at least 2 waypoints")
        self.wp_ned = np.array([_to_ned(np.asarray(w)) for w in waypoints])
        self.speed = speed
        self.yaw_ref = yaw
        self.reach_radius = reach_radius
        self.lookahead = lookahead
        self.ground = ground

        self.seg_idx = 0
        self.distance_traveled = 0.0

    def _get_segment(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = self.wp_ned[self.seg_idx]
        b = self.wp_ned[self.seg_idx + 1]
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            seg = np.array([0.0, 0.0, 0.0])
        else:
            seg = seg / L
        return a, b, seg

    def _advance(self, pos: np.ndarray, dt: float) -> None:
        a, b, _ = self._get_segment()
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-9:
            return
        proj = np.dot(pos - a, ab) / (L * L)
        reached = (proj >= 1.0) or (np.linalg.norm(pos - b) < self.reach_radius)
        if reached:
            self.seg_idx = min(self.seg_idx + 1, len(self.wp_ned) - 2)

    def desired(self, pos: np.ndarray, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (pos_ref_ned, vel_ref_ned, yaw_ref)."""
        self._advance(pos, dt)
        a, b, seg = self._get_segment()
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-9:
            return b, np.zeros(3), self.yaw_ref

        s = float(np.clip(np.dot(pos - a, ab) / max(L * L, 1e-9), 0.0, 1.0))
        ahead_s = s + min(self.lookahead / max(L, 1e-9), 1.0 - s)
        pos_ref = a + ahead_s * ab

        # Decelerate smoothly as the along-track distance to the waypoint
        # shrinks, so the feedforward velocity does not make the vehicle
        # overshoot it (cross-track error alone should not keep feeding
        # velocity along the track).
        dist_along = max(0.0, (1.0 - s) * L)
        arrival_scale = float(np.clip(dist_along / 3.0, 0.0, 1.0))
        vel_ref = seg * self.speed * arrival_scale
        return pos_ref, vel_ref, self.yaw_ref

    @property
    def completed(self) -> bool:
        return self.seg_idx >= len(self.wp_ned) - 2

    def remaining_ned(self) -> list[np.ndarray]:
        """Remaining waypoints (NED), starting at the current active target."""
        if self.completed:
            return []
        return [self.wp_ned[i].copy() for i in range(self.seg_idx + 1,
                                                    len(self.wp_ned))]

    def set_route_ned(self, route_ned: list[np.ndarray],
                      reset: bool = True) -> None:
        """Replace the remaining route (NED).

        ``route_ned`` is the *future* polyline after the current position
        (i.e. the first entry is the next target), not a full path that
        includes the aircraft's current position.  Used by the guardian
        predictive re-planner (``airlab.guardian.sim_bridge``) to swap the
        active mission onto a lower-risk corridor without re-constructing the
        whole mission.
        """
        arr = [np.asarray(w, dtype=float).reshape(3) for w in route_ned]
        if len(arr) < 2:
            raise ValueError("route needs at least 2 waypoints")
        self.wp_ned = np.array(arr)
        if reset:
            self.seg_idx = 0
            self.distance_traveled = 0.0

    @property
    def progress(self) -> float:
        # simple normalized progress on current segment
        return min(self.seg_idx / max(len(self.wp_ned) - 2, 1), 1.0)
