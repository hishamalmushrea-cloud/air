"""Threat engine: detect anything that threatens the aircraft or its mission.

Four independent evidence channels, each fused from *different* sensors so a
fault in one cannot self-confirm:

  * navigation spoofing / jamming   - GPS vs IMU dead-reckon vs magnetometer
  * obstacle / collision risk       - nearest projected clearance over horizon
  * environment anomaly             - wind vs expected envelope
  * cyber / command anomaly         - commanded state vs sane flight envelope

Every report carries a ``score`` in [0,1], an evidence list and a ``kind``.
The brain consumes these; it never uses a single channel to act alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -30.0, 30.0)))))


@dataclass
class Obstacle:
    pos: np.ndarray
    vel: np.ndarray
    radius: float = 0.5


@dataclass
class GuardianState:
    """A single fused, time-stamped snapshot the brain reasons over."""

    t: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))      # NED
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))    # commanded accel
    gps_pos: np.ndarray | None = None
    gps_vel: np.ndarray | None = None
    imu_dr_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_dr_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag_heading: float | None = None         # rad
    gps_course: float | None = None          # rad
    gps_signal_quality: float = 1.0          # 0..1 C/No-like health
    baro_ok: bool = True
    battery_frac: float = 1.0
    energy_required_frac: float = 0.0        # energy still needed to finish
    wind_est: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s2-ish
    obstacles: list[Obstacle] = field(default_factory=list)
    rab: float = 0.5                          # own collision radius


@dataclass
class ThreatReport:
    kind: str
    score: float
    evidence: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.score >= 0.55


def _gps_mag_consistency(state: GuardianState) -> float:
    """Heading mismatch between magnetometer and GPS course (spoof cue)."""
    if state.mag_heading is None or state.gps_course is None:
        return 0.0
    d = float(np.angle(np.exp(1j * (state.mag_heading - state.gps_course))))
    return float(np.abs(d))


class ThreatEngine:
    """Fuse independent channels into interpretable threat reports."""

    def __init__(self) -> None:
        # thresholds (tunable; values are design-in, documented in tests)
        self.gps_dr_div_thresh = 2.5   # m over the fusion cadence
        self.heading_mismatch_thresh = 0.35  # rad (~20 deg)
        self.safety_margin = 1.5       # m projected clearance to trigger
        self.wind_envelope = 1.6       # m/s^2 equivalent
        self.cyber_accel_envelope = 8.0  # m/s^2 total commanded accel sanity

    def evaluate(self, s: GuardianState) -> list[ThreatReport]:
        """Return all active + latent threat reports (score in [0,1])."""
        reports: list[ThreatReport] = []
        reports.append(self._nav(s))
        reports.append(self._obstacle(s))
        reports.append(self._env(s))
        reports.append(self._cyber(s))
        reports.append(self._battery(s))
        return [r for r in reports if r.score > 1e-6]

    # -- channels ---------------------------------------------------------- #
    def _nav(self, s: GuardianState) -> ThreatReport:
        # Dead-reckon residual: measure how far IMU DR has diverged from GPS.
        dr_gps = 0.0
        if s.gps_pos is not None:
            dr_gps = float(np.linalg.norm(s.imu_dr_pos - s.gps_pos))
        # Jamming: GPS absent but IMU/baro alive.
        jam_score = 0.0
        if s.gps_pos is None and s.baro_ok:
            jam_score = _sigmoid((1.0 - s.gps_signal_quality - 0.2) / 0.15)
        # Spoofing: GPS present but disagrees with independent channels.
        gps_present = s.gps_pos is not None
        head = 0.0
        if gps_present and s.mag_heading is not None and s.gps_course is not None:
            head = _gps_mag_consistency(s)
        dr_t = _sigmoid((dr_gps - self.gps_dr_div_thresh) / 0.8)
        head_t = _sigmoid((head - self.heading_mismatch_thresh) / 0.12)
        spoof = max(dr_t * gps_present, head_t * gps_present)
        # Overall navigation threat is the max of the two failure modes.
        score = float(np.clip(max(spoof, jam_score), 0.0, 1.0))
        ev = [] if score < 1e-6 else []
        if spoof >= 0.55 and gps_present:
            ev.append(f"gps_vs_imu_dr={dr_gps:.2f} m")
            if head >= self.heading_mismatch_thresh:
                ev.append(f"mag_vs_gps_heading={head:.2f} rad")
        if jam_score >= 0.55:
            ev.append("gps_lost_but_imu/baro_healthy")
        kind = "spoofing" if spoof >= jam_score else "jamming"
        return ThreatReport(kind, score, ev)

    def _obstacle(self, s: GuardianState) -> ThreatReport:
        if not s.obstacles:
            return ThreatReport("obstacle", 0.0)
        # project the closest approach over a short horizon using linear motion
        dt = 0.6
        worst = 1e9
        for o in s.obstacles:
            rel0 = s.pos - o.pos
            relv = s.vel - o.vel
            # time of closest approach
            denom = float(np.dot(relv, relv))
            tca = 0.0
            if denom > 1e-9:
                tca = float(np.clip(-np.dot(rel0, relv) / denom, 0.0, dt))
            rel = rel0 + relv * tca
            dist = float(np.linalg.norm(rel)) - o.radius - s.rab
            worst = min(worst, dist)
        margin = worst - self.safety_margin
        score = _sigmoid(-margin / 0.6)
        return ThreatReport("obstacle", score,
                            [f"min_clearance={worst:.2f} m"] if score > 1e-6 else [])

    def _env(self, s: GuardianState) -> ThreatReport:
        w = float(np.linalg.norm(s.wind_est))
        score = _sigmoid((w - self.wind_envelope) / 0.5)
        return ThreatReport("wind", score,
                            [f"wind={w:.2f} m/s2"] if score > 1e-6 else [])

    def _cyber(self, s: GuardianState) -> ThreatReport:
        a = float(np.linalg.norm(s.a_cmd))
        score = _sigmoid((a - self.cyber_accel_envelope) / 2.0)
        return ThreatReport("cyber", score,
                            [f"a_cmd={a:.2f} m/s2"] if score > 1e-6 else [])

    def _battery(self, s: GuardianState) -> ThreatReport:
        short = s.energy_required_frac - s.battery_frac
        score = _sigmoid((short - 0.02) / 0.05)
        return ThreatReport("battery", score,
                            [f"short={short:.2f} frac"] if score > 1e-6 else [])
