"""Guardian brain: turn fused state + threats into a safe maneuver decision.

The brain runs the threat engine every frame, then a bounded sense-and-avoid
planner, then picks a mode.  It also tracks which of its *declared* and
*undeclared* (not externally advertised) defensive capabilities were exercised,
so an operator can always inspect what the aircraft did and why.

Defensive modes:
  CRUISE_SAFE  - nominal, monitor only
  EVADE        - predicted collision risk; apply the chosen maneuver
  RECOVER_NAV  - GPS/jamming/spoofing evidence: drop the degraded source and
                 navigate on IMU/baro/mag (autonomous sensor-source recovery)
  SILENT       - RF emission throttling + defensive cloak when being tracked
  ABORT        - energy / envelope unsafe: land now
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .threats import GuardianState, ThreatEngine, ThreatReport
from .avoidance import EvasionPlanner, EvasionDecision

CRUISE_SAFE = "CRUISE_SAFE"
EVADE = "EVADE"
RECOVER_NAV = "RECOVER_NAV"
SILENT = "SILENT"
ABORT = "ABORT"


@dataclass
class BrainDecision:
    t: float
    mode: str
    reason: str = "nominal"
    threat: ThreatReport | None = None
    evasion: EvasionDecision | None = None
    declared_used: list[str] = field(default_factory=list)
    undeclared_used: list[str] = field(default_factory=list)
    compute_ms: float = 0.0
    energy_budget_mw: float = 0.0
    latency_ms: float = 0.0


def _score(existing: dict, report: ThreatReport) -> None:
    if report.score >= 0.55:
        existing[report.kind] = max(existing.get(report.kind, 0.0), report.score)


class GuardianBrain:
    def __init__(self, threat_engine: ThreatEngine | None = None,
                 evasion: EvasionPlanner | None = None) -> None:
        self.threat = threat_engine or ThreatEngine()
        self.evasion = evasion or EvasionPlanner()

    def decide(self, state: GuardianState, desire: np.ndarray,
               health_score: float | None = None) -> BrainDecision:
        """Produce one defensive decision for the current state.

        ``health_score`` is an optional aggregate maintenance-health in [0,1]
        (from :class:`airlab.guardian.health.HealthPrognosis`).  A critically
        low health is treated as an abort condition (predictive maintenance,
        master prompt §35) with priority below an immediate battery/wind
        emergency but above a soft collision risk.
        """
        reports = self.threat.evaluate(state)
        active: dict[str, float] = {}
        for r in reports:
            _score(active, r)

        health_abort = health_score is not None and health_score < 0.45

        ev = self.evasion.plan(state, desire)
        mode = CRUISE_SAFE
        reason = "nominal"
        decl: list[str] = []
        undecl: list[str] = []
        threat_rep: ThreatReport | None = None

        obstacle = active.get("obstacle", 0.0)
        nav = active.get("spoofing", 0.0) + active.get("jamming", 0.0)
        wind = active.get("wind", 0.0)
        cyber = active.get("cyber", 0.0)
        battery = active.get("battery", 0.0)

        # Priority order: safety-critical first, then degraded-nav, then cloak.
        if health_abort and battery < 0.55 and wind < 0.80:
            mode = ABORT
            reason = "predictive_maintenance_health_low"
            decl.append("predictive_maintenance_health")
            undecl.append("maintenance_abort")
            threat_rep = None
        elif battery >= 0.55 or wind >= 0.80:
            mode = ABORT
            reason = "energy/wind_envelope_unsafe"
            decl.append("energy_envelope_guard")
            if active.get("jamming", 0.0) >= 0.55 or cyber >= 0.55:
                undecl.append("silent_rf")
            threat_rep = active_report(reports, "battery" if battery >= wind else "wind")
        elif obstacle >= 0.55:
            mode = EVADE
            reason = f"collision_risk:{ev.reason}" if ev else "collision_risk"
            decl.append("predictive_sense_avoid")
            if ev and ev.evading:
                undecl.append("cloaked_evasion")
            if active.get("jamming", 0.0) >= 0.55 or cyber >= 0.55:
                undecl.append("silent_rf")
            threat_rep = active_report(reports, "obstacle")
        elif nav >= 0.55:
            # Spoofing: keep navigating, switch sources.  Jamming: also cut RF
            # emissions (defensive silence) so a tracker cannot use our link.
            if active.get("jamming", 0.0) >= 0.55:
                mode = SILENT
                reason = "jamming_rf_silence"
                undecl.append("silent_rf")
            else:
                mode = RECOVER_NAV
                reason = "navigation_integrity_degraded"
            decl.append("cross_sensor_nav_consistency")
            undecl.append("autonomous_source_switch")
            threat_rep = active_report(reports, "spoofing" if active.get("spoofing")
                                       else "jamming")
        elif cyber >= 0.55:
            mode = RECOVER_NAV
            reason = "command_anomaly"
            decl.append("command_envelope_guard")
            undecl.append("command_anomaly_reject")
            undecl.append("silent_rf")
            threat_rep = active_report(reports, "cyber")
        elif state.obstacles and ev and ev.evading:
            mode = EVADE
            reason = "tracked/margin_soft"
            decl.append("predictive_sense_avoid")
            undecl.append("cloaked_evasion")

        # Cost model: very small on-edge watchdog.  This is an estimate to feed
        # the "intelligence per watt" score, not a hardware measurement.
        compute = 1.4 + 0.9 * len(reports) + 0.6 * len(active)
        latency = 0.8 + compute * 0.25
        power = 90.0 if mode == CRUISE_SAFE else (140.0 if mode != ABORT else 70.0)

        return BrainDecision(
            t=state.t, mode=mode, reason=reason, threat=threat_rep, evasion=ev,
            declared_used=decl, undeclared_used=undecl,
            compute_ms=compute, energy_budget_mw=power, latency_ms=latency,
        )


def active_report(reports: list[ThreatReport], kind: str) -> ThreatReport | None:
    for r in reports:
        if r.kind == kind:
            return r
    return None
