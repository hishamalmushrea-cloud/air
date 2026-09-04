"""Nexus-Predator: the next-generation multirotor autonomy platform spec.

This file captures the *platform* that hosts the guardian brain: airframe,
sensor bundle, onboard compute and energy.  Speculative capabilities are scored
across the lab's 10-criteria rubric in ``specifications`` and flagged with a
quadrant (A = current/implemented, B = emerging, C = potential, D = speculative).
Nothing here is weaponised; all novel capabilities are defensive/resilience
features.

The platform is intentionally lightweight in the simulator: a point-mass
behaviour lab around ``GuardianBrain``, not a full rigid-body model.  Real
dynamics are already represented in ``simulator.py``; this module is the
autonomy/behaviour twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .threats import GuardianState, Obstacle
from .brain import GuardianBrain, BrainDecision, EVADE, ABORT, SILENT, RECOVER_NAV


@dataclass
class NexusSpec:
    name: str = "Nexus-Predator V2"
    mass_g: float = 1180.0
    hover_power_w: float = 112.0
    compute_topps: float = 0.35          # on-board edge TOPS (spiking-friendly)
    onnx_gops_per_w: float = 847.0       # neuromorphic efficiency (research target)
    max_speed_ms: float = 14.0
    endurance_min: float = 38.0
    sensors: list[str] = field(default_factory=lambda: [
        "IMU", "baro", "mag", "GNSS", "flow",
        "stereo IR", "micro-LiDAR", "RF ambient",
    ])
    # Quadrant A/B/C/D classification of the capability set.
    capabilities: dict[str, str] = field(default_factory=lambda: {
        "predictive_sense_avoid": "A",
        "cross_sensor_nav_consistency": "A",
        "learned_frame_trust": "A",
        "cloaked_evasion": "B",
        "autonomous_source_switch": "B",
        "silent_rf_mode": "C",
        "oracle_risk_world_model": "C",
        "emergent_adversarial_hardening": "D",
    })


@dataclass
class ScenarioMetrics:
    mode_histogram: dict[str, int]
    max_threat_reached: float
    final_clearance: float
    crashed: bool
    declared_used: set[str]
    undeclared_used: set[str]
    mean_compute_ms: float
    mean_energy_mw: float


class NexusAirV2:
    """Behaviour-lab wrapper: the new airframe + guardian brain.

    Not a full dyamics model — it integrates the point-mass motion of the own
    aircraft and external obstacles and asks the brain for a safe decision each
    step.  It demonstrates the *capabilities*, not the physics (physics lives in
    ``simulator.py``).
    """

    def __init__(self, brain: GuardianBrain | None = None) -> None:
        self.brain = brain or GuardianBrain()
        self.spec = NexusSpec()

    def run_step(self, state: GuardianState, desire: np.ndarray,
                 obstacle: Obstacle | None = None) -> tuple[BrainDecision, GuardianState]:
        s = state
        if obstacle is not None:
            s = GuardianState(**{**state.__dict__, "obstacles": [obstacle]})
        dec = self.brain.decide(s, desire)
        # Apply evasion offset (bounded, point-mass) for the kinematic showcase.
        if dec.mode == EVADE and dec.evasion is not None:
            s.vel = np.asarray(s.vel, dtype=float) + dec.evasion.offset_vel * 0.1
            s.pos = np.asarray(s.pos, dtype=float) + s.vel * 0.01
        else:
            s.pos = np.asarray(s.pos, dtype=float) + s.vel * 0.01
        return dec, s

    def simulate(self, initial: GuardianState, desire: np.ndarray,
                 n_steps: int = 60, obstacle_fn=None, step_dt: float = 0.05,
                 state_mod=None) -> ScenarioMetrics:
        """Integrate the point-mass + brain for ``n_steps``.

        ``obstacle_fn(k, pos)`` returns an :class:`Obstacle` (or ``None``) for
        step k.  ``state_mod(k, st)`` may edit the fused state before the brain
        sees it (e.g. drop GPS to simulate jamming, inject a spoofed position).
        The own state is integrated from the previous decision; the brain's
        evade offset is applied as a bounded velocity trim so the lab shows a
        *real* dodge, not just a label.
        """
        pos = np.asarray(initial.pos, dtype=float).copy()
        vel = np.asarray(initial.vel, dtype=float).copy()
        hist: dict[str, int] = {}
        max_threat = 0.0
        final_clear = 1e9
        declared: set[str] = set()
        undeclared: set[str] = set()
        compute = 0.0
        energy = 0.0
        crashed = False
        for k in range(n_steps):
            t = k * step_dt
            obstacle = obstacle_fn(k, pos) if obstacle_fn else None
            st = GuardianState(
                t=t, pos=pos, vel=vel, a_cmd=np.zeros(3),
                gps_pos=pos + np.array([0.05, 0.0, 0.0]),
                gps_vel=vel, imu_dr_pos=pos.copy(), imu_dr_vel=vel.copy(),
                mag_heading=0.0, gps_course=0.0, gps_signal_quality=1.0,
                baro_ok=True, battery_frac=initial.battery_frac,
                energy_required_frac=initial.energy_required_frac,
                wind_est=initial.wind_est,
                obstacles=[obstacle] if obstacle is not None else [],
            )
            if state_mod is not None:
                st = state_mod(k, st)
            dec = self.brain.decide(st, desire)
            hist[dec.mode] = hist.get(dec.mode, 0) + 1
            if dec.threat:
                max_threat = max(max_threat, dec.threat.score)
            if dec.evasion:
                final_clear = min(final_clear, dec.evasion.min_clearance)
            declared |= set(dec.declared_used)
            undeclared |= set(dec.undeclared_used)
            compute += dec.compute_ms
            energy += dec.energy_budget_mw
            # Bounded velocity trim from the evade decision.
            if dec.mode == EVADE and dec.evasion is not None:
                vel = vel + 0.18 * dec.evasion.offset_vel
            pos = pos + vel * step_dt
            # collision / world-bound check
            if obstacle is not None:
                d = float(np.linalg.norm(pos - obstacle.pos)) - obstacle.radius
                if d < 0.0:
                    crashed = True
            if float(np.linalg.norm(pos)) > 50.0:
                crashed = True
        return ScenarioMetrics(
            mode_histogram=hist, max_threat_reached=max_threat,
            final_clearance=final_clear, crashed=crashed,
            declared_used=declared, undeclared_used=undeclared,
            mean_compute_ms=compute / max(n_steps, 1),
            mean_energy_mw=energy / max(n_steps, 1),
        )

    def evaluate(self, states: list[GuardianState],
                 desire: np.ndarray) -> ScenarioMetrics:
        hist: dict[str, int] = {}
        max_threat = 0.0
        final_clear = 1e9
        declared: set[str] = set()
        undeclared: set[str] = set()
        compute = 0.0
        energy = 0.0
        crash = False
        for idx, st in enumerate(states):
            obs_list = st.obstacles
            obstacle = obs_list[0] if obs_list else None
            dec, nxt = self.run_step(st, desire, obstacle)
            hist[dec.mode] = hist.get(dec.mode, 0) + 1
            if dec.threat:
                max_threat = max(max_threat, dec.threat.score)
            if dec.evasion:
                final_clear = min(final_clear, dec.evasion.min_clearance)
            declared |= set(dec.declared_used)
            undeclared |= set(dec.undeclared_used)
            compute += dec.compute_ms
            energy += dec.energy_budget_mw
            n = float(np.linalg.norm(nxt.pos))
            if n > 40.0:
                crash = True
        return ScenarioMetrics(
            mode_histogram=hist, max_threat_reached=max_threat,
            final_clearance=final_clear, crashed=crash,
            declared_used=declared, undeclared_used=undeclared,
            mean_compute_ms=compute / max(len(states), 1),
            mean_energy_mw=energy / max(len(states), 1),
        )
