"""Bridge live simulator telemetry into the predictive-maintenance engine.

Implements program priority #2: ``SubsystemHealth`` should score *real* fused
flight-stack signals, not synthetic numpy features.  This module mines the
``Simulator``'s live state (battery fraction from the energy model, motor
throttle-to-achieved-thrust residual, a modelled lumped temperature, the
EKF/vehicle acceleration jitter, and cross-sensor disagreement) so the health
score is computed from what the aircraft actually experiences.

Safety: this is *diagnostic*.  It reads telemetry and emits health scores; the
guardian's normal (ABORT / EVADE / RECOVER_NAV / SILENT) decisions remain the
authority.  Nothing here ever commands the aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .health import SubsystemHealth, HealthPrognosis


@dataclass
class ThermalState:
    """First-order lumped thermal model driven by real power draw.

    This is a *model*, not a measurement: it gives the health engine a
    physically consistent temperature from the platform's actual power budget.
    Parameters follow a small 500 g-class edge compute + payload stack; the
    units are W, J/K, and 1/s.
    """

    ambient_c: float = 25.0
    capacitance_jpk: float = 1200.0      # J/K lumped thermal mass
    thermal_conductance_wpk: float = 2.2  # W/K to ambient
    power_idle_w: float = 6.0            # edge + avionics baseline
    temp_c: float = 25.0

    def step(self, power_w: float, dt: float) -> float:
        p = float(power_w) + self.power_idle_w
        # dT = (Q_in - Q_out)/C ; Q_out = k*(T - T_amb)
        q = p - self.thermal_conductance_wpk * (self.temp_c - self.ambient_c)
        self.temp_c += float(q) / self.capacitance_jpk * dt
        return self.temp_c


class TelemetryHealthBridge:
    """Feed a live ``Simulator`` into the guardian's health engine."""

    def __init__(self, sim, health: SubsystemHealth | None = None,
                 prognosis: HealthPrognosis | None = None,
                 thermal: ThermalState | None = None,
                 warmup_samples: int = 200) -> None:
        self.sim = sim
        self.health = health or SubsystemHealth(warmup_samples=warmup_samples)
        self.prognosis = prognosis or HealthPrognosis()
        self.thermal = thermal or ThermalState(
            ambient_c=float(sim.cfg.thermal_ambient_c))
        self._prev_est_vel = None
        self._vib_ema = 0.0
        self._motor_resid_ema = 0.0
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return self._history

    def step(self, dt: float) -> float:
        """Sample one frame of telemetry and return aggregate health."""
        sim = self.sim
        battery_frac = sim._battery_frac()
        motor_resid = self._motor_residual(sim, dt)
        gps_disagree = self._gps_disagree(sim)
        flow_disagree = float(sim._last_flow_mismatch)
        temp_c = self.thermal.step(sim.last_control[0], dt)
        vib = self._vibration(sim, dt)
        self.health.update(battery_frac, motor_resid, temp_c, vib,
                           gps_disagree, flow_disagree)
        agg = self.prognosis.aggregate(self.health.scores())
        self._history.append({
            "battery_frac": battery_frac,
            "motor_resid": motor_resid,
            "temp_c": temp_c,
            "vib": vib,
            "gps_disagree": gps_disagree,
            "flow_disagree": flow_disagree,
            "health": agg,
        })
        return agg

    def _motor_residual(self, sim, dt: float) -> float:
        """Throttle-efficiency observable: how much of the *commanded* thrust
        actually becomes lift.

        For a near-level quad ``g - a_z(ned)`` estimates the delivered thrust
        per unit mass.  A healthy motor therefore needs ``cmd ≈ g`` and the
        efficiency estimate is ≈ 1.  When ``motor_efficiency`` falls, the
        controller commands *more* throttle to hold altitude, so the estimated
        efficiency (and hence the residual ``|1 - eff|``) grows.  This is the
        observable a maintenance engine can use with only throttle + IMU.
        """
        g = 9.80665
        cmd = float(sim.last_control[0])
        a_ned_z = float(sim.vehicle.a_ned[2]) if hasattr(sim, "vehicle") else 0.0
        delivered = g - a_ned_z if abs(a_ned_z) < 1.5 * g else g
        eff = delivered / cmd if cmd > 1e-6 else 0.0
        resid = float(abs(1.0 - eff))
        # exponential smoothing so the health engine sees a stable trend
        self._motor_resid_ema = (0.90 * self._motor_resid_ema +
                                 0.10 * resid)
        return self._motor_resid_ema

    def _vibration(self, sim, dt: float) -> float:
        accel = np.asarray(sim.vehicle.a_ned, dtype=float)
        if self._prev_est_vel is None:
            self._prev_est_vel = accel.copy()
            return 0.0
        jitter = float(np.linalg.norm(accel - self._prev_est_vel))
        self._prev_est_vel = accel.copy()
        self._vib_ema = (0.90 * self._vib_ema + 0.10 * jitter)
        return self._vib_ema

    def _gps_disagree(self, sim) -> float:
        deg = 0.0
        if not sim._gps_available():
            deg = max(deg, 0.5)
        # cross-sensor disagreement: the guardian spoof detector uses
        # GPS-versus-IMU dead-reckon divergence.  Here we use the cheap,
        # always-available proxy: GPS/baro health vs the EKF position
        # covariance-driven uncertainty is already in the sim; for the health
        # engine we expose a smooth signal from actual GNSS availability and
        # the landmark detector's residual.
        lg = getattr(sim, "_landmark_residual", 0.0)
        fg = getattr(sim, "_factorgraph_residual", 0.0)
        return float(np.clip(deg + 0.25 * min(float(lg), 1.0)
                             + 0.10 * min(float(fg), 1.0), 0.0, 1.0))
