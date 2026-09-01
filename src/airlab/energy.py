"""Very small power/propulsion energy model for mission-level studies.

This is intentionally *not* a motor/propeller aero model.  It translates the
recorded specific-thrust command into a scalar power draw so energy can be
added to the mission metrics without pretending to be high fidelity.  The
power curve has a hover point and a super-linear term for high thrust, which is
the qualitative behaviour of a rotorcraft.

Units
-----
- ``specific_thrust``: N/kg (= m/s^2 of body- -z acceleration)
- ``power``: W
- ``energy``: Wh
"""

from __future__ import annotations

import numpy as np

G = 9.80665


class PowerModel:
    def __init__(
        self,
        p_hover: float = 120.0,     # W at hover
        thrust_hover: float = 9.80665,  # N/kg at hover
        thrust_exponent: float = 1.7,
        battery_capacity_wh: float = 100.0,
    ) -> None:
        self.p_hover = p_hover
        self.thrust_hover = thrust_hover
        self.thrust_exponent = thrust_exponent
        self.capacity_wh = battery_capacity_wh

    def power(self, specific_thrust: float) -> float:
        t = max(float(specific_thrust), 0.0)
        ratio = t / self.thrust_hover
        return self.p_hover * (ratio ** self.thrust_exponent)

    def energy(self, thrust_history: np.ndarray, dt: float) -> float:
        """Return total energy consumed (Wh) over a recorded thrust history."""
        thrust_history = np.asarray(thrust_history, dtype=float)
        if thrust_history.size == 0:
            return 0.0
        P = np.array([self.power(t) for t in thrust_history])
        return float(np.sum(P) * dt / 3600.0)

    def battery_discharge_frac(self, thrust_history: np.ndarray, dt: float) -> float:
        return self.energy(thrust_history, dt) / self.capacity_wh


DEFAULT_POWER = PowerModel()


def compute_energy(run, dt: float, model: PowerModel | None = None) -> dict:
    """Compute energy metrics from a SimRun's recorded throttle commands."""
    model = model if model is not None else DEFAULT_POWER
    if getattr(run, "control", None) is None or len(run.control) == 0:
        return {
            "energy_wh": 0.0,
            "power_mean_w": 0.0,
            "battery_used_frac": 0.0,
            "battery_remaining_frac": 1.0,
        }
    thrust = np.asarray(run.control, dtype=float)[:, 0]
    energy_wh = model.energy(thrust, dt)
    power_mean_w = float(np.mean([model.power(t) for t in thrust]))
    used_frac = model.battery_discharge_frac(thrust, dt)
    energy_left = max(0.0, model.capacity_wh - energy_wh)
    return {
        "energy_wh": energy_wh,
        "power_mean_w": power_mean_w,
        "battery_used_frac": used_frac,
        "battery_remaining_frac": energy_left / model.capacity_wh if model.capacity_wh > 0 else 0.0,
    }
