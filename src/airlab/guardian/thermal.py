"""Part-level low-watt thermal model (CPU/NPU, ESC, motor, battery).

Program priority #4.  The preliminary ``ThermalState`` in ``telemetry_health``
was a single lumped mass.  That cannot answer the master prompt's thermal
question (which part can overheat first?).  This is a small transparent thermal
network:

  nodes = battery, motor, esc, cpu_npu (each generates heat) + frame (passive)
  each live node C*dT/dt = P_in - k_amb*(T - T_amb) - k_frame*(T - T_frame)
  frame C*dT/dt = sum(k_frame*(T_i - T_frame)) - k_amb_frame*(T_frame - T_amb)

Heat inputs are derived from the actual flight power and compute load:
  * motor  : fraction of delivered propulsive power (copper/iron loss)
  * esc    : fraction of propulsive power (switching loss)
  * battery: fraction of total electrical power (internal-resistance heat)
  * cpu_npu: idle + compute_frac * edge_power_max (the actual 0.35 TOPS edge)

All numbers are **modelled**, not measured; they are documented so an operator
can tune them to a specific board (scientific-honesty rule). No black box.
"""

from __future__ import annotations

import numpy as np


class ThermalNode:
    def __init__(self, name: str, capacity_jpk: float,
                 conductance_wpk: float, max_temp_c: float,
                 temp_c: float | None = None,
                 ambient_c: float = 25.0) -> None:
        self.name = name
        self.capacity_jpk = float(capacity_jpk)
        self.conductance_wpk = float(conductance_wpk)
        self.max_temp_c = float(max_temp_c)
        self.ambient_c = float(ambient_c)
        self.temp_c = float(temp_c if temp_c is not None else ambient_c)

    @property
    def temp_r(self) -> float:
        return self.temp_c

    @property
    def margin_c(self) -> float:
        return self.max_temp_c - self.temp_c

    @property
    def ok(self) -> bool:
        return self.temp_c < self.max_temp_c


class PartThermalModel:
    """Small thermal network for the Nexus edge + propulsion stack."""

    def __init__(self, ambient_c: float = 25.0,
                 frame_capacity_jpk: float = 1800.0,
                 frame_conductance_wpk: float = 3.0,
                 node_frame_conductance_wpk: float = 0.8,
                 edge_power_max_w: float = 8.0,
                 edge_power_idle_w: float = 2.0,
                 motor_loss_frac: float = 0.12,
                 esc_loss_frac: float = 0.05,
                 battery_loss_frac: float = 0.06,
                 initial_temps: dict[str, float] | None = None) -> None:
        self.ambient_c = float(ambient_c)
        self.frame_conductance_wpk = float(frame_conductance_wpk)
        self.node_frame_conductance_wpk = float(node_frame_conductance_wpk)
        self.edge_power_max_w = float(edge_power_max_w)
        self.edge_power_idle_w = float(edge_power_idle_w)
        self.motor_loss_frac = float(motor_loss_frac)
        self.esc_loss_frac = float(esc_loss_frac)
        self.battery_loss_frac = float(battery_loss_frac)
        ini = initial_temps or {}

        # Compact low-watt edge/NPU + propulsion part-level limits.  The edge
        # module has a small thermal mass and poor spread (lightweight, no
        # active cooling); the motor/ESC are the propulsion hot spots.  These
        # are **modelled-declared** numbers, tune to a specific board.
        self.frame = ThermalNode("frame", 1000.0, 1.8, 80.0,
                                 temp_c=float(ini.get("frame", ambient_c)),
                                 ambient_c=ambient_c)
        self.cpu_npu = ThermalNode("cpu_npu", 40.0, 0.5, 55.0,
                                   temp_c=float(ini.get("cpu_npu", ambient_c)),
                                   ambient_c=ambient_c)
        self.esc = ThermalNode("esc", 120.0, 0.8, 85.0,
                               temp_c=float(ini.get("esc", ambient_c)),
                               ambient_c=ambient_c)
        self.motor = ThermalNode("motor", 200.0, 0.5, 90.0,
                                 temp_c=float(ini.get("motor", ambient_c)),
                                 ambient_c=ambient_c)
        self.battery = ThermalNode("battery", 900.0, 1.0, 45.0,
                                   temp_c=float(ini.get("battery", ambient_c)),
                                   ambient_c=ambient_c)
        self.nodes = [self.cpu_npu, self.esc, self.motor, self.battery]

    # ------------------------------------------------------------- heat in
    def node_power(self, throttle_power_w: float,
                   compute_frac: float = 0.3) -> dict[str, float]:
        tp = max(float(throttle_power_w), 0.0)
        cf = float(np.clip(compute_frac, 0.0, 1.0))
        return {
            "cpu_npu": self.edge_power_idle_w + cf * self.edge_power_max_w,
            "esc": self.esc_loss_frac * tp,
            "motor": self.motor_loss_frac * tp,
            "battery": self.battery_loss_frac * tp,
        }

    def step(self, throttle_power_w: float, dt: float,
             compute_frac: float = 0.3) -> dict[str, float]:
        """Advance one control cycle; returns per-node temperatures (°C)."""
        plants = [self.cpu_npu, self.esc, self.motor, self.battery]
        pwr = self.node_power(throttle_power_w, compute_frac)
        ambient = self.ambient_c
        # update live nodes -> frame coupling
        for node in plants:
            p = pwr[node.name]
            k_frame = self.node_frame_conductance_wpk
            frame = self.frame.temp_c
            d = p - node.conductance_wpk * (node.temp_c - ambient) \
                - k_frame * (node.temp_c - frame)
            node.temp_c += d / node.capacity_jpk * dt
        # frame receives coupling from all live nodes
        inflow = sum(self.node_frame_conductance_wpk * (n.temp_c - self.frame.temp_c)
                     for n in plants)
        dframe = inflow - self.frame.conductance_wpk * (self.frame.temp_c - ambient)
        self.frame.temp_c += dframe / self.frame.capacity_jpk * dt
        return self.temperatures()

    def temperatures(self) -> dict[str, float]:
        return {n.name: float(n.temp_c) for n in self.nodes}

    def margins(self) -> dict[str, float]:
        return {n.name: float(n.margin_c) for n in self.nodes}

    def max_temp(self) -> float:
        return max(float(n.temp_c) for n in self.nodes)

    def worst_node(self) -> str:
        return max(self.nodes, key=lambda n: n.temp_c).name

    def status(self, warn_frac: float = 0.85) -> dict[str, str]:
        return {n.name: ("critical" if n.temp_c >= n.max_temp_c
                         else "warn" if n.temp_c >= warn_frac * n.max_temp_c
                         else "ok") for n in self.nodes}

    def summary(self) -> dict:
        t = self.temperatures()
        return {
            "worst": self.worst_node(),
            "max_temp_c": round(self.max_temp(), 1),
            "temperatures": {k: round(v, 1) for k, v in t.items()},
            "status": self.status(),
        }
