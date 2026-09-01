"""Scenario generation for automated UAV experiments.

A scenario is a compact, reproducible description of a simulation run:
waypoints, wind, sensor noise scales, faults, and mission options.  The goal is
to turn the simulator into a *data factory*: a scenario defines an experiment,
and the batch runner produces a table of fully-labelled outcomes.

This module deliberately keeps scenarios serialisable (plain dicts / CSV) so
they can later be stored next to telemetry files and replayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from .simulator import SimConfig, Simulator
from .metrics import evaluate_mission, safety_metrics


@dataclass
class Scenario:
    """One reproducible experiment descriptor."""

    name: str = "scenario"
    duration: float = 40.0
    dt: float = 0.01

    # Mission
    waypoints: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (0.0, 0.0, 2.0),
        (8.0, 0.0, 3.0),
        (8.0, 8.0, 4.0),
        (-2.0, 8.0, 5.0),
        (-2.0, -2.0, 5.0),
    ])
    cruise_speed: float = 2.0

    # Environment / disturbances
    wind_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gps_outage: Optional[tuple[float, float]] = None

    # Sensor quality / fault scales
    gps_noise_scale: float = 1.0
    flow_sigma: float = 0.05
    flow_bias_walk: float = 0.002
    baro_drift: float = 0.01
    gyro_bias_scale: float = 1.0
    accel_bias_scale: float = 1.0
    imu_bias_ramp: float = 0.0

    # Vehicle
    mass: float = 1.3
    drag: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.20, 0.15]))

    # Sensor gating / corruption
    flow_enabled: bool = True
    flow_outage: Optional[tuple[float, float]] = None
    flow_scale: float = 1.0
    flow_bias_ramp: float = 0.0

    # Uncertainty-aware safety layer + independent landmark detector
    safety_enabled: bool = True
    landmark_enabled: bool = True

    def to_config(self) -> SimConfig:
        cfg = SimConfig()
        cfg.duration = self.duration
        cfg.dt = self.dt
        cfg.gps_outage = self.gps_outage

        cfg.wind_ned = np.asarray(self.wind_ned, dtype=float).reshape(3)
        cfg.waypoints = list(self.waypoints)
        cfg.cruise_speed = self.cruise_speed
        cfg.flow_enabled = self.flow_enabled
        cfg.flow_outage = self.flow_outage
        cfg.flow_scale = self.flow_scale
        cfg.flow_bias_ramp = self.flow_bias_ramp
        cfg.safety_enabled = self.safety_enabled
        cfg.landmark_enabled = self.landmark_enabled

        # Build a SensorConfig from the serialisable descriptor.
        cfg.sensor_kwargs = {
            "config": _sensor_config_from_scenario(self),
        }

        # vehicle
        cfg.vehicle_kwargs = {
            "mass": self.mass,
            "drag_linear": np.asarray(self.drag, dtype=float).reshape(3),
        }

        cfg.seed = int(abs(hash(self.name)) % (2**32))
        return cfg

    def as_dict(self) -> dict:
        d = asdict(self)
        d["wind_ned"] = list(np.asarray(self.wind_ned, dtype=float))
        d["drag"] = list(np.asarray(self.drag, dtype=float))
        return d


def _sensor_config_from_scenario(s: Scenario):
    """Create a SensorConfig with fields matched to the scenario descriptor."""
    from .sensors import SensorConfig

    c = SensorConfig()
    c.gps_noise_scale = s.gps_noise_scale
    c.flow_sigma = s.flow_sigma
    c.flow_bias_walk = s.flow_bias_walk
    c.baro_drift = s.baro_drift
    c.gyro_bias_init = s.gyro_bias_scale * c.gyro_bias_init
    c.accel_bias_init = s.accel_bias_scale * c.accel_bias_init
    c.imu_bias_ramp = s.imu_bias_ramp
    return c


def random_scenario(
    rng: np.random.Generator,
    duration: float = 40.0,
    index: int = 0,
    max_height: float = 8.0,
    max_range: float = 12.0,
    flow_enabled: bool = True,
) -> Scenario:
    """Generate a randomised but structurally sane mission scenario."""
    n_wp = int(rng.integers(3, 6))
    wps = [(0.0, 0.0, float(rng.uniform(1.5, 3.0)))]
    for _ in range(n_wp - 1):
        n = float(rng.uniform(-max_range, max_range))
        e = float(rng.uniform(-max_range, max_range))
        h = float(rng.uniform(2.0, max_height))
        wps.append((n, e, h))

    # Random wind (m/s-ish of acceleration)
    wind = rng.normal(0.0, 0.15, 3)
    wind[2] = abs(wind[2]) * 0.3

    # Random GPS outage with 60% probability (contiguous window)
    outage = None
    if rng.random() < 0.6:
        max_start = max(0.0, duration * 0.5)
        start = float(rng.uniform(duration * 0.2, max_start))
        end = float(rng.uniform(start + min(2.0, duration * 0.1),
                                min(duration, start + max(2.0, duration * 0.7))))
        outage = (min(start, end), max(start, end))

    s = Scenario(
        name=f"scen_{index:04d}",
        duration=duration,
        waypoints=wps,
        cruise_speed=float(rng.uniform(1.2, 2.5)),
        wind_ned=wind,
        gps_outage=outage,
        gps_noise_scale=float(rng.uniform(0.5, 2.0)),
        flow_sigma=float(rng.uniform(0.03, 0.12)),
        flow_bias_walk=float(rng.uniform(0.001, 0.008)),
        baro_drift=float(rng.uniform(0.005, 0.03)),
        gyro_bias_scale=float(rng.uniform(0.5, 2.0)),
        accel_bias_scale=float(rng.uniform(0.5, 2.0)),
        imu_bias_ramp=float(rng.uniform(0.0, 0.0002)),
        mass=float(rng.uniform(1.0, 1.7)),
        drag=np.array([
            *rng.uniform(0.12, 0.35, 2),
            rng.uniform(0.08, 0.25),
        ]),
        flow_enabled=flow_enabled,
    )
    # make sure the mission remains a useful loop
    s.name = f"scen_{index:04d}"
    return s


def run_scenario(s: Scenario, record: bool = True):
    """Run one scenario and return (metrics, SimRun).

    Metrics require recorded telemetry, so the ``record`` flag currently only
    documents whether the returned ``SimRun`` is expected to be useful; the
    internal recording is always enabled.
    """
    cfg = s.to_config()
    sim = Simulator(cfg)
    run = sim.run(record=True)
    m = evaluate_mission(
        run.true_pos, run.est_pos, run.ref_pos,
        run.ref_vel, run.est_vel, run.gps_available, cfg.dt,
    )
    m.update(safety_metrics(run, cfg.dt))
    return m, run


def scenario_energy(run, dt: float):
    """Simple power/energy estimate from recorded throttle commands."""
    from .energy import compute_energy
    return compute_energy(run, dt)
