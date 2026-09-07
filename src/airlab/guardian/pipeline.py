"""Telemetry data pipeline (edge -> storage -> analytics).

Program priority #5.  The learned risk prior (brief-27) and the thermal model
(brief-28) currently work from *simulated* labelled telemetry and modelled
constants.  This module turns the real simulator flight into a **reproducible
dataset**: it records the fused telemetry every frame, and exposes a table the
risk prior and thermal calibrator can fit on.

Design principles:
  * downstream-first: the recorded schema is the exact feature set the risk
    prior and thermal health engine consume (no ad-hoc dumps).
  * timestamps + integrity metadata (master prompt §34).
  * numpy/CSV only — no extra dependency, inspectable rows.
  * data is recorded from what the aircraft actually experiences (fused state,
    power, node temperatures, threat residuals), not from hand-picked vectors.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from .risk_prior import RiskPriorModel

# Canonical schema of the analytics table.  ``field`` is the column name.
_FIELDS = [
    "t_s", "time", "pos_n", "pos_e", "pos_d", "vel_n", "vel_e", "vel_d",
    "throttle", "power_w", "battery_frac", "temp_c", "cpu_npu_c", "esc_c",
    "motor_c", "battery_c", "motor_resid", "vib", "gps_disagree",
    "flow_disagree", "landmark_residual", "factorgraph_residual",
    "land", "mode", "reason",
]


@dataclass
class TelemetryDataset:
    rows: list[dict] = field(default_factory=list)
    path: str = ""
    meta: dict = field(default_factory=dict)

    def append(self, row: dict) -> None:
        self.rows.append({k: row.get(k, "") for k in _FIELDS})

    def __len__(self) -> int:
        return len(self.rows)

    def to_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        self.path = path
        return path

    def as_arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        out = {}
        for k in _FIELDS:
            vals = [float(r[k]) for r in self.rows if str(r.get(k, "")) != ""]
            if vals:
                out[k] = np.asarray(vals, dtype=float)
        return out


class RiskTelemetryDataset:
    """Turntable of (features, label) for the learned risk prior.

    Each row is a *risk sample*:  ``dist_m`` = distance to the nearest
    projected obstacle, ``jam`` = jamming/GNSS degradation, ``label`` = the
    observed risk (copied from the actual risk field sample at that point, so
    the prior can be calibrated against the *seen* danger rather than the
    analytic formula).
    """

    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float]] = []
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def add(self, dist_m: float, jam: float, label: float) -> None:
        self.samples.append((float(dist_m), float(jam), float(label)))
        self.n += 1

    def as_array(self) -> np.ndarray:
        if not self.samples:
            return np.zeros((0, 3))
        return np.asarray(self.samples, dtype=float)

    def fit_prior(self, prior: RiskPriorModel | None = None) -> RiskPriorModel:
        p = prior or RiskPriorModel()
        if self.n:
            p.fit(self.as_array())
        return p


class DataPipeline:
    """Record a live Simulator into a TelemetryDataset + RiskTelemetryDataset.

    Usage:
        pipe = DataPipeline(sim)
        run = sim.run()          # pipe records inside the run loop
        ds = pipe.dataset.to_csv("out/guardian/telemetry.csv")
        prior = pipe.risk.fit_prior()
    """

    def __init__(self, sim, dataset: TelemetryDataset | None = None,
                 risk: RiskTelemetryDataset | None = None) -> None:
        self.sim = sim
        self.dataset = dataset or TelemetryDataset()
        self.risk = risk or RiskTelemetryDataset()
        self._last_obstacles: list = []
        self._last_jam: list = []
        self._risk_fields: list = []

    # ------------------------------------------------------------ recording
    def record(self, dt: float, obstacles=None, jamming_centers=None,
               risk_field=None) -> None:
        """Append one frame of live telemetry and one risk sample."""
        sim = self.sim
        row = _telemetry_row(sim)
        self.dataset.append(row)
        self._last_obstacles = list(obstacles or [])
        self._last_jam = list(jamming_centers or [])
        self._risk_fields.append(risk_field)
        self._add_risk_sample(row, obstacles or [], jamming_centers or [],
                              risk_field)

    def _add_risk_sample(self, row: dict, obstacles, jamming_centers,
                         risk_field) -> None:
        p = np.array([row["pos_n"], row["pos_e"], row["pos_d"]], dtype=float)
        dist = 12.0
        if obstacles:
            d = min(float(np.linalg.norm(p - np.asarray(o.pos, dtype=float)))
                    for o in obstacles)
            dist = float(np.clip(d, 0.0, 12.0))
        # Jamming feature: use actual GNSS degradation first, then the reverse
        # distance to any declared jamming corridor.  A healthy run that simply
        # crosses a jam corridor still sees the corridor as high risk.
        jam = float(row.get("gps_disagree", 0.0))
        if jamming_centers:
            d_j = min(float(np.linalg.norm(p - np.asarray(c, dtype=float)))
                      for c in jamming_centers)
            jam = max(jam, float(np.exp(-0.5 * (d_j / 4.0) ** 2)))
        jam = float(np.clip(jam, 0.0, 1.0))
        label = risk_field.sample(p) if risk_field is not None else 0.0
        self.risk.add(dist, jam, float(label))


def _telemetry_row(sim) -> dict:
    b = getattr(sim, "guardian_health_bridge", None)
    th = getattr(b, "thermal", None)
    node_temps = th.temperatures() if th is not None else {}
    return {
        "t_s": float(sim.time),
        "time": float(sim.time),
        "pos_n": float(sim.ekf.pos[0]),
        "pos_e": float(sim.ekf.pos[1]),
        "pos_d": float(sim.ekf.pos[2]),
        "vel_n": float(sim.ekf.vel[0]),
        "vel_e": float(sim.ekf.vel[1]),
        "vel_d": float(sim.ekf.vel[2]),
        "throttle": float(sim.last_control[0]),
        "power_w": float(sim.power_model.power(sim.last_control[0])),
        "battery_frac": float(sim._battery_frac()),
        "temp_c": float(node_temps.get("cpu_npu", sim.cfg.thermal_ambient_c)),
        "cpu_npu_c": float(node_temps.get("cpu_npu", 0.0)),
        "esc_c": float(node_temps.get("esc", 0.0)),
        "motor_c": float(node_temps.get("motor", 0.0)),
        "battery_c": float(node_temps.get("battery", 0.0)),
        "motor_resid": float(getattr(b, "_motor_resid_ema", 0.0)),
        "vib": float(getattr(b, "_vib_ema", 0.0)),
        "gps_disagree": float(getattr(sim, "_landmark_residual", 0.0)),
        "flow_disagree": float(sim._last_flow_mismatch),
        "landmark_residual": float(getattr(sim, "_landmark_residual", 0.0)),
        "factorgraph_residual": float(getattr(sim, "_factorgraph_residual", 0.0)),
        "land": float(getattr(sim, "safety", None).state if hasattr(
            getattr(sim, "safety", None), "state") else 1.0),
        "mode": str(getattr(getattr(sim, "safety", None), "mode", "")),
        "reason": str(getattr(getattr(sim, "safety", None), "reason", "")),
    }
