"""Real PX4 / ROS telemetry reader (data pipeline source).

This is the "real log reader" data-pipeline source (priority #6).  It reads
the standard export format that PX4 itself produces (``ulog2csv``), so it can
ingest a real flight log without needing a proprietary parser.  It also accepts
a merged ROS-style CSV (ROS bag CSV / simple topic-per-file directory).

Honest scope
------------
There are **no real PX4/ROS flight logs** in this repository.  This module is
the adapter; ``write_test_fixture()`` creates a *simulated* PX4-schema fixture
used only for tests/demos, explicitly labelled :data:`FIXTURE_LABEL`.  Swapping
in a real ``ulog2csv`` directory is exactly what the reader is built for.

Topic mapping (standard ``ulog2csv`` names)
-------------------------------------------
* ``vehicle_local_position``   -> pos_ned, vel_ned
* ``vehicle_attitude``         -> attitude reference (not used for risk)
* ``battery_status``           -> battery_frac (from remaining/scale if present)
* ``sensor_combined``          -> measured accel (vibration proxy)
* ``vehicle_global_position``  -> satellites_used / eph (GPS quality -> jam)
* ``estimator_status``         -> pos_horiz_accuracy / gps check -> jam
* ``distance_sensor``/``obstacle_distance`` -> dist_m to nearest obstacle
* ``system_usage``/``cpu_load`` -> compute_frac (thermal calibration)
* ``board_temperature``/``cpu_temperature`` -> thermal measurement (if present)

Any missing signal is left as ``NaN``; the downstream dataset/prior records
only rows with the features it needs.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from .pipeline import TelemetryDataset, RiskTelemetryDataset

FIXTURE_LABEL = "simulated PX4-schema fixture (not a real flight)"

# Column-name alias -> canonical feature key.
_ALIASES = {
    "timestamp": ["timestamp", "time", "time_stamp"],
    "pos_n": ["x", "pos_n", "north"],
    "pos_e": ["y", "pos_e", "east"],
    "pos_d": ["z", "pos_d", "down"],
    "vel_n": ["vx", "vel_n"],
    "vel_e": ["vy", "vel_e"],
    "vel_d": ["vz", "vel_d"],
    "battery_frac": ["remaining", "battery_frac", "battery_remaining"],
    "battery_voltage": ["voltage_v", "voltage", "voltage_v"],
    "dist_m": ["dist_m", "obstacle_distance", "distance", "range"],
    "satellites": ["satellites_used", "satellites"],
    "eph": ["eph", "pos_horiz_accuracy"],
    "compute_frac": ["cpu_load", "cpu_load_avg", "compute_frac"],
    "board_temp_c": ["board_temperature", "cpu_temperature", "temperature",
                     "board_temp_c"],
    "gps_dropout": ["gps_dropout", "gps_out", "gps_fix", "fix_type"],
    "risk_label": ["risk_label", "label", "threat_label"],
    "vel_n_m": ["vel_n", "velocity_n", "vel_m_s0"],
}


def _find_aliases(cols: list[str], aliases: list[str]) -> str | None:
    for c in cols:
        if c in aliases:
            return c
    return None


@dataclass
class Px4LogResult:
    telemetry: TelemetryDataset = field(default_factory=TelemetryDataset)
    risk: RiskTelemetryDataset = field(default_factory=RiskTelemetryDataset)
    meta: dict = field(default_factory=dict)

    def fit_prior(self, prior=None):
        """Fit the learned risk prior from the imported real datasets."""
        from .risk_prior import RiskPriorModel
        p = prior or RiskPriorModel()
        if self.risk.n:
            p.fit(self.risk.as_array())
        return p


class Px4RosLogReader:
    """Read a directory (topic-per-file from ``ulog2csv``) or a single merged
    CSV file and normalise it into guardian telemetry/risk datasets."""

    def __init__(self) -> None:
        self.meta = {"source": "none", "files": [], "rows": 0}

    # ------------------------------------------------------------- public
    def load(self, path: str) -> Px4LogResult:
        if os.path.isdir(path):
            return self._load_dir(path)
        return self._load_single(path)

    def _load_dir(self, path: str) -> Px4LogResult:
        result = Px4LogResult()
        self.meta = {"source": path, "files": [], "rows": 0, "label": "dir"}
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".csv"):
                continue
            full = os.path.join(path, fname)
            r = self._load_single(full)
            self.meta["files"].append(fname)
            for row in r.telemetry.rows:
                if (row.get("pos_n") != "" and row.get("pos_e") != ""):
                    result.telemetry.append(row)
            for s in r.risk.samples:
                result.risk.add(*s)
            result.meta = self.meta
        self.meta["rows"] = len(result.telemetry)
        result.meta = self.meta
        return result

    def _load_single(self, path: str) -> Px4LogResult:
        result = Px4LogResult()
        self.meta = {"source": path, "files": [os.path.basename(path)],
                     "rows": 0, "label": "single"}
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return result
            cols = [h.strip() for h in header]
            features = {k: _find_aliases(cols, aliases)
                        for k, aliases in _ALIASES.items()}
            n = 0
            for raw in reader:
                if len(raw) < len(cols):
                    continue
                d = dict(zip(cols, raw))
                if not any(d.get(c, "") not in ("", "nan", "NaN")
                           for c in (features["pos_n"] or [], features["pos_e"] or [])):
                    # still may have velocity-only rows; keep them for thermal
                    # but risk needs pos.
                    pass
                row = self._row_from(d, features)
                result.telemetry.append(row)
                self._maybe_risk(row, result.risk)
                n += 1
        self.meta["rows"] = len(result.telemetry)
        result.meta = self.meta
        return result

    # ----------------------------------------------------------- helpers
    def _num(self, d: dict, name: str | None, default: float = float("nan")):
        if name is None:
            return default
        v = d.get(name, "")
        try:
            return float(v)
        except ValueError:
            return default

    def _row_from(self, d: dict, features: dict) -> dict:
        row = {
            "t_s": self._num(d, features["timestamp"]),
            "time": self._num(d, features["timestamp"]),
            "pos_n": self._num(d, features["pos_n"]),
            "pos_e": self._num(d, features["pos_e"]),
            "pos_d": self._num(d, features["pos_d"]),
            "vel_n": self._num(d, features["vel_n"]),
            "vel_e": self._num(d, features["vel_e"]),
            "vel_d": self._num(d, features["vel_d"]),
            "dist_m": self._num(d, features["dist_m"]),
            "satellites": self._num(d, features["satellites"]),
            "eph": self._num(d, features["eph"]),
            "compute_frac": self._num(d, features["compute_frac"]),
            "board_temp_c": self._num(d, features["board_temp_c"]),
            "risk_label": self._num(d, features["risk_label"]),
        }
        return row

    def _maybe_risk(self, row: dict, risk: RiskTelemetryDataset) -> None:
        dist = row.get("dist_m", float("nan"))
        if not np.isfinite(dist):
            return
        jam = 0.0
        sat = row.get("satellites", float("nan"))
        if np.isfinite(sat):
            jam = max(jam, float(np.clip(1.0 - sat / 12.0, 0.0, 1.0)))
        eph = row.get("eph", float("nan"))
        if np.isfinite(eph):
            jam = max(jam, float(np.clip(eph / 3.0, 0.0, 1.0)))
        # A real logger has no ground-truth risk label, so by default we emit a
        # NaN label (input distribution only, never a silent calibration).  A
        # fixture can supply ``risk_label`` so the supervised path is tested.
        label = row.get("risk_label", float("nan"))
        risk.add(float(dist), float(jam), float(label))


def write_test_fixture(path: str, n: int = 120,
                       with_risk_label: bool = False) -> str:
    """Write a *simulated* PX4-schema fixture for tests/demos.

    Explicitly labelled ``FIXTURE_LABEL`` so no one mistakes it for a real
    flight.  It is a single merged CSV modelled on PX4's ``ulog2csv`` topic
    columns, with an obstacle-distance column so the risk reader is exercised.
    ``with_risk_label=True`` adds a simulated analytical ``risk_label`` so the
    supervised prior path can be tested (still simulated, never real).
    """
    rng = np.random.default_rng(21)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["timestamp", "x", "y", "z", "vx", "vy", "vz", "remaining",
            "satellites_used", "eph", "cpu_load", "obstacle_distance"]
    if with_risk_label:
        cols.append("risk_label")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(n):
            x = 0.2 * k
            dist = float(np.clip(abs(np.sin(k * 0.15)) * 6.0, 0.05, 12.0))
            sat = 12 if k < 60 else 3
            eph = 0.2 if k < 60 else 1.5
            jam = float(np.clip(max((1.0 - sat / 12.0), eph / 3.0), 0.0, 1.0))
            row = [
                k * 0.05 * 1e6,                      # timestamp (us)
                x, 0.0 + 0.2 * np.sin(k * 0.1), -2.0,
                1.0, 0.0, 0.0,
                max(0.2, 1.0 - 0.002 * k),           # remaining
                sat,
                eph,
                0.3 if k < 60 else 0.9,              # cpu_load
                dist,
            ]
            if with_risk_label:
                # simulated analytic label: close obstacle + jam => risk
                lbl = float(np.clip(0.9 * np.exp(-0.5 * (dist / 2.0) ** 2)
                                    + 0.6 * jam, 0.0, 1.0))
                row.append(lbl)
            w.writerow(row)
    return path
