"""Predictive maintenance: subsystem health scores from telemetry.

Master prompt §35: from data detect motor degradation, battery aging, vibration
anomalies, temperature anomalies, sensor degradation → a **Health Score** per
subsystem.

Every score is computed against a *calibrated baseline* that is learned from the
first healthy window (same design-in trust approach as frame trust).  This keeps
the maintenance engine transparent and avoids a black-box threshold.

Subsystems:
  battery  - voltage/energy trend vs healthy discharge
  motor    - throttle-to-speed residual drift (proxy for efficiency loss)
  thermal  - temperature anomaly
  vib      - vibration proxy (e.g. throttle jitter / accel residual)
  sensor   - each independent sensor's disagreement time-fraction
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -30.0, 30.0)))))


@dataclass
class HealthScore:
    subsystem: str
    score: float                 # 0..1 (1 healthy)
    trend: float = 0.0           # per-second slope of the score
    status: str = "ok"           # ok / watch / warn / critical
    evidence: str = ""

    @property
    def degraded(self) -> bool:
        return self.score < 0.55


class SubsystemHealth:
    """Collect per-frame features and emit subsystem health scores."""

    def __init__(self, warmup_samples: int = 30) -> None:
        self.warmup_samples = warmup_samples
        self._bat_volt: list[float] = []
        self._motor_resid: list[float] = []
        self._temp: list[float] = []
        self._vib: list[float] = []
        self._sensor_gps: list[float] = []
        self._sensor_flow: list[float] = []
        # baselines learned during warmup
        self._bat_ref: float | None = None
        self._motor_p95: float | None = None
        self._temp_ref: float | None = None
        self._vib_p95: float | None = None
        self.calibrated = False

    def update(self, battery_frac: float, motor_resid_abs: float,
               temp_c: float, vib: float, gps_disagree: float,
               flow_disagree: float) -> None:
        self._bat_volt.append(float(battery_frac))
        self._motor_resid.append(float(motor_resid_abs))
        self._temp.append(float(temp_c))
        self._vib.append(float(vib))
        self._sensor_gps.append(float(gps_disagree))
        self._sensor_flow.append(float(flow_disagree))
        if len(self._bat_volt) >= self.warmup_samples and not self.calibrated:
            self._finalize()

    def _finalize(self) -> None:
        self._bat_ref = float(np.mean(np.asarray(self._bat_volt[-self.warmup_samples:])))
        self._motor_p95 = float(np.percentile(self._motor_resid[-self.warmup_samples:], 95))
        if self._motor_p95 < 1e-6:
            self._motor_p95 = 1e-6
        self._temp_ref = float(np.mean(self._temp[-self.warmup_samples:]))
        self._vib_p95 = float(np.percentile(self._vib[-self.warmup_samples:], 95))
        if self._vib_p95 < 1e-6:
            self._vib_p95 = 1e-6
        self.calibrated = True

    def scores(self) -> list[HealthScore]:
        if not self.calibrated or self._bat_ref is None:
            return [HealthScore("system", 1.0, status="ok",
                                evidence="warming_up")]

        n = len(self._bat_volt)
        out: list[HealthScore] = []
        # Battery: the reference is the healthy discharge level.  A normal
        # few-percent mission discharge must not read "warn"; only a drop of
        # ~10 % (deep discharge / sag / aging) starts to matter.
        last = float(self._bat_volt[-1])
        drop = max(0.0, self._bat_ref - last)
        bat_score = float(np.clip(1.0 - _sigmoid((drop - 0.10) / 0.08), 0.0, 1.0))
        t = self._trend(self._bat_volt, 0.25)
        out.append(HealthScore("battery", bat_score, trend=t,
                               status=_status(bat_score),
                               evidence=f"ref={self._bat_ref:.3f},last={last:.3f}"))

        # Motor: residual (throttle-to-speed drift) scaled by health p95.
        last_m = float(self._motor_resid[-1])
        m_ratio = last_m / self._motor_p95
        m_score = float(np.clip(1.0 - _sigmoid((m_ratio - 2.0) / 1.0), 0.0, 1.0))
        out.append(HealthScore("motor", m_score, trend=self._trend(self._motor_resid, 0.0),
                               status=_status(m_score),
                               evidence=f"p95={self._motor_p95:.4f},last={last_m:.4f}"))

        # Thermal: generic °C anomaly around baseline.
        last_t = float(self._temp[-1])
        dt = max(0.0, last_t - self._temp_ref)
        t_score = float(np.clip(1.0 - _sigmoid((dt - 12.0) / 8.0), 0.0, 1.0))
        out.append(HealthScore("thermal", t_score, trend=self._trend(self._temp, 0.0),
                               status=_status(t_score),
                               evidence=f"ref={self._temp_ref:.1f}C,last={last_t:.1f}C"))

        # Vibration proxy.
        last_v = float(self._vib[-1])
        v_ratio = last_v / self._vib_p95
        v_score = float(np.clip(1.0 - _sigmoid((v_ratio - 3.0) / 1.5), 0.0, 1.0))
        out.append(HealthScore("vibration", v_score,
                               status=_status(v_score),
                               evidence=f"p95={self._vib_p95:.4f},last={last_v:.4f}"))

        # Sensors: disagreement time-fraction over the last 100 samples.
        out.append(self._sensor_score("gps", self._sensor_gps))
        out.append(self._sensor_score("flow", self._sensor_flow))
        return out

    def _sensor_score(self, name: str, series: list[float]) -> HealthScore:
        last = 0.0
        if series:
            last = max(0.0, float(series[-1]))
        score = float(np.clip(1.0 - _sigmoid((last - 0.2) / 0.1), 0.0, 1.0))
        return HealthScore(f"sensor_{name}", score, status=_status(score),
                           evidence=f"disagree={last:.3f}")

    def _trend(self, series: list[float], default: float) -> float:
        if len(series) < 5:
            return default
        x = np.arange(len(series), dtype=float)
        y = np.asarray(series[-min(50, len(series)):], dtype=float)
        xx = np.arange(len(y), dtype=float)
        denom = float(np.sum((xx - xx.mean()) ** 2))
        if denom < 1e-9:
            return default
        slope = float(np.sum((xx - xx.mean()) * (y - y.mean())) / denom)
        return slope / max(float(np.mean(y)), 1e-9)


def _status(score: float) -> str:
    if score >= 0.85:
        return "ok"
    if score >= 0.70:
        return "watch"
    if score >= 0.55:
        return "warn"
    return "critical"


class HealthPrognosis:
    """Aggregate subsystem scores into one maintenance-health signal."""

    def __init__(self) -> None:
        self.history: list[float] = []

    def aggregate(self, scores: list[HealthScore]) -> float:
        if not scores:
            return 1.0
        # min is the weakest-link health: any critical subsystem drags it down.
        agg = float(min(s.score for s in scores))
        self.history.append(agg)
        return agg

    @property
    def trend(self) -> float:
        if len(self.history) < 2:
            return 0.0
        y = np.asarray(self.history, dtype=float)
        xx = np.arange(len(y), dtype=float)
        denom = float(np.sum((xx - xx.mean()) ** 2))
        if denom < 1e-9:
            return 0.0
        return float(np.sum((xx - xx.mean()) * (y - y.mean())) / denom)


def simulated_features(rng, k: int, battery_bad: bool = False,
                       motor_bad: bool = False, thermal_bad: bool = False,
                       vib_bad: bool = False) -> tuple[float, float, float, float, float, float]:
    """Deterministic feature generator for tests / demos (numpy rand)."""
    t = float(k)
    base = np.array([
        0.9 - 0.0008 * k,                                   # battery frac
        abs(0.02 * np.sin(k * 0.1)) + rng.normal(0, 0.004),  # motor residual
        35.0 + 0.02 * k,                                     # temp C
        abs(0.05 * np.sin(k * 0.2)) + rng.normal(0, 0.01),   # vib
        0.0 + rng.normal(0, 0.01),                           # gps disagree
        0.0 + rng.normal(0, 0.01),                           # flow disagree
    ])
    if battery_bad:
        base[0] = 0.5 - 0.01 * k
    if motor_bad:
        base[1] = abs(base[1]) + 0.4
    if thermal_bad:
        base[2] = 35.0 + 0.5 * k
    if vib_bad:
        base[3] = abs(base[3]) + 0.8
    return (base[0], base[1], base[2], base[3], base[4], base[5])
