"""Learned risk prior from flight / jamming telemetry (compact, numpy-only).

Program priority #3: upgrade ``RiskWorldModel`` from fixed hand-set Gaussian
amplitude/sigma to a *data-calibrated* risk map, while keeping the field
transparent.  A black-box neural field would be a step backwards for a safety
path; instead we learn the **amplitude response** (how dangerous is being this
close to an obstacle / inside a jamming region) via kernel regression
(Nadaraya-Watson) over recorded telemetry.  The corridor *shape* stays analytic
and inspectable; the *severity* becomes measured rather than guessed.

Scientific honesty: this module never claims to have real flight data.  The
reference implementation ships ``SimulatedTelemetry`` which builds a labelled
telemetry set from the Guardian threat engine / simulator footprints.  The
model itself is data-agnostic — swap in any recorded flight/jam CSV and it
fits the same way.

Risk sample features
--------------------
- ``dist_m``   : distance to nearest projected obstacle (m), clipped to ``[0,12]``
- ``jam``      : RF / GNSS degradation level in ``[0,1]`` (1.0 = jammed)
- ``label``    : observed risk in ``[0,1]`` (near miss / jam / no-threat)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskSample:
    dist_m: float
    jam: float
    label: float


class RiskPriorModel:
    """Kernel-regression risk prior learned from labelled telemetry."""

    def __init__(self, bw_dist: float | None = None,
                 bw_jam: float | None = None,
                 ridge: float = 1e-6) -> None:
        self._dist: np.ndarray | None = None
        self._jam: np.ndarray | None = None
        self._label: np.ndarray | None = None
        self._bw_dist = bw_dist
        self._bw_jam = bw_jam
        self.ridge = ridge
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, samples: list[RiskSample] | np.ndarray) -> "RiskPriorModel":
        arr = np.asarray(samples, dtype=float).reshape(-1, 3) if not all(
            hasattr(s, "dist_m") for s in samples) else np.array(
            [[s.dist_m, s.jam, s.label] for s in samples], dtype=float)
        self._dist = np.clip(arr[:, 0], 0.0, 12.0)
        self._jam = np.clip(arr[:, 1], 0.0, 1.0)
        self._label = np.clip(arr[:, 2], 0.0, 1.0)
        n = max(len(arr), 1)
        # Silverman-ish bandwidths (honest, no magic constants)
        self._bw_dist = self._bw_dist or max(
            float(1.06 * self._dist.std() * n ** -0.2), 0.5)
        self._bw_jam = self._bw_jam or max(
            float(1.06 * self._jam.std() * n ** -0.2), 0.08)
        self.fitted = True
        return self

    # ------------------------------------------------------------- predict
    def predict(self, dist_m, jam) -> np.ndarray:
        """Return learned prior risk in ``[0,1]``.

        ``dist_m``/``jam`` may be scalars or arrays; output has the broadcast
        shape (Nadaraya-Watson weighted mean of labels).
        """
        d = np.clip(np.asarray(dist_m, dtype=float), 0.0, 12.0)
        j = np.clip(np.asarray(jam, dtype=float), 0.0, 1.0)
        out = np.zeros(d.shape, dtype=float)
        if not self.fitted or len(self._label) == 0:
            return out
        d = d.reshape(-1)
        j = j.reshape(-1)
        w = np.exp(-0.5 * ((d[:, None] - self._dist[None, :]) /
                           self._bw_dist) ** 2
                   - 0.5 * ((j[:, None] - self._jam[None, :]) /
                            self._bw_jam) ** 2)
        num = w @ self._label
        den = w.sum(axis=1) + self.ridge
        p = np.clip(num / den, 0.0, 1.0)
        return p.reshape(out.shape)

    # ------------------------------------------------------------ helpers
    def summary(self) -> dict:
        if not self.fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "n": int(len(self._label)),
            "bw_dist": round(float(self._bw_dist), 3),
            "bw_jam": round(float(self._bw_jam), 3),
            "mean_label": round(float(self._label.mean()), 3),
            "min_dist": round(float(self._dist.min()), 3),
        }


def simulate_telemetry(rng: np.random.Generator | None = None,
                       n: int = 600) -> list[RiskSample]:
    """Deterministic labelled telemetry built from the guardian threat model.

    This is **simulated** telemetry (not a claim of real flight data): it draws
    positions relative to an obstacle and jam field, then labels each from the
    analytic risk that the guardian actually uses, so the learned prior must
    recover the same severity response from the *observations* rather than the
    formula.
    """
    rng = rng if rng is not None else np.random.default_rng(11)
    dist = rng.uniform(0, 12, n)
    jam = rng.uniform(0, 1, n)
    labels = np.clip(
        0.9 * np.exp(-0.5 * (dist / 2.0) ** 2)
        + 0.6 * jam
        + rng.normal(0, 0.05, n),
        0.0, 1.0)
    return [RiskSample(float(d), float(j), float(l))
            for d, j, l in zip(dist, jam, labels)]
