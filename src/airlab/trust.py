"""Per-frame informative-frame trust.

The analytic trust in :meth:`Simulator._detector_trust` is hand set
(``count/3``, ``rms/1.2``).  This module replaces those constants with a
*self-calibrated* model: during a short startup window the aircraft collects the
frame-*informativeness* signals it actually sees on a healthy run, then learns a
robust low reference and a bandwidth, plus a reference landmark count.  Every
later frame is scored against that learned healthy distribution, so a
degenerate-parallax scene (many features in a tight cone) is recognised as thin
even though its raw count is high.

The learner only measures *could this frame inform a geometric check?* — it
never uses the detector's own verdict, so a faulty detector cannot down-weight
itself (the same design-in rule as the availability weights).

Numpy-only, fully inspectable.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -30.0, 30.0)))))


class FrameTrustLearner:
    """Learn a reference "informative frame" distribution from startup data.

    Parameters
    ----------
    low_q / high_q :
        robust percentiles of the healthy calibration samples used to derive the
        reference (``ref``) and bandwidth (``band``).
    min_samples :
        must collect at least this many healthy samples before the model is
        considered calibrated.
    """

    def __init__(self, kind: str, low_q: float = 25.0, high_q: float = 90.0,
                 min_samples: int = 8) -> None:
        self.kind = kind
        self.low_q = float(low_q)
        self.high_q = float(high_q)
        self.min_samples = int(min_samples)
        self._samples: list[float] = []
        self._counts: list[float] = []
        self.calibrated = False
        self.ref: float | None = None
        self.band: float | None = None
        self.offset: float = 0.0
        self.count_ref: float | None = None
        self.n_calibrated = 0

    def calibrate(self, x: float, count: float | None = None) -> None:
        """Feed one *healthy* frame informativeness sample (no faults active).

        ``x`` is the raw signal (e.g. RMS pairwise angle for landmarks, factor
        conditioning for the graph).  ``count`` is the raw evidence count (e.g.
        number of landmarks) used to learn a count reference.
        """
        if self.calibrated or np.isnan(x):
            return
        self._samples.append(float(x))
        if count is not None:
            self._counts.append(float(count))
        if len(self._samples) >= self.min_samples:
            self._finalize()

    def _finalize(self) -> None:
        s = np.asarray(self._samples, dtype=float)
        lo = float(np.percentile(s, self.low_q))
        hi = float(np.percentile(s, self.high_q))
        # A degenerate scene -> rms ~ 0.04; a healthy spread field -> rms ~ 1.0.
        # A unit at the healthy LOW percentile should already be clearly usable
        # (target ~0.75), and a unit at the low_percentile - bandwidth should be
        # near the thin boundary (~0.12).  Choose the sigmoid offset so the low
        # percentile maps to the target and the bandwidth is (high-low)/2.  The
        # guard keeps a zero-width training window from producing a degenerate
        # flat line.
        span = max(float(hi - lo), max(abs(lo) * 0.05, 0.03))
        target_low = 0.75
        target_hi = 0.95
        self.ref = float(lo)
        # bandwidth such that the high percentile maps to target_hi and low maps
        # to target_low.
        z_lo = np.log(target_low / (1.0 - target_low))
        z_hi = np.log(target_hi / (1.0 - target_hi))
        self.offset = float(z_lo)
        self.band = float(span / max(z_hi - z_lo, 1e-6))
        # Count reference: robust median of the healthy evidence count.  This
        # replaces the hand-set /3 (min useful landmarks); it adapts to whatever
        # the camera actually sees on a healthy run.
        if self._counts:
            self.count_ref = float(np.median(np.asarray(self._counts)))
        else:
            self.count_ref = 3.0
        self.calibrated = True
        self.n_calibrated = len(s)

    def trust(self, x: float, count: float | None = None) -> float | None:
        """Map a frame signal to trust in [0,1]; None if not yet calibrated."""
        if not self.calibrated or self.ref is None or self.band is None:
            return None
        z = (float(x) - self.ref) / self.band + self.offset
        trust = float(np.clip(_sigmoid(z), 0.0, 1.0))
        if count is not None and self.count_ref:
            count_frac = float(np.clip(float(count) / self.count_ref, 0.0, 1.0))
            trust *= count_frac
        return float(np.clip(trust, 0.0, 1.0))

    @property
    def param_summary(self) -> str:
        if not self.calibrated:
            return f"{self.kind}:uncalibrated({len(self._samples)})"
        crefs = "" if self.count_ref is None else f",count_ref={self.count_ref:.2f}"
        return (f"{self.kind}:ref={self.ref:.3f},band={self.band:.3f},"
                f"n={self.n_calibrated}{crefs}")
