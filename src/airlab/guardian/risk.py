"""Oracle-style risk world model (transparent, numpy-only).

This is the "predict the threat corridor, not just dodge the next step" part of
the Nexus-Predator brain.  It builds a coarse 3-D risk field over the remaining
mission area from independent evidence:

  * static / dynamic obstacle projections (bounded Gaussians over the flight
    corridor),
  * jamming / spoofing corridors (areas where GPS was recently degraded),
  * an energy floor (feasibility is enforced by the replanner, not the field).

The model is *not* a black-box oracle.  It is a self-calibrated risk surrogate:
every contribution is a documented analytic field, so an operator can inspect
exactly why a route scored poorly (the same design-in/interpretability rule as
the rest of the stack).  We label it "oracle" only in the sense that it is the
planner's internal view of the world over the remaining mission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskField:
    origin: np.ndarray
    cell: float
    shape: tuple[int, int, int]
    risk: np.ndarray

    @property
    def xmin(self) -> float:
        return float(self.origin[0])

    @property
    def ymin(self) -> float:
        return float(self.origin[1])

    @property
    def zmin(self) -> float:
        return float(self.origin[2])

    def world_to_idx(self, p: np.ndarray) -> np.ndarray:
        idx = (np.asarray(p, dtype=float) - self.origin) / self.cell
        return idx

    def sample(self, p: np.ndarray) -> float:
        """Bilinear-ish sample of the risk field at a world point."""
        idx = self.world_to_idx(p)
        ix, iy, iz = idx
        if (ix < 0 or iy < 0 or iz < 0 or ix >= self.shape[0] - 1 or
                iy >= self.shape[1] - 1 or iz >= self.shape[2] - 1):
            return 0.0
        ix0, iy0, iz0 = int(np.floor(ix)), int(np.floor(iy)), int(np.floor(iz))
        fx, fy, fz = ix - ix0, iy - iy0, iz - iz0
        ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1
        r = (self.risk[ix0, iy0, iz0] * (1 - fx) * (1 - fy) * (1 - fz) +
             self.risk[ix1, iy0, iz0] * fx * (1 - fy) * (1 - fz) +
             self.risk[ix0, iy1, iz0] * (1 - fx) * fy * (1 - fz) +
             self.risk[ix1, iy1, iz0] * fx * fy * (1 - fz) +
             self.risk[ix0, iy0, iz1] * (1 - fx) * (1 - fy) * fz +
             self.risk[ix1, iy0, iz1] * fx * (1 - fy) * fz +
             self.risk[ix0, iy1, iz1] * (1 - fx) * fy * fz +
             self.risk[ix1, iy1, iz1] * fx * fy * fz)
        return float(np.clip(r, 0.0, None))


class RiskWorldModel:
    def __init__(self, cell: float = 1.0, obstacle_amp: float = 1.0,
                 obstacle_sigma: float = 2.0, jamming_amp: float = 0.7,
                 jamming_sigma: float = 4.0, horizon_s: float = 3.0,
                 sampling_step: float = 0.5, length_penalty: float = 0.02) -> None:
        self.cell = cell
        self.obstacle_amp = obstacle_amp
        self.obstacle_sigma = obstacle_sigma
        self.jamming_amp = jamming_amp
        self.jamming_sigma = jamming_sigma
        self.horizon_s = horizon_s
        self.sampling_step = sampling_step
        self.length_penalty = length_penalty

    def build(self, bounds: tuple[float, float, float, float, float, float],
              obstacles: list, jamming_centers: list | None = None) -> RiskField:
        """Build a risk field over ``bounds=(xmin,xmax,ymin,ymax,zmin,zmax)``.

        ``obstacles`` are objects with ``pos`` and ``vel`` (dynamic projection
        over ``horizon_s``).  ``jamming_centers`` are world points where GPS/
        RF degradation was observed.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        nx = max(2, int(np.ceil((xmax - xmin) / self.cell)))
        ny = max(2, int(np.ceil((ymax - ymin) / self.cell)))
        nz = max(2, int(np.ceil((zmax - zmin) / self.cell)))
        origin = np.array([xmin, ymin, zmin], dtype=float)
        field = np.zeros((nx, ny, nz), dtype=float)

        if obstacles:
            # project dynamic obstacles along their velocity for the horizon
            inner = self.horizon_s
            steps = max(1, int(round(inner / max(self.horizon_s / 6.0, 0.25))))
            for step_i in range(steps + 1):
                t = inner * step_i / max(steps, 1)
                for o in obstacles:
                    p = np.asarray(o.pos, dtype=float) + np.asarray(o.vel, dtype=float) * t
                    self._add_bump(field, origin, p, self.obstacle_amp,
                                   self.obstacle_sigma)

        for c in (jamming_centers or []):
            self._add_bump(field, origin, np.asarray(c, dtype=float),
                           self.jamming_amp, self.jamming_sigma)

        return RiskField(origin=origin, cell=self.cell, shape=(nx, ny, nz),
                         risk=field)

    def _add_bump(self, field: np.ndarray, origin: np.ndarray, p: np.ndarray,
                  amp: float, sigma: float) -> None:
        nx, ny, nz = field.shape
        half = max(1, int(np.ceil(3.0 * sigma / self.cell)))
        ci = (np.asarray(p, dtype=float) - origin) / self.cell
        c0 = int(np.round(ci[0]))
        c1 = int(np.round(ci[1]))
        c2 = int(np.round(ci[2]))
        for i in range(max(0, c0 - half), min(nx, c0 + half + 1)):
            for j in range(max(0, c1 - half), min(ny, c1 + half + 1)):
                for k in range(max(0, c2 - half), min(nz, c2 + half + 1)):
                    wp = origin + self.cell * np.array([i, j, k])
                    d = float(np.linalg.norm(wp - p))
                    field[i, j, k] = max(field[i, j, k],
                                         amp * float(np.exp(-0.5 * (d / sigma) ** 2)))

    def route_metrics(self, route: list[np.ndarray],
                      field: RiskField) -> tuple[float, float]:
        """Return (mean_risk_per_m, total_length_m) for a polyline route."""
        pts = [np.asarray(p, dtype=float).reshape(3) for p in route]
        if len(pts) == 0:
            return 0.0, 0.0
        total_risk = 0.0
        length = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-9:
                continue
            n = max(1, int(np.ceil(seg / self.sampling_step)))
            for s_i in range(n + 1):
                p = a + (b - a) * s_i / n
                total_risk += field.sample(p)
            length += seg
        mean_risk = total_risk / max(length, 1e-9)
        return mean_risk, length

    def route_score(self, route: list[np.ndarray],
                    field: RiskField) -> tuple[float, float, float]:
        """Return (score, mean_risk, length). Score penalises longer routes."""
        mean_risk, length = self.route_metrics(route, field)
        score = mean_risk + self.length_penalty * length
        return score, mean_risk, length
