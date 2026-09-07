"""Low-watt edge perception path (spiking / NeuViT-style, transparent).

Program priority #7.  Until now the mission bridge only ever received
*scripted* obstacles (``guardian_obstacles``).  This module is the first
sensing path: it converts a cheap low-power range/flow/ranging stream into
:class:`Obstacle` detections for the guardian oracle, and reports the
actual ``intelligence-per-watt`` numbers from declared hardware constants.

Honest classification (master prompt §4/§27)
--------------------------------------------
* The detector is a **transparent synthetic spiking-style** front-end: a
  range/depth frame is thresholded with a one-shot binary spike (LIF-style),
  clustered on the edge, and tracked frame-to-frame.  It is **simulated**
  behaviour, not a claim of running on a real NeuEdge/NeuViT chip.
* The neuromorphic efficiency number (`gops_per_w`) is a **declared research
  figure** (847 GOp/s/W) used for the energy budget, not a measurement on
  this code.
* The compute/energy numbers are **Estimated** for a 0.35 TOPS-class edge
  board; real figures must come from board power probes.

This is a safe, defensible substrate: the oracle consumes *sensed* obstacles,
so the route planner can avoid what the aircraft actually sees, not what a
scenario script told it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .threats import Obstacle


@dataclass
class PerceptionConfig:
    # Declared edge/NPU constants (low-watt path).  Estimated, not measured.
    edge_topps: float = 0.35
    edge_power_w: float = 8.0
    neuromorphic_gops_per_w: float = 847.0   # declared research figure
    spike_threshold_m: float = 2.0           # range gate (m)
    spike_voltage_gate_m: float = 0.3        # cluster separation (m)
    min_points: int = 3
    max_clusters: int = 3
    frame_hz: float = 10.0
    sensor_range_m: float = 8.0
    fov_deg: float = 60.0


@dataclass
class EdgePerception:
    obstacles: list[Obstacle] = field(default_factory=list)
    spike_count: int = 0
    cluster_count: int = 0
    frame_n: int = 0
    frame_hz: float = 10.0
    energy_w: float = 0.0
    gops_per_w: float = 0.0

    @property
    def topps_used(self) -> float:
        # symbolic: spikes/s scaled to a nominal TOPS; declared, not measured
        return float(self.frame_hz * max(self.frame_n, 1)) * 1e-12

    @property
    def intelligence_per_watt(self) -> float:
        if self.energy_w <= 0:
            return 0.0
        return self.frame_hz / self.energy_w


class SpikeVision:
    """Decode a sparse point/range stream into obstacle clusters.

    This is a *surrogate* front-end for a future SNN/NeuViT: it thresholds a
    low-resolution depth/range frame, applies a one-shot binary spike, clusters
    the active points on the edge, and tracks clusters.  It deliberately does
    **not** use an external ML library; every step is inspectable and
    numpy-only, and the numbers are declared estimates.
    """

    def __init__(self, cfg: PerceptionConfig | None = None) -> None:
        self.cfg = cfg or PerceptionConfig()
        self._prev: list[np.ndarray] = []
        self.frame = 0
        self._spike_count = 0

    def process(self, points: np.ndarray | None,
                dt: float = 0.1) -> EdgePerception:
        """Turn a sensor `points` array (N x 3 NED relative) into obstacles.

        ``points`` are *range-return points* (NED, relative to the aircraft).
        ``None``/empty means no obstruction detected.
        """
        cfg = self.cfg
        self.frame += 1
        self._spike_count = 0
        if points is None or len(points) == 0:
            self._prev = []
            return self._result([], 0)
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        # range gate
        dist = np.linalg.norm(pts, axis=1)
        gate = dist < cfg.sensor_range_m
        pts = pts[gate]
        self._spike_count = int(pts.shape[0])
        if pts.shape[0] < cfg.min_points:
            self._prev = []
            return self._result([], self._spike_count)
        clusters = self._cluster(pts)
        obstacles = []
        for c in clusters:
            cent = np.asarray(c, dtype=float).reshape(3)
            vel = self._vel(cent, dt)
            obstacles.append(Obstacle(pos=cent.copy(), vel=vel,
                                      radius=float(cfg.spike_voltage_gate_m)))
        return self._result(obstacles, self._spike_count)

    # ---------------------------------------------------------- internals
    def _cluster(self, pts: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        centers: list[list[np.ndarray]] = []
        for p in pts:
            if not centers or all(
                    float(np.linalg.norm(p - c[-1])) > cfg.spike_voltage_gate_m
                    for c in centers):
                centers.append([p.copy()])
            else:
                idx = min(range(len(centers)),
                          key=lambda i: float(np.linalg.norm(p - centers[i][-1])))
                c = centers[idx]
                c[-1] = 0.6 * c[-1] + 0.4 * p
                c.append(p.copy())
            if len(centers) > cfg.max_clusters * 2:
                break
        out = [c[-1] for c in centers if len(c) >= cfg.min_points]
        return out[:cfg.max_clusters]

    def _vel(self, cent: np.ndarray, dt: float) -> np.ndarray:
        if not self._prev or dt <= 0:
            return np.zeros(3)
        best = min(self._prev,
                   key=lambda p: float(np.linalg.norm(p - cent)))
        return (cent - best) / dt

    def _result(self, obstacles: list[Obstacle],
                spike_count: int) -> EdgePerception:
        cfg = self.cfg
        self._prev = [o.pos.copy() for o in obstacles]
        # Energy: edge at 8 W * (frame_hz / real-time stream rate) — a very
        # rough, declared estimate; no thermal model here (thermal.py owns it).
        energy = cfg.edge_power_w * min(float(cfg.frame_hz) / 10.0, 1.0)
        return EdgePerception(
            obstacles=obstacles,
            spike_count=spike_count,
            cluster_count=len(obstacles),
            frame_n=self.frame,
            frame_hz=cfg.frame_hz,
            energy_w=energy,
            gops_per_w=cfg.neuromorphic_gops_per_w,
        )


class PerceptionToGuardian:
    """Feed the spiking vision into the mission bridge / guardian oracle."""

    def __init__(self, vision: SpikeVision | None = None) -> None:
        self.vision = vision or SpikeVision()

    def obstacles(self, points: np.ndarray | None,
                  dt: float = 0.1) -> list[Obstacle]:
        return self.vision.process(points, dt=dt).obstacles
