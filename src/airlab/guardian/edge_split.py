"""Edge-vs-Ground compute split for the guardian (master prompt section 18).

Section 18 asks: what must run *onboard*, what can run at the *ground station*,
and what can run on a *server/cloud* — compared over latency, power,
reliability, bandwidth, privacy, and compute requirements.

This module is a transparent allocator over the guardian's compute tasks.  It
does **not** claim a real GCS or a real network.  It uses **declared estimates**
for link bandwidth, latency, power, and reliability, and encodes the one hard
rule from the safety architecture:

  * Safety-critical tasks (threat detection, defensive evasion, guardian brain,
    predictive re-planner) **must stay onboard** — a loss of link cannot be
    allowed to remove the aircraft's ability to protect itself.
  * Analytics-only tasks (long-range map accumulation, historical risk prior
    training, health trend analytics, mission post-processing) MAY move to the
    ground station or cloud.
  * Reliable link is never assumed for a safety path.

Output is a per-task placement + aggregate edge load / power / latency and a
compare table for onboard-vs-ground-vs-cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComputeTask:
    name: str
    topps: float                 # compute intensity (TOPS, estimated)
    power_w: float               # estimated average edge power
    latency_ms: float            # estimated onboard latency
    safety_critical: bool = False
    data_mbps: float = 0.0       # typical telemetry/data payload
    privacy_sensitive: bool = False


@dataclass
class PlacementResult:
    task: str
    location: str                # onboard / ground / cloud
    onboard: bool
    reason: str
    latency_ms: float
    power_w: float
    bandwidth_mbps: float
    privacy_risk: int            # 0 low .. 2 high


@dataclass
class SplitResult:
    placements: list[PlacementResult] = field(default_factory=list)
    onboard_topps: float = 0.0
    onboard_power_w: float = 0.0
    edge_latency_ms: float = 0.0
    ground_latency_ms: float = 0.0
    total_data_mbps: float = 0.0
    offloadable_data_mbps: float = 0.0
    num_safety_tasks: int = 0
    num_analytics_tasks: int = 0
    score: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "onboard_topps": round(self.onboard_topps, 3),
            "onboard_power_w": round(self.onboard_power_w, 2),
            "edge_latency_ms": round(self.edge_latency_ms, 2),
            "ground_latency_ms": round(self.ground_latency_ms, 2),
            "offloadable_data_mbps": round(self.offloadable_data_mbps, 2),
            "total_data_mbps": round(self.total_data_mbps, 2),
            "safety_tasks_onboard": self.num_safety_tasks,
            "analytics_tasks": self.num_analytics_tasks,
        }


@dataclass
class LinkEstimate:
    ground_rtt_ms: float = 120.0
    cloud_rtt_ms: float = 400.0
    link_bandwidth_mbps: float = 8.0
    ground_reliability: float = 0.97
    cloud_reliability: float = 0.90


class EdgeGroundSplit:
    """Classify and place guardian compute tasks on the {onboard, ground, cloud}
    spectrum using declared-estimate link/reliability constants."""

    def __init__(self, link: LinkEstimate | None = None,
                 edge_topps: float = 0.35,
                 edge_power_budget_w: float = 8.0) -> None:
        self.link = link or LinkEstimate()
        self.edge_topps = float(edge_topps)
        self.edge_power_budget_w = float(edge_power_budget_w)

    @staticmethod
    def default_tasks() -> list[ComputeTask]:
        """The guardian's compute tasks (declared-estimate intensities)."""
        return [
            # Safety-critical: MUST stay onboard.
            ComputeTask("threat_detection", 0.06, 1.2, 2.0, safety_critical=True),
            ComputeTask("guardian_evasion", 0.03, 0.8, 1.0, safety_critical=True),
            ComputeTask("predictive_replan", 0.05, 1.5, 12.0, safety_critical=True),
            ComputeTask("sensor_fusion", 0.04, 0.9, 1.0, safety_critical=True),
            # Analytics-only: MAY move to ground/cloud.
            ComputeTask("map_accumulation", 0.03, 0.5, 8.0,
                        data_mbps=2.0, privacy_sensitive=True),
            ComputeTask("risk_prior_training", 0.08, 2.0, 500.0,
                        data_mbps=4.0, privacy_sensitive=False),
            ComputeTask("health_trend_analytics", 0.02, 0.3, 60.0,
                        data_mbps=0.5, privacy_sensitive=True),
            ComputeTask("mission_post_process", 0.10, 3.0, 1000.0,
                        data_mbps=6.0, privacy_sensitive=False),
        ]

    def place(self, tasks: list[ComputeTask] | None = None) -> SplitResult:
        tasks = tasks if tasks is not None else self.default_tasks()
        res = SplitResult()
        for t in tasks:
            if t.safety_critical:
                loc = "onboard"
                reason = "safety-critical; link loss must not remove protection"
            elif t.privacy_sensitive and self.link.link_bandwidth_mbps < 4.0:
                loc = "onboard"
                reason = "privacy-sensitive and low-bandwidth link cannot offload"
            elif t.latency_ms > 250.0:
                loc = "cloud"
                reason = "latency-tolerant; low onboard compute demand"
            elif t.latency_ms > 30.0:
                loc = "ground"
                reason = "moderate-latency; best on GCS"
            else:
                loc = "onboard"
                reason = "low-latency analytics; cheap to run onboard"
            onboard = loc == "onboard"
            lat, power, bw, priv = self._placement_metrics(t, loc, onboard)
            res.placements.append(PlacementResult(
                task=t.name, location=loc, onboard=onboard, reason=reason,
                latency_ms=lat, power_w=power, bandwidth_mbps=bw,
                privacy_risk=priv))
            if onboard:
                res.onboard_topps += t.topps
                res.onboard_power_w += t.power_w
                res.edge_latency_ms = max(res.edge_latency_ms, t.latency_ms)
                if t.safety_critical:
                    res.num_safety_tasks += 1
            else:
                res.ground_latency_ms = max(res.ground_latency_ms, lat)
                res.offloadable_data_mbps += t.data_mbps
                res.num_analytics_tasks += 1
            res.total_data_mbps += t.data_mbps
        res.score = self._score(res)
        return res

    def _placement_metrics(self, t: ComputeTask, loc: str,
                           onboard: bool) -> tuple[float, float, float, int]:
        link = self.link
        if loc == "onboard":
            return t.latency_ms, t.power_w, 0.0, (1 if t.privacy_sensitive else 0)
        if loc == "ground":
            return t.latency_ms + link.ground_rtt_ms, 0.0, t.data_mbps, \
                (2 if t.privacy_sensitive else 1)
        return t.latency_ms + link.cloud_rtt_ms, 0.0, t.data_mbps, \
            (2 if t.privacy_sensitive else 1)

    def _score(self, res: SplitResult) -> dict:
        # Nominal edge budget fit: fraction of declared edge power/TOPS used.
        edge_power_frac = res.onboard_power_w / max(self.edge_power_budget_w,
                                                    1e-9)
        edge_topps_frac = res.onboard_topps / max(self.edge_topps, 1e-9)
        return {
            "edge_power_frac": round(float(np.clip(edge_power_frac, 0, 2)), 3),
            "edge_topps_frac": round(float(np.clip(edge_topps_frac, 0, 2)), 3),
            "safety_full_onboard": bool(res.num_safety_tasks > 0),
            "ground_offload_fraction": round(
                res.offloadable_data_mbps / max(res.total_data_mbps, 1e-9), 3),
        }
