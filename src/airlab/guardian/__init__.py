"""Nexus-Predator AI core (defensive / sense-and-avoid).

A research-grade autonomy brain for a next-generation multirotor.  This module
is the *decision layer* that turns raw fused state + obstacle detections + RF/
navigation health into a safe maneuver.  It deliberately implements the
defensive/evasive/safety functions only — no weapons, no targeting, no guidance
toward a threat.  Every "undeclared" capability is documented internally in the
research notes and marked as such; nothing here is hidden from the operator in a
way that opposes safety.

Design constraints (from the master prompt):
  * safety / recognition is design-in, not bolt-on;
  * every idea is scored on energy, weight, compute, heat, latency,
    deployability, and "intelligence per watt";
  * speculative capabilities are labelled (A/B/C/D quadrants), never presented
    as established fact.
"""

from .threats import GuardianState, ThreatEngine, ThreatReport, Obstacle
from .avoidance import EvasionPlanner, EvasionDecision
from .brain import GuardianBrain, BrainDecision
from .drone_nexus import NexusAirV2, NexusSpec
from .risk import RiskWorldModel, RiskField
from .replan import PredictiveRePlanner, ReplanResult
from .health import HealthScore, SubsystemHealth, HealthPrognosis, simulated_features

__all__ = [
    "GuardianState",
    "ThreatEngine",
    "ThreatReport",
    "Obstacle",
    "EvasionPlanner",
    "EvasionDecision",
    "GuardianBrain",
    "BrainDecision",
    "NexusAirV2",
    "NexusSpec",
    "RiskWorldModel",
    "RiskField",
    "PredictiveRePlanner",
    "ReplanResult",
    "HealthScore",
    "SubsystemHealth",
    "HealthPrognosis",
    "simulated_features",
]
