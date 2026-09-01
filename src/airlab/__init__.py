"""AIR Lab — Autonomous UAV Research & Innovation Lab.

A minimal, dependency-light but *architecturally honest* simulation of a
quadrotor autonomy stack:

    world         ->  sensors  ->  AHRS + nav EKF  ->  flight controller  ->  dynamics

The goal is not to replace PX4/Gazebo/AirSim, but to give us a readable,
extendable reference implementation of the *ideas* (sensor fusion, cascaded
control, fault injection, evaluation) that we can grow into a full digital
twin / scenario-generation lab.
"""

from __future__ import annotations

__version__ = "0.1.0"
