"""Precision humanitarian payload-drop model (digital twin).

Priority #13 of the Nexus-Predator program.  This module simulates the
three civil drop methods from the rescue-drone design doc
(`docs/rescue-drone/05-payload-drop-ops.md`):

* ``free_drop``         — low-altitude free fall with an airbag pouch
* ``guided_parachute``  — small round steerable parachute (declared CEP goal <= 5 m)
* ``winch``             — hover + line lower (declared accuracy <= 2 m)

and applies the hard humanitarian guard: **never drop over people**
(a free/guided ballistic release inside the people-exclusion ring is
rejected; the exactly-hovering winch remains the only allowed method there,
because it never leaves the vertical under the cleared hover point).

Honesty: every constant is *declared* engineering estimate, not measured.
The model is a transparent Monte-Carlo of simplified physics — honest
enough to compare methods and reject unsafe drops, not a flight-validated
ballistic solver.  No weapons, no ballistics targeting code: release point
is fixed above the target by definition; we only model *error*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

METHODS = ("free_drop", "guided_parachute", "winch")


@dataclass
class DropConfig:
    """Declared constants for the precision-drop digital twin (not measured)."""
    seed: int = 42
    mc_samples: int = 400
    g: float = 9.81
    rho_air_kg_m3: float = 1.225
    payload_mass_kg: float = 1.2
    canopy_mass_kg: float = 0.3
    canopy_area_m2: float = 0.5          # round chute ~Ø 0.8 m
    canopy_cd: float = 1.3
    guide_steer_mps: float = 3.5         # steering authority (declared)
    guide_efficiency_mean: float = 0.72  # attitude-tracking efficiency mean
    guide_efficiency_sigma: float = 0.08
    release_sigma_m: float = 0.6         # release-position noise (1 sigma)
    wind_sigma_mps: float = 1.0          # gust noise
    horizontal_lag_s: float = 1.2        # time to adopt the wind vector
    winch_rate_mps: float = 1.5          # line-lower speed
    winch_sigma_per_mps_m: float = 0.15  # line swing sigma per 1 m/s wind
    hover_sigma_m: float = 0.5           # hover-hold position sigma
    people_min_clearance_m: float = 5.0
    wind_max_mps: float = 12.0           # ops envelope (declared)
    free_drop_lag_s: float = 0.6         # shorter aerodynamic lag for a stiff pouch
    free_drop_slip: float = 0.75         # fraction of wind adopted during fall
    comp_sigma_frac: float = 0.08        # release-point misprediction (8% of mean drift)


@dataclass
class DropVerdict:
    method: str
    feasible: bool
    cep_m: float
    p90_m: float
    mean_drift_m: float
    descent_s: float
    reasons: list[str] = field(default_factory=list)


def cep50(samples_xy: np.ndarray) -> float:
    """Radius containing 50% of impacts (CEP50)."""
    r = np.linalg.norm(samples_xy, axis=1)
    return float(np.median(r))


def p90(samples_xy: np.ndarray) -> float:
    r = np.linalg.norm(samples_xy, axis=1)
    return float(np.quantile(r, 0.9))


def _descent_time(method: str, h_m: float, cfg: DropConfig) -> float:
    if method == "free_drop":
        return math.sqrt(2.0 * h_m / cfg.g)
    if method == "guided_parachute":
        m = cfg.payload_mass_kg + cfg.canopy_mass_kg
        vz = math.sqrt(2.0 * m * cfg.g /
                       (cfg.rho_air_kg_m3 * cfg.canopy_cd * cfg.canopy_area_m2))
        return h_m / vz
    return h_m / cfg.winch_rate_mps


def impact_samples(method: str, h_m: float, wind_mps: float,
                    wind_dir_rad: float = 0.0, seed: int = 42,
                    mc_samples: int | None = None,
                    config: DropConfig | None = None) -> np.ndarray:
    """Monte-Carlo impact offsets (x downwind, y crosswind) w.r.t. the aim
    point directly above which the release happens.  Simplified declared
    physics — honest enough for method comparison and the safety guard."""
    cfg = config or DropConfig(seed=seed)
    n = mc_samples or cfg.mc_samples
    rng = np.random.default_rng(seed)
    w = np.array([math.cos(wind_dir_rad), math.sin(wind_dir_rad)])
    gust = rng.normal(0.0, cfg.wind_sigma_mps, size=(n, 1))
    wind_vec = (wind_mps + gust) * w

    # Deterministic (known) part of the wind is aimed-off by the jettison
    # computer (CARP-style release point planning, civil heritage only):
    # residual deterministic error = mis-prediction of the *mean* drift.
    det_wind = wind_mps * w
    gust_vec = gust * w

    if method == "free_drop":
        t = _descent_time(method, h_m, cfg)
        eff_t = max(0.0, t - cfg.free_drop_lag_s)
        mis = rng.normal(0.0, cfg.comp_sigma_frac, size=(n, 1))
        drift = (1.0 + mis) * (gust_vec * cfg.free_drop_slip * eff_t) \
            + mis * (det_wind * cfg.free_drop_slip * eff_t)
        noise = rng.normal(0.0, cfg.release_sigma_m, size=(n, 2))
        return drift + noise

    if method == "guided_parachute":
        t = _descent_time(method, h_m, cfg)
        eff_t = max(0.0, t - cfg.horizontal_lag_s)
        # steering cancels both gust-derived and mis-predicted drift, up to
        # guide_steer * 0.9 * t of authority (declared physical limit).
        max_cancel = cfg.guide_steer_mps * 0.9 * t
        mis = rng.normal(0.0, cfg.comp_sigma_frac, size=(n, 1))
        raw = (gust_vec + mis * det_wind) * eff_t
        eff = np.clip(rng.normal(cfg.guide_efficiency_mean,
                                 cfg.guide_efficiency_sigma, size=(n, 1)),
                      0.4, 0.9)
        raw_mag = np.linalg.norm(raw, axis=1, keepdims=True)
        cancel = np.minimum(raw_mag, max_cancel) * eff
        residual = (raw_mag - cancel) * raw / np.maximum(raw_mag, 1e-9)
        noise = rng.normal(0.0, cfg.release_sigma_m, size=(n, 2))
        return residual + noise

    # winch — hang-line from a holding hover point; residual swing only
    t = _descent_time(method, h_m, cfg)
    sigma = cfg.hover_sigma_m + cfg.winch_sigma_per_mps_m * wind_mps
    return rng.normal(0.0, sigma, size=(n, 2))


def evaluate_method(method: str, h_m: float, wind_mps: float,
                    wind_dir_rad: float = 0.0, seed: int = 42,
                    config: DropConfig | None = None) -> DropVerdict:
    cfg = config or DropConfig(seed=seed)
    s = impact_samples(method, h_m, wind_mps, wind_dir_rad, seed=seed,
                       config=cfg)
    drift = s.mean(axis=0)
    return DropVerdict(
        method=method, feasible=True,
        cep_m=round(cep50(s), 2), p90_m=round(p90(s), 2),
        mean_drift_m=round(float(np.linalg.norm(drift)), 2),
        descent_s=round(_descent_time(method, h_m, cfg), 1),
    )


def plan_drop(h_m: float, wind_mps: float, people_clearance_m: float = 99.0,
              wind_dir_rad: float = 0.0, seed: int = 42,
              config: DropConfig | None = None) -> DropVerdict:
    """Humanitarian drop planner: evaluate the three methods and return the
    lowest-CEP verdict that passes the safety guards.  If nothing is safe,
    the returned verdict is free_drop-shaped with feasible=False and the
    reasons explain it."""
    cfg = config or DropConfig(seed=seed)
    best: DropVerdict | None = None
    rows: dict[str, DropVerdict] = {}
    for m in METHODS:
        v = evaluate_method(m, h_m, wind_mps, wind_dir_rad, seed, cfg)
        reasons: list[str] = []
        if wind_mps > cfg.wind_max_mps:
            reasons.append(f"wind {wind_mps:.1f} > envelope "
                           f"{cfg.wind_max_mps:.1f} m/s")
        if m != "winch" and people_clearance_m < cfg.people_min_clearance_m:
            reasons.append(f"people at {people_clearance_m:.1f} m "
                           f"< {cfg.people_min_clearance_m:.1f} m exclusion "
                           f"ring (ballistic release forbidden)")
        if reasons:
            v.feasible = False
            v.reasons = reasons
        rows[m] = v
        if v.feasible and (best is None or v.cep_m < best.cep_m):
            best = v
    if best is not None:
        return best
    # nothing feasible — report the (rejected) best-effort verdict so callers
    # can show WHY: pick the smallest CEP among rejected ones.
    v = min(rows.values(), key=lambda x: x.cep_m)
    v.reasons = sorted({r for x in rows.values() for r in x.reasons})
    return v
