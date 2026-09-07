"""Predictive re-planner: avoid the whole threat corridor, not just one step.

Uses a transparent beam search over the *remaining* waypoint sequence: at each
remaining waypoint it considers a small set of bounded lateral/vertical offsets,
propagates a coarse polyline, evaluates it against the risk world model, and
keeps the best K prefixes.  The result is a new waypoint sequence (with the same
number of legs) that minimises predicted risk while cost-ing extra mission
distance and checking energy feasibility against the platform's battery.

This is the capability that upgrades the guardian from "detect + dodge" to
"detect + avoid the threat corridor before it arrives."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .risk import RiskWorldModel, RiskField


@dataclass
class ReplanResult:
    route: list[np.ndarray]
    baseline: list[np.ndarray]
    risk_reduction: float            # baseline_mean_risk - replanned_mean_risk
    bas_risk: float
    repl_risk: float
    bas_length: float
    repl_length: float
    extra_distance_frac: float
    energy_heavy_required: float     # fraction of battery the replanned route needs
    feasible: bool
    min_clearance_m: float
    # Thermal-aware feasibility (brief-30): whether the replanned route keeps
    # every thermal node (cpu_npu / esc / motor / battery) under its limit.
    thermal_feasible: bool = True
    thermal_max_c: float = 0.0
    thermal_worst_node: str = ""
    thermal_margin_c: float = float("inf")
    # Thermal-aware trajectory modulation (brief-36): when the raw cruise
    # profile would overheat a node, the planner returns the reduced
    # compute_frac / throttle-power profile that keeps the same route feasible
    # while cooling the hot node (instead of only rejecting it).
    thermal_mitigated: bool = False
    thermal_compute_frac: float = 0.3
    thermal_power_frac: float = 1.0
    thermal_power_w: float = 0.0
    reasons: list[str] = field(default_factory=list)


class PredictiveRePlanner:
    def __init__(self, model: RiskWorldModel | None = None, beam: int = 6,
                 lateral_offsets=(-8.0, -5.0, -3.0, 0.0, 3.0, 5.0, 8.0),
                 vertical_offsets=(-2.0, -1.0, 0.0, 1.0, 2.0), cruise_speed: float = 3.0,
                 hover_power_w: float = 112.0, battery_capacity_wh: float = 71.0,
                 energy_reserve_frac: float = 0.15, min_clearance: float = 2.0,
                 sampling_step: float = 0.5,
                 thermal_aware: bool = False,
                 thermal_ambient_c: float = 25.0,
                 thermal_initial_temps: dict[str, float] | None = None) -> None:
        self.model = model or RiskWorldModel()
        self.beam = beam
        self.lateral_offsets = lateral_offsets
        self.vertical_offsets = vertical_offsets
        self.cruise_speed = cruise_speed
        self.hover_power_w = hover_power_w
        self.battery_capacity_wh = battery_capacity_wh
        self.energy_reserve_frac = energy_reserve_frac
        self.min_clearance = min_clearance
        self.sampling_step = sampling_step
        # Optional thermal-aware feasibility: when enabled, the planner also
        # checks that the replanned route keeps every thermal node (cpu_npu /
        # esc / motor / battery) under its limit over the flight time (energy
        # alone is not enough — a hot compute/ESC/motor can still force an
        # abort).  The model is simulated fresh each call, but it may be seeded
        # with the *current live* node temperatures so the prediction is not
        # restarted at ambient (priority #8).
        self.thermal_aware = bool(thermal_aware)
        self.thermal_ambient_c = float(thermal_ambient_c)
        self.thermal_initial_temps = dict(thermal_initial_temps or {})

    def plan(self, start: np.ndarray, remaining: list[np.ndarray],
             battery_frac: float, obstacles=None,
             jamming_centers=None) -> ReplanResult:
        """Re-plan the remaining waypoints from ``start``.

        Returns the original route (baseline) and the best re-planned route for
        the *remaining* mission, along with risk/energy/clearance metrics.
        """
        remaining = [np.asarray(w, dtype=float).reshape(3) for w in remaining]
        start = np.asarray(start, dtype=float).reshape(3)
        if not remaining:
            return self._empty(start)

        # build the world bounds around the current + remaining points
        all_pts = [start] + remaining
        if obstacles:
            for o in obstacles:
                all_pts.append(np.asarray(o.pos, dtype=float))
                all_pts.append(np.asarray(o.pos, dtype=float)
                               + np.asarray(o.vel, dtype=float) * 2.0)
        pts = np.asarray(all_pts, dtype=float)
        margin = 6.0
        bounds = (pts[:, 0].min() - margin, pts[:, 0].max() + margin,
                  pts[:, 1].min() - margin, pts[:, 1].max() + margin,
                  pts[:, 2].min() - margin, pts[:, 2].max() + margin)
        field = self.model.build(bounds, obstacles or [], jamming_centers)

        baseline = [start] + remaining
        bas_score, bas_risk, bas_len = self.model.route_score(baseline, field)

        # beam search over waypoint offsets
        beams: list[tuple[list[np.ndarray], float]] = [([start], 0.0)]
        for wp in remaining:
            out: list[tuple[list[np.ndarray], float]] = []
            for pts_list, score in beams:
                prev = pts_list[-1]
                for dlat in self.lateral_offsets:
                    for dvert in self.vertical_offsets:
                        cand_wp = wp + self._offset(prev, wp, dlat, dvert)
                        route = pts_list + [cand_wp]
                        s, _, _ = self.model.route_score(route, field)
                        out.append((route, s))
            out.sort(key=lambda x: x[1])
            beams = out[:self.beam]

        best_route, best_score = beams[0]
        best_route = best_route[1:]  # drop start; planner returns route incl start? keep start
        # trim: we want route = [start] + chosen_remaining (but beams already
        # contains start + all chosen waypoints)
        repl_route = beams[0][0]
        _, repl_risk, repl_len = self.model.route_score(repl_route, field)
        # distance + energy feasibility
        extra = max(0.0, repl_len - bas_len) / max(bas_len, 1e-9)
        time_h = (repl_len / max(self.cruise_speed, 0.1)) / 3600.0
        energy_h = self.hover_power_w * time_h
        req = (energy_h / max(self.battery_capacity_wh, 1e-9)) + self.energy_reserve_frac
        feasible = req <= battery_frac + 1e-9
        mc = self._min_clearance(repl_route, obstacles or [])
        therm_feasible, therm_max, therm_worst, therm_margin, therm_mit, \
            therm_cf, therm_pf, therm_pw = \
            self._thermal_feasibility(repl_route, req)

        reasons = []
        if repl_risk < bas_risk - 1e-6:
            reasons.append("risk_corridor_avoided")
        if extra > 0.05:
            reasons.append("extra_distance")
        if not feasible:
            reasons.append("energy_infeasible")
        if not therm_feasible:
            reasons.append("thermal_infeasible")

        return ReplanResult(
            route=repl_route, baseline=baseline,
            risk_reduction=bas_risk - repl_risk,
            bas_risk=bas_risk, repl_risk=repl_risk,
            bas_length=bas_len, repl_length=repl_len,
            extra_distance_frac=extra, energy_heavy_required=req,
            feasible=feasible and therm_feasible, min_clearance_m=mc,
            thermal_feasible=therm_feasible, thermal_max_c=therm_max,
            thermal_worst_node=therm_worst, thermal_margin_c=therm_margin,
            thermal_mitigated=therm_mit, thermal_compute_frac=therm_cf,
            thermal_power_frac=therm_pf, thermal_power_w=therm_pw,
            reasons=reasons,
        )

    def _empty(self, start: np.ndarray) -> ReplanResult:
        return ReplanResult(route=[start], baseline=[start],
                            risk_reduction=0.0, bas_risk=0.0, repl_risk=0.0,
                            bas_length=0.0, repl_length=0.0,
                            extra_distance_frac=0.0, energy_heavy_required=0.0,
                            feasible=True, min_clearance_m=float("inf"),
                            reasons=["no_remaining_waypoints"])

    def _thermal_feasibility(self, route, energy_req: float) -> tuple[
            bool, float, str, float, bool, float, float, float]:
        if not self.thermal_aware:
            return True, 0.0, "", float("inf"), False, 0.3, 1.0, 0.0
        # Nominal profile first.
        nom = self._sim_thermal(route, compute_frac=0.3, power_frac=1.0)
        if nom[0]:
            return True, nom[1], nom[2], nom[3], False, 0.3, 1.0, self.hover_power_w
        # The nominal route over-heats a node.  Search for the *least*
        # disruptive mitigation: reduce compute load (the edge/NPU) and/or
        # reduce propulsive power (slower, so the propulsion stack runs cooler
        # per second over a longer exposure).  We prefer staying close to
        # nominal, and keep the most disruptive (power) reduction for last.
        best: tuple[float, float, float] | None = None
        best_out: tuple[bool, float, str, float, float, float] | None = None
        for pf in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            for cf in (0.3, 0.25, 0.2, 0.15, 0.1, 0.05):
                ok, max_t, worst, margin = self._sim_thermal(
                    route, compute_frac=cf, power_frac=pf)
                if not ok:
                    continue
                # disruption = how far we move from the nominal profile.  The
                # grid order already prefers nominal -> least reduction; among
                # equal-disruption keep the smallest power hit (best cooling
                # for the same compute reduction).
                key = (0.3 - cf) + 0.7 * (1.0 - pf)
                if best is None or key < best[0] - 1e-9 or \
                        (abs(key - best[0]) < 1e-9 and pf > best[1]):
                    best = (key, pf, cf)
                    best_out = (ok, max_t, worst, margin, cf, pf)
        if best_out is None:
            return False, nom[1], nom[2], nom[3], False, 0.3, 1.0, \
                self.hover_power_w
        _, max_t, worst, margin, cf, pf = best_out
        return True, max_t, worst, margin, True, cf, pf, \
            self.hover_power_w * pf

    def _sim_thermal(self, route, compute_frac: float,
                     power_frac: float) -> tuple[bool, float, str, float]:
        """Simulate the part-level thermal model for a route at a profile.

        Returns (ok, max_temp_c, worst_node, margin_c).  A node is only ok if
        it stays under its OWN limit.
        """
        from .thermal import PartThermalModel
        model = PartThermalModel(ambient_c=self.thermal_ambient_c,
                                 initial_temps=self.thermal_initial_temps)
        pf = float(np.clip(power_frac, 0.1, 1.0))
        cf = float(np.clip(compute_frac, 0.0, 1.0))
        # Reducing propulsive power slows the aircraft, so exposure time grows
        # for the same route length (honest coupling, not a silent free lunch).
        speed = max(0.2, self.cruise_speed * pf)
        t_total = max(0.1, self.route_length(route) / max(speed, 0.2))
        power = self.hover_power_w * pf
        steps = max(1, int(min(t_total, 7200.0) / 0.5))
        dt = t_total / steps
        for _ in range(steps):
            model.step(power, dt, compute_frac=cf)
        worst = model.worst_node()
        max_t = model.max_temp()
        node = next((n for n in model.nodes if n.name == worst), None)
        margin = node.margin_c if node is not None else float("inf")
        worst_limit = node.max_temp_c if node is not None else 0.0
        return max_t < worst_limit, max_t, worst, margin

    @staticmethod
    def route_length(route) -> float:
        total = 0.0
        pts = [np.asarray(p, dtype=float).reshape(3) for p in route]
        for a, b in zip(pts[:-1], pts[1:]):
            total += float(np.linalg.norm(b - a))
        return total

    def _offset(self, a: np.ndarray, b: np.ndarray, dlat: float,
                dvert: float) -> np.ndarray:
        """A bounded lateral/vertical offset along the a->b direction."""
        d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        dxy = np.array([d[0], d[1], 0.0])
        mag = float(np.linalg.norm(dxy))
        off = np.zeros(3)
        if mag > 1e-6:
            perp = np.array([-dxy[1], dxy[0], 0.0]) / mag
            off = off + dlat * perp
        off[2] = off[2] + dvert
        return off

    def _min_clearance(self, route: list[np.ndarray], obstacles) -> float:
        worst = float("inf")
        for a, b in zip(route[:-1], route[1:]):
            seg = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(seg / self.sampling_step)))
            for s_i in range(n + 1):
                p = a + (b - a) * s_i / n
                for o in obstacles:
                    d = float(np.linalg.norm(p - o.pos)) - getattr(o, "radius", 0.5)
                    worst = min(worst, d)
        if not obstacles:
            return float("inf")
        if worst == float("inf"):
            return float("inf")
        return worst


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)
