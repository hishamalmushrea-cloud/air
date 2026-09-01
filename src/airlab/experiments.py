"""Batch experiment runners.

These are the "data factory" layer: they take many Scenario descriptors, run
each through the simulator, and aggregate navigation/safety/energy metrics into
rows that can be written to CSV or used for downstream analysis.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Iterable

import numpy as np

from .scenarios import Scenario, random_scenario, run_scenario, scenario_energy
from .metrics import average_of_metrics


def _row(scenario: Scenario, m: dict, e: dict) -> dict:
    d = scenario.as_dict()
    d.update({k: (float(v) if isinstance(v, (int, float)) else v)
              for k, v in m.items()})
    d.update(e)
    return d


def run_batch(
    scenarios: Iterable[Scenario],
    dt: float = 0.01,
    verbose: bool = True,
) -> tuple[list[dict], list[Scenario], list["object"]]:
    """Run a list of scenarios.

    Returns (rows, scenarios, runs).  ``runs`` can be large; callers that only
    want the aggregate CSV can ignore the third return value.
    """
    rows: list[dict] = []
    runs = []
    t0 = time.time()
    for i, s in enumerate(scenarios):
        m, run = run_scenario(s, record=True)
        e = scenario_energy(run, dt)
        row = _row(s, m, e)
        rows.append(row)
        runs.append(run)
        if verbose:
            print(f"[{i+1}] {s.name}: pos_rmse={m['pos_rmse']:.2f} "
                  f"in_bounds={m['in_bounds_frac']:.2f} "
                  f"collision={m['ground_collision_frac']:.2f} "
                  f"energy={e['energy_wh']:.2f}Wh", flush=True)
    if verbose:
        print(f"[batch] {len(scenarios)} scenarios in {time.time()-t0:.2f}s")
    return rows, list(scenarios), runs


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def random_batch(
    n: int = 20,
    duration: float = 40.0,
    seed: int = 42,
    max_height: float = 8.0,
    max_range: float = 12.0,
) -> tuple[list[dict], list[Scenario]]:
    """Generate and run ``n`` fully random scenarios."""
    rng = np.random.default_rng(seed)
    scenarios = [random_scenario(rng, duration=duration, index=i,
                                 max_height=max_height, max_range=max_range)
                 for i in range(n)]
    rows, scen, _ = run_batch(scenarios)
    return rows, scen


def degradation_study(
    outage_durations: Iterable[float] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    n_per_duration: int = 12,
    duration: float = 40.0,
    seed: int = 123,
    compare_flow: bool = True,
) -> list[dict]:
    """Stress test: how robust is the stack as GNSS outage length grows.

    For each outage length we run ``n_per_duration`` randomly generated
    missions, first with optical-flow aiding enabled and (optionally) again
    with it disabled.  This exposes the safety boundary: the current design
    survives long outages *because* of flow aiding, not despite them.
    """
    outage_durations = list(outage_durations)
    rng = np.random.default_rng(seed)
    rows = []

    # Pre-generate one coherent set of missions so the two arms share the same
    # weather/vehicle/waypoint draw for each outage length.
    base_scenarios = []
    for k in range(n_per_duration):
        s = random_scenario(rng, duration=duration, index=k)
        base_scenarios.append(s)

    variants = [("flow", True)] if not compare_flow else [("flow", True), ("no_flow", False)]

    for od in outage_durations:
        for label, flow_enabled in variants:
            scenarios = []
            for s in base_scenarios:
                s2 = Scenario(**{**s.as_dict(), "flow_enabled": flow_enabled})
                # as_dict converted fields to plain lists; reconstruct is fine
                # because to_config() calls np.asarray on the numeric fields.
                if od <= 0.0:
                    s2.gps_outage = None
                else:
                    start = max(0.0, duration - od - 1.0)
                    end = min(duration, start + od)
                    s2.gps_outage = (start, end)
                scenarios.append(s2)

            metrics_list = []
            for s in scenarios:
                m, _ = run_scenario(s, record=False)
                metrics_list.append(m)
            agg = average_of_metrics(metrics_list, [
                "pos_rmse", "horizontal_rmse", "vertical_rmse",
                "max_horizontal_err", "in_bounds_frac", "ground_collision_frac",
            ])
            row = {
                "outage_duration_s": float(od),
                "velocity_aiding": label,
                "n": n_per_duration,
                "mean_in_bounds": agg.get("in_bounds_frac", float("nan")),
                "mean_collision": agg.get("ground_collision_frac", float("nan")),
                "mean_pos_rmse": agg.get("pos_rmse", float("nan")),
                "mean_horizontal_rmse": agg.get("horizontal_rmse", float("nan")),
                "mean_vertical_rmse": agg.get("vertical_rmse", float("nan")),
                "mean_max_horizontal_err": agg.get("max_horizontal_err", float("nan")),
            }
            rows.append(row)
            print(f"[deg] outage={od:5.1f}s  aiding={label:8s} "
                  f"in_bounds={row['mean_in_bounds']:.3f}  "
                  f"collision={row['mean_collision']:.3f}  "
                  f"rmse={row['mean_pos_rmse']:.3f}")
    return rows


# ------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------- #
def corruption_study(
    flow_faults: list[dict] | None = None,
    outage_durations: Iterable[float] = (0.0, 15.0, 30.0),
    n_per_cell: int = 3,
    duration: float = 40.0,
    seed: int = 2026,
    safety_variants: tuple[bool, ...] = (True, False),
) -> list[dict]:
    """Study how corrupt velocity-aiding affects safety, with/without the
    uncertainty-aware safety layer.

    ``flow_faults`` is a list of dicts with keys: ``name``, ``outage``,
    ``scale``, ``bias_ramp``.  Default:
        none healthy
        dropout during GPS outage
        scale 1.8 (wrong magnitude)
        bias_ramp 0.25 (slowly rolling-speed error)
    """
    if flow_faults is None:
        flow_faults = [
            {"name": "none",      "outage": None, "scale": 1.0, "bias_ramp": 0.0},
            {"name": "dropout",   "outage": (10.0, 30.0), "scale": 1.0, "bias_ramp": 0.0},
            {"name": "scale1.5",  "outage": None, "scale": 1.5, "bias_ramp": 0.0},
            {"name": "scale1.8",  "outage": None, "scale": 1.8, "bias_ramp": 0.0},
            {"name": "bias.1",    "outage": None, "scale": 1.0, "bias_ramp": 0.10},
            {"name": "bias.25",   "outage": None, "scale": 1.0, "bias_ramp": 0.25},
        ]

    outage_durations = list(outage_durations)
    rng = np.random.default_rng(seed)
    base_scenarios = [random_scenario(rng, duration=duration, index=k)
                      for k in range(n_per_cell)]

    rows = []
    for fault in flow_faults:
        for od in outage_durations:
            for safety_enabled in safety_variants:
                scenarios = []
                for s in base_scenarios:
                    s2 = Scenario(**{**s.as_dict(),
                                     "flow_outage": fault["outage"],
                                     "flow_scale": float(fault["scale"]),
                                     "flow_bias_ramp": float(fault["bias_ramp"]),
                                     "safety_enabled": safety_enabled})
                    if od <= 0.0:
                        s2.gps_outage = None
                    else:
                        start = max(0.0, duration - od - 1.0)
                        end = min(duration, start + od)
                        s2.gps_outage = (start, end)
                    scenarios.append(s2)

                metrics_list = []
                for s in scenarios:
                    m, _ = run_scenario(s, record=False)
                    metrics_list.append(m)
                agg = average_of_metrics(metrics_list, [
                    "pos_rmse", "horizontal_rmse", "vertical_rmse",
                    "in_bounds_frac", "ground_collision_frac", "crash",
                    "landed", "safety_fraction", "mean_unc_horiz_std",
                    "mean_flow_health",
                ])
                outcomes = [m.get("safety_outcome", "none") for m in metrics_list]
                # For a *persistent* bias fault there is no way for the sensor
                # health self-report alone to catch it because it has no
                # independent ground truth.  We therefore separately record
                # what the safety layer actually achieved (landed_safely) and
                # still call *unintended crash* only when the ground contact is
                # not part of an intentional landing.
                row = {
                    "flow_fault": fault["name"],
                    "flow_scale": float(fault["scale"]),
                    "flow_bias_ramp": float(fault["bias_ramp"]),
                    "outage_duration_s": float(od),
                    "safety_layer": "on" if safety_enabled else "off",
                    "n": n_per_cell,
                    "mean_in_bounds": agg.get("in_bounds_frac", float("nan")),
                    "mean_collision": agg.get("crash", float("nan")),
                    "mean_ground_contact": agg.get("ground_collision_frac", float("nan")),
                    "mean_landed": agg.get("landed", float("nan")),
                    "mean_safety_fraction": agg.get("safety_fraction", float("nan")),
                    "mean_pos_rmse": agg.get("pos_rmse", float("nan")),
                    "mean_unc_std": agg.get("mean_unc_horiz_std", float("nan")),
                    "mean_flow_health": agg.get("mean_flow_health", float("nan")),
                    "outcomes": ";".join(outcomes),
                }
                rows.append(row)
                print(f"[corr] fault={fault['name']:8s} outage={od:5.1f}s "
                      f"safety={'on ' if safety_enabled else 'off'} "
                      f"in_bounds={row['mean_in_bounds']:.3f} "
                      f"unintended_crash={row['mean_collision']:.3f} "
                      f"landed={row['mean_landed']:.3f}")
    return rows


def landmark_study(
    faults: list[dict] | None = None,
    outage_durations: Iterable[float] = (0.0, 15.0, 30.0),
    n_per_cell: int = 3,
    duration: float = 40.0,
    seed: int = 404,
) -> list[dict]:
    """Ablate the *independent landmark detector* specifically.

    Contrasts: same corrupt velocity-aiding fault, same GNSS outage, same
    safety layer — but landmark detector ON vs OFF.  This isolates whether the
    structurally independent geometry check (and not a self-report) is what
    converts a likely crash into a controlled landing for "convincing but
    wrong" velocity faults (scale error, ramping bias).

    ``faults``: list of dicts with name/scale/bias_ramp.
    """
    if faults is None:
        faults = [
            {"name": "none",      "scale": 1.0, "bias_ramp": 0.0},
            {"name": "scale1.8",  "scale": 1.8, "bias_ramp": 0.0},
            {"name": "bias.25",   "scale": 1.0, "bias_ramp": 0.25},
        ]

    outage_durations = list(outage_durations)
    rng = np.random.default_rng(seed)
    base_scenarios = [random_scenario(rng, duration=duration, index=k)
                      for k in range(n_per_cell)]
    rows = []

    for fault in faults:
        for od in outage_durations:
            for lm_enabled in (True, False):
                scenarios = []
                for s in base_scenarios:
                    s2 = Scenario(**{**s.as_dict(),
                                     "flow_scale": float(fault["scale"]),
                                     "flow_bias_ramp": float(fault["bias_ramp"]),
                                     "safety_enabled": True,
                                     "landmark_enabled": lm_enabled})
                    if od <= 0.0:
                        s2.gps_outage = None
                    else:
                        start = max(0.0, duration - od - 1.0)
                        end = min(duration, start + od)
                        s2.gps_outage = (start, end)
                    scenarios.append(s2)

                metrics_list = []
                for s in scenarios:
                    m, _ = run_scenario(s, record=False)
                    metrics_list.append(m)
                agg = average_of_metrics(metrics_list, [
                    "in_bounds_frac", "crash", "landed", "mean_flow_health",
                ])
                outcomes = [m.get("safety_outcome", "none") for m in metrics_list]
                row = {
                    "flow_fault": fault["name"],
                    "flow_scale": float(fault["scale"]),
                    "flow_bias_ramp": float(fault["bias_ramp"]),
                    "outage_duration_s": float(od),
                    "landmark_detector": "on" if lm_enabled else "off",
                    "n": n_per_cell,
                    "mean_in_bounds": agg.get("in_bounds_frac", float("nan")),
                    "mean_crash": agg.get("crash", float("nan")),
                    "mean_landed": agg.get("landed", float("nan")),
                    "mean_flow_health": agg.get("mean_flow_health", float("nan")),
                    "outcomes": ";".join(outcomes),
                }
                rows.append(row)
                print(f"[lm] fault={fault['name']:8s} outage={od:5.1f}s "
                      f"lm={'on ' if lm_enabled else 'off'} "
                      f"crash={row['mean_crash']:.3f} "
                      f"landed={row['mean_landed']:.3f}")
    return rows


def landmark_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Ablate the independent landmark detector")
    ap.add_argument("--durations", type=str, default="0,15,30")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--out", type=str, default="out/landmark.csv")
    args = ap.parse_args(argv)

    durations = [float(x) for x in args.durations.split(",") if x != ""]
    rows = landmark_study(outage_durations=durations, n_per_cell=args.n,
                          duration=args.duration, seed=args.seed)
    write_csv(rows, args.out)
    print(f"[lm] wrote {args.out}")
    return 0


def factorgraph_live_study(
    faults: list[dict] | None = None,
    n_per_cell: int = 3,
    duration: float = 40.0,
    seed: int = 77,
) -> list[dict]:
    """Ablate the *calibrated factor graph as a live safety signal*.

    Safety layer ON in every cell.  We compare:
      - factor-graph live detector ON (landmark detector OFF, so the FG is the
        only independent signal)
      - factor-graph live detector OFF (landmark detector OFF too, so nothing
        independent is watching — only the flow self-report)

    This directly answers: "now that the FG is calibrated, does it convert a
    corrupt velocity fault into a safe reaction without false alarms on a
    healthy mission?"
    """
    if faults is None:
        faults = [
            {"name": "none",     "scale": 1.0, "bias_ramp": 0.0},
            {"name": "scale1.5", "scale": 1.5, "bias_ramp": 0.0},
            {"name": "bias.25",  "scale": 1.0, "bias_ramp": 0.25},
            {"name": "bias.1",   "scale": 1.0, "bias_ramp": 0.10},
        ]

    rng = np.random.default_rng(seed)
    base = [random_scenario(rng, duration=duration, index=k) for k in range(n_per_cell)]
    rows = []
    for fault in faults:
        for fg_enabled in (True, False):
            scenarios = []
            for s in base:
                s2 = Scenario(**{**s.as_dict(),
                                "flow_scale": float(fault["scale"]),
                                "flow_bias_ramp": float(fault["bias_ramp"]),
                                "safety_enabled": True,
                                "landmark_enabled": False,
                                "factorgraph_enabled": fg_enabled})
                scenarios.append(s2)
            metrics_list = []
            for s in scenarios:
                m, _ = run_scenario(s, record=False)
                metrics_list.append(m)
            agg = average_of_metrics(metrics_list, [
                "in_bounds_frac", "crash", "landed", "safety_fraction",
            ])
            outcomes = [m.get("safety_outcome", "none") for m in metrics_list]
            row = {
                "flow_fault": fault["name"],
                "flow_scale": float(fault["scale"]),
                "flow_bias_ramp": float(fault["bias_ramp"]),
                "fg_live": "on" if fg_enabled else "off",
                "n": n_per_cell,
                "mean_in_bounds": agg.get("in_bounds_frac", float("nan")),
                "mean_crash": agg.get("crash", float("nan")),
                "mean_landed": agg.get("landed", float("nan")),
                "mean_safety_fraction": agg.get("safety_fraction", float("nan")),
                "outcomes": ";".join(outcomes),
            }
            rows.append(row)
            print(f"[fg-live] fault={fault['name']:8s} "
                  f"fg_live={'on ' if fg_enabled else 'off'} "
                  f"in_bounds={row['mean_in_bounds']:.3f} "
                  f"crash={row['mean_crash']:.3f} "
                  f"landed={row['mean_landed']:.3f} "
                  f"outcome={row['outcomes']}")
    return rows


def factorgraph_live_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Ablate the calibrated factor-graph live safety signal")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", type=str, default="out/factorgraph_live.csv")
    args = ap.parse_args(argv)

    rows = factorgraph_live_study(n_per_cell=args.n, duration=args.duration,
                                  seed=args.seed)
    write_csv(rows, args.out)
    print(f"[fg-live] wrote {args.out}")
    return 0


def factorgraph_detector_study(
    faults: list[dict] | None = None,
    n_per_cell: int = 3,
    duration: float = 40.0,
    seed: int = 55,
) -> list[dict]:
    """Characterise the factor-graph consistency *detector* in isolation.

    Safety layer is DISABLED and flow is intentionally left on (so the graph
    has a 'suspect' velocity factor to test).  We record the max flow residual
    and the min factor-graph health over each mission.  The question is: does
    the post-optimisation flow residual cleanly separate a healthy source from
    a corrupt one, independent of the safety-loop reaction time?

    This is the diagnostic form of the multi-hypothesis idea: instead of asking
    "did we land safely?", we ask "did the graph's own residual detect the bad
    factor?".
    """
    if faults is None:
        faults = [
            {"name": "none",     "scale": 1.0, "bias_ramp": 0.0},
            {"name": "scale1.8", "scale": 1.8, "bias_ramp": 0.0},
            {"name": "bias.25",  "scale": 1.0, "bias_ramp": 0.25},
            {"name": "bias.2",   "scale": 1.0, "bias_ramp": 2.0},
        ]

    rng = np.random.default_rng(seed)
    base = [random_scenario(rng, duration=duration, index=k) for k in range(n_per_cell)]
    rows = []
    for fault in faults:
        scenarios = []
        for s in base:
            s2 = Scenario(**{**s.as_dict(),
                            "flow_scale": float(fault["scale"]),
                            "flow_bias_ramp": float(fault["bias_ramp"]),
                            "safety_enabled": False,
                            "factorgraph_enabled": True,
                            "landmark_enabled": True})
            scenarios.append(s2)

        fg_resids = []
        fg_healths = []
        lm_healths = []
        for s in scenarios:
            _, run = run_scenario(s, record=True)
            rr = np.asarray(run.factorgraph_residual, dtype=float)
            hh = np.asarray(run.factorgraph_health, dtype=float)
            ll = np.asarray(run.landmark_score, dtype=float)
            if rr.size:
                fg_resids.append(float(np.nanmax(rr)))
                fg_healths.append(float(np.min(hh)))
                lm_healths.append(float(np.min(ll)))

        row = {
            "flow_fault": fault["name"],
            "flow_scale": float(fault["scale"]),
            "flow_bias_ramp": float(fault["bias_ramp"]),
            "n": n_per_cell,
            "fg_max_residual_mps": float(np.mean(fg_resids)) if fg_resids else float("nan"),
            "fg_min_health": float(np.mean(fg_healths)) if fg_healths else float("nan"),
            "lm_min_health": float(np.mean(lm_healths)) if lm_healths else float("nan"),
        }
        rows.append(row)
        print(f"[fg] fault={fault['name']:8s} "
              f"residual={row['fg_max_residual_mps']:.3f} "
              f"health={row['fg_min_health']:.3f} "
              f"lm={row['lm_min_health']:.3f}")
    return rows


def consensus_study(
    faults: list[dict] | None = None,
    n_per_cell: int = 6,
    duration: float = 45.0,
    seed: int = 191,
    mission_aware: bool = False,
) -> list[dict]:
    """Compare consensus policies between the two independent detectors.

    Each cell uses the SAME random missions across policies so the comparison is
    paired.  The policies are:
      none      - no independent monitor (only flow self-report)
      lm_only   - landmark detector only
      fg_only   - calibrated factor graph only
      min       - worst-of (OR): any bad detector triggers
      max       - best-of (AND): both must agree a fault exists
      geom      - geometric mean (soft consensus)
    """
    if faults is None:
        faults = [
            {"name": "none",     "scale": 1.0, "bias_ramp": 0.0},
            {"name": "scale1.5", "scale": 1.5, "bias_ramp": 0.0},
            {"name": "bias.1",   "scale": 1.0, "bias_ramp": 0.10},
            {"name": "bias.25",  "scale": 1.0, "bias_ramp": 0.25},
        ]
    rng = np.random.default_rng(seed)
    base = [random_scenario(rng, duration=duration, index=k) for k in range(n_per_cell)]
    policies = [
        ("none",  False, False, "min"),
        ("lm_only", True, False, "min"),
        ("fg_only", False, True, "min"),
        ("min",   True, True, "min"),
        ("max",   True, True, "max"),
        ("geom",  True, True, "geom"),
    ]
    rows = []
    for fault in faults:
        for label, lm_on, fg_on, consensus in policies:
            scenarios = []
            for s in base:
                s2 = Scenario(**{**s.as_dict(),
                                 "flow_scale": float(fault["scale"]),
                                 "flow_bias_ramp": float(fault["bias_ramp"]),
                                 "safety_enabled": True,
                                 "landmark_enabled": lm_on,
                                 "factorgraph_enabled": fg_on,
                                 "detector_consensus": consensus,
                                 "mission_aware": mission_aware})
                scenarios.append(s2)
            metrics_list = []
            outcomes = []
            for s in scenarios:
                m, _ = run_scenario(s, record=False)
                metrics_list.append(m)
                outcomes.append(m.get("safety_outcome", "none"))
            agg = average_of_metrics(metrics_list, ["in_bounds_frac", "crash",
                                                    "landed", "safety_fraction"])
            row = {
                "flow_fault": fault["name"],
                "flow_scale": float(fault["scale"]),
                "flow_bias_ramp": float(fault["bias_ramp"]),
                "policy": label,
                "n": n_per_cell,
                "mean_in_bounds": agg.get("in_bounds_frac", float("nan")),
                "mean_crash": agg.get("crash", float("nan")),
                "mean_landed": agg.get("landed", float("nan")),
                "mean_safety_fraction": agg.get("safety_fraction", float("nan")),
                "outcomes": ";".join(outcomes),
            }
            rows.append(row)
            print(f"[cons] fault={fault['name']:8s} policy={label:8s} "
                  f"in_bounds={row['mean_in_bounds']:.3f} "
                  f"crash={row['mean_crash']:.3f} "
                  f"landed={row['mean_landed']:.3f} "
                  f"outcome={row['outcomes']}")
    return rows


def consensus_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Consensus policy comparison between independent detectors")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=191)
    ap.add_argument("--mission-aware", action="store_true")
    ap.add_argument("--out", type=str, default="out/consensus.csv")
    args = ap.parse_args(argv)
    rows = consensus_study(n_per_cell=args.n, duration=args.duration,
                           seed=args.seed, mission_aware=args.mission_aware)
    write_csv(rows, args.out)
    print(f"[cons] wrote {args.out}")
    return 0


def factorgraph_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Characterise the factor-graph consistency detector")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--out", type=str, default="out/factorgraph.csv")
    args = ap.parse_args(argv)

    rows = factorgraph_detector_study(n_per_cell=args.n, duration=args.duration,
                                      seed=args.seed)
    write_csv(rows, args.out)
    print(f"[fg] wrote {args.out}")
    return 0


def corruption_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run velocity-aiding corruption study")
    ap.add_argument("--durations", type=str, default="0,15,30")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="out/corruption.csv")
    args = ap.parse_args(argv)

    durations = [float(x) for x in args.durations.split(",") if x != ""]
    rows = corruption_study(outage_durations=durations, n_per_cell=args.n,
                            duration=args.duration, seed=args.seed)
    write_csv(rows, args.out)
    print(f"[corr] wrote {args.out}")
    return 0


def batch_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run a random batch of AIR Lab scenarios")
    ap.add_argument("-n", "--num", type=int, default=20)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="out/batch.csv")
    args = ap.parse_args(argv)

    rows, _ = random_batch(args.num, args.duration, args.seed)
    write_csv(rows, args.out)
    print(f"[batch] wrote {args.out}")

    if rows:
        best = min(rows, key=lambda r: r.get("pos_rmse", float("inf")))
        worst = max(rows, key=lambda r: r.get("pos_rmse", -float("inf")))
        print(f"[batch] best {best['name']} pos_rmse={best['pos_rmse']:.3f} "
              f"in_bounds={best['in_bounds_frac']:.2f} energy={best['energy_wh']:.2f}Wh")
        print(f"[batch] worst {worst['name']} pos_rmse={worst['pos_rmse']:.3f} "
              f"in_bounds={worst['in_bounds_frac']:.2f} energy={worst['energy_wh']:.2f}Wh")

        # quick aggregate
        for k in ["in_bounds_frac", "ground_collision_frac", "pos_rmse",
                  "horizontal_rmse", "vertical_rmse", "energy_wh"]:
            vals = [float(r[k]) for r in rows if k in r]
            if vals:
                print(f"[batch] mean {k:24s} = {float(np.mean(vals)):.3f}")
    return 0


def degradation_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run a GNSS-outage degradation study")
    ap.add_argument("--durations", type=str, default="0,5,10,15,20,25")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-flow", action="store_true", help="Also run with optical flow disabled.")
    ap.add_argument("--out", type=str, default="out/degradation.csv")
    args = ap.parse_args(argv)

    durations = [float(x) for x in args.durations.split(",") if x != ""]
    rows = degradation_study(durations, args.n, args.duration, args.seed,
                             compare_flow=not args.no_flow)
    write_csv(rows, args.out)
    print(f"[deg] wrote {args.out}")
    return 0
