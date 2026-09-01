"""Command-line entry point and demo.

    python -m airlab

Runs the baseline mission, then a GNSS-outage variant, prints a comparison
of evaluation metrics, and writes a small summary CSV + optional PNG traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .simulator import Simulator, SimConfig, SimRun
from .metrics import evaluate_mission


def _metrics(run: SimRun, dt: float) -> dict:
    return evaluate_mission(
        run.true_pos, run.est_pos, run.ref_pos,
        run.ref_vel, run.est_vel, run.gps_available, dt,
    )


def _plot(runs: dict[str, SimRun], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped: {exc}]")
        return

    fig, ax = plt.subplots(5, 1, figsize=(9, 12))
    for name, r in runs.items():
        T = np.asarray(r.t)
        true_pos = np.asarray(r.true_pos)
        est_pos = np.asarray(r.est_pos)
        ref_pos = np.asarray(r.ref_pos)
        true_rpy = np.asarray(r.true_rpy)
        est_rpy = np.asarray(r.est_rpy)
        cost = np.asarray(r.cost)
        ax[0].plot(T, true_pos[:, 0], label=f"{name} true N")
        ax[0].plot(T, est_pos[:, 0], "--", label=f"{name} est N")
        ax[0].plot(T, ref_pos[:, 0], ":", label=f"{name} ref N")
        ax[1].plot(T, true_pos[:, 1], label=f"{name} true E")
        ax[1].plot(T, est_pos[:, 1], "--", label=f"{name} est E")
        ax[2].plot(T, -true_pos[:, 2], label=f"{name} true alt")
        ax[2].plot(T, -est_pos[:, 2], "--", label=f"{name} est alt")
        ax[3].plot(T, np.rad2deg(true_rpy[:, 2]), label=f"{name} yaw")
        ax[3].plot(T, np.rad2deg(est_rpy[:, 2]), "--", label=f"{name} est yaw")
        ax[4].plot(T, cost, label=f"{name} est error")
    for i in range(5):
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=7, ncol=3)
    ax[0].set_ylabel("N (m)")
    ax[1].set_ylabel("E (m)")
    ax[2].set_ylabel("alt (m)")
    ax[3].set_ylabel("yaw (deg)")
    ax[4].set_ylabel("|est err| (m)")
    ax[4].set_xlabel("time (s)")
    fig.suptitle("AIR Lab — autonomy stack trace (true / EKF / ref)")
    fig.tight_layout()
    path = os.path.join(out_dir, "trace.png")
    fig.savefig(path, dpi=130)
    print(f"[plot] wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    # baseline
    cfg = SimConfig()
    cfg.duration = args.duration
    sim = Simulator(cfg)
    baseline = sim.run()

    # GNSS outage scenario
    cfg2 = SimConfig()
    cfg2.duration = args.duration
    cfg2.gps_outage = (args.duration * 0.3, args.duration * 0.6)
    sim2 = Simulator(cfg2)
    outage = sim2.run()

    runs = {"baseline": baseline, "gps_outage": outage}
    rows = []
    for name, r in runs.items():
        m = _metrics(r, cfg.dt)
        rows.append({"scenario": name, **m})
        print(f"[{name}] pos_rmse={m['pos_rmse']:.3f} m  "
              f"horiz={m['horizontal_rmse']:.3f}  vert={m['vertical_rmse']:.3f}  "
              f"in_bounds={m['in_bounds_frac']:.3f}  gps={m['gps_available_frac']:.2f}")

    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {csv_path}")

    # print a compact comparison
    print("\n=== comparison ===")
    for k in ["pos_rmse", "horizontal_rmse", "vertical_rmse", "in_bounds_frac", "gps_available_frac"]:
        a = rows[0][k]
        b = rows[1][k]
        print(f"{k:24s} baseline={a:.3f}  gnss_outage={b:.3f}  (delta={b - a:+.3f})")

    if args.plot:
        _plot(runs, args.outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
