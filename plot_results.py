#!/usr/bin/env python3
"""Generate summary charts from batch / degradation CSV outputs."""

import csv
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print(f"[plot] matplotlib unavailable: {exc}")
    raise SystemExit(1)


def _read(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_batch(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return
    rmse = np.array([float(r["pos_rmse"]) for r in rows if r.get("pos_rmse")])
    energy = np.array([float(r["energy_wh"]) for r in rows if r.get("energy_wh")])
    inb = np.array([float(r["in_bounds_frac"]) for r in rows if r.get("in_bounds_frac")])

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hist(rmse, bins=min(12, max(4, int(np.sqrt(len(rmse))))), color="#2b8cbe", alpha=0.8)
    ax[0].set_xlabel("position RMSE (m)")
    ax[0].set_ylabel("scenarios")
    ax[0].set_title(f"n={len(rmse)}\nmean {rmse.mean():.3f} m")

    ax[1].bar([0], [float(np.mean(energy))], color="#31a354", alpha=0.8)
    ax[1].set_ylabel("energy (Wh)")
    ax[1].set_title(f"mean {np.mean(energy):.3f} Wh")

    ax[2].bar([0], [float(np.mean(inb))], color="#de2d26", alpha=0.8)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("in-bounds fraction")
    ax[2].set_title(f"mean {np.mean(inb):.3f}")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] batch -> {out}")


def plot_degradation(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return

    durs = sorted({float(r["outage_duration_s"]) for r in rows})
    variants = sorted({r.get("velocity_aiding", "flow") for r in rows})
    markers = {"flow": "o", "no_flow": "s"}
    colors = {"flow": "#2b8cbe", "no_flow": "#de2d26"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for variant in variants:
        sub = [r for r in rows if r.get("velocity_aiding", "flow") == variant]
        if not sub:
            continue
        d = np.array([float(r["outage_duration_s"]) for r in sub])
        order = np.argsort(d)
        d = d[order]
        inb = np.array([float(r["mean_in_bounds"]) for r in sub])[order]
        col = np.array([float(r["mean_collision"]) for r in sub])[order]
        rmse = np.array([float(r["mean_pos_rmse"]) for r in sub])[order]

        marker = markers.get(variant, "o")
        clr = colors.get(variant, "#31a354")
        label = "+" if variant == "flow" else "no flow"
        ax[0].plot(d, inb, marker=marker, linestyle="-", color=clr,
                   label=f"in-bounds ({label})")
        ax[0].plot(d, col, marker=marker, linestyle="--", color=clr, alpha=0.5,
                   label=f"collision ({label})")
        ax[1].plot(d, rmse, marker=marker, linestyle="-", color=clr, label=label)

    ax[0].set_xlabel("GNSS outage duration (s)")
    ax[0].set_ylabel("fraction")
    ax[0].set_ylim(0, 1.1)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    ax[0].set_title("Safety under degradation (with vs without velocity aiding)")

    ax[1].set_xlabel("GNSS outage duration (s)")
    ax[1].set_ylabel("mean position RMSE (m)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_title("Tracking cost under degradation")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] degradation -> {out}")


def plot_corruption(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return
    faults = sorted({r.get("flow_fault", "none") for r in rows})
    durations = sorted({float(r["outage_duration_s"]) for r in rows})
    markers = {"none": "o", "dropout": "s", "scale1.5": "^", "scale1.8": "v",
               "bias.1": "X", "bias.25": "D"}
    colors = {"on": "#2b8cbe", "off": "#de2d26"}
    linestyles = {"on": "-", "off": "--"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for fault in faults:
        for safety in ("on", "off"):
            sub = [r for r in rows
                   if r.get("flow_fault") == fault and r.get("safety_layer") == safety]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r["outage_duration_s"]))
            d = np.array([float(r["outage_duration_s"]) for r in sub])
            inb = np.array([float(r["mean_in_bounds"]) for r in sub])
            col = np.array([float(r["mean_collision"]) for r in sub])
            land = np.array([float(r["mean_landed"]) for r in sub])
            marker = markers.get(fault, "o")
            clr = colors.get(safety, "#31a354")
            ls = linestyles.get(safety, "-")
            label = f"{fault} / safety {safety}"
            ax[0].plot(d, inb, marker=marker, linestyle=ls, color=clr, label=label)
            ax[1].plot(d, col, marker=marker, linestyle=ls, color=clr,
                       label=f"{fault} / safety {safety}")
            ax[1].plot(d, land, marker=marker, linestyle=":", color=clr,
                       alpha=0.6)

    ax[0].set_xlabel("GNSS outage duration (s)")
    ax[0].set_ylabel("mean in-bounds fraction")
    ax[0].set_ylim(0, 1.1)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("Safety (in-bounds) vs corrupt velocity aiding")

    ax[1].set_xlabel("GNSS outage duration (s)")
    ax[1].set_ylabel("fraction")
    ax[1].set_ylim(0, 1.1)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].set_title("Collision (solid) / landed (dotted)")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] corruption -> {out}")


def plot_landmark(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return
    faults = sorted({r.get("flow_fault", "none") for r in rows})
    durations = sorted({float(r["outage_duration_s"]) for r in rows})
    markers = {"none": "o", "scale1.5": "^", "scale1.8": "v",
               "bias.1": "X", "bias.25": "D"}
    colors = {"on": "#2b8cbe", "off": "#de2d26"}
    linestyles = {"on": "-", "off": "--"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for fault in faults:
        for lm in ("on", "off"):
            sub = [r for r in rows
                   if r.get("flow_fault") == fault and r.get("landmark_detector") == lm]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r["outage_duration_s"]))
            d = np.array([float(r["outage_duration_s"]) for r in sub])
            crash = np.array([float(r["mean_crash"]) for r in sub])
            landed = np.array([float(r["mean_landed"]) for r in sub])
            marker = markers.get(fault, "o")
            clr = colors.get(lm, "#31a354")
            ls = linestyles.get(lm, "-")
            label = f"{fault} / lm {lm}"
            ax[0].plot(d, crash, marker=marker, linestyle=ls, color=clr, label=label)
            ax[1].plot(d, landed, marker=marker, linestyle=ls, color=clr, label=label)

    ax[0].set_xlabel("GNSS outage duration (s)")
    ax[0].set_ylabel("mean unintended crash")
    ax[0].set_ylim(0, 1.1)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("Landmark detector: crash")

    ax[1].set_xlabel("GNSS outage duration (s)")
    ax[1].set_ylabel("mean landed safely")
    ax[1].set_ylim(0, 1.1)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].set_title("Landmark detector: controlled landing")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] landmark -> {out}")


def plot_factorgraph(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return
    names = [r.get("flow_fault", "?") for r in rows]
    res = np.array([float(r["fg_max_residual_mps"]) for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(names, res, color="#2b8cbe")
    ax[0].set_ylabel("max flow residual (m/s)")
    ax[0].set_title("Factor-graph flow residual")
    ax[0].tick_params(axis="x", rotation=25)

    lm = np.array([float(r["lm_min_health"]) for r in rows])
    ax[1].bar(names, lm, color="#31a354")
    ax[1].set_ylabel("min landmark health")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Landmark consistency (for comparison)")
    ax[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] factorgraph -> {out}")


def plot_factorgraph_live(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return
    faults = sorted({r.get("flow_fault", "none") for r in rows})
    markers = {"none": "o", "scale1.5": "^", "scale1.8": "v",
               "bias.1": "X", "bias.25": "D"}
    colors = {"on": "#2b8cbe", "off": "#de2d26"}
    linestyles = {"on": "-", "off": "--"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for fault in faults:
        for fg in ("on", "off"):
            sub = [r for r in rows if r.get("flow_fault") == fault and r.get("fg_live") == fg]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r["flow_bias_ramp"]))
            x = np.arange(len(sub))
            marker = markers.get(fault, "o")
            clr = colors.get(fg, "#31a354")
            ls = linestyles.get(fg, "-")
            crash = np.array([float(r["mean_crash"]) for r in sub])
            landed = np.array([float(r["mean_landed"]) for r in sub])
            label = f"{fault} / fg {fg}"
            ax[0].plot(x, crash, marker=marker, linestyle=ls, color=clr, label=label)
            ax[1].plot(x, landed, marker=marker, linestyle=ls, color=clr, label=label)
    ax[0].set_xticks(np.arange(len(faults)))
    ax[0].set_xticklabels(faults)
    ax[0].set_ylabel("mean unintended crash")
    ax[0].set_ylim(0, 1.1)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    ax[0].set_title("Factor-graph live: crash")
    ax[1].set_xticks(np.arange(len(faults)))
    ax[1].set_xticklabels(faults)
    ax[1].set_ylabel("mean landed safely")
    ax[1].set_ylim(0, 1.1)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=6)
    ax[1].set_title("Factor-graph live: controlled landing")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] factorgraph live -> {out}")


def plot_consensus(path: str, out: str) -> None:
    rows = _read(path)
    if not rows:
        return
    faults = sorted({r.get("flow_fault", "none") for r in rows})
    policies = ["none", "lm_only", "fg_only", "min", "max", "geom", "adaptive"]
    colors = {
        "none": "#666666", "lm_only": "#2b8cbe", "fg_only": "#de2d26",
        "min": "#31a354", "max": "#756bb1", "geom": "#ff7f00",
        "adaptive": "#e377c2",
    }
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.8 / len(policies)
    for f, fault in enumerate(faults):
        for p, pol in enumerate(policies):
            sub = [r for r in rows if r.get("flow_fault") == fault and r.get("policy") == pol]
            if not sub:
                continue
            crash = float(sub[0].get("mean_crash", float("nan")))
            landed = float(sub[0].get("mean_landed", float("nan")))
            x = f + (p - len(policies) / 2.0) * width
            ax[0].bar(x, crash, width=width, color=colors[pol], alpha=0.8,
                      label=pol if f == 0 else None)
            ax[1].bar(x, landed, width=width, color=colors[pol], alpha=0.8,
                      label=pol if f == 0 else None)
    for a in ax:
        a.set_xticks(np.arange(len(faults)))
        a.set_xticklabels(faults)
        a.grid(alpha=0.3, axis="y")
        a.legend(fontsize=8, ncol=2)
    ax[0].set_ylabel("mean unintended crash")
    ax[1].set_ylabel("mean landed fraction")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] consensus -> {out}")


def plot_consensus_roc(path: str, out: str) -> None:
    """Decision-cost ROC: landing-aggressiveness vs missed crash.

    For each policy we collect two numbers:
      x = how often it *acts* (lands) on faults that never crash when
          unwatched (none, mild scale/bias)  -> mission-completion cost
      y = how often it *fails* (unintended crash) on the hardest fault (bias.25)
          -> safety miss cost
    A good policy is near the bottom-left of this plot.
    """
    rows = _read(path)
    if not rows:
        return
    policies = sorted({r.get("policy", "") for r in rows})
    # "Benign" = faults where the *unwatched* vehicle stays in-bounds (>=0.88)
    # for the whole mission, so acting on them is a pure mission-completion cost
    # (not a justified safety response).  bias.05 is NOT benign: it stays crash
    # free but already leaves bounds (in_bounds ~0.41 on the unwatched arm).
    benign = {"none", "scale1.2", "scale1.5", "scale1.8"}
    colors = {
        "none": "#666666", "lm_only": "#2b8cbe", "fg_only": "#de2d26",
        "min": "#31a354", "max": "#756bb1", "geom": "#ff7f00",
        "adaptive": "#e377c2",
    }
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for pol in policies:
        sub = rows
        act = [float(r["mean_landed"]) for r in sub
               if r.get("policy") == pol and r.get("flow_fault") in benign]
        miss = [float(r["mean_crash"]) for r in sub
                if r.get("policy") == pol and r.get("flow_fault") == "bias.25"]
        if not act or not miss:
            continue
        x = float(np.mean(act))
        y = float(np.mean(miss))
        ax.scatter(x, y, s=90, color=colors.get(pol, "#31a354"), alpha=0.85,
                   zorder=3)
        ax.annotate(pol, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
    ax.set_xlabel("landing-aggressiveness (mean landed on benign faults)")
    ax.set_ylabel("missed crash (mean unintended crash @ bias.25)")
    ax.set_ylim(0, max(0.06, max([float(r["mean_crash"]) for r in rows
                                  if r.get("flow_fault") == "bias.25"] or [0.0]) * 1.2))
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title("Consensus decision cost (lower-left is better)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot] consensus ROC -> {out}")


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "out/batch.csv"
    deg = sys.argv[2] if len(sys.argv) > 2 else "out/degradation.csv"
    corr = sys.argv[3] if len(sys.argv) > 3 else "out/corruption.csv"
    lm = sys.argv[4] if len(sys.argv) > 4 else "out/landmark.csv"
    fg = sys.argv[5] if len(sys.argv) > 5 else "out/factorgraph.csv"
    fglive = sys.argv[6] if len(sys.argv) > 6 else "out/factorgraph_live.csv"
    consensus = sys.argv[7] if len(sys.argv) > 7 else "out/consensus.csv"
    outdir = "out"
    if os.path.exists(batch):
        plot_batch(batch, os.path.join(outdir, "batch.png"))
    if os.path.exists(deg):
        plot_degradation(deg, os.path.join(outdir, "degradation.png"))
    if os.path.exists(corr):
        plot_corruption(corr, os.path.join(outdir, "corruption.png"))
    if os.path.exists(lm):
        plot_landmark(lm, os.path.join(outdir, "landmark.png"))
    if os.path.exists(fg):
        plot_factorgraph(fg, os.path.join(outdir, "factorgraph.png"))
    if os.path.exists(fglive):
        plot_factorgraph_live(fglive, os.path.join(outdir, "factorgraph_live.png"))
    if os.path.exists(consensus):
        plot_consensus(consensus, os.path.join(outdir, "consensus.png"))
        plot_consensus_roc(consensus, os.path.join(outdir, "consensus_roc.png"))


if __name__ == "__main__":
    main()
