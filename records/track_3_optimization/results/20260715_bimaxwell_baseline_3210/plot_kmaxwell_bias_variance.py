#!/usr/bin/env python3
"""Filter bias-variance plot: lag (mean age) vs Nesterov noise gain.

Closed-form only. Does not load a checkpoint or touch a training run.

    python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/plot_kmaxwell_bias_variance.py
    python3 .../plot_kmaxwell_bias_variance.py --out logs/kmaxwell/bias_variance.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kmaxwell_kernel import (
    BIMAXWELL_BF,
    BIMAXWELL_BS,
    BIMAXWELL_W,
    BIMAXWELL_TAU_MIN,
    BIMAXWELL_TAU_MAX,
    build_kmaxwell_kernel,
    nesterov_filter_stats,
)

MU = 0.95


def point(family, name, betas, weights):
    lag_m, lag_x, noise = nesterov_filter_stats(betas, weights, mu=MU)
    return {
        "family": family,
        "name": name,
        "lag_m": lag_m,
        "lag_x": lag_x,
        "noise_gain": noise,
    }


def collect_points():
    pts = []
    for beta in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
        pts.append(point("single-EMA", f"beta={beta}", [beta], [1.0]))
    pts.append(point(
        "bimaxwell-exact", "exact",
        [BIMAXWELL_BF, BIMAXWELL_BS], [BIMAXWELL_W, 1.0 - BIMAXWELL_W]))
    for k in (2, 3, 4, 6, 8, 12, 16):
        _, betas, weights, _ = build_kmaxwell_kernel(
            k, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, False)
        pts.append(point("stage1-K", f"K={k}", betas, weights))
    for tmax in (25, 49, 100, 200, 400):
        _, betas, weights, _ = build_kmaxwell_kernel(
            4, BIMAXWELL_TAU_MIN, float(tmax), 1.0, False)
        pts.append(point("stage2-tau_max", f"tmax={tmax}", betas, weights))
    return pts


def write_csv(pts, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "name", "lag_m", "lag_x", "noise_gain"])
        w.writeheader()
        w.writerows(pts)


def try_plot(pts, path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; wrote CSV only")
        return False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {
        "single-EMA": dict(marker="o", linestyle="-", color="0.45"),
        "bimaxwell-exact": dict(marker="*", linestyle="none", color="C3", s=220, zorder=5),
        "stage1-K": dict(marker="s", linestyle="none", color="C0"),
        "stage2-tau_max": dict(marker="D", linestyle="none", color="C2"),
    }
    for family, style in styles.items():
        sub = [p for p in pts if p["family"] == family]
        if not sub:
            continue
        xs = [p["lag_m"] for p in sub]
        ys = [p["noise_gain"] for p in sub]
        kw = dict(style)
        s = kw.pop("s", 70)
        ax.scatter(xs, ys, label=family, s=s, **{k: v for k, v in kw.items() if k != "linestyle"})
        if family == "single-EMA":
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order], color="0.45", linewidth=1)
        for p in sub:
            ax.annotate(p["name"], (p["lag_m"], p["noise_gain"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("bias / lag  (mix mean age, steps)")
    ax.set_ylabel("variance / noise gain  (Nesterov filter ||h||^2)")
    ax.set_title("K-Maxwell filter bias-variance  (mu=0.95, closed form)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(HERE / "bias_variance.png"))
    p.add_argument("--csv", default=str(HERE / "bias_variance.csv"))
    args = p.parse_args()
    pts = collect_points()
    print(f"{'family':<18} {'name':<12} {'lag_m':>8} {'lag_x':>8} {'noise_gain':>12}")
    for pt in pts:
        print(f"{pt['family']:<18} {pt['name']:<12} {pt['lag_m']:8.3f} {pt['lag_x']:8.3f} {pt['noise_gain']:12.5f}")
    write_csv(pts, Path(args.csv))
    print(f"wrote {args.csv}")
    try_plot(pts, Path(args.out))


if __name__ == "__main__":
    main()
