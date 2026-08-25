#!/usr/bin/env python3
"""val_ema curves: K6_a35 on #46 vs published #46 A40 logs.

  python3 records/track_3_optimization/results/20260824_kmaxwell_2680/plot_cwd_compare.py
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EMA_RE = re.compile(r"step:(\d+)/\d+ val_loss:[0-9.]+(?: val_avg_loss:[0-9.]+)? val_ema_loss:([0-9.]+)")

FLEETS = {
    "cwd46": {
        "label": "#46 SOAP-CWD (n=8, 2690)",
        "color": "#4a4a4a",
        "globs": [ROOT / "records/track_3_optimization/results/20260619_cwd_rowfloor_tailema/A40_seed*.txt"],
        "pass_step": 2690,
    },
    "k6": {
        "label": "K6_a35 on #46 (n=8, 2680)",
        "color": "#d62728",
        "globs": [HERE / "H100_seed*.txt"],
        "pass_step": 2680,
    },
}


def parse_file(path: Path) -> dict[int, float]:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = EMA_RE.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def load_fleet(globs) -> list[dict[int, float]]:
    paths = []
    for g in globs:
        paths.extend(sorted(Path(g).parent.glob(Path(g).name)))
    runs = [parse_file(p) for p in paths]
    if not runs:
        raise SystemExit(f"no logs for {globs}")
    return runs


def mean_std(runs: list[dict[int, float]]):
    steps = sorted(set().union(*[r.keys() for r in runs]))
    xs, mu, sd = [], [], []
    for s in steps:
        vals = [r[s] for r in runs if s in r]
        if len(vals) < max(1, int(0.5 * len(runs))):
            continue
        xs.append(s)
        mu.append(float(np.mean(vals)))
        sd.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return np.array(xs), np.array(mu), np.array(sd)


def plot_panel(ax, fleets, xmin, xmax, ymin, ymax, title):
    ax.axhline(3.28, color="#888888", ls="--", lw=1.0, zorder=0)
    for key, spec in FLEETS.items():
        x, mu, sd = fleets[key]
        m = (x >= xmin) & (x <= xmax)
        ax.plot(x[m], mu[m], color=spec["color"], lw=2.0, label=spec["label"], zorder=3)
        ax.fill_between(x[m], mu[m] - sd[m], mu[m] + sd[m], color=spec["color"], alpha=0.18, lw=0, zorder=2)
        ps = spec["pass_step"]
        if xmin <= ps <= xmax:
            ax.axvline(ps, color=spec["color"], ls=":", lw=1.0, alpha=0.8, zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("step")
    ax.set_ylabel("val_ema")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    loaded = {k: load_fleet(v["globs"]) for k, v in FLEETS.items()}
    print({k: len(v) for k, v in loaded.items()})
    curves = {k: mean_std(v) for k, v in loaded.items()}

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plot_panel(ax, curves, 125, 2720, 3.26, 4.65, "SOAP-CWD val_ema (mean ± 1 sd)")
    fig.savefig(HERE / "figure.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plot_panel(ax, curves, 2550, 2720, 3.272, 3.292, "Target zone")
    ax.annotate("3.28", xy=(2555, 3.2804), color="#666666", fontsize=9)
    fig.savefig(HERE / "zoomed_figure.png")
    plt.close(fig)
    print("wrote", HERE / "figure.png", HERE / "zoomed_figure.png")


if __name__ == "__main__":
    main()
