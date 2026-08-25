#!/usr/bin/env python3
"""Val-loss curves: Muon anneal n=8 vs Track-3 #36 vs bi-Maxwell 3210.

  python3 logs/wr/plot_muon_compare.py
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VAL_RE = re.compile(r"step:(\d+)/\d+ val_loss:([0-9.]+)")

FLEETS = {
    "muon36": {
        "label": "#36 Muon (n=10, 3250)",
        "color": "#4a4a4a",
        "globs": [ROOT / "records/track_3_optimization/results/20260610_tuned_baseline_3250/*.txt"],
        "pass_step": 3250,
    },
    "bimaxwell": {
        "label": "bi-Maxwell (n=8, 3210)",
        "color": "#1f77b4",
        "globs": [
            ROOT / "records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/A800_seed*.txt",
        ],
        "pass_step": 3210,
    },
    "anneal": {
        "label": "K8 anneal 58→26 (n=8, 3160)",
        "color": "#d62728",
        "globs": [HERE / "muon_anneal_3160/seed*.stdout"],
        "pass_step": 3160,
    },
}


def parse_file(path: Path) -> dict[int, float]:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = VAL_RE.search(line)
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
    ax.set_ylabel("val loss")
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
    plot_panel(ax, curves, 125, 3250, 3.26, 4.65, "Muon val loss (mean ± 1 sd)")
    fig.savefig(HERE / "figure.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plot_panel(ax, curves, 2875, 3250, 3.268, 3.292, "Target zone")
    ax.annotate("3.28", xy=(2880, 3.2804), color="#666666", fontsize=9)
    fig.savefig(HERE / "zoomed_figure.png")
    plt.close(fig)

    out = HERE / "compare_tail.tsv"
    steps = [3150, 3160, 3170, 3175, 3180, 3200, 3210, 3225, 3250]
    with out.open("w") as f:
        f.write("step\tmuon36\tbimaxwell\tanneal\n")
        for s in steps:
            row = [str(s)]
            for key in ("muon36", "bimaxwell", "anneal"):
                x, mu, _ = curves[key]
                hit = np.where(x == s)[0]
                row.append(f"{mu[hit[0]]:.5f}" if len(hit) else "")
            f.write("\t".join(row) + "\n")
    print("wrote", HERE / "figure.png", HERE / "zoomed_figure.png", out)


if __name__ == "__main__":
    main()
