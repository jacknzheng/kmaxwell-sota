import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "gauge_norm_permatrix.json"
OUT = Path(__file__).resolve().parent / "gauge_factorization.png"

with DATA.open() as f:
    records = list(json.load(f).values())

norm = np.array([r["norm_slope"] for r in records])
residual = np.array([r["residual_slope"] for r in records])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
specs = [
    (axes[0], norm, "A. Weight-norm response", "#5b4ee8", (-1.5, 1.2)),
    (axes[1], residual, "B. Curvature left after removing norm", "#16877b", (-1.5, 1.2)),
]

bins = np.linspace(-1.5, 1.2, 28)
for ax, values, title, color, limits in specs:
    ax.hist(values, bins=bins, color=color, alpha=0.22, edgecolor="white")
    ymax = ax.get_ylim()[1]
    # Each short tick is one matrix; deterministic vertical jitter separates overlaps.
    order = np.argsort(values)
    levels = (np.arange(len(values)) % 5) * ymax * 0.018 + ymax * 0.025
    ax.scatter(values[order], levels, s=16, color=color, alpha=0.78,
               edgecolors="none", zorder=3)
    median = np.median(values)
    ax.axvline(median, color=color, lw=2)
    ax.axvline(0, color="#1f2937", lw=1, ls="--", alpha=0.55)
    ax.set_xlim(*limits)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xlabel("log–log slope versus learning-rate multiplier")
    ax.set_ylabel("number of matrices")
    ax.text(0.03, 0.91, f"median {median:+.2f}", transform=ax.transAxes,
            color=color, weight="bold")
    ax.grid(axis="y", color="#dbe1e8", lw=0.8, alpha=0.8)

axes[0].axvline(1, color="#d97706", lw=1.5, ls=":")
axes[0].text(1.0, axes[0].get_ylim()[1] * 0.78, "  exact norm law: +1",
             color="#9a5800", va="center", fontsize=9)

fig.suptitle("The inverse-curvature law splits into a universal norm term and a heterogeneous residual",
             fontsize=15, weight="bold", x=0.02, ha="left")
fig.savefig(OUT, dpi=190, facecolor="white")
print(OUT)
