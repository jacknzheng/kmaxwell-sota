import cmath
import math

import matplotlib.pyplot as plt
import numpy as np


PERIODS = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256])


def ema_response(beta: float, mu: float = 0.95) -> np.ndarray:
    values = []
    for period in PERIODS:
        z = cmath.exp(2j * math.pi / period)
        response = (1 - mu) + mu * (1 - beta) / (1 - beta / z)
        values.append(abs(response))
    return np.asarray(values)


# Unit-DC endpoint responses. The K-series are the corrected descriptor
# outputs; the EMA response is recomputed for the best tested .982 -> .944
# schedule's endpoint. These are filter descriptors, not realized Muon update
# amplitudes.
EMA = ema_response(0.944)
K8 = np.asarray([
    0.0826, 0.0850, 0.0897, 0.1022, 0.1172, 0.1500,
    0.1831, 0.2445, 0.2982, 0.4561, 0.6380, 0.8091,
])
K6 = np.asarray([
    0.0877, 0.0908, 0.0966, 0.1118, 0.1297, 0.1679,
    0.2055, 0.2735, 0.3316, 0.4979, 0.6820, 0.8441,
])
POWER = np.asarray([
    0.3998, 0.4236, 0.4576, 0.5168, 0.5631, 0.6287,
    0.6729, 0.7300, 0.7665, 0.8417, 0.8947, 0.9271,
])


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#17202a", alpha=0.10, linewidth=0.8)
    ax.tick_params(labelsize=10)


def make_empirical_trend():
    # H100 matched-control discovery results, scaled to K8's observed gain.
    # Exact values and the A100 stack sensitivity belong in the appendix.
    shares = np.array([0.00098, 0.00286, 0.00297]) / 0.00297
    labels = ["one scheduled\nEMA", "scheduled\nK6", "scheduled\nK8"]
    colors = ["#f59e0b", "#22c55e", "#0ea5e9"]

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.arange(len(labels))
    ax.bar(x, shares, width=0.58, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 1 / 3, 2 / 3, 1], ["0", "about ⅓", "about ⅔", "K8"])
    ax.set_ylabel("observed improvement relative to K8")
    ax.set_title("Scheduling one memory timescale helps, but leaves most of the gap")
    ax.text(
        0.03,
        0.96,
        "K6 ≈ K8: adding coefficients from six to eight did not help",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#334155",
    )
    style_axis(ax)
    fig.tight_layout()
    fig.savefig("empirical_trend.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_kernel_shape():
    colors = {"K8": "#0ea5e9", "K6": "#22c55e", "power": "#ef4444"}
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.6, 7.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    ax1.plot(PERIODS, K8 / EMA, color=colors["K8"], linewidth=2.7, label="K8 / EMA")
    ax1.plot(PERIODS, K6 / EMA, color=colors["K6"], linewidth=2.7, label="K6 / EMA")
    ax1.axhline(1, color="#64748b", linewidth=1.2)
    ax1.axvspan(64, 256, color="#6366f1", alpha=0.08)
    ax1.set_ylim(0.80, 1.31)
    ax1.set_ylabel("response relative to EMA")
    ax1.set_title("K-Maxwell / scheduled-EMA response ratio")
    ax1.legend(frameon=False, loc="upper right")
    ax1.text(
        0.985,
        0.08,
        "candidate test band: 64–256 steps per cycle",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    style_axis(ax1)

    ax2.plot(PERIODS, POWER / EMA, color=colors["power"], linewidth=2.7)
    ax2.axhline(1, color="#64748b", linewidth=1.2)
    ax2.set_ylabel("response relative to EMA")
    ax2.set_title("Matched-age power-law / scheduled-EMA response ratio")
    ax2.text(
        0.985,
        0.88,
        "much more fast variation survives",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#991b1b",
    )
    ax2.text(
        0.985,
        0.07,
        "constant input (period ∞): every ratio → 1",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    style_axis(ax2)

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(PERIODS)
    ax2.set_xticklabels([str(p) for p in PERIODS])
    ax2.set_xlabel("steps per cycle  →  slower variation")

    fig.text(
        0.5,
        0.006,
        "Unit-DC anchor: every kernel has response 1 to a constant gradient. Ratios show temporal shape, not Muon update size.",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig("kernel_shape_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_empirical_trend()
    make_kernel_shape()
