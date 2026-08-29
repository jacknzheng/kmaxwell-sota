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


def format_response_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(PERIODS)
    ax.set_xticklabels([str(p) for p in PERIODS])
    ax.set_xlim(1.8, 285)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("period of the repeating input (optimizer steps)")
    ax.set_ylabel("output amplitude / input amplitude")
    style_axis(ax)


def make_kmaxwell_response():
    fig, ax = plt.subplots(figsize=(8.6, 5.15))
    ax.plot(PERIODS, EMA, color="#64748b", linewidth=2.8, label="scheduled EMA")
    ax.plot(PERIODS, K6, color="#22c55e", linewidth=2.8, label="K6 endpoint")
    ax.plot(PERIODS, K8, color="#0ea5e9", linewidth=2.8, label="K8 endpoint")
    format_response_axis(ax)
    ax.set_title("How much of a repeating gradient remains after filtering?")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.95,
        "constant input (period → ∞): every curve → 1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#475569",
    )
    ax.text(
        0.98,
        0.05,
        "Calculated from endpoint kernel weights—not a training-run statistic",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    ax.annotate(
        "At period 8, K6 returns\namplitude 0.130 from a unit input",
        xy=(8, K6[4]),
        xytext=(14, 0.34),
        arrowprops={"arrowstyle": "-", "color": "#15803d", "lw": 1.1},
        fontsize=9.5,
        color="#166534",
    )
    fig.tight_layout()
    fig.savefig("kmaxwell_filter_response.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_powerlaw_response():
    fig, ax = plt.subplots(figsize=(8.6, 5.15))
    ax.plot(PERIODS, EMA, color="#64748b", linewidth=2.8, label="scheduled EMA")
    ax.plot(PERIODS, POWER, color="#ef4444", linewidth=2.8, label="nominally mean-lag-targeted power law")
    format_response_axis(ax)
    ax.set_title("Nominal mean-lag targeting produced a very different realized filter")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.05,
        "Calculated from endpoint kernel weights—not a training-run statistic",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#475569",
    )
    ax.annotate(
        "For a period-2 input:\npower law 0.400; EMA 0.077",
        xy=(2, POWER[0]),
        xytext=(4.2, 0.57),
        arrowprops={"arrowstyle": "-", "color": "#dc2626", "lw": 1.1},
        fontsize=9.5,
        color="#991b1b",
    )
    fig.tight_layout()
    fig.savefig("powerlaw_filter_response.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_empirical_trend()
    make_kmaxwell_response()
    make_powerlaw_response()
