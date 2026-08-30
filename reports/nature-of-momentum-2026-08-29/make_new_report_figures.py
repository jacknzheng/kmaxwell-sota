import matplotlib.pyplot as plt
import numpy as np


INK = "#17202a"
MUTED = "#5b6775"
PURPLE = "#5b4ee8"
GREEN = "#169873"
ORANGE = "#e28b18"
RED = "#cf4a47"


def finish_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aeb7c2")
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color=INK, alpha=0.09, linewidth=0.8)


def save(fig, name):
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(name, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gains():
    labels = ["scheduled\nsingle EMA", "scheduled\nK8"]
    values = np.array([0.000285, 0.00277]) * 1000
    errors = np.array([0.000062, 0.000065]) * 1000
    colors = [ORANGE, PURPLE]

    fig, ax = plt.subplots(figsize=(8.4, 4.65))
    x = np.arange(2)
    ax.bar(x, values, yerr=errors, width=0.58, color=colors, capsize=5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("validation-loss reduction vs bi-Maxwell  (×10⁻³)", color=INK)
    ax.set_title(
        "K8 gains about ten times as much as one scheduled EMA",
        loc="left",
        fontsize=15,
        color=INK,
        pad=18,
        fontweight="semibold",
    )
    ax.text(
        0,
        1.015,
        "Mean ± standard error across four paired seeds",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )
    finish_axis(ax)
    save(fig, "gain_decomposition.png")


def pinning():
    eta = np.array([0.60, 0.77, 1.15, 1.30, 1.70])
    curvature = np.array([2.56, 1.82, 1.147, 0.92, 0.675])
    ref_eta = np.geomspace(0.56, 1.82, 200)
    log_eta = np.log(eta)
    log_curvature = np.log(curvature)
    slope, log_intercept = np.polyfit(log_eta, log_curvature, 1)
    fitted_curve = np.exp(log_intercept) * ref_eta**slope
    inverse_intercept = np.exp(np.mean(log_curvature + log_eta))
    inverse_curve = inverse_intercept / ref_eta

    fig, ax = plt.subplots(figsize=(8.4, 4.85))
    ax.plot(
        ref_eta,
        inverse_curve,
        linestyle="--",
        linewidth=2.0,
        color="#98a2b3",
        label=f"slope −1 reference:  {inverse_intercept:.2f} · s⁻¹",
    )
    ax.plot(
        ref_eta,
        fitted_curve,
        linewidth=2.4,
        color=ORANGE,
        label=rf"log–log fit:  {np.exp(log_intercept):.2f} · $s^{{{slope:.2f}}}$",
    )
    ax.scatter(eta, curvature, s=78, color=PURPLE, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.54, 1.85)
    ax.set_ylim(0.58, 2.95)
    ax.set_xticks(eta, ["0.60", "0.77", "1.15", "1.30", "1.70"])
    ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0], ["0.6", "0.8", "1.0", "1.5", "2.0", "3.0"])
    ax.set_xlabel("learning-rate multiplier  s", color=INK)
    ax.set_ylabel("equilibrium curvature  (relative units)", color=INK)
    ax.set_title(
        "Equilibrium curvature scales approximately as s⁻¹·²⁸",
        loc="left",
        fontsize=15,
        color=INK,
        pad=18,
        fontweight="semibold",
    )
    ax.text(
        0,
        1.015,
        "Five accepted constant-learning-rate segments; both axes are logarithmic",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    finish_axis(ax)
    save(fig, "eos_pinning.png")


def flip_screen():
    labels = ["classical 1/F\nreference", "measured late\ntrajectory"]
    values = [1 / 11, 3]
    colors = ["#9aa5b1", RED]

    fig, ax = plt.subplots(figsize=(8.4, 4.55))
    ax.bar(np.arange(2), values, width=0.50, color=colors)
    ax.axhline(1, color=INK, alpha=0.35, linewidth=1.2)
    ax.text(-0.32, 1.05, "bi-Maxwell reference", ha="left", va="bottom", fontsize=9.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.055, 5.2)
    ax.set_yticks([0.1, 0.3, 1, 3], ["0.1×", "0.3×", "1×", "3×"])
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("curvature ratio vs bi-Maxwell  (log scale)", color=INK)
    ax.set_title(
        "The momentum-free screen was measured after period-2 energy decayed",
        loc="left",
        fontsize=15,
        color=INK,
        pad=18,
        fontweight="semibold",
    )
    ax.text(
        0,
        1.015,
        "Fork at step 2400; curvature measured over steps 2432–3231",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )
    finish_axis(ax)
    save(fig, "flip_prediction_screen.png")


if __name__ == "__main__":
    gains()
    pinning()
    flip_screen()
