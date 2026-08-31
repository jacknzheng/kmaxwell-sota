import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(7.4, 4.5))
F = np.array([0.000, 0.010, 0.020, 0.041, 0.060, 0.089])
loss = np.array([3.32022, 3.28497, 3.27881, 3.27832, 3.28252, 3.27691])
ax.axvspan(0.02, 0.09, color="#dff3ed", zorder=0)
ax.plot(F, loss, color="#5746d9", lw=2.2)
ax.scatter(F, loss, s=46, color="#5746d9", zorder=3)
ax.annotate("removing period-2 response\nbreaks training", (0, 3.320),
            xytext=(0.012, 3.312), arrowprops={"arrowstyle": "->", "color": "#596575"},
            color="#3c4653")
ax.text(0.052, 3.302, "broad working range", ha="center", color="#16877b", weight="bold")
ax.set(xlabel=r"Period-2 response $F=|H(\pi)|$", ylabel="Validation loss after continuation",
       title="Period-2 response behaves like a guardrail, not a tuning target")
ax.grid(axis="y", alpha=.18)
fig.tight_layout()
fig.savefig("period2_scan.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.65))
x = np.array([0.6, 1.0, 1.7])
series = {
    r"curvature $\lambda$": ([.560, -.014, -.876], "#5746d9", "o"),
    r"squared weight norm $\|W\|^2$": ([-.410, .004, .546], "#a35e00", "s"),
    r"residual $R=\lambda\|W\|^2$": ([.218, -.014, -.423], "#16877b", "D"),
}
for label, (y, color, marker) in series.items():
    ax.plot(x, y, color=color, marker=marker, ms=7, lw=2, label=label)
ax.axhline(0, color="#8b95a2", lw=1)
ax.axvline(1, color="#c8ced6", lw=1, ls="--")
ax.set_xscale("log")
ax.set_xticks(x, ["0.6×", "1×", "1.7×"])
ax.set(xlabel="Learning-rate multiplier assigned to a matrix",
       ylabel="Median log change from the shared starting state",
       title="Per-matrix learning rate causally moves the local sharpness state")
ax.legend(frameon=False, loc="lower left")
ax.grid(axis="y", alpha=.18)
fig.tight_layout()
fig.savefig("per_matrix_response.png", dpi=180)
plt.close(fig)
