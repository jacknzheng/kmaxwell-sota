import matplotlib.pyplot as plt
from pathlib import Path


batch_multiplier = [1, 4, 8, 16]
validation_loss_change = [-0.0106, -0.0044, -0.0023, 0.0]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
})

fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
ax.axhline(0.0, color="#7b8288", linewidth=1.2, zorder=1)
ax.plot(
    range(4),
    validation_loss_change,
    color="#74308d",
    marker="o",
    markersize=8,
    linewidth=2.5,
    zorder=2,
)
ax.set_xticks(range(4), ["1×\n0.52M", "4×\n2.10M", "8×\n4.19M", "16×\n8.39M"])
ax.set_xlabel("batch multiplier and tokens per step")
ax.set_ylabel("validation-loss change vs no momentum\n(negative is better)")
ax.set_title("Bi-Maxwell benefit reaches zero at 16× batch")
ax.grid(axis="y", color="#d9dde1", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(-0.0118, 0.0012)
ax.annotate(
    "measured zero",
    xy=(3, 0),
    xytext=(2.55, -0.0015),
    arrowprops={"arrowstyle": "-", "color": "#58606a"},
    color="#58606a",
)
fig.savefig(Path(__file__).with_name("batch_benefit.png"), dpi=150)
