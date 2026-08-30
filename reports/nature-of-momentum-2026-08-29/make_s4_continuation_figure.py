import json

import matplotlib.pyplot as plt
import numpy as np


INK = "#17202a"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
FLOOR = "#c7c9cc"


with open("s4_continuation_scores.json") as f:
    scores = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), sharey=True)
base = scores["1500"]

for ax, fork in zip(axes, ["1500", "2400", "2900"]):
    record = scores[fork]
    branch = record["branch"]

    for key, label, color in [("ka", "kernel A", BLUE), ("kb", "kernel B", ORANGE)]:
        gaps = record["loss"][key]["gaps"]
        steps = sorted(int(step) for step in gaps if step != "0")
        horizons = np.array([step - branch for step in steps])
        values = np.array([gaps[str(step)] for step in steps])
        ax.plot(horizons, values, marker="o", markersize=3.4, linewidth=2, color=color, label=label)

    # The registered duplicate continuation exists at the early fork. Its
    # absolute loss gap at the corresponding horizon is reused as the floor at
    # the later forks.
    floor_gaps = base["loss"]["dup"]["floor"]
    available = {int(step) - base["branch"]: value for step, value in floor_gaps.items() if step != "0"}
    common_horizons = sorted(
        set(available).intersection(
            int(step) - branch for step in record["loss"]["ka"]["gaps"] if step != "0"
        )
    )
    floor = np.array([available[h] for h in common_horizons])
    ax.fill_between(common_horizons, -floor, floor, color=FLOOR, alpha=0.55, linewidth=0, label="duplicate floor")

    ax.axhline(0, color="#7f8790", linewidth=1)
    ax.set_title(f"checkpoint {branch}", color=INK, fontsize=12)
    ax.set_xlabel("continuation steps", color=INK)
    ax.grid(axis="y", color=INK, alpha=0.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aeb7c2")
    ax.tick_params(colors=INK)

axes[0].set_ylabel("validation loss minus deployed bi-Maxwell", color=INK)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
fig.suptitle(
    "The one-step winner becomes worse than the deployed kernel",
    x=0.5,
    y=1.02,
    fontsize=16,
    color=INK,
    fontweight="semibold",
)
fig.text(
    0.5,
    0.945,
    "Positive values mean the matched kernel has higher validation loss",
    ha="center",
    fontsize=10.5,
    color="#5b6775",
)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig("s4_continuation_loss.png", dpi=210, bbox_inches="tight", facecolor="white")
plt.close(fig)
