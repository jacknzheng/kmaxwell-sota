import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# K-Maxwell mean buffer age vs training step (hippocampus/87):
# single EMA (beta=0.95, mean age 19) until step 1000, then the 8-buffer
# mixture anneals linearly from start weights (mean age 58) to end (26)
# over 2250 steps.
DECAYS = [0.75, 0.8228524398549413, 0.8779303386257296, 0.9175985472180883,
          0.9451809410725112, 0.9638939208460664, 0.976378696890324,
          0.9846153846153846]
W0 = np.array([0.005094, 0.010188, 0.015282, 0.020376, 0.025470, 0.030564, 0.035658, 0.857369])
W1 = np.array([0.032262, 0.064524, 0.096786, 0.129047, 0.161309, 0.193571, 0.225833, 0.096669])
ages = np.array([b / (1 - b) for b in DECAYS])
steps = np.arange(0, 3250)
mean_age = np.where(steps < 1000, 0.95 / 0.05 * np.ones_like(steps, dtype=float), 0.0)
frac = np.clip((steps - 1000) / 2250, 0, 1)
mix_age = np.array([( (1 - f) * W0 + f * W1) @ ages for f in frac])
mean_age = np.where(steps < 1000, 19.0, mix_age)

fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
ax.plot(steps, mean_age, lw=2.4, color="#1769aa", label="K-Maxwell mean buffer age (deployed schedule)")
marks = {1628: 49.1, 2528: 36.3, 3028: 29.2}
ax.scatter(list(marks), list(marks.values()), zorder=5, s=55, color="#9a2f0c",
           label="saved analysis states (Test A targets)")
for x, y in marks.items():
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(6, 8), fontsize=11)
ax.axvspan(0, 1000, color="#8a97a4", alpha=0.12)
ax.text(500, 55, "single-EMA phase", ha="center", fontsize=10, color="#4a5560")
ax.set_xlabel("training step")
ax.set_ylabel("mean buffer age (steps)")
ax.set_title("Test A target: the record kernel shortens its memory while noise rises")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_k8_age.png")
print("saved")
