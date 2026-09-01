import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DECAYS = [0.75, 0.8228524398549413, 0.8779303386257296, 0.9175985472180883,
          0.9451809410725112, 0.9638939208460664, 0.976378696890324, 0.9846153846153846]
W0 = np.array([0.005094, 0.010188, 0.015282, 0.020376, 0.025470, 0.030564, 0.035658, 0.857369])
W1 = np.array([0.032262, 0.064524, 0.096786, 0.129047, 0.161309, 0.193571, 0.225833, 0.096669])
ages = np.array([b / (1 - b) for b in DECAYS])
steps = np.arange(0, 3250)
frac = np.clip((steps - 1000) / 2250, 0, 1)
mix_age = np.array([((1 - f) * W0 + f * W1) @ ages for f in frac])
sched = np.where(steps < 1000, 19.0, mix_age)

res = {}
for F, st in ((1500, 1628), (2400, 2528)):
    d = torch.load(f"boxlogs/wstar_postfit_fork{F}.pt", map_location="cpu", weights_only=False)
    emp = np.array([r[3] for r in d["rows"]])
    res[st] = (np.median(emp), np.percentile(emp, 25), np.percentile(emp, 75))

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=140)
ax.plot(steps, sched, lw=2.4, color="#1769aa", label="K-Maxwell deployed mean age (shortens)")
xs = sorted(res)
med = [res[x][0] for x in xs]
lo = [res[x][0] - res[x][1] for x in xs]
hi = [res[x][2] - res[x][0] for x in xs]
ax.errorbar(xs, med, yerr=[lo, hi], fmt="s", ms=9, lw=2, capsize=5, color="#9a2f0c",
            label="measured tracking optimum $A^*$ (lengthens) — model-free, frozen protocol")
ax.annotate("0.5", (1628, med[0]), textcoords="offset points", xytext=(8, -14), fontsize=11, color="#9a2f0c")
ax.annotate("15.1", (2528, med[1]), textcoords="offset points", xytext=(8, 8), fontsize=11, color="#9a2f0c")
ax.scatter([3028], [29.2], marker="o", s=45, color="#1769aa")
ax.annotate("third state:\nclocks pending", (3028, 29.2), textcoords="offset points",
            xytext=(-6, 12), fontsize=9, ha="right", color="#4a5560")
ax.set_xlabel("training step")
ax.set_ylabel("mean buffer age (steps)")
ax.set_ylim(-2, 62)
ax.set_title("Test A: the tracking optimum moves OPPOSITE to the deployed schedule — rejected")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_testA.png")
print("saved")
