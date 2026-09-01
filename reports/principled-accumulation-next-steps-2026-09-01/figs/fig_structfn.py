import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = torch.load("boxlogs/wstar_clocks_fork2400.pt", map_location="cpu", weights_only=False)
# median-across-matrices normalized structure function
lags = [1, 2, 4, 8, 16, 32, 64]
curves = []
for n, v in d["grad"].items():
    s2 = v["sigma2"]
    if not v.get("S_med"):
        continue
    curves.append([v["S_med"].get(D, np.nan) / (2 * s2) for D in lags])
med = np.nanmedian(np.array(curves), axis=0)
q25 = np.nanpercentile(np.array(curves), 25, axis=0)
q75 = np.nanpercentile(np.array(curves), 75, axis=0)

fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
ax.axhline(1.0, color="#8a97a4", lw=1.4, ls="--")
ax.text(1.05, 1.03, "pure-noise floor  $S(\\Delta)=2\\sigma^2$", fontsize=10, color="#4a5560")
ax.fill_between(lags, q25, q75, alpha=0.18, color="#1769aa")
ax.plot(lags, med, "o-", lw=2.2, color="#1769aa",
        label="measured $S(\\Delta)/2\\sigma^2$, median over 72 matrices (step-2528 state)")
ref = med[1] * (np.array(lags) / 2.0)
ax.plot(lags, 1 + (med[1] - 1) * (np.array(lags) / 2.0) ** 2, ":", lw=1.8, color="#9a2f0c",
        label="locally linear drift would grow $\\propto\\Delta^2$")
ax.plot(lags, 1 + (med[1] - 1) * (np.array(lags) / 2.0) ** 0.29, "-", lw=1.6, color="#13795b",
        label="measured law: exponent $p\\approx0.29$ (sub-diffusive)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(lags); ax.set_xticklabels(lags)
ax.set_xlabel("lag $\\Delta$ (steps)")
ax.set_ylabel("$S(\\Delta)\\,/\\,2\\sigma^2$")
ax.set_ylim(0.8, max(q75.max(), 1 + (med[1]-1)*32**2)*1.2)
ax.set_title("The gradient's non-noise change is sub-diffusive, not drifting")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("fig_structfn.png")
print("saved")
