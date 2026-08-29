import cmath
import math

import matplotlib.pyplot as plt


PERIODS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256]


def ema_response(beta: float, mu: float = 0.95) -> list[float]:
    values = []
    for period in PERIODS:
        z = cmath.exp(2j * math.pi / period)
        response = (1 - mu) + mu * (1 - beta) / (1 - beta / z)
        values.append(abs(response))
    return values


# Exact end-phase values from Fable's J1 descriptor output. The EMA series is
# recomputed for the matched-H100 winner (.982 -> .944), correcting the older
# .952 comparator used in the first evidence-pack draft.
SERIES = {
    "bi-Maxwell": [0.0892, 0.0924, 0.0986, 0.1149, 0.1340, 0.1754,
                   0.2164, 0.2892, 0.3470, 0.4770, 0.5931, 0.7467],
    "scheduled K8": [0.0826, 0.0850, 0.0897, 0.1022, 0.1172, 0.1500,
                     0.1831, 0.2445, 0.2982, 0.4561, 0.6380, 0.8091],
    "scheduled K6": [0.0877, 0.0908, 0.0966, 0.1118, 0.1297, 0.1679,
                     0.2055, 0.2735, 0.3316, 0.4979, 0.6820, 0.8441],
    "scheduled EMA (.944 end)": ema_response(0.944),
    "matched-age power law": [0.3998, 0.4236, 0.4576, 0.5168, 0.5631, 0.6287,
                              0.6729, 0.7300, 0.7665, 0.8417, 0.8947, 0.9271],
}

colors = plt.cm.turbo([0.05, 0.25, 0.45, 0.67, 0.90])
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.4, 8.0), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})

for (label, values), color in zip(SERIES.items(), colors):
    ax.plot(PERIODS, values, linewidth=2.5, color=color, label=label)

ax.set_xscale("log", base=2)
ax.set_ylabel("transfer magnitude  |W|")
ax.set_title("End-phase momentum transfer response")
ax.grid(alpha=0.18)
ax.legend(frameon=False, ncol=2, fontsize=9)

ema = SERIES["scheduled EMA (.944 end)"]
for label in ("scheduled K8", "scheduled K6"):
    ratio = [a / b for a, b in zip(SERIES[label], ema)]
    color = colors[1] if label == "scheduled K8" else colors[2]
    ax2.plot(PERIODS, ratio, linewidth=2.5, color=color, label=f"{label} / EMA")

ax2.axhline(1, color="#777", linewidth=1)
ax2.axvspan(64, 256, color="#888", alpha=0.10, label="corrected weighted-difference band")
ax2.set_xscale("log", base=2)
ax2.set_xticks(PERIODS)
ax2.set_xticklabels([str(p) for p in PERIODS])
ax2.set_xlabel("oscillation period (training steps)")
ax2.set_ylabel("gain ratio")
ax2.grid(alpha=0.18)
ax2.legend(frameon=False, fontsize=9)

fig.text(0.5, 0.005,
         "Quasi-static kernel descriptors; they describe the filters but do not establish causality.",
         ha="center", fontsize=9, color="#555")
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig("frequency_response.png", dpi=180)
