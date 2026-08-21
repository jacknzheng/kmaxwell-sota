"""Deterministic K-Maxwell kernel: ages, EMA rates, mix weights.

No torch dependency so identity checks can run without a GPU env.
"""

from __future__ import annotations

import math

BIMAXWELL_BF = 0.85
BIMAXWELL_BS = 0.98
BIMAXWELL_W = 0.4385
BIMAXWELL_TAU_MIN = BIMAXWELL_BF / (1.0 - BIMAXWELL_BF)  # 5.666...
BIMAXWELL_TAU_MAX = BIMAXWELL_BS / (1.0 - BIMAXWELL_BS)  # 49.0
BIMAXWELL_START = 1000


def build_kmaxwell_kernel(k, tau_min, tau_max, sigma, bimaxwell_exact=False):
    """Return (tau, betas, weights, mean_age) as Python lists + float.

    Weights are a frozen scoring function, never sampled. With bimaxwell_exact,
    ignore the Gaussian recipe and emit the submission kernel.
    """
    if bimaxwell_exact:
        if k != 2:
            raise ValueError("--bimaxwell-exact requires --k 2")
        tau = [BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX]
        betas = [BIMAXWELL_BF, BIMAXWELL_BS]
        weights = [BIMAXWELL_W, 1.0 - BIMAXWELL_W]
        mean_age = sum(w * t for w, t in zip(weights, tau))
        return tau, betas, weights, mean_age

    if k < 2:
        raise ValueError(f"--k must be >= 2, got {k}")
    if not (tau_min > 0 and tau_max > 0 and tau_max >= tau_min):
        raise ValueError(f"need 0 < tau_min <= tau_max, got {tau_min}, {tau_max}")
    if sigma <= 0:
        raise ValueError(f"--sigma must be > 0, got {sigma}")

    log_min = math.log10(tau_min)
    log_max = math.log10(tau_max)
    tau = [10.0 ** (log_min + i * (log_max - log_min) / (k - 1)) for i in range(k)]

    tau_c = math.sqrt(tau_min * tau_max)
    log_c = math.log(tau_c)
    scores = [math.exp(- (math.log(t) - log_c) ** 2 / (2.0 * sigma * sigma)) for t in tau]
    z = sum(scores)
    weights = [s / z for s in scores]
    betas = [t / (t + 1.0) for t in tau]
    mean_age = sum(w * t for w, t in zip(weights, tau))
    return tau, betas, weights, mean_age


def format_kmaxwell_recipe(k, tau_min, tau_max, sigma, start, seed, bimaxwell_exact,
                           tau, betas, weights, mean_age):
    tau_list = ", ".join(f"{x:.6g}" for x in tau)
    beta_list = ", ".join(f"{x:.6g}" for x in betas)
    w_list = ", ".join(f"{x:.6g}" for x in weights)
    mode = "bimaxwell-exact" if bimaxwell_exact else "gaussian-log-tau"
    return (
        f"K-Maxwell recipe: mode={mode} k={k} tau_min={tau_min:.6g} tau_max={tau_max:.6g} "
        f"sigma={sigma:.6g} start={start} seed={seed} mean_age={mean_age:.6g}\n"
        f"  tau=[{tau_list}]\n"
        f"  beta=[{beta_list}]\n"
        f"  w=[{w_list}]"
    )
