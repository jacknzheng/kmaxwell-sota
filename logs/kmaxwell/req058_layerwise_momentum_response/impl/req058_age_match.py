"""REQ-058 exact-age control + stream-decay perturbation (the missing REQ-054 finite-history control).

Two pieces, both CPU-validatable:

1) Stream-decay memory intervention on the selected matrix: `beta_k(a) = beta_k ** (1/a)`, a in {0.5,1,2}.
   a=2 doubles each stream's exponential decay TIME (beta closer to 1, longer memory); a=0.5 halves it.
   The mixture weights w_k(t) are UNCHANGED (this is a memory intervention, not a weight reweighting).

2) Exact realized-age matched single-EMA control. The K-Maxwell mixture, with outer Nesterov blend nu, has
   a realized (finite-history) age; a single EMA whose stationary schedule is beta=A/(1+A) does NOT match it
   (REQ-054's schedule-only control). Here we track each stream's raw buffer mass M_k and unnormalized first
   age-moment P_k through the ACTUAL recurrence (including inherited clone history), form the mixture q_age,
   and solve a scalar beta(t) so the single EMA reproduces the mixture's realized q_age at EVERY step.

Recurrence per stream (raw buffer, feeding constant unit gradient for the age bookkeeping):
    M_new = beta*M + (1-beta)
    P_new = beta*(P + M)
Mixture over streams with normalized weights w_k: M_mix = sum_k w_k M_k ; P_mix = sum_k w_k P_k.
Outer blend nu (the Nesterov mix grad.lerp_(m_eff, nu) contributes a fresh-gradient component):
    q_mass = (1 - nu) + nu * M_mix
    q_age  = nu * P_mix / q_mass
Single-EMA control tracks scalar (m_mass, m_mom); target = mixture q_age; closed-form solve for beta in [0,1):
    beta = target_age / (nu*(m_mom + m_mass + target_age*(1 - m_mass)))   [design-review derivation]
then advance m_mom, m_mass with that beta; verify realized q_age matches target to ~1e-10 (no clipping).

This module is pure Python (no torch); tested by test_age_match.py against the design-review reference.
"""
from __future__ import annotations

DECAYS = [0.75, 0.822852439855, 0.877930338626, 0.917598547218,
          0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
START_W = [0.005093975, 0.010187949, 0.015281924, 0.020375898,
           0.025469873, 0.030563847, 0.035657822, 0.857368713]
END_W = [0.032261839, 0.064523678, 0.096785516, 0.129047355,
         0.161309194, 0.193571033, 0.225832871, 0.096668514]
NU = 0.95
SWITCH, ANNEAL_END = 2000, 2750  # REQ-054 window (perturbation applies to the mixed-update regime)


def perturb_decays(decays, a):
    """beta_k(a) = beta_k ** (1/a). a=1 identity; a=2 longer memory; a=0.5 shorter memory."""
    return [b ** (1.0 / a) for b in decays]


def normalized_weights(t):
    """REQ-054 mixture weights w_k(t), linearly annealed start->end over [SWITCH, ANNEAL_END], then
    normalized to sum 1 (the published normalized schedule)."""
    alpha = min(max((t - SWITCH) / (ANNEAL_END - SWITCH), 0.0), 1.0)
    w = [s + alpha * (e - s) for s, e in zip(START_W, END_W)]
    z = sum(w)
    return [x / z for x in w]


def clone_init(betas, n_warm=2001):
    """Production cloned-buffer switch: a baseline single-EMA update at the switch, then the buffer is cloned
    into all streams. Returns per-stream (M_k, P_k) and the shared scalar (mass, mom) at the clone point,
    replaying the design-review protocol (baseline update at switch then clone)."""
    mass = mom = 0.0
    for _ in range(n_warm):  # baseline update at switch, then clone
        mom, mass = NU * (mom + mass), NU * mass + (1 - NU)
    return [mass] * len(betas), [mom] * len(betas), mass, mom


def mixture_age(M_streams, P_streams, weights):
    M_mix = sum(w * m for w, m in zip(weights, M_streams))
    P_mix = sum(w * p for w, p in zip(weights, P_streams))
    q_mass = (1 - NU) + NU * M_mix
    q_age = NU * P_mix / q_mass if q_mass > 0 else 0.0
    return q_mass, q_age


def solve_matched_ema(betas, weights_fn, steps, warm=2001):
    """Track the mixture's realized q_age over `steps` and solve a single EMA beta(t) matching it each step.
    Returns dict with max age error, max mass diff, beta range, and per-step traces. All arms share the
    inherited clone history (`clone_init`). Verifies beta in [0,1) without clipping."""
    M_s, P_s, mass, mom = clone_init(betas, warm)
    max_age_err = max_mass_diff = 0.0
    betas_solved = []
    trace = []
    for t in range(SWITCH + 1, SWITCH + 1 + steps):
        w = weights_fn(t)
        for k, b in enumerate(betas):
            P_s[k], M_s[k] = b * (P_s[k] + M_s[k]), b * M_s[k] + (1 - b)
        target_mass, target_age = mixture_age(M_s, P_s, w)
        # closed-form single-EMA beta matching target_age given inherited (mom, mass)
        denom = NU * (mom + mass + target_age * (1 - mass))
        beta = target_age / denom if denom != 0 else 0.0
        assert 0.0 <= beta < 1.0, f"infeasible beta {beta} at t={t}"
        betas_solved.append(beta)
        mom, mass = beta * (mom + mass), beta * mass + (1 - beta)
        actual_mass = (1 - NU) + NU * mass
        actual_age = NU * mom / actual_mass if actual_mass > 0 else 0.0
        max_age_err = max(max_age_err, abs(actual_age - target_age))
        max_mass_diff = max(max_mass_diff, abs(actual_mass - target_mass))
        trace.append((t, target_age, actual_age, target_mass, actual_mass, beta))
    return dict(max_age_error=max_age_err, max_mass_difference=max_mass_diff,
                beta_range=[min(betas_solved), max(betas_solved)], n_steps=len(betas_solved), trace=trace)


if __name__ == "__main__":
    print("age-match solver; run test_age_match.py to validate.")
