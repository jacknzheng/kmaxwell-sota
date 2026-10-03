"""REQ-069 ensemble analysis core (CPU-testable). Operates on whole-model vectors from 10 data-seed branches
(each a sequence of 64 per-step dicts {name: 1-D array}), separately for the gradient and displacement streams.

Question: do independent minibatch sequences reproduce a candidate slow whole-model component? If branches
share a slow "river" but have independent fast/period-two bounce, the ensemble MEAN should (a) keep the slow
component, (b) reduce the alternation, (c) agree across two independent groups of five.

Caveats honored: branches inherit a shared initial bounce phase, so averaging may RETAIN deterministic
alternation (reported, not assumed gone); mean gradient at different weights != gradient at mean weights;
few independent branches -> report dispersion, not p-values.

Reuses req068_gram for the temporal Gram on the ensemble-mean sequence.
"""
from __future__ import annotations
import numpy as np


def _flat(vec_dict, names):
    return np.concatenate([vec_dict[n].astype(np.float64).ravel() for n in names])


def ensemble_mean_step(branch_dicts, names):
    """Mean whole-model vector across branches at one step. branch_dicts: list (len B) of {name: array}."""
    acc = {n: np.zeros_like(branch_dicts[0][n], dtype=np.float64) for n in names}
    for d in branch_dicts:
        for n in names:
            acc[n] += d[n].astype(np.float64)
    B = len(branch_dicts)
    return {n: acc[n] / B for n in names}


def group_agreement_step(branch_dicts, names, groupA, groupB):
    """Cosine between the two group-mean whole-model vectors at one step (independent-five-group check)."""
    mA = _flat(ensemble_mean_step([branch_dicts[b] for b in groupA], names), names)
    mB = _flat(ensemble_mean_step([branch_dicts[b] for b in groupB], names), names)
    na, nb = np.linalg.norm(mA), np.linalg.norm(mB)
    return float(mA @ mB / (na * nb)) if na > 0 and nb > 0 else float("nan")


def convergence_step(branch_dicts, names, subset_sizes=(1, 2, 5, 10)):
    """For each n, the norm of the mean-of-first-n branches and the across-branch dispersion
    (mean pairwise distance / mean norm) at one step. Shows how the ensemble estimate stabilizes."""
    flats = [_flat(d, names) for d in branch_dicts]
    out = {}
    for n in subset_sizes:
        n = min(n, len(flats))
        m = np.mean(flats[:n], axis=0)
        out[n] = {"mean_norm": float(np.linalg.norm(m))}
    # dispersion across all branches: mean ||v_b - mean|| / ||mean||
    allm = np.mean(flats, axis=0); dn = np.linalg.norm(allm)
    disp = float(np.mean([np.linalg.norm(f - allm) for f in flats]) / dn) if dn > 0 else float("nan")
    out["dispersion"] = disp
    # SNR: ||mean|| / rms(individual) -> if >~1/sqrt(B) the mean has coherent structure beyond noise
    rms = float(np.sqrt(np.mean([np.dot(f, f) for f in flats])))
    out["mean_to_rms"] = float(dn / rms) if rms > 0 else float("nan")
    return out


def lag_cosine_sequence(seq_flats, lags=(1, 2, 4, 8, 16, 32)):
    """Mean signed cosine at each lag for a sequence of flat vectors (list of 1-D arrays)."""
    n = len(seq_flats)
    norms = [np.linalg.norm(v) for v in seq_flats]
    out = {}
    for L in lags:
        if L >= n: continue
        cs = []
        for i in range(n - L):
            a, b = norms[i], norms[i + L]
            if a > 0 and b > 0:
                cs.append(float(seq_flats[i] @ seq_flats[i + L] / (a * b)))
        out[int(L)] = float(np.mean(cs)) if cs else None
    return out


def ensemble_summary(branch_seqs, names, groupA=range(5), groupB=range(5, 10),
                     subset_sizes=(1, 2, 5, 10)):
    """branch_seqs: list (len B) of sequences; each sequence is a list (len T) of {name: array}.
    Returns per-step group agreement, ensemble-mean lag cosine vs mean individual-branch lag cosine
    (alternation reduction), convergence at a representative step, and the ensemble-mean lag profile."""
    B = len(branch_seqs); T = len(branch_seqs[0])
    groupA, groupB = list(groupA), list(groupB)
    # ensemble mean sequence (flattened)
    mean_seq = [_flat(ensemble_mean_step([branch_seqs[b][t] for b in range(B)], names), names) for t in range(T)]
    # individual branch flattened sequences
    branch_flats = [[_flat(branch_seqs[b][t], names) for t in range(T)] for b in range(B)]
    mean_lag = lag_cosine_sequence(mean_seq)
    indiv_lag = {L: float(np.mean([lag_cosine_sequence(branch_flats[b], (L,)).get(L, np.nan)
                                   for b in range(B)])) for L in (1, 2, 4, 8)}
    group_agree = [group_agreement_step([branch_seqs[b][t] for b in range(B)], names, groupA, groupB)
                   for t in range(T)]
    conv_mid = convergence_step([branch_seqs[b][T // 2] for b in range(B)], names, subset_sizes)
    return {
        "n_branches": B, "n_steps": T, "groupA": groupA, "groupB": groupB,
        "ensemble_mean_lag_cosine": mean_lag,
        "mean_individual_branch_lag_cosine": indiv_lag,
        "alternation_reduction_lag1": {"individual": indiv_lag.get(1), "ensemble": mean_lag.get(1)},
        "group_agreement_per_step_mean": float(np.nanmean(group_agree)),
        "group_agreement_first_last": [group_agree[0], group_agree[-1]],
        "convergence_mid_step": conv_mid,
    }
