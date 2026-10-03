"""REQ-070 candidate-direction extraction (CPU-testable). At a probe offset o (absolute step t=fork+o),
for each branch b assemble the whole-model movement candidates to test for useful descent on independent
loss, using only causally-available information (gradients through g_t, completed displacements through
delta_{t-1}):

  baseline_update  : delta_{t-1}  (the optimizer's actual last movement = direct movement units)
  neg_gradient     : -g_t          (raw downhill; gradient units -> passed through copied optimizer state
                                     at probe time by the GPU driver, NOT re-conditioned here)
  temporal_avg     : mean of delta_{t-k..t-1} (ordinary temporal averaging of completed displacements)
  ensemble_mean_LOO: leave-one-branch-out mean of the OTHER branches' delta_{t-1} (never includes branch b's
                     own sampling noise in its target), evaluated at branch b's exact weights by the driver

All are whole-model dicts {name: vector}. Signs/units labeled. The ensemble_mean_LOO and temporal_avg are
offline-ish references; neg_gradient/baseline_update are causal. The GPU driver evaluates fresh probe loss
at theta_b(offset) + alpha * d/||d|| for alpha in {0,0.25,0.5,1,2} * ||baseline_update||.
"""
from __future__ import annotations
import numpy as np


def _axpy_mean(dicts, names):
    out = {n: np.zeros_like(dicts[0][n], dtype=np.float64) for n in names}
    for d in dicts:
        for n in names:
            out[n] += d[n].astype(np.float64)
    return {n: out[n] / len(dicts) for n in names}


def unit_norm(d, names):
    """||d|| over the whole model (flattened Euclidean)."""
    return float(np.sqrt(sum(float((d[n].astype(np.float64) ** 2).sum()) for n in names)))


def extract_candidates(disp_by_branch, grad_by_branch, names, branch, offset, temporal_k=8):
    """disp_by_branch[b] / grad_by_branch[b]: callables step_index->{name:array} for branch b over the window
    (step_index is the LOCAL offset 0..63). offset o: probe at local step o (abs t=fork+o). Causal: uses
    delta at o-1 (completed) and g at o. Returns {candidate_name: {name: vector}} for this branch+offset."""
    assert offset >= 1, "probe offset must be >=1 so a completed displacement exists"
    cands = {}
    cands["baseline_update"] = {n: v.astype(np.float64) for n, v in disp_by_branch[branch](offset - 1).items()}
    g = grad_by_branch[branch](offset)
    cands["neg_gradient"] = {n: -g[n].astype(np.float64) for n in names}
    # temporal average of completed displacements delta_{o-k..o-1}
    k = min(temporal_k, offset)
    hist = [disp_by_branch[branch](j) for j in range(offset - k, offset)]
    cands["temporal_avg"] = _axpy_mean(hist, names)
    # leave-one-out ensemble mean of OTHER branches' last completed displacement
    others = [disp_by_branch[b](offset - 1) for b in range(len(disp_by_branch)) if b != branch]
    cands["ensemble_mean_LOO"] = _axpy_mean(others, names)
    return cands


def perturbation(theta_flat, d_flat, alpha, baseline_norm):
    """theta + alpha * (d/||d||) * baseline_norm ... but spec: equal whole-model displacement LENGTHS =
    {0,0.25,0.5,1,2} * baseline_update_norm. So step = (alpha*baseline_norm) * d/||d||."""
    dn = np.linalg.norm(d_flat)
    if dn == 0:
        return theta_flat.copy(), True  # zero-norm candidate flagged
    return theta_flat + (alpha * baseline_norm) * (d_flat / dn), False
