# REQ-070 — do candidate directions give useful descent on independent loss? — **NEGATIVE (no useful river descent)**

**Harness `365c392d`, node 3yk4kew (1×8 H100), stopped after delivery.** Forward-pass loss probes (no
training) at the exact REQ-069 probe states θ_b(offset), offsets 32/48/63, all 10 branches. For each causal
candidate movement direction d, fresh held-out val loss measured at θ + (α·‖baseline_update‖)·d/‖d‖ for
α ∈ {0, 0.25, 0.5, 1, 2}, restoring exact weights after each test. Leave-one-branch-out ensemble (never uses
a branch's own sampling noise). `analysis/loss_probe.json`; `impl/req070_{directions,loss_probe}.py`.

## Result — mean loss Δ vs α=0 (10 branches; negative = descent). Zero-loss ≈ 3.54.

| candidate (offset 32/48/63 consistent) | α=0.25 | α=0.5 | α=1 | α=2 |
|:--|--:|--:|--:|--:|
| baseline_update (δ_{t−1}, optimizer units) | +0.008 | +0.019 | +0.050 | +0.147 |
| neg_gradient (−g_t, raw downhill) | **+17.7** | +19.4 | +21.3 | +22.2 |
| temporal_avg (mean of past δ) | +0.003 | +0.007 | +0.015 | +0.035 |
| ensemble_mean_LOO (leave-one-out river) | −0.002 | −0.002 | +0.001 | +0.024 |

**Best (most negative) Δ anywhere: −0.0021 nats** (ensemble_mean_LOO, offset 32, α=0.5) — negligible
(~0.06% of 3.54) and **not robust** (offset 48: −0.0005; offset 63: none). 

**No candidate direction provides useful descent on independent data.**
1. **baseline_update** monotonically *increases* loss with distance — the optimizer's own step is already
   near-optimal at this state; moving further in its last direction overshoots.
2. **neg_gradient** is catastrophic at every scale (+17…+23): a ‖baseline‖-length move along the raw,
   normalized −gradient massively increases loss. The optimizer's conditioning (Muon orthogonalization +
   per-group LR) is essential; the raw downhill direction is not usable at this scale.
3. **temporal_avg** (ordinary displacement averaging) gives no descent.
4. **ensemble_mean_LOO** — the candidate reproducible "river" direction — is at best **loss-neutral**
   (−0.002…+0.001 at small α), never a meaningful or robust descent. Consistent with REQ-069: it is a weak,
   incoherent, embedding-dominated shared-initial-condition artifact that does not point downhill.

## Conclusion — the river arc closes negative

- **REQ-068:** actual displacement persists step-to-step (candidate river) — but embedding-dominated and not
  low-rank-constant (bending river).
- **REQ-069:** independent data does **not** reproduce a common whole-model direction (two-group agreement
  decays 0.9→0 in 64 steps).
- **REQ-070:** no candidate direction (baseline update, raw gradient, temporal average, ensemble mean) gives
  **useful descent** on independent loss beyond the optimizer's own step (best −0.002, negligible/non-robust).

**Verdict: there is no useful, reproducible "river" descent direction here.** The locally-persistent
displacement REQ-068 found is real but non-actionable: it does not generalize across data (069) and does not
reduce independent loss (070). The optimizer's own update is already near-optimal, and smoothness/persistence
of the motion is **not** evidence of a useful descent direction — exactly the caution the request raised.

## Gating consequence
**REQ-071 (fit a causal gradient-history estimator and test it in training) is conditional on REQ-069/070
evidence of a useful slow component — which is absent.** Per its own gate, REQ-071 should **not** be launched
on this evidence. (REQ-072, the six-optimizer descriptive ablation, is independent and remains valid.)

## Caveats
- Euclidean whole-model geometry is embedding-dominated (δ); a layer-reweighted probe could be a follow-up,
  but the ensemble direction is loss-neutral regardless. neg_gradient was tested as a normalized direction;
  a variant conditioned through copied optimizer state would just approximate baseline_update (also no
  descent beyond its own step). 10 branches, 3 offsets; branch-paired, fresh held-out probe tokens.

## Files
`analysis/loss_probe.json` (full curves + per-branch). `impl/req070_directions.py` (+test 6/6),
`impl/req070_loss_probe.py`, `impl/run_req070.sh`. Probe states + branch tensors off-Git on durable FS.
Node 3yk4kew stopped. No secrets/weights/tensors committed.
