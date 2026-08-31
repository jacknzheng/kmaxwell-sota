# REQ-028 — the 8× batch point (completes the benefit-vs-batch curve)

**SHA `365c392d`, nodes `wdkeomq` (8× muon μ0) + `q4zpy53` (8× bimaxwell record) (8×H100 each, released), torch 2.10.0+cu128.** One more batch size turns REQ-026/027's 1×/4× pair into a three-point curve: does the bimaxwell momentum benefit keep falling toward zero or plateau? Two arms, seed 0, 8× batch, 750 steps from the shared step-2000 state; checkpoints +750 only. Both `ARM_EXIT=0`.

## Gates (PASS)

- **Data-budget (data-exhaustion check, per the request):** an 8× 750-step run from the fork needs `(skip 250 + 750) × 4.194M = 4.19B` tokens. Bootstrapped **48 fineweb shards = 4.80B tokens** ⇒ 0.61B margin, asserted `ok=True` on-node *before* launching each arm. (REQ-027's 35 shards would have been ~0.7B short — this is why 48 were fetched.)
- **Finite-loss smoke:** each config to step 2125 finite (b8x_mu0 = 3.32173); no NaN, no `StopIteration`.
- Tests green at the pinned SHA (from the shared bootstrap).

## Result — the benefit-vs-batch curve (`summary.tsv`)

final_val@2750: 8× μ0 = **3.20561**, 8× bimaxwell = **3.20328** ⇒ **8× benefit = −0.00233**.

| batch | batch_tokens | momentum benefit (bimax − μ0) @2750 | source |
|:------|-------------:|------------------------------------:|:-------|
| **1×** | 524288 | **−0.01063** | REQ-026/027 (seeds 0,1 mean) |
| **4×** | 2097152 | **−0.00438** | REQ-026/027 (seeds 0,1,2 mean) |
| **8×** | 4194304 | **−0.00233** | REQ-028 (seed 0) |

## Shape — keeps halving, no plateau

The benefit **continues to fall toward zero, with no sign of a plateau**. Per batch-doubling it roughly halves: 1×→4× is ÷2.4 over 4× batch (≈÷1.55 per doubling), and 4×→8× is **÷1.88 over a single doubling** (−0.00438 → −0.00233). A power-law-like decay in batch size — consistent with the denoising reading (the momentum benefit is largely absorbed by the larger batch's own gradient averaging) rather than a geometry floor. It remains **non-zero** at 8× (−0.0023, still ~10× the REQ-027 seed spread of ~2–7e-4), so momentum is not yet worthless at 8× — but the trend points at diminishing returns as batch grows.

*(Shape is the deliverable; no further interpretation per the request.)*

## Files

- `summary.tsv` — the 2 arms + the 3-point curve with per-doubling decay factors.
- `val_trajectories.txt` — raw per-step val_loss for both arms + the data-budget line + smoke.
- `manifest.tsv` / `make_req028_configs.py` — the frozen 2-arm design + generator (batch_tokens 4194304, skip_batches 250).
- `configs/` — the 2 configs. `logs/` — the 2 arm training logs.

Checkpoints (2750 per arm, offline curvature), `eos_shared_state`, FineWeb data, env dumps: on-box, **not** committed. Both nodes released.
