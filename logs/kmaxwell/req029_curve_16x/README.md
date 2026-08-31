# REQ-029 — curve cleanup: 16× pair + single-EMA at 8× (completes the benefit-vs-batch curve)

**SHA `365c392d`, nodes `31e5g4w` + `3m7dp23` (8×H100 each, released), torch 2.10.0+cu128.** Extends the momentum-benefit-vs-batch curve one more octave (16×) and fills the single-EMA point at 8×. Three arms, seed 0, 750-step continuations from the shared step-2000 state, checkpoint +750 only.

## The full curve (`summary.tsv`) — benefit keeps falling to ~zero; no plateau

momentum benefit = final_val(bimaxwell) − final_val(muon μ0), same batch, @2750:

| batch | batch_tokens | benefit (bimax − μ0) | source |
|:------|-------------:|---------------------:|:-------|
| 1× | 524288 | −0.01063 | REQ-026/027 |
| 4× | 2097152 | −0.00438 | REQ-026/027 |
| 8× | 4194304 | −0.00233 | REQ-028 |
| **16×** | **8388608** | **≈ 0.00000** | **REQ-029** |

At 16× the two kernels converge to an **indistinguishable** final loss (both 3.17362; |benefit| < 1e-5, well below the ~2e-4 seed-noise floor from REQ-027). The benefit decays visibly through the 16× run itself (−0.00065 @2500 → −0.00022 @2625 → <1e-5 @2750). **Shape: the momentum benefit keeps falling and reaches ~zero by 16× — no plateau.** This is the denoising story in full: at large batch the two-rate kernel's advantage is entirely absorbed by the batch's own gradient averaging.

The two 16× arms are verified-distinct runs (configs: μ0 = `muon mu=0.0`, bimax = `bimaxwell_muon mu=0.95`; their 2500/2625 trajectories differ — μ0: 3.20669/3.18948, bimax: 3.20604/3.18926) that converge to the same 5-decimal loss by 2750.

## figB — single-EMA benefit ≈ 0 at every batch (now including 8×)

single-EMA benefit = final_val(muon μ0.95) − final_val(muon μ0), @2750:

| batch | single-EMA benefit (μ0.95 − μ0) |
|:------|--------------------------------:|
| 1× | +0.00002 |
| 4× | −0.00007 |
| **8×** | **−0.00010** (3.20551 vs REQ-028 8× μ0 3.20561) |

Single-EMA momentum gives **no benefit over μ0 at any batch size** (all |Δ| ≤ 1e-4, at the noise floor) — the benefit in this program is specific to the two-rate bimaxwell structure, not momentum per se.

## Gate note — the 16× data-budget bug (found + fixed)

The first 16× pass **exhausted fineweb 17 steps short of 2750** (StopIteration @ step 2733) despite a passing token-count budget check. Root cause: `distributed_data_generator` moves to the next shard when `< batch_size+1` tokens remain, **discarding each shard's sub-batch tail**. At 16×, a 100M-token shard yields only `floor((100M−1)/8388608) = 11` usable batches, so 78 shards = **858** usable batches < `125 skip + 750 = 875` needed. The raw-token budget (7.8B ≥ 7.34B) was misleading. **Fix:** top up to **86 shards = 946 usable batches ≥ 875**, and use **usable batches** (`Σ floor(shard_tokens/batch_tokens)`) as the budget metric. Both 16× arms then completed to 2750. (8× is unaffected: 23 usable batches/shard.) Per-config finite smoke passed on all arms.

## Files

- `summary.tsv` — the 3 arms + the full 5-point curve (1×/4×/8×/16×) + the single-EMA figB row + the data-budget note.
- `val_trajectories.txt` — raw per-step val_loss for all three arms.
- `manifest.tsv` / `make_req029_configs.py` — the 3-arm design + generator.
- `configs/` — the 3 configs. `logs/` — the 3 arm training logs (16× logs are the 86-shard reruns).

Checkpoints (2750 per arm), `eos_shared_state`, FineWeb data, env dumps: on-box, **not** committed. Both nodes released.
