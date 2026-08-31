# REQ-027 — seed replicates for the REQ-026 batch × kernel finding

**SHA `365c392d`, nodes `qrv5913` (A) + `qzy15ow` (B) (8×H100 each, released), torch 2.10.0+cu128.** Firms up REQ-026's n=1 headline (bimaxwell momentum benefit ≈ halves at 4× batch; single-EMA benefit ≈ 0) by replicating with seeds 1 and 2 (REQ-026 was seed 0). Same shared step-2000 state and machinery; only `seed` differs. Ran in parallel across both nodes; all 6 arms `ARM_EXIT=0`, checkpoints at +750 only.

## Arms (6) + REQ-026 seed-0 reference — `summary.tsv`

| batch | kernel | seed 0 (REQ-026) | seed 1 | seed 2 |
|:------|:-------|-----------------:|-------:|-------:|
| 1× | mu0 | 3.34586 | 3.34582 | — |
| 1× | bimax | 3.33557 | 3.33485 | — |
| 4× | mu0 | 3.24333 | 3.24354 | 3.24280 |
| 4× | bimax | 3.23892 | 3.23895 | 3.23866 |

## Readout — the benefit-shrinkage holds across seeds (`readout.tsv`)

momentum benefit = final_val(bimax) − final_val(mu0), same batch, @2750:

| batch | per-seed benefit | mean | spread (range) |
|:------|:-----------------|-----:|---------------:|
| **1×** | −0.01029 (s0), −0.01097 (s1) | **−0.0106** | 6.8e-4 |
| **4×** | −0.00441 (s0), −0.00459 (s1), −0.00414 (s2) | **−0.0044** | 4.5e-4 |

**The "momentum benefit halves at 4× batch" finding is robust.** Every individual seed shows |1× benefit| > |4× benefit| — the 1× benefits ([−0.0110, −0.0103]) and 4× benefits ([−0.0046, −0.0041]) **do not overlap** — and the 1×-vs-4× gap (~6.2e-3) is ~10–25× the per-seed spread (~2–7e-4). Mean 1× −0.0106 → 4× −0.0044 (ratio ~2.4×), consistent with REQ-026's ~1.8–2.3× at the intermediate checkpoints.

## Seed mechanism (as requested — the exact binding, measured)

`seed_then_initialize_parameters` does `torch.manual_seed(1337 + config["seed"])`, then the record's param init. **In a fork continuation this has essentially no effect on the trajectory:** the fork's `load_training_state` overwrites params + optimizer with the shared step-2000 state, the data loader (`distributed_data_generator`) reads the token stream **sequentially** (contiguous per-rank slices, no RNG shuffle), and there is no dropout — so nothing downstream consumes the seed-perturbed RNG. A direct isolation check (same `b4x_mu0` config at seed=0 vs seed=1 to step 2125) gives **3.35602 vs 3.35584, Δ = 1.8e-4** — i.e. the same order as accumulated run-to-run NCCL/cuBLAS nondeterminism, not a genuine data-order resample.

**Consequence for interpretation:** the cross-seed spread reported above measures **run-to-run (nondeterminism) robustness**, not **data-distribution robustness** — seed does not make each replicate see a different data order in this harness. The finding is robust to the former (tight, non-overlapping ranges); testing the latter would require a harness change (seed-dependent data sampling / a different data window per seed), which was out of scope for "change only the seed." Flagging so the spread isn't over-read as data-resampling variance.

## Files

- `summary.tsv` — all 6 arms + REQ-026 seed-0, final val@2750, with the seed-mechanism header.
- `readout.tsv` — per-seed benefit + per-batch mean/min/max/range/std.
- `val_trajectories.txt` — raw per-step val_loss (all 6 arms) + the seed-isolation check.
- `manifest.tsv` / `make_req027_configs.py` — the frozen design + generator.
- `configs/` — the 6 configs. `logs/` — per-arm training logs.

Checkpoints (2750 per arm, for offline curvature), `eos_shared_state`, FineWeb data, env dumps: retained on-box, **not** committed.
