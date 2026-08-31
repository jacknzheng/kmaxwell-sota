# REQ-026 — batch-size × momentum-kernel grid (does momentum survive large batch?)

**SHA `365c392d` (codex/req025-newton-muon; none of these arms use the newton optimizers), node `wp9pj83` (8×H100, released), torch 2.10.0+cu128.** Six 750-step continuations from one shared serialized step-2000 state, grid = **batch {1×, 4×} × kernel {muon μ=0.0, muon μ=0.95, bimaxwell_muon record}**. Tests the standing question: does the momentum benefit persist at large batch (geometry story) or shrink toward zero (denoising story)?

## Design & gates (all PASS)

- **Shared fork state:** one `eos_shared_base` run dumped the step-2000 training state (base val@2000 = 3.44367); all 6 arms resume it. Same machinery as REQ-019/025.
- **Smoke gate (6/6 PASS):** each config ran to the first post-fork validation (step 2125) with finite loss before any full arm — no NaN (plain muon/bimaxwell is stable here; the REQ-025 alpha=0 NaN was specific to the newton hook's data-order shift, absent here). Smoke values in `summary.tsv`.
- **Tests green** at the pinned SHA (targeted muon/registry/config suites).
- **Config diff** vs the REQ-019 fork-continuation template (`make_eos_state_dependence_configs.py`), generator committed as `make_req026_configs.py`:
  - `batch_tokens`: **524288** (1×) / **2097152** (4×); `microbatch_sequences`=64 fixed → 4× is 8→32 accumulation steps, **identical per-forward memory** (no OOM).
  - blocks-group optimizer: `muon{mu:0.0}` / `muon{mu:0.95}` / `bimaxwell_muon{mu:0.95, fast_decay:0.85, slow_decay:0.98, fast_weight:0.4385, switch_step:1000}`; **`lr:0.025, weight_decay:0.05` fixed across all three** so the only varied axis is the momentum kernel.
  - `start_step:2000, stop_after_step:2750` (750-step continuation); checkpoints at 2250/2500/2750 (=+250/+500/+750) via `checkpoint_model_at_cadence every:250`.
  - `load_training_state.skip_batches` **token-aligned**: 2000 (1×) / **500** (4×) — both resume at the same ~1.05B-token data position. (A flat `skip_batches=2000` skipped 4× the tokens at 4×; the smoke caught it as data exhaustion / `StopIteration`. Fixed + downloaded 35 fineweb chunks for the 4× runs.)
  - `cool_down_learning_rate cooldown_frac:0.7`, **no** `fixed_eta_after` → **LR schedule identical across all 6 arms** (documented confound below).

## Result — final val_loss @ step 2750 (full table + per-checkpoint in `summary.tsv`)

| batch | μ=0 (no mom.) | μ=0.95 (single EMA) | bimaxwell (record) |
|:------|-------------:|--------------------:|-------------------:|
| **1×** | 3.34586 | 3.34588 | **3.33557** |
| **4×** | 3.24333 | 3.24326 | **3.23892** |

(4× reaches lower loss at matched step count because it sees 4× tokens/step — a batch confound; hence the readout below is **within-batch-size**.)

## Readout — momentum benefit (kernel − μ=0, same batch); `readout.tsv`

| batch | bimaxwell − μ0 @2250 | @2500 | @2750 | single-EMA(μ95) − μ0 @2750 |
|:------|--------------------:|------:|------:|---------------------------:|
| **1×** | −0.01787 | −0.01468 | −0.01029 | +0.00002 (noise) |
| **4×** | −0.00987 | −0.00680 | −0.00441 | −0.00007 (noise) |

**bimaxwell benefit 1×/4× ratio: 1.81 (@2250) → 2.16 (@2500) → 2.33 (@2750).**

## Verdict — DENOISING-leaning, but momentum is not free at large batch

- The **bimaxwell (two-rate record) kernel gives a clear, monotone benefit over no-momentum at both batch sizes**, but that benefit is **~1.8–2.3× smaller at 4× than 1×**, and the ratio widens through training. This favors the **denoising interpretation** (the momentum benefit shrinks as the batch grows) over the geometry expectation that it would persist undiminished.
- **It does not vanish**, though: at 4× the bimaxwell kernel still helps by ~0.0044 @2750 — momentum retains partial value at 4× batch, so the design should stay minibatch-aware rather than dropping momentum at scale.
- **Single-EMA momentum (μ=0.95 plain muon) shows no durable benefit over μ=0 at either batch** (all |Δ| ≤ ~0.0016, decaying to the ~1e-4 nondeterminism floor by 2750). The benefit here is **specific to the two-rate bimaxwell structure**, not momentum magnitude per se.

**Scope:** discovery experiment, **n=1 per cell** (no seed replication) — read the ~0.0002 single-EMA deltas as noise and the ~0.004–0.018 bimaxwell deltas as signal, but do not attach significance without replication. The LR schedule is shared across arms (unchanged, per request); the primary within-batch-size kernel comparison shares that confound so it is controlled, but the absolute 1× vs 4× loss levels are not LR-matched.

## Files

- `summary.tsv` — per-arm val_loss at every boundary (smoke@2125, 2250, 2500, 2750) + kernel/batch/μ.
- `readout.tsv` — momentum benefit (kernel − μ0) at each checkpoint, both batches, + the 1×/4× ratio.
- `val_trajectories.txt` — raw per-step val_loss for all 6 arms + the base.
- `manifest.tsv` / `make_req026_configs.py` — the frozen design + its generator.
- `configs/` — the 6 committed configs. `logs/` — per-arm training logs + the arms driver log.

Checkpoints (2000/2250/2500/2750 per arm, for the **offline curvature** measured later per the request), optimizer shards, `eos_shared_state`, FineWeb data, and env dumps are retained on-box and **not** committed.
