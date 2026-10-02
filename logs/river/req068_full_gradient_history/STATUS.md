# REQ-068 — full gradient history capture — plan + pre-launch forecast

**Status: RUNNING (planning + build) 2026-10-02.** Foundational deliverable for the river-direction
study (REQ-068→071). Pinned baseline: harness `365c392d695f95dc9a4fb89095e85a6a7b5d551e`, the simplified
optimization trainer (`records/track_3_optimization/run.py`), K-Maxwell/Muon + auxiliary AdamW baseline,
GPT vocab 50304 / 12 layers / dim 768, fineweb10B, batch 524288 tokens, microbatch 64, 3250 updates,
original LR/momentum schedules. (Exact recipe pinned in `recipe.md` when the capture hook lands.)

## Pre-launch storage/compute forecast (REQUIRED before launch)

Real model instantiated at the pinned config: **162,354,816 trainable params** (embed 38.63M, output head
38.68M, 72 hidden 2-D matrices 84.93M, 98 vectors/scalars 0.10M; dtypes bf16 + fp32).

| capture | per step | full run ×3250 |
|:--|--:|--:|
| gradient, **bf16 native (chosen)** | 324.7 MB | **1.055 TB** |
| gradient, fp32 | 649.4 MB | 2.111 TB |
| + optimizer displacements (optional, same size) | +324.7 MB | +1.055 TB |
| 3 fork checkpoints (model+opt+rng) @ 500/1500/2500 | — | ~2 GB |

**Decision:** capture the gradient in its **native precision losslessly** (bf16 where the param is bf16,
fp32 where fp32) — ≈**1.06 TB** for the full run. fp32 up-conversion is NOT used (it would not recover
precision absent in the source). Displacements captured separately only if the durable budget allows the
extra ~1 TB; gradient capture is mandatory and takes priority. This exceeds any node-local disk, so it
**must** stream to a retained durable volume (training checkpoint/shared volume), not the disposable node.

**Gating blocker to resolve before the full run:** provision ~1.1–2.2 TB durable storage reachable from the
training node (bf16-only vs bf16+displacements). Path: Baseten training shared/checkpoint volume
(training-storage). Benchmark sustained write ≥ the per-step 325 MB / step-time before committing the run;
if unavailable, report the concrete blocker — do NOT downsample, quantize, drop matrices, or shorten.

## Capture design

- **Point:** a `pre_optimizer` hook (loop.py:51), which runs **after** microbatch accumulation + the
  `dist.all_reduce(p.grad, SUM)` and **before** `apply_updates()` (Muon's `grad.lerp_` mutates `.grad`
  in place). Clone every param's `.grad` there → matches "after accumulation/sync/unscale, before
  clip/momentum/Muon/decay; copy before the optimizer mutates `.grad`."
- **Normalization:** the harness gradient is **token-SUM** (reduction="sum"). Store the raw native value
  losslessly and record tokens-per-update so analysis can form mean-per-token gradients without changing
  training inputs. Record inactive/missing grads explicitly (None → flagged), never fabricated zeros.
- **One observation = one optimizer update** (not microbatch); record the global update index 0..3249.
- **Streaming:** bounded chunks (default 50 updates/chunk ≈ 16 GB bf16) to the durable volume; per-chunk
  SHA-256; never hold the full run in GPU/host memory. Sharded ranks: save each rank's owned slices +
  an exact reconstruction map (Muon ownership = size-descending sort, rank r owns r,r+W,...); no duplicate
  replicas of the same shard.
- **Manifest (committed to Git):** param schema (names, shapes, dtypes, flatten order/offsets, tying,
  sharding), per-chunk hashes + step coverage, tokens/update, artifact sizes, code/config/seed provenance,
  retrieval instructions (no credentials).
- **Reader (committed):** reconstruct every parameter gradient for any requested step; stream per-matrix
  time series across steps.
- **Fork states:** retain checkpoint/optimizer/RNG/data-cursor off Git at steps **500, 1500, 2500** (for
  REQ-069/070), with hashes + retrieval paths.
- **Displacements (separate, optional):** snapshot θ before `apply_updates` and diff after → actual
  per-step parameter update (optimizer + weight decay effects), labeled updates (not gradients).

## Smoke tests (CPU/1-node, before the full run)
round-trip tensor equality; no `.grad` aliasing (captured copy ≠ mutated post-step grad); accumulation/
reduction timing (capture is post-all_reduce); uninterrupted-vs-resumed indexing continuity; logging-on
vs logging-off training parity (loss trajectory identical to tolerance). Post-run: verify all 3250 update
indices + all params/shards present — no gaps/dupes/partial chunks; else report incomplete capture.

## Build order
1. (this chunk) forecast + design + mark RUNNING.
2. capture hook + manifest writer + reader + chunked streamer — CPU-validated smoke tests (free).
3. storage provisioning + sustained-write benchmark (gating).
4. full 3250-step run with capture + fork dumps; post-run verification + descriptive analysis.

## Update 2026-10-02: config generator done + REQ-072 absorbed
- `impl/make_req068_config.py` emits the baseline capture config: standard **Muon mu=0.95** on hidden
  matrices + canonical aux AdamW (embed lr0.7 / head lr0.004 / scalars lr0.015, all wd0.001; blocks lr0.025
  wd0.05). Validated against the harness parser: all 15 hooks resolve, 4 optimizer groups parse,
  pre_optimizer order capture->dump[500,1500,2500]->cooldown, teardown finalize->mark_finished.
- Pipeline CPU-complete: capture core (6/6) + hook patch (registers) + config (parses). Remaining before
  launch: durable storage provisioning (gating) + GPU in-harness smoke (logging parity, round-trip, all
  3250 indices) + the full run.
- **REQ-072** (six optimizer ablations: AdamW/SGD/Muon ×±momentum, full histories + spectrograms) reuses
  this exact capture infra (the writer is optimizer-agnostic; muon-mom arm == this baseline). Program-wide
  storage now dominant: REQ-068 ~1.06 TB + REQ-072 ~6×1.06=6.3 TB + REQ-069 branches. Consolidated storage
  forecast + provisioning plan is the next gating step before any launch.
