# REQ-073 — Q/K/MLP spectra through Muon, momentum, and effective batch size

**Status: RUNNING (planning/build) 2026-10-03.** Harness `365c392d`. Builds on REQ-068 (whole-model δ was
99.994% embedding-dominated → must look at hidden matrices) and REQ-072 (momentum low-passes the period-two
gradient). REQ-073 adds the **batch-size (noise) axis** and the **full Muon pipeline decomposition** on
specific hidden matrices.

## Selected matrices (frozen, block 5 = middle; separates matrix TYPE from depth)
| matrix | name | shape |
|:--|:--|:--|
| Q | `blocks.5.attn.q.weight` | 768×768 |
| K | `blocks.5.attn.k.weight` | 768×768 |
| MLP out | `blocks.5.mlp.proj.weight` | 768×3072 |
Total 3,538,944 params (verified). (Distinct from REQ-072's three-depth attn.proj selection.)

## Four arms (3250 updates, identical init + ordered token stream, one seed)
| arm | effective batch | hidden-matrix Muon momentum |
|:--|:--|:--|
| B-mom | B=524,288 tok | mu=0.95 Nesterov |
| B-nomom | B | mu=0 |
| 4B-mom | 4B=2,097,152 tok (4× accumulation) | mu=0.95 |
| 4B-nomom | 4B | mu=0 |
Aux AdamW FIXED all arms (β1=0.9, β2=0.95, embed/head/scalar LRs, decay, schedule). Only hidden-matrix
first-moment momentum switches. 4B = 4× grad accumulation before one update; **divide the 4B summed grad by 4
before clipping/optimizer** (preserve the B token-sum convention). Primary contrast at equal optimizer-update
count; secondary at matched cumulative tokens.

## Four pipeline stages, every step, per selected matrix (lossless native)
1. raw g_t (after accum/sync/unscale, before clip/momentum/mutation); record mean-per-token normalization.
2. **Muon input after momentum/Nesterov** (the `grad.lerp_(momentum,mu)` result, BEFORE orthogonalization) —
   NEW intermediate; store the transform input, not the raw momentum buffer.
3. post-polar direction (after `zeropower_via_newtonschulz5` + shape scaling, before lr/decay) = Muon u_t.
4. actual displacement δ_t (lr/decay decomposition documented).

## KEY build challenge — per-rank sharded capture
Muon computes stages 2 & 3 only on each matrix's OWNING rank (size-descending ownership: rank r owns indices
r, r+8, …). So a master-only stash misses matrices owned by other ranks (the REQ-072 Muon gap). Stages 2/3
are **intermediates** — the wd=0 displacement trick (REQ-072 workaround) cannot recover them. **Fix: capture
stages 2/3 on ALL ranks; each rank writes its owned selected matrices to a rank-tagged stream; the reader maps
each matrix to its owner rank.** Stages 1 (grad) & 4 (disp) are replicated → captured on master as before.

## Forecast (REQUIRED)
3,538,944 params × 4 bytes fp32 = 14.2 MB/stage/step; ×3250 = 46 GB/stage/arm; **4 stages × 4 arms ≈ 736 GB
fp32** (368 GB bf16) + checkpoints @500/1500/2500. Fits `csi-storage-sfs` (54 TB free). Census dtypes at build.

## Analyses
- STFT spectra (reuse `req072_spectrogram`) per stage/arm/matrix: where in the pipeline does period-two→low-f
  happen, and does 4B (less noise) reduce period-two independently of momentum?
- Noise variance at checkpoints: 8 disjoint groups of 4 B-batches at fixed weights; B vs 4B sampling variance
  (4B ≈ ¼ B variance under independence); pilot-verify averaged-B == accumulated-4B at identical weights.
- Shadow-pipeline replay: replay the recorded raw hidden-matrix history through Muon mom on/off (fresh/warmed).

## Build order
1. (this) selection + plan + forecast + RUNNING.
2. per-rank 4-stage capture (stage-2 instrumentation + owner-rank writers) — CPU-validated.
3. 4B accumulation + divide-by-4 hook; 4 configs.
4. 4 runs + stage spectra + noise + shadow analyses.
