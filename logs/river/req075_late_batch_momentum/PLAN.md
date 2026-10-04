# REQ-075 — late-training large batch vs momentum: plan, forecast, provenance

**Status: RUNNING (2026-10-04).** Session `01YKPwGqPuEuP2wzzHuCEhok`. This doc is the pre-launch plan +
resource forecast the request requires before any training launch. Provisioning is staged; forecasts below
are published before Stage 2/3 training.

## Baseline recipe (from REQ-073 `make_req073_configs.py`, verified in-repo)
- B = **524,288 tokens/update**; microbatch_sequences = 64; world_size 8; train_steps 3250; seed 0.
- Muon (hidden `blocks.*.weight`): lr **0.025 fixed** (NOT batch-scaled), mu on=0.95 (Nesterov), off=0; wd=0.
- Aux AdamW (fixed, still momentum): embed lr 0.7, head `proj.weight` lr 0.004, catch-all 0.015; all wd=0.
- Fork dumps via `dump_training_state_at_steps` @ [500,1500,2500] → `{dur}/{arm}/forks` on durable FS.
- 16B target = **8,388,608 tokens** = 16×B ⇒ keep microbatch (64 seqs) fixed, accumulation 16×, divide
  accumulated grad by 16 (as REQ-073's `scale_grad_for_4B` does for 4×). One optimizer step per 16B mean.

## OPEN provenance questions (must resolve on-box before using the checkpoint — gate the whole study)
1. **Does the B-mom @2500 fork contain COMPLETE state** (model + Muon/AdamW optimizer buffers + RNG +
   data cursor), or only weights? `dump_training_state_at_steps` lives in the kmaxwell-sota harness
   `harness/hooks.py` (cloned per-box, not in this repo) — read it on the box. If incomplete, reproduce
   @2500 from the pinned recipe or declare the nearest verified late checkpoint (record the difference).
2. **Is there a token-indexed LR schedule** (warmup/cooldown) in the baseline loop, or are the param-group
   LRs constant? REQ-075 Stage 3 must "preserve the original token-indexed LR schedule and endpoint
   mapping." Inspect the loop/config for a scheduler hook.
3. **Exact @2500 provenance**: hash, token count (2500×524288 = 1,310,720,000), schedule position, data
   cursor. "immediately before update 2500" = state at the start of update 2500.
4. **Data coverage** (see risk below): how many fineweb10B chunks are staged on the box / durable FS, and
   where is the update-2500 cursor.

## Resource forecast (per stage; 162M-param GPT, 1×8-H100 node; throughput assumed ~0.5M tok/s/node,
to be re-measured on first arm and this doc updated)

| Stage | Tokens (train) | GPU-hours (1 node) | Raw capture (durable, off-Git) | Committed derived |
|---|---|---|---|---|
| 1 frozen-noise (64→128 B-probes, no opt step) | 34–67M | <0.5 | ~0.9 GB (opt: per-probe grads) | <10 MB |
| 2 mechanism pilot (256 upd × 5 arms; 2×B + 3×16B) | **6.71B** | ~3.7 | ~36 GB (4 stage × 3 mat × 256 × 5) | tens of MB |
| 3 equal-token (4 cfg × 3 seed @386M + LR search ×8 @386M) | **7.72B** | ~4.3 | ~150–250 GB (736-upd B arms dominate) | hundreds of MB |
| **Total** | **~14.5B** | **~8.5 node-h** (≈4.3 wall-h on 2 nodes) | ~300–400 GB | <1 GB |

I/O: dominated by capture writes (chunked, native dtype, SHA-256) + fineweb reads; same pattern as REQ-073.

## ⚠ Data-coverage risk (must verify before Stage 2/3; request forbids silent cycling / REQ-042 scope creep)
- Stage-2 **16B arms alone want 3 × 2.147B = 6.44B tokens** over 256 updates — *more than the whole
  remaining baseline budget* and approaching the ~10B fineweb10B corpus. Stage 3 adds ~7.7B more.
- The continuations must draw **unused** corpus beyond the baseline's consumed 1.31B @2500, with **3
  distinct paired data-order seeds** for Stage 3. Total distinct-token demand (~14B) **exceeds a single
  10B corpus** if seeds must be fully disjoint.
- **Mitigations to decide on-box** (per the request's own escape hatch): (a) confirm actual staged corpus
  size; (b) if insufficient, **declare a shorter common pilot horizon** for Stage 2 (e.g. 64–128 updates)
  and a reduced Stage-3 token budget, documenting the spectral/statistical limitations, rather than cycling
  data; (c) paired seeds may share the token *window* with different *ordering* where the spec allows
  ("same ordered token stream" within a seed) — clarify disjointness need vs ordering.

## Staged execution (≤2 nodes, per constraint; stop nodes after each stage)
1. **Node 1 up** → resolve provenance Q1–Q4 + data coverage; build Stage-1 frozen-noise harness (reuse
   REQ-073 stage capture + REQ-074 readers), CPU-validate; run Stage 1; predeclare the low-noise criterion
   **before** labeling 16B. Pull derived arrays, stop node.
2. Publish updated forecast (measured throughput) → Stage 2 pilot (horizon confirmed against coverage).
3. Stage 3 equal-token (inherited-LR first, then bounded 0.5/1/2 LR search), 3 paired seeds, matched WD.
4. Useful-descent probes (REQ-074 time-resolved arrays; independent-loss direction probes) + final report.

Raw tensors/checkpoints stay off-Git on durable FS with manifests/checksums; only derived arrays, configs,
readers, figures committed. Measured findings kept separate from interpretation.
