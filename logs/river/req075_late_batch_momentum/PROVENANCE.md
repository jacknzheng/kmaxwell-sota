# REQ-075 — checkpoint & recipe provenance (resolved from harness source 365c392d)

All answers below are read directly from the pinned harness
`records/track_3_optimization/harness/hooks.py` @ `365c392d695f95dc9a4fb89095e85a6a7b5d551e`
(the same sha REQ-068→074 used) and the REQ-073 config generator. No reconstruction needed — the
retained fork is a complete state, not a fresh-state substitute.

## Common late checkpoint: REQ-073 **B-mom fork @ step 2500** (of 3250)
- Location (durable FS, verified present on wd8vemq):
  `/root/.cache/user_artifacts/req073/B-mom/forks/`
  - `train_state_model_step002500.pt` (572 MB) — full model state_dict (rank 0, CPU).
  - `train_state_step002500_rank{0..7}of8.pt` (507 MB ea) — **per-rank** optimizer state.
- `dump_training_state_at_steps` writes **pre-update at the pinned step** (params = x_2500, i.e.
  *immediately before* update 2500 executes) ⇒ exactly REQ-075's requested state.
- Each per-rank file carries: every optimizer group's `state_dict()` (**Muon momentum is owner-sharded,
  so all 8 ranks are required to resume**) **plus `muon_steps_seen`** (not in state_dict; drives the
  Bi-Maxwell switch). ⇒ complete optimizer history retained.
- Exact token count at x_2500 = 2500 × 524,288 = **1,310,720,000 tokens**. Schedule position: see LR below.

## Resume path: `load_training_state(state_dir, step=2500, skip_batches=N)`
- Setup hook (place after `broadcast_initial_parameters`): loads model + this rank's optimizer shard +
  `muon_steps_seen`, then fast-forwards the **training-token stream by `skip_batches`**. Loop must be
  launched with `start_step = 2500`.
- **The data cursor is NOT stored in the dump** — it is reconstructed by `skip_batches`. To continue the
  baseline's own stream set `skip_batches = 2500` (the B-batches consumed by step 2500). Branches diverge
  in data by choosing different `skip_batches` (REQ-069 used `1500 + b*64`). This is deterministic given
  the fixed fineweb shard ordering — not a fresh-state experiment.
- **RNG**: the speedrun GPT has no dropout and params are already loaded, so the only stochasticity is data
  order, fully controlled by `skip_batches`. (Confirm no dropout in `model_gpt.py` during Stage-1 build.)

## LR schedule: `cool_down_learning_rate(cooldown_frac=0.7, shape="linear")`
- eta from `progress = state["step"] / train_steps`: eta=1 while progress < 0.3, then linear decay to 0
  across the last 70%. Every group lr = base × eta. Baseline train_steps=3250.
- At step 2500: progress 0.769, u=(0.769−0.3)/0.7 = 0.670 ⇒ **eta ≈ 0.330** (stable→decay, ~67% into
  cooldown). Group base LRs: Muon 0.025, embed 0.7, head 0.004, catch-all 0.015 (all wd=0).
- **Cross-batch mapping (critical for 16B arms):** the hook is **step-indexed**, so running a 16B arm with
  one `state["step"]` increment per optimizer update would misread the schedule (46 steps ≠ 736 B-steps of
  token progress). REQ-075 requires the **token-indexed** schedule. Implementation: a REQ-075 cooldown
  variant that computes eta from **token position** `(2500*B + tokens_since_fork)/(3250*B)` with the same
  linear cool-down, so B and 16B arms get identical eta at equal tokens. Common token endpoint = 736B
  rounded to a multiple of 16B = **736 B-updates = 46 16B-updates**, token-step 3236 (truncated from 3250;
  record the truncated endpoint). Do NOT stretch the 16B arm to 736 optimizer updates.

## Data module (`data_fineweb.py`): **no silent cycling**
- `iterate_batches_single_process` (offline diagnostics: Stage-1 probes) and the training generator glob
  `.bin` shards **sorted, relative to cwd** (run from `/root/kmaxwell-sota` ⇒ data at
  `data/fineweb10B/*.bin`; the REQ-063/064 cwd gotcha). Both are bounded by `total_tokens` and **return at
  corpus exhaustion** — they do not wrap. ⇒ an over-long horizon ends short, never cycles silently.
- Shard header magic 20240520; tokens uint16; stride by rank so reduced partial sums == single-worker sums.

## Consequence for the data-coverage risk (see PLAN.md)
- Probe data (Stage 1) and the 3 continuation seeds (Stages 2/3) must occupy **disjoint** shard regions
  from each other and from the baseline's consumed 0..1.31B region, with no wrap. Total distinct-token
  demand (~14B) likely exceeds one 10B corpus ⇒ **declare a shorter common horizon** (the request's own
  escape hatch) once the on-box chunk census is known, rather than cycle. Chunk census pending venv/data
  staging on wd8vemq.
