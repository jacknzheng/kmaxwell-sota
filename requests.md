# requests.md — active queue for Jerry's agent

This branch is watched by Jerry's autonomous agent every 10 minutes.

To ask for work:

1. Append one request block using the template below.
2. Commit and push it to the `jerry-agent` branch.
3. Jerry changes `OPEN` to `RUNNING`, then `DONE`, `FAILED`, or
   `NEEDS-INFO`, and writes results under the same block.

Keep this file as an active queue, not a permanent results archive. Delete
completed and superseded requests after their useful code, logs, and summaries
have landed in the appropriate repository paths.

Next request number: **REQ-034**.


## REQ-033: does the ANNEALED K-Maxwell kernel survive a change of batch size?

- status: OPEN
- agent status: **RUNNING (2026-09-02 ~21:35Z)** — accepted; runs on a 2nd node in parallel with REQ-032's scaling-sdpo box (within my 2-node ceiling: REQ-032=qrvdr53, this=1 more; REQ-032's arm is API-bound so the kmaxwell node doesn't contend). Provisioning an 8×H100 kmaxwell box, bootstrapping @ `365c392d` (venv019 torch2.10cu128 + **30 fineweb chunks** [2x binding case=29]). Plan: regenerate `eos_shared_base` with the dump at **step 1000** (= transition_step_to_kmaxwell), record base val@1000; generate 12 configs (batch{0.25x,0.5x,2x}×kernel{A muon μ0.95, B bimaxwell record, C annealed_weights_muon shipped decays, D annealed_weights_muon batch-rescaled decays}) from the exact keys you gave (batch_tokens/microbatch_sequences/token-aligned skip 4000/2000/500; start 1000→stop 3250; C/D switch_step 1000 anneal_steps 2250; checkpoint 3250 only; dense val [3000,3250]/10). Hard gates first: **usable-batch budget assert** (REQ-029 metric) + **20-step finite-loss smoke** per config before any full arm. 1x row omitted as specified (out-of-band n=8 context only, never differenced). Deliverable → `logs/kmaxwell/req033_annealed_batch/`.
- agent status update: **ROOT-CAUSE FOUND + WORKAROUND APPLIED, marathon re-running (2026-09-02 ~23:25Z).** Base ran clean: **val@1000 = 3.65169**, step-1000 state dumped (model + 8/8 ranks), and **all 4 gates GREEN** — tests pass at the pin, usable-batch budget OK (2x tightest at +100, = your 29-of-30 binding case), mbs-divisibility OK, and all 13 configs resolve at launch. **But the first 12-arm pass came back ALL-NaN**: every arm — including plain-muon control A — diverged to NaN within **10 steps** of the fork. I isolated it with 6 controls: it is **NOT** the base/fork (1x *and* 2x fork@1000 both descend cleanly to ~3.54–3.61), **NOT** the LR (0.25× LR still NaN), **NOT** the optimizer step or data (frozen LR≈0 *still* NaN → the **gradient** itself is NaN). **Root cause: a `torch.compile` shape bug at `microbatch_sequences < 64`.** mbs=64 (1x, 2x) is stable; mbs=16 and mbs=32 NaN *regardless of batch_tokens* (verified: 1x batch with mbs=16/32 also NaN). **Eager (`TORCHDYNAMO_DISABLE=1`) fixes it** — 0.25x runs finite (val ~3.82). Since 0.25x/0.5x *require* mbs ≤ 16/32 (the per-GPU-seq cap you flagged), I run those **8 small-batch arms eager** (identical math, ~25× slower) and the **4 2x arms compiled** (mbs=64, fast). Your readout is a **within-batch** contrast, so the eager/compiled split does not affect it. Re-running now on the one node: b0.25x_A eager full at step ~1660/3250; ETA ~6–7h for all 12. The README will document the compile-NaN bug and the eager/compiled split. **If you'd prefer I stop and instead patch the compile path, or descope to 2x-only, say the word — otherwise I proceed to the full curve.**
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: 365c392d695f95dc9a4fb89095e85a6a7b5d551e (same as REQ-026/027/028/029)
- priority: arms are 2250 steps each (3x the REQ-026/029 forks), 12 of them. ONE
  node works but will be slow; TWO nodes preferred if within your ceiling — split
  by batch size, never by arm within a batch size (arms in a row must share a node
  so the within-batch contrast is same-hardware).

**Question.** REQ-026→029 established the benefit-vs-batch curve for the **frozen
two-rate bi-Maxwell** kernel: `1x -0.01063, 4x -0.00438, 8x -0.00233, 16x ~0.00000`
— halving per doubling, no plateau, fully absorbed by large-batch averaging.

That curve was never measured for the **annealed K=8 K-Maxwell** kernel of
[PR #357](https://github.com/KellerJordan/modded-nanogpt/pull/357) (verified: no
mention of the annealed kernel anywhere in req026/028/029). That kernel is a
different object — its mean age *changes over training* (58 -> 26 steps), and that
extra degree of freedom is exactly what could have absorbed batch-specific noise
structure. It is also the kernel in the open PR, so its batch-robustness is
load-bearing for the claim.

**Every timescale in the recipe is in STEPS**, but momentum reduces gradient
variance and noise scales as 1/batch. The tokens each buffer averages over is
`age x batch_tokens`. So the tuned ages are only meaningful at the batch size they
were tuned at. Arm D tests the obvious repair: scale the ages with the batch ratio.

### Expected work — 12 arms, 2250-step continuations from a shared step-1000 state

Same `eos_shared_base` machinery as REQ-026/029, but **dump the shared state at
step 1000, not 2000** — 1000 is `transition_step_to_kmaxwell`, the step at which
PR #357's kernel first engages. Forking there means arms C/D run the **full,
uncompressed 58 -> 26 anneal over the PR's own 2250 steps, ending at the real step
3250**. Record the base val@1000 as REQ-026 recorded val@2000 = 3.44367.
Grid = batch {0.25x, 0.5x, 2x} x kernel {A, B, C, D} = 12. **The 1x row is
deliberately omitted**: 1x full-run baselines already exist at n=8
(`logs/kmaxwell/{bimaxwell339_n8,ablation_anneal_n8}`), and re-running 1x here as a
step-1000 fork would not be comparable to them anyway (fork-vs-full-run). The
consequence is stated plainly in the readout: **the curve has no cell at the batch
size the recipe was tuned at**, so it is read as a trend across 0.25x / 0.5x / 2x,
with the existing n=8 1x runs as out-of-band context only — never differenced
against these arms.

| kernel | optimizer | meaning |
|---|---|---|
| **A** | `muon{mu:0.95}` | single-EMA control |
| **B** | `bimaxwell_muon` record settings | the already-curved reference |
| **C** | `annealed_weights_muon`, ages AS SHIPPED | PR #357 kernel, unmodified |
| **D** | `annealed_weights_muon`, ages x (1x/batch) | ages rescaled to the batch |

**The 12 arms explicitly:**

| # | batch | kernel | # | batch | kernel |
|---|---|---|---|---|---|
| 1 | 0.25x | A single-EMA | 7 | 0.5x | C shipped |
| 2 | 0.25x | B bimaxwell | 8 | 0.5x | D rescaled |
| 3 | 0.25x | C shipped | 9 | 2x | A single-EMA |
| 4 | 0.25x | D rescaled | 10 | 2x | B bimaxwell |
| 5 | 0.5x | A single-EMA | 11 | 2x | C shipped |
| 6 | 0.5x | B bimaxwell | 12 | 2x | D rescaled |

Arm B is included **so the annealed curve can be laid directly against the
bi-Maxwell curve on the same axes, same fork, same node** — REQ-026/029 measured
bi-Maxwell against `mu:0.0`, not against single-EMA, so a fresh B arm is what makes
the two curves comparable.

### Exact config keys

Batch axis (`microbatch_sequences` must shrink at small batch — per-GPU sequences
IS the microbatch bound there; 64 is illegal at 0.25x):

| batch | `batch_tokens` | `microbatch_sequences` | `skip_batches` (token-aligned) | fineweb chunks |
|---|---|---|---|---|
| 0.25x | 131072 | 16 | 4000 | 9 |
| 0.5x | 262144 | 32 | 2000 | 12 |
| 2x | 1048576 | 64 | 500 | 29 |

All three skips are exact integers -> every arm resumes at the same ~0.524B-token
data position. Budget in **usable batches** (`sum floor(shard_tokens/batch_tokens)`),
the REQ-029 metric, not raw tokens; chunk counts above already use it. **Bootstrap 30
chunks** — 2x is the binding case at 29.

`lr: 0.025, weight_decay: 0.05, mu: 0.95` fixed across ALL arms so the kernel is the
only varied axis within a batch size. `start_step: 1000, stop_after_step: 3250`,
`cool_down_learning_rate cooldown_frac: 0.7`, no `fixed_eta_after` — identical
schedule across arms, same documented confound as REQ-026 (within-batch-size
comparison controls for it). Checkpoint at 3250 only (+2250). No Lanczos.

Dense validation over `[3000, 3250]` every 10 steps, as the PR's own records do, so
the tail is resolved; the standard cadence elsewhere.

**Arm B** (`bimaxwell_muon`): `fast_decay: 0.85, slow_decay: 0.98,
fast_weight: 0.4385, switch_step: 1000` — the record settings from REQ-026.

**Arms C/D** (`annealed_weights_muon`): `switch_step: 1000`,
`anneal_steps: 2250` — i.e. **exactly the PR's own schedule, uncompressed.** The
fork point IS the switch step, so the shared base is plain Muon with no K-buffers to
inherit and the buffers lazy-init from the existing momentum precisely as they do in
the PR at step 1000. Each arm is therefore a faithful continuation of PR #357's own
schedule from step 1000 onward at its batch size, not an approximation of it.

Weight lists are **scale-invariant** — identical for C and every D. Only `decays`
changes. (Derived via `km/solve.py:solve_weights(k=8, shape='linear')`; the arm-C
decays reproduce the PR's published `kmaxwell_decay_rates` to <1e-12.)

```yaml
start_weights: [0.005093975, 0.010187949, 0.015281924, 0.020375898,
                0.025469873, 0.030563847, 0.035657822, 0.857368713]   # mean age 58*s
end_weights:   [0.032261839, 0.064523678, 0.096785516, 0.129047355,
                0.161309194, 0.193571033, 0.225832871, 0.096668514]   # mean age 26*s
```

`decays` per arm (tau = 8 log-spaced ages; beta = tau/(tau+1)):

```yaml
# C, all batches: tau [3, 64], age 58->26   (PR #357 as shipped)
[0.75, 0.822852439855, 0.877930338626, 0.917598547218,
 0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]

# D @ 0.25x: tau [12, 256], age 232->104   (x4)
[0.923076923077, 0.948927596166, 0.966407077116, 0.978042648561,
 0.985707613901, 0.99072224263, 0.993988168757, 0.996108949416]

# D @ 0.5x: tau [6, 128], age 116->52   (x2)
[0.857142857143, 0.902818485868, 0.934997769159, 0.957028830199,
 0.971818015605, 0.981615056307, 0.988048189779, 0.992248062016]

# D @ 2x: tau [1.5, 32], age 29->13   (x0.5)
[0.6, 0.699022338163, 0.782420529533, 0.84774327017,
 0.896059786816, 0.930304280845, 0.953847574219, 0.969696969697]
```

### Gates (hard)

1. Per-config 20-step finite-loss smoke before any full arm (REQ-025 precedent).
2. Usable-batch budget assert per config BEFORE launch (REQ-029 precedent — the 16x
   run exhausted fineweb 17 steps short because raw-token budgeting is wrong).
3. Tests green at the pinned SHA.
4. `microbatch_sequences` divides `batch_tokens/(8*1024)` — at 0.25x/0.5x the
   default 64 is illegal and will trip the accumulation assert.

### Artifacts

`logs/kmaxwell/req033_annealed_batch_curve/{README.md,summary.tsv,readout.tsv,
val_trajectories.txt,manifest.tsv,make_req033_configs.py,configs/,logs/}` — the
REQ-026/029 shape.

### Readout

Primary: **momentum benefit = final_val(kernel) - final_val(arm A), within batch
size**, at step 3250. Absolute loss is NOT comparable across batch sizes (2x sees
8x the tokens of 0.25x); only the within-batch contrast is.

Closing table, directly against the existing bi-Maxwell curve:

```
batch  batch_tokens  benefit(C-A)  benefit(D-A)  benefit(B-A)  bimax-mu0 (REQ-029)
```

The shape is the deliverable, no interpretation needed:

- **C decays toward zero like bi-Maxwell** -> the annealed kernel is also a
  denoiser; the PR's gain is batch-specific. Expected, given REQ-029.
- **C holds flat where bi-Maxwell decayed** -> the anneal buys something genuinely
  different from the frozen two-rate kernel. Would be the interesting result.
- **C decays but D holds** -> the *idea* is right and only the constants were fitted
  to one batch size; the recipe should be reparameterised in tokens, not steps.

**Known limitation to record in the README, do not paper over it:** age-scaling does
NOT fully restore constant update noise. Using `nesterov_filter_stats`, relative
update-noise variance vs 1x at the anneal endpoint is: arm C = {4.00, 2.00, 0.50}
at {0.25x, 0.5x, 2x}; arm D = {1.28, 1.11, 0.92}. Arm D removes most
but not all of it — the Nesterov term `h0 = (1-mu) + mu*sum(w*(1-beta))` has a floor
of `(1-mu)^2 = 0.0025` that no memory-lengthening removes. Exact noise-matching was
computed and rejected: it needs mean age ~837 at 0.25x, which is bias-dominated and
no longer the same hypothesis. If a follow-up wants true noise-matching, the knob is
co-scaling `mu`, not the ages.

**Seeds:** seed 0, n=1 per cell, matching REQ-026/028/029 discovery convention. Per
REQ-027 the harness reads tokens sequentially with no RNG shuffle and the fork loads
params+optimizer, so `seed` does not resample data order — cross-seed spread is
~2e-4 nondeterminism. Read deltas below ~5e-4 as noise. If a cell lands in that
band and the shape hinges on it, file a follow-up for replicates rather than
over-reading n=1.


## REQ-032: diligence (answer_free + answer_bearing) + tau2 (gold + step_hint) — 500 steps, H200/27B preferred

- status: OPEN
- agent status: **NEEDS-INFO — path decision (H200 blocked by my operator ceiling) (2026-09-02 ~17:55Z).** Accepted and ready to run (REQ-031 at step 175/200 on qv16djq, ~1.5h from done; I have room for a 2nd box within my ceiling). **One blocker before I provision:** your **preferred Path A = 8× H200 + Qwen3.8-27B**, and its whole scientific point is testing whether a *bigger* model breaks REQ-024's flat 8B judge curve. But my **standing hard operator constraint (Jerry) is "max 2 H100 nodes, nothing more"** — I'm **not authorized to provision H200** autonomously. Path B (H100/8B) *is* within my ceiling, but it re-tests the *same 8B* that was flat, so it won't answer the model-size question. **I don't want to silently default to the weaker path.** Please pick: **(A)** authorize H200 for this (i.e., lift my H100-only ceiling for REQ-032) → I run Path A (H200/27B, 500 steps, 4 arms); or **(B)** you're fine with H100/8B at the longer 500-step horizon → say so and I start Path B immediately (per your own fallback). Either way I'll run the 4 arms **sequentially on the 2nd box** (diligence answer_free/bearing first — those address the flat-curve question and don't share the tau2 user-sim DeepSeek load with REQ-031's gold; the two tau2 arms run last, by which point REQ-031 gold is finished, so no shared-key throttle overlap). Everything else is ready (same ac07c90 bootstrap + vllm/torchvision/audio patches + DeepSeek user-sim as REQ-031, which is working). Holding provisioning for your call.
- agent status: **RUNNING — Path B (H100/8B), started after >1h with no answer (2026-09-02 ~20:00Z).** Your request says "if H200 is unavailable, **do not wait** — take 8× H100," and my operator ceiling makes H200 unavailable *to me* → per your own default I'm proceeding on **Path B: 8× H100 + `Qwen/Qwen3-8B`**, 500 steps, eval every 25. **The H200/27B option stays open** — if you authorize H200 (lift my H100 ceiling), I'll switch REQ-032 to Path A; the two tau2 arms run *last* in the sequence (~35h each), so you have a long window to redirect before any tau2 compute is spent. **Arm order (per your spec):** (1) diligence `answer_free`, (2) diligence `answer_bearing` FIRST — these directly test whether the longer 500-step horizon breaks REQ-024's flat ~0.10 diligence-judge curve (a valid *horizon* question even on 8B, orthogonal to model size); then (3) tau2 `gold`, (4) tau2 `step_hint`. Provisioning a fresh 8×H100 box now (0 nodes in use → within my 2-node ceiling). Same ac07c90 bootstrap as REQ-031. Deliverable → `logs/async_sdpo_req032/`.
- requested: Jack / 2026-09-02 PDT
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact SHA: `ac07c90c1590823427a0a01f66bef6f69f0c3cf4`
- prior diligence/gold diagnosis: https://github.com/jacknzheng/kmaxwell-sota/tree/jerry-agent/logs/async_sdpo_req024
- do **not** preempt REQ-031 on qv16djq (8×H100, 8B gold). Provision a **separate** box. Within the standing 2-node ceiling, run in parallel with REQ-031 only if both boxes are authorized; otherwise queue behind REQ-031.

Follow-up to [REQ-024](https://github.com/jacknzheng/kmaxwell-sota/tree/jerry-agent/logs/async_sdpo_req024): diligence judge curves were flat at ~0.10 over 200 steps on 8B despite a live `teacher_minus_student_logp` gap. This fleet re-runs four arms at **500 steps** with eval every **25**, preferring a larger model on H200.

Four arms, **sequential on one 8-GPU node** unless Jack authorizes a second box for overlap:

1. diligence `answer_free`
2. diligence `answer_bearing`
3. tau2 `gold`
4. tau2 `step_hint`

### Hardware + model (pick one path; record which in README)

**Preferred — Path A (H200 + 27B)**

- Request **8× NVIDIA H200**.
- `model.model=Qwen/Qwen3.8-27B`
- Map: `cuda:0-3` policy vLLM TP=4; `cuda:4-7` FSDP2 trainer ×4.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (scripts already export this). Keep default `generator.engine.disable_custom_all_reduce=True`.

**Fallback — Path B (H100 + current 8B), only if H200 is unavailable**

- If H200 cannot be provisioned, **do not wait** — take **8× H100** and use the same model as REQ-024 / current default:
  - `model.model=Qwen/Qwen3-8B`
- Same 4+4 map. State clearly in README + `summary.tsv`: `path=B`, GPU model, and that Path A was unavailable (include the provisioner error / empty H200 inventory if any).
- Do **not** invent a different 9B / Qwen3.5 slug. Path B is exactly the REQ-024 stack at longer horizon.

Confirm **8 GPUs visible** before each arm; stop and record if fewer. `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. `torchrun --nproc-per-node=4`.

### Schedule (both paths)

```text
trainer.total_steps=500
judge.eval_interval=25
logging.checkpoint_interval=50
trainer.batch_size=16
total_num_gpus=8
generator.engine.n_rollout_gpus=4
trainer.n_trainer_gpus=4
```

Eval at steps **0 (baseline), 25, 50, …, 475, 500 (final)**. Do not use 200.

Let `MODEL` be `Qwen/Qwen3.8-27B` (Path A) or `Qwen/Qwen3-8B` (Path B).

### Frozen checkout

```bash
git fetch origin fix/hint-output-budget
git checkout ac07c90c1590823427a0a01f66bef6f69f0c3cf4
uv run --no-sync pytest -q -m 'not network'
```

Record SHA. Expect 228 passed, 2 skipped. No sandbox / no bwrap / no `--privileged` (banking path removed at this SHA).

### Preflight (once per box)

1. `uv sync --extra tau2`.
2. If vLLM missing: `uv pip install vllm==0.26.0`. If torchvision/torchaudio CUDA major checks abort import, apply the same no-op `_check_cuda_version()` patch as REQ-031.
3. Tau2 data: sparse-clone **data only** at `a2c024725189473d2d7cea3a5cfdbcc67478e41f` → `TAU2_DATA_DIR` (retail + airline).
4. DeepSeek `deepseek/deepseek-v4-flash` preflight (thinking off); record UTC + HTTP status.
5. Diligence judge: if eval stalls >30 min with zero held-out completions, stop that arm and document — do not burn the box.
6. Keys required: `PARALLEL_API_KEY`, `OPENROUTER_API_KEY`, `WANDB_API_KEY`. Never log secrets.

### Arm launches

```bash
# 1 — diligence answer_free
bash scripts/run_diligencebench.sh answer_free \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25

# 2 — diligence answer_bearing
bash scripts/run_diligencebench.sh answer_bearing \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25
```

W&B: `sdpo-diligence`. Graph judge metrics on default `_step` (not `eval/launched_at_step`). Watch bearing for weight-sync / EngineCore hangs (REQ-024); resume once with `logging.resume_from=latest`, then stop + evidence if it recrashes.

```bash
# 3 — tau2 gold
bash scripts/run_taubench.sh gold \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25

# 4 — tau2 step_hint
bash scripts/run_taubench.sh step_hint \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25 \
  "data.domains=[retail,airline]" \
  generator.engine.max_model_len=32768
```

`gold` / `run_taubench.sh` already pin retail+airline and `max_model_len=32768`. User-sim: DeepSeek V4 Flash, thinking off. W&B: `sdpo-tau2`. Ignore W&B `data.dataset_name=paperinstruments/diligence-bench` on tau2 runs — that field is unused when `data.dataset=tau2`; confirm domains + `eval/pass1*`.

### Training diagnostics (required in summary)

Per arm from `training.jsonl`: mean / p50 / p95 of `teacher_minus_student_logp`; count of steps with `|gap| < 1e-3`; loss @ step 10 vs final.

### Required artifacts

```text
logs/async_sdpo_req032/
  README.md          # path A or B, GPU model, MODEL slug, link to req024
  summary.tsv
  diligence-answer_free/
  diligence-answer_bearing/
  tau2-gold/
  tau2-step_hint/
```

Per arm: `args.txt`, `config.yaml`, `ARTIFACTS.txt`, `console.log`, `train.log`, `rank*.log`, `api_failures.jsonl`, `evaluations.jsonl`, `rollouts.jsonl`, `training.jsonl`, `vllm.jsonl`. Gzip large text. `git add -f` for ignored log patterns. No secrets, env dumps, weights, or checkpoint tensors.

`summary.tsv`: path, GPU, MODEL, SHA, CLI, steps completed, wall hours, eval at 0 / 250 / 500, gap stats, failure counters, W&B URL.

### Success

All four arms reach step 500 (or documented stop after one resume attempt); Path A or B recorded; eval every 25 through 500; diligence 30/30 held-out (or documented provider failure); tau2 `eval/pass1*` on W&B default step axis; gap stats in summary.


## REQ-031: tau2 gold only — DeepSeek user-sim, no sandbox, unprivileged GPU box

- status: OPEN
- agent status: **RUNNING (2026-09-01 ~00:05Z)** — this supersedes REQ-024's gold; the fixes here (DeepSeek user-sim instead of the 429-limited Nemotron, banking/bwrap removed, no sandbox) target exactly the walls I hit. Provisioning ONE **8-GPU** box **qv16djq** (not self-capping — 1 node, within my standing 2-node ceiling). Plan: checkout `ac07c90` on `fix/hint-output-budget`, `uv run --no-sync pytest -q -m 'not network'` (expect 228p/2s) + record SHA, `uv sync --extra tau2`, sparse-clone tau2 data-only at `a2c0247` → `TAU2_DATA_DIR`, DeepSeek `deepseek-v4-flash` preflight ping (thinking off, temperature 0) with UTC+status, then `bash scripts/run_taubench.sh gold trainer.total_steps=200` (CUDA_VISIBLE_DEVICES=0-7, nproc-per-node=4, cuda:0-3 vLLM TP4 + cuda:4-7 FSDP2). NO bwrap / NO --privileged / NO Nemotron / NO :free. Will confirm 8 GPUs visible before launch (stop+record if fewer). Artifacts → `logs/async_sdpo_req031/{README,summary.tsv,tau2-gold/}` with the W&B eval/pass1* step curve.
- agent status: **PREFLIGHT PASS → gold RUNNING (2026-09-02 ~06:10Z).** Bootstrap on qv16djq: checkout **ac07c90** (clean), offline `pytest -m 'not network'` = **228 passed, 2 deselected** ✓, `uv sync --extra tau2` ✓, tau2 data sparse-clone @ a2c0247 (retail+airline tasks.json) → TAU2_DATA_DIR ✓. **8 GPUs visible** ✓. **DeepSeek preflight: status 200** (provider SiliconFlow, no 402) at UTC 2026-09-02T05:55:55Z ✓. Confirmed **ac07c90 already contains my REQ-024 tau2 patch** (solo_mode=False, env_kwargs strip, `_retry_transient` 429-hardening — the owner applied it). Generated `config.yaml` verified: user_llm=**openrouter/deepseek/deepseek-v4-flash**, user-sim `default_user_llm_args` = **temperature 0.0 + reasoning{enabled:false,effort:none}** ✓, domains retail+airline, max_model_len 32768, batch_size 16, total_steps 200, n_rollout_gpus 4, n_trainer_gpus 4, eval_interval 25, checkpoint_interval 50, total_num_gpus 8 (judge.model is still nemotron:free but gold never calls it ✓). **Venv fixes needed (not in the request, same class as REQ-024):** vllm isn't a declared dep → `uv pip install vllm==0.26.0`; and torch 2.11.0+cu128 has no matching cu128 torch**vision**/torch**audio** (vllm pulled cu13 builds), whose fatal CUDA-major checks abort a text-only run — patched both `_check_cuda_version()` to no-op (no vision/audio CUDA ops are used; verified Qwen3ForCausalLM + vllm import clean). Gold now past model-load: **Qwen3ForCausalLM resolved, V1 vLLM engine v0.26.0 initializing, weights loading**; no 429s. Baseline eval (full official test @ step 0) next.
- agent status: **DONE (2026-09-02 ~19:55Z) — full 200-step gold run COMPLETE, GOLD_EXIT=0, node released.** Deliverable: `logs/async_sdpo_req031/{README.md,summary.tsv,tau2-gold/}` (config, args, ARTIFACTS, console.log.gz, train.log.gz, rank{1,2,3}.log, evaluations.jsonl.gz [source of truth], rollouts/training/vllm/api_failures.jsonl.gz). **W&B: https://wandb.ai/jacknzheng-united-states-department-of-state/sdpo-tau2/runs/8xt6ktgo** (eval/pass1, pass1_retail, pass1_airline on default `_step`). **Full pass^1 curve (n=60 each, 0 rollout errors everywhere):** baseline 0.283 → 25 **0.433** → 50 0.333 → 75 0.25 → 100 0.333 → 125 0.333 → 150 0.333 → 175 0.283 → final@200 **0.300** (retail 0.40→0.375, airline 0.05→0.15). Shape: the 8B policy peaks at step 25 then oscillates around baseline — **no sustained gain over 200 steps**, mirroring REQ-024's flat-8B finding (this is exactly REQ-032's motivation for 27B). **This completed where REQ-024's gold deadlocked:** DeepSeek user-sim (preflight 200 @05:55:55Z) + no sandbox + the retry-hardening absorbed **2231 rate-limits with api_failures=0**. Preflight recorded: pytest 228p/2s, `uv pip install vllm==0.26.0` (not a declared dep), torchvision/torchaudio `_check_cuda_version()` no-op patch (torch 2.11 has no cu128 build; safe for text-only), tau2 data @a2c0247, 8 GPUs, 4+4 map. **Node qv16djq released** (frees my 2-node budget for REQ-032 once you pick the path).
- requested: Jack / 2026-09-01 PDT
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact SHA: `ac07c90c1590823427a0a01f66bef6f69f0c3cf4`
- supersedes gold on REQ-024 (`ecf6fd8`) and withdrawn REQ-030. Do **not** reopen those.
- do not collide with kmaxwell REQ-021–029 (those numbers are already used)

Tau2 **gold only**. Retail + airline. No diligence, no banking, no hint GPU, no Nemotron user-sim.

Banking and the `srt`/`bwrap` host sandbox were **removed** at this SHA. Do not run `setup_tau2_sandbox.sh` (the file is gone). Do not install bubblewrap. Do not request `--privileged` or `seccomp=unconfined`. Those host-policy changes can break NCCL / the GPU driver. Retail and airline use in-process DB tools only.

Request an **8-GPU** box. Jack has operator permission. If fewer than 8 devices are visible, record the listing, update this block, and stop. Do not self-cap at 2, do not use `NPROC=1`.

### Hardware map (required)

```text
cuda:0-3  policy vLLM TP=4
cuda:4-7  FSDP2 trainer x4
```

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. `torchrun --nproc-per-node=4`. `gold` skips the hint LLM.

### Frozen checkout

```bash
git fetch origin fix/hint-output-budget
git checkout ac07c90c1590823427a0a01f66bef6f69f0c3cf4
uv run --no-sync pytest -q -m 'not network'
```

Record that SHA. Offline suite on the authoring machine: 228 passed, 2 skipped.

Resolved defaults that must appear in `config.yaml`:

```text
model.model=Qwen/Qwen3-8B
total_num_gpus=8
generator.engine.n_rollout_gpus=4
trainer.n_trainer_gpus=4
trainer.batch_size=16
trainer.total_steps=200
generator.engine.max_model_len=32768
generator.hint.prompt=gold
data.dataset=tau2
data.domains=[retail,airline]
data.user_llm=openrouter/deepseek/deepseek-v4-flash
judge.eval_interval=25
logging.checkpoint_interval=50
```

`gold` never calls `judge.model`. Do not point `data.user_llm` at Nemotron or any `:free` slug. User-sim payloads must include `temperature: 0` and `reasoning: {enabled: false, effort: "none"}`. Do not enable thinking.

### Preflight (no sandbox)

1. `uv sync --extra tau2` (not `--extra knowledge`; that extra is gone).
2. If `.deps/tau2-bench/data` is missing, sparse-clone **data only** at pinned `a2c024725189473d2d7cea3a5cfdbcc67478e41f` (retail/airline `tasks.json`). Set `TAU2_DATA_DIR` to that `data/` dir. This is task JSON, not a sandbox.
3. OpenRouter DeepSeek is funded. Send one real user-sim-shaped request to `deepseek/deepseek-v4-flash` (thinking off) and record status plus UTC timestamp, never keys. If the box returns 402, report which credential differs and stop.

Do not eval the train split. Held-out is tau2's official `loader("test")`.

### Single launch (baseline → train → final)

```bash
bash scripts/run_taubench.sh gold trainer.total_steps=200
```

The script already sets `data.domains=[retail,airline]` and `max_model_len=32768`. One process scores the full official test set at:

1. **baseline** — `eval/phase=baseline`, wandb `step=0`, before rollout
2. **interval** — 25, 50, …, 175 (`eval/phase=interval`)
3. **final** — after step 200 (`eval/phase=final`, wandb `step=200`)

W&B must graph `eval/pass1`, `eval/pass1_retail`, `eval/pass1_airline` on the **default `_step` x-axis**. Do not set panel x to `eval/launched_at_step`. File `evaluations.jsonl` remains the source of truth.

Resume after recoverable crashes with `logging.resume_from=latest`. Skip the in-run baseline on resume.

### Required artifacts

```text
logs/async_sdpo_req031/
  README.md
  summary.tsv
  tau2-gold/
```

Include `args.txt`, `config.yaml`, `ARTIFACTS.txt`, `console.log`, `train.log`, every `rankN.log`, `api_failures.jsonl`, `evaluations.jsonl`, `rollouts.jsonl`, `training.jsonl`, `vllm.jsonl`. Use `git add -f` for ignored log patterns. Gzip large text losslessly. Never commit secrets, env dumps, weights, or checkpoint tensors.

`summary.tsv`: exact SHA and CLI, visible GPU count, completed steps, wall time, overall and per-domain pass^1 at baseline / each interval / final, user-sim 429 count, every failure counter. README links raw dirs and the W&B URL.

### Success

8 visible GPUs; **no privileged / no bwrap**; user-sim is DeepSeek V4 Flash with thinking off; full official retail+airline test scored at 0, 25, …, 175, 200; those `eval/pass1*` points visible on W&B default step charts; `evaluations.jsonl` complete.

If a definitive external blocker stops the run, commit and push progress plus raw logs, update this block, and stop. Soft-reset / rebase if a push is rejected. Do not force-push.


## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```

---

## REQ-024: 8-GPU 4+4 OpenRouter DeepSeek SDPO fleet

- status: **DONE for diligence; gold not delivered** — keep this block as the diagnosis record. Do **not** reopen gold on `ecf6fd8`. Gold follow-up is REQ-031 at `ac07c90` (no sandbox).
- agent status: **DILIGENCE DELIVERED / gold STOPPED (deadlocked on provider) — node released (2026-08-31 ~13:35Z).** Update to the above: I let gold grind and it **deadlocked at the first eval boundary**, not just slowed — `evaluation_started=1` at 12:53Z with **0 held-out tasks completed 37+ min later** (443× upstream 429). At that throughput the first eval won't finish, let alone 200 steps, so continuing only burned the node at ~zero output. I **stopped the run and released wgmy26w** (both nodes now free). This is fully **resumable** and preserved (config in `tau2-gold-attempt/gold-config.yaml`, deadlock evidence in `tau2-gold-attempt/DEADLOCK-EVIDENCE.txt`): apply the patch, re-provision, relaunch `bash scripts/run_taubench.sh gold trainer.total_steps=200 "data.domains=[retail,airline]" generator.engine.max_model_len=32768` — but **gold needs a dedicated Nemotron provider key (BYOK)** or a less-contended user-sim/judge model to get through; I don't have one. **REQ-024 net: the core deliverable is done** — both diligence arms @200 + hint-gate PASS + the 3-fix tau2 patch, all in `logs/async_sdpo_req024/` (see `README.md`). Gold (Ablation A) is the only unshipped piece and it's blocked on external OpenRouter capacity, not on anything in the code. Ready to resume gold the moment a working provider key/model is available. **Owner 2026-09-01:** gold resume is REQ-031 at `ac07c90`; do not resume via withdrawn REQ-030 or `ecf6fd8`.
- agent status: **DILIGENCE ARMS DELIVERED; tau2 gold BLOCKED-IN-PRACTICE on OpenRouter provider capacity — DECISION NEEDED (2026-08-31 ~12:15Z).** Two more blockers hit on gold after the ones above: (3) the tau2-bench **data dir was missing** → cloned at pinned a2c0247; (4) even the **paid** `nemotron-3-super-120b` user-sim/judge returns **upstream shared-pool 429s** ("temporarily rate-limited upstream / engine_overloaded", DeepInfra+DigitalOcean), and the harness re-raised on them → the run died at step 8. I added **retry-with-backoff hardening** to the user-sim path (3rd fix; the delivered patch now carries all three: solo_mode, env_kwargs, retry). Gold now **survives** (relaunched, run 112840) — but it is **provider-throttled to a crawl: step 13/200 after 45 min, 233 × 429 absorbed via backoff, ≈3.5 min/step ⇒ ~11h to 200** (first eval boundary ~40 min out). Nothing I can change server-side speeds up an external shared-pool rate limit. **Your call:** (a) let it grind (it should finish overnight if the pool doesn't worsen; the 25/50/… evals land as it goes), (b) accept a **partial gold** (e.g. stop at the first 1–2 eval boundaries for the retail+airline pass^1 readout), or (c) drop in a **dedicated Nemotron provider key** (BYOK, per OpenRouter's own remedy hint) to lift the limit — I don't have one. **The core REQ-024 deliverable (both diligence arms @200 + hint-gate validation) is done and collected**; gold is Ablation A on top. I'll keep it running and land whatever evals complete unless you say otherwise. Updated patch: `logs/async_sdpo_req024/tau2-solo_mode-fix.patch` (git am → push to `fix/hint-output-budget`).
- requested: Jack / 2026-08-30 20:59 PDT
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact SHA: `ecf6fd84e21281ff1460169da61006e879886e5e`
- supersedes: REQ-018, REQ-020
- prior evidence: `logs/async_sdpo_req011/`, `logs/async_sdpo_req015/`, `logs/async_sdpo_req018/`

REQ-020 cannot run: Baseten workstations cap at 8 GPUs, so the 9-GPU local-hint
map is not provisionable. REQ-018 validated the hint-length fix at `3bd7def`
(0 drops / 0 `openrouter_length` over ~478 attempts/arm) and then died on
free-Nemotron 429s. This request puts hints on OpenRouter
`deepseek/deepseek-v4-flash`, restores the proven 4+4 split on 8 GPUs,
and keeps thinking off for both hints and the diligence judge.

REQ-018 and REQ-020 are removed from this queue. Their logs stay under
`logs/async_sdpo_req018/`. Do not preempt REQ-022 or REQ-023; use a
separate 8-GPU box.

Request an **8-GPU** box. Jack has operator permission. The operator default
of 2 GPUs is not enough. Do not self-cap at 2, do not start a local hint
engine, do not use `NPROC=1`, and do not recreate the 4+3+1 or 9-GPU maps.
If `nvidia-smi` / `torch.cuda.device_count()` shows fewer than 8 visible
devices, record the listing, update this block, and stop.

### Hardware map (required)

```text
cuda:0-3  policy vLLM TP=4
cuda:4-7  FSDP2 trainer x4
```

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. `torchrun --nproc-per-node=4`.
Hints are remote OpenRouter calls, not a GPU process.

### Frozen checkout

Do **not** use SHA `14db9fb` / `e2ff718` (local hint GPU), `3bd7def`
(free-Nemotron hints), or `b4d523d` (invalid `-latest` slug). Fetch this
exact commit. Do not recreate this as an on-box patch.

```bash
git fetch origin fix/hint-output-budget
git checkout ecf6fd84e21281ff1460169da61006e879886e5e
uv run --no-sync pytest -q -m 'not network'
```

Record that SHA. Offline suite at this SHA: `229 passed, 2 skipped`.

Resolved defaults that must appear in `config.yaml`:

```text
model.model=Qwen/Qwen3-8B
total_num_gpus=8
generator.engine.n_rollout_gpus=4
trainer.n_trainer_gpus=4
trainer.batch_size=16
generator.hint.backend=openrouter
generator.hint.model=deepseek/deepseek-v4-flash
generator.hint.reasoning_enabled=false
generator.hint.max_tokens=2048
judge.model=nvidia/nemotron-3-super-120b-a12b:free
judge.reasoning_enabled=false
data.user_llm=openrouter/nvidia/nemotron-3-super-120b-a12b:free
judge.eval_interval=25
logging.checkpoint_interval=50
```

Every hint and judge OpenRouter payload must include
`reasoning: {enabled: false, effort: "none"}`. Do not enable thinking.

### Preflight

OpenRouter (DeepSeek hints, Nemotron judge / user-sim) and Parallel Search
are funded. Before renting or restarting a box, send one real request to
each and record status plus UTC timestamp, never keys. The hint preflight
must use `deepseek/deepseek-v4-flash` through `build_error_hint` /
`generate_hint` (`backend=openrouter`). If the box still returns 402,
report which credential differs and stop that dependent arm. Do not put
credentials in this repository.

### Diligence arms

Preserved checkpoints on `qkpx8dw` / `wp2znpq` were lost in the operator
reset. Restart both diligence arms from step 0 on `ecf6fd8`. That is
intended: this SHA changes the hint model and the GPU split.

### Hint validation gate

After at least 100 post-switch hint attempts on each diligence arm,
report:

- attempts, successes, total drops, and `drops / attempts`
- every `hint_drop_*` cause, especially `openrouter_length`, `openrouter_429`,
  `timeout`, `empty`
- confirm the hint model slug is `deepseek/deepseek-v4-flash`

Gate passes when the total drop rate is below 5% while OpenRouter is
healthy and there are zero `hint_drop_openrouter_length` failures. If the
gate fails, preserve raw artifacts, diagnose, update this block, and stop.
Do not raise concurrency, switch hint models, enable reasoning, start a
local hint GPU, or train with an empty hint.

### Tau2

Do not delay diligence on tau2. Resolve the already-reported tau2-bench
`a2c0247` `get_environment` API drift, add a test covering the pinned
signature, commit that follow-up to `scaling-sdpo`, then launch tau2 `gold`
on retail+airline only. `gold` skips the hint LLM. Do not use banking on
Baseten.

### Frozen experiment

Finish all three arms at `trainer.total_steps=200`:

1. diligence `answer_free`
2. diligence `answer_bearing`
3. tau2 `gold` on retail+airline

Runtime per arm:

- 4 vLLM rollout GPUs + 4 FSDP2 trainer ranks
- `model.model=Qwen/Qwen3-8B`
- `trainer.mini_batch_size=2`
- `generator.engine.max_model_len=32768` for tau2
- never fall back to 4B, `stealth/ox-alpha`, `NPROC=1`, 2 GPUs, or a
  local hint engine

Resume after recoverable crashes. A rollout without its required hint must
be dropped. Keep search, hint, user-simulator, judge, sandbox, empty-episode,
stale-rollout, and weight-sync failures as separate counters.

W&B must receive a `samples` table each logged train step with prompt,
output, `hint_free`, and `hint_bearing`. File artifacts remain the source
of truth.

### Evaluation contract

Held-out eval runs synchronously at every 25-step boundary (25, 50, 75, 100,
125, 150, 175, 200). Do not skip a boundary because a prior eval is still
running.

Write every eval to `evaluations.jsonl`. A W&B screenshot is not a
substitute.

- Diligence: per-task query, response, token counts, normalized/raw judge
  scores, per-section earned/possible/fraction, judge error; plus aggregate
  `judge_score`, `judge_n`, `judge_errors`, section scores, requested count,
  and rollout-error count.
- Tau2: per-task domain, query, full transcript, token counts, pass^1, and
  rollout error; plus overall and per-domain pass^1, sample counts, requested
  count, and rollout-error count.

Every row and aggregate must include `launched_at_step` and `policy_version`.
Commit `evaluation_started`, task-completed/task-failed, and
`evaluation_completed`/`evaluation_failed` even when an API or sandbox
failure occurs.

### Required artifacts

Commit and push to this `jerry-agent` branch:

```text
logs/async_sdpo_req024/
  README.md
  summary.tsv
  hint-fix-validation.tsv
  diligence-answer-free/
  diligence-answer-bearing/
  tau2-gold/
```

For every arm include `args.txt`, `config.yaml`, `ARTIFACTS.txt`,
`console.log`, `train.log`, every `rankN.log`, `api_failures.jsonl`,
`evaluations.jsonl`, `rollouts.jsonl`, `sandbox.jsonl`, `training.jsonl`,
`vllm.jsonl`, sandbox setup logs, and a checkpoint/resume manifest. Use
`git add -f` for ignored log patterns. Gzip large text artifacts losslessly
and document decompression. Never commit secrets, environment dumps, model
weights, or checkpoint tensor files.

`summary.tsv` must include exact SHA and CLI, visible GPU count, completed
steps, wall time, checkpoint history, teacher-minus-student gap, dead-gap
fraction, clipping, staleness, every failure counter, and held-out metrics.
The README must link each row to its raw directory.

Success: 8 visible GPUs, hint gate passes, thinking stays off, and all three
arms reach step 200. If a definitive external blocker stops the run, first
commit and push all progress and raw failure logs, then update this block
with exact evidence and stop rather than restarting from zero.

---

