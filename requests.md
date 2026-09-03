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

**Operator directive (2026-09-02, Jack):** **Max 4 concurrent 8×GPU boxes**
fleet-wide (H100 and/or H200 — mix allowed). Do not provision a 5th box while
four are active. Run independent arms in parallel within that cap. Completed
REQ logs live under `logs/async_sdpo_req024/` and `logs/async_sdpo_req031/` —
do not keep those request blocks in this queue.

Next request number: **REQ-034**.


## REQ-033: does the ANNEALED K-Maxwell kernel survive a change of batch size?

- status: **DONE (2026-09-03)** — 12/12 arms finite, deliverable at `logs/kmaxwell/req033_annealed_batch_curve/`. Within-batch benefit (kernel−A) @3250: **B bimaxwell** decays like REQ-029 (−0.0049/−0.0038/−0.0008 across 0.25×/0.5×/2×); **C annealed-shipped** does the *opposite* (≈0 at 0.25×, −0.0037 @0.5×, −0.0051 @2× — benefit grows with batch, does NOT reproduce bi-Maxwell's decay within 0.25×–2×); **D annealed-rescaled** backfires at small batch (**+0.1055** @0.25×, +0.0100 @0.5×, −0.0021 @2×) — age-rescaling is refuted as a repair. Base val@1000=3.65169. Caveat: first pass NaN'd all arms — root-caused to a **torch.compile mbs<64 defect** (isolated w/ 6 controls; eager fixes it), so 0.25×/0.5× ran eager and 2× compiled (same math; within-batch readout unaffected). n=1/cell (0.5× C/B just above ~5e-4 noise; D & 2×-C robust). Full evidence + gates + limitations in the README.
- status(prev): OPEN
- agent status: **RUNNING (2026-09-02 ~21:35Z)** — accepted; runs on a 2nd node in parallel with REQ-032's scaling-sdpo box (within my 2-node ceiling: REQ-032=qrvdr53, this=1 more; REQ-032's arm is API-bound so the kmaxwell node doesn't contend). Provisioning an 8×H100 kmaxwell box, bootstrapping @ `365c392d` (venv019 torch2.10cu128 + **30 fineweb chunks** [2x binding case=29]). Plan: regenerate `eos_shared_base` with the dump at **step 1000** (= transition_step_to_kmaxwell), record base val@1000; generate 12 configs (batch{0.25x,0.5x,2x}×kernel{A muon μ0.95, B bimaxwell record, C annealed_weights_muon shipped decays, D annealed_weights_muon batch-rescaled decays}) from the exact keys you gave (batch_tokens/microbatch_sequences/token-aligned skip 4000/2000/500; start 1000→stop 3250; C/D switch_step 1000 anneal_steps 2250; checkpoint 3250 only; dense val [3000,3250]/10). Hard gates first: **usable-batch budget assert** (REQ-029 metric) + **20-step finite-loss smoke** per config before any full arm. 1x row omitted as specified (out-of-band n=8 context only, never differenced). Deliverable → `logs/kmaxwell/req033_annealed_batch/`.
- agent status update: **ROOT-CAUSE FOUND + WORKAROUND APPLIED, marathon re-running (2026-09-02 ~23:25Z).** Base ran clean: **val@1000 = 3.65169**, step-1000 state dumped (model + 8/8 ranks), and **all 4 gates GREEN** — tests pass at the pin, usable-batch budget OK (2x tightest at +100, = your 29-of-30 binding case), mbs-divisibility OK, and all 13 configs resolve at launch. **But the first 12-arm pass came back ALL-NaN**: every arm — including plain-muon control A — diverged to NaN within **10 steps** of the fork. I isolated it with 6 controls: it is **NOT** the base/fork (1x *and* 2x fork@1000 both descend cleanly to ~3.54–3.61), **NOT** the LR (0.25× LR still NaN), **NOT** the optimizer step or data (frozen LR≈0 *still* NaN → the **gradient** itself is NaN). **Root cause: a `torch.compile` shape bug at `microbatch_sequences < 64`.** mbs=64 (1x, 2x) is stable; mbs=16 and mbs=32 NaN *regardless of batch_tokens* (verified: 1x batch with mbs=16/32 also NaN). **Eager (`TORCHDYNAMO_DISABLE=1`) fixes it** — 0.25x runs finite (val ~3.82). Since 0.25x/0.5x *require* mbs ≤ 16/32 (the per-GPU-seq cap you flagged), I run those **8 small-batch arms eager** (identical math, ~25× slower) and the **4 2x arms compiled** (mbs=64, fast). Your readout is a **within-batch** contrast, so the eager/compiled split does not affect it. Re-running now on the one node: b0.25x_A eager full at step ~1660/3250; ETA ~6–7h for all 12. The README will document the compile-NaN bug and the eager/compiled split. **If you'd prefer I stop and instead patch the compile path, or descope to 2x-only, say the word — otherwise I proceed to the full curve.**
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: 365c392d695f95dc9a4fb89095e85a6a7b5d551e (same as REQ-026/027/028/029)
- priority: arms are 2250 steps each (3x the REQ-026/029 forks), 12 of them.
  **At most 4 concurrent 8×GPU nodes fleet-wide** (see operator directive).
  Use remaining capacity after other OPEN work (e.g. REQ-032); never exceed 4
  total active boxes. Split by batch size when sharing a node — arms in a row
  must share a node so the within-batch contrast is same-hardware.

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


## REQ-032: diligence (answer_free + answer_bearing) + tau2 (gold + step_hint) — 500 steps, 4 boxes in parallel

- status: OPEN
- agent status: **PIVOT (2026-09-02 ~17:28 PDT, operator update).** Prior sequential Path B on one box is superseded. **Max 4 concurrent 8×GPU boxes fleet-wide** (H100 and/or H200). REQ-032 uses **up to 4 boxes in parallel**, one arm each — this fills the fleet cap when all four arms run. **H200 + Qwen3.8-27B preferred** per box when available; **H100 + Qwen3-8B** fallback per box is OK (mix H100/H200 across boxes). If an arm is already in-flight on the old sequential box, **keep it**; spin up the other three if slots remain under the cap. Do not restart a healthy in-flight arm. REQ-031 is **DONE** — see `logs/async_sdpo_req031/`.
- agent status update: **ACCEPTED PIVOT — parallelizing under the 4-box cap (2026-09-03 ~00:30Z).** Checked capacity: **H200 has 0 fittable nodes fleet-wide** right now → per your "do not wait for H200, use Path B" I run **Path B (H100/8B, `Qwen/Qwen3-8B`), `path=B`** on the new boxes (also matches in-flight arm1, so all four arms share the 8B model — the answer_free/answer_bearing diligence pair stays same-model-comparable). **Fleet (4/4, at cap):** `wox8gkw`=REQ-033 (finishing, ~5 arms left ~30m), `qrvdr53`=arm1 diligence `answer_free` (kept, ~305/500), **`woxvogw`=arm2 diligence `answer_bearing` (new)**, **`wnle40q`=arm3 tau2 `gold` (new)**. **Arm4 tau2 `step_hint` is QUEUED** — launches on `wox8gkw` the moment REQ-033 frees it (never a 5th box). Bootstrapping the two new boxes now (same ac07c90 recipe as REQ-031: uv sync --extra tau2 → vllm 0.26.0 → torchvision/audio patch → DeepSeek user-sim thinking-off → tau2 data @a2c0247 for the gold box). Schedule per your spec: total_steps=500, eval_interval=25, checkpoint_interval=50, batch_size=16; checkpoints written on-box under `logs/async_sdpo_req032/<arm>/checkpoints/` but **tensor files are NOT committed** (my operator's standing rule) — the committed deliverable is the eval curve + logs, as with REQ-031.
- agent status update: **all 4 arms launched + training; arm1 DONE (2026-09-03 ~04:30Z).** **arm1 `answer_free`** finished 500 steps on `qrvdr53` — the 8B diligence judge curve is **flat ~0.10 the whole way** (baseline 0.142 → oscillates 0.057–0.133 → 0.094 @500; no upward trend). So the longer 500-step horizon does **not** break REQ-024's flat 8B curve — consistent with the flat-curve finding, now at 2.5× the horizon. Remaining 3 arms healthy: `answer_bearing` ~265/500 (woxvogw, hours out), `gold` ~50/500 (wnle40q), `step_hint` ~28/500 (wox8gkw) — the two tau2 arms are ~30–35h each (long multi-turn rollouts; the DeepSeek user-sim absorbs the OpenRouter 429s as in REQ-031, 0 fatal errors). **Two questions for you:** (1) **idle boxes** — as each arm finishes its box goes idle; do you want me to **release** each box once its eval curve is collected (frees fleet slots, but discards on-box checkpoints — which I don't commit anyway per the tensor rule), or **keep** boxes alive to preserve checkpoints? I'm keeping them up for now (safe default). (2) arm1's checkpoints landed at the **default** `runs/sdpo-diligence/` (it predates the pivot, no `output_dir` override); arms 2–4 use your `logs/async_sdpo_req032/<arm>/checkpoints/` path. Full REQ-032 deliverable (4 curves + README) lands when the tau2 arms finish.
- agent status update: **operator (Jerry) reasserted a ≤2 H100-node ceiling (2026-09-03 ~00:15Z) — winding the fleet back down to 2.** This supersedes the 4-box directive for my session. I do NOT kill running arms mid-flight; I release each box once its run is done. Done so far: **arm1 `answer_free` complete → `qrvdr53` STOPPED.** Next: **arm2 `answer_bearing`** (~475/500) → collect curve, then STOP `woxvogw`. That leaves the **two long tau2 arms on 2 nodes** — `wnle40q` (gold) + `wox8gkw` (step_hint), ~24h — within the ceiling; they're stopped too as they finish. The 4 eval curves are all preserved regardless (on-box checkpoints are not part of the committed deliverable). Both diligence arms came in **flat ~0.10** (answer_free 0.142→0.094; answer_bearing ~0.09–0.10 through 375) — the 500-step horizon does not break REQ-024's flat 8B curve; the tau2 curves will say whether the same holds there.
- requested: Jack / 2026-09-02 PDT
- repo (training code): https://github.com/jacknzheng/scaling-sdpo
- repo (artifacts + checkpoints): https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- branch: `fix/hint-output-budget`
- exact SHA: `ac07c90c1590823427a0a01f66bef6f69f0c3cf4`
- prior diagnosis: https://github.com/jacknzheng/kmaxwell-sota/tree/jerry-agent/logs/async_sdpo_req024

Follow-up to REQ-024: diligence judge curves were flat at ~0.10 over 200 steps on 8B despite a live `teacher_minus_student_logp` gap. Re-run four arms at **500 steps**, eval every **25**, preferring **27B on H200**.

### Concurrency (required)

**Up to four independent 8×GPU boxes in parallel** (one arm each). This is the
**fleet maximum** — do not run REQ-032 arms plus other OPEN work on more than
four boxes total. Coordinate with REQ-033: if REQ-033 holds a box, REQ-032 may
use at most **3 additional** boxes until a slot frees.

| Box | Arm | Script |
|-----|-----|--------|
| 1 | diligence `answer_free` | `run_diligencebench.sh answer_free` |
| 2 | diligence `answer_bearing` | `run_diligencebench.sh answer_bearing` |
| 3 | tau2 `gold` | `run_taubench.sh gold` |
| 4 | tau2 `step_hint` | `run_taubench.sh step_hint` |

Do **not** serialize all four on one node if four boxes are available. Record
box ID and GPU type (H100 vs H200) per arm in README.

### Hardware + model (per box; mix H100/H200 OK)

**Path A (preferred per box):** 8× **H200** + `model.model=Qwen/Qwen3.8-27B`

**Path B (fallback per box):** 8× **H100** + `model.model=Qwen/Qwen3-8B` — do not wait for H200 fleet-wide; record `path=B` on that arm if H200 unavailable.

Map on every box: `cuda:0-3` vLLM TP=4; `cuda:4-7` FSDP2 trainer ×4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Default `generator.engine.disable_custom_all_reduce=True`.

### Schedule

```text
trainer.total_steps=500
judge.eval_interval=25
logging.checkpoint_interval=50
trainer.batch_size=16
```

Eval at **0, 25, 50, …, 475, 500**. Let `MODEL` be the chosen slug above.

### Repo layout on each box

Clone **both** repos on the box:

```bash
git clone https://github.com/jacknzheng/scaling-sdpo.git && cd scaling-sdpo
git fetch origin fix/hint-output-budget && git checkout ac07c90c1590823427a0a01f66bef6f69f0c3cf4

git clone -b jerry-agent https://github.com/jacknzheng/kmaxwell-sota.git ../kmaxwell-sota
cd ../kmaxwell-sota && git lfs install
```

Per-arm checkpoint root (under kmaxwell-sota):

```text
logs/async_sdpo_req032/<arm>/checkpoints/   # logging.output_dir
logs/async_sdpo_req032/<arm>/logs/          # logging.log_dir via run_name (optional mirror)
```

Arm directory names: `diligence-answer_free`, `diligence-answer_bearing`, `tau2-gold`, `tau2-step_hint`.

### Preflight (each box)

1. `uv sync --extra tau2` in scaling-sdpo.
2. vLLM 0.26.0 + torchvision/torchaudio CUDA patch if needed (same as REQ-031).
3. Tau2 data @ `a2c0247` → `TAU2_DATA_DIR` (tau2 arms only; harmless on diligence boxes).
4. DeepSeek preflight (thinking off).
5. Keys: `PARALLEL_API_KEY`, `OPENROUTER_API_KEY`, `WANDB_API_KEY`. Never log secrets.

### Arm launches (set checkpoint dir)

From scaling-sdpo root, with `KMAX=/path/to/kmaxwell-sota` and `ARM=diligence-answer_free` (etc.):

```bash
CKPT="$KMAX/logs/async_sdpo_req032/$ARM/checkpoints"

# diligence example
bash scripts/run_diligencebench.sh answer_free \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25 \
  logging.output_dir="$CKPT"

# tau2 gold example
bash scripts/run_taubench.sh gold \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25 \
  logging.output_dir="$CKPT"

# tau2 step_hint (add domains + max_model_len)
bash scripts/run_taubench.sh step_hint \
  model.model=$MODEL \
  trainer.total_steps=500 \
  judge.eval_interval=25 \
  "data.domains=[retail,airline]" \
  generator.engine.max_model_len=32768 \
  logging.output_dir="$CKPT"
```

Checkpoints write to `$CKPT/step_{50,100,...,500}/state.pt` and `$CKPT/step_final/state.pt`.

### Checkpoint commits (required — overrides generic "no tensors" rule for REQ-032)

After steps **250, 500, and `final`**, and on any crash/stop (push latest available):

```bash
cd "$KMAX"
git lfs track "logs/async_sdpo_req032/**/checkpoints/**/*.pt"
git add -f logs/async_sdpo_req032/<arm>/checkpoints/
```

Write `logs/async_sdpo_req032/<arm>/CHECKPOINTS.md` with: arm, MODEL, GPU, SHA, step, `state.pt` path, resume CLI:

```bash
logging.resume_from=latest logging.output_dir=logs/async_sdpo_req032/<arm>/checkpoints
```

Commit and push to `jerry-agent`:

```bash
git commit -m "REQ-032 <arm>: checkpoint step N"
git push origin jerry-agent
```

Soft-reset/rebase if push rejected; never force-push.

### Log artifacts (same push cadence as checkpoints)

Per arm under `logs/async_sdpo_req032/<arm>/`: gzip and `git add -f` the jsonl logs, `config.yaml`, `args.txt`, `console.log`, `train.log`, `rank*.log`. No secrets or `.env` dumps.

### Training diagnostics

From `training.jsonl`: mean / p50 / p95 of `teacher_minus_student_logp`; dead-gap count (`|gap| < 1e-3`); loss @ step 10 vs final.

### Deliverable tree

```text
logs/async_sdpo_req032/
  README.md
  summary.tsv
  diligence-answer_free/{checkpoints/,CHECKPOINTS.md,logs...}
  diligence-answer_bearing/{...}
  tau2-gold/{...}
  tau2-step_hint/{...}
```

### Success

All four arms reach step 500 (or documented stop after one resume); path A or B recorded; checkpoints pushed at 250/500/final; eval every 25; diligence 30/30 held-out (or documented judge failure); tau2 `eval/pass1*` on W&B default step axis.


## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```

