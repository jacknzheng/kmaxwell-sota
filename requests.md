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

Next request number: **REQ-036**.


## REQ-034: K-Maxwell on the fork@2000 batch ladder — 1× → 16×

- status: OPEN
- requested: Jack / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: 365c392d695f95dc9a4fb89095e85a6a7b5d551e (same as REQ-026/027/028/029/033)
- **node budget: operator permission has been granted for ≥10 H100 nodes. The ≤2-node
  ceiling does not apply to REQ-034.** Provision as many boxes as the 6 arms can use
  (one arm per box) and run them in parallel; the 16× arm is the long pole, so start it
  first. Arms are short (750 steps each, ~10 cost units total), so this completes in
  one pass rather than queueing behind REQ-032's long tau2 arms.

**Why.** REQ-026→029 built the momentum-benefit-vs-batch curve for the frozen
bi-Maxwell kernel — `1x −0.01063, 4x −0.00438, 8x −0.00233, 16x ~0` — a clean decay
to zero. REQ-033 measured the **annealed K-Maxwell** kernel, but on a *different
protocol* (fork@1000, 2250 steps, @3250) and a *different range* (0.25×–2×), so its
numbers cannot be laid on the same axis as that curve. Plotting them together
produces a false zigzag; they are not the same measurement.

This request puts K-Maxwell on the **exact bi-Maxwell protocol** — same shared
step-2000 state, same 750-step window, same @2750 readout, same batch ladder — so
the two kernels finally sit on one axis over 1×–16×.

The open question it answers: REQ-033 found K-Maxwell's benefit **grows** with batch
across 0.25×–2× while bi-Maxwell's shrinks. Does K-Maxwell keep its gain at 4×, 8×,
16×, where bi-Maxwell's went to exactly zero — or does it also get absorbed once the
batch is large enough?

### Expected work — 6 arms, 750-step continuations from the shared step-2000 state

Same `eos_shared_base` machinery as REQ-026/028/029 (base val@2000 = 3.44367).

| # | batch | kernel | note |
|---|---|---|---|
| 1 | 1× | `annealed_weights_muon` | |
| 2 | 2× | `annealed_weights_muon` | |
| 3 | 4× | `annealed_weights_muon` | |
| 4 | 8× | `annealed_weights_muon` | |
| 5 | 16× | `annealed_weights_muon` | |
| 6 | 2× | `muon{mu:0.0}` | **the one missing control** |

**Only 2× needs a fresh control.** μ=0 already exists at 1× (3.34586, REQ-026),
4× (3.24333, REQ-026), 8× (3.20561, REQ-028) and 16× (3.17362, REQ-029) — all at
this exact fork and horizon. Difference the new K-Maxwell arms against those stored
values; do **not** re-run them.

### Exact config keys

Copy the REQ-026 fork-continuation template (`make_req026_configs.py`) and change
only the blocks-group optimizer and the batch keys:

| batch | `batch_tokens` | `microbatch_sequences` | `skip_batches` | fineweb chunks |
|---|---|---|---|---|
| 1× | 524288 | 64 | 2000 | 15 |
| 2× | 1048576 | 64 | 1000 | 19 |
| 4× | 2097152 | 64 | 500 | 27 |
| 8× | 4194304 | 64 | 250 | 44 |
| 16× | 8388608 | 64 | 125 | **80** |

All skips are exact integers → every arm resumes at the same ~1.049B-token data
position, exactly as in REQ-026/028/029. `microbatch_sequences` stays **64** at every
batch (larger batches just run more accumulation steps) — this both preserves
per-forward memory and **avoids the torch.compile mbs<64 NaN bug found in REQ-033**;
no eager fallback is needed here.

Budget in **usable batches** (`Σ floor(shard_tokens/batch_tokens)`), the REQ-029
metric, not raw tokens — the chunk counts above already use it. Bootstrap **86**
chunks (REQ-029's verified 16× figure) and every arm fits.

`start_step: 2000, stop_after_step: 2750`, `lr: 0.025, weight_decay: 0.05, mu: 0.95`,
`cool_down_learning_rate cooldown_frac: 0.7`, no `fixed_eta_after`, checkpoint +750
only, no Lanczos — all identical to REQ-026/028/029.

**Kernel** (`annealed_weights_muon`): `switch_step: 2000`, `anneal_steps: 750`.

**Note the deviation and why.** PR #357 switches at step 1000 and anneals to 3250.
Forking at 2000 puts the switch already in the past, and the shared base is plain
Muon with no K-buffers to inherit, so buffers lazy-init at the fork exactly as they
do in the PR at its own switch step. The 58→26 sweep is then **compressed into the
750-step window** so every arm sees the full kernel trajectory rather than a slice of
it. This tests the *kernel*, not the PR's absolute timetable — state that plainly in
the README. (Forking at 1000 instead was considered and rejected: 1000 is not
divisible by 16, so 16× cannot token-align — 62.5 batches — and a 2250-step window at
8×/16× needs 104/211 chunks, exceeding FineWeb10B's 103.)

Decays and weights are the shipped PR #357 values, identical at every batch (no
rescaling in this request — REQ-033 already refuted age-rescaling as a repair):

```yaml
decays: [0.75, 0.822852439855, 0.877930338626, 0.917598547218,
         0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
start_weights: [0.005093975, 0.010187949, 0.015281924, 0.020375898,
                0.025469873, 0.030563847, 0.035657822, 0.857368713]   # mean age 58
end_weights:   [0.032261839, 0.064523678, 0.096785516, 0.129047355,
                0.161309194, 0.193571033, 0.225832871, 0.096668514]   # mean age 26
```

### Gates (hard)

1. Per-config 20-step finite-loss smoke before any full arm.
2. Usable-batch budget assert per config BEFORE launch (REQ-029 precedent — the 16×
   first pass exhausted fineweb 17 steps short on raw-token budgeting).
3. Tests green at the pinned SHA.
4. Confirm the 2× μ=0 control's val@2000 matches the shared base (3.44367) before
   trusting any 2× difference.

### Artifacts

`logs/kmaxwell/req034_kmaxwell_batch_ladder/{README.md,summary.tsv,readout.tsv,
val_trajectories.txt,manifest.tsv,make_req034_configs.py,configs/,logs/}` — the
REQ-026/029 shape.

### Readout

`benefit = final_val(kmaxwell) − final_val(μ0)`, same batch, @2750 — the identical
statistic as the bi-Maxwell curve. Closing table:

```
batch  batch_tokens  benefit(kmax−mu0)  benefit(bimax−mu0)   source of mu0
1x     524288                           −0.01063             REQ-026
2x     1048576                          (none)               THIS REQUEST
4x     2097152                          −0.00438             REQ-026
8x     4194304                          −0.00233             REQ-028
16x    8388608                          ~0.00000             REQ-029
```

The shape is the deliverable, no interpretation needed:

- **K-Maxwell also decays to ~0 by 16×** → both kernels are denoisers; REQ-033's
  "anti-decay" was a 0.25×–2× window effect, and the annealed kernel buys nothing at
  large batch either.
- **K-Maxwell holds its gain at 8×/16× where bi-Maxwell went to zero** → the anneal
  is doing something structurally different from noise-averaging, and it is the
  large-batch-durable kernel. This would be the headline result.
- **K-Maxwell peaks mid-ladder** → there is an optimal batch for the kernel; report
  where.

n=1/cell, seed 0, matching REQ-026/028/029 discovery convention. Noise floor ~2e-4
(REQ-027); read |Δ| < ~5e-4 as noise. If the 16× cell lands inside that band and the
verdict hinges on it, file a follow-up for replicates rather than over-reading n=1.


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


## REQ-035: what sets the per-matrix equilibrium curvature constant C? (seed-replicated discriminator)

- status: OPEN
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `ebf53cd` (the REQ-019/022 serialized-fork-state design, unchanged)
- **node budget: operator permission has been granted for up to 10 H100/H200 nodes.
  The earlier 4-box fleet cap does not apply to REQ-035.** Arms are independent — run
  one arm per box in parallel. If this conflicts with a standing ceiling, the operator
  (Jack) has explicitly enabled this budget; proceed.

**Background — what the committed data already settles (no new compute).** Re-analysis of
the REQ-019/022 per-matrix JSONs (72 Muon matrices, 11-point s ladder, forks 1500 and 2000):

1. Each matrix obeys `log10 lam = log10 C - k*log10(s)`; k = 1.38 +/- 0.45; per-matrix fit
   residual ~0.034 dex. C spreads 0.376 dex (2.4x) across matrices.
2. **The run-to-run noise floor is ~0.100 dex**, from the duplicate arms (s=1.00 vs
   s=1.00dup: rms 0.118 dex; s=0.85 vs s=0.85dup: rms 0.084 dex) — same fork, same s, same
   age, differing only by run randomness. Explainable signal is sqrt(0.376^2 - 0.100^2) =
   0.363 dex, so a perfect model scores ~0.100 dex, not 0. (Note: the median |delta| is only
   0.019-0.030 dex; the distribution is heavy-tailed, so rms is the honest scale.)
3. Leave-one-matrix-out scoring of log C against that floor:

   | model | LOO (dex) | share of explainable signal |
   |---|---:|---:|
   | mean only | 0.382 | 0% |
   | weight norm | 0.384 | 0% |
   | architecture (type + depth) | 0.356 | 12% |
   | gradient block norm g | 0.347 | 16% |
   | Lanczos spectral gap + neg-eig fraction | 0.347 | 16% |
   | **curvature-along-gradient (C_grad)** | **0.201** | **77%** |
   | C_grad + type | 0.188 | 81% |
   | C_grad + type + depth | 0.175 | 84% |

4. **Weight norm has zero cross-sectional power** (slope +0.01, R^2 0.000); normalizing to
   lam*||W||^2 makes the spread 48% worse. Norm is a within-matrix channel only (REQ-023's
   raw -1.29 vs gauge -0.57), not a between-matrix one.
5. **Depth is real but type-specific and sign-flipping**: corr(logC, depth) = -0.71 for
   attn.v, +0.49 for mlp.fc, ~0 for the other four types; pooled it cancels to R^2 = 0.000.
   The previously reported "R^2 up to 0.83" reproduces exactly (attn.v, blocks 2-11, 4 params
   on 10 points) and is not chance (shuffle p = 0.002), but extrapolates badly (in-sample
   0.064 dex -> held-out blocks 0-1: 0.221 dex). Giving each type its own depth slope makes
   out-of-sample error WORSE than the mean (0.398 vs 0.382).
6. **The anisotropy A = lam_top / lam_grad is learning-rate-independent** (slope +0.07 +/-
   0.27, vs +1.38 for lam itself) and stable across states (corr +0.936, median shift 0.042
   dex). But `log C = log C_grad + log A` is an algebraic identity, not an explanation: it
   scores R^2 = 1.000 tautologically, and as a cross-state predictor (0.110 dex) it does no
   better than simply re-measuring C (0.099 dex). Recorded so it is not rediscovered.
7. **Instrument note:** the committed `top_eigenvalue` is the RAW Ritz value — the geometric
   tail correction is NOT applied (correction factor 1.000 at median and p90); `residual_tail`
   is diagnostic only. Applying it ourselves shifts C by 0.025 dex median and changes no
   ranking above. Gate attrition is strongly type-dependent (attn.q 10%, attn.v 66%) and is
   explained by the spectral gap (corr(log gap, log tail) = -0.81 across types).

**The open question.** C_grad captures 77% of the explainable signal alone; no architectural
covariate exceeds 16%. But C_grad is itself a measured curvature, so this relocates the
question rather than answering it. Nothing reaches the 0.100 dex floor. Three hypotheses
remain live and are distinguished by the arms below.

**Common settings for all arms:** 8-iteration Lanczos, fixed 131072-token batch, identical
measurement code and settings to REQ-019; curvature at the last 3 checkpoints of each arm.

### Arm A — seed replication (n=4). The load-bearing arm. 4 boxes.

Four independent seeds (0,1,2,3), each trained from scratch to step 1500, each forked into
the ladder s in {0.60, 1.00, 1.70}. **Registered question: is C a property of the
architecture, or of the individual trained network?**

- median |delta log C| across seed pairs **<= 0.10 dex** (the noise floor) => C is
  seed-independent; architecture determines it; the covariate hunt is justified.
- **>= 0.20 dex** => C is a learned per-network property; every static covariate model is
  then bounded away from the floor by construction, and the program pivots to state variables.
- in between => report the seed-reproducible fraction; that becomes the true ceiling for any
  covariate model, replacing 0.363 dex of "explainable" signal.

Also report corr(C_seed_i, C_seed_j), and separately whether the **type ordering** (attn.v
highest, attn.proj lowest) reproduces across seeds even if the levels do not.
**This arm is worth running even if every other arm is dropped.**

### Arm B — depth-sweep discriminator. 3 boxes.

Same recipe at 6, 12, and 24 blocks. Distinguishes absolute depth (block 6 of the 24-block
model matches block 6 of the 12-block model) from relative depth d/D (matches block 12).
Registered on attn.v and mlp.fc — the only two types with real depth structure. If neither
matches, depth is a proxy for a local quantity and the architecture hypothesis is dead.

### Arm C — shape / update-geometry sweep. 2 boxes.

The Muon shape factor sqrt(max(1, rows/cols)) currently takes only two values (2.0 for
mlp.fc, 1.0 for everything else), so update geometry has almost no natural variance to
explain anything with — despite REQ-023 showing a strong causal exponent of -1.3. Vary
head_dim and mlp_ratio so the factor spans {0.5, 1, 2, 4}. Registered: does C shift by the
-1.3 exponent REQ-023 measured, or does the shape factor act only through the effective LR
and not through C?

### Arm D — norm-pinning control. 1 box.

Project each Muon matrix back to a held Frobenius norm after every optimizer step; half the
matrices pinned +25%, half -25%, balanced by type, untouched matrices as internal control.
Registered band from REQ-023's gauge slope (-0.57): a held +25% norm change moves C by
**-0.055 dex** if the norm channel is causal at equilibrium, **0** if norm is purely a
transient channel. This is the one remaining test of the weight-norm hypothesis after its
cross-sectional death.

### Ordering

Arm A first, and alone if capacity is tight — B, C and D are only interpretable once A says
whether C is seed-stable. With 10 boxes available, run A (4) + B (3) + C (2) + D (1)
concurrently.

### Success criteria

- Arm A reports cross-seed median |delta log C| with an explicit verdict against the two
  registered bands and comparison to the 0.100 dex floor.
- Every arm commits `per_matrix_curvature.json` with the existing field set — critically
  including `curvature_along_gradient`, `curvature_along_polar`, and `gradient_block_norm`,
  which carry the entire result above.
- A `summary.tsv` reporting per matrix: C, k, C_grad, and A.
- Shared-state gate as in REQ-019 (identical sha256, zero abs-diff, LR = base x mult).
- **Do not** apply the geometric-tail correction silently — commit raw Ritz values plus
  `residual_tail`, as REQ-019 does, so the correction stays reversible.

### Artifacts

`logs/kmaxwell/req035_C_mechanism/<arm>/`

## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```

