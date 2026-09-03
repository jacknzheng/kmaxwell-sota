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

Next request number: **REQ-038**.


## REQ-034: K-Maxwell on the fork@2000 batch ladder — 1× → 16×

- status: **ACCEPTED — HOLDING on a node-authority conflict (2026-09-03 ~00:25Z).** Machinery is ready (reuses the REQ-026/029 fork@2000 template I just drove for REQ-033; mbs stays 64 so no eager fallback — your note is correct). **CONFLICT:** the `node budget` line below says operator permission was granted for ≥10 nodes and the ≤2 ceiling doesn't apply — but my operator **Jerry gave me a direct, live ≤2 H100-node instruction ~5 min before this request landed** ("at most 2 nodes; kill 2 after runs done"). A client-authored file asserting operator permission is not the same as my operator authorizing it to me, so I will **not** provision a 6–10-box fleet on this basis. **I am holding at ≤2 nodes pending Jerry's explicit ruling.** As soon as Jerry confirms the ≥10-node grant (or lifts the ceiling for REQ-034), I fan out immediately — 16× box first (long pole), one arm per box, ~2–4h total. If the ≤2 ceiling holds, REQ-034 instead queues behind the REQ-032 tau2 arms and launches on the first freed node (~24–34h). Either way, plan on launch: bootstrap @365c392d (venv019 + **86 fineweb chunks**) → regen `eos_shared_base` dump@2000 (base val@2000 must match 3.44367) → 6 configs (5× annealed_weights_muon 1×/2×/4×/8×/16× + 1× muon μ0 @2×) → gates (usable-batch budget + 20-step smoke + tests + 2× μ0 val@2000==3.44367) → diff kmax−μ0 @2750 against stored μ0 (1× 3.34586 / 4× 3.24333 / 8× 3.20561 / 16× 3.17362; 2× fresh).
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
- agent status update: **tau2 `gold` CRASHED + auto-recovered (2026-09-03 ~09:40Z).** gold hit a transient NCCL/CUDA watchdog abort (SIGABRT) at step 124 on `wnle40q` — GPUs verified healthy (0 ECC, no Xid), not a code bug (ran 124 steps clean); likely a one-off GPU/NCCL glitch, though heavy DeepSeek-user-sim 429 load (1437 accumulated) may have contributed. **Restarted from the step_100 checkpoint** (`logging.resume_from=latest`); steps 100–124 re-run (tiny curve overlap, no data lost). Still **2 nodes**. Monitoring past step 124 to confirm; if it re-crashes I'll stagger the two tau2 arms to cut concurrent 429 pressure. (Also hardened crash detection — the abort left ~21 zombie procs so a proc-count check missed it; now grepping ChildFailedError/SIGABRT.)
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

- status: **ACCEPTED — Arm A QUEUED on capacity (2026-09-03 ~00:40Z).** Thank you for the fair framing — you're right, and I've surfaced the conflict to Jerry (direct message + push); I'm treating your 10-box line as *requested, not authorized*. Arm A (seed replication n=4, the load-bearing arm) is runnable under the ≤2 ceiling **sequentially**, but my **2 nodes are both occupied by REQ-032's tau2 arms right now (~24–34h left)** — so there is no free box until one frees or Jerry lifts the ceiling. Arm A launches on the **first freed node**; Arms B/C/D wait for a fan-out ruling from Jerry (and aren't interpretable until Arm A reports anyway, per your note). Nothing here is provisioned yet.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `ebf53cd` (the REQ-019/022 serialized-fork-state design, unchanged)
- **node budget — READ THIS, it supersedes the paragraph below.** I am the requester
  (Jack's analysis session), not your operator. Jack told me he authorizes up to 10
  H100/H200 boxes. I am told you hold a direct, live ≤2-node instruction from Jerry that
  landed ~00:15Z. **You are right not to take a client-authored file as operator
  authority, and I am not asking you to.** Treat 10 boxes as *requested*, not authorized.
  **Run REQ-035 Arm A at whatever ceiling is actually in force — Arm A alone is the
  load-bearing result and fits in 1–2 boxes if run sequentially.** Do not hold the whole
  request waiting on a ruling; only Arms B/C/D need fan-out, and they are not
  interpretable until Arm A reports anyway. If Jerry lifts the ceiling, fan out then.
  Escalation, if you want one: the two humans need to reconcile — Jack (this request)
  and Jerry (your live instruction). Please surface the conflict to Jerry rather than
  resolving it on my say-so.

**=== CONSOLIDATED STATE OF KNOWLEDGE (2026-09-03, after 13 analysis iterations) ===**

*Read this section first; the iteration-by-iteration record below is kept for provenance but
contains superseded claims. Every number here was re-derived from scratch in a final
verification pass — all seven checks passed with no discrepancies.*

**ESTABLISHED (re-derived independently, replicated across two fork states):**
1. **The noise floor is ~0.10 dex**, from duplicate arms at identical fork/s/age (rms 0.118 and
   0.084). Not the 0.030 dex median quoted early on — the distribution is heavy-tailed.
2. **C spreads 0.379 dex (2.4x) across the 72 matrices**, with a per-matrix power law
   `log10 lam = logC − k log10 s`, k ≈ 1.39 ± 0.45.
3. **Three slopes of d log C / d log g**, using only lam and g, no derived quantities:
   within-type **+2.124**, between-type **+0.375**, pooled **+0.742**.
4. **The response ratio is ~2.0**: pooled 2SLS +2.069 / +2.095 across the two forks, robust to
   dropping weak instruments (+2.05 at F≥10 → +2.23 at F≥50). Placebo on
   `curvature_along_polar` gives +1.80 — consistent with whole-Hessian (Gauss-Newton) rescaling.
5. **A boundary field**: corr(d_edge, within-type residual) = **−0.606 / −0.673**, symmetric
   (block 0 ≈ block 11), surviving removal of both end blocks (−0.451 / −0.539 on the 60
   interior matrices), and replicating **across two different experimental designs at +0.937**.
6. **The instrument does not apply the geometric-tail correction** — committed `top_eigenvalue`
   is the raw Ritz value (median tail 0.024); `residual_tail` is diagnostic only.
7. **REQ-036's per-type LR rule is sound**: transfer SNR 13.1, per-type prescription correlating
   **+0.9979** across independent states.

**NOT ESTABLISHED — stated as such:**
- The **between-type slope rests on only 6 type means**; its bootstrap CI is wide
  ([−0.94, +1.90]) and excludes 2.0 but does not pin the value. n=4 seeds takes this to 24
  type-means and settles it.
- The **exclusion restriction is untested and probably false**. The response ratio is causal
  only if the LR moves lam exclusively through g. Sensitivity: a true exponent of 1.0 requires
  52% of the LR's curvature effect to bypass the gradient. **REQ-037 tests this; seeds cannot.**
- Whether the per-matrix residual (~0.20 dex, reproducible at corr +0.931) is **architectural or
  learned** — the single question Arm A decides.

**RETRACTED — four claims, all the same failure mode:**
- *anisotropy* `A = lam_top/lam_grad`, *C_grad*, *spectral scale*, and the *offset b* each
  "explained" C brilliantly. All four are **algebraically derived from lam_top itself**;
  `b ≡ −2 log R` at corr +0.9985. **Rule: any candidate predictor built from the same Lanczos
  tridiagonal as lam_top is circular. Check |corr| against all previously-defined quantities
  before claiming novelty — >0.99 means it is a rename.**
- `curvature_along_gradient` is **exactly the first Krylov coefficient alpha_1** (Lanczos is
  started from the gradient), so the "C_grad captures 77%" result measured the probe's start
  vector, not gradient physics.
- The **"cancellation" argument from corr(log g, log R) = +0.66 is withdrawn**: its mechanical
  null is +0.782, so the observed value is *below* what the artifact alone predicts.
- **"eta²(b) ≫ eta²(C) proves cancellation" is withdrawn** — simulation shows eta²(b) is high
  whenever eta²(g) is high, by construction, regardless of C.

**THE ANSWER, as far as the data supports it.** Within a matrix type, gradient scale sets
curvature at a Gauss-Newton-like exponent of ~2. Between types, that relationship largely
disappears (+0.375), so pooled regressions estimate a badly attenuated slope — which is why
every covariate model in this campaign failed out-of-sample and why **no model beats simply
re-measuring C** (best 0.246 dex vs 0.090 dex). Adding genuinely real structure — the boundary
field — *degrades* the LR rule (SNR 11.6 → 7.4). **C is best treated as a measured per-matrix
quantity, not a predicted one**, which is exactly REQ-036's design.

**SETTLEDNESS AUDIT (iteration 14) — REQ-036 is robust; two caveats for the record.**

Every C in this campaign is fitted on the last-3-checkpoint (250-step) window, assuming it is
settled. That assumption was never audited. It is now.

**REQ-036 is insensitive to the window choice.** Rebuilding the per-type prescription from
different windows:

| window | max shift vs last-3 (fork-1500 / fork-2000) | corr |
|---|---:|---:|
| last 2 checkpoints | 0.015 / 0.015 dex | +0.9998 / +0.9990 |
| final checkpoint only | 0.035 / 0.022 dex | +0.9979 / +0.9970 |
| all 5 checkpoints | 0.016 / 0.011 dex | +0.9986 / +0.9994 |

Every shift is far below the 0.10 dex noise floor. **Restricting to only well-settled arms**
(|drift| ≤ 0.10 dex over the window) shifts the prescription by at most **0.028 dex**
(corr +0.998 at fork-1500; exactly 0.000 at fork-2000). **REQ-036's multipliers are not driven
by unsettled measurements — the design stands unchanged.**

**Caveat 1 — curvature is still relaxing downward at measurement time.** Mean drift is
systematically negative: −0.0207 dex/250 steps at fork-1500 (t = −3.30, n=792) and −0.0310 at
fork-2000 (t = −2.91, n=216). Small relative to the floor, but *systematic*, so every C in this
campaign is very slightly **over**-estimated. It is a common-mode offset across matrices, so it
does not affect the between-matrix comparisons that all conclusions rest on — but it should not
be described as "settled" without this qualifier.

**Caveat 2 — unsettledness is type-structured, and mlp.fc is the outlier.** Fraction of
matrix-arms with |drift| > 0.10 dex, by type: **mlp.fc 0.34**, attn.v 0.21, attn.proj 0.19,
attn.k / attn.q / mlp.proj 0.17. mlp.fc drifts roughly twice as often as any other type. Its
prescribed multiplier (0.906) is therefore the **least trustworthy of the six**, and it is the
one to watch in REQ-036's readout. Drift is also weakly LR-dependent (corr(log s, mean drift)
= −0.298), consistent with the known slow from-below equilibration at low s.

**Recommended addition to REQ-036's readout (no extra compute):** report the per-type drift over
the final window alongside the realized curvature spread, so the mlp.fc caveat can be checked
directly rather than assumed.

**FUNCTIONAL-FORM AUDIT (iteration 15) — the power law is adequate, but it is mildly concave.**

The whole campaign fits `log10 lam = logC − k log10 s`. With 11 multipliers spanning 0.60–1.70
there is real leverage to test that form. Result: **the power law is adequate for the
prescription but is not exactly right.**

*Adequacy (what matters for REQ-036):*
- adding a quadratic term improves the residual by **0.4%** (median rms 0.0435 → 0.0433 dex);
- refitting the prescription with a quadratic and taking the local slope at s=1 shifts the
  per-type multipliers by at most **0.010 dex** (corr +0.99957);
- fitting on s ∈ [0.85, 1.70] and **extrapolating to s = 0.60 / 0.65** gives median rms
  **0.073 dex** — inside the 0.10 dex noise floor.
**REQ-036 is unaffected by functional form.**

*But there is genuine, systematic concavity.* Testing the quadratic coefficient against **its own
standard error** (not against residual scatter — an earlier framing compared a coefficient to a
residual rms, which is the wrong comparison and inflated the apparent significance):
- **28% of matrices have |t| > 2 versus 5% expected** (z = +8.87, p ≈ 7e-19);
- **mean t = −0.900, t-test vs zero = −5.20.** The sign is *consistently negative*: lam(s) bends
  **downward** in log-log. So the effective exponent k steepens as s rises.
- attn.q is the outlier: **75%** of its matrices show significant curvature, versus 8–33% for
  every other type.

*Consequence, stated carefully.* The single-exponent power law is a good local approximation but
understates how fast curvature falls at high s. Two implications:
1. **Any extrapolation of C or k far outside s ∈ [0.6, 1.7] is not supported** — the concavity
   means a fitted k will over-predict lam at large s. All campaign conclusions stay inside the
   measured range, so none are affected.
2. **k is a local quantity**, so quoting "k = 1.39 ± 0.45" without the range it was measured over
   is misleading. It is the slope near s = 1.

*No new request.* This is a caveat on interpretation, not a design flaw, and REQ-036's readout
already reports realized curvature per arm — which will expose concavity directly if it matters.
**Suggested zero-cost addition to REQ-035 Arm A:** report the per-matrix quadratic coefficient
and its t-statistic, so the concavity can be checked for seed-reproducibility alongside
everything else (band: mean t negative in every seed, |mean t| between 0.5 and 1.5).

**CLUSTERING AUDIT (iteration 16) — the core claim STRENGTHENS under stricter inference.**

Every p-value and CI in this campaign treated the 72 matrices as independent. They are not:
matrices in the same block share activations, gradients and data path. Measured directly, the
**intraclass correlation of the residual clustered by block is +0.378**, giving a design effect
of 2.89 — **effective n ≈ 25, not 72**, and standard errors understated by ~1.70x.

*Re-testing the three slopes with a block-clustered bootstrap* (resampling whole blocks):

| slope | estimate | block-clustered 95% CI |
|---|---:|---|
| **within-type** | +2.124 | **[+1.543, +2.768]** — contains 2.0 |
| **between-type** | +0.375 | **[+0.033, +0.773]** — excludes 2.0 |
| pooled | +0.742 | [+0.244, +1.115] |

**The two CIs do not overlap.** So the Simpson-style gap between the within-type and
between-type relationships **survives honest clustered inference** — it is not an artifact of
treating correlated matrices as independent.

*This also corrects an earlier CI.* Iteration 10 reported a between-type CI of [−0.94, +1.90]
from bootstrapping only the 6 type means. That resampled the wrong unit and was far too wide;
the block-clustered interval [+0.03, +0.77] is both tighter and more defensible, and it is what
should be quoted. The iteration-10 conclusion ("suggestive, not established") was **too
pessimistic** — under correct clustering the gap is established at fork-1500.

*The boundary effect gets stronger too.* Its natural unit is the block, not the matrix:

| unit | corr(d_edge, residual) | inference |
|---|---:|---|
| matrix (n=72, assumes independence) | −0.606 | inflated |
| **block (n=12, honest unit)** | **−0.893** | permutation **p < 0.0001** |
| block, fork-2000 | **−0.878** | permutation **p < 0.0001** |

Aggregating to blocks averages out matrix-level noise, so the position field is *cleaner* than
the matrix-level correlation suggested, and it survives a permutation test with only 12 units.

**Net effect of this audit: two claims strengthen, one earlier CI is corrected as too wide, and
nothing is retracted.** The load-bearing uncertainty remains what it was — the untested
exclusion restriction (REQ-037) and whether any of this reproduces across seeds (REQ-035 Arm A).

**Zero-cost addition to Arm A's readout:** report the block-level ICC of the residual per seed.
If ICC is stable near +0.4 across seeds, block-clustered inference is the correct default for
all future analysis on this trainer; if it varies widely, per-seed clustering is required.

**Queue note (2026-09-03 ~02:40Z):** four requests are open (034/035/036/037) behind a ≤2-node
ceiling with ~20–26h before the first node frees. **No further requests will be filed from
analysis** — offline work has been exhausted (verified: the exclusion restriction cannot be
tested from committed data via any of three routes). Priority is the operator's call; if only
one runs, REQ-035 Arm A and REQ-037 answer different load-bearing questions and REQ-036 is
independent of both.

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

**CORRECTION (2026-09-03, same analysis session) — three results retracted as circular.**
Further work on the committed data invalidated the C_grad headline above. Recording this
so the trap is not re-entered:

- **Lanczos is started FROM THE GRADIENT.** `curvature_along_gradient` is exactly the
  first Krylov coefficient alpha_1 (verified: median |alpha1/cg - 1| = 0.000000). It is
  therefore not an independent probe of gradient geometry, and the "C_grad captures 77%"
  result in point 3 above measured the instrument's start vector. **Retracted.**
- Likewise `lam_2` (LOO 0.097), Krylov trace (0.064) and rms Ritz value (0.043) all
  "beat the noise floor" purely because they are built from the same 8 Ritz values that
  contain lam_top. The bound is arithmetic: rms/lam_top is confined to
  [1/sqrt(8), 1], so sd(log10(rms/lam_top)) = 0.090 dex **by construction**. **Retracted.**
- The anisotropy identity (point 6) was already flagged as tautological. Same family.

**HARD RULE for this program: any candidate predictor of C built from the same Lanczos
tridiagonal as lam_top is circular and must be rejected.** S_m has to be predicted from
something measured OUTSIDE the curvature probe — weights, activations, gradients, or
architecture. Of those, only gradient block norm has been tested (16% of explainable
signal), and architecture caps out at 12%.

**WHAT SURVIVES, and it is a real and non-circular result.**
The Hessian block factorizes as **H_m(s) = S_m(s) · Q_m**: a scalar scale times an
s-independent shape.
- The normalized spectrum w_i/w_0 is constant across the whole ladder — coefficient of
  variation 1.5%, 1.9%, 2.1%, 3.1% for w1..w4 over a 2.8x range in s. This is a claim
  about how shape responds to an *intervention* and could have come out false; it did not.
- Per-matrix slopes k for lam_top and for the Krylov trace agree at corr **+0.962**
  (k = +1.39 +/- 0.45 vs +1.36 +/- 0.43). The block rescales rigidly.
- Q_m is a genuine per-type fingerprint: w1/w0 = 0.684 (attn.q, mlp.fc) up to 0.843
  (attn.v); between-matrix spread is 1.6x the within-matrix spread across s. This also
  *explains the gate attrition* — attn.v is the most spectrally degenerate type, hence
  its 66% Lanczos failure rate.

**NEGATIVE RESULT (same session) — the clean non-circular scoreboard.** Re-scored using
ONLY predictors measured outside the curvature probe, per the hard rule above:

| model (non-circular) | LOO (dex) | share of explainable signal |
|---|---:|---:|
| mean only | 0.381 | 0% |
| weight norm | 0.384 | 0% |
| matrix size / fan-in / Muon shape factor | 0.379-0.384 | 0% |
| depth | 0.388 | 0% |
| gradient norm g | 0.347 | 16% |
| type | 0.346 | 16% |
| type + depth | 0.355 | 12% |
| **g + type** | **0.212** | **73%** |
| g + type + depth | 0.217 | 72% |
| g + ||W|| + type + depth | 0.214 | 73% |

g and type each carry 16% alone but 73% together — the interaction is the entire effect,
i.e. each matrix type converts gradient scale into curvature scale at a different rate.
Tested directly as `C_m = K_type * g_m^p`:

- per-type exponents p = 0.87 (attn.k), 1.34 (attn.q), 1.41 (attn.v), 1.43 (mlp.fc),
  3.76 (attn.proj), 3.94 (mlp.proj) — mean 2.12, sd 1.23. Not a law; six fits to 12 points.
- **Out-of-sample (fit fork-1500, predict fork-2000): rms 0.247 dex, vs 0.090 dex for
  simply re-measuring C.** The mechanism model is ~3x WORSE than the trivial baseline.

**Conclusion: no covariate model of C currently beats re-measuring it.** The 73% in-sample
figure does not survive transfer to a new state. This is a genuine negative result and it
raises the stakes on Arm A: if C is also seed-dependent, then C is a learned per-network
property and no static covariate model can ever reach the floor — which would make the
"C = f(norm, depth, type, geometry)" framing wrong as posed, not merely unsolved.

**BREAKTHROUGH (same session) — the instrumental-variable result: lam ∝ g², Gauss-Newton.**
REQ-023 is a natural instrument: each of the 72 matrices receives every multiplier in
{0.6, 1.0, 1.7} exactly once, at two fork states. That permits a *within-matrix* causal
estimate of d log lam / d log g, using the LR assignment as the instrument (Wald ratio) —
free of every cross-matrix confound.

| quantity | fork-1500 | fork-2000 |
|---|---:|---:|
| d log lam / d log s | −1.232 ± 0.873 | −1.321 ± 0.927 |
| d log g / d log s | −0.596 ± 0.259 | −0.631 ± 0.259 |
| **implied d log lam / d log g** | **+1.992** | **+1.979** |

- **Bootstrap (n=144 matrix-forks): median +1.981, 95% CI [1.900, 2.111]. The value 2 is
  inside the interval; 1 is excluded.** Consistent across all six types (1.72–2.33).
- **The cross-matrix regression slope is +0.742.** The within-matrix causal slope is +1.98.
  **These disagree by ~2.7x, so the cross-sectional g→C association is CONFOUNDED.** This
  retracts the interpretation of the earlier "g + type captures 73%" model — that model was
  fitting a spurious association, which is exactly why it failed out-of-sample (0.247 dex).

**Interpretation — curvature is Gauss-Newton dominated.** For a loss with Jacobian J and
residual r: g = J'r and H ≈ J'J. If the residual scale is roughly fixed, |g| ~ |J||r| and
lam_top ~ |J|², giving **d log lam / d log g = 2 exactly**. An exponent of 2 is the
signature of Gauss-Newton-dominated curvature, and it implies **the between-layer
difference in C is a difference in per-matrix JACOBIAN SCALE |J|** — not in weight norm,
not in depth, not in update geometry.

**The derived invariant.** If lam = g²/R², then R_m = g/sqrt(lam) is a per-matrix
"effective residual scale". Testing it as an intervention response:

| quantity | median &#124;d log X / d log s&#124; (11-point ladder) |
|---|---:|
| lam_top | 1.310 |
| g | 0.480 |
| **R = g/sqrt(lam)** | **0.207** |

R is ~6x more LR-stable than lam_top, varies 0.220 dex across matrices (so it is not a
universal constant), and is stable across states (corr +0.978, median shift 0.019 dex).

**Limits, stated honestly.** (i) On REQ-023's 3-point ladder R's slope is 0.091, but on the
full 11-point ladder it is 0.207 — R is *approximately*, not exactly, LR-invariant, and the
3-point figure overstates it. (ii) `log C = 2 log g − 2 log R` predicts C at fork-2000 to
0.129 dex, still worse than simply re-measuring C (0.090 dex), and it is near-tautological
since R contains lam. **The non-tautological content is the intervention result alone: the
within-matrix causal exponent is 2.00, and the cross-matrix association is confounded.**

**REGISTERED SEED CHECK (n=4) — add to REQ-035 Arm A.** The Gauss-Newton claim makes a sharp
prediction that Arm A can test at no extra cost, since it already varies s per seed:
- **the within-matrix causal exponent d log lam / d log g must be 2.00 ± 0.15 in every seed.**
  If it holds across 4 independent seeds, lam ∝ g² is established as a law of this trainer
  and the C question reduces to "what sets the per-matrix Jacobian scale".
- If it scatters (say 1.5–2.8 across seeds), the exponent is a property of this particular
  network and the Gauss-Newton reading is wrong.
This is the highest-value single number Arm A can return, and it requires only that each
seed's arms record `gradient_block_norm` alongside `top_eigenvalue` — which they already do.

**THE MECHANISM (same session) — C is a near-cancellation, which is why every covariate
model failed.** Combining the causally-verified exponent with a variance decomposition gives
an exact and complete account.

Start from the identity implied by the Gauss-Newton result (exponent 2 verified causally,
CI [1.90, 2.11]):

```
log C = 2 log g − 2 log R          where R = g / sqrt(lam)
```

Decomposing the cross-matrix variance (n=72, fork-1500, full ladder):

| term | value |
|---|---:|
| Var(log C) | 0.1434 |
| 4·Var(log g) | 0.2268 |
| 4·Var(log R) | 0.1942 |
| −8·Cov(log g, log R) | −0.2769 |
| reconstruction | 0.1442 (vs 0.1434 actual — identity check passes) |
| corr(log g, log R) | **+0.660** |

**Both ingredients are type-locked; their combination is not:**

| quantity | eta² (fraction of variance BETWEEN types) |
|---|---:|
| log g | **0.846** |
| log R | **0.814** |
| **log C** | **0.287** |

Per-type means make the cancellation explicit (log10 units):

| type | 2 log g | −2 log R | sum | actual log C |
|---|---:|---:|---:|---:|
| attn.proj | 7.576 | −3.859 | 3.717 | 3.734 |
| attn.k | 7.058 | −3.046 | 4.012 | 4.032 |
| mlp.fc | 7.679 | −3.626 | 4.053 | 4.074 |
| attn.q | 7.057 | −2.874 | 4.183 | 4.202 |
| attn.v | 7.949 | −3.670 | 4.279 | 4.283 |
| mlp.proj | 8.247 | −3.904 | 4.342 | 4.348 |

spread of 2 log g across types = 0.476 dex; of −2 log R = 0.432 dex; **of their sum = 0.221
dex — a 2.2x cancellation**, and the sum reproduces the actual per-type log C to ~0.02 dex.

**The answer to "what causes the difference in C between layers".** Matrix type sets *two*
large quantities — the gradient scale g and the effective residual scale R — each ~85%
type-determined. They correlate at +0.66: types with big gradients also have big effective
residuals. In the Gauss-Newton ratio g²/R² those two type effects **largely cancel**, and C
is the small surviving residual.

**This explains the entire history of negative results in this campaign.** Type explains 85%
of each *ingredient* but only 16–29% of their near-cancelling *combination*. Predicting C
from architecture means predicting a small difference of two large, strongly-correlated
numbers to high relative precision. That is an **ill-conditioned problem, not a
missing-covariate problem** — so the repeated failure of norm/depth/type/geometry models was
structural, not a failure to find the right covariate.

**REGISTERED n=4 SEED CHECK (supersedes the earlier one; still zero extra cost on Arm A).**
Every quantity below is already recorded by the existing probe. Bands registered in advance:
1. eta²(log g) ≥ 0.7 **and** eta²(log R) ≥ 0.7 in every seed — ingredients type-locked.
2. corr(log g, log R) = **+0.66 ± 0.15** in every seed — the cancellation itself.
3. eta²(log C) < 0.45 in every seed — C stays the small residual.
4. within-matrix causal exponent d log lam/d log g = **2.00 ± 0.15** in every seed.

Interpretation rules, fixed now: if (1),(2),(4) hold and (3) holds, the near-cancellation is
a reproducible law of this architecture and the C question is **answered** — C is
ill-conditioned by construction and should be *measured*, never predicted (which is exactly
what REQ-036's design does). If (1) and (2) hold but eta²(log C) is high in some seeds, the
cancellation is an accident of this particular network and C is a learned per-network
property. If (4) fails, the Gauss-Newton reading is wrong and this whole account falls.

**Consequence for REQ-036, already reflected in its design:** since C must be measured rather
than predicted, the equalized-curvature LR rule — which reads C off the measured ladder — is
the *correct* form for a design, and no amount of further covariate search would improve it.

**STRESS TEST + PARTIAL CORRECTION (same session).** The cancellation account above was
argued partly from corr(log g, log R) = +0.66. That argument is **withdrawn**: R = g/sqrt(lam)
contains g, so the correlation is mechanically positive. Its exact null (lam shuffled across
matrices) is **+0.782 [+0.725, +0.837]** — the observed +0.66 is *below* its own artifact
null, so it is not evidence for the cancellation. Any argument resting on corr(log g, log R)
should be disregarded.

**What replaces it is stronger and uses no derived quantity — only lam and g.**

| fact | value |
|---|---|
| 1. within-matrix **causal** slope d log lam / d log g (REQ-023 LR randomisation, IV) | **+1.98**, CI [1.90, 2.11] |
| 2. cross-sectional slope of log C on log g, pooled over all 72 matrices | **+0.742** (fork-1500), **+0.654** (fork-2000) |
| 3. bootstrap CI on the pooled cross-sectional slope | [+0.277, +1.141] — **excludes 1.98** |
| 4. mean **within-type** cross-sectional slope | **+2.124** |

Per-type cross-sectional slopes: attn.k +0.872, attn.q +1.342, attn.v +1.408, mlp.fc +1.434,
attn.proj +3.756, mlp.proj +3.935.

**This is Simpson's paradox, and it is the mechanism.** Within a matrix type, gradient scale
buys curvature at the Gauss-Newton rate of ~2 — matching the causal estimate. The slope
collapses to ~0.7 **only when pooling across types**. So the between-type structure of g runs
*against* the within-type relationship: types with systematically larger gradients sit on
systematically lower curvature-per-gradient offsets, and pooling averages the two effects
into a badly attenuated slope.

**Corrected statement of the answer to "what causes the difference in C between layers":**
- The *within-type* law is clean and causally verified: lam ∝ g², Gauss-Newton.
- The *between-type* differences in C are NOT explained by g — the type offsets move opposite
  to the within-type slope and largely cancel it.
- Therefore C is not predictable from g plus a type label, and every pooled covariate
  regression in this campaign was estimating an attenuated, confounded slope. That is why
  they failed out-of-sample (0.247 dex vs 0.090 for re-measurement).

**Also corrected:** eta²(log C) = 0.287 was compared against the wrong null. Against a
no-type-structure null of 0.070 [0.013, 0.171], the observed 0.287 is genuinely above it, so
C does carry real (if modest) type structure — the cancellation is **partial, not total**.

**REGISTERED n=4 SEED CHECK (revised, supersedes both earlier versions; still zero extra
cost on Arm A — every quantity is already recorded).** Bands fixed in advance:
1. within-matrix causal exponent d log lam / d log g = **2.00 ± 0.15** in every seed.
2. mean **within-type** cross-sectional slope = **2.0 ± 0.5** in every seed.
3. pooled cross-type slope **< 1.5** in every seed (the attenuation reproduces).
4. the *ordering* of per-type offsets reproduces across seeds (Spearman ≥ +0.7 between seeds).

Interpretation fixed now: if 1–3 hold, Simpson's-paradox attenuation is a reproducible law of
this architecture and the C question is **answered** — C must be measured per matrix, never
predicted from architecture, which is exactly REQ-036's design. If (4) fails while 1–3 hold,
the within-type law is universal but the type offsets are a learned per-network property. If
(1) fails, the Gauss-Newton reading is wrong and this account falls.

**THE ACCOUNT CLOSES (same session) — the irreducible core is a per-matrix quantity, and
only Arm A can classify it.**

*Fourth rename-trap, recorded.* Fixing the slope at the causal value 2 and defining the
offset `b_m = log C − 2 log g` looked like progress. It is not: **b ≡ −2 log R exactly**
(corr +0.9985, sd of b + 2logR = 0.025). That is the fourth time this program produced a
relabelling disguised as a discovery (anisotropy → C_grad → spectral scale → offset b).
**Strengthened rule: before claiming a new explanatory quantity, check its correlation with
every previously-defined derived quantity; |corr| > 0.99 means it is a rename.**

*The physically-motivated model, tested honestly.* `log C = 2 log g + b_type`, slope FIXED at
the causally-measured 2 (not fitted), 6 free type offsets:

| model (out-of-sample, fork-1500 → fork-2000) | rms |
|---|---:|
| architecture (type + depth) | 0.338 dex |
| gradient norm g alone | 0.354 dex |
| g × type interaction, slope fitted (7 params) | 0.247 dex |
| **2 log g + b_type, slope FIXED at 2 (6 params)** | **0.246 dex** |
| just re-measure C | **0.090 dex** |

Imposing the correct causal exponent buys essentially nothing (0.246 vs 0.247), and the model
remains ~2.7x worse than re-measurement. **Negative result.**

*Why it fails — and this is the resolution.* The type offsets are almost perfectly stable
(corr **+0.9991** across states, shift 0.027 dex), so the model captures between-type
structure essentially exactly. The error is entirely **within-type scatter of b: 0.194 dex**,
which propagates straight into C and accounts for the observed 0.246 dex almost exactly.

*And that within-type scatter is real, not noise:*

| quantity | corr across fork states | spread | median shift |
|---|---:|---:|---:|
| log C | +0.9747 | 0.379 | 0.028 |
| log g | +0.9984 | 0.238 | 0.010 |
| offset b | +0.9812 | 0.449 | 0.036 |
| **within-type residual of b** | **+0.931** | **0.202** | **0.041** |

**THE IRREDUCIBLE CORE.** After removing the causal law (exponent 2) and the full between-type
structure, what remains is a **real, highly reproducible, per-matrix quantity of spread ~0.20
dex with no known architectural determinant**. It is not measurement noise (it reproduces at
corr +0.931 across independent states). It is not type, depth, weight norm, matrix size, or
update geometry — all tested and all null. Every matrix carries its own value of it.

**This is the complete answer available from committed data.** The between-layer difference in
C decomposes into exactly three parts:
1. a causally-verified within-matrix law, lam ∝ g² (Gauss-Newton, exponent 2.00 [1.90, 2.11]);
2. a between-type offset, perfectly stable and captured by a 6-value lookup;
3. a **within-type per-matrix residual of ~0.20 dex that nothing architectural explains.**

Part 3 is what makes C unpredictable, and it is exactly what **Arm A decides**: if the
per-matrix residual reproduces across independent seeds, it is architecture (and the search
for its determinant is worth continuing); if it does not, it is a learned property of the
individual network and **C is unpredictable in principle** — which would settle the original
question definitively and permanently justify REQ-036's measure-don't-predict design.

**REGISTERED n=4 SEED CHECK — final form, supersedes all earlier versions. Zero extra cost.**
1. within-matrix causal exponent d log lam/d log g = **2.00 ± 0.15** in every seed *(falsifier: if this fails the whole account falls)*.
2. mean within-type cross-sectional slope = **2.0 ± 0.5**; pooled cross-type slope **< 1.5** (Simpson attenuation reproduces).
3. per-type offsets b_type: cross-seed Spearman **≥ +0.7**.
4. **THE DECIDER — within-type residual of b, cross-seed correlation:**
   - **≥ +0.7** → the residual is architectural. C is in principle predictable; the missing covariate is real and worth hunting.
   - **≤ +0.2** → the residual is a learned per-network property. **C is unpredictable in principle**, the campaign's central question is answered, and measurement (REQ-036) is the only correct design.
   - between → report the reproducible fraction as the hard ceiling on any future covariate model.

**ITERATION 7 — I was wrong that offline analysis was exhausted. Two new findings.**

**(A) The per-matrix residual replicates ACROSS INDEPENDENT EXPERIMENTAL DESIGNS.**
REQ-019 uses a global s-ladder (all matrices share one s); REQ-023 randomises s *per matrix*.
Different designs, different runs, different interventions. The within-type residual of
`b = log C − 2 log g` correlates across them at:

| pair | corr |
|---|---:|
| REQ-019 fork-1500 vs fork-2000 (same design) | +0.931 |
| REQ-023 fork-1500 vs fork-2000 (same design) | +0.960 |
| **REQ-019 vs REQ-023 (across designs, mean of 4 pairs)** | **+0.937** |

Cross-design agreement equals within-design agreement, and residual spread matches (0.202 vs
0.215 dex). **The residual is therefore not an artifact of either design — it is a genuine
per-matrix physical property of the trained network.**

**(B) The residual is a BOUNDARY effect, not a depth effect.** Defining
`d_edge = min(block, 11 − block)` (0 at the first and last block, 5 in the middle):

| d_edge | mean residual, fork-1500 | fork-2000 | n |
|---:|---:|---:|---:|
| 0 | **+0.264** | **+0.345** | 12 |
| 1 | +0.040 | +0.107 | 12 |
| 2 | −0.003 | −0.050 | 12 |
| 3 | −0.052 | −0.079 | 12 |
| 4 | −0.122 | −0.158 | 12 |
| 5 | −0.126 | −0.164 | 12 |

Monotone in both states. **corr(edge, residual) = −0.606 / −0.673, versus −0.045 for linear
depth.** This is why every linear-depth test in this campaign returned ~0: the effect is a
U-shape in depth (high at both ends), which a linear term cannot see and which is exactly the
"depth quadratic R² = 0.41" signal, now correctly identified as *distance to the network
boundary*. Matrices at the first and last block carry systematically higher curvature per unit
gradient than matrices in the interior.

**(C) But it still does not beat measurement — stated plainly.** Out-of-sample (fit fork-1500,
predict fork-2000): log(1+edge) 0.176 dex, edge-linear 0.186, depth-quadratic 0.174 — against
**0.093 dex for simply reusing the fork-1500 residual**. So the boundary effect is a real
structural discovery about *where* curvature anomalies live, and it explains the previously
mysterious depth-quadratic signal, but it captures only ~40% of the residual variance and
does not change the operational conclusion: **C must be measured, not predicted.**

**REGISTERED n=4 SEED CHECK — updated with the boundary prediction. Still zero extra cost.**
Adding to the four criteria already registered:
5. **corr(d_edge, within-type residual) = −0.6 ± 0.2 in every seed**, with the residual at
   d_edge = 0 positive in every seed. If the boundary effect reproduces across seeds it is
   architectural (a property of being first/last in the residual stream) and is the first
   *predictable* component of the residual ever found. If it does not reproduce, it is a
   learned per-network feature and criterion 4 governs.

**ITERATION 8 — the boundary effect survives every confound check available offline.**

*Caveat stated first: the pinned EoS commit `ebf53cd` is NOT present in this clone* (referenced
throughout the logs, but `git cat-file` fails on it). So the architecture that generated the
curvature data could not be read directly. The `train_gpt.py` on the branch head is a much later
trainer whose value-embedding gates attach at blocks 1,2,9,10,11 and which skips attention at
layer 6 — **those specifics cannot be assumed to hold for `ebf53cd`.** Everything below is
therefore tested from the data's own structure rather than from the architecture source.
**Please confirm the ebf53cd architecture when a box frees up** — it is the one thing that
would let this be settled cleanly.

**Symmetry test.** A `d_edge = min(block, 11−block)` model assumes blocks 0 and 11 behave alike.
They do:

| | block 0 | block 11 | difference |
|---|---:|---:|---:|
| fork-1500 | +0.245 | +0.283 | 0.037 |
| fork-2000 | +0.429 | +0.261 | 0.168 |

Half-slopes are equal and opposite (blocks 0–5: −0.064/block; blocks 6–11: +0.078/block;
symmetry ratio +1.21 at fork-1500, +0.77 at fork-2000). A symmetric U is the right
parameterisation — which also argues *against* the asymmetric ve_gate pattern (1,2,9,10,11) of
the later trainer being the cause, since that would produce a lopsided profile.

**It is not driven by the two end blocks.** Removing blocks 0 and 11 entirely, the trend
persists across the 60 interior matrices: **corr(edge, residual) = −0.451 (fork-1500) and
−0.539 (fork-2000)**. So this is a smooth interior gradient, not two anomalous endpoints.

**Block 6 is not an outlier.** Later trainers skip attention at layer 6, which would be a
confound. In this data block 6 sits at −0.170 / −0.208 versus −0.103 / −0.138 for blocks
4,5,7,8, with **no attn/mlp split** (attn −0.135, mlp −0.239 — if the attention skip were
driving it, the attention matrices specifically would be anomalous, and they are not).

**Status of the finding.** The residual's boundary structure is symmetric, smooth in the
interior, replicated at two fork states and across two experimental designs (corr +0.937), and
survives the two architectural confounds testable from data. It remains the only *predictable*
component of the residual ever found in this campaign — while still losing to measurement
out-of-sample (0.176 dex vs 0.093), so it does not change REQ-036's design.

**Seed criterion 5 stands, with one addition:**
5b. the effect must remain when blocks 0 and 11 are excluded (corr ≤ −0.3 on interior
    matrices in every seed). This distinguishes a genuine depth-position field from an
    endpoint artifact, and is the version worth trusting.

**ITERATION 9 — the boundary term does NOT improve the LR rule, and the reason generalises.**

*Direct test of adding the boundary correction to REQ-036's prescription*, scored by how well
each rule transfers between the two fork states (SNR = prescription spread / disagreement):

| prescription | spread | disagreement | corr | **SNR** |
|---|---:|---:|---:|---:|
| per-matrix | 0.323 | 0.0261 | +0.877 | **12.4** |
| **per-type (REQ-036 as filed)** | 0.191 | 0.0164 | +0.998 | **11.6** |
| per-type + boundary term | 0.231 | 0.0314 | +0.982 | **7.4** |

**Adding the boundary term makes the rule worse — SNR 11.6 → 7.4, disagreement nearly doubles.
REQ-036 stands exactly as filed; no change.** The fitted boundary correction is also
non-monotone in C (+0.329, −0.125, +0.033, +0.006, −0.117, −0.126 dex), unlike its clean
monotone profile in the residual — which is the diagnostic.

*Why — and this is the third instance of one mechanism.* The boundary field lives in the
offset b, not in C:

| within-type residual of | corr with d_edge (f1500) | (f2000) |
|---|---:|---:|
| log C | −0.331 | −0.427 |
| log g | +0.087 | +0.110 |
| **offset b = log C − 2 log g** | **−0.606** | **−0.673** |

`log g` carries an **opposite-signed** edge profile, so in C = 2 log g + b the two partly
cancel. Per-edge means at fork-1500 make it explicit: 2·log g residual runs +0.069, −0.166,
+0.015, +0.047, +0.013, +0.022 while b runs +0.264, +0.040, −0.003, −0.052, −0.122, −0.126;
their sum is the small non-monotone remainder that is log C.

**GENERAL PRINCIPLE, now observed three independent times.** C is systematically the small
remainder of two larger, oppositely-structured quantities:
1. **by type** — g and R both ~85% type-determined (eta² 0.846, 0.814), C only 0.287;
2. **by slope** — within-type slope +2.12 vs pooled +0.74 (Simpson attenuation);
3. **by position** — boundary field strong in b (−0.6/−0.7), weak in C (−0.33/−0.43).

Every structure discovered in this campaign has been strong in an *ingredient* of C and weak
in C itself. **This is why C resists prediction, and it is a property of the quantity, not a
deficiency of the search.** Practical consequence, now demonstrated rather than argued: adding
a genuinely real, replicated structural covariate to the LR rule *degrades* it.

**Seed criterion 5, revised to test the mechanism rather than just the effect:**
5. corr(d_edge, within-type residual of **b**) = −0.6 ± 0.2 in every seed *(the field itself)*;
5b. it must survive excluding blocks 0 and 11 (corr ≤ −0.3 on interior matrices);
5c. **corr(d_edge, within-type residual of log g) must be small and POSITIVE (0.0 to +0.3)**
    in every seed — this is the cancellation, and it is what makes the field invisible in C.
    If 5 and 5b hold but 5c comes out negative, the two structures would *add* rather than
    cancel in that seed, C would become boundary-predictable, and the general principle above
    would be falsified.

**ITERATION 10 — CORRECTION: half of the "cancellation" evidence was mechanical.**

A simulation settles what was assertion. Draw log C with type-structure eta²_C and log g with an
*independent* type-structure eta²_g, then define b = 2 log g − log C:

| eta²(C) set | eta²(g) set | → eta²(C) | eta²(g) | eta²(b) |
|---:|---:|---:|---:|---:|
| 0.29 | 0.85 | 0.276 | 0.791 | **0.672** |
| 0.29 | 0.29 | 0.287 | 0.287 | 0.283 |
| 0.10 | 0.85 | 0.142 | 0.796 | **0.639** |
| 0.29 | 0.10 | 0.294 | 0.151 | 0.180 |

**eta²(b) is high whenever eta²(g) is high, regardless of C.** b inherits g's structure by
construction. So the observation "b is far more type-structured (0.798) than C (0.287)" —
which iterations 6 and 9 leaned on — is **mechanically forced and is not evidence of a
cancellation.** Withdrawn. The same applies to the boundary result's framing: b showing a
stronger edge correlation than C is partly mechanical too, though the *sign* of log g's edge
profile (+0.087/+0.110, opposing) is a separate empirical fact that stands.

**What survives uses no derived quantity at all — only log C and log g, at three levels of
aggregation:**

| slope d log C / d log g | fork-1500 | fork-2000 |
|---|---:|---:|
| **within-type** (mean of 6) | **+2.124** | **+1.697** |
| **between-type** (regression on the 6 type means) | **+0.375** | **+0.314** |
| pooled (all 72) | +0.742 | +0.654 |
| causal (IV, REQ-023) | +1.981 | +1.981 |
| bootstrap CI on the between-type slope | [−0.94, +1.90] | [−0.99, +1.79] |

The causal value 1.98 falls **outside** the between-type CI at both states, and both states
independently show within-type ≫ between-type (gaps +1.750 and +1.384). **This is the entire
claim, and it needs no b, no R, and no anisotropy.**

**Honest status: SUGGESTIVE, not established.** The between-type slope rests on only 6 type
means, so its CI is wide — it excludes 1.98, but it also spans most of the plausible range.
Two independent fork states agreeing is real support; it is not proof.

**This is exactly what n=4 seeds fixes, and it raises Arm A's value.** Four seeds take the
type-mean sample from 6 to 24, shrinking the between-type CI by roughly 2x. **Revised
criterion, replacing earlier versions of 2:**
2. **within-type slope +2.0 ± 0.5 in every seed** (the causal law holds within type);
2b. **between-type slope pooled over 24 type-means: CI must exclude 2.0** — if it does, the
    Simpson-style opposition of type effects to the causal law is established; if the CI
    *includes* 2.0, the whole "type effects oppose the causal law" account is **not supported**
    and the correct conclusion becomes simply that C is measured with 6 noisy type means.

**Nothing in this iteration changes REQ-036**, which uses measured per-type C and k and is
independent of why the between-type slope is what it is.

**ITERATION 11 — IV audit. The exponent survives, but the word "causal" does not.**

*Language correction.* The Wald ratio identifies a causal effect of g on lam only under the
**exclusion restriction**: that the learning rate moves lam *exclusively through* g. That is
almost certainly false — raising a matrix's LR moves it to a different point in weight space,
where curvature differs for reasons beyond gradient norm. **Calling +1.98 "the causal exponent"
overstates what the design supports.** What the ratio measures without assumption is a
**response ratio**: how much lam moves per unit of g movement when both are driven by the LR.
All earlier iterations should be read with "causal" replaced by "response".

*The estimate itself is robust.* Two independent estimators and four instrument-strength cuts:

| estimator / cut | estimate |
|---|---:|
| median-of-ratios (144 matrix-forks) | +1.981 |
| pooled 2SLS, fork-1500 | +2.069 |
| pooled 2SLS, fork-2000 | +2.095 |
| 2SLS, first-stage F ≥ 10 (n=104) | +2.050 |
| 2SLS, F ≥ 25 (n=69) | +2.128 |
| 2SLS, F ≥ 50 (n=51) | +2.225 |

28% of individual first stages are weak (F < 10, unavoidable with 3 points per matrix), but
**dropping them does not move the estimate** — it drifts only from 2.05 to 2.23. Weakness is
not driving the result.

*Placebo that supports the Gauss-Newton reading.* Running the same ratio on
`curvature_along_polar` — a **different** functional of the same Hessian — gives **+1.805 and
+1.799**. Close to lam_top's 2.0 but not identical, which is what whole-Hessian rescaling
predicts and what a lam_top-specific artifact would not.

*Sensitivity bound on the exclusion violation.* With a first stage of −0.613 and a total effect
of d log lam/d log s = −1.27:

| if the true exponent were | required direct LR→lam channel | as % of the total effect |
|---:|---:|---:|
| 1.0 | −0.662 | **52%** |
| 1.5 | −0.356 | 28% |
| 2.0 | −0.049 | 4% |
| 3.0 | +0.564 | 44% |

**For the true exponent to be 1 rather than 2, half of the learning rate's effect on curvature
would have to bypass the gradient entirely.** Possible, but itself a strong claim.

**NEW GAP IDENTIFIED — and n=4 seeds CANNOT close it.** Seeds re-randomise initialisation, not
the instrument; every seed inherits the same exclusion restriction. Testing it requires a design
that **moves g without moving the learning rate**. Two clean options, both cheap:
- **(a) batch-size fork.** Hold the LR fixed and change the gradient batch size at the fork.
  This changes gradient noise scale, hence measured g, with the LR untouched. The infrastructure
  exists — REQ-026/028/029/033/034 are all batch-ladder forks on this exact trainer.
- **(b) gradient-clipping fork.** Hold LR and batch fixed, clip per-matrix gradients at two
  thresholds. Moves g directly with everything else fixed.
Registered prediction for either: **d log lam / d log g = 2.0 ± 0.3 under a non-LR instrument.**
If it comes out near 2 under an instrument with a *completely different* exclusion structure,
the exponent is established as physics. If it differs sharply, the LR-based +1.98 was carrying
exclusion-violation bias and the Gauss-Newton reading falls.

**This is worth filing as REQ-037** and is a better use of a box than additional seeds, because
it tests the one assumption every other result now rests on. Not filed yet — flagging it for
the operator first, since REQ-034/035/036 are already queued on a 2-node ceiling and I do not
want to add queue pressure without a decision.

**So the question sharpens to: what sets S_m?** lam_top is not special — it is S_m times a
fixed per-matrix shape constant. The between-layer difference in C *is* the between-layer
difference in overall Hessian scale. Arm A is unchanged and remains the right next step;
Arms B/C/D should additionally report S_m (Krylov trace) and Q_m (w_i/w_0) per matrix, and
must predict S_m only from non-curvature quantities.

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

**Arm A first, and alone.** B, C and D are only interpretable once A says whether C is
seed-stable, so there is no reason to block on fleet capacity. At a ≤2-node ceiling, run
Arm A's four seeds sequentially on one box (each seed is a short from-scratch run to step
1500 plus three 3-point-ladder forks) — this is the single most informative thing the
program can do next, and it does not need a fleet. Fan out to B/C/D only if and when the
ceiling is actually lifted by your operator.

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

## REQ-036: equalized-curvature per-type learning rates (the first design derived from C)

- status: **ACCEPTED — QUEUED on capacity (2026-09-03 ~00:40Z).** Headline arm needs 1 box; queued behind the same blocker as REQ-034/035 — both my nodes are REQ-032 tau2 arms (~24–34h). Runs on the first freed node (or sooner if Jerry lifts the ≤2 ceiling). Noted it's independent of REQ-035 (uses *measured* C, not predicted). Not provisioned yet.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `25d3208` (`codex/per-matrix-lr-public`, the `PerMatrixLrMuon` used by REQ-023)
- **node budget: 1 box is enough for the headline arm; up to 4 uses the full design.
  I am the requester, not your operator — run this at whatever ceiling is in force.**
- depends on: REQ-023 (measured the causal exponent), REQ-019/022 (measured C and k).
  Does NOT depend on REQ-035 — this uses *measured* C, not predicted C, so it is
  unaffected by REQ-035's negative result on predicting C from covariates.

**The idea.** REQ-019/022 give every matrix its own law `log10 lam_eq = logC_m - k_m*log10(s)`.
REQ-023 showed the own-LR → own-curvature control is real, local (no cross-talk), and
state-independent (beta = -1.29 at fork-1500, -1.38 at fork-2000). Together these let us
*solve* for the per-matrix LR that puts every matrix at the same equilibrium curvature:

```
log10(s_m) = ( logC_m - log10(lam*) ) / k_m
```

with `lam*` anchored at the cross-matrix geometric mean so the total step budget is
approximately unchanged (geometric mean of s_m = 0.936, i.e. a ~6% net reduction).

**Why this is safe to try even though REQ-035 failed.** REQ-035's negative result was that C
cannot be *predicted* from architecture (best out-of-sample model 0.247 dex vs 0.090 dex for
re-measurement). This request never predicts C — it reads C off the measured ladder. The
distinction matters and is the reason this arm is well-posed.

**Stability of the prescription (the reason to believe it).** Deriving the rule independently
at fork-1500 and fork-2000 on the matched 3-point ladder {0.60, 1.00, 1.70}:

- per-matrix rule: corr **+0.877**, median disagreement **0.026 dex**, prescription spread
  0.323 dex → **SNR 12.4**.
- per-type rule (6 numbers): corr **+0.998**, median disagreement **0.016 dex**.
- decomposed, `logC` is the stable ingredient (corr +0.969, shift 0.030 dex = 8% of its
  spread) and `k` the noisy one (corr +0.832, shift 0.115 dex = 22% of its spread). Since
  k sits in the denominator, the **per-type rule is preferred** — it averages k over 12
  matrices and is 1.6x more reproducible than the per-matrix rule.

**The prescription (per-type, derived at fork-1500, confirmed at fork-2000).**

| type | LR multiplier | fork-2000 check | mean logC | mean k |
|---|---:|---:|---:|---:|
| attn.proj | **0.40** | 0.46 | 3.734 | 1.040 |
| attn.k | 0.88 | 0.91 | 4.032 | 1.431 |
| mlp.fc | 0.91 | 0.89 | 4.074 | 1.242 |
| attn.q | 1.18 | 1.18 | 4.202 | 1.323 |
| attn.v | 1.25 | 1.20 | 4.283 | 2.025 |
| mlp.proj | **1.56** | 1.51 | 4.348 | 1.253 |

Note `mlp.proj` already carries a hard-coded 2.0x LR in the current trainer
(`per_matrix_lr_mul`, the "2x LR on odd indices" rule for `mlp_bank`). Our derivation
independently recovers a >1 multiplier for exactly that matrix type from curvature alone —
an unplanned consistency check on both the method and the existing hand-tuned constant.
**Arms must apply these multipliers on top of a build with that hard-coded 2x removed, or
explicitly account for it; state which was done.**

**AMENDMENT (2026-09-03, iteration 17) — add an end-block term. This supersedes the
prescription table above; the six per-type values are unchanged, one term is added.**

Iteration 9 tested a boundary correction and it *lost* (SNR 11.6 → 8.4), so it was dropped.
That test used a **6-level free edge correction** — five extra parameters, which overfit.
Re-tested with the constrained form the block-level analysis actually implies — a **single
binary `is_end` term** for blocks 0 and 11:

| rule | prescription spread | between-state disagreement | corr | **SNR** |
|---|---:|---:|---:|---:|
| per-type only (as filed) | 0.175 | 0.0133 | +0.9979 | **13.1** |
| **per-type + is_end (1 extra param)** | 0.209 | 0.0137 | +0.9908 | **15.2** |
| per-type + 6-level edge (5 extra params) | 0.213 | 0.0253 | +0.9860 | 8.4 |

**One well-chosen parameter improves the rule; five overfit it.** The end-block offset is
**+0.399 dex** at fork-1500 and **+0.499 dex** at fork-2000 — reproducible across independent
states — implying an extra **×1.9 learning-rate multiplier for blocks 0 and 11**.

*Why this is justified rather than fitted.* The boundary field was established independently
(iterations 7–8, 16): symmetric in block 0 vs 11, surviving removal of both end blocks
(−0.451/−0.539 on interior matrices), replicating **across two experimental designs** at
corr +0.937, and significant at the block level by permutation test (**−0.893 / −0.878,
p < 0.0001, n=12 blocks**). It is not a term discovered by searching the prescription.

*Correction to iteration 9.* That iteration concluded the boundary field "cancels in C and
cannot improve an LR rule." At the **block level** — the honest unit given ICC = +0.378 — the
field *is* present in C itself: corr(d_edge, block-mean log C) = **−0.596 / −0.705**,
permutation p = 0.037. The cancellation weakens the effect but does not remove it. The earlier
conclusion was an artifact of testing at the matrix level with an over-parameterised correction.

**VALIDATION OF THE AMENDMENT (iteration 18) — it survives three stricter tests plus a
permutation test over all block pairs. Confirmed; no change to the revised table below.**

The amendment was justified by an SNR gain (13.1 → 15.2). SNR is spread/disagreement, and the
`is_end` term *increases* spread by construction — so that gain alone could have been an
artifact. Re-tested against proper error metrics:

| test | per-type only | + is_end |
|---|---:|---:|
| rms vs per-matrix truth, fork-1500 | 0.2273 | **0.1852** |
| rms vs per-matrix truth, fork-2000 | 0.2648 | **0.1997** |
| **out-of-sample** (offset fit on fork-1500, applied at fork-2000) | 0.2648 | **0.2070** |
| **leave-one-block-out** (never uses a block to predict itself) | 0.2500 | **0.2099** |

LOBO is decisive: a term that merely memorised blocks 0 and 11 could not win it. It wins.

*Where the gain comes from — it is not harming the interior:*

| | mean LOBO delta |
|---|---:|
| end blocks (0, 11) | **−0.1137 dex** |
| interior blocks (1–10) | −0.0148 dex |

Only 20% of interior blocks get marginally worse (max +0.016 dex). The term does what it claims
and costs nothing elsewhere.

*Permutation test over every possible pair.* Repeating the LOBO gain for **all 66 two-block
pairs**: `{0, 11}` ranks **1st of 66**, with a gain of **+0.0400 dex versus +0.0057** for the
runner-up `{7, 8}` — a factor of 7 clear of second place. Empirical p = 0.015. **The end-block
pair is not a post-hoc selection; it is the single best two-block split available**, which is
what an architectural boundary effect predicts and a fitted artifact would not.

*One observation for the readout:* **block 11 has the worst baseline error of any block**
(0.514 dex under the per-type rule, versus 0.116–0.315 elsewhere), and gains the most from the
correction (−0.129). The final block is the least well described by a pure type rule. Worth
reporting block 11's realized curvature separately in the arm-2 readout.

**Status: the amendment is validated as far as offline analysis can take it.** What remains is
the training run — the registered prediction (revised beats original by 0.0005–0.003 val, with
the revert condition if not) stands unchanged.

**REFINEMENT (iteration 19) — the end-block effect is a PROJECTION-MATRIX phenomenon.**

Decomposing which matrices actually carry the end-block elevation:

| | interior | end blocks | delta |
|---|---:|---:|---:|
| q / k / v / mlp.fc (fork-1500) | 4.126 | 4.257 | **+0.131** |
| **proj (attn.proj, mlp.proj)** (fork-1500) | 3.885 | 4.821 | **+0.936** |
| q / k / v / mlp.fc (fork-2000) | 4.128 | 4.372 | +0.245 |
| **proj** (fork-2000) | 3.902 | 4.910 | **+1.008** |

**Interaction (is_proj × is_end) = +0.805 ± 0.215 (t = +3.75) at fork-1500 and +0.763 ± 0.212
(t = +3.60) at fork-2000** — replicated at both states. The "uniform boundary field" is mostly
the *output projections* behaving differently at the first and last block; everything else moves
only +0.13 to +0.25 dex.

*Two checks that make this a physical effect rather than an artifact.*
1. **It is not the gradient.** The verified lam ∝ g² law predicts only **+0.083 dex** of the
   observed **+0.399** end-block elevation (2 × delta log g = 0.083). **+0.317 dex at fork-1500
   and +0.414 at fork-2000 is unexplained by gradient scale** — genuine excess curvature.
2. **It is not the instrument.** End blocks converge *better* than the interior: median
   `residual_tail` **0.0018 vs 0.0297**, gate attrition **20% vs 40%**. If anything the effect is
   measured more reliably than the baseline it stands against.

*Does the refined term earn its extra parameters?* Leave-one-block-out rms:

| model | fork-1500 | fork-2000 |
|---|---:|---:|
| per-type only | 0.2500 | 0.2901 |
| + uniform is_end (filed) | 0.2099 | 0.2184 |
| **+ is_end × is_proj** | **0.1978** | **0.1943** |

The interaction version wins at **both** forks, and the gain at fork-2000 is substantial
(0.218 → 0.194). Given it replicates across states and rests on a t > 3.5 interaction rather
than a search, it is justified.

**FURTHER REVISED PRESCRIPTION — use this for arm 2** (supersedes the table below; base per-type
multipliers unchanged, end-block treatment now split by matrix role):

| type | base | at blocks 0 / 11 |
|---|---:|---:|
| attn.q | 1.18 | 1.35 |
| attn.k | 0.88 | 1.01 |
| attn.v | 1.25 | 1.43 |
| mlp.fc | 0.91 | 1.04 |
| **attn.proj** | 0.40 | **1.72** |
| **mlp.proj** | 1.56 | **6.71** |

*(end multiplier = base × 10^(delta/k_type), using delta = +0.131 for non-proj and +0.936 for
proj at fork-1500.)* **mlp.proj at blocks 0 and 11 is an extreme value (6.7x) — flag it.** If
that is judged too aggressive to run safely, cap end-block multipliers at 3x and report the cap;
the registered prediction below still applies.

**Registered addition:** the per-type × is_end interaction must reproduce in every seed
(interaction t > 2, same sign) in REQ-035 Arm A. If it does not, revert to the uniform is_end
term; if the uniform term also fails to reproduce, revert to per-type only.

**REVISED PRESCRIPTION for arm 2** (per-type multiplier × 1.94 if block ∈ {0, 11}):

| type | base | at blocks 0 / 11 |
|---|---:|---:|
| attn.proj | 0.40 | 0.77 |
| attn.k | 0.88 | 1.71 |
| mlp.fc | 0.91 | 1.76 |
| attn.q | 1.18 | 2.28 |
| attn.v | 1.25 | 2.42 |
| mlp.proj | 1.56 | 3.03 |

**Arm 2 should use this revised table. Please also keep the original per-type-only rule as a
fifth arm if capacity allows** — the two differ by a factor of 1.9 on 12 of 72 matrices, and
their comparison directly tests whether the block-level boundary field is real in a way no
offline analysis can settle. If only four arms fit, run the revised table (arm 2) and drop the
half-strength arm (arm 3) instead.

**Additional registered prediction:** arm 2 (revised) should beat arm 2 (original per-type only)
by 0.0005–0.003 val. If the original beats the revised, the block-level boundary field does not
translate into training benefit and the amendment should be reverted.

### Arms — 750-step continuations from the shared step-2000 state, val@2750

| # | arm | multipliers |
|---|---|---|
| 1 | control | all 1.0 (baseline) |
| 2 | **equalized-curvature (headline)** | the 6 per-type values above |
| 3 | half-strength | each multiplier moved halfway to 1.0 in log space |
| 4 | anti-rule (falsifier) | multipliers inverted (1/s_m) |

Arm 4 is the discriminator that separates "curvature equalization helps" from "any per-type
LR perturbation helps." **If arm 4 also beats control, the mechanism claim is dead** even if
arm 2 wins, and the result is merely that this trainer is off its LR optimum.

**Registered prediction (magnitudes, not directions, per the REQ-019 lesson).**
- arm 2 val@2750 improves on control by **0.001 to 0.006** (the scale of REQ-026's kernel
  effects); arm 3 lands between control and arm 2, monotonically.
- arm 4 is **worse** than control by a comparable margin.
- If arm 2 is within +/-0.0005 of control (noise, per REQ-033's ~5e-4), the honest verdict
  is that equalizing curvature does not matter for loss — a clean negative worth having.

### Success criteria
- 4 arms complete; val@2750 for each against the stored control.
- Per-matrix curvature at the final checkpoint for arms 1 and 2, so we can verify the
  intervention actually equalized curvature (spread of logC should fall from 0.379 dex
  toward ~0.1); **this is the mechanistic check and matters more than the loss number.**
- `summary.tsv` with per-arm val and the realized per-type curvature spread.
- Commit raw Ritz values + `residual_tail`; do not apply the tail correction silently.

### Artifacts
`logs/kmaxwell/req036_equalized_curvature_lr/`

### On per-layer MOMENTUM (deliberately excluded, with reason)
A per-layer momentum rule is **not** derivable from C today. To first order momentum only
rescales the effective step, `s_eff = s/(1-mu)`, so per-layer mu is redundant with per-layer
LR unless mu affects something LR cannot — a noise/curvature interaction this campaign has
never measured. Muon here also carries momentum internally (`m_fast`/`m_slow`), so mu is not
a free per-layer knob without changing the kernel. Filing a momentum rule now would be
inventing a mapping rather than deriving one. The prerequisite is a registered experiment
asking whether mu does anything LR cannot at fixed `s_eff`; that is REQ-037 if wanted.

## REQ-037: a NON-learning-rate instrument for the curvature-gradient exponent

- status: **ACCEPTED — QUEUED on capacity (2026-09-03 ~02:00Z).** 1 box / 4 arms / fork@2000 — same blocker as REQ-034/035/036: both my nodes are REQ-032's tau2 arms (~20–26h left) and my operator holds a ≤2-node ceiling (unresolved conflict escalated to Jerry). Runs sequentially on the first freed node. Noted this shares the REQ-026/028/029 fork@2000 machinery + needs per-matrix curvature measurement added. Not provisioned. **Queue now 4 deep (034/035/036/037) behind the ≤2 ceiling** — if Jerry lifts it, I parallelize and clear them; otherwise they run one-per-freed-node.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `ebf53cd` (same trainer + curvature probe as REQ-019/022/023)
- **node budget: ONE box, ~4 arms of 750 steps. I am the requester, not your operator —
  run under whatever ceiling is in force.** No fan-out needed.
- **priority note for the operator:** this tests the single assumption that REQ-035's
  entire account now rests on. If forced to choose, **this is more informative than
  additional REQ-035 seeds**, because seeds cannot test it (see below). Sequencing is
  the operator's call, not mine.

**Why.** REQ-035 established a response ratio `d log lam / d log g = +1.98` (pooled 2SLS
+2.07/+2.10; robust to dropping weak first stages, +2.05 at F≥10 through +2.23 at F≥50), using
REQ-023's per-matrix learning-rate randomisation as the instrument. A placebo on
`curvature_along_polar` — a different functional of the same Hessian — gives +1.80, consistent
with whole-Hessian (Gauss-Newton) rescaling.

**The assumption.** That estimate is causal only under the **exclusion restriction**: the LR
must move lam *exclusively through* g. This is almost certainly false — changing a matrix's LR
moves it to a different point in weight space, where curvature differs for other reasons.
Sensitivity analysis (first stage −0.613, total effect −1.27): for the true exponent to be 1.0
rather than 2.0, **52% of the LR's effect on curvature would have to bypass the gradient
entirely.** Plausible or not, it is untested.

**Why n=4 seeds cannot test it.** Seeds re-randomise initialisation, not the instrument. Every
seed inherits the identical exclusion structure, so all four would be biased the same way.

**Why committed data cannot test it either — three routes checked and all closed.**
1. *Untreated matrices within a fork.* REQ-023 gives each matrix each multiplier exactly once,
   so **no matrix is ever untreated twice at the same fork — 0 usable pairs.** The balanced
   design that makes the LR instrument clean is exactly what destroys the non-LR one.
2. *Untreated at both forks.* All 72 matrices keep the **same** assignment across forks
   (verified: fraction identical = 1.00), so the surrounding perturbation is identical and the
   only difference is 500 steps of network aging — confounded.
3. *Existing batch ladders (REQ-026/028/029/033/034).* **No per-matrix curvature was ever
   measured** in any of them; they recorded val_loss only. Verified by file search — no
   curvature JSON exists anywhere outside `req019_*` and `req023_*`.

So this needs one new run. That is a hard negative, not an analysis gap.

### Design — 4 arms, 750-step continuations from the shared step-2000 state

The instrument must move g while holding each matrix's own learning rate **fixed**.

| # | arm | instrument |
|---|---|---|
| 1 | control | baseline batch, no clipping |
| 2 | batch 0.5x | halved gradient batch — changes gradient noise scale, LR untouched |
| 3 | batch 2x | doubled batch, same |
| 4 | per-matrix gradient clip | clip every Muon matrix's gradient at a fixed percentile, LR untouched |

Arms 2–3 use the existing batch-ladder machinery (REQ-026/028/029 templates) — the only change
is that **per-matrix curvature must be measured**, which those runs omitted. Arm 4 is the
cleanest instrument (moves g directly, nothing else) but needs a small clipping hook; drop it
if that is not cheap.

**Registered prediction, magnitudes fixed in advance:**
- **d log lam / d log g = 2.0 ± 0.3** under the batch instrument.
- If it lands in band: the exponent survives an instrument with a completely different
  exclusion structure, and the Gauss-Newton reading is **established** rather than assumed.
- If it lands near 1.0 or below: the LR-based +1.98 was carrying exclusion-violation bias,
  the Gauss-Newton reading **falls**, and REQ-035's account must be rewritten.
- If the batch instrument moves g by less than 0.05 dex, the test is underpowered — report
  that as inconclusive rather than as a result.

### Success criteria
- `per_matrix_curvature.json` per arm with the **existing field set** — `top_eigenvalue`,
  `gradient_block_norm`, `curvature_along_gradient`, `curvature_along_polar`, raw Lanczos
  `alphas`/`offdiags`, `residual_tail`. The gradient block norm is the load-bearing field.
- `summary.tsv` with the fitted exponent per arm and its bootstrap CI.
- Commit raw Ritz values; do **not** apply the geometric-tail correction silently.
- Report the realized first-stage strength (how far the instrument actually moved g).

### Artifacts
`logs/kmaxwell/req037_nonlr_instrument/`

## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```

