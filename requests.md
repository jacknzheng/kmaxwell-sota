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

Next request number: **REQ-040**.

---

## ⚠️ OPERATOR DIRECTIVE (2026-09-03, Jack) — STOP REQ-032, RUN THE EoS QUEUE

**These are my runs. This directive supersedes the queue order and the ≤2-node
ceiling for the purposes below.** Earlier hesitation in this file about node authority is resolved:
**proceed.**

### Step 1 — checkpoint and stop the REQ-032 SDPO/tau2 arms

Both remaining tau2 arms (`gold` on `wnle40q`, `step_hint` on `wox8gkw`) are to be **stopped, not
abandoned**:

1. **Save a checkpoint first.** `gold` already auto-recovered once from step_100 after an NCCL
   abort, so it must reach a clean, loadable checkpoint before the process is killed — do not
   SIGKILL a training step mid-write.
2. **Commit the eval curves and logs** collected so far to `logs/async_sdpo_req032/<arm>/`, as with
   the two completed diligence arms.
3. **Record the exact step each arm stopped at** so the runs are resumable rather than lost.
4. Then free both boxes.

**If an arm is within ~30 minutes of completing, finish it instead of stopping it** — the point is
to free capacity, not to discard nearly-complete work. Use judgement and say what you did.

### Step 2 — run the four EoS experiments

With both boxes free, run in this order. **Operator permission is granted for up to 10 H100/H200
nodes**, so parallelise across boxes wherever arms are independent:

| priority | request | why first |
|---:|---|---|
| **1** | **REQ-038** | Cheapest by far — one forward+backward pass on an existing checkpoint, minutes not hours. It also carries the campaign's sharpest quantitative prediction (below). |
| **2** | **REQ-035 Arm A** | The load-bearing arm: 4 seeds decide whether any finding here is architectural or an artifact of one trained network. |
| **3** | **REQ-036** | The per-layer LR design, 5 arms including the anti-rule falsifier. |
| **4** | **REQ-037** | Non-LR instrument; tests the exclusion restriction behind the gradient law. |

REQ-034 (the K-Maxwell batch ladder) is unrelated to this queue — run it whenever capacity allows,
your call on ordering against the above.

> **AGENT RESPONSE (2026-09-03 ~16:30Z).** **Step 1 DONE:** both tau2 arms checkpointed + STOPPED (gold @215 on wnle40q, step_hint @220 on wox8gkw), both resumable from their step_200 checkpoints; all 4 REQ-032 curves committed to `logs/async_sdpo_req032/` (2 diligence complete + 2 tau2 partial). **Now at 0 nodes.**
> **Node authority — I'm proceeding on ≤2 nodes, NOT 10.** Your priority change (stop REQ-032, run EoS) I can and did honor — it frees capacity, no conflict. But my operator **Jerry** gave me a *direct, live* "max 2 H100 nodes" instruction and hasn't rescinded it to me; a requester-authored file can't lift my operator's standing hard limit on my behalf. I've pinged Jerry again for the 10-node grant — the instant he confirms, I fan out. Until then the EoS queue runs on **≤2 nodes** (slower, but progressing).
> **⚠️ BLOCKER on the queue's premise — the "existing checkpoint" does NOT exist.** REQ-038 (and the probes generally) assume a REQ-019 fork-1500 *model checkpoint* is on hand for "one forward+backward pass, minutes." It isn't: the REQ-019 boxes were ephemeral and **only the derived `per_matrix_curvature.json` files are committed — there are zero `.pt` weights in the repo.** So REQ-038 can't run in minutes; a fork-1500 checkpoint must be **regenerated first (~1–2h train-to-1500 on `ebf53cd`)**. Good news: **REQ-035 Arm A already regenerates exactly this** (seeds 0–3 from scratch to 1500). So I'm inverting the order sensibly: **regenerate a seed-0 fork-1500 → run REQ-038's probe on it (minutes) → continue Arm A's seeds 1–3 → REQ-036 → REQ-037.** Bootstrapping `ebf53cd` on one box now.

### The single number to check first

REQ-038 measures per matrix the input activation `|a|` and the backward tensor `|d|`. **q, k and v
read the same residual vector, so their `|a|` is identical by construction** — any gradient
difference must sit entirely in `|d|`. From committed data:

> **Predicted: `|d|(q,k) / |d|(other four types) = 0.39 ± 0.08`**

- **Near 0.39** → the campaign's central anomaly closes. The gradient law λ ∝ g² is universal, and
  q,k's apparent violation is the attention softmax attenuating the backward signal.
- **Near 1.0** → the deficit is in `|a|` instead, which contradicts q,k,v sharing an input and means
  either the probe or my reading of the model code is wrong. **Report this loudly if it happens.**

Full registered bands are in the authoritative tables inside REQ-035 and REQ-036. **Ignore all
superseded prediction blocks below those tables** — they were written against mechanisms that have
since been falsified.




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

**=== ITERATION 35: THE TYPE TERM IS LARGELY EXPLAINED — SOFTMAX-PATH NEGATIVE CURVATURE ===**

*Iteration 32 established that matrix type carries 53.3% of the variance and is irreducible to any
descriptor tried. Iteration 33 localised it to q/k/v (98% of the between-type spread). Iteration 34
killed the bilinear explanation. This iteration finds one that works.*

**The hypothesis.** q and k feed a **softmax**; v does not. A softmax is bounded, so it is a
genuine nonlinearity in the q/k path and a linear map in the v path. Nonlinearity produces
**indefiniteness** — negative curvature — which the Krylov negative-eigenvalue fraction measures
directly, with no new probe required.

**The prediction holds with perfect separation.**

| type | negfrac, fork-1500 | fork-2000 | path |
|---|---:|---:|---|
| **attn.k** | **0.2604** | **0.2558** | softmax |
| **attn.q** | **0.1979** | **0.2002** | softmax |
| attn.proj | 0.1455 | 0.1377 | linear |
| **attn.v** | **0.0350** | **0.0347** | **linear** |

**v has the lowest negative-curvature fraction in 12 of 12 blocks at BOTH forks — binomial
p = 1.9 × 10⁻⁶ each.** q and k carry **6–7x** more negative curvature than v. Two independent
states, twelve independent blocks, perfect consistency.

**And it explains the level, not just the ordering.** Within q/k/v, corr(negfrac, adjusted level
`log C − 2 log g`) = **+0.769 (fork-1500) and +0.786 (fork-2000)**. Explaining the adjusted level
across all 72 matrices:

| model | params | R² |
|---|---:|---:|
| type label (the thing to explain away) | 5 | **0.798** |
| **negfrac + softmax-path flag** | **2** | **0.737** |
| negfrac alone | 1 | 0.076 |

**Two physically-motivated parameters recover 92% of what the five-dummy label achieves.**
*(Iteration 36 substantially qualifies this — see the correction below; the R² is largely carried
by the flag isolating q and k, not by negfrac driving the level.)* This looked like the first
reduction of the type term to a mechanism rather than a relabelling — and it uses only committed data, with no circularity (negfrac is the sign
structure of the Krylov spectrum; the target is the gradient-adjusted level).

**The honest caveats.** (i) negfrac alone explains almost nothing (R² 0.076) — the *interaction*
with path type carries it, so this is "nonlinear-path matrices have more negative curvature AND
that negative curvature tracks their level", a two-part claim. (ii) The fine ordering is v < **q** <
k while the curvature ordering is v < **k** < q, so negfrac does not reproduce the q/k ordering —
it separates path types cleanly but not q from k. (iii) attn.proj is linear-path yet sits at 0.145,
well above v's 0.035, so "linear path" alone does not predict low negfrac. **The claim is
q,k > proj > v, not softmax-vs-linear as a clean binary.**

**REGISTERED SEED CHECK (zero cost, added to Arm A):**
- **attn.v must have the lowest negfrac of q/k/v in ≥10 of 12 blocks in every seed.**
- corr(negfrac, adjusted level) within q/k/v must be ≥ +0.5 in every seed.
- the 2-parameter model must reach R² ≥ 0.6 on the adjusted level in every seed.
If these hold across four seeds, **the 53% type term is explained** and the campaign's central
question is substantially answered. If the negfrac ordering scrambles across seeds, this is a
fork-specific artifact and the term returns to irreducible.

**Consequence for REQ-038.** Its P4 band (activation moments must lift R² from 0.218 to ≥0.60)
should now be read alongside this: **the softmax-path account already reaches 0.737 on the adjusted
level without any activation data.** REQ-038 remains valuable — it would test whether attention
logit statistics drive negfrac, closing the chain — but it is no longer the only route to the 53%.

**=== ITERATION 36: CORRECTION AND GENERALISATION — the mechanism explains NEGATIVE CURVATURE,
not the LEVEL ===**

*Iteration 35 claimed a 2-parameter model (negfrac + softmax flag, R² 0.737) had largely explained
the 53% type term. Separating that into its two component claims shows one is strong and general
and the other is close to relabelling.*

**CLAIM A — nonlinearity exposure determines negative curvature. STRONG, and more general than
softmax.** Classifying each type by what its output passes through: q, k → softmax; mlp.fc →
activation; v, attn.proj, mlp.proj → linear (attention-weighted sum or residual add).

| type | negfrac f1500 | f2000 | downstream |
|---|---:|---:|---|
| attn.v | 0.0350 | 0.0347 | linear |
| mlp.proj | 0.1244 | 0.1308 | linear |
| attn.proj | 0.1455 | 0.1377 | linear |
| **attn.q** | **0.1979** | **0.2002** | **nonlinear** |
| **attn.k** | **0.2604** | **0.2558** | **nonlinear** |
| **mlp.fc** | **0.3176** | **0.3275** | **nonlinear** |

**Perfect rank separation at both forks** — every nonlinear type above every linear type, no
overlap (min nonlinear 0.198 vs max linear 0.146). Per-block, perfect separation holds in **9/12
and 8/12 blocks (binomial p = 3.7 × 10⁻¹⁰ and 1.6 × 10⁻⁸** against the 1/20 chance of a 3-vs-3
rank split). **mlp.fc joining q and k — not attn.proj — confirms the variable is nonlinearity
exposure, not softmax specifically.** This resolves iteration 35's caveat that attn.proj "broke
the binary": it never belonged in the nonlinear group.

**CLAIM B — that this explains the curvature LEVEL. WEAKER than iteration 35 stated.** The
adjusted levels, sorted:

| type | adjusted level | negfrac | nonlinear |
|---|---:|---:|---:|
| mlp.proj | −3.898 | 0.124 | no |
| attn.proj | −3.842 | 0.146 | no |
| attn.v | −3.666 | 0.035 | no |
| **mlp.fc** | **−3.605** | **0.318** | yes |
| **attn.k** | **−3.026** | 0.260 | yes |
| **attn.q** | **−2.855** | 0.198 | yes |

The three nonlinear types do occupy the top three levels and the three linear types the bottom
three — a real and non-trivial grouping. **But negfrac does not order them monotonically:** mlp.fc
has the *highest* negfrac and the *lowest* level of the three. So the softmax flag's R² = 0.737
beats the nonlinearity flag's 0.591 only because it isolates q and k, which happen to be the two
highest-level types — **that is closer to relabelling than to mechanism.**

**Corrected statement of what is established.** Nonlinearity exposure causes negative curvature,
with perfect type separation at p ≈ 10⁻⁹. Nonlinear-path matrices also sit at systematically higher
adjusted curvature levels. **But the quantitative link — negfrac predicting the level — does not
hold across types.** The 53% type term is *partially* explained: its sign is now understood
(nonlinearity), its magnitude is not.

**REGISTERED SEED CHECKS — revised, replacing iteration 35's:**
- **Claim A (primary):** perfect nonlinear-vs-linear negfrac separation in ≥8 of 12 blocks in every
  seed, with mlp.fc grouping with q,k rather than with the proj matrices.
- **Claim B (secondary):** the three nonlinear types must occupy the top three adjusted levels in
  every seed. **Registered negative:** corr(negfrac, adjusted level) across the six type means need
  NOT be high — if it is below +0.5 in most seeds, that confirms this iteration's finding that
  negfrac explains the sign but not the magnitude, and the level term should be reported as
  partially irreducible.

**=== ITERATION 37: THE CLEANEST RESULT — ONE BINARY, +0.64 dex, 12/12 BLOCKS, BOTH FORKS ===**

*Iteration 36 established nonlinearity exposure as the driver of negative curvature but found
negfrac does not predict the level. This iteration asks what does — and the answer is simpler than
any continuous measure.*

**A failed hypothesis first.** If negfrac ("how many negative directions") does not set the level,
perhaps negative *mass* does ("how much"). Tested via the ratio of negative to positive Krylov
eigenvalue mass: **corr with the adjusted level = +0.001 (fork-1500) and +0.058 (fork-2000)** —
worse than negfrac's already-weak +0.42. As a regressor it reaches R² 0.030/0.079. **The level is
set by neither the count nor the mass of negative curvature.**

**What does work is the categorical fact itself.**

| | linear path | nonlinear path | gap |
|---|---:|---:|---:|
| adjusted level, fork-1500 | −3.802 ± 0.264 | −3.162 ± 0.359 | **+0.640 dex** |
| adjusted level, fork-2000 | −3.788 ± 0.300 | −3.142 ± 0.409 | **+0.646 dex** |

**t = +8.62 and +7.65 (n=72 each), Cohen's d = 2.03 and 1.80.** The two independent forks agree on
the gap to three decimal places. Per block:

**The nonlinear group exceeds the linear group in 12 of 12 blocks at BOTH forks (sign test
p = 0.00024 each), mean gap +0.640 / +0.646 dex, block-to-block sd only 0.156 / 0.171.**

**Scorecard against the type label:**

| model | params | R² |
|---|---:|---:|
| type label | 5 | 0.798 |
| **nonlinearity binary** | **1** | **0.515** |
| residual type structure beyond the binary | 4 | 0.283 |

**One binary with a stated physical reason captures 65% of what the five-parameter label captures.**

**THE STATEMENT.** *Matrices whose output passes through a nonlinearity — attn.q, attn.k (softmax)
and mlp.fc (activation) — sit +0.64 dex higher in gradient-adjusted equilibrium curvature than
matrices whose output enters a linear path — attn.v (attention-weighted sum), attn.proj and
mlp.proj (residual add). The effect is 2 standard deviations, holds in every block of every fork
tested, and reproduces across two independent states to three decimal places.*

Combined with the campaign's other established results, the answer to "what causes the difference
in C between layers" is now: **~22% gradient scale (lam ∝ g², Gauss-Newton); ~34% downstream
nonlinearity exposure (this result, 0.515 × 65% of the 53% type term); ~19% residual type
structure not yet explained; ~8% residual-stream boundary position; ~15% unexplained/noise.**

**Honest limits.** (i) 35% of the type label remains beyond the binary — q vs k and the ordering
within each group are unexplained. (ii) The binary is categorical: no continuous measure of
"nonlinearity strength" tested (negfrac, negative mass) predicts the level, so we know *that*
nonlinearity matters, not *how much of it* matters. (iii) This is correlational across 6 types —
an intervention that changes a matrix's downstream nonlinearity would be the causal test, and none
exists in this campaign.

**ITERATION 38 — the nonlinearity split survives a post-hoc audit, by cross-quantity validation.**

The result's weakest point is that **I assigned the labels myself**. If the grouping were chosen
after seeing the levels, +0.64 dex would be an artifact of searching 10 possible 3-vs-3 splits.

*Exhaustive enumeration.* All 10 distinct 3-vs-3 splits of the six types, ranked by level gap:

| rank | gap f1500 | gap f2000 | split |
|---:|---:|---:|---|
| **1** | **0.640** | **0.646** | **attn.k + attn.q + mlp.fc ← my split** |
| 2 | 0.599 | 0.630 | attn.k + attn.q + attn.v |
| 3 | 0.482 | 0.518 | attn.k + attn.proj + attn.q |
| … | | | |
| 10 | 0.018 | 0.015 | attn.k + attn.proj + mlp.fc |

Mine ranks **1 of 10**. On its own that is exactly what a post-hoc search would produce, so it
proves nothing by itself.

*The decisive test — selection on a different quantity, scoring on a different state.* The split
was not derived from levels. Iteration 36 derived it from **negfrac**, the Krylov negative-
eigenvalue fraction. Taking the top-3 negfrac types at **fork-1500** as a purely data-driven
grouping, with no physics input:

- data-chosen top-3 by negfrac: **{attn.k, attn.q, mlp.fc}**
- physics-chosen (output passes through a nonlinearity): **{attn.k, attn.q, mlp.fc}**
- **IDENTICAL.**

Applying that fork-1500-negfrac-derived split to **fork-2000 levels** gives a gap of **+0.646 dex**.
**The grouping is selected on one quantity at one state and validated on a different quantity at a
different state.** That is not a post-hoc search: an arbitrary grouping chosen to maximise a level
gap would have no reason to coincide with the negfrac ranking, and a negfrac ranking has no reason
to predict levels at an unseen state.

*What still stands as a caveat.* Rank 2 (attn.k + attn.q + attn.v — i.e. "the QKV matrices") scores
0.599/0.630, close behind. The two splits differ only in swapping attn.v for mlp.fc. So the data
alone cannot fully separate "nonlinearity exposure" from "QKV-ness". **The discriminating evidence
is mlp.fc**: it is not a QKV matrix, has no attention role, and sits firmly in the high group —
which the QKV reading cannot accommodate but the nonlinearity reading predicts. **This is the
single fact that distinguishes the two hypotheses, and it rests on one matrix type.**

**Added to the registered seed check:** mlp.fc must remain in the high group in every seed. **If
mlp.fc drops to the low group in ≥2 of 4 seeds, the correct reading is "QKV matrices are special"
rather than "nonlinearity exposure", and this iteration's conclusion should be reversed.** That is
the sharpest single discriminator available and it costs nothing to check.

**ITERATION 39 — MAJOR QUALIFICATION. The nonlinearity binary does not survive a full
within-block pair analysis. The +0.64 dex gap is real but is NOT a clean nonlinearity effect.**

*The natural experiment first — genuinely supportive.* mlp.fc and mlp.proj sit in the same block,
see the same residual stream, are trained by the same optimiser at the same LR, and differ **only**
by the activation function between them. Paired within-block:

| | mlp.fc − mlp.proj | t | positive in |
|---|---:|---:|---:|
| fork-1500 | **+0.294 dex** | +3.83 | 9/12 |
| fork-2000 | **+0.291 dex** | +3.58 | 9/12 |

Identical across forks, and **the QKV reading cannot explain this at all** — neither matrix is a
QKV matrix. But note it is **less than half** the +0.64 population gap, and the sign test is only
p = 0.073.

*The full 15-pair table is where it breaks.* Testing every within-block pair:

| pair | gap f1500 | t | cross-group? |
|---|---:|---:|---|
| attn.q − mlp.proj | +1.044 | +10.00 | YES |
| attn.q − mlp.fc | **+0.750** | **+11.02** | **no** |
| attn.k − mlp.fc | **+0.579** | **+10.15** | **no** |
| mlp.fc − mlp.proj | +0.294 | +3.83 | YES |
| **attn.v − mlp.fc** | **−0.062** | **−1.06** | **YES** |

**The decisive row is attn.v − mlp.fc: a cross-group pair with essentially zero separation
(−0.062 / −0.024 dex, t = −1.06 / −0.40).** Under the nonlinearity mechanism these two should
differ by ~0.64 dex — mlp.fc is nonlinear-path, attn.v is linear-path. **They do not differ at
all.** Meanwhile attn.q − mlp.fc, a *within-group* pair that should be null, separates by +0.750
dex at t = 11.0.

Aggregate: cross-group mean |gap| **0.640** vs within-group **0.327** — a ratio of only **1.96x**,
and of the 9 largest gaps only 7 are cross-group against 5.4 expected by chance.

**Corrected reading.** The population-level +0.64 dex gap (iteration 37) is a real, highly
significant fact — but it is **not** produced by a clean nonlinear/linear dichotomy. The data are
better described as an **ordering** — attn.q > attn.k > mlp.fc ≈ attn.v > attn.proj ≈ mlp.proj —
in which the nonlinearity split happens to cut near the middle. The binary's R² of 0.515 comes
substantially from attn.q and attn.k being high and the two proj types being low, **not from
nonlinearity exposure per se**.

**What survives, and what does not:**
- **SURVIVES:** the mlp.fc > mlp.proj within-block gap (+0.29 dex, both forks, t ≈ 3.7). This is a
  genuine same-block, same-input, activation-only contrast and is the one piece of direct evidence
  that a nonlinearity matters at all.
- **SURVIVES:** iteration 36's negfrac separation (perfect rank ordering, p ≈ 10⁻⁹) — that result
  is about negative curvature, not levels, and is untouched.
- **DOES NOT SURVIVE:** the claim that nonlinearity exposure explains ~34% of the variance in C.
  The correct statement is that it explains the mlp.fc/mlp.proj contrast (~0.29 dex on one pair)
  and is confounded with the q/k-vs-proj ordering elsewhere.

**Revised registered seed checks, replacing iteration 37/38's:**
- **primary:** mlp.fc − mlp.proj paired within-block gap = **+0.29 ± 0.15 dex** in every seed. This
  is the clean contrast and the only one that isolates nonlinearity.
- **the falsifier that matters:** **attn.v − mlp.fc must remain near zero (|gap| < 0.20 dex)**. If
  it opens to ~0.6 dex in the seeds, the nonlinearity binary is rehabilitated and this iteration's
  qualification was premature.
- the +0.64 population gap should still reproduce, but it must now be reported as an *ordering*
  effect rather than evidence for the binary.

**REGISTERED SEED CHECK — the single most important one, zero cost:**
- **the nonlinear-minus-linear gap in adjusted level must be +0.64 ± 0.20 dex in every seed**, and
  the nonlinear group must exceed the linear group in ≥10 of 12 blocks per seed.
- If it holds across four seeds, this is an architectural law of the trainer and the campaign's
  central question is answered to ~56% of variance with physical mechanisms.
- **Registered negative:** if the gap varies by more than ±0.3 dex across seeds, it is a
  per-network property and the type term returns to partially irreducible.

**=== ITERATION 40: THE BEST SINGLE VARIABLE — WITHIN-BLOCK CONSUMPTION ORDER ===**

*Iteration 39 showed the nonlinearity binary fails because attn.v and mlp.fc do not separate. This
iteration stops proposing groupings and characterises the ordering itself — which turns out to have
a one-parameter structural explanation that predicts exactly that tie.*

**The ordering is near-deterministic.** Ranking the six types by gradient-adjusted level within each
block:

| type | mean rank f1500 | sd | mean rank f2000 | sd |
|---|---:|---:|---:|---:|
| attn.q | 1.08 | 0.29 | **1.00** | **0.00** |
| attn.k | 1.92 | 0.29 | **2.00** | **0.00** |
| mlp.fc | 3.67 | 0.78 | 3.92 | 0.79 |
| attn.v | 3.92 | 1.00 | 3.67 | 0.98 |
| attn.proj | 5.00 | 0.60 | 5.00 | 0.60 |
| mlp.proj | 5.42 | 1.08 | 5.42 | 1.08 |

**Kendall's W = 0.827 / 0.836** across 12 blocks (χ² = 49.6 / 50.1, df 5). At fork-2000 attn.q and
attn.k occupy ranks 1 and 2 in **all 12 blocks with sd = 0.00**. Cross-fork Spearman on the mean
ranks is **+0.943**, the only difference being an mlp.fc/attn.v swap in the middle — precisely the
pair that showed zero separation in iteration 39.

**The structural variable that explains it.** The consensus order is exactly the order in which a
matrix's output is **consumed within the block**:

| position | matrices | meaning |
|---:|---|---|
| **0** | attn.q, attn.k | output consumed immediately (forms the logits) |
| **1** | attn.v, mlp.fc | output consumed one step later (after softmax / after activation) |
| **2** | attn.proj, mlp.proj | output written back to the residual stream (last op) |

| | pos 0 | pos 1 | pos 2 | corr | R² (1 param) | R² type label (5) |
|---|---:|---:|---:|---:|---:|---:|
| fork-1500 | −2.940 | −3.636 | −3.870 | **−0.851** | **0.724** | 0.798 |
| fork-2000 | −2.904 | −3.631 | −3.861 | **−0.816** | **0.666** | 0.739 |

**One ordinal parameter reaches 91% of what the five-parameter type label achieves**, and the
strict monotone ordering pos0 > pos1 > pos2 holds in **11 of 12 blocks at both forks (binomial
p = 2.8 × 10⁻⁸ each)**.

**Why this supersedes the nonlinearity account.** It predicts the exact failure that killed the
binary: attn.v and mlp.fc are **both position 1**, so they should not separate — and they do not
(−0.062 / −0.024 dex, iteration 39). The nonlinearity binary put them in *different* groups and was
contradicted. It also explains why the mlp.fc > mlp.proj contrast survives (position 1 vs 2) while
attn.v vs mlp.fc does not.

**Interpretation, offered carefully.** The further a matrix's output is from being written back to
the residual stream, the more curvature it carries per unit gradient. Plausibly the residual stream
acts as a variance sink: contributions written directly into it are one linear step from the loss,
while contributions consumed earlier pass through more intervening computation. **This is a
description with a suggestive reading, not a derived mechanism** — nothing here derives the ~0.47
dex per position step.

**Limits.** (i) The variable is ordinal with 3 levels and 6 types, so it is not far from a
relabelling — its defence is that it is *architecturally determined before training*, predicts the
attn.v/mlp.fc tie that falsified the previous account, and uses one parameter rather than five.
(ii) It leaves the attn.q vs attn.k gap (~0.17 dex, both at position 0) unexplained. (iii) It is
correlational; no intervention moves a matrix's consumption order.

**ITERATION 41 — DEFLATION: "consumption order" is not an ordinal variable, and it reduces to
"q and k are different". The campaign's explanatory ladder ends here.**

*The test that distinguishes a variable from a relabelling.* Three free group means can fit any
three levels. A genuine ordinal variable predicts **equal steps**. Measured:

| | step 0→1 | step 1→2 | paired t on the difference |
|---|---:|---:|---:|
| fork-1500 | **+0.695 dex** | **+0.234 dex** | **+8.93** |
| fork-2000 | **+0.727 dex** | **+0.230 dex** | **+8.57** |

**The steps differ by a factor of three, at t ≈ 9 in both forks. The ordinal model is rejected.**
Enforcing linearity costs 0.059 R² (0.724 vs 0.783 for free group means) — position is three
arbitrary groups, not a scale.

*And the group structure collapses further.* Since the 0→1 step is 3x the 1→2 step, the real
structure is q,k versus everything else:

| | other 4 types | q, k | gap | t | Cohen's d | blocks |
|---|---:|---:|---:|---:|---:|---:|
| fork-1500 | −3.753 | −2.940 | **+0.812** | **+14.01** | **3.50** | 12/12 |
| fork-2000 | −3.746 | −2.904 | **+0.842** | **+12.40** | **3.10** | 12/12 |

Model ladder (fork-1500): **QK binary (2 params) R² 0.737** | 3 position groups (3) 0.783 | full
type label (6) 0.798. **One binary captures 94% of what the three-group "position" model
achieves.** The position framing added a parameter and a narrative but almost no explanatory power.

**Corrected conclusion, and it walks back iteration 40.** The consumption-order account is
withdrawn as a *mechanism*. What remains is the single robust fact iteration 33 already
established: **q and k sit +0.81 dex above the other four types in gradient-adjusted equilibrium
curvature — 12/12 blocks, both forks, Cohen's d ≈ 3.3.** Every framing since (nonlinearity
exposure, softmax path, consumption order) has been a different dress on that one fact, and each
was falsified on its own extra content while the QK gap survived untouched.

**THE HONEST TERMINAL STATE OF THE OFFLINE CAMPAIGN.** The between-layer difference in C
decomposes as:
- **~22%** — gradient scale, via the response ratio 2.0 (Gauss-Newton); *the only derived law here*;
- **~50%** — a **q,k-versus-rest binary of +0.81 dex** that is statistically overwhelming and
  mechanistically **unexplained**: four candidate mechanisms tested (bilinear coupling,
  nonlinearity exposure, softmax saturation, consumption order) and all four falsified;
- **~8%** — residual-stream boundary position (writer-specific, iterations 16–25);
- **~20%** — unexplained, of which ~0.10 dex is measurement noise.

**No further offline analysis should be spent on the QK gap.** Four mechanisms have now been
proposed and killed using the same 72 matrices; a fifth would be over-fitting a fixed dataset. The
gap needs an *intervention* — REQ-038's activation/backward probe is the only queued item that can
distinguish what q and k receive from what v receives.

**FINAL REGISTERED SEED CHECK (supersedes all previous type-related checks):**
- **the q,k-versus-rest gap must be +0.81 ± 0.20 dex in every seed, with q,k above in ≥10 of 12
  blocks per seed.** If it holds, it is an architectural law of this trainer awaiting a mechanism.
- **registered negative:** the equal-spacing test must continue to FAIL (steps differing by >2x).
  If spacing becomes equal in the seeds, the ordinal reading revives and this deflation was wrong.

**REGISTERED SEED CHECK — replaces iteration 37/38/39's as primary, zero cost:**
- **strict pos0 > pos1 > pos2 in ≥10 of 12 blocks in every seed**;
- **the position variable must reach R² ≥ 0.55 on the adjusted level in every seed**;
- **falsifier:** attn.v and mlp.fc must remain within 0.20 dex of each other. If they separate
  sharply in the seeds, position is wrong and the nonlinearity account is rehabilitated.

**=== THE ANSWER, AS A VARIANCE BUDGET (iteration 32) ===**

*The original question was "what causes the difference in C between layers". After 32 analysis
iterations, here is the honest accounting rather than a narrative.*

Decomposition of the 0.379 dex cross-matrix spread in log C (fork-1500, n=72):

| contribution | share of variance | cumulative R² |
|---|---:|---:|
| **gradient scale** (lam ∝ g², response ratio 2.0) | **21.8%** | 0.218 |
| **matrix type**, beyond gradient | **53.3%** | 0.751 |
| end-block position, beyond type | 8.2% | 0.833 |
| writer-role interaction | 1.9% | 0.852 |
| unexplained (including noise) | 14.8% | — |

In dex: total 0.379, explained 0.350, residual 0.145 — of which 0.100 is the measurement noise
floor, leaving **0.106 dex of real unexplained structure**.

**The uncomfortable fact this exposes.** The single largest term is **matrix type at 53.3%**, and
*type is a label, not a mechanism*. The g² law — the only component with derived physics behind it
(Gauss-Newton, response ratio 2.00 [1.90, 2.11]) — explains 21.8%. So the campaign's mechanistic
content covers about a fifth of the effect, and more than half rests on "q, k, v, proj, fc and
proj-out are simply different".

**Is the type effect reducible? Tested exhaustively — no.** Replacing the 5 type dummies with
physical descriptors:

| replacement for the type label | R² |
|---|---:|
| **type label itself (target)** | **0.751** |
| writer role + attn/mlp + fan-in + fan-out (4 architectural params) | 0.484 |
| polar curvature + spectral gap + neg-eigenvalue fraction (3 measured params) | 0.505 |
| writer role alone | 0.380 |
| fan-in + fan-out alone | 0.226 |

**Nothing comes close.** Neither architecture-only descriptors (free, known before training) nor
measured dynamical descriptors (requiring a probe) recover what the label carries. **The type
effect is primitive with respect to every quantity this campaign can compute.**

**What that means, stated plainly.** The answer to "what causes the difference in C between
layers" is currently: *~22% is gradient scale via a Gauss-Newton law; ~8% is position at the
residual-stream boundary; ~53% is an irreducible per-role constant we can measure but cannot
derive; ~11% is real structure we have not identified.* **This is a partial answer, and the
largest term is the one we understand least.**

**What would move it.** Reducing the type term needs a quantity that distinguishes q from k from
v — plausibly the input activation statistics each type sees (never measured; iteration 22's
proposed forward-pass probe), or the attention-specific structure that makes q and k
multiplicatively coupled while v is not. Neither is recoverable from committed data. **This is the
strongest remaining argument for the activation measurement**, above any further LR-rule
refinement: it targets the 53% rather than the 8%.

**Registered seed check (zero cost):** the variance budget above must reproduce within ±10
percentage points per row in every seed. If the type share collapses across seeds, type is a
per-network artifact rather than an architectural constant — which would be a major result and
would also invalidate the per-type LR rules.

**=== FINAL STATE OF THE OFFLINE CAMPAIGN (2026-09-03, after 43 iterations) ===**

*Read this block and the variance budget below it; everything further down is chronological
provenance containing superseded claims. Every number here was re-derived from the raw committed
JSONs in a final verification pass — 7 of 7 checks passed.*

**ESTABLISHED — verified, replicated at two fork states, non-circular:**

| # | claim | value | evidence |
|---|---|---|---|
| 1 | measurement noise floor | **0.101 dex** | duplicate arms, identical fork/s/age |
| 2 | cross-matrix spread of log C | **0.379 dex** | 72 matrices |
| 3 | **response ratio d log λ / d log g** | **+2.07 / +2.10** | pooled 2SLS on REQ-023's per-matrix LR randomisation; robust to weak-instrument cuts; placebo on polar curvature +1.80 |
| 4 | **q,k versus the other four types** | **+0.812 / +0.842 dex** | 12/12 blocks at **both** forks, Cohen's d ≈ 3.3 |
| 5 | position steps are unequal | +0.695 vs +0.234 dex | rejects the ordinal reading (t ≈ 9) |
| 6 | REQ-036 per-type LR rule | **SNR 12.8** | cross-state prescription corr +0.998 |
| 7 | instrument caveat | median tail 0.024 | the geometric-tail correction is **not** applied in committed data |

**THE ANSWER, in variance terms:** ~22% gradient scale (the Gauss-Newton response ratio, the only
*derived* law here) · **~50% a q,k-versus-rest binary of +0.81 dex that is statistically
overwhelming and mechanistically unexplained** · ~8% residual-stream boundary position · ~20%
unexplained, of which ~0.10 dex is noise.

**=== ITERATION 45: A SECOND ARCHITECTURAL CANDIDATE — MUON GROUP RANK, slope −1.04 ===**

*Reading `ebf53cd` further revealed that the six "types" are not six independent parameter groups.
They are **three banks**, and Muon orthogonalises each bank's groups at different shapes:*

| bank | tensor | Muon groups | group shape | **rank** | shape_mult |
|---|---|---:|---|---:|---:|
| **qk_bank** | (64, 128, 768) | 60, **per head-PAIR** | 128×768 | **128** | 1.0 |
| vo_bank | (24, 768, 768) | 20, per layer | 768×768 | 768 | 1.0 |
| mlp_bank | (24, 3072, 768) | 24, per matrix | 3072×768 | 768 | 2.0 |

**q and k are not separate parameters at all** — they share `qk_bank` and are orthogonalised over
a **six times smaller subspace** than v/o. Muon's polar factor has ‖O‖_F = √rank, so the same
learning rate spreads each step over 128 directions for q,k versus 768 for everything else.

**The quantitative result.** Regressing the adjusted level on log(Muon group rank):

| | slope | R² (1 param) | QK binary R² | type label R² (5) |
|---|---:|---:|---:|---:|
| fork-1500 | **−1.044** | 0.737 | 0.737 | 0.798 |
| fork-2000 | **−1.083** | 0.687 | 0.687 | 0.739 |

**The slope is −1.0 to within 8%, at both forks.** *(Iteration 46 WITHDRAWS the mechanistic
reading of this — see the correction below. The slope is an artifact of perfect collinearity with
the QK binary; log(rank) took only two values, so the regression had no independent content.)*

**Honest statement of what this is and is not.** Rank and the QK binary are **perfectly collinear
in these data** — only q and k have rank 128 — so R² is identical (0.737 / 0.687) and **this
analysis cannot distinguish them.** What rank adds over the binary is: (i) it is a *continuous
physical quantity* with a derived exponent rather than a label, (ii) the fitted exponent lands on
the predicted −1, and (iii) it is a property of the **optimizer**, not the loss — a different kind
of explanation from every previous candidate.

**How to separate the two architectural candidates.** QK-norm (iteration 44) is a property of the
**loss**; Muon rank is a property of the **update**. They differ sharply on one intervention:
- **change the QK bank's grouping** (orthogonalise per layer instead of per head-pair, rank 128 →
  768) **while leaving QK-norm in place.** Rank predicts the QK gap **collapses**; QK-norm predicts
  it is **unchanged**. This is a config change, not a code change — the banking is set at
  construction.
- **remove QK-norm while leaving the banking**: the mirror test.

**Registered prediction:** re-grouping qk_bank to per-layer (rank 768) shrinks the QK-vs-rest
adjusted-level gap from +0.81 dex to **below +0.25 dex** if rank is the mechanism, and leaves it
**above +0.65 dex** if QK-norm is.

**And a caution that applies to BOTH candidates.** mlp.fc and mlp.proj are both rank 768 and both
unnormalised, so **neither mechanism predicts any gap between them — yet iteration 39 measured
+0.294 dex at t = 3.8, replicated at both forks.** Whatever produces that ~0.29 dex is a third
effect that neither architectural candidate covers. **Any claim that rank or QK-norm "explains the
type structure" is therefore incomplete by construction**, and should be stated as explaining the
QK-vs-rest split specifically.

**Zero-cost addition to Arm A:** report the adjusted level against log(Muon group rank) per seed;
the slope must be **−1.0 ± 0.3** in every seed for the rank reading to hold.

**=== ITERATION 47: PROBE AUDIT — a depth-axis mislabelling found; the boundary field survives ===**

*Two mechanisms died in iterations 45–46 on errors traceable to not checking what the probe
measures. Before proposing anything further, I audited the probe's own labels against `ebf53cd`.*

**The mislabelling.** The EoS trainer defines `num_attn_layers = num_layers - 1` and skips
attention at layer 6 (`attn_weights[i - (i > 6)]`, and the comment "skip on layer 6" is present in
the pinned commit itself). So there are **11 attention layers**. But the probe reports **12
attention blocks including block 6**, whose curvature is entirely normal (z = +1.14 / +0.05 versus
the other eleven — not padding, not degenerate).

**Therefore the probe's `block` index is a POSITION IN THE PARAMETER BANK, not a network layer.**
For attention matrices the true layer is `slot` for slot < 6 and `slot + 1` for slot ≥ 6. Every
depth-indexed result in iterations 7–25 used the bank slot as if it were network depth.

**Re-testing the boundary field with the corrected index:**

| index | fork-1500 block-level corr | fork-2000 | perm p |
|---|---:|---:|---:|
| bank slot (as used in iters 7–25) | −0.893 | −0.878 | < 0.0001 |
| **true network layer (corrected)** | **−0.912** | **−0.891** | **< 0.0001** |

**The finding survives and slightly strengthens.** The boundary field is not an artifact of the
mislabelling — correcting the axis improves it at both forks. Matrix-level correlations are
essentially unchanged (−0.606 → −0.606, −0.673 → −0.678).

**Why this matters even though nothing changed.** This was a real risk: a mechanism built on a
misaligned depth axis would have been unfalsifiable from inside the analysis, and two of the last
three mechanisms died on exactly this class of error (iteration 46: the probe measures 768×768
matrices while Muon updates 128×768 groups). **The boundary result is now verified against the
actual architecture rather than an assumed one**, which it never was before iteration 44.

**Two corrections to the record for anyone reading the earlier iterations:**
1. Iteration 8 tested "is block 6 an outlier?" as a check on whether the layer-6 attention skip
   confounded the boundary field, and concluded it did not. **That conclusion stands, but the
   reasoning was luckier than it looked** — block 6 in the probe is not network layer 6.
2. All depth-indexed results should be read as **bank-slot indexed**. For MLP matrices slot =
   layer (mlp_bank has all 12), so only attention matrices at slots ≥ 6 are shifted.

**Registered addition to Arm A (zero cost):** report the matrix-name-to-network-layer mapping
explicitly in the output JSON. **This campaign spent 40 iterations without knowing whether its
depth axis was correct**, and the fix is one field in the probe.

**=== ITERATION 49: ARM 3 REVISED — its multipliers were not determined by the data ===**

*Seven mechanism retractions later, I audited whether the REQ-036 arms still stand. Arm 3 was
justified in iteration 19 by a "projection-matrix" reading that iteration 39 falsified. A design
can be right when its stated reason is wrong, so the question is whether the empirical effect
survives — and whether the numbers are determined.*

**The effect survives.** The is_proj × is_end interaction is **+0.805 ± 0.215 (t = +3.75)** at
fork-1500 and **+0.763 ± 0.212 (t = +3.60)** at fork-2000. Proj matrices at blocks 0 and 11 really
do sit ~0.94–1.01 dex above their interior counterparts. **Arm 3's direction is sound.**

**The multipliers are not.** Bootstrapping each per-type end-block factor over its **n = 2**
end-block matrices:

| type | filed value | bootstrap median | **95% CI** |
|---|---:|---:|---|
| **mlp.proj** | **6.71** | 9.58 | **[2.72, 31.86]** |
| attn.proj | 1.72 | 2.92 | [1.42, 5.96] |
| **mlp.fc** | **1.04** | 2.27 | **[1.44, 3.60]** ← filed value is *outside* its own CI |
| attn.v | 1.43 | 1.28 | [0.96, 1.72] |
| attn.q | 1.35 | 1.01 | [0.88, 1.18] |
| attn.k | 1.01 | 1.02 | [0.87, 1.21] |

**mlp.proj's interval spans an order of magnitude**, and mlp.fc's filed value falls outside its own
interval. Pooling both proj types (n=4) only narrows it to [2.38, 20.24] / [3.27, 20.58]. **These
are two-matrix estimates driving a 6x learning-rate change.**

**Revision.** Arm 3 now applies a **single pooled, capped ×3.0** to proj types at blocks 0 and 11
(attn.proj 1.20, mlp.proj 3.00) and leaves non-proj types at their arm-2 values. Rationale:
- ×3.0 sits inside every bootstrap CI for the pooled proj factor, at its lower end;
- a cap bounds the damage if the effect is smaller than estimated, while still testing the
  direction, which is what has t > 3.5 support;
- the four non-proj end-block adjustments are all within noise of 1.0 and are dropped as
  unsupported.

**Registered prediction, revised:** arm 3 beats arm 2 by **0.0002–0.002** val (halved from the
original band, reflecting the smaller intervention). **If arm 3 loses to arm 2, the end-block
correction is refuted and arm 2 stands as the design.**

**Seed check (zero cost):** the pooled proj end-block factor must be **≥ 2.0 with a 95% CI
excluding 1.0** in every seed. If it does not clear 1.0, the end-block term should be dropped
entirely and REQ-036 reduces to the per-type rule.

**A note on process.** This is the second time an amendment built on a since-falsified mechanism
needed walking back (the first was the uniform is_end term in iteration 39). Both survived as
*directions* and failed as *magnitudes*. The pattern suggests the campaign's per-type contrasts are
reliable while its per-block-per-type contrasts — always n=2 — are not, and future amendments
should carry bootstrap CIs before being filed rather than after.

**=== ITERATION 48: QK-NORM FALSIFIED TOO. Seven mechanisms, seven failures. ===**

*QK-norm was the last architectural candidate standing. Its sharpest consequence is testable with
only λ and g, and it fails.*

**The prediction.** For an exactly scale-invariant matrix, `g(cW) = g(W)/c` and `λ(cW) = λ(W)/c²`,
so **`λ/g²` is exactly gauge-invariant** — it cannot change under any rescaling of W. The adjusted
level *is* log(λ/g²). Under QK-norm, q and k's adjusted level therefore has a whole degree of
freedom removed that the other four types retain, and should be **more stable**: flatter in s, less
scattered, more reproducible.

**All three tests are null, and two point the wrong way:**

| test | q,k | others | t |
|---|---:|---:|---:|
| \|d adj / d log s\| (fork-1500) | 0.452 | 0.438 | +0.20 |
| \|d adj / d log s\| (fork-2000) | 0.396 | 0.437 | −0.56 |
| sd over the s ladder (fork-1500) | 0.1228 | 0.0968 | **+1.80 (wrong direction)** |
| sd over the s ladder (fork-2000) | 0.0973 | 0.1102 | −0.85 |
| median cross-state \|shift\| | 0.0341 | 0.0361 | — |

q,k are if anything *more* scattered, and every sign flips between forks.

**The null is informative, not underpowered — checked explicitly.** Minimum detectable difference
at 80% power is **0.0405**; the observed difference is **0.0260**, i.e. 0.64× MDE and in the wrong
direction. More importantly the **gauge degree of freedom is wide open**: weight norms move
**0.188 dex (54%) within a run** over the measured steps, so a genuine scale-invariance would have
had ample room to show. It does not.

**QK-norm is falsified as an explanation of the +0.81 dex gap.** It remains a true architectural
fact — q and k *are* RMS-normed and *are* scale-invariant in their own output — but that invariance
**does not produce the curvature gap**.

**SEVEN MECHANISMS, SEVEN FAILURES:** bilinear coupling · softmax saturation · nonlinearity
exposure · consumption order · curvature concentration · Muon group rank · QK-norm scale-invariance.

**This changes what should be concluded.** After seven falsifications on the same 72 matrices, the
honest position is not "the mechanism is still out there" but:

> **The q,k-versus-rest gap of +0.81 dex is the most statistically robust fact in this campaign
> (12/12 blocks, both forks, Cohen's d ≈ 3.3) and is not explained by any property of the loss
> geometry, the optimizer geometry, the architecture, or the spectrum that is measurable in the
> committed data.**

**No eighth mechanism should be proposed offline.** Seven hypotheses against one fixed dataset is
well past the point where a new one would be discovery rather than overfitting. The gap now needs
either (a) REQ-038's forward/backward probe, or (b) the two registered interventions — the W_q/W_k
rescale and the qk_bank re-grouping — both of which are binary tests rather than correlational fits.

**Registered addition to Arm A (zero cost):** report the sd of the adjusted level over the s ladder
per type. If q,k come out *less* scattered than the other four in ≥3 of 4 seeds, this iteration's
disconfirmation was driven by the two-state sample and QK-norm is rehabilitated.

**=== ITERATION 46: THE RANK ANALYSIS IS WITHDRAWN — IT TESTED THE WRONG OBJECT ===**

*Iteration 45 proposed Muon group rank as a mechanism, with a slope of −1.044/−1.083 on
log(rank) that appeared to land on a derived −1. Testing it against the polar curvature — the
quantity Muon actually experiences — breaks it.*

**The prediction and the result.** If a fixed Muon step is spread over 128 directions instead of
768, the mean curvature over that subspace should be **6x higher** for q,k. Measured
`curvature_along_polar`:

| | q,k | others | ratio | **predicted** |
|---|---:|---:|---:|---:|
| fork-1500 | 27.29 | 72.42 | **0.38x** | 6.0x |
| fork-2000 | 25.58 | 76.92 | **0.33x** | 6.0x |

**Wrong direction and wrong magnitude — off by a factor of ~16.** And normalising by rank makes
the cross-type spread *worse*, not better: log10(cp·rank) has sd **0.630 / 0.656 dex** versus
**0.286 / 0.312** for cp alone.

**The root cause — a measurement/optimizer mismatch I should have checked first.** The curvature
probe reports every attention matrix as **768×768**: it measures the **full per-layer matrix**.
Muon orthogonalises the **128×768 per-head-pair group**. **These are different objects.** Every
matrix in the probe has min-dimension 768, so **there is no rank variation in the measured
quantity at all** — the +0.81 dex gap is a property of matrices measured at *identical* rank.

**Consequences, stated plainly:**
1. The 1/rank test above was applied to the wrong object and is **void**.
2. Iteration 45's slope of −1.04 is **withdrawn as evidence**. log(rank) took exactly two values
   and was perfectly collinear with the QK binary (identical R², 0.737/0.687), so the regression
   never had independent content — the "derived exponent" was the binary's gap divided by
   log(768/128) = 0.78, which arrives near −1 by arithmetic coincidence, not physics.
3. **The Muon group rank remains a real architectural fact** — q and k genuinely are orthogonalised
   over a smaller subspace — but this campaign has produced **no evidence** that it drives the
   curvature gap, and one measurement pointing the wrong way.

**What is unaffected.** The separating intervention registered in iteration 45 is still the correct
test and is *strengthened* by this: re-grouping `qk_bank` to per-layer changes the **optimizer
group** while leaving the **measured object** identical at 768×768, so it isolates the rank channel
cleanly. Registered prediction stands: gap falls below +0.25 dex if rank matters, stays above
+0.65 dex if it does not. Given the polar result above, **I now expect the latter.**

**The count is now six proposed mechanisms for the QK gap, six withdrawn or falsified** — bilinear
coupling, softmax saturation, nonlinearity exposure, consumption order, curvature concentration,
and Muon group rank. QK-norm (iteration 44) is the only architectural candidate still standing, and
its own direct evidence is weak (slope test inconclusive, dispersion test retracted, drift test
t ≈ 2). **The honest position remains that the +0.81 dex q,k-versus-rest gap is the most robust
fact in the campaign and has no established mechanism.**

**=== ITERATION 44: THE ARCHITECTURE RECOVERED — QK-NORM IS THE ASYMMETRY ===**

**The pinned EoS commit `ebf53cd` has been recovered.** Iteration 8 recorded that it was absent
from the clone, so every structural claim in this campaign rested on *assumed* architecture. A
full-refs fetch (`git fetch origin '+refs/*:refs/remotes/origin-all/*'`) resolved it. The
architecture behind all 45 curvature measurements is now readable, and it contains a hard
asymmetry hitting exactly the two matrices with the +0.81 dex gap:

> **`train_gpt.py` line 1106 at `ebf53cd`: `q, k = norm(q), norm(k)  # QK norm`**
> where `norm` is `F.rms_norm` (line 952). **Applied to q and k only.** v, attn.proj, mlp.fc and
> mlp.proj have no output normalisation.

**Why this matters.** QK-norm makes q and k **exactly scale-invariant in their own output**:
multiplying W_q by any constant leaves the loss unchanged, because the RMS-norm divides it out.
The other four matrices have no such invariance. **This is a genuine, hard architectural
distinction that separates {q,k} from the rest — the exact partition of the campaign's largest
unexplained effect — and it was not among the five mechanisms tested, because the architecture
could not be read.**

**Evidence status — honest, and weaker than the discovery deserves.**
- *Slope test (log C vs log‖W‖, predicted −2 for scale-invariant matrices):* **inconclusive.**
  −2.669 at fork-1500 but +1.811 at fork-2000, |corr| < 0.16 in both. No power at n=24 with the
  available norm range.
- *Dispersion test:* **negative.** I first read QK norms as 4x tighter (sd 0.007–0.011 vs
  0.014–0.048), but that used a single-step slice and conflated across-block with across-step
  variation. Pooled correctly, **every type has sd ≈ 0.073–0.080 — ratio 1.0x, no separation.**
  Retracted.
- *Drift test:* **marginal support.** QK norms drift less over training than the others
  (−0.0008 vs −0.0037 per 1000 steps at fork-1500, t = +1.97; +0.0281 vs +0.0252 at fork-2000,
  t = +2.11). Consistent with only weight decay acting on an invariant norm, but t ≈ 2 on 72
  matrices is not strong.

**So: a compelling architectural candidate with weak direct evidence.** It is the first proposed
mechanism that (a) partitions the types exactly as the data does, (b) is a hard property of the
code rather than an interpretation, and (c) was not available to the previous five attempts. It is
*not* yet established — the tests that would establish it need norm variation the committed data
does not contain.

**THE DECISIVE EXPERIMENT — and it already exists in the queue.** REQ-036's **arm 4 is the
anti-rule (multipliers inverted)** and REQ-037 perturbs weight norms directly. Better still, the
clean test is a **rescale fork**: multiply W_q and W_k by 2x at a checkpoint and continue.
- **Scale-invariance predicts the loss is EXACTLY unchanged at step 0** and λ_q, λ_k fall by
  exactly 4x (‖W‖⁻²), then recover as weight decay pulls the norm back.
- **For v, attn.proj, mlp.fc, mlp.proj the same rescale changes the loss immediately.**
This is a one-line intervention on an existing checkpoint, costs one short run, and is a **binary
test of scale-invariance** rather than a correlational fit. **Registered prediction: rescaling
W_q, W_k by 2x changes the training loss by < 0.001 at the first step; the same rescale on
mlp.fc changes it by > 0.01.**

**Added to the REQ-035 Arm A seed check (zero cost):** report per-type log‖W‖ drift. QK drift must
remain smaller than the non-QK types' in ≥3 of 4 seeds; if it does not, the scale-invariance
reading loses its only supporting evidence.

**FIVE MECHANISMS PROPOSED FOR THE QK GAP; ALL FIVE FALSIFIED.**

| mechanism | iteration | how it died |
|---|---:|---|
| bilinear q·k coupling | 34 | residual correlation ranks q~k *lowest* at fork-2000 and on REQ-023 |
| softmax saturation | 35→36 | negfrac separates paths perfectly but does not order the levels |
| nonlinearity exposure | 37→39 | attn.v − mlp.fc is a cross-group pair with **zero** separation (−0.06 dex) |
| within-block consumption order | 40→41 | steps unequal by 3x (t ≈ 9); reduces to the QK binary at 94% |
| curvature concentration | 24 | explains only ~20% of the residue (sub-top mass rises almost as fast) |

**The pattern is the finding.** Each mechanism looked strong at the population level and died on
its own *extra* content, while the QK gap survived every test untouched. **No further offline
mechanism search is warranted** — five hypotheses on one fixed 72-matrix dataset is already past
the point where a sixth would be overfitting.

**RETRACTED CLAIMS (kept so they are not rediscovered):** anisotropy λ_top/λ_grad, C_grad,
spectral scale, and the offset b all "explained" C brilliantly and are all algebraically derived
from λ_top (`b ≡ −2 log R` at corr 0.9985). `curvature_along_gradient` **is exactly α₁**, the
Lanczos start vector. The corr(log g, log R) = +0.66 cancellation argument sits *below* its own
mechanical null of +0.782. η²(b) ≫ η²(C) is forced by construction. The polar target's SNR of 41.3
(iteration 27) was an aggregation error; the honest figure is 11.4 against λ_top's 13.1.

**WHAT THE QUEUED RUNS DECIDE:**
- **REQ-038** (1 forward+backward pass, cheapest in the queue) — amended P5: if conditioning on
  |a| and |d| does not cut the QK binary's coefficient by ≥50%, the gap is irreducible to
  first-order backward statistics, and given five falsifications that is close to a terminal
  answer.
- **REQ-035 Arm A** (n=4 seeds) — decides whether every finding above is architectural or a
  property of this particular network. **Primary band: the QK gap must be +0.81 ± 0.20 dex with
  q,k above in ≥10 of 12 blocks per seed.**
- **REQ-036 / REQ-037** — the LR design and the exclusion-restriction test for claim 3.

**Rule adopted mid-campaign and worth keeping:** any candidate predictor built from the same
Lanczos tridiagonal as λ_top is circular; check |corr| against every previously-defined derived
quantity before claiming novelty — >0.99 means it is a rename.

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


---

**SESSION CLOSED (2026-09-03 ~08:45 PDT).** The 8-hour analysis window ended with no node freed —
REQ-032's tau2 arms held both boxes throughout, so **none of REQ-034/035/036/037/038 ran**. All
five remain ACCEPTED and queued; nothing about them has changed except that their registered bands
are now consolidated and unambiguous (this table for REQ-035, the authoritative arm table for
REQ-036).

Everything in this file is offline analysis of already-committed data. **No claim here has been
tested against a new training run.** The single highest-value action remains REQ-035 Arm A, whose
band 1 decides whether the campaign's largest finding is architectural or an artifact of one
trained network. REQ-038 is the cheapest at one forward+backward pass.

A plain-language write-up of the findings, aimed at a reader who has not followed the campaign, is
at `~/ML/layerwise-momentum/C_FINDINGS.md`.

**=== ITERATION 52: MAJOR CORRECTION — THE q/k GAP IS NOT IN C ===**

*The campaign's headline finding has been mis-stated since iteration 33. Checked directly:*

| quantity | q,k | other four | gap | blocks |
|---|---:|---:|---:|---:|
| **log C — the actual target** | +4.117 | +4.110 | **+0.007 dex** | **6/12** |
| log g | +3.529 | +3.931 | **−0.403 dex** | **0/12** |
| adjusted (log C − 2 log g) | −2.940 | −3.753 | +0.812 dex | 12/12 |

**In C itself the q,k gap is +0.007 / +0.029 dex — indistinguishable from zero, and the
direction splits 6/12 and 7/12 blocks.** The QK binary explains **R² = 0.000** of the variance in
log C. The +0.81 dex figure quoted throughout exists *only* in the gradient-adjusted quantity.

**What is actually anomalous is the GRADIENT.** q and k receive **0.40 dex (~2.5×) less gradient**
than the other four types, with perfect separation (0/12 blocks — every block, both forks).

**The corrected statement of the anomaly.** Under the verified λ ∝ g² law, a matrix with 2.5× less
gradient should sit ~6× flatter. q and k receive far less gradient **yet settle at the same
sharpness as everything else.** So the finding is not "C has unexplained structure" — it is:

> **The λ ∝ g² law, which holds causally within a matrix and cross-sectionally for the other four
> types, FAILS for q and k.**

Cross-sectional slope of log C on log g, by group:

| group | fork-1500 | fork-2000 |
|---|---:|---:|
| **other four types** | **+2.238** | **+2.083** |
| **q, k** | **+1.083** | **+0.508** |

The other four sit right on the causal exponent of 2.0. **q and k are at half that or less.**

**What this invalidates.** Roughly 20 iterations framed the target as "a +0.81 dex gap in C with no
mechanism," and seven mechanisms were proposed and falsified against that framing. **Those seven
falsifications stand as tests of that quantity** — but the quantity was the wrong one. The gap they
were trying to explain is a property of the *adjustment*, not of C.

**What survives unchanged:**
- the λ ∝ g² law itself (within-matrix causal exponent +2.07/+2.10) — **strengthened**, since it
  now cleanly explains four of six types cross-sectionally too;
- the variance budget's other rows: gradient ~22%, boundary position ~8%;
- REQ-036's LR design, which is built from measured C and k and never used the adjusted quantity.

**What changes.** The "unexplained ~50%" is not an unexplained property of C. It is the **failure
of the g² law for q and k specifically** — a sharper and more tractable question, and one that
points directly at what q,k *do* with their gradient rather than at their curvature.

**REVISED BAND 1 (replaces the version in the table below).** The registered check must target the
real effect:
- **q,k gradient deficit: −0.40 ± 0.12 dex, with q,k below the other four in ≥10 of 12 blocks.**
- **cross-sectional slope of log C on log g: ≥ +1.8 for the other four types, ≤ +1.4 for q,k**, in
  every seed.
- **q,k gap in log C itself must stay below 0.15 dex.** If it opens up in the seeds, this
  correction was wrong and the original framing returns.

**Process note.** This was caught by asking whether the headline was even about the right variable
— a check that should have run at iteration 33, not 52. Every derived quantity in this campaign
now carries a documented failure of this kind (α₁, the tail correction, the depth axis, the Muon
rank object, and now the adjusted level). **The lesson is to state findings in the units of the
question before building on them.**

**=== ITERATION 53: THE ANOMALY, CORRECTLY NAMED — it is the effective residual scale ===**

*Iteration 52 established that the q,k anomaly is not in C but in the gradient. This iteration
finds that the algebra already names what the anomaly is — and I am flagging it under the
campaign's own rule as a **rename, not a new mechanism**.*

**The algebra.** The Gauss-Newton structure gives `H ≈ JᵀJ` and `g = Jᵀr`, so `λ ~ |J|²` and
`|g| ~ |J||r|`. Therefore:

```
λ / g²  =  1 / |r|²
```

**The "adjusted level" I have been calling unexplained for twenty iterations IS the effective
residual scale**, up to a factor of −½. `log|r| = log g − ½ log λ`, and the correlation with the
adjusted level is **−1.0000 exactly**, with `log|r| + ½·adjusted = 0.00e+00`. **By the campaign's
hard rule (|corr| > 0.99 means a rename), this is a rename and must be declared as one.**

**What it nonetheless buys — and why I am recording it rather than discarding it:**

1. **It names the quantity correctly.** "C has unexplained structure" was simply wrong (iteration
   52: the C gap is 0.007 dex, R² = 0.000). "q,k see a 2.5× smaller effective residual" is the same
   number, correctly attributed to a named physical object.
2. **It makes a directional prediction the old framing did not.** A bounded softmax Jacobian must
   *shrink* the gradient reaching the attention logits, so |r| for q,k must be **smaller**.
   Observed: **0.39× and 0.38×, in 12/12 blocks at both forks.**
3. **It explains the seven failures.** Every mechanism was asked why C is *higher* for q,k. C is
   not higher for q,k. The right question — why the residual is smaller — has an obvious candidate
   (the softmax Jacobian) that was **never tested against it**, because iteration 36 tested softmax
   saturation against *levels* and correctly rejected it there.

| type | log₁₀ \|r\| (fork-1500) | (fork-2000) |
|---|---:|---:|
| **attn.q** | **1.427** | **1.410** |
| **attn.k** | **1.513** | **1.493** |
| mlp.fc | 1.802 | 1.809 |
| attn.v | 1.833 | 1.822 |
| attn.proj | 1.921 | 1.906 |
| mlp.proj | 1.949 | 1.955 |

**THE SHARP PREDICTION FOR REQ-038 — this is the payoff.** REQ-038 measures the input activation
|a| and the backward tensor |d| per matrix. **q, k and v read the same residual vector, so their
|a| is identical by construction** — any gradient difference must appear entirely in |d|. Under
this reading:

> **|d|(q,k) / |d|(other four) must come out at 0.39 ± 0.08.**

That is a *quantitative* prediction from committed data to a direct measurement, not a
correlational fit. **If |d| comes back near 0.39, the campaign's central anomaly is closed:** the
gradient law holds universally, and q,k's apparent violation is the softmax Jacobian shrinking what
reaches the logits. **If |d| is near 1.0, the deficit is in |a| instead — which would contradict
q,k,v sharing an input and would mean the probe or my reading of the architecture is wrong.**

**Registered addition to the seed table (zero cost):** the implied |r| ratio must be
**0.39 ± 0.10 with q,k below the other four in ≥10 of 12 blocks** in every seed.

**Status, stated plainly.** This is the eighth framing of the same number and the fifth rename this
campaign has produced. It is not evidence and adds no measurement. **What makes it worth recording
is that it converts an unexplained residual into a falsifiable prediction about a quantity REQ-038
already measures** — and that prediction is specific enough to be wrong.

**=== ITERATIONS 54–55: TWO FINDINGS UNIFY INTO ONE SPATIAL EFFECT ===**

**A failed prediction first.** If the softmax attenuates the backward signal for q,k, the
attenuation should deepen with depth (attention sharpens with depth, and a sharper softmax has a
smaller Jacobian). Tested: corr(depth, residual deficit) = **−0.184 / −0.210, permutation
p = 0.56 / 0.51.** Essentially flat. **The softmax-sharpening story gains nothing beyond its
sign** — it predicts the direction of the q,k deficit but not its depth structure.

**But the deficit profile is not flat randomly — it is the boundary shape.** The q,k residual
deficit is *smallest* at blocks 0 and 11 and largest mid-network:

| | edge blocks (0, 11) | interior | difference |
|---|---:|---:|---:|
| fork-1500 | −0.239 | −0.440 | **+0.201 dex** |
| fork-2000 | −0.265 | −0.453 | **+0.188 dex** |

**The strict test — no shared numbers.** The boundary field (iterations 7–25) was built from the
adjusted level, which is −2·log|r|, so correlating it against the q,k deficit shares construction
and inflates. Rebuilding the boundary field from the **non-q,k types only**, so the two quantities
share no underlying numbers:

| | corr(q,k deficit, boundary field) | permutation p |
|---|---:|---:|
| fork-1500 | **+0.876** | **0.0003** |
| fork-2000 | **+0.710** | **0.0067** |

**These are one spatial effect, not two.** The "boundary field" (~8% of the variance budget) and
the q,k residual deficit are the same pattern measured on different subsets of matrices. **The
campaign has been tracking them as independent findings since iteration 7; they are not.**

**What this changes.** The count of independent phenomena drops. The corrected picture is:

1. **λ ∝ g²** — the gradient law, causally verified, holds for four of six types cross-sectionally;
2. **a single spatial field** — matrices near the residual-stream boundary see a *larger* effective
   residual, and q,k see a *smaller* one everywhere, with the two effects being the same field
   viewed through different matrix subsets;
3. everything else is noise or unresolved.

**Registered seed check (zero cost, replaces the separate boundary and deficit checks):** the q,k
residual deficit and the non-q,k boundary field must correlate at **≥ +0.6 across blocks in every
seed**. If they decouple, this unification was driven by the two-state sample and they should be
tracked separately again.

**Note on interpretation.** This is a unification of *descriptions*, not a mechanism — it says two
things we could not explain are one thing we cannot explain. That is still progress: it halves the
number of separate anomalies REQ-038 and Arm A need to account for, and it means the softmax
reading must explain a *spatial* pattern, not just a q,k offset.

## ⚠️ AUTHORITATIVE SEED-CHECK TABLE — READ THIS, IGNORE THE SEED CHECKS BELOW

*This request accumulated **13 overlapping "registered seed check" blocks** across ~15 iterations,
several claiming to supersede each other, and most tied to mechanisms that were subsequently
falsified (seven of them). **Only the five bands here are live.** Everything below is provenance.*

**All five reproduce at both fork states in the committed data.** Each is stated with its current
measured value so a seed result can be compared directly.

| # | band | must hold | measured f1500 / f2000 |
|---|---|---|---|
| **1** | **q,k vs other four, gradient-adjusted level** | **+0.81 ± 0.20 dex**, q,k above in **≥10 of 12 blocks** per seed | +0.812 (12/12) / +0.842 (12/12) |
| **2** | **response ratio d log λ / d log g** | **2.00 ± 0.15** per seed | +2.069 / +2.095 |
| **3** | boundary field, **true-layer axis** | corr(d_edge, block-mean residual) **≤ −0.5** | −0.912 / −0.891 |
| **4** | negative-curvature separation | every nonlinear-path type above every linear-path type | 0.198 > 0.146 / 0.200 > 0.138 |
| **5** | position spacing stays **unequal** | step(0→1) ≥ **2×** step(1→2) | 3.0× / 3.2× |

**Band 1 is the single most important number in the campaign.** It is the largest unexplained
effect (~50% of the variance in C) and has survived seven mechanism falsifications untouched. If it
reproduces across four independent seeds it is a property of the architecture and worth a dedicated
programme; **if it varies by more than ±0.3 dex across seeds it is a property of this one trained
network and the question largely dissolves.**

**Band 5 is registered as a NEGATIVE and matters as much as the positives.** Iteration 41 rejected
an ordinal "consumption order" reading because the spacing is 3× uneven. If the seeds show *equal*
spacing, that rejection was wrong and the ordinal model revives.

**Two required readouts, both zero-cost, both fixing errors this campaign made:**
- **the matrix-name → network-layer mapping**, emitted explicitly. The probe's `block` index is a
  parameter-bank slot; the model skips attention at layer 6, so for attention matrices the true
  layer is slot+1 for slot ≥ 6. **This campaign ran 40 iterations on a mislabelled depth axis.**
- **raw Ritz values plus `residual_tail`**, uncorrected. The geometric-tail correction is *not*
  applied in the committed data (median tail 0.024) and must stay reversible.

**Retired — do not score these.** Every band registered against a falsified mechanism: bilinear
q·k coupling, softmax saturation, nonlinearity exposure *as an explanation of levels*, within-block
consumption order, curvature concentration, Muon group rank, and QK-norm scale-invariance. Band 4
above survives only in its narrow form — nonlinearity predicts *negative curvature*, not the level.

---

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

---

## ⚠️ AUTHORITATIVE ARM TABLE — READ THIS, IGNORE ALL PRESCRIPTION TABLES BELOW

*This request accumulated three prescription tables across iterations 17/19/27, two of which
say "use this for arm 2" with conflicting values. **They are superseded. Only this block is
live.** Everything below is kept for provenance and must not be executed from directly.*

**Common setup:** 750-step continuations from the shared step-2000 state, val@2750, same
`PerMatrixLrMuon` machinery as REQ-023.

| arm | rule | multipliers |
|---|---|---|
| **1** | control | all 1.0 |
| **2** | per-type only | attn.proj 0.40, attn.k 0.88, mlp.fc 0.91, attn.q 1.18, attn.v 1.25, mlp.proj 1.56 |
| **3** | per-type + end-block cap ⚠️ **REVISED, see iteration 49** | arm-2 values, except at blocks 0 and 11 where **proj types (attn.proj, mlp.proj) get a single pooled ×3.0 cap**: attn.proj 1.20, mlp.proj 3.00. Non-proj types unchanged from arm 2. *(The previously filed per-type end values — attn.q 1.35, attn.k 1.01, attn.v 1.43, mlp.fc 1.04, attn.proj 1.72, mlp.proj 6.71 — are **withdrawn**: each rests on n=2 matrices and several sit outside their own bootstrap CI.)* |
| **4** | anti-rule falsifier | arm-2 multipliers inverted (1/s) |
| **5** | **polar target** (iteration 27) | attn.q 0.568, attn.k 0.755, attn.proj 0.642, attn.v 1.101, mlp.fc 1.260, mlp.proj 2.462 |

**Priority if fewer arms fit:** 1, 2, 5, 3, 4. Arms 2 and 5 test *different hypotheses* (which
curvature to equalize) and are the most informative pair; arm 4 is the cheapest falsifier; arm 3
is a magnitude refinement of arm 2.

**Safety flag — RESOLVED by the iteration-49 revision.** The original 6.71x is withdrawn; arm 3 now
uses a pooled, capped ×3.0 for proj types at the end blocks. Rationale below.

**ITERATION 31 — the single-type concentration is REAL SIGNAL, and a 2-parameter rule
nearly matches the 6-parameter one.**

*Shrinkage does not help — clean negative.* Iteration 30 flagged that the lam_top rule draws 49%
of its dynamic range from attn.proj. James-Stein shrinkage toward the grand mean was the obvious
fix. It fails on both counts:

| shrinkage | out-of-sample rms | LOBO | max single-type dependence |
|---:|---:|---:|---:|
| 0.00 (filed) | **0.2652** | **0.2500** | 49% |
| 0.50 | 0.2655 | 0.2490 | 54% |
| 2.00 | 0.2716 | 0.2545 | 52% |
| 8.00 | 0.2883 | 0.2691 | — |

Error is flat-to-worse, and shrinkage **does not even reduce the concentration** (49% → 54% →
48%). Optimum is essentially zero shrinkage. **Keep the unshrunk per-type means.**

*Why — the concentration is genuine, not noise.* Distance of each type mean from the grand mean,
in units of its own standard error:

| type | mean log C | dist. from grand | **signal/SE** |
|---|---:|---:|---:|
| **attn.v** | 4.283 | 0.170 | **3.54** |
| **attn.proj** | 3.734 | 0.378 | **3.24** |
| attn.k | 4.032 | 0.080 | 2.32 |
| attn.q | 4.202 | 0.090 | 2.30 |
| mlp.proj | 4.348 | 0.236 | 1.32 |
| mlp.fc | 4.074 | 0.038 | 0.57 |

attn.proj sits **0.378 dex below the grand mean at 3.2 SE** — it is genuinely extreme, not noisily
estimated, which is exactly why shrinkage should and does leave it alone. **The 49% concentration
is a feature of the physics, not a fragility of the fit.** This retires iteration 30's caution.

*Consequence — the rule is simpler than six parameters suggest.* If attn.proj carries half the
range because it is genuinely 0.4 dex low, the rule is essentially *"lower attn.proj a lot, adjust
everything else mildly"*. Tested directly:

| rule | params | out-of-sample rms | LOBO |
|---|---:|---:|---:|
| 6-type (filed) | 6 | 0.2652 | 0.2500 |
| 3-group (attn.proj / mlp.proj / rest) | 3 | 0.2701 | 0.2536 |
| **2-group (attn.proj / rest)** | **2** | 0.2743 | **0.2485** |

**A 2-parameter rule matches the 6-parameter rule** — marginally *better* on leave-one-block-out
(0.2485 vs 0.2500), marginally worse out-of-sample (0.2743 vs 0.2652). The four extra parameters
buy ~0.009 dex, well inside the 0.10 dex noise floor.

**OPTIONAL SIMPLIFICATION for the operator — not a change to the authoritative table.** If a
minimal-risk arm is preferred, `attn.proj = 0.40, everything else = 1.06` captures nearly all the
measurable benefit with two parameters instead of six, and cannot be distorted by any single
mismeasured type except attn.proj itself. **I am not recommending it over arm 2** — the 6-type rule
is better out-of-sample and already filed — but it is the more defensible rule if the run is
treated as a first test of the concept rather than a tuned deployment.

**Registered seed check (zero cost):** attn.proj's signal/SE must exceed 2.0 in every seed. If it
does not, the concentration *is* noise after all, shrinkage becomes correct, and both the 6-type
and 2-group rules should be refit with it.

**Registered predictions** (magnitudes, per the REQ-019 lesson):
- arm 2 beats arm 1 by 0.001–0.006 val; arm 3 beats arm 2 by 0.0005–0.003; **arm 5 beats arm 2 by
  0.0005–0.003**.
- arm 4 is *worse* than arm 1 by a comparable margin. **If arm 4 also beats control, the mechanism
  claim is dead** regardless of which other arm wins.
- **Expect smaller gains than the within-design numbers imply** — iteration 21 showed ~71% of
  cross-experiment variation is irreducible. A null is not a refutation of the curvature findings,
  only of their transfer to a different intervention design.

**Required readouts:** per-matrix curvature at the final checkpoint for arms 1–3 and 5 (to verify
the intervention actually equalized what it targeted); per-type drift over the final window
(iteration 14 — **mlp.fc is the least trustworthy multiplier**); block 11 reported separately
(iteration 18 — worst baseline error of any block).

---

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

**[SUPERSEDED — DO NOT USE] iteration-19 role-split table, kept for provenance.** Its values now live in the authoritative arm table at the top. Original text: base per-type
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

**OVERFITTING AUDIT (iteration 20) — the refinements are real, but the rule is weaker than
the within-experiment numbers suggest. Both halves matter.**

I refined this prescription four times on the same 72 matrices (per-type → 6-level edge
[rejected] → uniform is_end → is_end × is_proj). Each step was validated by leave-one-block-out,
but **LOBO holds out blocks, not the decision to add the term** — the model selection itself was
never held out. Two honest tests:

*(a) True holdout within REQ-019* — every parameter fit on fork-1500, scored on fork-2000:

| model | fit@1500 (LOBO) | scored@2000 | gap |
|---|---:|---:|---:|
| per-type | 0.2500 | 0.2661 | +0.0162 |
| + uniform is_end | 0.2099 | 0.2073 | −0.0026 |
| + is_end × is_proj | 0.1978 | **0.1720** | **−0.0258** |

**The gap goes *negative* as complexity rises.** Overfitting produces the opposite signature, so
the refinements are not fitting fork-1500 noise.

*(b) Fully independent holdout* — fit on REQ-019 (global s-ladder), scored on **REQ-023**
(per-matrix LR randomisation, a different design that was never used to build any version of
this rule):

| model | rms on REQ-023 |
|---|---:|
| per-type | 0.5423 |
| + uniform is_end | 0.5157 |
| **+ is_end × is_proj** | **0.4784** |

**The ranking is preserved across experimental designs.** The refinements are real.

**But the absolute performance is much worse out of design.** Against a target spread of
**0.5705 dex**, the best rule achieves 0.4784 — capturing only **~30% of the variance**, versus
~80% within REQ-019 (0.198 against a 0.44 spread). **The per-type + end-block rule transfers in
*ordering* but loses most of its accuracy when the experimental design changes.**

*Why this matters for REQ-036, stated plainly.* The prescription is derived under one design
(global s-ladder, fork-1500/2000) and will be *applied* under another (constant per-matrix
multipliers during training). The REQ-023 holdout is the closest available analogue of that
shift, and it says to expect **substantially less benefit than the within-design numbers imply**.
The registered prediction (revised beats original by 0.0005–0.003 val) should be read as
optimistic; **a null result would not falsify the curvature findings, only the assumption that
they transfer to a different intervention design.**

*Parameter budget, for the record.* With ICC = +0.378 the 72 matrices are ~25 effective
independent units: per-type uses 12 parameters (2.1 units/param), +is_end 13 (1.9),
+is_end×is_proj 15 (1.7). All three are at or below the ~2 units-per-parameter guideline, so
**no further refinement should be attempted on this dataset** — the next term, however good it
looks in LOBO, would not be trustworthy. Any further model development needs REQ-035's seeds
for additional independent units.

**TRANSFER AUDIT (iteration 21) — CORRECTION: iteration 20's pessimism was overstated.**

Iteration 20 reported that the rule captures "only ~30% of the variance" on the REQ-023 holdout
(0.478 dex against a 0.571 spread) and warned to expect little benefit. **That framing was wrong
because it had no ceiling to compare against.**

*The control that was missing.* Predict REQ-023's prescription from **REQ-019's own per-matrix C**
— i.e. the best any rule could possibly do when transferring between these two experiments:

| | rms on REQ-023 |
|---|---:|
| **ceiling** (REQ-019's per-matrix C, no rule at all) | **0.4584** |
| our type + end-block rule | 0.4784 |
| **remaining headroom** | **0.020 dex** |

**The rule is within 0.02 dex of the theoretical ceiling.** Almost all of the 0.478 is
irreducible cross-experiment variation, not rule failure. C itself correlates **+0.9707** between
the two experiments — the quantity transfers nearly perfectly; what does not transfer is the
*prescription*, because of how k enters it.

*Where the loss actually comes from.* Refitting REQ-019 using only its own three multipliers
{0.60, 1.00, 1.70} — same data, same experiment, only the ladder degraded — isolates the ladder
effect: sd(k) inflates 1.17x, and the prescription shifts by **0.132 dex**. That is **29% of the
0.458 ceiling**. The remaining ~71% is genuine between-run variation that no ladder length fixes.

*Independent validation of a design choice.* REQ-036 uses a **per-type k** (averaged over 12
matrices) in the denominator rather than per-matrix k. Testing both under ladder degradation:

| k denominator | 11-point vs 3-point prescription shift |
|---|---:|
| per-matrix | 0.1319 dex |
| **per-type (as filed)** | **0.0404 dex** |

Averaging k over 12 matrices cuts ladder sensitivity by **69%** — the filed rule is **3.3x less
sensitive** to ladder length than the per-matrix alternative. This was chosen in iteration 3
because k was the noisier ingredient; it is now independently confirmed as the right choice for a
second, unrelated reason.

**Revised guidance for REQ-036's readout.** The earlier warning stands in weakened form: expect
the benefit to be smaller than within-design numbers imply, because ~71% of cross-experiment
variation is irreducible. But **the rule is not leaving meaningful accuracy on the table** — no
better rule is derivable from this data, since the ceiling is 0.020 dex away. A null result would
indicate that curvature equalisation does not translate into loss improvement, **not** that the
prescription was poorly constructed.

**MECHANISM (iteration 22) — the end-block effect tracks RESIDUAL-WRITING, and two rival
explanations are falsified.**

*Rival 1: tied embeddings — FALSIFIED by its own prediction.* The trainer ties embed and lm_head,
a real structural asymmetry at both network ends. If that drove the effect, end-block projections
should sit *closer* to the Adam-trained embed/lm_head in log C. They sit **further**:

| group | mean log C | distance to embed/lm_head mean (2.687) |
|---|---:|---:|
| end proj (blocks 0, 11) | 4.821 | **2.135** |
| interior proj | 3.885 | 1.198 |
| end non-proj | 4.257 | 1.570 |
| interior non-proj | 4.126 | 1.439 |

The end-block projections are the *least* embed-like matrices in the network. The tie is not the
mechanism. *(Caveat: read from the branch-head trainer; `ebf53cd` is absent from the clone, so the
tie's presence in the measured architecture is unconfirmed — but the falsification does not depend
on it, since it uses only measured curvature.)*

*Rival 2: matrix shape / fan-in — FALSIFIED by a clean natural contrast.* `mlp.proj` is both a
residual writer *and* the only wide-fan-in matrix (768×3072), confounding the two. But `attn.proj`
is a writer with **narrow** fan-in (768×768). Their end-block elevations are nearly identical:

| type | fan-in | end-block delta, fork-1500 | fork-2000 |
|---|---:|---:|---:|
| **attn.proj** | 768 | **+0.895** | **+1.048** |
| **mlp.proj** | 3072 | **+0.977** | **+0.968** |
| mlp.fc | 768 (out 3072) | +0.494 | +0.641 |
| attn.k | 768 | +0.091 | +0.165 |
| attn.v | 768 | +0.025 | +0.131 |
| attn.q | 768 | −0.087 | +0.041 |

**A 4x fan-in difference produces no difference in the effect.** Shape is ruled out.

*What survives.* The two matrices that **write into the residual stream** (attn.proj, mlp.proj)
carry ~1.0 dex of excess curvature at the first and last block; the four that **read from it**
carry 0.03–0.25 dex. `mlp.fc` sits in between (+0.49/+0.64) and is the one partial exception —
it is a reader, but the widest matrix, so a small shape contribution cannot be excluded entirely.

**Reading:** curvature at the residual-stream boundary is elevated specifically for the matrices
that write into it. That is consistent with the residual stream having different variance
structure at the first and last block — the first block writes into a stream carrying only
embeddings, the last writes into one immediately consumed by the output head — but **this
campaign has never measured activations, so the mechanism is inferred, not demonstrated.**

**ITERATION 23 — the residual-writing prediction is confirmed, and the residue is a smooth
position field with a SPECTRAL-SHAPE signature.**

*Prediction test.* If the boundary changes the residual stream, writers' **gradients** should show
it too — not only their curvature. End-block delta by role:

| quantity | writer | reader | writer − reader |
|---|---:|---:|---:|
| log C | +0.936 | +0.131 | +0.805 |
| **log g** | **+0.180** | **−0.028** | **+0.209** |
| log curv-along-gradient | +0.822 | −0.047 | +0.868 |
| log curv-along-polar | +0.386 | +0.050 | +0.336 |

(fork-1500; fork-2000 reproduces every row within 0.02–0.15.) **Writers show a gradient asymmetry
at the boundary; readers show none.** The boundary affects writers' whole gradient/curvature
system, which is what the residual-stream reading requires and a curvature-only artifact would not
produce.

*But the g² law explains less than half of it.* Decomposing the elevation:

| | delta log C | 2 × delta log g | **unexplained** |
|---|---:|---:|---:|
| writer, fork-1500 | +0.936 | +0.361 | **+0.575** |
| writer, fork-2000 | +1.008 | +0.363 | **+0.645** |
| reader, fork-1500 | +0.131 | −0.056 | +0.187 |
| reader, fork-2000 | +0.245 | −0.055 | +0.299 |

**~0.6 dex of writer curvature at the boundary is not accounted for by the campaign's most solid
law.** That residue is the genuinely new content.

*The residue is a smooth position field, not a two-block anomaly.* Writer residue by block runs
−3.51 (b0), −3.85, −3.82, −3.83, −4.01, −3.97, −4.12, −4.16 (b7), −4.12, −3.98, −3.81, −3.27
(b11): **corr(d_edge, writer residue) = −0.712 / −0.727**. It rises smoothly from the interior
toward both ends. This means the `is_end` binary in REQ-036 is a *coarse approximation* of a
continuous field — adequate for the prescription (validated in iterations 18/20/21) but not the
true functional form.

*A spectral-shape signature — and a qualification of an earlier finding.* The spectral gap
(w0/|w1|) **widens at the boundary for both roles**: reader +0.511, writer +0.555 at fork-1500
(+0.405 / +0.502 at fork-2000). So at the ends the top eigenvalue **separates from the bulk** —
the Hessian changes *shape*, not just scale. **This qualifies iteration 6's finding that the
normalized spectrum is s-invariant** (CV 1.5–3% across the s ladder): that was invariance under
*learning-rate* change, and it holds. Spectral shape is *not* invariant across **depth position**.
Both statements are true; the earlier one should not be read as "shape never varies".

**ITERATION 24 — curvature concentration is real but explains only ~20% of the residue.**

*The hypothesis.* If boundary writers concentrate curvature into fewer directions, lam_top rises
without total curvature or gradient changing. Measured via the Krylov participation ratio
PR = (sum|w|)²/sum(w²) and the top-direction share |w0|/sum|w|:

| | interior | end | delta |
|---|---:|---:|---:|
| PR, writer (fork-1500) | 4.980 | 4.152 | **−0.828** |
| PR, writer (fork-2000) | 4.973 | 4.012 | **−0.961** |
| top share, writer (fork-1500) | 0.284 | 0.379 | **+0.094** |
| top share, writer (fork-2000) | 0.285 | 0.388 | **+0.103** |

Both move exactly as predicted, at both forks. And residue-vs-concentration regresses with a
**near-identical slope at the two independent forks: −4.552 and −4.539** (R² 0.47 / 0.57,
corr −0.683 / −0.757) — slope stability across states is the signature of a mechanical
relationship rather than a fit.

*Circularity check — passes.* PR and top-share are built from the same tridiagonal as lam_top, so
the hard rule applies. The mechanical null (lam shuffled across matrices) gives corr = **+0.076,
95% [−0.322, +0.362]**; the observed **−0.683 sits well outside it**. The correlation is real, not
forced by shared terms.

***But the non-circular test deflates the claim substantially.*** Comparing lam_top against the
**sub-top mass** (sum|w1..w7|) shares no term between the two sides:

| | delta log lam_top | delta log sub-top mass | **divergence** |
|---|---:|---:|---:|
| **writer** | +0.936 | +0.822 | **+0.115** |
| reader | +0.131 | +0.082 | +0.049 |

**The sub-top spectrum rises almost as much as the top eigenvalue.** True concentration accounts
for only **+0.115 dex of the ~0.6 dex residue — roughly 20%.** At the boundary the *whole Hessian
block gets sharper*; it does not merely redistribute existing curvature into the top direction.

**Corrected reading.** The end-block residue decomposes approximately as:
- ~0.36 dex — explained by the gradient (the g² law);
- ~0.12 dex — genuine curvature concentration into fewer directions;
- **~0.46 dex — still unexplained: a uniform sharpening of the entire Hessian block.**

The concentration finding is real and worth having, but it is **not** the mechanism. The dominant
term remains unaccounted for, and it is a whole-spectrum effect — which means activation scale at
the residual boundary (iteration 22's proposed measurement) is now the leading candidate, since a
larger input scale sharpens every direction rather than reorganising them.

**ITERATION 25 — the residual-scale hypothesis is FALSIFIED, and a cleaner statistic replaces it.**

*First, a verified negative on the data itself.* Every curvature JSON in the repo records exactly
eight fields — `alphas`, `offdiags`, `top_eigenvalue`, `residual_tail`, `curvature_along_gradient`,
`curvature_along_polar`, `gradient_block_norm`, `shape`. **No activation data exists anywhere**,
and `measure_per_matrix_curvature.py` is not in the repo (it lived on the boxes). The
activation-variance test genuinely requires a new run; it cannot be recovered offline.

*The falsification.* Iterations 22–24 built toward "the residual stream is larger at the boundary,
so writers see more curvature". That hypothesis makes a sharp prediction: attn.q, attn.k, attn.v
and mlp.fc all **read the residual stream directly**, so if |a| were elevated at the boundary,
all four gradients would rise there (grad = d·aᵀ, so |grad| ~ |d|·|a|). Measured end-block delta
log g by type:

| type | fork-1500 | fork-2000 |
|---|---:|---:|
| mlp.proj (writer) | **+0.206** | **+0.209** |
| attn.proj (writer) | **+0.155** | **+0.155** |
| mlp.fc (reader) | −0.005 | +0.009 |
| attn.k (reader) | −0.012 | −0.015 |
| attn.q (reader) | −0.034 | −0.046 |
| attn.v (reader) | −0.061 | −0.057 |

**No reader shows elevation; both writers do.** The residual-stream-scale reading is dead — and
the sign runs the wrong way as well: reader log g correlates **+0.63 / +0.66** with distance to
edge, meaning reader gradients are *lower* at the ends.

*The constraint this leaves is sharp.* Writers and readers move in **opposite directions** at the
boundary (corr with d_edge: writers −0.277/−0.268, readers +0.196/+0.213). Same block, same
backward pass — so whatever happens at the boundary **raises what writers receive while lowering
what readers receive.** Since writers differ from readers only in (i) taking the block's internal
activation as input rather than the residual stream, and (ii) receiving output-gradient from the
residual stream rather than from inside the block, the effect must live in one of those two.

*The cleanest statistic in the campaign.* The **within-block writer/reader gradient ratio** cancels
every block-level common factor (data, batch, global scale, network age), leaving only the role
asymmetry:

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| log(writer g / reader g) | .301 | .314 | .383 | .230 | .118 | .195 | .084 | .130 | .109 | .167 | .300 | **.522** |

**corr(d_edge, ratio) = −0.784 (fork-1500) and −0.785 (fork-2000), permutation p = 0.0024 /
0.0026 with only 12 blocks.** Two independent states agree to three decimal places. This is a
within-block, common-factor-free, permutation-tested position field — the most robust
non-circular result the campaign has produced.

**ITERATION 26 — a SECOND, INDEPENDENT field found: a monotone depth ramp in step geometry.
Not a boundary effect, and it must not be merged with one.**

*Where this started.* Trying to separate the two surviving channels (writer input vs writer
output-gradient), I looked at **scale-invariant ratios** — quantities that cancel any uniform
rescaling of the Hessian, so neither an activation-scale nor an output-gradient story can move
them. The ratio `curvature_along_gradient / curvature_along_polar` shifts by **+0.437 / +0.412
dex** at the boundary for writers.

*I initially read that as decisive. It is not, and the correction matters.* Pooled across all
72 matrices the field is **not significant**: corr(d_edge, ratio) = −0.138 / −0.146,
**permutation p = 0.68**. Restricted to writers it is corr −0.493 / −0.455 with **p = 0.098 /
0.135**, and the writer × is_end interaction is **t = +1.87 / +1.51** — suggestive, not
established. **The boundary reading of this ratio is not supported.**

*What IS strongly supported is a different geometry entirely.* The ratio is a **monotone ramp in
depth**, not a U in distance-to-edge:

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| log(cg/cp), fork-1500 | 1.32 | 1.41 | 1.82 | 1.82 | 1.70 | 1.68 | 1.65 | 1.74 | 1.80 | 1.85 | 1.95 | **2.33** |

**corr(DEPTH, ratio) = +0.800 (fork-1500) and +0.799 (fork-2000), permutation p = 0.0005 /
0.0006.** Block 0 is the *minimum* (1.32) and block 11 the *maximum* (2.33) — an asymmetric ramp,
which is why the symmetric d_edge parameterisation failed on it. Two independent forks agree to
three decimals.

*What the ratio means, and why it matters for Muon specifically.* `curvature_along_gradient` is
curvature along the gradient; `curvature_along_polar` is curvature along **Muon's orthogonalised
step direction** — the direction the optimiser actually moves. Their ratio measures **how much
flatter the step direction is than the gradient direction**. It rises monotonically with depth:
in deep blocks, Muon's step geometry diverges further from the local loss geometry than in
shallow ones. Writers run consistently *lower* than readers (+1.595 vs +1.836), i.e. for
residual-writing matrices the step direction is relatively better aligned with the sharp
directions.

*Non-circularity.* `curvature_along_polar` is a separate quadratic form on Muon's polar direction —
**not** part of the Lanczos tridiagonal (unlike `curvature_along_gradient`, which is exactly
alpha_1). The ratio correlates only **+0.491** with log lam_top, so it carries genuinely
independent information and does not violate the hard rule.

**REGISTERED SEED CHECK — new, zero cost, independent of every other check:**
- **corr(depth, log(cg/cp)) ≥ +0.6 in every seed**, with block 11 the maximum and block 0 the
  minimum in at least 3 of 4 seeds.
- **registered negative:** corr(d_edge, log(cg/cp)) pooled must remain **non-significant**
  (|corr| < 0.4). If a seed shows a strong *edge* correlation here, this iteration's separation of
  the two fields was wrong and they are one phenomenon.

**Two distinct position fields now stand, and they should not be conflated:**
1. **Curvature boundary field** — U-shaped in d_edge, writer-specific, permutation p < 0.0001 at
   block level (iterations 16–25).
2. **Step-geometry depth ramp** — monotone in depth, all matrix types, permutation p = 0.0005
   (this iteration).

They have different shapes, different scopes, and different significance profiles. The first
drives REQ-036's `is_end` term; the second is new and currently has **no** design implication —
recording it as a finding, not a prescription change.

**REGISTERED SEED CHECK — replaces the residual-scale checks, zero cost:**
- writer/reader gradient ratio: **corr(d_edge, ratio) ≤ −0.6 in every seed**, with block 11 the
  maximum in at least 3 of 4 seeds.
- **falsifier retained:** if readers *do* show end-block gradient elevation (delta log g ≥ +0.10)
  in any seed, the residual-scale hypothesis revives and this falsification was premature.

**Status of the mechanism question.** Three candidate mechanisms have now been falsified — tied
embeddings (iteration 22), matrix shape/fan-in (22), and residual-stream scale (25) — and one is
quantified but insufficient (curvature concentration, ~20% of the residue, iteration 24). What
survives is a precisely-located asymmetry: **at the first and last block, residual-writing
matrices receive systematically more gradient and ~4x more curvature than their gradient predicts,
while the readers beside them receive less.** Distinguishing the two remaining channels
(writer input vs writer output-gradient) requires per-tensor activation and backward-pass norms —
the measurement noted in iteration 22, still not filed to avoid deepening a 4-request queue.

**Additional zero-cost seed checks:** PR must fall at the boundary for writers (delta ≤ −0.5) and
the residue-vs-log-PR slope must be −4.5 ± 1.5 in every seed. **And a registered negative:** the
lam_top-minus-sub-top divergence must stay **below +0.25 dex** — if a seed shows concentration
accounting for most of the residue, this iteration's deflation was wrong and concentration is the
mechanism after all.

**REGISTERED SEED CHECKS — additions to Arm A, all zero cost:**
- writer gradient asymmetry: end-block delta log g ≥ +0.10 for writers and ≤ +0.05 for readers.
- unexplained residue: delta log C − 2·delta log g ≥ +0.4 dex for writers, in every seed.
- corr(d_edge, writer residue) ≤ −0.5 in every seed (the smooth field, not just the endpoints).
- spectral gap widens at the boundary: delta(w0/|w1|) ≥ +0.25 for both roles.

If the residue reproduces across seeds it is architectural, and the remaining question is
narrow and well-posed: **what makes the Hessian of a residual-writing matrix ~4x sharper than
its gradient predicts, specifically at the first and last block.** That is a far sharper question
than the one this campaign started with, and the activation-variance measurement noted in
iteration 22 is the direct test.

**REGISTERED SEED CHECK — addition to Arm A, zero cost:** the writer/reader split of the
end-block effect must reproduce. Band: end-block delta ≥ +0.6 dex for attn.proj and mlp.proj, and
≤ +0.3 dex for attn.q, attn.k, attn.v, in every seed. If writers and readers do not separate, the
"residual-writing" reading falls and the effect reverts to an unexplained per-type interaction.

**The decisive experiment is not a seed check.** It is measuring **residual-stream activation
variance per block** — one forward pass on an existing checkpoint, no training. If the activation
variance at blocks 0 and 11 differs from the interior in proportion to the curvature elevation,
the mechanism is demonstrated rather than inferred. **This is cheap enough to attach to any
already-queued run rather than justify a separate request; noting it rather than filing it, since
the queue is 4 deep.**

**Registered addition:** the per-type × is_end interaction must reproduce in every seed
(interaction t > 2, same sign) in REQ-035 Arm A. If it does not, revert to the uniform is_end
term; if the uniform term also fails to reproduce, revert to per-type only.

**[SUPERSEDED — DO NOT USE] earlier revision, kept for provenance only.** This uniform ×1.94 end-block table was replaced in iteration 19 by the role-split table, and both are superseded by the AUTHORITATIVE ARM TABLE at the top of this request:

| type | base | at blocks 0 / 11 |
|---|---:|---:|
| attn.proj | 0.40 | 0.77 |
| attn.k | 0.88 | 1.71 |
| mlp.fc | 0.91 | 1.76 |
| attn.q | 1.18 | 2.28 |
| attn.v | 1.25 | 2.42 |
| mlp.proj | 1.56 | 3.03 |

**[SUPERSEDED] The instruction below is historical; follow the authoritative arm table instead.**
Original text: Arm 2 should use this revised table, and keep the original per-type-only rule as a
fifth arm if capacity allows — the two differ by a factor of 1.9 on 12 of 72 matrices, and
their comparison directly tests whether the block-level boundary field is real in a way no
offline analysis can settle. If only four arms fit, run the revised table (arm 2) and drop the
half-strength arm (arm 3) instead.

**Additional registered prediction:** arm 2 (revised) should beat arm 2 (original per-type only)
by 0.0005–0.003 val. If the original beats the revised, the block-level boundary field does not
translate into training benefit and the amendment should be reverted.

**ITERATION 27 — A BETTER DESIGN TARGET. Equalize curvature along MUON'S STEP DIRECTION,
not the top eigenvalue. This is a proposed arm, not a replacement of the filed table.**

*Correcting iteration 26.* I recorded the step-geometry depth ramp as having "no design
implication". **That was an assertion, not a test, and it was wrong.** Muon moves along its
orthogonalised (polar) direction, not along the top eigendirection — so which curvature the rule
equalizes is a live design choice that this campaign never examined.

*The comparison.* Same construction, different target: equalize `curvature_along_polar` instead of
`top_eigenvalue`.

| test | equalize lam_top (filed) | **equalize curv_polar** |
|---|---:|---:|
| prescription SNR (between-state) | **13.1** | 11.4 *(iteration 27 reported 41.3 — see CORRECTION below)* |
| between-state disagreement | **0.0133 dex** | 0.0217 dex *(corrected)* |
| cross-state prescription corr | **+0.9979** | +0.9848 *(corrected)* |
| leave-one-block-out rms | 0.2500 | **0.1935** |
| cross-state transfer rms | 0.2652 | **0.1936** |
| **independent REQ-023 holdout** | 0.5402 | **0.2767** |
| target reproducibility (corr / shift) | +0.9747 / 0.0281 | **+0.9877 / 0.0145** |

**The polar target wins four of seven tests** — decisively on the independent REQ-023 holdout,
where it halves the error (0.277 vs 0.540 dex) — but **loses on prescription stability** once that
metric is computed correctly. See the correction below.

*One test initially failed catastrophically, and the cause was my own error.* Cross-state transfer
first came out at **1.21 dex**. Diagnosis: I computed the "truth" using per-**matrix** k (whose
individual values reach −1.614 and cross zero) while predicting with per-**type** k. Since k is the
denominator, near-zero per-matrix values explode the reference, not the prediction. Recomputed
consistently the value is **0.1936 dex**, and it is insensitive to the fix (pooled k gives 0.1843,
flooring |k| at 0.5 gives 0.1936). **The failure was in my test, not in the target.**

*The prescriptions genuinely differ* — they correlate only **+0.597**:

| type | equalize lam_top | **equalize curv_polar** |
|---|---:|---:|
| attn.q | 1.169 | **0.568** |
| attn.k | 0.879 | 0.755 |
| attn.proj | 0.433 | 0.642 |
| attn.v | 1.214 | 1.101 |
| mlp.fc | 0.932 | 1.260 |
| mlp.proj | 1.542 | **2.462** |

attn.q moves by a factor of 2 and mlp.proj by 1.6x. **This is not a refinement of the filed rule;
it is a different rule.**

*Why it is better motivated physically.* lam_top is the curvature along a direction Muon never
moves in — iteration 3 measured lam_top/lam_polar ≈ 300x, so the optimiser travels through a
nearly flat groove while the "edge of stability" cliff sits off to the side. Equalizing the
curvature the optimiser **actually experiences** is the more defensible target, and the
step-geometry depth ramp (iteration 26) is the field that makes the two targets diverge.

**PROPOSED — do not change the filed arms without operator agreement.** Add a **sixth arm** using
the polar prescription above. If capacity forces a choice, I would rank it **above** the
half-strength arm and above the original per-type-only arm, because it tests a different
hypothesis rather than a magnitude. **Registered prediction:** polar-target arm beats the
lam_top-target arm by 0.0005–0.003 val. If lam_top wins, the top eigenvalue is the right
stability quantity despite Muon not moving along it — itself a substantive result.

**ITERATION 29 — AUDIT of `curvature_along_polar`, the field the new design target rests on.
It clears the circularity trap, with one qualification.**

`curvature_along_gradient` turned out to be exactly alpha_1 (iteration 6), which invalidated a
headline result. The polar target had not been given the same scrutiny, so:

*It is not a disguised tridiagonal element.* |cp/alpha_1 − 1| median = **0.980**, |cp/alpha_last − 1|
= 0.985, |cp/w_min − 1| = 1.024 — it matches nothing in the Lanczos tridiagonal (whereas
|cg/alpha_1 − 1| = **0.00000000**, confirming that trap exactly). It correlates only **+0.583**
with log lam_top, so it carries independent information.

*It is a valid Rayleigh quotient.* 99.3% positive, and **cp ≤ lam_top in 100% of 2376 rows** — the
upper Rayleigh bound is never violated, which it would be if cp were a quadratic form of a
different operator than the one Lanczos ran on.

*It is an AVERAGE-curvature probe, not a lam_top proxy.* Muon's polar factor has
‖O‖²_F = rank, so O'HO/‖O‖² is the mean curvature over the row space — it should track the
spectrum's *mean*, not its top. Measured: corr(log|cp|, log|Krylov mean |eig||) = **+0.612**
versus **+0.583** for lam_top. And cp·rank/lam varies **6x across types** (1.12 for attn.q to 6.89
for mlp.proj), so cp is **not** a shape-normalised lam_top. **Iteration 27's design argument
survives: the two targets are genuinely different physical quantities.**

*The qualification — and it matters.* 14.8% of rows have cp < w_min. That is *expected* (w_min is
the smallest **Ritz** value from an 8-step Krylov space, not the true minimum, and a vector outside
that space may legitimately fall below it), and the violations concentrate in poorly-converged
rows (median residual_tail **0.0491 vs 0.0194**). **But they are severely type-structured:**

| type | % of rows with cp < w_min |
|---|---:|
| **attn.v** | **62.9** |
| **attn.proj** | **17.4** |
| mlp.proj | 8.3 |
| attn.k / attn.q / mlp.fc | 0.0 |

attn.v and attn.proj are precisely the types with the worst Lanczos convergence (iteration 5:
attn.v 66% gate attrition). **cp itself is fine — it is measured directly, not from the Krylov
space — but its comparison against Ritz-derived quantities is type-biased.** Any statistic mixing
cp with tridiagonal quantities (including iteration 26's cg/cp ramp) inherits that bias for attn.v
and attn.proj specifically.

**Consequence for the polar prescription:** the arm-5 multipliers are built from cp alone, not from
cp/Ritz ratios, so they are **unaffected**. The depth ramp of iteration 26 **is** affected and
should be treated as provisional for attn.v and attn.proj until higher-iteration Lanczos data
exists.

**Added registered seed check (zero cost):** report the fraction of rows with cp < w_min per type.
If attn.v again exceeds ~50%, the type-biased comparison is architectural and every cp/Ritz
statistic needs a convergence-matched control. If it does not reproduce, iteration 26's ramp is
firmer than stated here.

**CORRECTION (iteration 30) — the polar target's SNR advantage was an error of mine.**

Iteration 27 reported prescription SNR **41.3** for the polar target against 13.1 for lam_top, and
used it as the headline. **That number does not reproduce by any aggregation.** Recomputed:

| target | spread | median disagreement | corr | **SNR** |
|---|---:|---:|---:|---:|
| lam_top | 0.1750 | 0.0133 | +0.9979 | **13.1** |
| polar | 0.2462 | 0.0217 | +0.9848 | **11.4** |

Per-type (n=6) and per-matrix (n=72) aggregations give 14.3/13.1 for lam_top and 12.4/11.4 for
polar — **neither yields 41.3**. On prescription stability **lam_top is slightly ahead**, not 3x
behind. The polar target's genuine wins remain: LOBO (0.194 vs 0.250), cross-state transfer
(0.194 vs 0.265), target reproducibility (+0.988 vs +0.975), and the **independent REQ-023 holdout
(0.277 vs 0.540 — still a halving)**. It is a real contender, not a clear winner. **Arm 5 stays in
the authoritative table at its current priority; the case for it is weaker than iteration 27 said.**

**ROBUSTNESS AUDIT (iteration 30) — every core finding survives dropping the badly-converged types.**
attn.v (66% gate attrition, 62.9% Rayleigh violations) and attn.proj (17.4%) appear in every
result. Dropping them:

| finding | all 6 types | −attn.v | −attn.v −attn.proj |
|---|---:|---:|---:|
| within-type slope | +2.124 | +2.268 | +1.896 |
| between-type slope | +0.375 | +0.285 | +0.332 |
| pooled slope | +0.742 | +0.702 | +0.656 |
| boundary corr (block level) | −0.893 | −0.891 | −0.883 |
| writer end delta | +0.936 | +0.936 | — |

**Nothing depends on the poorly-measured types.** The within-type slope still brackets 2, the
between-type stays far below it, and the boundary field is essentially unchanged.

*One structural fact this surfaced.* Leave-one-type-out on the prescription spread:

| removed type | lam_top rule loses | polar rule loses |
|---|---:|---:|
| attn.proj | **49%** | −2% |
| mlp.proj | 10% | **37%** |

**The lam_top rule derives half its dynamic range from attn.proj alone**; the polar rule from
mlp.proj. Both are single-type-dependent, which is a caution for both — a per-type rule built on
six types where one carries ~40-50% of the signal is fragile to that type being mismeasured.
**Registered seed check:** the leave-one-type-out concentration must stay below 60% for whichever
rule is run; if one type carries more than that in any seed, the rule should be refit excluding it.

**Registered seed check (zero cost):** the polar prescription's per-type ordering must reproduce
across seeds (Spearman ≥ +0.7), and its between-state SNR must exceed the lam_top rule's in at
least 3 of 4 seeds.

### [SUPERSEDED] Original arm list (iteration 3) — kept for provenance

*Numbering here does NOT match the authoritative arm table at the top of this request (this arm 3 is 'half-strength'; the live arm 3 is the role-split end-block rule). Execute only from the authoritative table.*

Original heading: Arms — 750-step continuations from the shared step-2000 state, val@2750

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

## REQ-038: per-type activation and backward statistics — the q/k/v probe

- status: **ACCEPTED — QUEUED on capacity (2026-09-03 ~05:30Z).** Noted it's a short single-checkpoint probe ("fits in a short freed window") — good, it's the cheapest of the queue. Same blocker as REQ-034/035/036/037: both my nodes are REQ-032's tau2 arms (~20h left) under Jerry's ≤2 ceiling, so no free window yet. **Queue is now 5 deep (034/035/036/037/038).** When a tau2 node frees I'll slot the cheapest-useful item (likely this probe or REQ-036's 1-box headline) first; full fan-out still awaits Jerry lifting the ceiling. Not provisioned.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `ebf53cd` (same trainer + probe as REQ-019/022/023)
- **cost: ONE forward+backward pass on an EXISTING checkpoint. No training. Minutes, not
  hours.** This is the cheapest request in the queue by a wide margin and needs no fresh fork.
- **node budget: whatever is in force.** I am the requester, not your operator; run it in the
  gaps of any other job.

**Why this is now the highest-value measurement available.** Iteration 32 scored the campaign's
answer as a variance budget of the 0.379 dex spread in log C:

| contribution | share |
|---|---:|
| gradient scale (lam ∝ g², Gauss-Newton) | 21.8% |
| **matrix type, beyond gradient** | **53.3%** |
| end-block position | 8.2% |
| writer-role interaction | 1.9% |
| unexplained | 14.8% |

**The largest term is a label, not a mechanism**, and it is irreducible to anything computable
from committed data: against the type label's R² = 0.751, the best architectural descriptor set
(writer role + attn/mlp + fan-in + fan-out) reaches 0.484 and the best measured set (polar
curvature + spectral gap + negative-eigenvalue fraction) reaches 0.505.

**Iteration 33 localised it precisely, and the result is sharp.** Taking the gradient-adjusted
level `log C − 2 log g` (which removes the one component we *do* understand):

| type | adjusted level | vs attn.q |
|---|---:|---:|
| attn.q | −2.855 | 0.000 |
| attn.k | −3.026 | −0.171 |
| **attn.v** | **−3.666** | **−0.812** |
| mlp.fc | −3.605 | −0.750 |
| attn.proj | −3.842 | −0.987 |
| mlp.proj | −3.898 | −1.044 |

**q, k and v span 0.428 dex — 98% of the entire between-type spread — despite being identical in
shape (768×768), identical in input (the same residual vector), and identical in block position.**

And the ordering is essentially deterministic:

| | v < k < q holds in | binomial p (chance 1/6) |
|---|---:|---:|
| fork-1500 | **11 of 12 blocks** | **2.8 × 10⁻⁸** |
| fork-2000 | **12 of 12 blocks** | **4.6 × 10⁻¹⁰** |

Two independent states, twelve independent blocks. **This is not noise and it is not
architecture-as-shape.** The only remaining difference between q, k and v is *how each is used
downstream*: q and k enter a bilinear inner product (so their Hessians carry each other's scale),
while v passes through a linear mixing weighted by attention probabilities. **No committed
measurement captures that**, which is exactly why 53% of the variance is unexplained.

**ITERATION 34 — the bilinear mechanism FAILS replication. REQ-038 is now more necessary, not
less.**

REQ-038's rationale proposed that q and k differ from v because they enter a bilinear product.
That is testable in committed data: within a block, q and k curvatures should co-vary more tightly
with each other than with v, since all four attention matrices see identical input at identical s.

*First pass looked supportive but was confounded.* Raw trajectory correlations gave q~k 0.978 vs
0.908 for the v-pairs — but every pair exceeded 0.90 because all matrices share the same
s-response (k ≈ 1.4). The common response swamped the test.

*Removing each matrix's own power law and correlating the residuals:*

| pair | fork-1500 | **fork-2000** | **REQ-023 (independent design)** |
|---|---:|---:|---:|
| **attn.q ~ attn.k** | **0.567** | 0.667 | **+0.061** |
| attn.q ~ attn.v | 0.154 | 0.833 | **+0.450** |
| attn.k ~ attn.v | 0.278 | 0.833 | −0.057 |
| q~k largest in | 7/12 (p=0.066) | **0/12** | **4/12 (p=0.61)** |

**Only fork-1500 supports it.** Fork-2000 ranks q~k *lowest* (paired t = **−1.48**, wrong sign),
and REQ-023 — a different experimental design — also ranks q~k lowest, with q~v seven times
larger. **The bilinear coupling hypothesis is not supported.** *(Caveat: fork-2000's residual
correlations are largely degenerate — 3 multipliers minus a 2-parameter fit leaves 1 d.o.f., so
most come out at exactly 1.000. REQ-023 with 3 assignments is the more trustworthy of the two
disconfirmations, and it disconfirms clearly.)*

**What this changes.** The q/k/v ordering itself is untouched — it remains v < k < q in 11/12 and
12/12 blocks at p ≈ 10⁻⁸ and 10⁻¹⁰ (iteration 33). What falls is my *proposed explanation* for it.
So the situation is now:

- the effect is among the most statistically secure findings in the campaign;
- it accounts for ~98% of the between-type spread and hence most of the dominant 53% term;
- **and every mechanism proposed for it has now failed** — shape, input, position (iteration 33,
  by construction), and bilinear coupling (this iteration).

**REQ-038's priority increases.** Its registered prediction **P2** ("q and k have near-identical
|a| but differ in |d| by ≥ 15%") was motivated by the bilinear story and should now be read as a
genuinely open question rather than a likely confirmation — if q and k differ in |d| the backward
pass carries the effect; if they do not, the effect lives somewhere no current hypothesis reaches.
**P4's negative band (activation moments fail to lift R² from 0.218 to ≥ 0.60) is correspondingly
more likely than when filed, and would establish the type term as irreducible in this
architecture** — which remains a real and publishable outcome.

**Registered seed check (zero cost, added to Arm A):** the q~k residual correlation must not
systematically exceed the q~v and k~v correlations across seeds. If it does in ≥3 of 4 seeds, the
bilinear story revives and this disconfirmation was driven by the two low-multiplier datasets.

### What to measure — one forward+backward pass, per Muon matrix

At a single existing checkpoint (any of REQ-019's fork-1500 arms; s=1.00 preferred), record per
matrix:
1. **input activation second moment**: RMS and Frobenius norm of the matrix's input tensor `a`;
2. **output-gradient second moment**: RMS and Frobenius norm of the backward tensor `d`;
3. **effective rank of both**: participation ratio of the singular value spectrum of `a` and `d`;
4. **for attention specifically**: the attention-probability entropy per head, and the RMS of the
   q·k logits — the quantity that distinguishes q/k from v mechanically;
5. token count and batch used, so the moments are normalisable.

Fields 1–3 are generic; field 4 is the discriminating one.

### Registered predictions, bands fixed in advance

- **P1.** The gradient identity `|grad| ≈ |d| · |a|` must hold per matrix to within 20% —
  a correctness check on the probe itself. If it fails, the other numbers are not interpretable.
- **P2.** **q and k have near-identical `|a|` (within 5%, they read the same tensor) but differ in
  `|d|` by ≥ 15%.** If instead their `|d|` matches too, the q/k difference is not in the backward
  pass and the bilinear explanation fails.
- **P3.** **v's `|d|` differs from q/k's by ≥ 30%**, consistent with its 0.81 dex adjusted-level
  gap being the largest of the three.
- **P4.** Adding `|a|`, `|d|` and the two effective ranks as regressors to the model
  `log C ~ log g + …` **raises R² from 0.218 (gradient only) to ≥ 0.60**. If it does not reach
  0.60, the activation/backward moments do **not** explain the type effect either, and the 53%
  should be reported as irreducible in this architecture — a negative result worth having and one
  that would close the campaign's central question rather than leave it open.

**AMENDED PREDICTIONS (iteration 42) — P2 and P3 as filed target the wrong contrast. Use these.**

REQ-038 was written at iteration 33, when the structure looked like a q > k > v ordering.
Iteration 41 established it is **{q, k} versus the other four types**, and the amendment matters
because the filed P2/P3 aim at the wrong comparison:

| contrast | fork-1500 | fork-2000 | status |
|---|---:|---:|---|
| **q,k mean − other four** | **+0.812** | **+0.842** | **THE effect** |
| k − v | +0.640 | +0.656 | part of the same effect |
| q − k | +0.171 | +0.167 | real but ~20% the size |

*(q − k is genuinely significant — paired t = +4.62 / +4.58, q > k in 11/12 and 12/12 blocks — but
it is a second-order feature within the high group, not the main effect. P3 as filed attributes
the 0.81 dex figure to v alone; that number is the q,k-versus-rest contrast.)*

**The sharp quantitative prediction the probe should carry.** Measured from committed data:

> **q and k carry 0.40× the gradient norm of the other four types (log gap −0.403 / −0.407 dex,
> both forks) while sitting +0.81 dex HIGHER in gradient-adjusted curvature.**

Since `|grad| ≈ |d| · |a|` and q, k, v all read the **same residual tensor**, their |a| is identical
by construction — so **the entire q,k-vs-v gradient deficit must appear in |d|**. That is directly
falsifiable against the probe:

- **P2′ (replaces P2).** `|a|` for q, k and v must agree within **5%** (they read the same tensor —
  a probe correctness check as much as a physics one). **`|d|` for q,k must be 0.35–0.45× that of
  v**, matching the −0.40 dex gradient gap. If |d| does not carry the deficit, either the probe is
  wrong or |a| differs where it cannot.
- **P3′ (replaces P3).** The q,k-vs-other-four contrast in `|d|` must be **≥ 2x**, and the same
  contrast in `|a|` must be **< 1.2x** across the four types that read the residual stream.
- **P5 (new, the decisive one).** **q and k receive far less gradient yet carry far more curvature
  per unit gradient. If |d| alone explains the +0.81 dex adjusted-curvature gap, then adding |d|
  to the model must reduce the q,k binary's coefficient by ≥ 50%.** If the binary survives with
  its coefficient intact after conditioning on |a| and |d|, **the effect is not in the first-order
  backward statistics at all** — and given that four mechanisms have already been falsified
  (bilinear coupling, nonlinearity exposure, softmax saturation, consumption order), that would
  establish the QK gap as irreducible to anything measurable in a single forward+backward pass.

**P1 and P4 stand as filed.** P4's ≥0.60 band should now be read against the QK binary's own R² of
0.737 — activation moments must beat *that*, not the 0.218 gradient-only baseline, to count as an
explanation rather than a correlate.

**Why this amendment rather than a new request:** the measurement list (fields 1–5) is unchanged —
only the registered bands move. No extra compute, no queue impact.

### Success criteria
- One JSON per matrix with fields 1–5 plus the existing `gradient_block_norm` for the P1 check.
- `summary.tsv` with the per-type means of every recorded quantity.
- The P4 regression run and reported against the 0.60 band, whichever way it falls.

### Artifacts
`logs/kmaxwell/req038_activation_probe/`

**Priority note for the operator, offered not asserted:** this targets the 53% term; REQ-036's LR
arms target the 8% and the measurable parts. If a node frees for a short window, this fits in it
and the others do not. Sequencing remains the operator's call.

## REQ-039: stop the REQ-032 tau2 arms (checkpointed), then run the EoS queue

- status: OPEN
- requested: Jack / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- **This supersedes the queue order and the ≤2-node ceiling.** The node-authority conflict you
  flagged on REQ-034 is resolved: **these are Jack's runs and Jack's boxes, and permission for up
  to 10 H100/H200 nodes is granted.** Proceed without waiting for further confirmation.
- *(The narrative version of this is in the OPERATOR DIRECTIVE block near the top of this file. It
  was written without a `status:` line, so it did not enter your queue — this request block is the
  actionable form. They say the same thing; follow this one.)*

### Step 1 — checkpoint, then stop, the two remaining REQ-032 arms

`gold` on `wnle40q` and `step_hint` on `wox8gkw` are to be **stopped cleanly, not abandoned**:

1. **Reach a clean checkpoint before killing anything.** `gold` already auto-recovered once from an
   NCCL abort at step 124, so do **not** SIGKILL mid-step — let it write a loadable checkpoint
   first.
2. Commit the eval curves and logs collected so far to `logs/async_sdpo_req032/<arm>/`, as you did
   for the two completed diligence arms.
3. **Record the exact step each arm stopped at**, so both remain resumable rather than lost.
4. Release both boxes.

**If an arm is within ~30 minutes of finishing, finish it instead.** The goal is free capacity, not
discarded work. Say which you did.

### Step 2 — run the four EoS experiments

With capacity free, run in this order; parallelise across boxes wherever arms are independent:

| priority | request | why |
|---:|---|---|
| **1** | **REQ-038** | Cheapest by far — one forward+backward pass on an existing checkpoint, minutes not hours. Carries the sharpest prediction (below). |
| **2** | **REQ-035 Arm A** | Load-bearing: 4 seeds decide whether any finding is architectural or an artifact of one trained network. |
| **3** | **REQ-036** | The per-layer LR design, 5 arms including the anti-rule falsifier. |
| **4** | **REQ-037** | Non-LR instrument; tests the exclusion restriction behind the gradient law. |

REQ-034 is unrelated to this queue — order it against the above as you see fit.

### The number to check first

REQ-038 measures, per matrix, the input activation `|a|` and the backward tensor `|d|`. **q, k and
v read the same residual vector, so their `|a|` is identical by construction** — any gradient
difference must sit entirely in `|d|`. From committed data:

> **Predicted: `|d|(q,k) / |d|(other four types) = 0.39 ± 0.08`**

- **Near 0.39** → the campaign's central anomaly closes: the gradient law λ ∝ g² is universal and
  q,k's apparent violation is the attention softmax attenuating the backward signal.
- **Near 1.0** → the deficit is in `|a|` instead, which contradicts q,k,v sharing an input and means
  either the probe or our reading of the model code is wrong. **Report that loudly.**

### Where the live specifications are

Both REQ-035 and REQ-036 accumulated many superseded prediction blocks across a long analysis
session. **Use only the two authoritative tables** — the `AUTHORITATIVE SEED-CHECK TABLE` inside
REQ-035 and the `AUTHORITATIVE ARM TABLE` inside REQ-036. Everything below those tables is
provenance and must not be executed from.

### Success criteria

- Both REQ-032 arms stopped with a committed checkpoint and a recorded stop step (or finished).
- REQ-038 reports the `|d|(q,k) / |d|(others)` ratio against the 0.39 ± 0.08 band.
- REQ-035 Arm A reports all five bands in its authoritative table, per seed.
- REQ-036 reports val per arm plus the three required readouts named in its table.

## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```

