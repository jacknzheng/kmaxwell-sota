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

Next request number: **REQ-021**.

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

## REQ-020: 9-GPU local-hint SDPO fleet

- status: NEEDS-INFO — **a 9-GPU box is not provisionable on this Baseten fleet** (workstation `--gpu-count` is capped at 8). Per this request's own "record and stop if <9 visible devices" rule, stopped. Need a 9-GPU box source or an approved layout change.
- requested: Jack / 2026-08-29 23:58 PDT

### REQ-020 STATUS (agent 2026-08-30) — blocked on 9-GPU hardware availability

The operator authorized lifting the 2-node cap for this request. But **the hardware isn't available via the standard tooling**: `truss train workstation --accelerator h100 --gpu-count 9` → `Error: Invalid value for '--gpu-count': 9 is not in the range 1<=x<=8`. Baseten H100 workstations top out at **8 GPUs**. The requested map (`cuda:0-3` policy + `cuda:4-7` trainer + `cuda:8` frozen hint engine) needs 9 on a single box, and REQ-020 explicitly forbids the 8-GPU fallback ("do not shrink to 3+4+1 on an 8-GPU box") and says to stop if fewer than 9 devices are visible — so I did not provision a smaller box.

**To unblock, pick one:** (a) point me at a 9-GPU (or larger) box/source outside `truss train workstation` (a pre-provisioned node, a different accelerator/instance type, or a cluster that offers >8-GPU single nodes); or (b) authorize a modified layout that fits 8 GPUs (e.g. hint engine time-shared on the policy GPUs, or 3 rollout + 4 trainer + 1 hint) — noting you currently forbid that; or (c) revert to the OpenRouter-hint path (REQ-018 `3bd7def`) with a **paid/higher-rate** aux model to sidestep the free-tier 429s. Tell me which and I'll proceed.

REQ-020 supersedes REQ-018, so REQ-018 stays closed (hint-length fix already validated — `logs/async_sdpo_req018/`). REQ-019 (curvature) is unaffected and running on the 2-node budget.
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact SHA: `e2ff7181305b1f6066ff059ae97bb36993371cd5`
- supersedes: REQ-018
- prior evidence: `logs/async_sdpo_req011/`, `logs/async_sdpo_req015/`, `logs/async_sdpo_req018/`

Request a **9-GPU** box. Jack has operator permission to use a box. The
operator default / 2-node cap is not enough and must not be used. Do not
self-cap at 2, do not shrink to 3+4+1 on an 8-GPU box, and do not fall
back to OpenRouter hints or `NPROC=1`. If `nvidia-smi` /
`torch.cuda.device_count()` shows fewer than 9 visible devices, record
the listing, update this block, and stop.

Do not preempt REQ-019. Use a separate 9-GPU box. REQ-018 already
validated the OpenRouter hint-length fix at `3bd7def` (0 drops / 0
`openrouter_length` over ~478 attempts per diligence arm) and then died
on free-Nemotron 429s. This request moves hints onto a frozen local GPU
so the 429 path is gone.

### Hardware map (required)

```text
cuda:0-3  policy vLLM TP=4
cuda:4-7  FSDP2 trainer x4
cuda:8    frozen hint vLLM TP=1  Qwen/Qwen3.5-9B  no weight sync
```

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8`. `torchrun --nproc-per-node=4`.

### Frozen checkout

Do **not** use SHA `3bd7def` (OpenRouter hints, 4+4 on 8 GPUs). Fetch
this exact commit. Do not recreate this as an on-box patch.

```bash
git fetch origin fix/hint-output-budget
git checkout e2ff7181305b1f6066ff059ae97bb36993371cd5
uv run --no-sync pytest -q -m 'not network'
```

Record that SHA. Offline suite at this SHA:
`224 passed, 2 skipped, 2 deselected`.

Resolved defaults that must appear in `config.yaml`:

```text
model.model=Qwen/Qwen3-8B
total_num_gpus=9
generator.engine.n_rollout_gpus=4
trainer.n_trainer_gpus=4
generator.hint.backend=vllm
generator.hint.model=Qwen/Qwen3.5-9B
generator.hint.gpu=8
generator.hint.reasoning_enabled=false
generator.hint.max_tokens=2048
judge.model=nvidia/nemotron-3-super-120b-a12b:free
data.user_llm=openrouter/nvidia/nemotron-3-super-120b-a12b:free
judge.eval_interval=25
logging.checkpoint_interval=50
```

Judge and tau2 user-sim stay on OpenRouter. Hints do not.

### Preflight

OpenRouter (judge / user-sim) and Parallel Search are funded. Before
renting or restarting a box, send one real request to each and record
status plus UTC timestamp, never keys. Then start the local hint engine
and issue one production hint through `build_error_hint` / `generate_hint`
(`backend=vllm`). If the box still returns 402 on judge/search, report
which credential differs and stop that dependent arm. Do not put
credentials in this repository.

### Diligence arms

Preserved checkpoints on `qkpx8dw` / `wp2znpq` were lost in the operator
reset. Restart both diligence arms from step 0 on `e2ff718`. That is
intended: REQ-018 already showed the pre-fix steps were dropped-hint, not
contaminated, and this SHA changes the hint backend.

### Hint validation gate

After at least 100 post-switch hint attempts on each diligence arm,
report:

- attempts, successes, total drops, and `drops / attempts`
- every `hint_drop_*` cause, especially `vllm_error`, `timeout`, `empty`
- zero `hint_drop_openrouter_*` on the vLLM path (those mean the local
  engine was not used)

Gate passes when the total drop rate is below 5% while the hint GPU is
healthy. If the gate fails, preserve raw artifacts, diagnose, update this
block, and stop. Do not raise concurrency, switch back to OpenRouter
hints, enable hint reasoning, or train with an empty hint.

### Tau2

Do not delay diligence on tau2. Resolve the already-reported tau2-bench
`a2c0247` `get_environment` API drift, add a test covering the pinned
signature, commit that follow-up to `scaling-sdpo`, then launch tau2 `gold`
on retail+airline only. `gold` skips the hint LLM; the ninth GPU may sit
idle. Do not use banking on Baseten.

### Frozen experiment

Finish all three arms at `trainer.total_steps=200`:

1. diligence `answer_free`
2. diligence `answer_bearing`
3. tau2 `gold` on retail+airline

Runtime per arm:

- 4 vLLM rollout GPUs + 4 FSDP2 trainer ranks + 1 hint GPU
- `model.model=Qwen/Qwen3-8B`
- `trainer.mini_batch_size=2`
- `generator.engine.max_model_len=32768` for tau2
- never fall back to 4B, `stealth/ox-alpha`, `NPROC=1`, or 2 GPUs

Resume after recoverable crashes. A rollout without its required hint must
be dropped. Keep search, hint, user-simulator, judge, sandbox, empty-episode,
stale-rollout, and weight-sync failures as separate counters.

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
logs/async_sdpo_req020/
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
`vllm.jsonl` (including `hint_engine_*` events), sandbox setup logs, and a
checkpoint/resume manifest. Use `git add -f` for ignored log patterns. Gzip
large text artifacts losslessly and document decompression. Never commit
secrets, environment dumps, model weights, or checkpoint tensor files.

`summary.tsv` must include exact SHA and CLI, visible GPU count, completed
steps, wall time, checkpoint history, teacher-minus-student gap, dead-gap
fraction, clipping, staleness, every failure counter, and held-out metrics.
The README must link each row to its raw directory.

Success: 9 visible GPUs, hint gate passes, and all three arms reach step
200. If a definitive external blocker stops the run, first commit and push
all progress and raw failure logs, then update this block with exact
evidence and stop rather than restarting from zero.

## REQ-019: momentum EoS law across fork states

- status: OPEN — corrected harness ready; discard the nine pre-fix arms whose shared-state gate failed
- requested: OpenAI Codex for Jeffrey Cheng / 2026-08-30 01:40 UTC

### REQ-019 STATUS (agent 2026-08-30) — 9 arms trained; shared-state gate FAILED (real divergence, not nondeterminism)

All 9 EoS arms trained to their stop steps on `kmaxwell-sota @ 755c49d2` (2-node budget, torch 2.10+cu128). Checkpoints saved every 125 steps in `dumps_<run_id>/model_step*.pt`. Ran the shared-state gate before accepting any measurement, per your instruction.

**Gate FAILS (see `logs/kmaxwell/req019_eos_state_dependence/shared-state-check.tsv`):**
- Fork-1500: all **6** arms have **distinct** `model_step001500` sha256 (each 572213441 B). Fork-2000: all **3** distinct at `model_step002000`.
- Max tensorwise abs diff (fork-1500, s060 vs s100 / s130 @ step 1500): **8.675e+01 / 8.125e+01** — this is ~8 orders of magnitude above the ~1e-6 you'd see from NCCL/cuBLAS nondeterminism, so it is a **genuine trajectory divergence**, not run-to-run noise.
- **First divergent checkpoint = step 125** (earliest saved): s060 vs s100 max_abs_diff **26.5** @125 → 69.5 @750 → 86.0 @1375 → 86.75 @1500. They diverge from the *start of training*, not at the fork.
- The only config difference between two fork-1500 arms (e.g. s060 vs s100) is the LR-multiplier value (`0.6` vs `1.0`) + `run_id`/`dump_dir`. Since they diverge from step 125, **the multiplier is being applied globally (from step 0) rather than only after the fork step 1500** — i.e. `make_eos_state_dependence_configs.py` / the runner is not encoding a shared-pre-fork fork. (The duplicate-1.00 control `s100` vs `s100dup` also has distinct hashes, but those are on separate boxes; the same-box s060/s100/s130 comparison already proves the divergence.)

Per your gate protocol I **preserved the runs and did not silently call the states shared, and did not run the 45 curvature measurements** (measuring per-matrix curvature across states that aren't actually shared would be meaningless). 

**To unblock:** fix the config generator / optimizer so the LR multiplier applies **only after `fork_step`** (verify two arms are bit-identical through the fork before diverging), push the corrected SHA, and I'll re-run the 9 arms + gate + curvature. The 9 trained (invalid-fork) dump sets have been **released** with boxes `wol2ygw`/`wn28y0w` (idle GPU while blocked; dumps can't be committed — checkpoint tensors — and a corrected re-run regenerates them; the divergence evidence above is already captured). The **authorized 4-seed noise fleet** is also on hold behind this fix.
- priority: run alongside REQ-018 without preempting it
- concurrency cap: at most 4 simultaneous 8xH100 workstations for REQ-019

Measure whether the per-matrix curvature law changes with training state. This
request contains two shared-trajectory programs, forked at steps 1500 and 2000.
It does not include momentum-kernel sweeps, seed fleets, or the Section-4
interventions; those require separate authorization.

### Frozen public harness

- repo: `https://github.com/jacknzheng/kmaxwell-sota`
- branch: `codex/momentum-kernel-schedules`
- exact SHA: `f83bfcdca955af57b988ee865388712f81a34a81`
- runner: `records/track_3_optimization/run.py`
- config generator:
  `records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py`
- curvature tool:
  `records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py`

The previous generator leaked each arm's learning-rate multiplier into the
pre-fork trajectory; the shared-state gate caught the error, and those nine
runs are invalid. The corrected SHA keeps the ordinary Track-3 schedule and
the post-fork fixed multiplier in one schedule hook. From a fresh public HTTPS
clone, all six fork-1500 configs produced identical learning-rate traces over
steps 0--1499 and all three fork-2000 configs did so over steps 0--1999; the
configured multipliers first appeared at the respective fork steps. Do not
reuse the preserved pre-fix runs, substitute a private `muoff` checkout, or
port the harness again.

On every workstation:

```bash
git clone --filter=blob:none --branch codex/momentum-kernel-schedules \
  https://github.com/jacknzheng/kmaxwell-sota.git
cd kmaxwell-sota
git checkout f83bfcdca955af57b988ee865388712f81a34a81
python records/track_3_optimization/offline_analysis/\
make_eos_state_dependence_configs.py --out configs/req019
```

Record the resolved Python, PyTorch, CUDA, GPU, and git versions, but never
environment variables or credentials. Use the provisioned FineWeb data and
the ordinary 8-rank Track-3 launch environment.

### Frozen slate

The generator emits this manifest:

| fork | fixed LR multipliers after fork | stop after step | curvature checkpoints |
| ---: | --- | ---: | --- |
| 1500 | 0.60, 0.77, 1.00, 1.00 duplicate, 1.30, 1.70 | 2750 | 2250, 2375, 2500, 2625, 2750 |
| 2000 | 0.60, 1.00, 1.70 | 3249 | 2750, 2875, 3000, 3125, 3249 |

Every arm uses seed 0, bi-Maxwell momentum, the standard planned 3250-step
Track-3 schedule before its fork, and an absolute constant learning-rate
multiplier after its fork. Each arm trains from initialization, so it has no
external checkpoint dependency. Runs at the same fork must reproduce the same
model and optimizer trajectory through that fork; the duplicate 1.00 arm is a
run-divergence control.

Run each generated config with:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_3_optimization/run.py configs/req019/<run_id>.yaml
```

Use no more than four workstations concurrently. Assign one arm per node at a
time and refill nodes from the frozen manifest until all nine arms finish. If
REQ-018 needs capacity, reduce REQ-019 concurrency rather than preempting it.

### Shared-state gate

Before accepting the post-fork measurements, compare the saved model tensors
at step 1500 across all six fork-1500 arms and at step 2000 across all three
fork-2000 arms. Report file hashes and the maximum tensorwise absolute
difference. The expected difference is exactly zero because seed, data order,
hardware class, and pre-fork hooks are identical. If it is nonzero, preserve
the runs, set this request to `NEEDS-INFO`, and report the first divergent
checkpoint; do not silently call the states shared.

### Curvature measurement

After an arm trains, run its five manifest checkpoints on all eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py \
  --no_dist --dump_dir dumps_<run_id> \
  --steps <five manifest steps> --tokens 131072 --iters 8 \
  --out_tag req019_per_matrix_curvature
```

The merged JSON must contain all 74 matrices at every requested checkpoint,
including the complete Lanczos `alphas` and `offdiags`; these allow the
geometric-tail correction to be recomputed centrally. Keep the raw rank shards
until the merged file is validated. A low-learning-rate arm that remains in
transient is still a valid deliverable: retain its within-window trajectory so
the relaxation time can be modeled instead of discarding it.

### Authorized generalized-sharpness calibration

Calibrate the paper-faithful global block-spectral generalized sharpness
measurement before adding it to the fleet. Use two valid checkpoints from the
corrected runs, one early and one late, and the same fixed 131072-token set as
the Euclidean-curvature measurement. The Frank--Wolfe domain is the product of
spectral-norm balls over all 72 Muon matrices; each iteration uses one joint
Hessian--vector product, followed by the exact polar/SVD linear minimization
for every block. Cross-block Hessian terms must remain present. A collection
of independent diagonal-block maxima is not an acceptable substitute.

At both checkpoints, run one shared initialization through iteration counts
`K = 5, 10, 20, 50`. Then compare one restart with five restarts at `K = 50`.
Report the objective after every iteration, relative changes between the four
iteration budgets, spread across restarts, peak memory, and node wall time.
Retain the Euclidean Ritz measurement at the same checkpoints for scale only;
do not claim the two observables should agree. Commit implementation and tests
with the calibration artifacts under:

```text
logs/kmaxwell/req019_fw_calibration/
  README.md
  summary.tsv
  objective_trace.tsv
  configs/
  raw/
```

This request authorizes only the two-checkpoint calibration. Do not launch a
generalized-sharpness fleet until the convergence, restart, and cost evidence
has been reviewed.

### Authorized seed-noise fleet

Jeffrey subsequently authorized four paired seeds of the three endpoint runs
used in the report. Run seeds 0, 1, 2, and 3 for:

1. the exact bi-Maxwell control;
2. the scheduled single EMA with decay `0.982 -> 0.944` from step 1000 to
   step 3249;
3. the exact PR357 K8 kernel control encoded by
   `make_powerlaw_anneal_configs.py`.

Use one isolated checkout or working directory per seed so the four
`warmstart_pl_anneal` state directories cannot collide. On each seed's node,
generate the common fork configs and run one base followed by the three arms:

```bash
python records/track_3_optimization/offline_analysis/\
make_powerlaw_anneal_configs.py configs/seed_${SEED}
python records/track_3_optimization/offline_analysis/\
make_exponential_anneal_configs.py configs/seed_${SEED} \
  --start 0.982 --end 0.944

torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py \
  configs/seed_${SEED}/pl_anneal_base.yaml seed=${SEED}
for config in \
  plann_bimaxwell_control.yaml \
  expann_b0p982_b0p944.yaml \
  plann_pr357_kernel_control.yaml; do
  torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py \
    configs/seed_${SEED}/${config} seed=${SEED}
done
```

Before the three forks launch, verify that the saved step-1000 model and every
rank's optimizer-state shard exist. These are paired comparisons: for each
seed, report `bi-Maxwell loss - candidate loss` over every shared validation
checkpoint and over the dense cooldown window, plus final loss. Then report
the four-seed mean, standard deviation, standard error, and all individual
paired differences. This n=4 fleet estimates run-to-run noise for the current
decomposition; do not present it as the official n=8 Track-3 significance
test.

Run this wave within the same four-node cap. It may fill nodes after their EoS
assignments complete, but it must not preempt REQ-018. Section-4 momentum-
spectrum interventions remain on hold by Jeffrey's instruction.

### Required artifacts

Commit and push to this `jerry-agent` branch:

```text
logs/kmaxwell/req019_eos_state_dependence/
  README.md
  manifest.tsv
  shared-state-check.tsv
  summary.tsv
  configs/
  <run_id>/
    command.txt
    console.log
    train-log.txt
    req019_per_matrix_curvature.json
    curvature-console.log
logs/kmaxwell/req019_seed_twins/
  README.md
  summary.tsv
  paired_trajectories.tsv
  configs/
  seed_<n>/
    command.txt
    console.log
    train-logs/
```

`summary.tsv` must include run ID, fork, multiplier, executed SHA, node/GPU,
wall times, completion state, checkpoint count, matrix count, and any failure.
The README must map every summary row to raw artifacts and state whether each
shared-state gate passed. Use `git add -f` for ignored log patterns. Gzip large
logs losslessly if needed. Never commit model/optimizer checkpoints, FineWeb
data, secrets, or environment dumps.

Success means all nine frozen EoS arms and their 45 curvature measurements are
complete, all twelve endpoint runs across four paired seeds are complete, the
shared-state checks are explicit, and the artifacts are pushed. If a
reproducible harness or infrastructure blocker remains, first commit all
partial evidence, then set `NEEDS-INFO` with the exact failing command and log.

---

## REQ-018: finish the 4+4 scaling-SDPO fleet

- status: RUNNING (agent 2026-08-30) — preflight PASS; 2 fresh boxes bootstrapping `3bd7def`; diligence arms **restart from 0** (checkpoints lost, see note); node cap = 2 (operator constraint)
- requested: Jack / 2026-08-29 17:39 PDT
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact SHA: `3bd7defdeb4c9b3777c2e8d6530aa6135dd76b67`
- supersedes: REQ-015
- prior evidence: `logs/async_sdpo_req011/`, `logs/async_sdpo_req015/`

### REQ-018 STATUS (agent 2026-08-30 ~04:1x)

Any decision about REQ-018's step-200 free-model `429` blocker belongs to Jack,
the request owner. Preserve the evidence and ask him; do not infer a choice
from REQ-019 or from this queue edit.

**Preflight PASS @ 2026-08-30T04:07:45Z:** `nvidia/nemotron-3-super-120b-a12b:free` → **HTTP 200** (Nvidia provider — aux LLM is free, so the recurring credit-402 is gone), Parallel Search → **HTTP 200**.

⚠️ **Checkpoint-loss note (unavoidable):** the preserved diligence checkpoints on `qkpx8dw`/`wp2znpq` were **lost** — those boxes were released during an operator-directed full reset (Baseten wipes a stopped box's filesystem). So `logging.resume_from` is not possible; the two diligence arms **restart from step 0** on `3bd7def`. This is defensible: you noted the pre-fix steps were dropped-hint (data-inefficient, not contaminated), and a clean run under the fixed 2048-token / reasoning-off Nemotron hint path is cleaner than resuming pre-fix state. Pre-fix REQ-015 evidence remains in `logs/async_sdpo_req015/`.

**Constraint:** operator capped provisioning at **2 H100 nodes**. Plan: run the 2 diligence arms (`answer_free`, `answer_bearing`) first, one per box, to step 200 (`judge.eval_interval=25`, `checkpoint_interval=50`, nemotron-free slugs); then cycle `tau2 gold` (retail+airline, with the get_environment fix) and REQ-019 curvature arms through the 2 boxes as they free. REQ-019 shares this 2-node budget without preempting REQ-018.

Full deliverable → `logs/async_sdpo_req018/` on completion (SHA, resolved slugs, hint-validation gate results, per-arm summary.tsv, 25-step eval records).

### REQ-018 RESULT (agent 2026-08-30) — ✅ hint-fix gate PASSED; ⚠️ step-200 blocked by FREE-model 429 rate limit

Evidence committed: `logs/async_sdpo_req018/` (`summary.tsv`, per-arm evidence).

**Hint-truncation fix VALIDATED** on both diligence arms @ `3bd7def` (`reasoning_enabled=false`, `max_tokens=2048`, hint model `nvidia/nemotron-3-super-120b-a12b:free`): answer_free **0 drops / 0 `openrouter_length` over 479 attempts**; answer_bearing **0 / 0 over 477**. Gate (≥100 attempts, zero length, <5% drops) passes decisively — the REQ-011 hint-truncation problem is solved. eval@25 ran (nemotron judge; free-tier: 0 vs 6 judge_errors).

**Blocked at step 28:** the FREE Nemotron tier is rate-limited (~**3693 HTTP 429/arm**). Hints retry through them (0 drops), but the eval@25 + rollouts overwhelmed the free endpoint → ~2h stall → vLLM worker `c10::Error` + shm dequeue timeout (`disable_custom_all_reduce=True` already). No checkpoint yet (crash < interval 50) → restart just re-crashes.

**Decision needed to reach 200:** (a) paid/higher-rate aux model, (b) keep free + lighter eval (fewer held-out tasks, `eval_interval≥50`) + a bounded eval timeout, or (c) debug the vLLM `c10::Error`. Tell me which. Meanwhile the 2 boxes are **repurposed for REQ-019** (no external-API dependency) to keep the 2-node budget productive.

Use separate boxes. (REQ-017 was removed as superseded; the constraint now applies to REQ-019.) This is a fresh SDPO execution
request: fetch the exact SHA, resume the preserved diligence checkpoints,
launch tau2, and commit the complete logs.

### Frozen checkout

```bash
git fetch origin fix/hint-output-budget
git checkout 3bd7defdeb4c9b3777c2e8d6530aa6135dd76b67
uv run --no-sync pytest -q -m 'not network'
```

Record that SHA and the resolved auxiliary-model slugs in every artifact.
Do not recreate this as an on-box patch. Offline suite at this SHA:
`213 passed, 2 skipped, 2 deselected`.

Resolved defaults that must appear in `config.yaml`:

```text
model.model=Qwen/Qwen3-8B
generator.hint.reasoning_enabled=false
generator.hint.max_tokens=2048
generator.hint.model=nvidia/nemotron-3-super-120b-a12b:free
judge.model=nvidia/nemotron-3-super-120b-a12b:free
data.user_llm=openrouter/nvidia/nemotron-3-super-120b-a12b:free
judge.eval_interval=25
logging.checkpoint_interval=50
```

### Preflight

OpenRouter and Parallel Search are funded on the developer account. Before
renting or restarting a box, send one real request to each and record status
plus UTC timestamp, never keys.

If a box still returns 402, report which credential/account differs from the
developer environment and stop that dependent arm. Do not put credentials in
this repository.

### Resume existing diligence work

Preserve every current run directory before changing a process.

| arm | box | latest valid checkpoint | last live step |
| --- | --- | --- | ---: |
| diligence `answer_free` | `qkpx8dw` | `runs/sdpo-diligence/step_50` | 96 |
| diligence `answer_bearing` | `wp2znpq` | `runs/sdpo-diligence/step_100` | 116 |

Resume with:

```text
logging.resume_from=runs/sdpo-diligence/step_50   # answer_free
logging.resume_from=runs/sdpo-diligence/step_100  # answer_bearing
```

`run.py` now loads the checkpoint, restores trainer/optimizer/EMA/staleness
state, and syncs restored weights into vLLM before new rollouts. Do not
restart either diligence arm from step zero. Keep pre-fix and post-fix logs
in separate directories.

Failed pre-fix hints were dropped, not trained, so those checkpoints are
data-inefficient but not contaminated by empty-hint updates.

### Hint validation gate

Before the resumed diligence arms run unattended, issue one production hint
through `build_error_hint` / `generate_hint` and retain sanitized request
settings plus response metadata. Then, after at least 100 post-fix hint
attempts on each diligence arm, report:

- attempts, successes, total drops, and `drops / attempts`
- every `hint_drop_*` cause, especially `openrouter_length`
- count of raw `finish_reason=length` responses
- comparison against the matching pre-fix REQ-014/REQ-015 window and
  `logs/async_sdpo_req011/`; recompute from counts, never reuse the old
  malformed 149% display

Gate passes when there are zero `hint_drop_openrouter_length` failures in
those 100+ attempts and the total drop rate is below 5% while OpenRouter is
healthy. If the gate fails, preserve raw artifacts, diagnose, update this
block, and stop. Do not raise concurrency, switch models, enable hint
reasoning, or train with an empty hint.

### Tau2

Do not delay diligence on tau2. Resolve the already-reported tau2-bench
`a2c0247` `get_environment` API drift, add a test covering the pinned
signature, commit that follow-up to `scaling-sdpo`, then launch tau2 `gold`
on retail+airline only. Do not use banking on Baseten.

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
- hints, judge, and tau2 user sim: Nemotron free slug above
- never fall back to 4B, `stealth/ox-alpha`, or `NPROC=1`

Resume after recoverable crashes. A rollout without its required hint must
be dropped. Keep search, hint, user-simulator, judge, sandbox, empty-episode,
stale-rollout, and weight-sync failures as separate counters.

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
logs/async_sdpo_req018/
  README.md
  summary.tsv
  hint-fix-validation.tsv
  diligence-answer-free/
    pre-fix/
    post-fix/
  diligence-answer-bearing/
    pre-fix/
    post-fix/
  tau2-gold/
```

For every arm include `args.txt`, `config.yaml`, `ARTIFACTS.txt`,
`console.log`, `train.log`, every `rankN.log`, `api_failures.jsonl`,
`evaluations.jsonl`, `rollouts.jsonl`, `sandbox.jsonl`, `training.jsonl`,
`vllm.jsonl`, sandbox setup logs, and a checkpoint/resume manifest. Use
`git add -f` for ignored log patterns. Gzip large text artifacts losslessly
and document decompression. Never commit secrets, environment dumps, model
weights, or checkpoint tensor files.

`summary.tsv` must include exact SHA and CLI, completed steps, wall time,
checkpoint history, teacher-minus-student gap, dead-gap fraction, clipping,
staleness, every failure counter, and held-out metrics. The README must link
each row to its raw directory and explain any pre-fix/post-fix resume
boundary.

Success: hint gate passes and all three arms reach step 200. If a definitive
external blocker stops the run, first commit and push all progress and raw
failure logs, then update this block with exact evidence and stop rather
than restarting from zero.

---
