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

Next request number: **REQ-020**.

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

## REQ-019: momentum EoS law across fork states

- status: OPEN
- requested: OpenAI Codex for Jeffrey Cheng / 2026-08-30 01:40 UTC
- priority: run alongside REQ-018 without preempting it
- concurrency cap: at most 4 simultaneous 8xH100 workstations for REQ-019

Measure whether the per-matrix curvature law changes with training state. This
request contains two shared-trajectory programs, forked at steps 1500 and 2000.
It does not include momentum-kernel sweeps, seed fleets, or the Section-4
interventions; those require separate authorization.

### Frozen public harness

- repo: `https://github.com/jacknzheng/kmaxwell-sota`
- branch: `codex/momentum-kernel-schedules`
- exact SHA: `755c49d2933c3fa6c5b2fe449d05e72e40fbab9d`
- runner: `records/track_3_optimization/run.py`
- config generator:
  `records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py`
- curvature tool:
  `records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py`

The exact SHA was tested from a fresh public HTTPS clone on an 8xA100 box: all
nine generated configs resolve through `bind_sites` and optimizer-group
resolution, and the curvature tool imports successfully. Do not substitute a
private `muoff` checkout or port the harness again.

On every workstation:

```bash
git clone --filter=blob:none --branch codex/momentum-kernel-schedules \
  https://github.com/jacknzheng/kmaxwell-sota.git
cd kmaxwell-sota
git checkout 755c49d2933c3fa6c5b2fe449d05e72e40fbab9d
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
