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

## REQ-019: 9-GPU local-hint SDPO fleet

- status: OPEN
- requested: Jack / 2026-08-29 22:30 PDT
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact SHA: `e2ff7181305b1f6066ff059ae97bb36993371cd5`
- supersedes: REQ-018
- prior evidence: `logs/async_sdpo_req011/`, `logs/async_sdpo_req015/`

Request a **9-GPU** box. Jack has operator permission to use a box. The
operator default of 2 GPUs is not enough and must not be used. Do not
self-cap at 2, do not shrink to 3+4+1 on an 8-GPU box, and do not fall
back to OpenRouter hints or `NPROC=1`. If `nvidia-smi` / 
`torch.cuda.device_count()` shows fewer than 9 visible devices, record
the listing, update this block, and stop.

Do not preempt REQ-017. Use a separate 9-GPU box.

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

`run.py` loads the checkpoint, restores trainer/optimizer/EMA/staleness
state, and syncs restored weights into the **rollout** vLLM before new
rollouts. The hint engine is frozen and must not receive those weights.
Do not restart either diligence arm from step zero.

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
logs/async_sdpo_req019/
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

---

## REQ-017: broad momentum-kernel discovery maps

- status: OPEN
- requested: OpenAI Codex for Jeffrey Cheng / 2026-08-29 UTC

Use the available fleet (historically six simultaneous 8xH100 workstations) as
an adaptive discovery engine. This supersedes REQ-016's narrow n=8 plan. Do
not run seed fleets: all discovery runs are seed 0 from a common frozen
step-1000 state. Stop and preserve every workstation when its assigned work is
harvested.

### Code and scientific baseline

- source repo: `https://github.com/jacknzheng/kmaxwell-sota`
- source branch: `codex/momentum-kernel-schedules`
- exact source commit: `9a82be0161100256c450fe4479df02910f008dcb`
- single-EMA implementation: `AnnealedDecayMuon`
- power-law implementation: `AnnealedWeightsMuon` plus
  `finite_history_power_law_weights`
- generators and summarizer: `records/track_3_optimization/offline_analysis/`

Run on the plain Track-3 bi-Maxwell/#339 ablation stack, not #46 CWD: CWD's
tail EMA is known to mask momentum-schedule effects. Preserve every
non-momentum choice. Use the exact source implementation directly where
compatible; otherwise port it mechanically and document the diff and executed
SHA. In particular, keep the scheduled EMA decay separate from fixed Nesterov
blend `mu=0.95`, and preserve power-law finite-history normalization, invisible
stream warming, and endpoint indexing.

For seed 0, run the unchanged baseline to step 1000 once, save a complete
checkpoint, and fork every arm from it. Verify fork equality. Continue every
arm through step 3250 with dense cooldown validation; do not stop at first
crossing.

### Wave 0: only two parity anchors

1. Exact bi-Maxwell control.
2. Scheduled single EMA beta 0.978 -> 0.952.

Compare these with the supplied A100 observations (3.27836 and 3.27804 final,
respectively). Small hardware numeric differences are acceptable; investigate
large trajectory or ordering disagreement before launching the maps. Do not
duplicate any other A100 cells merely for parity.

### Map A: scheduled single EMA, 36 cells

Specify endpoints in EMA mean age `a`, converting with `beta=a/(a+1)`. For this
first map retain the source implementation's interpolation linear in beta.

```text
start mean age in {12, 19, 30, 45, 60, 90}
end mean age   in {8, 12, 19, 26, 38, 60}
```

Run the complete Cartesian product, including increasing, decreasing, and
constant/near-constant memory. This deliberately removes the monotonicity
constraints from REQ-016.

### Map B: scheduled power law, 49 cells

Use the normalized, finite-history endpoint solver and interpolate the two
realized endpoint kernels as implemented in the pinned source.

```text
gamma_1000 in {0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75}
gamma_end  in {0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75}
```

Run the complete Cartesian product unless a numerical correctness failure is
found. Signed mixture coefficients are expected; non-finite training is not.

### Scheduling and adaptive execution

Keep all available boxes busy and dispatch disjoint cells. Interleave maps A
and B enough that an implementation issue in one family does not idle the
fleet. Commit a progress snapshot after every six-run wave (or approximately
hourly) with completed cells and live assignments.

After the two complete maps, do not start seed replication. Use remaining
capacity adaptively in this order:

1. If an optimum is on a boundary, extend that boundary with one-dimensional
   coarse points; do not reflexively launch another full square grid.
2. At the best two single-EMA endpoint pairs, compare interpolation linear in
   beta, mean age, and flip transmission `(1-beta)/(1+beta)`.
3. At the best two endpoint pairs from each family, sweep switch step in
   `{500, 750, 1000, 1250, 1500}` while holding the end fixed at 3249.
4. If capacity remains, sweep anneal completion in
   `{2000, 2500, 2900, 3250}` at the best switch/endpoint settings.

If results make an adaptive choice ambiguous, update REQ-017 to NEEDS-INFO
with the relevant small table rather than spending dozens of runs on both
branches. The human collaborator's judgment is an intended part of the loop.

### Analysis and artifacts

For every run retain the full validation trajectory and report final loss plus
paired candidate-minus-bi-Maxwell differences over all common checkpoints and
the dense cooldown window. Use `control - candidate` for columns labeled
improvement.

At step 1001, midpoint, and final, tabulate each realized kernel's:

- mean age and age variance
- flip transmission
- squared-kernel/noise gain
- mass in ages 0--1, 0--4, 0--16, and beyond 64

Produce heatmaps/tables for both endpoint maps and simple regressions or rank
correlations testing whether performance collapses onto any descriptor. These
are diagnostics, not proof of sufficiency.

Commit code, configs, commands, raw logs, and:

```text
logs/kmaxwell/req017_momentum_maps/
  README.md
  progress.tsv
  single_ema_map.tsv
  powerlaw_map.tsv
  trajectories.tsv
  kernel_descriptors.tsv
  fork_manifest.tsv
```

The completion criterion is broad empirical maps and clearly identified next
questions, not Track-3 statistical significance. Record exact source/executed
SHAs, hardware, runtimes, failures, and every adaptive follow-up decision.
