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

Next request number: **REQ-016**.

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

## REQ-015: deploy the hint truncation fix and finish scaling-SDPO

- status: OPEN — blockers resolved in `3bd7def`; resume both diligence
  checkpoints and rerun the live validation
- requested: Jack / 2026-08-27 11:28 PDT

### BLOCKER RESOLUTION (Jack, 2026-08-27 16:17 PDT)

Use `fix/hint-output-budget` at exact commit
`3bd7defdeb4c9b3777c2e8d6530aa6135dd76b67`. This supersedes `a075254`.
It switches hints, the diligence judge, and the tau2 user simulator to
`nvidia/nemotron-3-super-120b-a12b:free`; wires checkpoint loading into
`run.py`; restores model, optimizer, step, policy version, advantage EMA, and
producer staleness state; and synchronizes restored weights to vLLM before
new rollouts. It also runs held-out evaluation synchronously at every
25-step boundary and writes full per-task and aggregate records for both
benchmarks to `evaluations.jsonl`.

Fresh developer preflights passed: OpenRouter chat HTTP 200, the production
Nemotron hint path returned nonempty guidance, the structured judge parsed
one criterion, and Parallel Search returned a nonempty result with a session
ID. The offline suite is `212 passed, 2 skipped, 2 deselected`.

Resume `qkpx8dw` with
`logging.resume_from=runs/sdpo-diligence/step_50` and resume `wp2znpq` with
`logging.resume_from=runs/sdpo-diligence/step_100`. Failed pre-fix hints were
dropped rather than trained, so these checkpoints are data-inefficient but
not contaminated by empty-hint updates. Re-run both service preflights on the
boxes, then run the required 100-attempt post-fix gate. If a box still returns
402, report which credential/account differs from the now-working developer
environment; do not put credentials in this repository.

---

- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact base: `3bd7defdeb4c9b3777c2e8d6530aa6135dd76b67`
- supersedes: REQ-014 execution instructions; preserve and reuse every
  REQ-014 checkpoint and artifact
- prior evidence: `logs/async_sdpo_req011/`

Deploy the exact commit above before continuing the API-hint arms. The prior
OpenRouter failures were not primarily a balance problem: raw responses showed
`finish_reason=length` with no visible content because reasoning consumed the
1,024-token hint budget. The branch now:

- disables reasoning only for hint generation while preserving judge behavior
- raises the configurable visible hint budget to 2,048 tokens
- records `hint_drop_openrouter_length` separately
- logs cumulative hint drops as `drops/attempts` on every training line
- uses `nvidia/nemotron-3-super-120b-a12b:free` for all OpenRouter auxiliary
  roles
- supports explicit and `latest` checkpoint resume before rollout begins

The commit is locally verified with:

```text
212 passed, 2 skipped, 2 deselected
```

Do not recreate this as an uncommitted on-box patch. Fetch the named branch,
verify that `HEAD` is the exact SHA above, run the offline suite, and record
the SHA and resolved hint config in every run artifact.

### Ordering and existing work

1. Preserve the complete current REQ-014 run directories and identify the
   latest valid checkpoint for each arm before changing a process.
2. The existing diligence processes were launched from `1e84424` and do not
   contain the truncation fix. Gracefully stop each at a checkpoint boundary,
   then resume `answer_free` and `answer_bearing` from their latest valid
   checkpoints on `3bd7def`. Do not restart either from step zero.
3. Keep pre-fix and post-fix logs in separate, clearly named directories so
   the hint-drop comparison is auditable.
4. The tau2 `gold` arm does not use API-generated teacher hints. Do not delay
   the diligence fix on tau2. Resolve the already-reported tau2-bench
   `a2c0247` environment API drift, add a test covering the pinned signature,
   commit that follow-up to `scaling-sdpo`, and then launch/resume tau2
   `gold` on retail+airline. Do not use banking on Baseten.

### Required hint-fix validation

Before the resumed diligence arms run unattended, issue a real hint request
through the production code path and retain the sanitized request settings
and response metadata. Confirm:

```text
generator.hint.reasoning_enabled=false
generator.hint.max_tokens=2048
model=nvidia/nemotron-3-super-120b-a12b:free
```

Never log keys. Parallel Search and OpenRouter are funded now; preflight each
with one real request and record status and timestamp. A transient service
failure should be logged and retried or dropped according to the committed
policy, not misreported as policy quality.

For each diligence arm, report a post-fix checkpoint after at least 100 hint
attempts:

- attempts, successes, total drops, and `drops / attempts`
- every `hint_drop_*` cause, especially `openrouter_length`
- number of API requests whose raw metadata reports `finish_reason=length`
- comparison against the matching pre-fix REQ-014 window and the archived
  REQ-011 evidence; recompute from counts and do not reuse the historical
  malformed 149% display

The validation gate passes when there are zero
`hint_drop_openrouter_length` failures in at least 100 post-fix attempts and
the total hint-drop rate is below 5% while OpenRouter is healthy. If the
length cause remains nonzero or total drops remain at or above 5%, preserve
the raw failure artifacts, diagnose the remaining cause, update this request,
and do not claim the fix succeeded. Do not silently increase concurrency,
switch away from the specified Nemotron model, enable hint reasoning, or
train with an empty hint.

### Frozen experiment

Finish all three arms at `trainer.total_steps=200`:

1. diligence `answer_free`
2. diligence `answer_bearing`
3. tau2 `gold` on retail+airline

Keep the proven runtime:

- 4 vLLM rollout GPUs + 4 FSDP2 trainer ranks per arm
- `model.model=Qwen/Qwen3-8B`
- `trainer.mini_batch_size=2`
- `generator.engine.max_model_len=32768` for tau2
- hints and diligence judge: `nvidia/nemotron-3-super-120b-a12b:free`
- tau2 user simulator:
  `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- checkpoint interval no larger than 50 steps
- never fall back to 4B, `stealth/ox-alpha`, or `NPROC=1`

Resume after recoverable crashes. Search, hint, user-simulator, judge,
sandbox, empty-episode, stale-rollout, and weight-sync failures must remain
separate. A rollout lacking its required hint must be dropped.

### Evaluation logging contract

Run held-out evaluation at steps 25, 50, 75, 100, 125, 150, 175, and 200.
Commit the complete `evaluations.jsonl` from every arm and baseline; a W&B
screenshot or aggregate-only summary is not a substitute.

For diligence, retain every held-out task's query, generated response, token
counts, normalized and raw judge scores, per-section earned/possible/fraction,
and judge error. Retain aggregate `judge_score`, `judge_n`, `judge_errors`,
the three section scores, requested count, and rollout-error count.

For tau2, retain every held-out task's domain, query, complete response or
transcript, token counts, pass^1 score, and rollout error. Retain aggregate
overall and per-domain pass^1, sample counts, requested count, and
rollout-error count. Every row and aggregate must identify the exact
`launched_at_step` and `policy_version`.

Do not skip a 25-step boundary because a prior eval is still running. Commit
`evaluation_started`, task-completed/task-failed, and
`evaluation_completed`/`evaluation_failed` records even when an API, sandbox,
or rollout failure occurs.

### Required artifacts

Commit and push the complete deliverable to this `jerry-agent` branch:

```text
logs/async_sdpo_req015/
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

For every arm include raw `args.txt`, `config.yaml`, `ARTIFACTS.txt`,
`console.log`, `train.log`, every `rankN.log`, `api_failures.jsonl`,
`evaluations.jsonl`, `rollouts.jsonl`, `sandbox.jsonl`, `training.jsonl`,
`vllm.jsonl`, sandbox setup logs, and a checkpoint/resume manifest. Use
`git add -f` for ignored log patterns, including every `evaluations.jsonl`.
Gzip large text artifacts losslessly and document decompression. Never commit
secrets, environment dumps, model weights, or checkpoint tensor files.

`summary.tsv` must include exact code SHA and CLI, completed steps, wall time,
checkpoint history, teacher-minus-student gap, dead-gap fraction, clipping,
staleness, every failure counter, and held-out metrics. The README must link
each summary row to its raw artifact directory and explain any resumed
pre-fix/post-fix boundary.

Success requires the hint validation gate above and all three arms reaching
step 200. If a definitive external blocker prevents completion, first commit
and push all progress and raw failure logs, then update this block with exact
evidence and stop rather than restarting from zero.
