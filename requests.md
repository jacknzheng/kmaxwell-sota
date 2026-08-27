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

- status: NEEDS-INFO — hint fix deployed+validated, but **(A) both OpenRouter & Parallel Search return 402 again**, and **(B) resume-from-checkpoint is not wired into run.py in b7b0bda**. Per your "external blocker → preserve, report, stop" clause: stopped, checkpoints preserved, not restarting from zero.
- requested: Jack / 2026-08-27 11:28 PDT

### REQ-015 STATUS (agent 2026-08-27 ~22:4x)

**Hint fix DEPLOYED + config VALIDATED.** Fetched `fix/hint-output-budget`, checked out `b7b0bda65805506ea4eafa7f12bc0fd7d4c6a48a` on both diligence boxes (HEAD SHA verified `=b7b0bda…`), offline suite passed. Resolved hint config confirmed via `train.config.Config()`:
```
generator.hint.reasoning_enabled = False   ✓
generator.hint.max_tokens        = 2048    ✓
model                            = z-ai/glm-5.3-flash  ✓
```
(The ≥100-attempt live validation gate — zero `hint_drop_openrouter_length`, total drops <5% — **could not run**: OpenRouter is 402, see below.)

**BLOCKER A — both paid services 402 again (preflight @ 2026-08-27T22:40:43Z):**
- OpenRouter GLM-5.3 Flash: **HTTP 402**
- Parallel Search (`api.parallel.ai`): **HTTP 402**

Both balances drained again during the multi-hour runs. Per your step-1 rule (and the closing "external blocker" clause), I did not burn GPUs retrying. **Fund both services** to proceed. (This is the 3rd credit-exhaustion event of the campaign — the fleet burns balance fast; a larger buffer would prevent the recurring stalls.)

**BLOCKER B — resume-from-checkpoint is not wired into b7b0bda's entrypoint.** `train/trainer.py` defines `load_state_from_checkpoint(...)` (and the save side works — `runs/sdpo-diligence/step_N/state.pt` written every 50 steps), but a whole-tree grep finds **no caller**: `run.py`'s `SDPOTrainer(...)` construction (`run.py:187`) and `train()` (`run.py:474`) never load a checkpoint, and the loop starts at `trainer.state.step = 0`. So "resume from the latest valid checkpoint, do not restart from zero" is **not achievable as-committed** — the checkpoints save but are never loaded on launch (the "verified resume" is a unit-tested mechanism, not integrated into the run path). **Decision needed:**
- (a) I wire the resume-load into `run.py` (detect latest `step_N` in `output_dir` → `torch.load(state.pt)` → the existing `load_state_dict` path → set start step) and commit to `scaling-sdpo` — this is the "verified resume" wiring REQ-014 step 4 asked for and is absent; ~contained change but touches your training entrypoint, or
- (b) accept a **clean restart from 0 on b7b0bda** — arguably *better* here: the pre-fix steps (dil_free→96, dil_bearing→116) were trained with the truncated-hint bug b7b0bda fixes, so carrying them forward would contaminate the post-fix run. Say which you prefer.

**State preserved (not restarted):** diligence processes stopped at checkpoint boundaries. Latest valid checkpoints on-box: **dil_free = `step_50`** (was at step 96), **dil_bearing = `step_100`** (was at step 116). ⚠️ The 3 boxes (`qkpx8dw` dil_free, `wp2znpq` dil_bearing, `w5ymlv3` tau2) are **held up idle** to preserve these on-box checkpoints — stopping a Baseten box wipes its filesystem, and the state.pt tensors are too large to archive to git (you also forbid committing checkpoint tensors). If this will be a while, tell me and I'll stop the boxes (forfeiting the checkpoints → restart-from-0 when funded).

**tau2 `gold` follow-up (get_environment API drift fix + test + commit):** not done yet — it needs push access to `scaling-sdpo` and a funded OpenRouter/Parallel to smoke-test the fixed path, so it's gated on Blocker A too.

**To unblock:** (1) fund OpenRouter + Parallel Search; (2) answer Blocker B (wire-resume vs restart-0); (3) if you want the tau2 harness fix committed to `scaling-sdpo`, confirm I have push access to that repo. Then I resume/relaunch the diligence arms on b7b0bda, run the ≥100-attempt hint-fix validation gate, fix+launch tau2, and produce the full `logs/async_sdpo_req015/` deliverable.

---

- repo: https://github.com/jacknzheng/scaling-sdpo
- branch: `fix/hint-output-budget`
- exact base: `b7b0bda65805506ea4eafa7f12bc0fd7d4c6a48a`
- supersedes: REQ-014 execution instructions; preserve and reuse every
  REQ-014 checkpoint and artifact
- prior evidence: `logs/async_sdpo_req011/`

Deploy the exact commit above before continuing the API-hint arms. The prior
OpenRouter failures were not primarily a balance problem: raw responses showed
`finish_reason=length` with no visible content because GLM reasoning consumed
the 1,024-token hint budget. Commit `b7b0bda`:

- disables reasoning only for hint generation while preserving judge behavior
- raises the configurable visible hint budget to 2,048 tokens
- records `hint_drop_openrouter_length` separately
- logs cumulative hint drops as `drops/attempts` on every training line

The commit is locally verified with:

```text
210 passed, 2 skipped, 2 deselected
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
   checkpoints on `b7b0bda`. Do not restart either from step zero.
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
model=z-ai/glm-5.3-flash
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
switch models, enable reasoning, or train with an empty hint.

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
- hints and diligence judge: `z-ai/glm-5.3-flash`
- tau2 user simulator: `openrouter/z-ai/glm-5.3-flash`
- checkpoint interval no larger than 50 steps
- never fall back to 4B, `stealth/ox-alpha`, or `NPROC=1`

Resume after recoverable crashes. Search, hint, user-simulator, judge,
sandbox, empty-episode, stale-rollout, and weight-sync failures must remain
separate. A rollout lacking its required hint must be dropped.

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
`rollouts.jsonl`, `sandbox.jsonl`, `training.jsonl`, `vllm.jsonl`, sandbox
setup logs, and a checkpoint/resume manifest. Use `git add -f` for ignored log
patterns. Gzip large text artifacts losslessly and document decompression.
Never commit secrets, environment dumps, model weights, or checkpoint tensor
files.

`summary.tsv` must include exact code SHA and CLI, completed steps, wall time,
checkpoint history, teacher-minus-student gap, dead-gap fraction, clipping,
staleness, every failure counter, and held-out metrics. The README must link
each summary row to its raw artifact directory and explain any resumed
pre-fix/post-fix boundary.

Success requires the hint validation gate above and all three arms reaching
step 200. If a definitive external blocker prevents completion, first commit
and push all progress and raw failure logs, then update this block with exact
evidence and stop rather than restarting from zero.
