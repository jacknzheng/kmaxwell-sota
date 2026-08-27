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

Next request number: **REQ-013**.

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

## REQ-011: finish the 4+4 async-sdpo fleet

- status: RUNNING
- requested: Jack / 2026-08-26 12:25 PDT
- repo: https://github.com/jacknzheng/async-sdpo
- base: `main` at `c3f6139`, plus the rebased REQ-002 logging patch

Continue the three currently running 8×H100 arms to
`trainer.total_steps=200`. Do not restart healthy runs merely because this
queue file was compacted.

### Frozen runtime

- 4 vLLM rollout GPUs + 4 FSDP2 trainer ranks
- `model.model=Qwen/Qwen3-8B`
- `trainer.mini_batch_size=2`
- hints and diligence judge: `z-ai/glm-5.3-flash`
- tau2 user simulator: `openrouter/z-ai/glm-5.3-flash`
- same already-provisioned environment variables and persistent caches
- never fall back to 4B or `NPROC=1`

`stealth/ox-alpha` is retired and returns 404. OpenRouter identified
`z-ai/glm-5.3-flash` as the same underlying model's permanent slug. Do not
switch back to the retired slug.

The 27B model was attempted with 128 GB shared memory but vLLM TP=4 failed in
`custom_all_reduce.cuh` because the workstation needs
`NCCL_P2P_DISABLE=1`. The documented 8B fallback is proven and is now the
frozen model for these runs.

### Active arms

1. tau2 `gold`
2. diligence `answer_free`
3. diligence `answer_bearing`

All three have passed initialization, rollout, compilation, backward, weight
sync, and optimizer steps. Last confirmed live diagnostics:

| arm | confirmed step | teacher−student gap | staleness |
| --- | ---: | ---: | ---: |
| tau2 `gold` | 5 | −0.0397 | 1.35 (max 3) |
| diligence `answer_free` | 19 | −0.1581 | 1.00 (max 2) |
| diligence `answer_bearing` | 25 | −0.0652 | 1.13 (max 2) |

### Reliability requirements

- Keep the OpenRouter model fixed to GLM-5.3 Flash.
- Record hint failures by cause. A high hint-drop rate is a data-efficiency
  issue; if it prevents progress, reduce concurrency or increase bounded
  retries rather than silently training with empty hints.
- A rollout with no hint must be dropped because an unhinted teacher is
  effectively the student and contributes a near-zero SDPO signal.
- Diligence judge failures are off the gradient path, but must be reported
  explicitly rather than silently scored as model failures.
- Keep tau2 sandbox failures separate from policy failures.

### Completion and write-back

For each arm, preserve the run directory and produce `summary.tsv` with:

- repository SHA, full CLI, resolved model and hint/judge slugs
- completed steps and wall time
- mean teacher−student gap and fraction with `abs(gap) < 1e-3`
- advantage and low/high ratio clipping fractions
- mean/max staleness
- hint attempts, successes, drops, and drops by cause
- sandbox failures and empty episodes
- held-out metric (`pass1` for tau2; rubric judge score and judge-error count
  for diligence)
- paths or URLs for console logs, train logs, config, checkpoints, and W&B

When all three arms finish—or if a definitive blocker stops them—update this
block with the final result and status. Do not launch unrelated nanogpt work
while these active SDPO runs need the boxes.
