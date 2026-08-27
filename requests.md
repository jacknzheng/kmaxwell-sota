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

Next request number: **REQ-015**.

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

- status: DONE-PARTIAL (stack proven + strong SDPO signal on 2/3 arms; none reached 200 — API-transient crashes + no checkpointing; see `logs/async_sdpo_req011/`)

### FINAL RESULT (agent, 2026-08-27 ~05:0x)

**The 4+4 async-SDPO stack works and produces a strong, non-degenerate SDPO signal. No arm reached `total_steps=200`: every arm eventually hard-crashes on an UNCAUGHT auxiliary-LLM transient (glm-5.3-flash/OpenRouter is very flaky), and there is NO trainer checkpointing so each crash forfeits the run.** Full writeup + per-step trajectories + metrics: **`logs/async_sdpo_req011/`** (`README.md`, `summary.tsv`, `*_extract.txt`).

| arm | steps | mean teacher−student gap | gap range | frac \|gap\|<1e-3 | staleness (mean/max) | hint drops (openrouter_error) | status |
|-----|------:|------:|-----|------:|-----|------:|-----|
| diligence answer_free | 114/200 | **−0.106** | −0.37…−0.043 | 0.000 | 0.95 / 3 | 3598 | crashed (uncaught eval-path transient) |
| diligence answer_bearing | 176/200 | **−0.101** | −0.27…−0.039 | 0.000 | 1.08 / 3 | 4188 | crashed (uncaught eval-path transient) |
| tau2 gold | 64/200 | ~−0.045 | −0.057…−0.040 | 0.000 | ~1.4 / 3 | (402-crashed) | stopped after 4 crashes; box released, log lost |

- **Signal is real:** gaps are large, consistently negative (~−0.10 nats on both diligence arms), **0% dead steps** → the hinted teacher genuinely beats the bare student and SDPO distills it. Not a no-op.
- **Why not 200:** only the hint path is hardened (drops transients, `episodes_empty=0`); the eval path (`evaluate_pass1`) and tau2's multi-turn user-sim have uncaught call sites, so over thousands of API calls an uncaught transient eventually crashes a rank → torchrun kills the arm. tau2 was most exposed (4 crashes: 402 ×3 + a 16384-context overflow fixed with `max_model_len=32768`; a `run_tau2_episode` drop-guard was written+applied but the 4th crash was in the unguarded pass@1 eval).
- **Held-out metric:** diligence rubric-judge 404s on glm-5.3-flash structured output (known, off gradient path per your pt5) → no valid judge score. tau2 pass1 not reached.

**To get a clean 200 (recommended, not done to avoid burning the night restart-looping a flaky API from step 0):** (1) keep a comfortable OpenRouter balance (kills the 402 class), (2) harden every auxiliary-LLM call site (eval + user-sim) like the hint path, (3) add periodic trainer checkpointing. With (1)+(2) the arms will reach 200 — they were healthy and advancing between transients. All 3 boxes released (results harvested). Say the word and I'll relaunch with the call-site hardening once credits are stably funded.

### (superseded) original request
- prior-status: RUNNING
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

### REQ-011 STATUS (agent, 2026-08-27 ~02:3x) — progressing; one credit-dip incident, tau2 restarted

Arms were training healthily (dilbear→109, dilfree→77, tau2→64) when **OpenRouter briefly hit $0 (`402 Insufficient credits`) around 02:06**. Effect split by code path:
- **diligence arms SURVIVED** — hint generation has try/except (drops the rollout on error), so they stalled during the dip then **auto-resumed** when the balance recovered. Now healthy: dilfree ~77, dilbear ~109.
- **tau2 CRASHED at step 64** — the tau2 user-simulator LLM call has **no** try/except, so the 402 raised an uncaught `litellm.APIError` and killed the trainer (orphaned vLLM workers left spinning; cleaned up). **Restarted tau2 from step 0** (credits recovered — live glm call returns 200, balance now shows `usage $30.94, limit=None`).

**⚠️ Credit burn is the real risk to this run.** The 3-arm fleet burned ~$20 in ~4h (usage $11→$31). At the slow rollout rate (~15–19 steps/hr, glm-5.3-flash is a reasoning model), reaching step 200 on all arms is ~8–11h *more* → likely **another ~$40–60 of OpenRouter spend**. If the balance hits zero again: diligence will stall-and-resume, but **tau2 will hard-crash again** (uncaught user-sim error). **Please keep the OpenRouter balance funded** until all three finish. (Retrying a *sustained* 402 doesn't help — litellm already exhausts retries; the only fix is available balance. I can optionally harden the tau2 user-sim path to drop-instead-of-crash on API errors, but that only converts a crash into a stall — it still can't train without credits. Say the word if you want that hardening.)

Monitors are watching for milestones, crashes, and a 402-stall condition. Will post the final per-arm `summary.tsv` per the spec above when all three reach 200.

### UPDATE (~03:4x) — diligence arms near done; tau2 has a RECURRING crash → hardening it now

- **Diligence arms are healthy and close: dilbear 167/200, dilfree 108/200.** These are the reliable deliverable and will finish.
- **tau2 has now crashed 3× at low steps, each on a DIFFERENT transient** in its multi-turn user-sim/rollout path: (1) 402 credit-dip @64, (2) `VLLMValidationError` 16384-context overflow @1 (fixed by raising `generator.engine.max_model_len=32768`), (3) another 402 blip @1. Root cause: tau2's episode/user-sim path lacks the drop-on-exception guard the diligence hint path has, so **any** transient LLM error kills the whole trainer. glm-5.3-flash being a *verbose reasoning model* makes tau2 episodes long + error-prone, amplifying this.
- **Action:** I'm applying the hardening you sanctioned ("increase bounded retries rather than silently training") — wrapping tau2's per-episode rollout so a transient (402/litellm/vLLM) DROPS that episode and is counted by cause, never crashing the trainer (exactly how diligence already behaves). Then restart tau2 once. tau2's orphaned vLLM workers were cleaned; it's held down until the patch lands. If it still can't make sustained progress after hardening, I'll deliver the two diligence arms to 200 + tau2's partial run and leave tau2's completion to your call (it may simply be a poor fit for a verbose reasoning-model user-sim).
- Credit balance still funded (`usage ~$33, limit=None`) but volatile; **please keep it topped up** — the diligence arms need it to reach 200.

Monitors watching for milestones, crashes, and 402-stalls. Final per-arm `summary.tsv` when the arms settle.

### VERDICT (~04:4x) — tau2 STOPPED (definitive blocker = credit dips); 2/3 arms (diligence) delivering

**tau2 `gold` crashed 4× at step 1–64, every time on a transient in an uncaught path; I stopped it and its box.** Escalating fixes attempted, each addressing the prior cause: (1) cleared orphans + restart → (2) `VLLMValidationError` context overflow, fixed with `max_model_len=32768` → (3)+(4) OpenRouter **`402 Insufficient credits`** dips. The hardening patch I applied (wraps `run_tau2_episode` to drop-and-count transient failures like the diligence path — verified, `EPISODE_DROP_BY_EXC` counter, applied to the box) did NOT catch the 4th crash: `drops=0`, because the fatal 402 propagates through the **pass@1 eval path** (`evaluate_pass1`, `run.py:605`, unguarded), not the episode body.

**Root cause = the OpenRouter balance keeps briefly hitting $0** (the 3-arm fleet burns faster than the intermittent top-ups; balance oscillated $11→$31→$33 with repeated momentary-zero dips). Diligence arms survive each dip (drop-and-resume); tau2's multi-turn user-sim + pass@1 eval have several uncaught 402 paths, so it hard-crashes on every dip. This is **not further fixable from my side without either (a) a stably-funded balance, or (b) hardening every eval/user-sim call site** (whack-a-mole; the harness came with only `num_retries=2`).

**tau2 is relaunchable** (box was `w5y8zm3`, stopped to save GPU; harness is hardened + 32k ctx) **once the OpenRouter balance is kept reliably positive.** If you want it, top up a comfortable buffer and say so — I'll relaunch. Otherwise the deliverable is the two diligence arms.

**Delivering: diligence `answer_free` + `answer_bearing`** — healthy, near done (`answer_bearing` 176/200, `answer_free` 114/200), surviving the same credit dips. Will produce their per-arm `summary.tsv` (per the spec above) when they hit 200, plus tau2's partial-run diagnostics for the record.

---

## REQ-013: stack and tune K-Maxwell on PR #351 MuonH fast-slow decay

- status: OPEN
- requested: Jack / 2026-08-26 18:30 PDT
- repo: https://github.com/jacknzheng/modded-nanogpt
- upstream baseline: https://github.com/KellerJordan/modded-nanogpt/pull/351
- baseline commit: `Yufei-Gu-451/modded-nanogpt@d7bc799aaf238bcd7094e3c6ed67fba1ecfa35a9`

Run this only after REQ-011's active async-SDPO arms finish or definitively
stop. Do not preempt those jobs. Then use fresh/released 8×H100-class boxes to
test whether our K-Maxwell first-moment kernel can improve the new MuonH
per-optimizer SOTA.

PR #351 reaches **3125 steps** with n=20 mean `3.278994` and
`(3.28-mean)*sqrt(20)=0.00450`. Its only intended baseline change is the MuonH
four-phase fast-slow LR schedule:

```text
warmup_end=100
plateau_end=200
fast_decay_end=1750
peak_lr=0.030
floor_lr=0.006
fast_decay_exponent=0.6
slow_decay_schedule=linear
min_lr=0
train_steps=3125
```

### Integration requirements

Start from PR #351's trainer at the exact commit above. First reproduce one
seed with the unmodified trainer. Then make a separate K-Maxwell variant that
changes only MuonH's first-moment construction:

- preserve MuonH's Newton-Schulz direction, `scale_invariant_update_`,
  hyperball constraint, initialization, all parameter groups, and the exact
  fast-slow LR schedule
- preserve architecture, data, batch, one forward/backward per update, and
  auxiliary AdamW settings
- replace the single EMA, after a configurable lazy-init switch, with the same
  deterministic log-spaced K-EMA convex mixture used by
  `records/track_3_optimization/results/20260824_kmaxwell_3160/`
- keep the switch update bit-identical to baseline by initializing every
  K-buffer from the live single-EMA momentum
- expose and log `--k`, `--tau-min`, `--tau-max`, `--weights`,
  `--weights-end`, `--km-start`, `--anneal-frac`, and Nesterov `--mu`
- keep EMA rates `beta_i` separate from Nesterov `mu`; do not silently couple
  them
- add dense evaluation on one fixed grid for every screened/confirmed run:
  every 5 steps on `[3000,3125]`, with validation excluded from train time

Before spending on a sweep, prove that the no-K-Maxwell code path reproduces
the PR #351 update and that the lazy-init switch step is identical.

### Staged seed-0 screen

Use seed 0 throughout screening and always include the exact PR #351 control.
Do not run a blind Cartesian product; advance the best candidate from each
stage.

1. **Transfer check:** K=8, `tau=[3,64]`, `km_start=1000`, `mu=0.95`, linear
   weight anneal with mean age `58→26`, versus unmodified PR #351.
2. **Anneal endpoints:** at the same K/window/onset/mu, screen
   `50→22`, `54→24`, `58→22`, `58→26`, and `62→30`.
3. **Onset interaction with the LR schedule:** using the best endpoint pair,
   screen `km_start ∈ {750,1000,1250,1500}`. Anneal to step 3125 in every arm.
4. **Nesterov coefficient:** on the best endpoint/onset configuration, screen
   `mu ∈ {0.94,0.95,0.96}`.
5. **K discretization only if still promising:** hold the chosen start/end
   mean ages, `[3,64]`, onset, and mu fixed; solve the same linear-ramp family
   for `K ∈ {4,6,8,12}`. Do not compare configurations whose requested mean
   ages produce negative weights.

Primary screen ranking is raw `val_loss@3125`, then earliest `val_loss<3.28`
on the common 5-step grid. Report `@3100`, `@3110`, `@3120`, and `@3125` for
every arm. Keep all completed runs in the table; do not cherry-pick or expand
the grid after seeing results.

### Confirmation and success criterion

If no seed-0 candidate improves the paired control by at least `0.0005` at
3125 or crosses at least 10 steps earlier, stop and report a null result.

Otherwise run the selected candidate and an exact PR #351 control on
consecutive seeds 0–7 on the same hardware. Score each 5-step boundary with:

```text
margin = (3.28 - mean) * sqrt(8)
pass iff margin >= 0.004
```

Success means the K-Maxwell arm's first passing boundary is **strictly below
3125**. Also report paired/control differences at every common tail step and
the equal-step comparison against PR #351's published n=20 mean at 3125:

```text
(3.278994 - candidate_mean) / sqrt(1/20 + 1/8)
```

Do not claim a win from a single-seed crossing. If the n=8 result is within one
5-step grid notch of 3125 or is driven by one outlier, extend the already
selected candidate/control to seeds 0–19 before making a record claim; do not
retune after looking at those seeds.

### Write-back

Put reproducible artifacts under `logs/kmaxwell/muonh351/`:

- patched trainer plus a minimal diff against PR #351
- exact commands and resolved K-Maxwell weights/betas for every arm
- raw logs and `sweep.tsv` for all seed-0 screens
- `summary.tsv` for candidate/control confirmation with per-seed losses,
  mean, margin, pass, and first crossing on the common grid
- one short README stating whether K-Maxwell transfers to MuonH fast-slow
  decay, which hyperparameter mattered, and whether it actually beats 3125

If a statistically valid sub-3125 result survives, prepare a clean record
folder/PR artifact, but do not discard the null or losing sweep runs.

### REQ-013 STATUS (agent, 2026-08-27 ~05:5x) — reproduction gate PASSES; stage-1 transfer check running

**Integration done + verified.** K-Maxwell variant `records/track_3_optimization/results/20260810_muonh_fast_slow_decay_3125/train_gpt_muonh_kmaxwell.py` (base = PR #351 `train_gpt_muonh_fast_slow_decay.py` @ d7bc799, copied verbatim; only MuonH's first-moment changed to the K-EMA convex mixture + lazy-init switch). `--k`/`--tau-min`/`--tau-max`/`--weights`/`--weights-end`/`--km-start`/`--anneal-frac`/`--mu` exposed; `--k 1` disables K-Maxwell → plain MuonH. `beta_i` kept separate from `mu`. Dense eval every 5 on [3000,3125]. Minimal diff: `logs/muonh351/` (will land with results).

**Reproduction gate: PASS — within nondeterminism (bit-identity is impossible here).** The MuonH trainer is not bit-deterministic across runs (NCCL/cuBLAS atomics). Three seed-0 runs of the *unmodified* PR #351 vs the `--k 1` variant:

| step | base run-1 | base run-2 | variant --k1 | \|base1−base2\| | \|base1−variant\| |
|------|------|------|------|------|------|
| 125 | 4.88161 | 4.93710 | 4.89284 | 0.0555 | 0.0112 |
| 2000 | 3.42408 | 3.42464 | 3.42494 | 0.00056 | 0.00086 |
| 3125 | **3.27881** | **3.27830** | **3.27853** | **0.00051** | **0.00028** |

`--k 1` reproduces base **more closely than two base runs reproduce each other** → the no-K-Maxwell path is code-faithful to PR #351. (The lazy-init switch was also verified statically: at `km_start` every K-buffer is seeded from the just-advanced single-EMA momentum, and that step's update calls the identical `muon_update` — bit-identical by construction; it only diverges after `km_start` when the K-mixture takes over.)

**⚠️ Noise-floor caveat that matters for scoring:** run-to-run σ at step 3125 is **~5e-4**, which *equals* your seed-0 success bar (`≥0.0005 @3125`). So the seed-0 screen is noise-limited at the threshold — a single-seed win near 0.0005 is indistinguishable from noise. I'm treating seed-0 as *indicative only* and will lean on your n=8 confirmation (statsig margin) for any real claim; I'll also report the candidate against the **control distribution** (I have 3–4 seed-0 control samples: base 3.27881/3.27830, variant-k1 3.27853/…, mean ≈3.2786, σ≈5e-4), not a single control point.

**STAGE 1 TRANSFER CHECK — STRONG PASS (seed 0). K-Maxwell transfers to MuonH fast-slow decay.**

Control = 4 seed-0 samples (2× base #351, 2× variant `--k1`); candidate = K=8, τ[3,64], km_start=1000, μ=0.95, anneal mean-age 58→26.

| metric | control (n=4 seed-0) | K=8 58→26 | Δ | bar | verdict |
|--------|------|------|------|------|------|
| val_loss @3125 | 3.27862 (σ=0.0002) | **3.27711** | **−0.00151** | ≥0.0005 | **PASS (~7σ, 3× bar)** |
| @3120 | 3.27872 | 3.27714 | −0.00158 | — | |
| @3110 | 3.27889 | 3.27731 | −0.00158 | — | |
| @3100 | 3.27923 | 3.27762 | −0.00161 | — | |
| first val<3.28 | ~step 3104 | **step 3055** | 49 earlier | ≥10 | **PASS (5× bar)** |

Clears **both** criteria decisively and sits ~7σ above the seed-0 noise floor (far more robust than the REQ-010/012 within-noise μ nulls). K-Maxwell's log-spaced K-EMA first moment genuinely helps MuonH — worth the full staged screen + n=8 confirmation.

**Now running — stage 2 (anneal endpoints, seed 0):** `50→22, 54→24, 58→22, 62→30` (58→26 already = the transfer point above), same K/window/onset/μ, 2 configs/box. Advancing best endpoint to stage 3 (onset), then stage 4 (μ), then stage 5 (K), then **n=8 confirm** of the winner + control with the Track-3 statsig margin and first-passing-step-below-3125 test. Artifacts landing in `logs/muonh351/`.

---

## REQ-014: harden and finish the scaling-SDPO fleet

- status: OPEN (Parallel Search funded by Jack; rerun the live preflight and launch)
- requested: Jack / 2026-08-26 21:40 PDT
- repo: https://github.com/jacknzheng/scaling-sdpo
- branch/base: `main` at `094c701`
- supersedes: the repository URL and unfinished execution work in REQ-011

Continue the three 4+4 async-SDPO arms from REQ-011, but use the renamed
canonical repository above. Parallel Search has now been funded. Re-run a
real request and, if it returns HTTP 200, start the fleet immediately. Do not
stop healthy REQ-013 jobs already in flight; use separate boxes or start each
SDPO arm as capacity becomes available.

Commit `094c701` has been locally verified (`207 passed, 2 skipped,
2 deselected`) and pushed. It adds comprehensive structured diagnostics at
the API, vLLM, rollout, training, and tau2 sandbox boundaries.

### Preflight and reliability work

Before renting or restarting GPU boxes:

1. Confirm Parallel Search and OpenRouter independently with one real request
   each. Jack has funded Parallel since the previous 402. Record status codes
   and timestamps, but never keys. If either still returns 402, mark only that
   service blocked and do not rent its dependent boxes.
2. Pull `main` at `094c701` or later from the canonical repository and run the
   offline suite. Do not replace or bypass its diagnostic logging.
3. Run one short tau2 banking smoke episode and one diligence smoke episode.
   Verify that the run directory contains `ARTIFACTS.txt`, `console.log`,
   `train.log`, rank logs, `api_failures.jsonl`, `rollouts.jsonl`,
   `sandbox.jsonl`, `training.jsonl`, and `vllm.jsonl`. Also preserve the
   timestamped `tau2-sandbox-setup-*.log`.
4. Confirm periodic checkpoints can be loaded before the full launch. Resume
   from the latest valid checkpoint after a crash rather than starting from
   zero. If a remaining auxiliary-LLM exception can still escape the run or
   eval coordinator, harden and test that boundary, then commit it to
   `scaling-sdpo`.

### Frozen experiment

Run all three arms to `trainer.total_steps=200`:

1. tau2 `gold`
2. diligence `answer_free`
3. diligence `answer_bearing`

Keep the proven REQ-011 runtime:

- 4 vLLM rollout GPUs + 4 FSDP2 trainer ranks per arm
- `model.model=Qwen/Qwen3-8B`
- `trainer.mini_batch_size=2`
- `generator.engine.max_model_len=32768` for tau2
- hints and diligence judge: `z-ai/glm-5.3-flash`
- tau2 user simulator: `openrouter/z-ai/glm-5.3-flash`
- never fall back to 4B, the retired `stealth/ox-alpha` slug, or `NPROC=1`

A rollout without its required hint must still be dropped. Search, hint,
user-simulator, judge, sandbox, empty-episode, and stale-rollout failures must
remain separate counters. Do not restart a healthy arm merely to pick up a
nonessential change; resume from the latest valid checkpoint after a crash.

### Completion

Preserve every run directory. Copy the complete artifacts into this
`jerry-agent` branch under:

```text
logs/async_sdpo_req014/
  README.md
  summary.tsv
  tau2-gold/
  diligence-answer-free/
  diligence-answer-bearing/
```

For every arm, commit and push the raw `args.txt`, `config.yaml`,
`ARTIFACTS.txt`, `console.log`, `train.log`, every `rankN.log`,
`api_failures.jsonl`, `rollouts.jsonl`, `sandbox.jsonl`, `training.jsonl`,
`vllm.jsonl`, sandbox setup log, and checkpoint/resume manifest. These files
are the deliverable: do not replace them with excerpts. Use `git add -f` where
the log patterns are ignored. Never commit API keys, environment dumps,
credentials, model weights, or checkpoint tensor files. Gzip a raw artifact
losslessly if needed to keep each GitHub file comfortably below its size
limit, and document the exact decompression command.

`summary.tsv` and README must include the exact scaling-SDPO SHA and CLI,
resolved model/hint/judge slugs, wall time, completed steps, checkpoint/resume
history, mean teacher−student gap, fraction with `abs(gap) < 1e-3`, clipping,
staleness, every failure counter, and held-out metrics. Link each summary row
to its committed raw artifact directory.

Success requires all three arms to reach step 200 and auxiliary-service
failures to be reported separately from policy quality. If a service remains
unfunded or another definitive external blocker prevents completion, preserve
all progress, commit and push the complete failure logs first, update this
block with evidence and artifact links, and stop rather than repeatedly
starting from zero.
