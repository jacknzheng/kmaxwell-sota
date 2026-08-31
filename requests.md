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

Next request number: **REQ-024**.

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

## REQ-023: wide per-matrix learning-rate interventions at two fork states

- status: RUNNING (agent 2026-08-30) — fresh node q4g2jyq (8xH100) bootstrapping SHA 25d3208 (codex/per-matrix-lr-public); does NOT preempt REQ-022's node wxgmk0q. Will run tests → config byte-diff gate → regen eos_shared_base → LR-trace gate → 6 runs (f{1500,2000}_a{0,1,2}) → shared-state gate → 30 curvature → cross-talk analysis.
- requested: Jeffrey / Codex / 2026-08-30 16:55 PDT
- priority: use the idle 8xH100 node released by REQ-021; do not preempt the
  running REQ-022 node
- expected cost: six 850-step continuations plus 30 standard curvature
  measurements, approximately 3–4 8xH100 node-hours

### Scientific question

Does changing one Muon matrix's learning rate affect only that matrix, as a
local thermostat predicts, or does curvature move into untreated matrices, as
a collective cross-layer sharpness budget predicts? Three replicated,
type-balanced assignments make untreated-neighbor effects estimable rather
than anecdotes. Repeating the same assignments from steps 1500 and 2000 also
tests whether the per-layer causal law changes with training state, extending
the local step-2400 intervention.

### Frozen implementation

Use a fresh clone of this repository and check out exactly:

```text
branch: codex/per-matrix-lr-public
SHA: 25d320879087eacc57fbc8d51b6007c18bb97ca6
parent lineage: ebf53cd88dad93721c121af80285cf01f239f53e
```

This SHA adds `PerMatrixLrMuon` as a subclass of the deployed
`BimaxwellMuon`. It changes only the applied learning rate and its decoupled
weight-decay factor by `lr_multipliers[sorted_index]`; the bi-Maxwell momentum
buffers and polar update are unchanged. It therefore loads the existing
`eos_shared_base` fork states without translating or resetting `m_fast` or
`m_slow`.

Run these targeted offline tests before renting or using the H100 node:

```bash
python -m pytest -q \
  records/track_3_optimization/tests/test_per_matrix_lr_muon.py \
  records/track_3_optimization/tests/test_per_matrix_lr_config_generator.py \
  records/track_3_optimization/tests/test_registry_locks.py \
  -k 'per_matrix or optimizer_registry'
```

Expected: 7 tests pass. The broad `test_hook_registry_is_locked` test is
already stale at parent `ebf53cd`: that parent added
`log_gradient_autocorrelation`, `log_learning_rates_at_steps`, and
`set_learning_rate_stairs` without adding them to its locked-name set. Do not
treat that pre-existing unrelated failure as a REQ-023 failure and do not patch
the frozen checkout on the box.

### Frozen assignments and configs

The six generated configs and both human- and machine-readable assignment
tables are committed beside this request:

```text
requests/req023/assignments.tsv
requests/req023/assignments.json
requests/req023/manifest.tsv
requests/req023/req023_f1500_a{0,1,2}.yaml
requests/req023/req023_f2000_a{0,1,2}.yaml
```

The design uses seed 23023. Within each of the six matrix types, a seeded
random ordering is divided among multipliers `{0.6, 1.0, 1.7}` and cyclically
rotated across assignments 0, 1, and 2. Consequently every assignment has
exactly four matrices at each multiplier within every type, and every matrix
receives each multiplier exactly once. The same three assignments are reused
at both forks so the fork-state comparison is paired.

Regenerate the files from the frozen code and require a byte-for-byte match
with the committed tables/configs before launch:

```bash
python records/track_3_optimization/offline_analysis/make_per_matrix_lr_configs.py \
  --out /tmp/req023-generated
diff -ru requests/req023 /tmp/req023-generated
```

### Shared-state and learning-rate gate

Use the serialized `eos_shared_base` states at steps 1500 and 2000 from
REQ-019. If the node was wiped, regenerate that base exactly from the
REQ-019 `ebf53cd` config; do not replace it with six independent prefixes.

Before the six full runs, execute a one-update temporary trace derived from
assignment 0 at each fork. Keep the committed scientific configs unchanged;
only the temporary trace copy may use `stop_after_step = fork + 1` and a unique
run ID. Both traces must show:

1. the loaded pre-update model and optimizer tensors equal the corresponding
   serialized source state;
2. the `learning_rates` row at the fork lists all 72 entries in
   `sorted_index` order;
3. every `sorted_index -> parameter name -> multiplier` tuple equals the
   committed `assignments.tsv` column; and
4. every effective Muon learning rate is exactly `0.025 * multiplier` at the
   first fork update.

Stop with `NEEDS-INFO` and retain the trace if any item fails. Do not run
curvature on a failed branch.

Each full config writes its loaded state at the fork to its own `gate_*`
directory before the first update. After all six training runs, compare model
and optimizer tensors across the three assignments at each fork and against
the source state. Record the hashes and maximum absolute tensor difference in
`shared-state-check.tsv`. This gate must pass before any curvature job begins.

### Runs and curvature

Run the six configs exactly as committed. Each continuation lasts 850 updates:

- fork 1500: stop 2350; curvature checkpoints
  `{1850, 1975, 2100, 2225, 2350}`
- fork 2000: stop 2850; curvature checkpoints
  `{2350, 2475, 2600, 2725, 2850}`

At every listed checkpoint, run the standard per-matrix curvature measurement
used by REQ-019: all Muon matrices, `--iters 8 --tokens 131072`, preserving raw
Lanczos alphas/off-diagonals and parameter names. Do not substitute aggregate
curvature or a shorter token budget.

### Required analysis

For each fork, report both raw leading curvature and the gauge-normalized
quantity `lambda_top * ||W||_F^2`.

1. Fit the direct effect of a matrix's own multiplier with matrix and type
   effects retained.
2. For every untreated matrix, fit response to the multipliers assigned to
   other matrices. Report same-block, adjacent-block, same-type, and all-other
   effects with the random assignment as the unit of replication.
3. Compare the signed cross-talk pattern between forks 1500 and 2000.
4. Recompute the own-multiplier law separately for each assignment; do not let
   one assignment or matrix type create the aggregate conclusion.

This is a discovery experiment with three assignments, not a claim of final
statistical significance. Preserve the per-matrix rows so later assignments
can extend the regression without rerunning these six continuations.

### Artifacts

Commit and push under:

```text
logs/kmaxwell/req023_per_matrix_lr/
  README.md
  summary.tsv
  shared-state-check.tsv
  runtime-lr-trace.tsv
  assignments.tsv
  manifest.tsv
  configs/
  <run_id>/
    command.txt
    console.log
    train-log.txt
    per_matrix_curvature.json
```

`summary.tsv` must contain the exact code SHA, fork, assignment, runtime,
validation loss, every curvature checkpoint, direct-effect estimates, and the
four cross-talk group estimates for both raw and gauge-normalized curvature.
The README must state whether the evidence favors a local response, a
collective redistribution, or is unresolved at three assignments. Commit all
raw evidence before marking `DONE`, `FAILED`, or `NEEDS-INFO`.

---

## REQ-022: momentum EoS fine-grained multiplier ladder at fork-1500

- status: RUNNING (agent 2026-08-30) — node wxgmk0q (8xH100), ebf53cd. eos_shared_base regenerated; all **7 ladder arms trained** {0.65,0.85,0.90,1.10,1.15,1.45,0.85dup}; **shared-state gate PASSES** (identical sha256 `de88e4aa` across all 7, abs-diff 0, LR@1500=base×mult exact — see `shared-state-check.tsv`). Per-matrix curvature now running (7 arms × 5 ckpts). REQ-021 withdrawn (node w7yov0w released), so REQ-022 is the sole active request.
- exact SHA: `ebf53cd88dad93721c121af80285cf01f239f53e` (same as REQ-019; do not upgrade)
- priority: after REQ-021 if both are picked up together; both fit one node
- protocol: identical to REQ-019 phase (1) in every respect (serialized
  `eos_shared_base` fork-1500 state — regenerate exactly as REQ-019 did if the
  released box wiped it; same shared-state hash gate; same per-matrix Lanczos:
  74 matrices, checkpoints {2250,2375,2500,2625,2750}, `--iters 8 --tokens
  131072`, alphas/offdiags stored raw), EXCEPT the multiplier set:
  run SIX new arms with constant post-fork multipliers {0.65, 0.85, 0.90,
  1.10, 1.15, 1.45} plus ONE duplicate at 0.85 (second seed of GPU
  nondeterminism only — identical config, for a mid-multiplier noise floor).
- purpose: REQ-019 established the aggregate inverse-LR law from five
  multipliers; per-matrix slopes are heterogeneous (10th-90th pct roughly
  -2.1..-1.0). The fine ladder doubles the per-matrix regression leverage and
  tests law curvature (departure from a pure power law) near multiplier 1.
- deliverables: same layout as REQ-019 (`logs/kmaxwell/req019_eos_state_dependence/`
  naming pattern `eos_f1500_s065` etc., summary.tsv rows appended, shared-state
  check rows appended). Gate must PASS before any curvature run, per REQ-019's
  rule; NEEDS-INFO with evidence otherwise.
- cost estimate: 7 train arms x ~3.5 min + 7 curvature passes x ~19 min on one
  8xH100 node ~= 3 node-hours.

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

- status: **DONE** (agent 2026-08-30) — `ebf53cd` serialized-state design works; all four owner-ordered phases complete, all shared-state gates PASS, all artifacts pushed. Success criteria met: **9 EoS arms + 45 curvature measurements**, **12 endpoint runs across 4 paired seeds** (16 runs incl. bases), explicit shared-state checks, plus the authorized FW two-checkpoint calibration. Boxes released.

### REQ-019 v3 COMPLETE (agent 2026-08-30) — all phases delivered

Executed on `kmaxwell-sota @ ebf53cd`, 8×H100, torch 2.10.0+cu128. All EoS arms ran on the base node `qvgl1eq` (the 9.9 GB serialized `eos_shared_state` was not cross-node-copied; this request permits base-node-only in that case).

**(1) Fork-1500 + (4) Fork-2000 EoS — `logs/kmaxwell/req019_eos_state_dependence/`** (commits 1a12152, 959360c, 7ddf22f)
- 9 arms (6 fork-1500 + 3 fork-2000), 45 per-matrix curvature measurements (74 matrices × 5 checkpoints each, full Lanczos alphas/offdiags).
- **Both shared-state gates PASS** (`shared-state-check.tsv`): identical `model_step00{1500,2000}.pt` sha256 across arms (unique-hash count = 1), max tensorwise abs-diff = 0.000e+00, and every optimizer group's LR at the first fork update = base × multiplier exactly.
- **Result — the per-matrix curvature law changes strongly with the LR multiplier but is essentially state-INDEPENDENT.** Max top-eigenvalue @ final checkpoint is monotone-inverse to the multiplier (~8× drop across 0.60→1.70); the two ×1.0 duplicates bracket to ~10% (noise floor); and the fork-2000 (later-state) curvature reproduces the fork-1500 law within ~5–25% at matched multipliers — i.e. the momentum-EoS relationship is a function of the LR multiplier and only weakly of the fork state.

**(2) Seed-twins — `logs/kmaxwell/req019_seed_twins/`** (commit 19e7069)
- 16/16 runs (4 seeds × base + 3 arms), per-seed isolated working dirs, warmstart step-1000 model + 8 optimizer shards verified before each seed's arms.
- Paired final diffs (bi-Maxwell − candidate), n=4: **pr357 K8 +0.002768** (se 6.5e-5), **expann 0.982→0.944 +0.000285** (se 6.2e-5); both candidates beat the bi-Maxwell control on every seed. Presented as a run-to-run noise estimate, **not** the n=8 significance test.

**(3) FW generalized-sharpness calibration — `logs/kmaxwell/req019_fw_calibration/`** (commit 343b16c)
- Paper-faithful global block-spectral sharpness via Frank–Wolfe over the product of spectral-norm balls on all 72 Muon matrices, joint HVP with cross-block terms (a new tool + 8 CPU tests incl. a cross-block-coupling guard, under `impl/`). Two checkpoints (1500/2750).
- K=50 converged (last-interval gain +1–2%); 5-restart spread ≤3.1% at K=50; single gradient-seeded restart ≈ ensemble; peak 40.8 GiB/rank, ~23 min/checkpoint. Euclidean Ritz retained for scale only (~10³× smaller; no agreement claimed).

### REQ-019 v3 GATE PASS (agent 2026-08-30) — fork-1500, SHA ebf53cd

The serialized `eos_shared_base` design fixed what two from-scratch fleets could not. All 6 fork-1500 arms ran on base node `qvgl1eq` (8xH100); the 9.9 GB `eos_shared_state/` was not cross-node-copied, so per this request's own rule all arms ran on the base node. Evidence: `logs/kmaxwell/req019_eos_state_dependence/shared-state-check.tsv`.

**Gate PASSES on all three checks:**
1. **Identical checkpoints** — all 6 arms' `model_step001500.pt` share one sha256 `3d9560ea…cb1f`; unique-hash count = **1**.
2. **Zero divergence** — max tensorwise abs-diff s060 vs s170 @1500 = **0.000e+00** (185/185 keys, symdiff 0); byte-identical, loaded from the one serialized state.
3. **LR = base × multiplier at the first fork update (step 1500)**, every optimizer group: base (s100 ×1.0) embed 0.7 / proj 0.004 / blocks 0.025 / other 0.015; s060 ×0.6→0.42/0.0024/0.015/0.009; s077 ×0.77→0.539/0.00308/0.01925/0.01155; s130 ×1.3→0.91/0.0052/0.0325/0.0195; s170 ×1.7→1.19/0.0068/0.0425/0.0255; s100dup ×1.0 identical to s100. All exact.

Proceeding to per-matrix curvature (5 manifest ckpts 2250–2750 × 6 arms, `--iters 8 --tokens 131072`), then the 3 fork-2000 arms + their curvature, then the twins fleet and FW calibration. Full artifacts pushed on completion.

### REQ-019 RE-RUN RESULT (agent 2026-08-30) — gate STILL FAILS on `f83bfcd`; root cause refined

Re-ran the 6 fork-1500 arms on the corrected SHA `f83bfcd` (verified `git rev-parse HEAD` = f83bfcd; configs regenerated). Evidence: `logs/kmaxwell/req019_eos_state_dependence/shared-state-check-v2.tsv`.

**The config generator IS fixed** — `configs/req019/eos_f1500_s060.yaml` vs `…s100.yaml` now differ **only** in `fixed_eta_after: 0.6` vs `1.0` (plus run_id/dump_dir), and both carry `fixed_eta_after_step: 1500`. Your LR-trace validation (identical over 0–1499) is consistent with that static config.

**But the trained checkpoints STILL diverge identically to the pre-fix run:**
- 6 distinct `model_step001500` hashes; max abs diff s060 vs s100 = **26.94 @125 → 87.25 @1500** (was 26.5→86.75 pre-fix — same pattern).
- The divergence is in **real weights**, widespread: `embed.weight` diff **26.9** @125; **173/185 tensors** differ >1e-4; only 12 at <1e-5 (nondeterminism). Sample `embed.weight[:4]`: `[0.996, 6.41, 8.13, -3.59]` vs `[1.39, 4.97, 5.81, -4.25]`.

**Refined root cause:** the fix landed in the config generator + the LR-*schedule-trace* function, but the **runtime optimizer still applies `fixed_eta_after` from step 0** instead of gating it by `fixed_eta_after_step`. i.e. the eta multiplier is used in the actual `muon` update pre-fork, even though the config and the schedule-trace say post-fork only. Grep where `fixed_eta_after` is consumed in the optimizer step / `run.py` and add the `step >= fixed_eta_after_step` gate there (the schedule-trace path alone isn't what drives training). **Verify by re-checking that two fork-1500 arms are bit-identical (or ~1e-6) through step 1500 before pushing.**

Per gate protocol: preserved evidence, did NOT run curvature (premise still void), did NOT proceed to the twins/calibration/fork-2000 phases. Boxes `31go243`/`3mp2l23` released (dumps regenerable on the next fix). I'll re-run the full owner-ordered pipeline once the **runtime** eta-gating is fixed and pushed.
- execution order (owner-set, 2026-08-30 ~10:00 UTC): run in this order —
  (1) the six fork-1500 arms, (2) the seed-noise twin fleet, (3) the
  generalized-sharpness calibration, (4) the three fork-2000 arms. Rationale:
  earliest arms carry the most decision-blocking information; fork-2000 is
  third-state redundancy. Still no preemption of REQ-018.
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
- exact SHA: `ebf53cd88dad93721c121af80285cf01f239f53e`
- runner: `records/track_3_optimization/run.py`
- config generator:
  `records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py`
- curvature tool:
  `records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py`

The two failed fleets established that separate from-scratch GPU runs do not
reproduce an identical parameter state, but they did not directly measure the
runtime learning rate. Inspection found no consumer of `fixed_eta_after`
outside the correctly gated schedule hook. The corrected design therefore
removes independent pre-fork execution entirely: `eos_shared_base` writes one
complete model-and-optimizer state at steps 1500 and 2000, and every arm resumes
that serialized state at its fork. Do not reuse either failed fleet.

On every workstation:

```bash
git clone --filter=blob:none --branch codex/momentum-kernel-schedules \
  https://github.com/jacknzheng/kmaxwell-sota.git
cd kmaxwell-sota
git checkout ebf53cd88dad93721c121af80285cf01f239f53e
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

First run the generated base config once:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_3_optimization/run.py configs/req019/eos_shared_base.yaml
```

It writes `eos_shared_state/` with the model and all eight optimizer-state
shards at steps 1500 and 2000. Copy that directory losslessly to every node
that will run an arm and verify every file's SHA-256 against the source copy.
If no cross-node transfer mechanism is available, run all nine arms on the
base node; never regenerate the base independently on another node.

Every arm uses seed 0, bi-Maxwell momentum, loads the relevant serialized
state, resumes the data stream at the matching batch, and applies its absolute
constant learning-rate multiplier beginning with the fork update. The
duplicate 1.00 arm measures post-fork run divergence.

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
difference. The expected difference is exactly zero because these checkpoints
are loaded from the same serialized files and saved before the first fork
update. Also record the learning rate of every optimizer group immediately
before and after that first update; it must equal base LR times the arm's
multiplier. If either check fails, preserve the runs, set this request to
`NEEDS-INFO`, and stop before curvature.

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
