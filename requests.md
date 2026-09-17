# Experiment requests

Active queue for the `jerry-agent` branch. Next request number: **REQ-068**.

Use this file for pending experiments, execution status, and links to results. Consolidated
findings belong in [FINDINGS.md](FINDINGS.md). Completed and blocked specifications are preserved
in the [September 16 archive](requests_archive_20260916.md).

## Active queue

| Request | Status | Work and dependencies |
|---|---|---|
| [REQ-062](#req-062-second-seed-of-six-momentum-kernel-runs-that-each-test-one-property) | OPEN | Existing independent muoff second-seed study; preserve its priority and limits. |
| [REQ-066](#req-066-adapt-step-size-with-gradient-direction-consistency) | OPEN | Test the multiplicative 1 + cos θ step-size rule with recent displacement references. |
| [REQ-067](#req-067-use-a-100-step-reference-in-prodigys-step-size-estimator) | OPEN | Compare Prodigy's fixed w₀ reference with exact rolling wₜ₋₁₀₀ in rₜ. Independent of REQ-066's outcome. |

**Pickup:** check live jobs and newly delivered artifacts before scheduling; do not interrupt
running work. REQ-066/067 use available capacity without displacing REQ-062. REQ-063's
control repair and REQ-064's prediction experiment have been delivered; do not rerun them
because of old scheduling text. **REQ-065 is BLOCKED: REQ-064 reported H1 FAIL.** The new
step-size experiments do not depend on that failed layer-wise momentum gate.

This cleanup records requests only; it does not report new training runs or a live job handle.

## Operating constraints

- **At most two nodes fleet-wide**, including other experiments.
- Preserve the existing **eight Lanczos iterations**; record convergence diagnostics.
- Commit code, configs, logs, figures, and derived measurements. **Never commit model weights,
  optimizer tensors, checkpoints, secrets, or environment dumps.**
- Record independent base-state hashes, data cursors, code SHA, actual LR traces, exact checkpoint
  steps, and the operator/loss normalization used by each probe.
- On pickup, change OPEN to RUNNING and record the live job or host/session/process handle,
  start time in UTC, progress/log location, and eventually the terminal exit state.
- Monitor every **20 minutes** as requested by Jack. A status label alone does not verify a live job.
- Reuse a live base between dependent experiments; benchmark training and probe costs separately.
- Use the interpretation and validation safeguards in archived REQ-051 and [FINDINGS.md](FINDINGS.md).
- Append new numbered requests; update existing status in place. Do not prepend iteration diaries
  or duplicate historical status tables.

## Deferred work

| Request | State | Disposition |
|---|---|---|
| REQ-065 | BLOCKED | REQ-064 reported H1 FAIL on September 16. Do not launch the conditional policy trial. [Preserved specification](requests_archive_20260916.md#req-065-test-an-unconstrained-layer-wise-policy-against-the-strongest-globals). |
| REQ-049 | OPTIONAL | Four-seed replication of the crossed per-matrix LR test; does not displace 050–052. Original specification remains in the history linked below. |
| REQ-042 | BLOCKED | 32×/64× batch runs exceed the available corpus. Requires a data or run-length decision; no looping/repetition is authorized by this cleanup. |
| REQ-035 B/C/D | NOT RUN | Arm A is complete. Preserve as deferred work; do not dispatch automatically. |

[Archived request specifications](https://github.com/jacknzheng/kmaxwell-sota/blob/28d00746aa80d71caf1fb8cb38b2e336b4c5d2d9/requests.md)
include the deferred designs. Completed results and retractions belong in [FINDINGS.md](FINDINGS.md)
and the linked experiment directories, rather than this queue.

## Delivered work and interpretation

DONE means artifacts were delivered, not that every requested control or mechanism was
established. The [September 11 audit](logs/kmaxwell/layerwise_momentum_design_20260911/README.md)
and [September 15 audit](logs/kmaxwell/layerwise_momentum_audit_20260915/README.md) take precedence
over stronger historical claims. Original specifications, result paragraphs, and audit notes
are retained in the archive; unresolved controls remain unresolved.

| Request | Status | Result and limits |
|---|---|---|
| [REQ-050](requests_archive_20260916.md#req-050-curvature-at-initialisation-and-early-training) | DONE | Establish when the depth-curvature profile appears. |
| [REQ-051](requests_archive_20260916.md#req-051-decompose-why-each-matrix-has-a-different-lr-to-curvature-response) | DONE; audited | Four-seed LR responses delivered; the combined decomposition uses mismatched probe batches. |
| [REQ-052](requests_archive_20260916.md#req-052-matched-uniform-versus-mixed-lr-controls-for-req-051) | DONE; audited | LR-scope comparison delivered; recorded bases differ from REQ-051, so exact pairing remains unverified. |
| [REQ-053](requests_archive_20260916.md#req-053-what-makes-mlpproj-different--expansion-ratio-vs-nonlinearity) | DONE | Separate the ReLU² input from the fan-in shape as the source of `mlp.proj`'s excess elasticity. |
| [REQ-054](requests_archive_20260916.md#req-054-annealed-single-ema-matched-to-k-maxwells-scheduled-memory-age) | DONE; audited | K-Maxwell beats the scheduled EMA at 1×; exact realized-age and larger-batch controls remain open. |
| [REQ-055](requests_archive_20260916.md#req-055-downhill-alignment-and-loss-curvature-of-the-actual-post-muon-update) | DONE; audited | One-seed geometry delivered; equivalence and per-step-mechanism claims remain unresolved. See REQ-057/058. |
| [REQ-056](requests_archive_20260916.md#req-056-test-k-maxwell-memory-in-standard-adam) | DONE; audited | K-Maxwell vs ordinary Adam is inconclusive (n=3); gain over the scheduled EMA does not isolate exact-age kernel shape. |
| [REQ-057](requests_archive_20260916.md#req-057-validate-layer-wise-spectral-sharpness-and-cross-layer-coupling) | DONE; audited | Repeatable pilot rankings; ~97% coupling is at a diagnostic direction, not the realized momentum step. Some numerical checks remain open. |
| [REQ-058](requests_archive_20260916.md#req-058-test-whether-layer-sharpness-predicts-the-response-to-momentum) | DELIVERED; gate not established | Raw-S_i correlation replicated; registered incremental prediction test absent, nomom control unverified, one base lacks the 64-update endpoint. |
| [REQ-059](requests_archive_20260916.md#req-059-verify-a-sharpness-guided-layer-wise-momentum-policy) | DONE; negative, audited | Balanced raw-S_i policy loses to global a=0.5; corrected statistics retain that result. Control/provenance repairs in REQ-063. |
| [REQ-060](requests_archive_20260916.md#req-060-identify-loss-cubic-feedback-separately-from-muon-normalization) | DELIVERED; mechanism unresolved | Simplified gradient-centered nonlinearities measured; actual-buffer causal attribution and removal replay not established. |
| [REQ-063](requests_archive_20260916.md#req-063-verify-restored-optimizer-controls-and-repair-the-returned-evidence) | DONE 2026-09-15 | Evidence repaired (corrected REQ-059 stats keep the negative; REQ-058 fork-1500 relabeled 60-update). nomom mu-overwrite reproduced + fixed in CPU and **on silicon**: returned "nomom" was mu=0.95 (==ordinary Muon), true mu=0 is distinct + best. Exact-age EMA / names / a=1 verified; replay reproducible. Box stopped. |
| [REQ-064](requests_archive_20260916.md#req-064-predict-local-memory-improvements-beyond-a-strong-global-setting) | DONE 2026-09-16; negative | 156/156 continuations delivered; reported H1 FAIL. Sharpness did not beat the type/depth prior; REQ-065 remains blocked. [Results](logs/kmaxwell/req064_local_memory_prediction/README.md). |

## REQ-062: second seed of six momentum-kernel runs that each test one property

- status: OPEN
- requested: Codex for Jeffrey Cheng / 2026-09-14 UTC
- priority: independent of REQ-057 to REQ-060; use available capacity without displacing them
- resource limit: **one node; 6 node-hours maximum**

Self-contained. This uses none of the K-Maxwell code, states, or vocabulary from REQ-054/057/058.
It is a separate codebase with its own harness, its own kernels, and its own data path.

Every claim in our momentum work rests on one run per configuration against a run-to-run spread
measured on different hardware. This request supplies a second seed so each claim has two runs and
a spread measured on yours.

### Source and setup

```bash
git clone https://github.com/jeffreycider/muoff
cd muoff && git checkout 3634020          # branch jcheng/momentum-kernel-schedules
bash boxlogs/jstudy/pod_bootstrap.sh      # MANDATORY, see below
python3 data/cached_fineweb10B.py         # 103 shards -> data/fineweb10B/
```

**The bootstrap script is mandatory and blocking.** torch 2.7.0 deadlocks inside the
ProcessGroupNCCL watchdog: one rank stops posting collectives and the rest wait until a timer kills
the job. That bug cost us about a third of our attempts this week. The script picks a torch wheel
matching your driver, builds a virtual environment, refuses to continue if it resolves to 2.7, and
runs a 30-step eight-rank smoke test with every NCCL workaround flag unset. **If the smoke test
fails, stop and report. Do not tune NCCL environment variables** — we spent a week doing that and
none of it touched the cause.

Data: `data/cached_fineweb10B.py` fetches the 103 shards the configs expect at
`data/fineweb10B/fineweb_{train,val}_*.bin`.

### What to run

Seven runs, all at **seed 1**. The seed is the top-level `seed:` key in each yaml; it is `0` in the
committed files and is the only field to change. Change nothing else.

**Run this one first — the other six fork from its step-1500 dump and cannot start until it has
written that dump.**

1. `efconfigs/lr_sweep/e2_prod_dump1500.yaml` — 3250 steps; writes the fork state `s4fork_1500_prod`
   at step 1500 and supplies the seed-1 reference loss.

Then these six, in any order among themselves:

2. `efconfigs/lr_sweep/decouple_L90_k15.yaml`
3. `efconfigs/lr_sweep/lowrec_L24_k15.yaml`
4. `efconfigs/lr_sweep/notch_p000.yaml`
5. `efconfigs/lr_sweep/notch_p003.yaml`
6. `efconfigs/lr_sweep/notch_p006.yaml`
7. `efconfigs/lr_sweep/overshoot_k6.yaml`

Runner:

```bash
bash boxlogs/jstudy/run_batch_v4.sh e2_prod_dump1500
bash boxlogs/jstudy/run_batch_v4.sh decouple_L90_k15 lowrec_L24_k15 notch_p000 \
                                    notch_p003 notch_p006 overshoot_k6
```

Each run is 3250 steps, about 20 minutes at 0.35 s/step on eight A100s.

One inconsistency to know about: `run_batch_v4.sh` still exports `NCCL_P2P_LEVEL=NVL` by default,
and its header comments still describe that as a fix. It is not one. It was our mistaken diagnosis
of the torch deadlock and the comments are stale. With the bootstrap venv the flag does nothing;
leave it or unset it, and tell us if unsetting it changes anything.

### Success criteria

- The smoke test passes, and no run needs an NCCL workaround flag beyond the stale default noted above.
- Run 1 completes 3250 steps. Its final validation loss within 0.003 of 3.2726 is expected; a larger
  difference is a useful result, so report it rather than retrying.
- Runs 2 to 7 each complete 3250 steps and log validation loss at every 250-step boundary.
- A run that stops early is reported as failed. Do not score a partial log.

### Required artifacts

```text
logs/muoff/req062_second_seed/
  README.md            hardware, torch version from the bootstrap, runtimes, attempts, deviations
  results.csv          config, seed, final_val_loss, status, duration_sec
  trajectories.tsv     config, step, val_loss at every 250-step boundary
  run logs             the full stdout of every run, including its `dacf step:` lines
  smoke.log            bootstrap smoke-test output
```

**Do not return the fork dumps.** They are about 178 GB each and we do not need them.

The number we want most is your own spread: if you can afford one repeat of run 1, its two values
give us a floor measured on your hardware, which is what every claim below is quoted against.

### What each run decides

Runs 2 and 3 move one property each of the kernel the schedule ends on — memory length, and the
weight on the newest gradient — with the others matched by construction, to find which one sets
final loss. Runs 4 to 6 vary the kernel's response at period two across 0, 0.03 and 0.06 with
everything else matched, to settle two archived experiments of ours that disagree by sixty times the
spread. Run 7 pushes the stability threshold below the range we have tested.

### Not included

No secrets. No new code beyond what is in the commit. No dependency on any K-Maxwell state or result.

## REQ-066: adapt step size with gradient-direction consistency

- status: **OPEN**
- requested: Jack / 2026-09-16 PDT
- priority and dependencies: new optimizer study; preserve REQ-062 and live jobs
- resource limit: **two nodes fleet-wide**; benchmark the pilot and record a finite GPU-hour
  budget before expansion
- artifacts: `logs/kmaxwell/req066_direction_consistency/`

**Question:** Can we take larger steps while recent gradients keep pointing downhill, and
smaller steps when the direction changes, using a Prodigy-inspired step-size controller?
Think of continuing confidently along a straight path, then slowing down at a turn.

Source: Bernstein and Newhouse, [Old Optimizer, New Norm: An Anthology](https://arxiv.org/pdf/2409.20325),
Story III, equations 24–30 and the discussion following equation 30. The paper interprets
Prodigy's step size using gradient/displacement alignment and mentions a rule akin to
ηₜ₊₁ = ηₜ × (1 + cos θ) in preliminary experiments. The precise comparisons below are our
experimental choices, not a claimed reproduction of an established result.

### Rule and comparisons

Here wₜ means weights before optimizer update t, gₜ means that update's raw loss gradient,
and ηₜ is the positive adaptive step-size scale. For a lookback of k completed optimizer
updates, define:

```text
aₜ = wₘₐₓ₍₀,ₜ₋ₖ₎                   reference weights
dₜ = aₜ − wₜ                       reverse of the recent weight displacement
cₜ = dot(gₜ, dₜ) / (‖gₜ‖₂ × ‖dₜ‖₂) = cos θₜ
ηₜ₊₁ = ηₜ × (1 + cₜ)
```

The dot product measures how much two directions agree; ‖·‖₂ is their ordinary length.
Use **reference minus current**, not current minus reference: cₜ > 0 should mean that
continuing along the recent descent path still reduces loss locally. Positive cₜ increases
the scale; zero leaves it unchanged; negative cₜ decreases it. This is gradient-versus-path
consistency, not the cosine between gₜ and gₜ₋₁. Those coincide only for particular update rules.

Start with one fixed, verified Muon recipe and its existing update direction, momentum,
relative matrix scaling, auxiliary optimizers, weight decay, and training schedule. Freeze
the exact recipe/code SHA before the pilot. Apply one global controller to the Muon matrix
group; form the cosine by summing dot products and squared norms across that same group
and all distributed shards. Preserve raw gradients before Muon's in-place transformations.
Do not average per-matrix cosines or adapt each layer separately in this first test.

| Arm | Step-size rule | Reference |
|---|---|---|
| baseline | Existing scheduled LR; adaptive multiplier fixed at 1 | None |
| cosine-1 | Multiplicative 1 + cₜ controller | wₜ₋₁ |
| cosine-100 | Same controller | wₘₐₓ₍₀,ₜ₋₁₀₀₎ |

Factor the actual LR as ηₜ × qₜ, where qₜ is the unchanged dimensionless warmup/cooldown
schedule. Initialize η₀ to the baseline's nominal LR, use ηₜ for update t, and use cₜ to
set ηₜ₊₁. Do not recursively multiply by qₜ or remove the schedule in just one arm. Keep
the weight-decay displacement on the baseline schedule so adapting the loss-gradient step
does not also change regularization strength. Document this separation in the resolved config.

For zero gradient or zero displacement, hold η unchanged. Clamp finite numerical cosine
values to [−1, 1]. Because exact cₜ = −1 would otherwise set η permanently to zero, and
persistent positive cₜ can grow η exponentially, predeclare a positive η floor and finite
ceiling shared by the cosine arms. Record both the raw candidate and applied η, plus bound
hit rates. Label the training implementation as bounded 1 + cos θ. Any damping, smoothing,
different initialization, or gradient-versus-gradient variant is a separately named follow-up;
do not silently alter the registered arms after seeing their losses.

### Pilot, evaluation, and deliverables

1. Verify the sign and update timing on tiny deterministic examples: aligned, perpendicular,
   opposite, and zero directions. Check that disabling the controller reproduces the baseline.
   Verify exact lookback, startup, and resumed-run parity, including distributed reductions.
2. Pilot all three arms from the same initialization and token order on one development seed
   through **500 completed updates**. Record validation loss at 0, 100, 250, and 500, stability,
   bound hit rates, memory, and runtime. Freeze η bounds and any implementation fixes before
   confirmation runs. A changed pilot is development evidence, not an independent replication.
3. If correctness and measured cost permit, run all three frozen arms from initialization
   through **3250 updates on three fresh paired seeds**, reserved before outcomes are opened.
   Use the same evaluation tokens and evaluate every 250 updates and at 3250. The primary
   comparison is each cosine arm minus baseline validation loss at 3250; comparing the two
   windows is secondary. Do not select a favorable earlier endpoint or drop failed seeds.

Publish every paired seed difference, their mean and uncertainty, train/validation curves,
time to common loss thresholds, and total GPU-hours. Three seeds are a small comparison;
report inconclusive results when uncertainty does not separate the methods. Log cₜ, ηₜ,
actual LR, update norm, displacement norm, gradient norm, and bound/zero-direction events.
Keep the seed/data/code manifests, configs, launch commands, and raw scalar logs. Report
whether any gain survives the added runtime and memory cost. A gain with frequent bound hits
is evidence for the bounded controller, not for the unrestricted recurrence.

## REQ-067: use a 100-step reference in Prodigy's step-size estimator

- status: **OPEN**
- requested: Jack / 2026-09-16 PDT
- priority and dependencies: independent of REQ-066's outcome; share correctness checks and
  measurement conventions where applicable
- resource limit: **two nodes fleet-wide**; benchmark memory/runtime and record a finite
  GPU-hour budget before expansion
- artifacts: `logs/kmaxwell/req067_prodigy_rolling_reference/`

**Question:** Does Prodigy's step-size estimate become more useful during training if it
compares the current weights with the weights exactly **100 optimizer updates ago**, instead
of always comparing with initialization? The recent reference asks whether the last stretch
of the path still points downhill, even after training has moved far from its starting point.

Use the same [paper](https://arxiv.org/pdf/2409.20325), Story III, especially equations 24–27.
The proposed change is the reference inside rₜ, the scalar running estimate used to choose
the step size. With b = √β₂, its simplified notation is:

```text
standard: rₜ = b × rₜ₋₁ + (1 − b) × ηₜ² × dot(gₜ, w₀ − wₜ)
rolling:  rₜ = b × rₜ₋₁ + (1 − b) × ηₜ² × dot(gₜ, wₘₐₓ₍₀,ₜ₋₁₀₀₎ − wₜ)
```

β₂ sets how quickly old information fades. **Replace only w₀ in this inner-product term.**
Keep rₜ's smoothing, the sₜ accumulator and its norm, the moment updates, η initialization,
schedule handling, stabilization, and all other Prodigy settings identical between arms.
Pin and document the implementation and its mapping to the paper; the displayed equations
omit implementation details and must not be substituted for a full production baseline.

| Arm | Optimizer | Reference in rₜ |
|---|---|---|
| prodigy-init | Verified standard Prodigy | w₀ |
| prodigy-100 | Identical Prodigy except the reference term | wₘₐₓ₍₀,ₜ₋₁₀₀₎ |

Use the same parameter grouping and reference coverage in both arms. Keep the same model,
data, initialization, token order, batch size, and schedule. REQ-066's Muon controller and
this Prodigy comparison test different update rules; report their results separately.

### Exact window and interpretation

- For t < 100 use w₀. Read wₜ₋₁₀₀ before update t; advance history after the update. Count
  completed optimizer updates, not microbatches or gradient-accumulation passes.
- Keep a rolling history that supplies the exact lagged weights. A snapshot refreshed every
  100 steps has a varying age and is a different experiment. Account for the substantial
  cost of retaining roughly 100 parameter snapshots; benchmark storage, precision, and any
  CPU-transfer overhead before training. If exact history does not fit the budget, report
  that limitation rather than silently using a stale or compressed approximation.
- Do not reset rₜ or sₜ every 100 steps, truncate their smoothing, change the norm, or divide
  the displacement by 100. Those would confound the reference replacement.
- Standard Prodigy's ηₜ₊₁ = max(ηₜ, rₜ / ‖sₜ‖₁) keeps its adaptive scale nondecreasing.
  Changing the reference alone therefore **cannot lower that scale**; it can slow or stop
  further growth. An external cooldown schedule can still lower the actual LR. REQ-066
  separately tests a controller whose scale can both increase and decrease.

### Pilot, evaluation, and deliverables

First check parity with the baseline when both references are w₀, including the startup
period; then verify lag selection at updates 99, 100, 101, and 200 on a known trajectory.
Check rₜ and η against a direct small-tensor calculation. Preserve the full rolling history
and optimizer state on resume and check uninterrupted-versus-resumed parity. Handle a zero
denominator exactly as the pinned baseline does; log signed rₜ and invalid-value events.

Pilot both arms from initialization on one development seed through **500 updates**, with
the REQ-066 evaluation points. Freeze the implementation and settings before running both
arms on **three fresh paired seeds through 3250 updates** if correctness and cost permit.
Evaluate every 250 updates and at 3250. Primary outcome: rolling-minus-initialization
reference validation loss at 3250. Report all paired differences and uncertainty, failures,
time to common loss thresholds, runtime, peak GPU/CPU memory, and history-transfer costs.
Do not tune the 100-step window using confirmation outcomes.

Log rₜ, ‖sₜ‖₁, the candidate and accepted η, actual LR, the signed gradient/reference inner
product, displacement norm, and cosine. Record how often the max rule rejects a candidate.
This distinguishes an ineffective reference signal from a useful signal prevented from
lowering the scale by the inherited max rule. Commit configs, code, manifests, scalar logs,
curves, and a result summary; keep parameter histories and checkpoints off Git.

## Template

```md
