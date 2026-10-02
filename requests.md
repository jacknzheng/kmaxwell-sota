# Experiment requests

Active queue for the `jerry-agent` branch. Next request number: **REQ-073**.

Use this file for pending experiments, execution status, and links to results. Consolidated
findings belong in [FINDINGS.md](FINDINGS.md). Completed and blocked specifications are preserved
in the [September 16 archive](requests_archive_20260916.md).

## Active queue

| Request | Status | Work and dependencies |
|---|---|---|
| [REQ-062](#req-062-second-seed-of-six-momentum-kernel-runs-that-each-test-one-property) | OPEN | Existing independent muoff second-seed study; preserve its priority and limits. |
| [REQ-068](#req-068-record-the-full-gradient-history-for-an-entire-nanogpt-run) | RUNNING | Capture run reported launched; Jack revised future capture to three selected full matrices. See scope amendment below and STATUS.md. |
| [REQ-069](#req-069-ten-data-seed-branches-to-estimate-reproducible-local-motion) | OPEN | Jack: ten branches from identical model/optimizer state; average selected-matrix gradients and updates. |
| [REQ-070](#req-070-test-candidate-directions-on-independent-loss-and-curvature-probes) | OPEN | Jack: test directions from REQ-069 on independent data before claiming a river direction. |
| [REQ-071](#req-071-fit-a-causal-gradient-history-estimator-and-test-it-in-training) | OPEN | Jack: fit an estimator against independent references; conditional on REQ-069/070 evidence. |
| [REQ-072](#req-072-six-optimizer-ablations-with-full-gradient-histories-and-spectrograms) | OPEN | Jack: AdamW / SGD / Muon, each with and without momentum; every-step histories of selected full matrices and six spectrograms. |

**Pickup:** check live jobs and newly delivered artifacts before scheduling; do not interrupt
running work. Preserve REQ-062's priority and limits. For Jack's new work, prioritize REQ-068's
complete capture and REQ-069's middle-training pilot, then REQ-070 and conditional REQ-071.
REQ-072 is an independent descriptive optimizer study; reuse the capture infrastructure without
waiting for REQ-069/070/071 results or interrupting live work. These requests do not depend on the
archived layer-wise momentum gate.

This queue update requests experiments; it does not report new runs or a live job handle.
Completed entries and Jack's previous OPEN REQ-066/067 have been removed from this queue at
Jack's request. Existing result files, archives, deferred work, and Git history are unchanged.

## Capture scope amendment — Jack, 2026-10-02 PDT

**This supersedes the all-parameter history requirements in REQ-068 through REQ-072.** Save
**a few selected full matrices at every optimizer step**, not the entire model. Training still
updates the complete model, and run lengths and six optimizer ablations remain unchanged.

- Default selection: the **attention output-projection weight matrix in the first, middle, and
  last transformer blocks** — three complete matrices. For the 12-block baseline these are
  zero-based blocks 0, 5, and 11 (expected `blocks.0.attn.proj.weight`,
  `blocks.5.attn.proj.weight`, `blocks.11.attn.proj.weight`; resolve exact harness names).
  Freeze a manifest of exact names, shapes, dtypes, and parameter counts before outcomes are
  inspected. Use the identical selection across all six arms, seeds, branches, and time steps.
- These must be complete matrices with all entries retained at every step for the full requested
  duration. Do not replace them with sampled entries, norms, or periodic snapshots. No embeddings,
  output head, bias/scalar tensors, or other weight matrices need persistent histories by default.
  Do not add a separate large sampled-coordinate recording requirement.
- REQ-068 requires raw gradients for the selected matrices. REQ-069 also retains their actual
  displacements. REQ-072 requires raw gradients, conditioned directions, and actual displacements
  for those same matrices in all six runs. Existing timing/precision/readback checks still apply.
- Forecast from the actual selected shapes and each stream's native dtype. Illustratively, three
  768-by-768 matrices have 1,769,472 entries: 3250 steps require **11.50 GB per history in 16-bit,
  or 23.00 GB in 32-bit** (decimal units). Six arms and three histories require about **207 GB
  or 414 GB**, respectively, before checkpoints/metadata. Mixed dtypes require an exact forecast.
- Report exact per-matrix measurements and an explicitly labeled **selected-matrix aggregate**.
  It is not a whole-model spectrum or proof of a universal river direction. Inexpensive whole-model
  scalar summaries computed during training are allowed, but must not require full tensor history.
- Preserve completed artifacts and do not cancel/restart a live training run merely for this edit.
  Check the actual job status on pickup. If supported safely, change capture scope only at a recorded
  clean chunk/resume boundary while retaining every step for the selected matrices. Otherwise extract
  the selected histories from the already-running full capture and use the narrower writer for all
  subsequent runs. Record provenance; extra historical tensors are not required future deliverables.
  Update the existing writer/config to select these matrices before copying or serializing tensors,
  and make smoke-test coverage assertions refer to this manifest rather than all model parameters.

## Operating constraints

- **At most two nodes fleet-wide**, including other experiments.
- Preserve the existing **eight Lanczos iterations**; record convergence diagnostics.
- Commit code, configs, logs, figures, and derived measurements. **Never commit model weights,
  optimizer tensors, checkpoints, secrets, or environment dumps.** Full gradient/update tensor
  histories must also stay off Git; commit their manifests, checksums, readers, and derived summaries.
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
| REQ-049 | OPTIONAL | Four-seed replication of the crossed per-matrix LR test; remains deferred; do not dispatch automatically. Original specification remains in the history linked below. |
| REQ-042 | BLOCKED | 32×/64× batch runs exceed the available corpus. Requires a data or run-length decision; no looping/repetition is authorized by this cleanup. |
| REQ-035 B/C/D | NOT RUN | Preserve these unrun arms as deferred work; do not dispatch automatically. |

[Archived request specifications](https://github.com/jacknzheng/kmaxwell-sota/blob/28d00746aa80d71caf1fb8cb38b2e336b4c5d2d9/requests.md)
include the deferred designs. Completed results and retractions belong in [FINDINGS.md](FINDINGS.md)
and the linked experiment directories, rather than this queue.

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

## REQ-068: record the full gradient history for an entire nanoGPT run

- status: **RUNNING (last reported: full run launched; check live handle on pickup)** — see
  `logs/river/req068_full_gradient_history/STATUS.md`. Jack's 2026-10-02 amendment replaces the
  full-model capture requirement with the three-matrix manifest above. Recompute storage from
  that selection; the previous ~1.1 TB requirement is historical, not a gate for future launches.
- requested: Jack / 2026-10-01 PDT
- priority: first deliverable for the river-direction study; preserve REQ-062 and live jobs
- artifacts: `logs/river/req068_full_gradient_history/`
- resource limit: existing two-node fleet-wide limit; measure storage/I/O and record a finite
  GPU-hour and storage budget before the full run

**Required:** complete one baseline nanoGPT training run and persist the **entire raw gradient
matrix for each of the three selected attention output projections at every optimizer step,
throughout the entire run**, following the shared capture scope amendment. Full-model histories
are no longer required. Norms, sampled coordinates, projections, spectra, short capture windows,
or every-k-step snapshots are not substitutes for the selected matrices' complete histories.

### Baseline and indexing

Pin one verified Muon + auxiliary-optimizer baseline, code SHA, architecture, dataset, token
order, batch size, loss normalization, and LR/momentum schedules before launching. Use the
simplified optimization trainer where suitable; record the exact selected recipe. Target
**3250 completed optimizer updates from initialization**, with ordinary validation through the
final endpoint. Do not switch to a smaller model or shorter run and call that full delivery.

Define theta_t as the weights before update t and g_t as that update's gradient. Capture g_t
**after all microbatch accumulation and distributed synchronization, after any loss-scale
unscaling, and before clipping, momentum, Muon transforms, weight decay, or optimizer mutation**.
Copy before optimizer code mutates `.grad`. In a sharded implementation save all shards plus an
exact reconstruction map; avoid storing duplicate replicas. Record missing/inactive gradients
explicitly rather than silently fabricating zeros. One observation means one optimizer update,
not one microbatch; distinguish attempted/skipped updates if applicable.

Save the actual gradient values at their native precision, losslessly; converting to float32
is allowed but must not be described as recovering precision absent in the source. Record the
exact conversion to mean-per-token gradients for comparable analysis without changing the
baseline training inputs. Record parameter names, shapes, dtypes, flatten order/offsets, tying,
sharding, tokens per update, global update index, and code/config/seed provenance.

### Storage, replay, and acceptance

- Stream bounded chunks to durable external storage or a retained durable volume. Do not keep
  the full run in GPU memory or leave the only copy on a disposable training node. Publish the
  artifact location and retrieval instructions; keep credentials out of manifests and logs.
- Forecast bytes before launch from the selected matrices' actual counts/dtypes/step count, including
  updates/checkpoints. Benchmark sustained writes and free capacity. If storage is unavailable,
  report the concrete blocker; do not silently downsample, quantize, drop a selected matrix, or shorten.
- Keep raw tensors outside Git. Commit a machine-readable manifest with chunk hashes, step
  coverage, parameter schema, shapes, dtypes, and artifact sizes, plus a reader that reconstructs
  every selected matrix's gradient for any requested step and streams per-matrix time series.
- Record every-step actual displacements of the selected matrices separately where affordable, including the
  effects of the optimizer and weight decay; clearly label them as updates rather than gradients.
  Their storage can be scoped independently, but every-step raw capture of the selected matrices is mandatory.
- Retain checkpoint/optimizer/RNG/data-cursor states off Git at predeclared analysis forks
  (initially steps 500, 1500, and 2500), plus any filter state needed for later continuations.
  Record hashes and retrieval paths so independent probes can be performed at the correct state.
- Smoke-test round-trip tensor equality, accumulation/reduction timing, no gradient aliasing,
  uninterrupted-versus-resumed indexing, and logging-on versus logging-off training parity.
  Verify all 3250 expected update indices and all selected matrices/shards after the full run;
  no gaps, duplicates, or silently partial chunks. Report failures as incomplete capture.
- Deliver capture coverage, checksums, validation trajectory, peak memory, I/O volume, training
  time excluding/including capture overhead, and a worked example loading one complete step.

Initial descriptive analysis: gradient norms, signed direction similarity and lag correlations,
period-two evidence, and per-matrix/selected-matrix-aggregate comparisons. A smooth average is a candidate signal,
not proof of a valley floor. Curvature probes can use retained states under REQ-070; storing a
full Hessian or measuring it every step is not required for this capture request.

## REQ-069: ten data-seed branches to estimate reproducible local motion

- status: **OPEN**
- requested: Jack / 2026-10-01 PDT
- dependencies: verified baseline and fork states from REQ-068, or an exactly matched retained base
- artifacts: `logs/river/req069_ten_branch_ensemble/`
- priority: start with the step-1500 pilot; then repeat at steps 500 and 2500 within the recorded budget

**Question:** do ten nearby training trajectories reveal a reproducible average gradient/update
direction that changes slowly, with less period-two oscillation than individual trajectories?

At each fork clone **identical model weights and full optimizer state** into ten branches.
Use ten independent data-sampling seeds, with identical batch size, architecture, LR/momentum
schedule, and starting schedule position. Document other RNG streams and dropout handling.
Do not independently reinitialize the networks: their coordinate-wise gradients need not be
aligned. Identical seeds and token sequences would merely replay the same trajectory.

Run **100 completed updates per branch**. Preserve the entire gradient matrices and actual optimizer
displacements for the shared three-matrix selection at every branch step, using REQ-068's conventions.
Hash initial states and record the sampled token indices/cursors. Run branches sequentially if needed to
respect fleet limits. Probe runs must not consume or modify another branch's training RNG/cursor.

### Registered analysis

- Average raw mean-per-token gradients across branches at each relative update, and separately
  average actual parameter displacements. Analyze per selected matrix and their labeled aggregate. Do not
  confuse averaging raw vectors with averaging unit directions; report norms and dispersion.
- Split branches into two predeclared independent groups of five and compare their averages
  at matched steps. Repeat summary estimates with 1, 2, 5, and 10 branches to assess convergence;
  report uncertainty across branches, not by pretending correlated time points are new seeds.
- Measure directional agreement at lags 1, 2, 4, 8, 16, and 32; mark near-zero vectors invalid.
  Compare with individual branches and ordinary temporal averaging. Avoid overlapping-window
  agreement as the only evidence, since shared inputs mechanically induce agreement.
- Measure period-two residuals after a predeclared local trend removal; preserve gradient signs.
  Record selected-matrix separation, loss spread, and update norms to identify when branches have
  moved too far apart for a local interpretation. Do not align/reorder steps to maximize agreement.
- All branches inherit a common initial oscillation phase, so averaging may retain deterministic
  bouncing. A negative result for cancellation is informative. Different branch locations also
  mean the ensemble gradient is not generally the gradient at the ensemble-average weights.

Deliver raw-artifact manifests, per-step/per-layer measurements, cosine-versus-lag plots,
two-group agreement, amplitude/alternation diagnostics, and all branches including failures.
Call a reproducible average a **candidate local motion**, not established ground-truth river
direction. No paired weight-perturbation or valley-settling experiment is requested at this stage.

## REQ-070: test candidate directions on independent loss and curvature probes

- status: **OPEN**
- requested: Jack / 2026-10-01 PDT
- dependencies: REQ-069 candidate directions and exact retained evaluation states
- artifacts: `logs/river/req070_independent_direction_tests/`

**Question:** is a reproducible ensemble direction useful for descent, beyond merely being smooth?

Before inspecting results, select probe steps (initially branch steps 20, 50, and 99), independent
probe data, and displacement lengths. Retain exact weights and optimizer states at those steps.
Compare raw gradient, two-step average, an exponential moving average, and the ensemble direction.
For each evaluated branch, use a **leave-one-branch-out ensemble** so its sampling noise is not
part of its own reference. Evaluate the candidate at that branch's own parameter state; report the
spread across branches rather than assuming one average direction is valid at every location.
Use the shared selected-matrix coordinates for all candidate comparisons. Temporarily move only
those matrices, keeping other parameters fixed, and evaluate the full model's probe loss. Baseline
and ensemble directions must use exactly the same parameter support and norm convention. Report
this as a selected-matrix intervention, not a recovered full-model direction. Separate full model/
optimizer fork checkpoints remain permitted for these probes; no full-model gradient history is needed.

For a proposed movement vector d, evaluate fresh probe loss at theta + alpha*d/||d||, restoring
the exact original state after each test. For gradient candidates d is the negative gradient.
Use equal displacement lengths across candidates: initially 0.25, 0.5, 1, and 2 times that state's
baseline update norm, plus zero. Define a shared fallback or mark the test invalid for zero norms.
Publish full loss-versus-distance curves, paired loss changes, norms, and failures. Do not pick
the best displacement on final validation data and report that selection as an unbiased result.

Run distinct comparisons for raw-gradient geometry and actual optimizer movement. For the latter,
pass each candidate through an identically copied baseline optimizer state; preserve parameter-group
scaling, schedule, and weight decay, and avoid mutating the live trajectory. Muon/Adam transforms
can change the result substantially; a useful raw gradient is not automatically a useful update.

At selected states, optionally measure directional loss curvature for the candidate and removed
alternating component via Hessian-vector products on a fixed independent probe objective. Preserve
the existing eight Lanczos iterations if spectral probes are used, and report convergence. Verify
derivatives and loss normalization, using an appropriate diagnostic attention backend if necessary.
These are diagnostics, not a requirement for full Hessian storage or every-step curvature capture.

Use separate data for any probe-setting selection and final confirmation. Report useful-descent
and reproducibility evidence separately. Lower curvature or greater consistency alone does not
establish a river direction. Do not add the deferred weight-displacement/settling experiment.

## REQ-071: fit a causal gradient-history estimator and test it in training

- status: **OPEN**
- requested: Jack / 2026-10-01 PDT
- dependencies: proceed to optimizer trials only if REQ-069/070 establish a reproducible, useful
  reference; otherwise report why the reference is inadequate before expanding
- artifacts: `logs/river/req071_history_estimator/`

**Question:** can one run's recent gradients estimate the useful reference without the cost of
an ensemble? Fit an estimator to independent targets rather than trying an unrestricted filter zoo.

Start with a shared linear temporal estimator, with window W in {4, 8, 20, 40}:

$$
\widehat{s}_t = \sum_{j=0}^{W-1} c_j g_{t-j}.
$$

Fit coefficients against leave-one-branch-out reference gradients on development trajectories.
Fit and score only the shared selected matrices, with the same loss normalization and parameter
coverage for inputs and targets. Do not infer an unmeasured full-model target. As an interpretable
candidate enforce preservation of constants and current-step linear trends, and cancellation of
constant-amplitude period-two oscillation:

$$
\sum_j c_j = 1, \qquad \sum_j j c_j = 0, \qquad \sum_j (-1)^j c_j = 0.
$$

Penalize large coefficients to control noise amplification; select regularization and W only on
development data. Predeclare any weighting across layers. Compare with unfiltered gradients,
pair averaging, and a tuned exponential moving average with comparable tuning budget. Estimate the
current signal using only present/past gradients; no centered windows or future leakage.

Split development and evaluation by whole trajectory/base lineage. Do not randomly split overlapping
windows or let an evaluated branch enter its own target. Report that branches sharing initialization
are conditional replicates; use fresh initialization lineages for independent confirmation. Score
magnitude error, direction error, residual alternation, and response delay when direction changes.
Include a low/no-alternation control to test whether the method erases useful signal. Failed prediction
is evidence against this estimator/history window, not proof that no possible estimator can work.

If offline and independent-loss evidence are favorable, branch paired **200-update** continuations
from identical model/optimizer states: original optimizer, simple averaging baseline, and fitted
estimator. Predeclare insertion point (initially raw synchronized gradients before momentum/Muon),
parameter coverage, startup/history handling, and immutable coefficient settings. Initially apply
averaging/the fitted estimator only to the selected matrices; leave all other parameters on the
unchanged baseline optimizer. Initialize from shared selected-matrix history and thereafter use
each branch's own gradients. Any whole-model extension is a separately justified follow-up.
Keep data orders paired and verify the restored optimizer actually has the intended settings.

If those continuations justify expansion, compare the frozen methods through **3250 updates on
three fresh paired initialization/data seeds**, with comparable development LR tuning budgets.
Evaluate at least every 250 steps and at the final endpoint on untouched validation data. Report all
seed differences, uncertainty, failures, validation loss at equal tokens, time-to-loss, runtime,
GPU-hours, and history memory/storage. These three seeds give limited precision; an inconclusive
result is valid. Gradients becoming smoother alone is not a successful optimizer result.

## REQ-072: six optimizer ablations with full gradient histories and spectrograms

- status: **OPEN**
- requested: Jack / 2026-10-02 PDT
- dependencies: reuse REQ-068 capture/reader infrastructure after verification; independent of
  the river-reference and learned-filter results; do not interrupt live jobs
- artifacts: `logs/river/req072_optimizer_spectrograms/`
- resource limit: existing **two nodes fleet-wide**; benchmark and record finite GPU-hour,
  durable-storage, and I/O budgets for all six runs before launching the full comparison

**Question:** what temporal frequencies are present in the gradients after each optimizer's
conditioning, and how does turning momentum off change them? Deliver **six ablations, a
spectrogram for each, and histories of the selected full matrices at every optimizer step for every arm**.
This is descriptive analysis of the optimizers, not a search for the best temporal filter.

### Six arms and matched training

| Arm | Optimizer | First-moment momentum |
|---|---|---|
| adamw-mom | AdamW | beta1 = 0.9 |
| adamw-nomom | AdamW | beta1 = 0 |
| sgd-mom | SGD | momentum = 0.9, dampening = 0, Nesterov off |
| sgd-nomom | SGD | momentum = 0, dampening = 0, Nesterov off |
| muon-mom | Standard Muon | mu = 0.95, standard Nesterov direction |
| muon-nomom | Same Muon conditioning | mu = 0, Nesterov off; condition the current gradient |

For AdamW keep **beta2 = 0.95**, epsilon, bias correction, and other settings identical within
its pair. "No momentum" means no first-moment averaging, **not** removal of the running squared-
gradient scale. Keep that adaptive conditioning. For Muon retain the identical orthogonalization,
iteration count, shape scaling, and numerical settings across its pair; do not substitute a
K-Maxwell or other multi-timescale memory kernel. Specify SGD's sum-style momentum recurrence
and resulting scale so amplitude differences are not mistaken for spectral effects.

Use AdamW and SGD for their full model parameter sets. Muon uses the baseline's supported hidden
matrices and auxiliary AdamW for the remaining parameters. For muon-mom use auxiliary beta1 = 0.9;
for muon-nomom use auxiliary beta1 = 0, with beta2 = 0.95 and other auxiliary settings held fixed.
Thus the no-momentum arms have no first-moment momentum anywhere. Document the full parameter-group
map and label Muon auxiliary panels as AdamW; the primary optimizer comparison is on matching
selected hidden matrices. Optional auxiliary/whole-model scalar summaries are separate and do
not add tensor-history requirements; do not label selected-matrix spectra as whole-model spectra.

Run all six from **identical initial model weights and identical training token order**, with
fresh optimizer state, fixed architecture/dataset/batch size, and identical evaluation tokens.
Target **3250 completed updates per arm**, initially one paired initialization/data seed (six
full runs). This is an exploratory paired comparison, not a statistically replicated performance
claim. Each optimizer generates its own trajectory; do not present differences as a pure filter
transfer function evaluated on identical gradients.

Use a short separate development pilot to select stable optimizer-specific LR scales, with a
matched tuning budget. Within each momentum on/off pair use the same LR scales and schedule;
freeze these before confirmation. Different optimizers need not share the same numeric LR.
Keep clipping and regularization rules fixed within pairs, record them, and separate weight
decay from the measured conditioned direction. Use explicit decoupled weight decay for this
study; set SGD's built-in coupled weight_decay to zero and apply the declared decay separately,
so decay does not enter its momentum buffer. Label this implementation choice. Record actual
LR/decay/momentum traces and clipping factors.
Any arm that diverges or stops is reported as failed/incomplete with all observations retained;
do not silently change its settings halfway through or report its partial run as full delivery.

### Mandatory every-step histories for selected full matrices: raw AND post-conditioning

For **each matrix in the shared three-matrix manifest, at every step in all six complete runs**,
store losslessly:

1. **Raw loss gradient g_t:** after accumulation, synchronization, and loss-scale unscaling,
   before clipping/regularization/momentum/optimizer mutations. Save the actual native values
   and the exact normalization needed to obtain mean-per-token gradients.
2. **Conditioned direction u_t:** the exact data-gradient-driven tensor after clipping (if used),
   momentum/bias correction, adaptive scaling or Muon orthogonalization, and fixed shape scaling,
   but **before learning-rate multiplication and decoupled weight decay**. AdamW uses its
   bias-corrected first moment divided by sqrt(second moment) + epsilon; SGD uses its momentum
   direction or current gradient; Muon uses its final scaled orthogonalized direction. Record
   the optimizer responsible for each selected matrix. Auxiliary tensor histories are not required.
3. **Actual parameter displacement:** post-step weights minus pre-step weights, including LR,
   decay, and finite-precision effects, stored separately so direction changes can be distinguished
   from schedule effects. Record the update decomposition and any additional transforms.

Retain each selected matrix's **entire matrix** across the whole run. The fixed subset of parameter
matrices is intentional; do not subsample entries within those matrices. No skipped steps, norm-only
logs, sampled coordinates, projections, or saved spectrograms may substitute for their histories.
Copy tensors before in-place updates overwrite them; do not reconstruct u_t merely by dividing a rounded weight difference by the LR.
Use bounded chunks, an exact parameter/shard schema, native-precision lossless storage, checksums,
resume-safe step indices, and a reader for any arm/step/selected matrix. Durable tensor artifacts stay
**outside Git**; commit retrieval instructions, manifests, schemas, code, and derived results.
Forecast selected-matrix storage for all three streams and all six arms, not just one raw-gradient run. Report
storage blockers rather than quietly reducing coverage or precision. REQ-068's complete selected-matrix raw-
gradient run remains required; reuse a run here only if its full config and capture requirements
match exactly, with an explicit artifact cross-reference.

### Fourier analysis and spectrogram construction

Analyze **variation over optimizer steps**, not spatial Fourier transforms across matrix rows or
columns. A matrix supplies many signed scalar time series, one per entry. Compute the temporal
short-time Fourier transform (STFT) for each coordinate, then sum squared Fourier magnitudes
across coordinates. Do not first take gradient norms, absolute values, or a signed coordinate
average: those can hide sign-flipping oscillation or cancel unrelated coordinates.

For a parameter tensor flattened to coordinates i, use the same window and normalization for
all arms. The aggregate power is, up to the explicitly recorded PSD normalization:

$$
P_m(\tau,f)=\sum_{i\in m}\left|\sum_{j=0}^{W-1}h_j\,u_{\tau+j,i}\,
 e^{-2\pi\mathrm{i}fj}\right|^2.
$$

Here m names a parameter tensor, tau is a window's start step, h is the window, and f is cycles
per optimizer step. Normalize by window energy and sampling frequency consistently; report total
power and mean-per-coordinate power. Compute in coordinate blocks to avoid loading the entire
history into RAM. Sum the three selected matrices' powers for a labeled selected-matrix aggregate;
this is not a global/whole-model spectrum. Keep the selection fixed across all arms.

- Primary STFT: **128-step Hann window, hop 16, sampling frequency 1 sample/update**, no temporal
  subsampling, real one-sided frequencies from 0 to **0.5 cycles/update**. Show period two explicitly
  at **0.5**, the Nyquist endpoint. Correctly scale DC/Nyquist versus interior one-sided bins.
- Show raw/non-detrended spectra (including DC) and a separately labeled per-window linear-detrended
  version, so the changing trend does not get silently deleted. Detrending is for interpretation,
  not evidence that everything removed was noise. Include 32- and 256-step windows as sensitivity
  checks for short changes versus frequency resolution, with the same rule across all arms.
- Use only complete windows for primary plots; mark any padded boundary windows. Gaps or skipped
  updates break contiguity: split the series and flag incomplete runs rather than interpolating.
- Make the primary panels from u_t; also show matched raw-gradient and actual-displacement panels
  to reveal what conditioning changes versus what the LR/decay schedule changes.
- Produce whole-run Fourier power spectra and early/middle/late summaries alongside spectrograms;
  mark LR phase boundaries and distinguish nonstationary whole-run spectra from local behavior.
- Plot both absolute log power and per-window normalized spectral-power fractions, with shared
  color limits within each comparable six-arm figure. Report the numerical floor and flag zero-
  energy windows. Different optimizer scales should not masquerade as different frequency content.
- Report power near period two (predeclare band 0.45–0.5 cycles/update), low-frequency power
  (0–0.05), total energy, signed lag-one/lag-two agreement, and losses alongside the plots.
  Display each selected matrix separately as well as their aggregate, so the larger-energy matrix
  cannot hide the others. Do not generalize three attention projections to every matrix family.

Use a documented STFT implementation such as [SciPy ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html).
For AdamW's distinction between first- and second-moment state, follow the pinned implementation
and [AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).

### Validation and deliverables

- Verify optimizer capture against direct small-tensor calculations and confirm instrumentation
  does not change the updates. Assert true zero-momentum behavior at startup and after restoring
  state: no saved momentum hyperparameter may overwrite the ablation. Check every parameter group.
- Validate spectra with a constant signal, a known sinusoid, exact alternating signs, and white
  noise. Include a sign-flipping vector of constant norm to catch accidental norm-before-FFT bugs.
  Check one-sided energy scaling, Nyquist visibility, chunk boundaries, tensor round trips, and
  complete step/selected-matrix/shard coverage for all six arms.
- Deliver **six individually labeled conditioned-direction spectrograms**, a shared-scale 3-by-2
  optimizer-by-momentum comparison, raw/conditioned/displacement comparisons, per-layer plots,
  whole-run Fourier spectra, band-power tables, and machine-readable spectral arrays.
- Deliver all six runs' complete selected-matrix histories with manifests/readers, resolved configs, code SHA, seeds/data
  manifests, launch commands, stdout, validation curves, runtime/GPU-hours, peak memory, storage
  volume, and capture overhead. Figures alone do not complete this request.
- Explain whether momentum suppresses, shifts, or amplifies alternating/high-frequency structure
  and whether conditioning changes that pattern. With one paired seed, keep conclusions descriptive;
  a smooth spectrogram alone does not establish a river direction or better optimization.

## Template

```md
## REQ-NNN: short experiment title

- status: **OPEN**
- requested: Name / YYYY-MM-DD timezone
- dependencies:
- artifacts:

Question, registered comparisons, resource limits, validation, and deliverables.
```
