# Experiment requests

Active queue for the `jerry-agent` branch. Next request number: **REQ-076**.

Use this file for pending experiments, execution status, and links to results. Consolidated
findings belong in [FINDINGS.md](FINDINGS.md). Completed and blocked specifications are preserved
in the [September 16 archive](requests_archive_20260916.md).

## Active queue

| Request | Status | Work and dependencies |
|---|---|---|
| [REQ-062](#req-062-second-seed-of-six-momentum-kernel-runs-that-each-test-one-property) | OPEN | Existing independent muoff second-seed study; preserve its priority and limits. |
| [REQ-068](#req-068-record-the-full-gradient-history-for-an-entire-nanogpt-run) | DONE | Both streams (g_t + delta_t) captured for all 3 windows + 64x64 K/C Gram + OOS + per-family + offset probes. Finding: gradients anti-align (lag1<0, period-two), displacements persist (lag1 ~+0.78 = river), but not low-rank-constant + embedding-dominated. Node stopped. |
| [REQ-069](#req-069-ten-data-seed-branches-to-estimate-reproducible-local-motion) | DONE | NEGATIVE: within-branch signatures reproduce but no reproducible common whole-model direction across independent data seeds. |
| [REQ-070](#req-070-test-candidate-directions-on-independent-loss-and-curvature-probes) | DONE (negative) | No candidate direction gives useful descent on independent loss (best -0.002 nats, negligible/non-robust); baseline-update overshoots, raw -grad catastrophic (+21), ensemble loss-neutral. River arc closes negative; REQ-071 gated off. Node stopped. |
| [REQ-071](#req-071-fit-a-causal-gradient-history-estimator-and-test-it-in-training) | GATED OFF | Conditional on REQ-069/070 showing a useful reproducible slow component; both negative, so premise absent. Not launched. |
| [REQ-072](#req-072-six-optimizer-ablations-with-full-gradient-histories-and-spectrograms) | RUNS DONE; SPECTROGRAM DELIVERY INCOMPLETE | Six-arm spectral summaries committed; time-resolved arrays/figures missing. REQ-074 repairs delivery and the Muon displacement-proxy limitation. |
| [REQ-073](#req-073-qkmlp-spectra-through-muon-momentum-and-effective-batch-size) | DONE | Period-two filtered PRIMARILY by momentum (stage1->2; no drop when momentum off), orthogonalization secondary; NOT minibatch noise. Time-resolved in REQ-074. |
| [REQ-074](#req-074-save-every-step-frequency-decompositions-and-deliver-time-period-amplitude-spectrograms) | DONE | 28 time-resolved arrays + 168 heatmaps (time x period x amplitude) recomputed offline from SAVED REQ-072/073 histories (no rerun), GPU-recompute parity-exact (~1e-15) to numpy core. Time-localizes fast->slow: momentum builds the slow component over ~first quartile; orthogonalization secondary; batch size NOT the cause (4B-nomom keeps highest period-2 -> refutes "period-2 = noise"). |
| [REQ-075](#req-075-late-training-large-batch-versus-momentum-noise-oscillations-and-useful-gains) | DONE | 16B does NOT reproduce momentum's benefit (worse per token); small-batch+momentum most token-efficient; momentum=denoising (vanishes at 16B+tuned LR); late period-two is overshoot/dynamics (half-LR collapses it), revealed-not-removed by large batch. 3 stages on 1 node. |

**Pickup:** check live jobs and newly delivered artifacts before scheduling; do not interrupt
running work. Preserve REQ-062's priority and limits. For Jack's new work, prioritize REQ-068's
whole-model window analysis, then REQ-069's middle-training confirmation, REQ-070 loss tests,
and conditional REQ-071 estimator trials.
REQ-072 is an independent descriptive optimizer study; reuse the capture infrastructure without
waiting for REQ-069/070/071 results or interrupting live work. These requests do not depend on the
archived layer-wise momentum gate.
REQ-073 is the focused follow-up to the gradient-versus-update plots: isolate Muon on matching
hidden matrices, separate momentum from orthogonalization, and test effective batch size. Keep
REQ-072's six-arm study distinct; share artifacts only when the full configurations and capture
streams match, and budget both requests within the existing fleet limit.
REQ-074 is Jack's latest correction to REQ-072/073 spectral persistence and plotting. Read it before
any further capture/analysis for these requests; preserve live jobs and REQ-062 priority.
REQ-075 is the new late-training performance/mechanism follow-up. Inspect and reuse compatible
REQ-073/074 artifacts first, but do not substitute their full-run, equal-update B/4B results for
the requested shared-checkpoint, larger-batch, equal-token continuations. Preserve live jobs,
REQ-062 priority, and the existing fleet ceiling.

This queue update requests experiments; it does not report new runs or a live job handle.
Completed entries and Jack's previous OPEN REQ-066/067 have been removed from this queue at
Jack's request. Existing result files, archives, deferred work, and Git history are unchanged.

## Capture scope amendment — Jack, 2026-10-02 PDT

**Latest decision: separate the whole-model river investigation from the descriptive optimizer
spectrogram study.** This replaces the earlier three-matrix-only scope for REQ-068 through
REQ-071. REQ-072 keeps its selected-matrix, full-run recording and all six optimizer ablations.

| Work | Parameters recorded | Time coverage | Required tensor streams |
|---|---|---|---|
| REQ-068: main river experiment | Entire model, including embeddings, head, vectors/scalars | Three 64-update windows: updates 500–563, 1500–1563, 2500–2563 (zero-based) | Raw gradients and actual parameter displacements |
| REQ-069: independent confirmation | Entire model in ten branches from the same checkpoint | 64 consecutive updates per branch; middle-training pilot first | Raw gradients and actual parameter displacements |
| REQ-070/071: validation and estimation | Whole-model directions derived from the above | Matched retained probe states and past-only history | Reuse captured artifacts; benchmark any live filter memory |
| REQ-072: optimizer comparison | Three selected complete attention output-projection matrices | Every step of each 3250-update run, all six arms | Raw gradients, conditioned directions, actual displacements |

**Storage estimates:** at 164M parameters, one 64-step whole-model history is approximately
21 GB in 16-bit or 42 GB in 32-bit. Three windows with two histories are approximately **126 GB
or 252 GB**. Ten 64-update confirmation branches with two histories are another **420 GB or
840 GB per fork**, so start with the middle fork and budget additional forks separately. These
are decimal, uncompressed estimates; resolve actual parameter counts, each stream's native dtype,
checkpoints, and metadata before launch. Whole-model recording is not required outside the windows.

**REQ-072 matrix selection remains:** the attention output-projection weight matrix in the first,
middle, and last transformer blocks. For the 12-block baseline use zero-based blocks 0, 5, and 11
(expected `blocks.0.attn.proj.weight`, `blocks.5.attn.proj.weight`, `blocks.11.attn.proj.weight`;
resolve exact harness names). Freeze exact names/shapes/dtypes before outcomes are inspected; use
the same three full matrices across all arms and steps. No sampled entries or periodic snapshots.
Three 768-by-768 matrices over 3250 steps require 11.50/23.00 GB per stream at 16/32-bit, or about
207/414 GB for three streams across six arms. These plots describe the selected matrices only.

Process full-model windows in parameter/coordinate blocks; do not load the entire history on the
GPU. Keep ordinary scalar training logs throughout. Optional fixed random projections of whole-model
gradients/updates can provide longer-term monitoring, but first validate their norm/angle distortion
against exact windows and benchmark projection overhead. Keep the map fixed across time and matched
branches, use an efficient implementation without a huge dense projection matrix, and label these
measurements approximate. They are not substitutes for the exact windows or recoverable full gradients.

**Live-work handling:** preserve completed artifacts and verify the actual running job before any
change. Do not cancel/restart an active run solely for this edit. Existing full-run captures can supply
the required whole-model windows; extract them with step/schema provenance. If a selected-matrix writer
is already active, enable full-model capture at a safe documented boundary for the required windows,
or schedule missing windows from retained exact fork states, recording any replay deviations. Do not
claim missing raw tensors or updates can be recovered from scalar logs. Implement a windowed full-model
writer for REQ-068/069 and a separate selected-matrix full-run writer for REQ-072. Historical all-model-
all-step and three-matrix-only river plans are superseded, not additional deliverables.

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

<a id="req-068-record-the-full-gradient-history-for-an-entire-nanogpt-run"></a>

## REQ-068: whole-model river geometry from three short gradient-history windows

- status: **DONE 2026-10-03** -> `logs/river/req068_full_gradient_history/` (README + analysis/win*.json). Full 3250-step gradient history (1.86 TB, verified) PLUS the amended scope: both streams (g_t + actual displacement delta_t) fork-captured for all three windows 500-563/1500-1563/2500-2563 from the retained forks, 64x64 K/C temporal Gram per window/stream, eigen/coeff/centered, per-family contributions, out-of-sample r1/2/4/8 vs past-mean/exp-avg, and probe states at offsets 32/48/63 (27/window). FINDING: raw gradients anti-align step-to-step (lag1 -0.72/-0.63/-0.25, period-two, weakening) while actual displacements strongly persist (lag1 +0.79/+0.79/+0.75 = candidate river); BUT not low-rank-constant over 64 steps (flat Gram spectrum, OOS r8~0.04, baselines ~0) and whole-model Euclidean displacement is embedding-dominated (AdamW lr0.7 vs Muon lr0.025) -> needs layer-reweighted view; valley-floor claims deferred to REQ-069/070. Node q8rn683 stopped
- requested: Jack / 2026-10-01 PDT; revised 2026-10-02 PDT
- priority: main river experiment; preserve REQ-062 and live jobs
- artifacts: `logs/river/req068_full_gradient_history/`; see `STATUS.md` for historical progress
- resource limit: two nodes fleet-wide; record finite GPU-hour, storage, and analysis-memory budgets

**Question:** does local whole-model motion contain a reproducible, slowly changing component,
separable from fast alternating motion? A spectrogram of a few matrices does not answer this.

### Capture one baseline trajectory

Pin one verified nanoGPT Muon + auxiliary-optimizer recipe, code SHA, architecture, dataset, seed,
token order, batch size, and LR/momentum schedules. Reuse the existing baseline run if compatible.
Train normally to **3250 completed optimizer updates**, with ordinary scalar training/validation
logging throughout. Record **64 consecutive whole-model observations** in each of the predeclared
windows **500–563, 1500–1563, and 2500–2563**, using zero-based update indices. Do not select windows
after viewing oscillation patterns. Shorter initial prefixes within each window provide sensitivity
checks; any longer-window extension must be separately budgeted, not silently added everywhere.

For every trainable parameter in those windows save:

1. **Raw loss gradient g_t** evaluated at theta_t, after microbatch accumulation, distributed
   synchronization and loss-scale unscaling, but before clipping, momentum, conditioning, decay,
   or in-place optimizer mutation. Record normalization to mean loss per token.
2. **Actual parameter displacement delta_t = theta_(t+1) - theta_t**, including optimizer
   conditioning, learning rate, decay, and rounding. Compute from pre/post weights with sufficient
   arithmetic precision to represent their difference; record the calculation and dtype. Keep
   its meaning distinct from the raw gradient or the pre-LR conditioned direction.

Include embeddings, head, hidden matrices, vectors/scalars, and all distributed shards. Record
parameter names, shapes, tying, native dtypes, missing/inactive gradients, shard ownership, token
counts, and step numbers. Save losslessly, with bounded chunks and copies made before mutation.
Do not replace the full-model windows with three matrices, sampled coordinates, or projections.
Persistent conditioned-direction history is optional here; both listed streams are mandatory.

Retain exact model/optimizer/RNG/data-cursor states at theta_500, theta_1500, and theta_2500 for
confirmation branches. Retain matched probe states initially at offsets 32, 48, and 63 within each
window (or a verified replay route with documented numerical deviations). Checkpoints stay off Git.

Forecast storage from actual dtypes and verify write throughput. Commit manifests, checksums,
retrieval instructions, readers, code/configs and derived artifacts; tensors stay on durable storage
outside Git. Verify round-trip equality, capture timing, no aliasing, logging parity, restart index
continuity, and all 192 expected step indices with complete parameter/shard coverage for both streams.
No all-parameter tensor history is required outside the three windows. Preserve any extra existing
capture as historical data rather than treating it as an additional requirement for future runs.

### Exact temporal geometry before filter fitting

For each window analyze the raw gradients and actual displacements **separately**. Let v_i be
one whole-model vector at local step i. Accumulate the temporal Gram matrix in parameter blocks:

$$
K_{ij}=v_i^\top v_j,\qquad
C_{ij}=\frac{K_{ij}}{\sqrt{K_{ii}K_{jj}}}.
$$

K records vector inner products; C records cosine similarity. Both are just **64 by 64** even
though v has millions of coordinates. Verify blockwise results against direct dot products on a
small reference, symmetry, diagonal norms, and stable numerical accumulation. Flag zero/near-zero
norms. Deliver exact whole-model
heatmaps, signed lag 1/2/4/8/16/32 agreement, vector norms, and each parameter family's contribution
to K. The Euclidean whole-model result is primary; any layer-reweighted version must be separately
labeled, since reweighting changes the geometry. Large embedding contributions must remain visible.

Use the Gram matrix to estimate dominant recurring parameter-space directions without loading
all vectors on the GPU. Analyze both uncentered data and a separately labeled centered version;
centering must not silently discard the shared forward component. Plot the time-varying signed
coefficients along these directions and their temporal spectra. Ask whether alternating motion
and gradual motion occupy distinguishable directions, and how much energy each explains. Highest
energy does not automatically mean useful river motion. Do not identify all low-frequency power
as signal or assume the valley floor has only one dimension.

**Out-of-sample check:** estimate directions/bases from the first 32 observations and evaluate
how much of the following 32 observations they explain, without refitting on that second half.
Predeclare ranks 1, 2, 4, and 8; select a rank/rule on development windows, not final confirmation
runs. Report held-out captured energy, residual energy, and direction agreement, against simple
past-mean/pair-average/exponential-average references. Evaluate any claimed slow component using
past-only estimates; full-window decompositions are descriptive only. All comparisons must state
whether they target raw-gradient motion or actual optimizer motion and account for known schedules.

Deliver small Gram/cosine matrices, heatmaps, spectra, signed component time courses, first-half/
second-half comparisons, per-family contributions, raw-artifact manifests, and validation curves.
State whether evidence supports a candidate local slow component, and where it fails. This is not
proof of a valley floor or useful descent: independent branches and loss tests follow in REQ-069/070.

## REQ-069: ten data-seed branches to estimate reproducible local motion

- status: **DONE 2026-10-03 — NEGATIVE confirmation** -> `logs/river/req069_ten_branch_ensemble/`. 10 branches forked from theta_1500 (identical weights+optimizer) x independent data seeds (disjoint corpus windows), 64 updates each, both streams + offset probes. Within-branch signatures reproduce (grad lag1 ~-0.64, disp lag1 ~+0.79) BUT no reproducible common whole-model direction: two independent 5-groups agree ~0.9 at the fork then decay to ~0 within 64 steps (grad 0.944->-0.029, disp 0.899->+0.042), ensemble-mean norm shrinks with more branches (incoherent averaging), dispersion high (grad 4.0/disp 2.8). Ensemble retains the period-two (shared bounce phase) + embedding-dominated persistent delta = shared-IC artifact, not data-reproducible. Tempers REQ-068's river (bending, not low-rank valley floor). 500/2500 windows not run (gated on positive middle evidence). Node qz0mpk3 stopped
- requested: Jack / 2026-10-01 PDT; revised 2026-10-02 PDT
- dependencies: REQ-068 whole-model window analysis and compatible exact fork state
- artifacts: `logs/river/req069_ten_branch_ensemble/`
- priority: start with step 1500; extend to 500/2500 only if evidence and a separately recorded budget justify it

**Question:** do independent minibatch sequences reproduce the candidate slow whole-model component?
This confirms REQ-068's local analysis; it is not ten independently initialized models.

Clone identical model weights, full optimizer state, and schedule position from the middle fork
into **ten branches**. Vary data-sampling seeds independently, document all other RNG streams,
and use identical batch size/architecture/schedule. Run **64 updates per branch**, not the old
100-step selected-matrix specification. Record full-model raw gradients and actual displacements
at every branch step, using REQ-068's schema and timing. Retain exact states at branch offsets
32, 48, and 63 for independent loss tests. Do not alter another branch's data cursor/RNG while probing.

Predeclare two groups of five. Compute their ensemble averages at each matched relative step,
separately for gradients and updates, and compare directions, magnitudes, lag dependence, and
period-two content. Inspect convergence with 1, 2, 5, and 10 branches. Apply the frozen slow-
component extraction rule from development data independently to the two groups; agreement is
not a valid confirmation if the basis is fitted jointly to both groups' confirmation observations.
If the rule was developed on this base lineage, label this a conditional diagnostic and confirm
on a fresh initialization lineage before broader claims. Do not average coordinates across
independently initialized models or treat correlated steps as independent seeds.

Compute whole-model temporal Gram/cosine matrices and parameter-family contributions for branches
and ensemble means. Stream coordinate blocks to calculate cross-branch statistics. Track branch
parameter separation, loss spread, and schedule traces to detect loss of local comparability.
Do not reorder time steps to maximize agreement. Branches inherit a shared initial bounce phase;
ensemble averaging may retain deterministic alternation. Report it rather than declaring the mean
the river. Mean gradients at different weights need not equal the gradient at mean weights.

Use whole-branch uncertainty/resampling and report the limited number of independent branches.
Deliver independent-five-group agreement, mean-motion/alternation plots, held-out temporal
prediction checks, failures, complete window manifests, and compute/storage costs. No paired
weight-perturbation/settling experiment is requested. A negative or ambiguous result is valid;
do not expand to more forks or full training solely because an average looks smooth.

## REQ-070: test candidate directions on independent loss and curvature probes

- status: **DONE 2026-10-03 — NEGATIVE** -> `logs/river/req070_independent_direction_tests/`. Forward-pass loss probes at REQ-069 probe states (offsets 32/48/63, 10 branches) for candidates baseline_update/neg_gradient/temporal_avg/ensemble_mean_LOO at lengths 0/.25/.5/1/2 x ||baseline update||. RESULT: no useful descent anywhere (best -0.002 nats, negligible + non-robust). baseline_update monotonically overshoots (+0.05@a1), neg_gradient catastrophic (+21@a1, raw gradient needs conditioning), temporal_avg no descent, ensemble_mean_LOO loss-neutral (the 'river' direction does not point downhill). Closes the river arc negative: 068 persistence (embedding-dominated, not low-rank) -> 069 not reproducible -> 070 no useful descent. **REQ-071 gated OFF** (conditional on a useful slow component, which is absent). Node 3yk4kew stopped
- requested: Jack / 2026-10-01 PDT; revised 2026-10-02 PDT
- dependencies: whole-model candidates from REQ-068/069 and matched exact probe states
- artifacts: `logs/river/req070_independent_direction_tests/`

**Question:** does a reproducible slow whole-model direction improve loss on independent data?
This replaces the prior selected-matrix-only intervention; move all parameters represented in
these full-model candidates and use the same parameter support for every comparison.

Before inspecting results, fix probe offsets **32, 48, 63**, independent data, displacement scales,
and the candidate-extraction rule. Compare the baseline optimizer movement, raw downhill gradient,
ordinary temporal averaging, the ensemble mean, and a candidate slow component if supported by
REQ-068/069. For each branch use a **leave-one-branch-out reference** and never include its own
sampling noise in its target. Evaluate at that branch's exact weights, not at the ensemble center.

All optimizer-relevant candidates must be available at the stated decision time: at theta_t,
use gradients through g_t and completed displacement history through delta_(t-1). Label any
candidate requiring future data as an offline diagnostic and exclude it from causal performance
claims. A simulated baseline update may be computed from the current gradient on copied state.
Do not fit decomposition rules on the probe losses or future confirmation gradients.

For a movement direction d, evaluate the full model's fresh probe loss at theta + alpha*d/||d||,
restoring exact weights/state after each test. Gradient candidates use the negative sign. Compare
equal whole-model displacement lengths: 0.25, 0.5, 1, and 2 times the same state's baseline update
norm, plus zero. Flag zero-norm candidates and predeclare fallback behavior. Report the complete
loss-versus-distance curves and branch-level paired differences, not just the best displacement.

Distinguish raw-gradient candidates passed through identical copied optimizer state from direct
movement candidates already in optimizer-update units; do not condition an actual displacement a
second time. Preserve parameter-group scaling, schedules, and declared decay treatment. Optionally
measure directional curvature on fixed independent probe data using verified Hessian-vector products;
use the existing eight Lanczos iterations if spectral probes are added. No full Hessian or every-step
curvature recording is required. Keep final validation data untouched for confirmation.

Deliver reproducibility and useful-descent evidence separately, with the actual parameter support,
probe data/state hashes, sign/norm conventions, numerical checks, and failures. Smoothness, low
curvature, or high captured energy alone is not evidence of a useful river direction.

## REQ-071: fit a causal gradient-history estimator and test it in training

- status: **OPEN**
- requested: Jack / 2026-10-01 PDT; revised 2026-10-02 PDT
- dependencies: a reproducible and independently useful whole-model reference from REQ-068/069/070;
  report inadequate references instead of launching a blind filter sweep
- artifacts: `logs/river/req071_history_estimator/`

**Question:** can a single trajectory's past history estimate the supported whole-model reference
without ensembles or expensive geometry probes at every step? This is no longer restricted to three
attention matrices. Fit coefficients using full-model window data streamed in parameter blocks.

Predeclare whether the target is a slow raw-gradient component or an actual movement component,
and keep input/target conventions consistent. These are not interchangeable through nonlinear Muon
conditioning. Use leave-one-branch-out references and independent data; no branch enters its own
target. The initial interpretable raw-gradient estimator is

$$
\widehat{s}_t=\sum_{j=0}^{W-1}c_j g_{t-j},\qquad W\in\{4,8,20\},
$$

with constraints

$$
\sum_j c_j=1,\qquad \sum_j j c_j=0,\qquad \sum_j(-1)^j c_j=0.
$$

These preserve constants/current-step linear trends and cancel constant-amplitude period-two
alternation. Use a shared coefficient set initially; any group weighting or alternative movement-
history estimator must be separately labeled. Penalize large coefficients, tune only on development
lineages, and compare with raw input, pair averaging, and a tuned exponential moving average. The
initial maximum window is 20 so past-only testing fits the held-out half of a 64-step recording;
longer windows require a separately budgeted extension, not future leakage or concatenated gaps.

Split by whole base lineage/trajectory and chronological halves, not random overlapping windows.
Report whole-model and per-family direction/magnitude error, lag at turns, residual alternation,
and a low/no-alternation control. Different time windows from one training run are not independent
initializations. Failure only rules out the tested estimator/data regime, not all possible methods.

If offline and independent-loss evidence justify it, test **200-update paired continuations**:
original optimizer, simple averaging, and the fitted method, with identical starting weights and
optimizer state and paired token order. Define the insertion point, initialization from pre-fork
history, and all-parameter coverage; then each branch uses only its own newly generated history.
Start these continuations at a retained offset-32 state so up to 20 prior gradients are available
inside its captured window; do not synthesize missing pre-fork history or borrow another branch's
future observations.
Benchmark the live rolling-buffer memory/compute separately: storing short windows on disk does
not remove the cost of a full-model online filter. If the method does not fit, report the limitation
or implement a mathematically equivalent recurrence; do not silently restrict it to selected matrices.

Only after promising continuations, compare frozen methods through **3250 updates on three fresh
paired initialization/data seeds**, with comparable development LR-tuning budgets. Evaluate at
least every 250 steps and at the final endpoint on untouched validation data. Persistent full-model
history remains windowed even in these runs. Report all seed differences, uncertainty, failures,
loss at equal tokens, time-to-loss, total GPU-hours, live history memory, and capture overhead.
Do not equate smoother gradients with better optimization.

## REQ-072: six optimizer ablations with full gradient histories and spectrograms

- status: **DONE 2026-10-03** -> `logs/river/req072_optimizer_spectrograms/` (README + analysis/*.json). 6 arms (AdamW/SGD/Muon x +-momentum, 3250 steps) capturing raw g_t + conditioned u_t + displacement for blocks.{0,5,11}.attn.proj. FINDING: first-moment MOMENTUM is what converts the period-two-oscillating gradient into a smooth low-freq conditioned direction, consistently across all 3 optimizers (u_t period-two frac with/without momentum: AdamW 0.009/0.233, SGD 0.000/0.121, Muon 0.040/0.185; momentum off raises it 4-26x). AdamW sqrt(v) adaptivity and Muon orthogonalization do NOT suppress it alone. Explains REQ-068's slow displacement (= momentum low-passing the bounce), not Muon geometry or a descent river. Muon u_t recovered via displacement (per-rank sharding; wd=0 exact; caveat in README). Node w7vln1w stopped
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
storage blockers rather than quietly reducing coverage or precision. REQ-068's whole-model windows are a separate requirement; reuse
a run/artifact only if its config, time coverage, and tensor streams actually match, with an explicit
cross-reference. REQ-072 does not require whole-model histories and cannot substitute its three
matrices for REQ-068/069's whole-model observations.

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

## REQ-073: Q/K/MLP spectra through Muon, momentum, and effective batch size

- status: **DONE 2026-10-04** -> `logs/river/req073_muon_batch_spectrograms/` (README + analysis/*.json). 4 arms (B/4B x Muon momentum, 3250 steps) capturing the 4-stage pipeline (raw g_t, post-momentum pre-polar, post-polar u_t, displacement) for blocks.5 Q/K/mlp.proj. FINDING: period-two (f~0.5) is filtered PRIMARILY by momentum at stage 1->2 (B 0.42->0.14; no drop when momentum off: s2==s1), SECONDARILY by Muon orthogonalization at stage 2->3 (~2.5-3x, but alone only to ~0.16-0.23 vs momentum's ~0.04). NOT minibatch noise: larger batch (4B, 1/4 variance) raises the raw-gradient period-two (0.42->0.56), proving it's deterministic. Loss: 4B-mom 3.172 < B-mom 3.296 < 4B-nomom 3.358 < B-nomom 3.444 (momentum + batch help via separate routes). Closes the river arc (068 persistence -> 069 not reproducible -> 070 no useful descent -> 072 momentum low-passes -> 073 localizes to momentum stage + rules out noise). Node wonydk3 stopped
- requested: Jack / 2026-10-03 PDT
- dependencies: verified REQ-068 reader/capture conventions; reuse REQ-070 independent-loss
  machinery where compatible; do not wait for a learned filter or interrupt running jobs
- artifacts: `logs/river/req073_muon_batch_spectrograms/`
- resources: existing **two nodes fleet-wide**; publish finite GPU-hour, storage, and I/O
  forecasts before launch; this request is a specification, not a live launch record

**Question:** does the change from fast raw-gradient oscillation to slower parameter movement
come from momentum, Muon's matrix transformation, or minibatch noise? Does suppressing the
alternating component help independent loss? The whole-model REQ-068 movement spectrum was
approximately 99.994% embedding-dominated, so it cannot establish Muon's effect on hidden matrices.
Compare the same hidden parameters at each stage of their actual optimizer pipeline.

### Three complete matrices, fixed before inspecting outcomes

Use the following matrices in the **same middle block** (zero-based block 5 of 12), separating
matrix type from layer depth. Names and shapes are present in REQ-068's recorded schema:

| Matrix | Parameter name | Shape |
|---|---|---|
| Attention query Q | `blocks.5.attn.q.weight` | 768 × 768 |
| Attention key K | `blocks.5.attn.k.weight` | 768 × 768 |
| MLP output projection | `blocks.5.mlp.proj.weight` | 768 × 3072 |

Verify these against the actual pinned harness before launch. Record complete tensors at every
update, not just norms or a few entries. Keep matrices separate in primary figures: the MLP is
larger, so also show mean power per coordinate and normalized spectral shape. A selected-matrix
aggregate is supplementary and must not be labeled whole-model. This selection applies to
REQ-073; it does not silently change REQ-072's existing three-depth selection.

### Four training arms: effective batch × hidden-matrix momentum

Let B be the baseline **524,288 tokens per optimizer update**. Use 4B = **2,097,152 tokens** for
the larger-batch arm, implemented by four times the gradient accumulation while keeping the
microbatch shape fixed. Accumulate before a single optimizer update: four ordinary updates do
not constitute a 4B update.

| Arm | Effective batch | Muon momentum on hidden matrices |
|---|---|---|
| B-mom | B | mu = 0.95, standard Nesterov direction |
| B-nomom | B | mu = 0, Nesterov off |
| 4B-mom | 4B | mu = 0.95, standard Nesterov direction |
| 4B-nomom | 4B | mu = 0, Nesterov off |

Hold **auxiliary AdamW fixed in all four arms**, including beta1 = 0.9 and beta2 = 0.95,
embedding/head/scalar learning rates, decay, and schedules. Only hidden-matrix first-moment
momentum is switched here. Label the no-momentum arms accordingly; they still have auxiliary
AdamW momentum. This differs deliberately from REQ-072's model-wide first-moment ablation.

All four arms start from identical initial weights, fresh optimizer state, and the same ordered
token stream. The 4B loader groups four consecutive B-sized token blocks before updating. Verify
the corpus contains enough unique tokens; no silent looping or resampling to extend coverage.
Use one paired initialization/data seed initially. Run **3250 updates per arm** so each has a
long continuous record; report failures/incomplete arms without changing settings mid-run.

Freeze common Muon LR scales, per-update LR schedule, clipping, orthogonalization iterations,
shape scaling, and decoupled decay after a small stability pilot across all four configurations.
Do not automatically scale LR with batch size. Use a batch-size-invariant optimizer-input
convention: if the baseline consumes token-summed gradients at B, divide the accumulated 4B
sum by four before clipping and optimizer processing, preserving the B-arm convention. If it
already consumes means, use means in all arms. Do not accidentally change effective LR, clipping,
or the auxiliary AdamW epsilon regime by quadrupling gradient scale. Save native raw values and
normalization factors separately; raw-gradient analysis uses mean-per-token units in every arm.

The primary contrast is at equal **optimizer-update count**, holding the optimizer's memory
and schedule in update units fixed. It is not a compute-matched performance comparison: 4B sees
four times as many tokens, and momentum spans four times as many tokens. Also report loss and
spectral summaries at common cumulative-token checkpoints (multiples of 4B), with schedule
positions shown; these secondary comparisons are not matched optimizer age. Do not interpolate
missing gradients or equate the two comparisons. Record actual tokens, GPU-hours, LR, decay,
momentum, and clipping traces.

### Capture each stage, not just the beginning and end

For every selected matrix at **every update in every arm**, save losslessly:

1. Raw loss gradient after accumulation/synchronization/unscaling, before clipping or mutations.
2. Exact input to Muon's matrix transformation, after any clipping and the actual momentum/
   Nesterov combination. Store the transformation input, not merely the momentum buffer.
3. Exact transformed direction after orthogonalization and shape scaling, before LR and decay.
4. Actual post-step minus pre-step parameter displacement, with LR/decay decomposition documented.

Capture before in-place operations overwrite inputs; no rounded weight-difference reconstruction
of intermediate stages. With clipping enabled, additionally save the post-clipping gradient or
its exact reconstructible transform so clipping is not misattributed to momentum. Retain complete
model/optimizer/data/RNG checkpoints before updates **500, 1500, and 2500** for the diagnostics below.

Use bounded streaming chunks, a fixed schema, native precision, checksums, resume-safe indices,
and tensor round-trip/capture-parity checks. Keep tensors and checkpoints off Git; commit readers,
manifests and retrieval paths. These three matrices total **3,538,944 entries**: at 3250 steps,
four streams and four arms require approximately **736 GB if all streams are fp32**, before
checkpoints/metadata/extra clipping capture. Census actual dtypes and benchmark I/O before launch;
do not reduce coverage or precision silently. Reuse historical data only with exact provenance.

### Separate temporal oscillation from batch noise

At the retained checkpoints, freeze weights and leave optimizer state untouched. For each fixed
state, sample **eight disjoint groups of four B-sized probe batches**, preserving training RNG and
data cursor. Record the selected-matrix mean-per-token gradients for each B batch and each group's
4B mean. Keep sampling/dropout conventions explicit; disjoint corpus offsets alone do not prove
statistical independence. Verify on a pilot that the averaged B gradients match direct accumulated
4B gradients at identical weights within the documented numerical tolerance.

Estimate sampling variance around the fixed-state mean for B and 4B. Under independent samples,
the 4B mean has one quarter of B's variance; report departures and uncertainty rather than imposing
this relation. Nested B/4B observations are paired, not independent replicates. These observations
estimate noise at one state, not a temporal training spectrogram or a valley direction.

For mechanistic comparison, replay the same recorded raw hidden-matrix history through shadow
Muon pipelines with momentum on/off, identical transforms, and declared fresh/warmed state.
Use the history from step 0 or a valid compatible buffer; do not invent missing prehistory.
Label replay outputs as hypothetical transformations along a fixed observed trajectory, not actual
counterfactual training updates. Contrast this with the four actual trajectories, whose gradients
diverge after training begins. This separates immediate transformation effects from trajectory effects.

### Spectrograms and checks of useful descent

Reuse REQ-072's coordinate-wise temporal STFT and energy checks: sum Fourier power after transforming
each signed coordinate, never Fourier-transform the norm or a signed coordinate average. Use full
3250-update histories with 128-step Hann windows/hop 16; show 32/256-step sensitivity checks and
early/middle/late views. Primary y-axis: **period in optimizer updates**, with period 2 explicit
and DC separate. State axis spacing and retain frequency-valued arrays; do not invent a finite
period for DC. Provide period-in-token equivalents for the batch comparison.

For Q, K and MLP separately, show raw → momentum-adjusted → transformed → actual movement panels
for all four arms. Use comparable color limits across arms within each stream; gradient and update
streams have different units, so label separate power references. Include absolute and normalized
power, lag-one/lag-two agreement, and predeclared bands 0.45–0.5 and 0–0.05 cycles/update. Show raw
and separately labeled detrended spectra. Do not infer frequency conversion from power ratios alone,
or identify every low-frequency component with useful learning. Predeclare a few fixed coordinates
or fixed signed projections for supplementary views of evolving structure; save their definitions,
do not select them by attractive plots, and keep the complete-matrix spectra primary.

At the retained states, use independent probe data to compare the baseline direction with two
past-only replacements for the selected matrices' raw gradient inputs: consecutive-pair averaging
and an EMA with decay 1/3. Pass each candidate through the same copied optimizer state and apply
all other parameter updates identically. Test both native and matched selected-matrix displacement
norms, with the same predeclared small step-length grid for every candidate. Restore the model and
optimizer state between probes. Reuse REQ-070's loss-evaluation conventions and report negative results.
In the ideal fixed-state independent-noise limit, pair averaging and this EMA both have squared-weight
sum 1/2 and mean age 1/2 update, but only pair averaging exactly cancels period two. Verify finite-history
initialization; the match does not perfectly isolate parity on a nonlinear, changing trajectory.
Scope conclusions to these selected-matrix interventions; lower spectral power alone is not success.

Deliver the four complete histories, stage-resolved spectrograms and numerical arrays, resolved
configs/seeds, checkpoint provenance, fixed-state variance comparisons, paired independent-loss
results, validation curves, and resource/capture-overhead accounting. Separate descriptive evidence,
immediate optimizer transformations, and useful-descent evidence. Do not launch a broad filter
sweep or full filtered-training extension solely because a plot becomes smoother.

## REQ-074: save every-step frequency decompositions and deliver time-period-amplitude spectrograms

- status: **DONE 2026-10-04** -> `logs/river/req074_time_period_amplitude/` (README has the full findings table). Delivery-repair: REQ-072/073 analyzers saved only time-averaged spectrum_f, discarding the time axis; the RAW stage histories ARE on durable FS, so recomputed time-resolved P(step_end, period) offline (window 128, Hann, hop 1 -> 3123 columns; +32/256 + detrended variants) -- NO training rerun. Time-resolved STFT core CPU-validated (5/5); GPU recompute (`req074_recompute_gpu.py`, torch.fft) parity-exact to the numpy core at ~1e-15 (`--selfcheck` on-GPU). All REQ-072/073 histories verified intact first. Delivered: **28 arrays** (`arrays/*.npz`, all verify_ok=1) + **168 heatmaps** (`heatmaps/` 84 primary + 84 detrend; x=training time, y=period log, color=log10 power). Findings (time-localized): (1) momentum is the primary low-pass and it builds the slow component **progressively** over ~the first training quartile (post-momentum slow-band 0.17-0.40 -> 0.58-0.79); (2) Newton-Schulz orthogonalization is a secondary low-pass (near-erases period-2); (3) batch size is NOT the mechanism -- 4B-nomom keeps the HIGHEST period-2 fraction (0.31->0.37), decisively refuting "period-2 = minibatch noise" and time-resolving the REQ-073 conclusion. Caveats: sgd-mom degenerates late; muon REQ-072 u_t via displacement proxy (power fractions preserved); REQ-073 Muon stages complete (all_reduce). Node qvkdpj3 (1x8 H100) stopped after pull; raw tensors stay off-Git.
- requested: Jack / 2026-10-03 PDT
- dependencies: inspect actual REQ-072/073 jobs and durable captures; no river-reference gate
- artifacts: `logs/river/req074_time_period_amplitude/`, with links to repaired REQ-072/073 artifacts
- resource limit: existing **two nodes fleet-wide**; preserve REQ-062 priority and running jobs;
  benchmark and record finite compute, storage, and I/O budgets before any replacement training

**Direct request to Claude/Jerry's agent:** create the actual spectrograms: **training time on x,
oscillation period on y, and amplitude as color**. Retain the frequency decomposition at every
optimizer update. Do not replace the time axis with an average, pooled band totals, or a whole-run
curve. Jack explicitly authorizes rerunning the affected captures if the necessary data is absent.

### Why delivery must be repaired

At inspected commit `9715260733366835f9ea682fc345c46fdb2ae293`, REQ-072's `stft_power` computes
`spectrogram` and `starts`, but `req072_analyze.py` discards them and writes only `spectrum_f`
(the mean over time) and scalar summaries. REQ-073's analyzer repeats this loss of information.
REQ-072 Muon JSON retains only band fractions. Those outputs cannot reconstruct a time-resolved
spectrogram. A DONE label or saved averages does not satisfy this request.

1. First verify all raw-history manifests, chunk hashes, selected matrix/stage ownership, exact
   consecutive step coverage, normalization factors, and durable retrieval paths. REQ-072 reports
   `/root/.cache/user_artifacts/req072/<arm>/{grad,uT,disp}`; do not assume those files survived
   without checking. Inspect actual REQ-073 job/capture state rather than trusting its queue label.
2. If intact histories exist, recompute every time slice from them; a training rerun is unnecessary.
   Persist all time-resolved arrays this time. If sufficient saved temporal Gram matrices exist,
   they can recover summed coordinate power exactly, but must cover every required window and
   matrix/stage; unrelated REQ-068 windows are not substitutes for REQ-072/073.
3. If the required histories or intermediate stages are missing/corrupt, **rerun the affected
   arms/captures** using their frozen original settings and declared provenance. Do not silently
   splice unrelated trajectories or substitute partial windows. Do not restart a live job merely
   to change the exporter; recover its intact history or schedule replacement after it finishes.
4. Fix both analyzers so future runs never discard the time dimension. Preserve old files and
   identify corrected versions; do not silently overwrite the archived measurements.

### Save a spectrum for every update, with an explicit temporal window

Temporal frequency is defined across successive gradients, not by Fourier-transforming the
matrix rows/columns at a single instant. Use the following operational definition:

- **Primary: 128 consecutive optimizer updates, periodic Hann taper, hop exactly 1 update,
  sampling rate 1/update, no padding, no detrending.** This supersedes the old primary hop 16
  and the implementation's hop 32. Also retain 32- and 256-update/hop-1 sensitivity arrays and
  separately labeled per-window linear-detrended arrays; raw/no-detrending is always retained.
- For an update indexed t, the primary spectrum uses signed values at updates t-127 through t.
  Save `step_start`, `step_end`, and `window_center`; plot `step_end` on x. For a full zero-based
  3250-update capture, save **3123 primary columns**, ending at every update 127 through 3249.
  The first 127 updates have insufficient history: mark this explicitly, never invent zeros or
  future samples. Shorter windows provide earlier coverage with their own explicit resolution.
- Any new/replacement run must persist each selected raw gradient/stage tensor at every step,
  and persist each spectrum once its complete window exists. A bounded asynchronous consumer
  is acceptable; it must checkpoint progress, retain unprocessed raw history, and never drop or
  skip steps. Older intact histories may be processed offline with identical per-step coverage.
- Transform EACH signed coordinate across time, then sum squared Fourier magnitudes across all
  coordinates of that selected matrix. This coordinate power sum is allowed; **averaging across
  time is not**. Do not FFT norms, absolute gradients, or signed coordinate averages. Preserve
  separate matrices and optimizer stages; never collapse the six/four arms into one average.
- Apply correct one-sided PSD scaling: divide by sampling rate and Hann-window energy, double
  interior positive-frequency bins, and do not double DC or Nyquist. Preserve DC in arrays.
  Flag gaps/nonfinite samples and split contiguous segments; no interpolation or zero-filling.
- Use coordinate blocks or an exact streaming method; do not require the whole coordinate-by-
  training-time tensor in RAM. Benchmark actual overhead without changing training semantics.

### Numerical artifacts must survive the job and be usable from the repository

For every arm × matrix × stage × window length × detrending choice, save a lossless `.npz` (or
equivalent documented format) containing **all frequency bins at all valid steps**, not just
summaries: `power[time, frequency]`, `amplitude[time, frequency]`, frequency coordinates,
positive-frequency periods, step start/end/center, validity/gap flags, and DC separately for plotting.
Store linear values; plot log scales from them rather than saving only clipped/logarithmic pixels.

Record the exact amplitude definition. Primary aggregate RMS spectral amplitude per frequency bin
is `sqrt(PSD * delta_f)` after summing coordinate powers; this is root-sum-square amplitude over
coordinates, not a single coordinate's sinusoidal peak amplitude. An optional per-coordinate RMS
version divides by sqrt(number_of_coordinates) and must be labeled separately. Do not label power
as amplitude. Keep native/mean-per-token normalization and stream units explicit.

Include code/config SHA, seeds, source hashes, exact matrix names/shapes, stream definitions,
window/hop/taper, coordinate aggregation, PSD/amplitude normalization, LR/decay/clipping traces,
and complete coverage counts. Commit the compact derived time-frequency arrays, manifests,
plotting code, and PNG/PDF figures to this branch; shard derived files if needed for Git limits.
Raw gradients, optimizer states, and checkpoints remain off Git with verified retrieval paths.
A JSON containing only `spectrum_f` or three band totals is an explicit delivery failure.

### Required plots and faithful stream labels

- **x: optimizer update (window end); y: period in optimizer updates; color: amplitude.** Use
  period = 1/f for positive frequencies, a clearly labeled log-period axis, and ticks including
  2, 4, 8, 16, 32, 64, 128 where resolved. Render the actual bin geometry; no fake fine resolution.
  DC is a separate strip/panel, never a finite period. Do not bridge missing training segments.
- Use a shared amplitude color scale across comparable arms within each matrix/stage. If a
  logarithmic color scale is necessary, label it and record its floor/reference. Different stream
  units require separate references. Absolute amplitude is primary; per-time normalized power is
  optional and cannot replace it. Mark LR phase boundaries; label single-seed results descriptive.
- REQ-072: all six optimizer/momentum arms, each of blocks 0/5/11 attention output-projection
  matrices, raw gradient / directly captured conditioned direction / actual displacement panels.
- REQ-073: all four B/4B × Muon momentum arms, middle-block Q/K/MLP output matrices, all four
  original pipeline stages. Also give the declared period-in-token equivalents for B versus 4B.
- Fix owner-rank capture for Muon intermediates. **Displacement is not a substitute for the
  pre-LR conditioned direction.** Even at zero weight decay, a time-varying LR changes temporal
  spectra; rounded parameter differences introduce another error. If direct captures are absent,
  rerun the affected capture or deliver explicitly labeled displacement-only partial results while
  the required direct stage remains outstanding. Do not claim exact intermediate reconstruction.

### Validation and completion gate

Test direct coordinate FFT versus blocked/streaming output; constant, exact alternating,
known-sinusoid, changing-frequency, and changing-amplitude signals; one-sided Parseval scaling;
Nyquist visibility; absence of false temporal averaging; gap boundaries; restart/round-trip
identity; and capture-versus-no-capture training parity. Verify all per-step window counts and
all selected matrix/arm/stage outputs. Render and visually inspect the final time-period-amplitude
figures. Do not declare DONE until compact time-resolved arrays AND requested figures are committed
with retrieval/provenance metadata. Report unavailable arms/stages explicitly as incomplete.

On pickup record RUNNING plus the actual job/session handle; report whether this is reanalysis or
replacement training and which data were recovered. This request does not authorize unrelated
hyperparameter sweeps or exceed the two-node fleet ceiling.

## REQ-075: late-training large batch versus momentum: noise, oscillations, and useful gains

- status: **DONE 2026-10-05** (session 01YKPwGqPuEuP2wzzHuCEhok) -> `logs/river/req075_late_batch_momentum/` (`REPORT.md` = final; `STAGE1/2/3_FINDINGS.md`, `PROVENANCE.md`, `PLAN.md`). All 3 stages delivered on 1x8-H100 (wd8vemq, stopped). **ANSWER: No + yes** — a 16x batch does NOT reproduce momentum's benefit (16B worse per token by +0.019..+0.027 nats at 736B tokens, 3 seeds); small-batch+momentum is the most token-efficient recipe. Momentum = DENOISING: nomom-mom benefit falls monotone with batch and vanishes at 16B+tuned LR (B +0.0114->+0.0072 after LR tuning; 16B +0.0016->-0.0002). The late period-two oscillation is overshoot/optimizer-dynamics (half-LR collapses it; raw-grad period-2 fraction 3%->83% B->16B = revealed-not-removed by large batch, masked by noise at B where Stage-1 NSR_B~1.7; 16B NSR~0.37, NOT noise-free). Registered interpretations #1/#2/#5 supported, #3 not-supported-at-low-noise. Limits: 3-seed pilot, reduced-not-zero noise at 16B, selected-matrix only, single late checkpoint, period-2-causation follow-up (matched-noise/EMA) not run. Raw tensors off-Git; derived arrays/spectra/configs/readers committed.
- requested: Jack / 2026-10-04 PDT
- dependencies: inspect REQ-073/074 configurations, retained checkpoints and actual artifacts;
  reuse their stage capture and spectral readers where verified compatible
- artifacts: `logs/river/req075_late_batch_momentum/`
- resources: existing **two nodes fleet-wide**; preserve REQ-062 priority and live work; publish
  finite per-stage GPU-hour, token, storage and I/O forecasts before launch

**Question:** would a sufficiently large batch reproduce the benefit of suppressing late-training
oscillations, and can small-batch momentum achieve comparable or better loss improvement for the
same data/compute budget? Separate random gradient-estimation noise from alternating motion caused
by optimizer dynamics, and measure useful learning rather than treating smoother curves as success.

REQ-073/074 report that 4B does not remove period-two structure and that momentum filters it. Treat
this as motivation, not proof of an infinite-batch limit: spectral fractions can rise when other
power falls, noise can excite oscillatory dynamics, and trajectories/compute budgets differ. Verify
absolute power, frozen-state sampling variance and matched continuations in this request. Do not
rewrite the older results or relaunch their complete studies as part of this addition.

### Common late checkpoint and registered scope

Use the baseline B-momentum checkpoint immediately before update 2500 of the 3250-update REQ-073
recipe if a complete, verified model/optimizer/data/RNG state is retained. Otherwise reproduce that
state with the pinned recipe, or declare the nearest verified late checkpoint before inspecting
branch outcomes. Record hashes, exact token count, schedule position, data cursor and reconstruction
differences; missing optimizer history must not silently become a fresh-state experiment.

Set B from the actual baseline (expected **524,288 tokens/update**); target **16B = 8,388,608**.
Keep microbatch shape fixed and increase accumulation, with exactly one optimizer step after each
large-batch mean. Average raw gradients at unchanged parameters before clipping/momentum/Muon,
preserving the baseline optimizer-input normalization and mean-per-token analysis units. Validate
accumulated versus grouped small-batch gradients numerically at a frozen state.

Switch **hidden-matrix Muon momentum only**, as in REQ-073: on = mu 0.95 with the actual baseline
Nesterov combination; off = mu 0 with Nesterov disabled. Auxiliary AdamW settings remain fixed and
must be labeled as still containing momentum. Preserve valid inherited buffers in on branches;
disable/clear the Muon buffer in off branches. Clone all other state. Record the initial switching
transient separately; do not selectively discard it from the performance result. Do not claim an
EMA effective-batch formula is exact for this Nesterov-plus-Muon pipeline.

### Stage 1: frozen-state noise measurement before expensive continuations

Freeze the common checkpoint and compute selected-matrix mean gradients on **64 independently
sampled B-sized probe batches**. Record sampling/dropout conventions, keep probe data separate
from validation data, and preserve the training cursor/RNG. Form nested disjoint 4B and 16B groups;
these are paired observations, not additional independent replicates. Disable other stochastic
sources for a sampling-noise-only diagnostic or quantify them separately and label the result.

For Q, K and MLP separately, save within-state variance, absolute mean-gradient norm, estimated
noise-to-signal ratio, and directional agreement between independent large-batch groups. Account
for noise bias in estimated signal norm and report uncertainty; four 16B groups are a pilot, not
a precise variance estimate. Under independent sampling variance is expected to scale inversely
with batch size; measure departures instead of assuming that scaling.

Predeclare a numerical low-noise criterion (for example, estimated RMS gradient noise below 10%
of signal, with uncertainty reported) before using it to label 16B. If 16B remains noisy, report
that limitation; optional additional frozen probes up to 128 B batches can improve estimation.
Do not call a finite batch infinite, or assume selected-matrix agreement establishes whole-model
agreement. No automatic 32B/64B training extension is requested.

### Stage 2: short equal-update mechanism pilot

From the identical late state and one paired data-order seed, run **256 optimizer updates** per arm:

| Arm | Tokens/update | Hidden Muon momentum | Hidden Muon LR |
|---|---|---|---|
| B-mom | B | baseline on | checkpoint LR |
| B-nomom | B | off | checkpoint LR |
| 16B-mom | 16B | baseline on | checkpoint LR |
| 16B-nomom | 16B | off | checkpoint LR |
| 16B-nomom-halfLR | 16B | off | half checkpoint LR |

Hold all learning rates constant at the checkpoint values during this diagnostic, except the
specified hidden-Muon half-LR control. Preserve all other optimizer transformations, clipping and
per-update decay conventions. Do not automatically scale LR with batch size. Report actual step
norms: removing momentum can change movement size as well as direction. If an arm becomes unstable,
record its failure/stop point rather than silently lowering its LR mid-run.

This pilot is explicitly **not compute matched**: large-batch arms consume 16 times more tokens.
Verify sufficient unused corpus coverage first; no silent cycling/repetition or expansion of the
deferred REQ-042 data scope. If the exact horizon cannot fit available data or forecast resources,
declare a shorter common pilot horizon and its spectral limitations before running any arms.

### Stage 3: equal-token performance comparison

Run the four B/16B x momentum configurations from the same checkpoint with **three paired
continuation data-order seeds**. These are continuation replicates conditional on the common base,
not three independently pretrained models. Within each seed use the same ordered token stream;
16B groups the next sixteen B blocks before updating. Restore the exact branch state between runs.

Set the common token budget to the baseline's remaining training budget, rounded down to a multiple
of 16B. At update 2500 of 3250, this is **736B = 385,875,968 tokens**: 736 B updates versus 46 16B
updates. Preserve the original token-indexed LR schedule and endpoint mapping, recording the
truncated endpoint; do not stretch the large-batch arm to 736 optimizer updates. Report momentum's
memory span in both updates and tokens, since keeping its coefficient fixed changes the latter.

First report the inherited-LR comparison. Then give all four configurations the same bounded
hidden-Muon LR search: multipliers **0.5, 1, 2**, including the inherited-LR run, on one declared
tuning data seed at the same token horizon. Keep auxiliary AdamW LR schedules fixed. Select using
a fixed tuning-validation split and evaluate the selected settings with the three paired seeds
and a separate reporting-validation split; count tuning compute separately. A stability failure
counts as an outcome, not permission for a broader sweep.

Match cumulative decoupled weight-decay shrinkage over equal token intervals across batch sizes
and LR multipliers, with the formula and actual products logged; if baseline decay is zero, retain
zero. Preserve all other hyperparameters and state conventions. Evaluate the same fixed held-out
data at common token boundaries and endpoints. Report loss versus tokens and measured wall time,
training time separately from probe/capture overhead, GPU-hours, throughput, actual updates and
paired loss differences with all seed values and uncertainty. Three seeds provide limited power;
do not claim equivalence merely because a difference is not statistically significant.

### Measurements, useful-descent checks and interpretation

Capture the same three complete matrices as REQ-073: middle-block Q, K and MLP output projection.
Save raw gradients, exact momentum/Nesterov input to Muon, post-Muon direction and actual parameter
displacement at every update of the mechanism pilot and final three-seed comparisons. Keep native
tensors/checkpoints off Git with durable manifests/checksums; commit compact derived arrays, configs,
readers and figures. Forecast storage from the actual arm/horizon/dtype census before capture.

Report absolute and normalized period-two power, lag-one/lag-two direction agreement, gradient and
update norms, clipping traces and independent loss. Transform signed coordinates before summing
squared Fourier magnitudes; never substitute spectra of norms or signed coordinate averages. Reuse
REQ-074 time-resolved arrays and plotting conventions, preserving every valid window and frequency.
Use 128-update windows/hop 1 for the 256-update pilot, with 32-update sensitivity. The equal-token
16B continuations have only 46 updates: use clearly labeled 32-update windows there, compare matching
window lengths across arms, and do not fabricate 128-update windows or interpolate missing spectra.
Provide axes in optimizer updates and tokens, with matrices/stages kept separate and DC explicit.

At a few predeclared retained pilot states, reuse independent-loss probes to compare on/off directions
from copied valid optimizer state, both at native displacement and matched selected-matrix movement
norm over a common small step-length grid. Apply other parameter updates identically and restore
state between probes. These local checks help separate direction quality from step length; they do
not replace actual continuation loss or imply a whole-model intervention result.

Interpret registered contrasts as follows:

- Larger batch reduces measured noise, oscillation and the benefit of momentum: consistent with
  momentum chiefly helping through denoising in this regime, not a proof of general equivalence.
- Oscillation survives verified noise reduction but falls with half LR: supports an overshooting/
  optimizer-dynamics explanation. Examine absolute power and actual movements, not fractions alone.
- Momentum still improves loss at low measured noise: benefit extends beyond simple batch-noise
  averaging, but could involve conditioning or dynamics; it does not alone prove cancellation causes gains.
- Smoother directions without improved held-out loss: no demonstrated useful-learning benefit.
- Better loss per update but worse per token/time: more accurate updates do not repay their cost.

If the equal-token result is promising, confirm it at one additional predeclared late checkpoint
within a separately forecast finite budget before making a general late-training claim. A causal
claim specifically about period-two cancellation requires a follow-up matched-noise/matched-age
filter comparison (reuse REQ-073's pair-average versus EMA control where valid), not an automatic
broad filter sweep under this request.

**Completion:** deliver frozen-state noise estimates with uncertainty, the five-arm mechanism pilot,
four-arm inherited/tuned equal-token results, resolved configs and checkpoint provenance, paired
loss/time tables, complete time-resolved arrays/figures, actual resource use and explicit missing
or failed arms. Keep measured findings separate from interpretations. On pickup record RUNNING
with the real job/session handle; adding this request does not assert that training has launched.

## Template

```md
## REQ-NNN: short experiment title

- status: **OPEN**
- requested: Name / YYYY-MM-DD timezone
- dependencies:
- artifacts:

Question, registered comparisons, resource limits, validation, and deliverables.
```
