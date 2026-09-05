# Experiment requests

Active queue for the `jerry-agent` branch. Next request number: **REQ-053**.

The findings are consolidated in [FINDINGS.md](FINDINGS.md). Update that file when evidence changes;
keep this queue for runnable specifications, concise status updates, and result links.

## Run order and status

| Order | Request | Status | Work |
|---|---|---|---|
| 1 | [REQ-050](#req-050-curvature-at-initialisation-and-early-training) | OPEN | Establish when the depth-curvature profile appears. |
| 2 | [REQ-051](#req-051-decompose-why-each-matrix-has-a-different-lr-to-curvature-response) | OPEN | Measure LR-response components across four seeds and six matrix LR levels. |
| With 051 | [REQ-052](#req-052-matched-uniform-versus-mixed-lr-controls-for-req-051) | OPEN | Compare mixed, uniform-Muon, and full-global LR using the same bases. |

These are queue states; no GPU execution handle has been supplied. Do not interrupt running work.
Run the REQ-051 pilot first, then coordinate REQ-052 while each base checkpoint is available.

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
- Use the interpretation and validation safeguards in REQ-051 and [FINDINGS.md](FINDINGS.md).
- Append new numbered requests; update existing status in place. Do not prepend iteration diaries
  or duplicate historical status tables.

## Deferred work

| Request | State | Disposition |
|---|---|---|
| REQ-049 | OPTIONAL | Four-seed replication of the crossed per-matrix LR test; does not displace 050–052. Original specification remains in the history linked below. |
| REQ-042 | BLOCKED | 32×/64× batch runs exceed the available corpus. Requires a data or run-length decision; no looping/repetition is authorized by this cleanup. |
| REQ-035 B/C/D | NOT RUN | Arm A is complete. Preserve as deferred work; do not dispatch automatically. |

Completed REQ-034, REQ-035 Arm A, REQ-036, REQ-037 arms 1–3, REQ-038/041, and REQ-043–048
are summarized with result links in FINDINGS.md. REQ-037 arm 4 was delivered as REQ-046.
The old REQ-040 dispatch order names completed work and is superseded by the active run order above.

[Full pre-cleanup queue and analysis history](https://github.com/jacknzheng/kmaxwell-sota/blob/28d00746aa80d71caf1fb8cb38b2e336b4c5d2d9/requests.md)
preserves the original requests, criteria, deferred specifications, and retracted findings.
No experiment results or historical files have been deleted.

## REQ-050: curvature at initialisation and early training

- status: **OPEN**
- requested: 2026-09-04; cost premise corrected in iteration 193
- priority: first; at most two nodes fleet-wide
- question: Is the depth-curvature profile visible at initialization, or does it emerge during early training?

Use the existing `measure_per_matrix_curvature.py` probe at **steps 0, 125, 250, 500, 1000,
and 1500**. Start with one seed; expand to four independent seeds if the measured cost permits,
as in the original request. Each run from step 0 passes through all six measurement points.
Earlier checkpoints are not retained: regenerate the training state rather than assuming a probe-only run.

The recorded training-time estimate is about **4 minutes per seed / 16.2 minutes for four seeds**
at 0.162 seconds per step. This excludes probing and setup; benchmark and report those separately.

Preserved registered criteria (band 55):

1. **INHERITED:** at step 0, cubic profile R² ≥ 0.70, minimum in blocks 4–8, and correlation
   ≥ +0.70 with the late step-2750 profile.
2. **LEARNED-EARLY:** step 0 has cubic R² < 0.30 or an edge minimum, but the profile is present
   by steps 500–1500.

Keep the raw profiles and report cases meeting neither criterion as inconclusive; the original
claim that ambiguity was impossible was narrative, not a third numerical criterion.
Include the exact measurement steps, per-seed evidence, initialization provenance, code, and
result location when updating this request. Observing early structure alone does not prove its cause.

**Probe-repeat requirement, added iteration 224.** REQ-048's twelve files were found to be 4 seeds x
3 **probe repeats** of a single checkpoint (all report `step = 2750`), not three training steps.
That accident made probe reliability estimable, and the numbers matter for this request's design:
single-probe `log lam` carries a within-matrix sd of **0.349** against a between-matrix sd of
**0.347**, i.e. **40.5% of its variance is measurement noise**. `log n_eff` is far cleaner
(within 0.181, between 0.556).

Therefore: run **at least 3 probe repeats at every measurement step**, with different probe seeds,
and commit each repeat separately rather than pre-averaging. Rationale:

- Criterion 1 tests a **cubic R² >= 0.70** and a **correlation >= +0.70** with the late profile.
  Both are attenuated by probe noise. With single probes on `lam`, reliability is ~0.75 per
  measurement, so a true correlation of 0.80 would be observed near 0.60 and would **fail a
  criterion it should pass**. Three repeats raise reliability to ~0.90 and make the thresholds
  mean what they were written to mean.
- Committing repeats separately preserves the ability to estimate reliability at each step and to
  disattenuate; pre-averaging discards it permanently.
- Cost is probe time only, not training time; the 16.2-minute training estimate is unchanged. Report
  measured probe cost per repeat so the repeat count can be revisited.

If probe cost makes 3 repeats at all six steps infeasible under the two-node ceiling, prefer
**3 repeats at steps 0 and 1500** (the two the criteria actually compare) over one repeat everywhere.

**Judge the registered criteria within seeds, added iteration 226.** Criterion 1 asks for a cubic
R² and a correlation with the late profile; do **not** additionally require cross-seed sign or
threshold agreement as corroboration. In this design the `log C` residual is **76% shared across
seeds** (mean off-diagonal correlation +0.758 after type and block are absorbed), so four seeds are
close to one structural test repeated four times. A pure-noise structural predictor achieves 4/4
sign agreement **58%** of the time.

Four seeds are still the right design -- they test robustness to **initialisation**, which is what
criterion 1 needs. But report each seed's result on its own matrices, with standard errors clustered
by block, and treat a criterion as met on **within-seed effect size**. For reference, the standing
concentration result under exactly this treatment gives t = -9.4 to -17.0 with partial R² 0.34-0.56
per seed; that is the standard of evidence a new claim should be held to.

### Pre-registered hypothesis H1, added iteration 227 -- test on REQ-050/051 data, not on REQ-048

Iteration 227 found that a **type-by-depth interaction** accounts for ~31% of the reproducible
residual in `log C` after type, block and `log n_eff` are absorbed. Per-type depth slopes of that
residual (dex per block, REQ-048, 4 seeds):

| `mlp.proj` | `attn.q` | `mlp.fc` | `attn.k` | `attn.proj` | `attn.v` |
|---|---|---|---|---|---|
| **+0.0298** | +0.0127 | +0.0111 | −0.0051 | −0.0212 | **−0.0274** |

The grouping that separates this ordering cleanly is **the attention output path (`attn.v`,
`attn.proj`) negative vs all others positive-or-zero**. That grouping was read off these slopes, so
it is a hypothesis generated by the data and **must not** be scored on REQ-048.

**Register these predictions before the new runs, and report them whether they pass or fail:**

1. **H1-sign:** in new data, `attn.v` and `attn.proj` have negative residual-vs-depth slopes and
   `mlp.proj` has a positive one, judged **within each seed** (rule 32), not by cross-seed sign
   counting.
2. **H1-size:** the per-type slope spread is at least 0.03 dex/block (observed 0.057; half is a
   conservative floor).
3. **H1-share:** per-type depth slopes recover an R² gain of at least 0.15 on the residual within
   each seed (observed 0.288-0.468).
4. **Falsifier:** if the standing **writer-vs-internal** split (`attn.proj` + `mlp.proj` together)
   separates the new slopes better than the attention-output-path grouping, H1 is wrong. On REQ-048
   it fails outright -- the two writers sit at opposite extremes.

REQ-050 tests H1 across training steps (does the interaction appear with the bowl, or later?).
REQ-051 tests it causally, since it varies per-type LR directly. Neither requires more than the
two-node ceiling already in force.

**Clarification from iteration 228 -- expect the RAW slopes to look null.** The type-by-depth effect
is **suppressed** in `log C`: each type's direct curvature drift with depth is largely offset by its
concentration drifting the other way. Across the six types, sd(direct) = 0.0219 but
sd(total) = 0.0098, with corr(via `n_eff`, direct) = **−0.920**. Without a concentration control the
R² gain is only 0.065; with one it is 0.44–0.50.

Therefore, when scoring H1:

- Evaluate the slope criteria on residuals **after** controlling concentration, exactly as H1 states.
  A near-null raw `log C` slope is **consistent with H1 being true** and must not be reported as a
  refutation.
- Report the decomposition `total = beta x (slope of log n_eff) + direct` per type as a
  **description**, and publish all three columns. But do **not** score cancellation from a
  correlation between `via` and `direct`, or from the identity closing: `direct` is defined as
  `total - via`, so that correlation is mechanically negative (mean −0.81 even for independent
  components) and the identity holds for any data. Iteration 228 made this error and it is retracted.
- Score cancellation with a **pairing permutation** instead: hold each type's curvature-depth slope
  and each type's concentration-depth slope fixed, permute which concentration profile pairs with
  which type, and compare the observed sd(total) to that null. On REQ-048 this gives p = 0.004-0.034
  across seeds, with breaking the pairing roughly doubling the spread.
- A `lam`-free control (`n_eff_bulk = trace(H)²/(trace(H²) − lam²)`, or `participation_ratio`) gives
  the same answer on REQ-048 (0.437 / 0.460 vs 0.446) and is preferred where available, since it
  shares no term with the outcome.

## REQ-051: decompose why each matrix has a different LR-to-curvature response

- status: **OPEN**
- **primary quantitative target, added iteration 230:** measure the **causal** elasticity
  `k = d(log lam)/d(log g)`. Observationally, on REQ-048 with type and block absorbed, `k = 3.173`
  (per seed 3.28/3.27/2.99/3.15, cluster-robust by block, t vs 2 = +17.1), and `k = 2.569` with a
  lam-free concentration control (t vs 2 = +4.88). **k = 2 is the gauge-invariant value** — a scalar
  rescaling a matrix's whole contribution moves `lam` by c² and `g` by c, leaving `C = lam/g²`
  untouched. So observationally the between-matrix variation is **not** a pure gauge rescaling, and
  `d(log C)/d(log g) = +1.17` (+0.57 with the concentration control), worth 0.17–0.21 dex of C across
  ±1 sd of residual `log g`, partial R² 0.30–0.39.
  **The confound this cannot resolve:** `g` and `lam` are measured at the same step and both respond
  to the same training dynamics, so the observational k may be confounded. REQ-051 varies per-matrix
  LR causally and is the only instrument in the queue that can separate them. Report the causal k
  with its distance from 2, per seed, using within-seed effect sizes (rule 32).
- **second target, added iteration 231 — the moment ladder.** Under the reparametrisation gauge
  (`W = c·V` gives `g ~ c`, every Hessian moment `~ c²`), **all** spectral moments must have
  elasticity **+2** wrt `log g`. Observationally on REQ-048 they do not, and they deviate in a strict
  order: `trace(H)` **+1.107**, `sqrt(trace(H²))` **+2.424**, `lam_top` **+3.173**. Within-seed
  differences are `lam − trace` +2.066 (t = +26.3), `lam − rms` +0.749 (t = +13.3), `rms − trace`
  +1.317 (t = +19.7), all 4/4 seeds; a permutation null on `log g` gives p = 0.0000. Physically:
  **matrices with larger gradients hold their curvature in fewer directions.**
  Report the same three elasticities under **causal** LR variation. Two outcomes are informative:
  the ladder survives (concentration genuinely responds to the gradient side), or it collapses toward
  +2 for all three moments (the observational ladder was confounded by joint response to training
  dynamics). Report whichever occurs, per seed, with within-seed effect sizes.
- requested: Jack / Codex, 2026-09-05 PDT
- priority: **high, after the already-open REQ-050; do not interrupt work already running**
- repo: `https://github.com/jacknzheng/kmaxwell-sota`, branch `jerry-agent`
- implementation base: use the committed per-matrix-LR machinery from REQ-023/045 and the
  activation/backward probes from REQ-043/047; record the exact final code SHA
- resource constraint: **at most 2 nodes total**

### Question

REQ-023 and REQ-045 establish that a matrix's own LR changes its equilibrium top curvature with a
pooled elasticity near `-1.16`, while the separately identified neighbour-LR coefficient is null.
REQ-043/047 explain the q/k weight-gradient deficit as a backward-magnitude plus token-alignment
effect. What remains unanswered is **why the own-LR curvature elasticity differs across matrices,
types, and blocks**.

Use the repository's current notation throughout:

- `lambda` = per-matrix `top_eigenvalue`;
- `g` = raw same-minibatch weight-gradient Frobenius norm;
- `C_gauge = lambda / g^2` — call this `C_gauge`, not merely `C`, to avoid confusing it with the
  older power-law intercept;
- `rho = g / (||a||_F ||d||_F)` = REQ-043's `align_ratio`;
- for any positive quantity `x`, `k_x = -d log(x) / d log(own LR multiplier)`.

Two decompositions must be evaluated on **the same state and the same minibatch**:

```text
k_lambda = 2*k_g + k_C_gauge
k_g      = k_a + k_d + k_rho

therefore:
k_lambda = 2*(k_a + k_d + k_rho) + k_C_gauge
```

The first follows from `C_gauge = lambda/g^2`. The second follows exactly from
`g = ||a||_F ||d||_F rho` for the hooked bias-free Linear. These are accounting identities, not by
themselves causal mechanisms; their value is that they reveal **which measured component carries the
between-matrix variation in LR response**.

### Design: six-level, within-matrix LR curves on four independent bases

Train **four genuinely independent seeds** to a serialized fork at step 2000. From each base, run six
750-step continuation arms to step 2750. Use per-matrix multipliers

```text
{0.50, 0.65, 0.85, 1.00, 1.30, 1.70}
```

with a cyclic Latin assignment across arms:

- every one of the 72 Muon matrices receives every multiplier exactly once across the six arms;
- within each arm, each multiplier is assigned to exactly 12 matrices;
- stratify within matrix type: each of the six types contributes exactly two matrices to every
  multiplier in every arm;
- redraw/rotate the block-to-level mapping independently by seed;
- keep the full-network multiplier histogram identical in every arm, so the experiment changes a
  matrix's own LR without changing the network-wide LR distribution;
- write the effective LR multiplier explicitly into every output row.

Within each seed, all six arms must load the exact same serialized model, optimizer, scheduler, and
data cursor, then consume the same post-fork minibatch sequence; only the per-matrix LR assignment may
differ. Record and verify the base-state hash. Across seeds, the four base hashes must be distinct.

REQ-045 has already identified the neighbour channel as null. Still report each matrix's others-mean
multiplier and the mechanical own/others correlation; do not reinterpret this balanced ladder as a new
neighbour-effect test.

Save checkpoints at **2050 and 2750**. Step 2050 is a mandatory early-response measurement, not a
conditional branch. Step 2750 is the equilibrium endpoint. Full checkpoints may be retained locally
for the probes but must not be committed.

### Measurements

At both 2050 and 2750, on one fixed, recorded validation minibatch shared across all arms within a
seed, run a combined same-state probe that records per matrix:

- `a_frob`, `a_rms`, `a_eff_rank`;
- `d_frob`, `d_rms`, `d_eff_rank`;
- raw `grad_frob` and `align_ratio`;
- REQ-047's `d_token_participation`, `da_cos_mean`, and `grad_rank1_frac`;
- `weight_frob`;
- momentum/polar-input norm, post-polar update Frobenius norm and spectral norm;
- realized relative update `effective_lr * ||polar_update||_F / ||W||_F`.

At 2750 also record, from the same loss scaling and token batch:

- `top_eigenvalue`;
- `gradient_block_norm`, plus an explicit equality/scale check against `grad_frob`;
- `curvature_along_polar`;
- the usual Lanczos convergence diagnostics.

Do not add another uniform pre-polar gradient multiplier: REQ-046 already proves that Muon's polar map
normalizes that intervention away. Do not mix REQ-043 gradients from one minibatch with curvature from
another; if the curvature code and hook code cannot share one pass, record the exact normalization
factor and require the per-matrix `gradient_block_norm`/`grad_frob` ratio to be constant across arms.

### Analyses and registered decisions

1. **Own-LR inverse law.** Fit each matrix's six-point `log lambda ~ log multiplier` curve separately,
   then summarize by seed, type, and block. The inverse-law hypothesis passes if, in every seed,
   at least 90% of matrices have `k_lambda > 0` and the seed-median `k_lambda` lies in `[0.9, 1.5]`.
   Report disagreement rather than pooling it away.

2. **Power law versus saturation.** For each matrix compare
   `lambda=A*m^(-k)` against `lambda=lambda_floor+A*m^(-k)` using leave-one-LR-level-out prediction.
   Call saturation supported only if the positive-floor model reduces held-out RMSE by at least 15%
   in at least three of four seeds. Otherwise retain the simpler power law. Report which types/blocks,
   if any, support a floor.

3. **Exact response accounting.** Require maximum absolute residual below `1e-5` for both
   `k_lambda-(2*k_g+k_C_gauge)` and `k_g-(k_a+k_d+k_rho)`, after documenting log base and normalization.
   A larger residual is a probe mismatch and must be fixed before interpretation.

4. **Gauge-restoration hypothesis.** The prior prediction is that the LR response is carried almost
   entirely by `g`, with `C_gauge` nearly restored. It passes if each seed has
   `|mean(k_C_gauge)| < 0.15` and the pooled magnitude of `k_C_gauge` is less than 15% of
   `k_lambda`. Failure means the small pooled REQ-023/045 result hides systematic matrix-level
   heterogeneity and must be revised.

5. **Which part of the gradient response differs by layer?** Within each seed, remove matrix-type
   means and report the variance/covariance decomposition of `k_g = k_a+k_d+k_rho` across blocks.
   Also report type-specific values. The registered q/k/v prediction is that `attn.v` has larger
   `k_lambda` than the mean of q/k in at least three of four seeds. Do not call the largest component
   causal; label it the component carrying the response variation.

6. **Early prediction of final sharpness response.** Predict the step-2750 `k_lambda` using only
   step-2050 response features (`k_a`, `k_d`, `k_rho`, early `k_g`, effective relative update,
   effective-rank and token-coherence responses), with one entire seed held out. Compare against the
   existing matrix-type prior (`R^2=0.217`, RMSE `0.427`). The early-response hypothesis passes if
   held-out-seed `R^2 >= 0.32` and RMSE `<= 0.40`; otherwise conclude that the first 50 steps do not
   predict equilibrium reaction well enough.

7. **Depth discipline.** Fit depth only within type and seed. Compare linear depth, free block effects,
   and type-by-depth models under held-out-seed evaluation. No pooled “deep layers react more” claim is
   allowed unless its sign holds in all six types and at least three of four seeds.

### Required artifacts

Write code, configs, logs, and results to:

`logs/kmaxwell/req051_lr_elasticity_decomposition/`

Include:

- `README.md` with design, provenance, results, caveats, and explicit pass/fail for all seven decisions;
- `assignments.json` with seed/arm/matrix/type/block/multiplier and balance checks;
- `manifest.tsv` with independent-base state hashes, checkpoint steps, data cursor, node, config, SHA,
  and exit status;
- `per_matrix_measurements.tsv` for the same-state 2050/2750 fields;
- `elasticities.tsv` containing every per-matrix `k_lambda`, `k_g`, `k_C_gauge`, `k_a`, `k_d`, and
  `k_rho` plus identity residuals;
- `model_comparison.tsv` for power versus saturation and the held-out-seed early predictor;
- raw JSONs, config generator, combined probe, and analysis script;
- figures showing all 12 blocks together, unified by metric: per-type/per-seed `k_lambda`, the
  `k_a/k_d/k_rho/k_C_gauge` decomposition, six-level LR curves with held-out predictions, and early
  predicted versus final `k_lambda`.

Commit no model weights, optimizer tensors, secrets, or full checkpoints. Preserve disagreement across
seeds and matrices in the tables; do not report only pooled averages.

### Execution and interpretation addendum — 2026-09-05, before execution

Jack has requested continued execution and monitoring every 20 minutes until the LR-response question
is resolved. Please acknowledge this request when picked up. On starting REQ-050/051, record the live
scheduler job ID or host/session/process handle, start time in UTC, current seed/arm/step, and a concise
log location. On completion or failure, record the terminal exit state. A queue status alone is not
proof that a process is running. Preserve the fleet-wide two-node ceiling and existing running work.

These clarifications are part of REQ-051 and do not add training arms:

- **Scientific scope.** The balanced ladder estimates own-LR response under reassignment of the other
  matrices' LRs. A fixed multiplier histogram does not hold the other weights or trajectories fixed.
  REQ-045 failed to detect its particular aggregate neighbour channel; it did not prove that every
  cross-layer effect is zero. Carry that limitation into the per-matrix response interpretation.
- **No theorem about LR invariance.** `C_gauge=lambda/g^2` is a defined ratio. Its small response is an
  empirical hypothesis; changing LR is not a parameter-coordinate transformation. Uniformly scaling
  the loss also scales both its Hessian and gradient linearly, so it does not leave this ratio
  unchanged. Do not label a restored ratio a proof of general gauge invariance or gradient mediation.
- **Shared operator and normalization.** Record whether curvature is the true block Hessian, GGN,
  or another operator; record loss reduction, token count, batch hash, dtype, and probe seed. Do not
  log-transform nonpositive eigenvalues or signed directional curvature silently. Record exclusions.
  Separate actual training-update telemetry from a hypothetical polar update recomputed on eval data.
  Hooking the eval probe must not mutate the training optimizer or momentum state.
- **Accounting checks are unit checks.** Fit every component with identical observations, regressors,
  weights, and masks. Since `rho` and `C_gauge` are defined using `g` and `lambda`, the two slope
  identities are algebraically guaranteed. Independently verify `sum_t d_t a_t^T` against the autograd
  gradient before computing `rho`; passing the slope identities alone cannot verify hook correctness.
- **Mean restoration versus matrix restoration.** Retain decision 4 as the *pooled* test, but additionally
  report the distribution of `k_C_gauge`, its RMS, type means, and seed consistency at every block.
  Large positive and negative responses may cancel. A small pooled coefficient establishes only a
  small mean response. Report uncertain heterogeneous effects without declaring them absent.
- **Prediction comparison on the new data.** The old `R2=0.217`/`RMSE=0.427` came from REQ-035's global
  three-level LR ladder and are historical context. Refit mean-only, type-only, and type-plus-block
  baselines using exactly the new training folds. Require the early-feature predictor to reduce RMSE
  by at least 10% relative to the strongest of these baselines, with improvement in at least three
  held-out seeds; also report the originally specified absolute thresholds. Any regularization or
  feature selection must use training seeds only. Separate baseline-only prediction from prediction
  after observing the six early treatment responses; the latter requires calibration interventions.
  Use separate fixed eval minibatches for early features and final outcomes to reduce shared probe
  noise; all fields within an identity must still use the same minibatch and state.
- **Dependence and uncertainty.** Four seeds are four independent network replicates, not 288.
  Report four held-out errors, four per-seed effects, and probe/fit uncertainty. Do not manufacture
  narrow confidence intervals by treating blocks, arms, or checkpoints as independent networks.
  For saturation compare error in log-lambda, constrain floor >= 0, A > 0, k >= 0, and report
  boundary/failed fits. Six points may not identify a floor; an inconclusive outcome is allowed.
- **Budget and stopping gate.** This is 4 regenerated bases plus 24 continuation arms, 48 inexpensive
  signal-probe states and 24 endpoint curvature-probe states. Benchmark one base/arm/probe first,
  report training and probing time separately, and repair any operator/hook/state mismatch before
  launching the remaining arms. Do not assume endpoint probing is cheap because training is short.
  Call step 2750 the planned endpoint; establish stationarity from available late training/probe
  diagnostics before calling it equilibrium. If those diagnostics are insufficient, flag that
  limitation and propose a bounded follow-up instead of extrapolating a time course from one point.

## REQ-052: matched uniform-versus-mixed LR controls for REQ-051

- status: **OPEN**
- requested: Jack, 2026-09-05 PDT, continuing the LR/sharpness experiment goal
- priority: coordinate with REQ-051 while its four base checkpoints are live; REQ-050 and already
  running work retain priority. Do not interrupt or duplicate a running job.
- branch: `jerry-agent`; resource ceiling: **two nodes fleet-wide**
- incremental scope: **five control arms per seed, four seeds, 20 continuations total**;
  reuse the four serialized REQ-051 step-2000 bases and probe implementation

### Evidence motivating the test

The newly proposed writer/internal sensitivity difference (band 67) holds in REQ-023, but does not
transfer to REQ-035's global LR experiment. An independent raw-data audit of the two REQ-023 forks
and all four REQ-035 seed archives gives the following. Writers are `attn.proj` and `mlp.proj`;
internal matrices are q, k, v and mlp.fc. Each group has equal representation at all 12 blocks.

| design / state | mean k(writers) minus mean k(internal) | mean k(v) minus equal-type mean k(q,k) |
|---|---:|---:|
| REQ-023 mixed LR, fork1500, endpoint2250 | +0.924 | -0.085 |
| REQ-023 mixed LR, fork2000, endpoint2750 | +1.165 | -0.500 |
| REQ-035 global LR, seed0, endpoint2250 | -0.194 | +0.367 |
| REQ-035 global LR, seed1, endpoint2250 | -0.125 | +0.400 |
| REQ-035 global LR, seed2, endpoint2250 | -0.006 | +0.455 |
| REQ-035 global LR, seed3, endpoint2250 | -0.199 | +0.356 |

Method: for each matrix at the stated checkpoint, fit `k=-OLS_slope(log lambda,log multiplier)`
over 0.6/1.0/1.7 using assignments.tsv for REQ-023 and the LR tag for REQ-035; then average within
the prespecified groups. Averaging the REQ-035 late 2250-2750 lambda values first also gives negative
writer/internal contrasts in all four seeds (-0.277,-0.163,-0.157,-0.162). REQ-023's window-mean
log-lambda contrasts are positive at both forks (+0.758,+0.899). Thus a simple endpoint mismatch
does not resolve the discrepancy. Different code, initialization, optimizer treatment scope, and
training protocols remain possible explanations; this comparison alone does not establish causality.

Five checkpoints from one REQ-023 continuation are dependent measurements, not five independent seeds.
The four-seed criterion in band 67 is untested by that experiment. Preserve the original claim as a
hypothesis rather than marking its seed-replication criterion satisfied.

### Exact added arms and controls

For every REQ-051 seed, load its exact step-2000 model/optimizer/scheduler/data-cursor state and run:

1. all 72 Muon matrix multipliers = **0.65**;
2. all 72 Muon matrix multipliers = **1.00**;
3. all 72 Muon matrix multipliers = **1.70**.

Use the **same per_matrix_lr_muon implementation** as REQ-051, with all 72 entries equal in these
three arms. Keep embedding/output/other AdamW learning rates fixed for the Muon-only arms.

**Runtime audit resolved the historical scope question before launch (2026-09-05).** REQ-019's
`eos_f1500_s{060,100,170}/train-log.txt`, `learning_rates step:1500`, records:

| multiplier | embedding AdamW | output AdamW | Muon | other AdamW |
|---|---:|---:|---:|---:|
| 0.60 | 0.42 | 0.0024 | 0.015 | 0.009 |
| 1.00 | 0.70 | 0.0040 | 0.025 | 0.015 |
| 1.70 | 1.19 | 0.0068 | 0.0425 | 0.0255 |

REQ-023's runtime traces keep the AdamW entries at 0.70/0.004/0.015 across assignments. The
REQ-035 README describes reuse of the REQ-019 ladder, but does not commit its runtime LR traces;
the directly verified all-optimizer scope is REQ-019. Its writer/internal contrasts are also
negative at both matched endpoints: -0.345 (fork1500,2250) and -0.330 (fork2000,2750), while
v-minus-qk is +0.325 and +0.317. Thus the same ranking reversal is present in a global dataset
whose actual LR treatment is directly logged.

To separate uniform Muon scope from non-Muon LR changes, add just **two further arms per seed**:

4. **full_global065:** all 72 Muon multipliers = 0.65 AND all non-Muon optimizer group LRs = 0.65x
   their reference values;
5. **full_global170:** all 72 Muon multipliers = 1.70 AND all non-Muon optimizer group LRs = 1.70x
   their reference values.

Reuse arm 2 (all multipliers 1.00, reference non-Muon LRs) as the shared full-global 1.00 control.
Do not duplicate it. In full-global arms apply the Muon scale exactly once: set per-matrix entries
to s with the base Muon LR unchanged, and separately scale non-Muon groups. Verify actual traces
after restoring optimizer state and applying hooks, since loading state can overwrite YAML LRs.

All arms keep weight-decay coefficients, kernel/momentum, batch, data order, and stop step fixed.
Changing LR also changes realized decoupled weight decay; log its update separately from the
gradient-driven update and do not describe the result as isolating those two channels.
Record actual per-group runtime LR traces and full base-state hashes.

Measure early features at **2050** and endpoint curvature plus signal features at **2750**, with the
same REQ-051 operator, normalization, minibatch pairing, checkpoint semantics, and no-state-mutation
checks. The all-1 arm supplies a common reference trajectory. Training is 20x750 extra steps; the
40 signal-probe states and 20 endpoint curvature-probe states must be costed separately using the
REQ-051 pilot. Do not omit the probe cost or assume old checkpoints exist.

Execution order: once a seed's base is generated, schedule its six mixed arms and five control arms
within the fleet ceiling before releasing that base. REQ-051's measured pilot must pass first. If
REQ-051 has already released its bases when picked up, record that condition and the full regeneration
cost before proceeding; do not silently treat these as free probe-only runs.

### Registered analysis

Primary comparisons use **the same three LR levels** in each design: 0.65, 1.00, 1.70. For each
matrix select its corresponding three REQ-051 mixed-arm observations, fit k_mixed, and fit k_uniform
from the three Muon-only controls. Also fit k_full_global from full_global065, shared all-1,
and full_global170. The six-point REQ-051 fits remain secondary for this comparison.

Report per seed and block:

- `D_m = k_uniform,m - k_mixed,m`;
- writer/internal contrast in each design and their paired difference;
- v-versus-q/k contrast in each design and their paired difference;
- corresponding k_g, k_C_gauge, k_a, k_d and k_rho changes, using the same identity safeguards.

Repeat these contrasts for full-global minus uniform-Muon and full-global minus mixed. The
full-global-minus-uniform-Muon contrast holds each Muon LR fixed at a given level and changes
only the non-Muon LR group. Its effect identifies sensitivity to those accompanying LR changes
in this matched setup; it does not identify which individual non-Muon parameter caused it.
Apply the same practical threshold/seed-sign criteria below separately to each contrast and
report all three comparisons. The direction of the intermediate uniform-Muon condition is open.
The historical directional prediction primarily concerns **full-global versus mixed** scope.

The candidate prediction is that the writer/internal contrast is **larger in mixed than uniform**
LR, and the v-versus-q/k contrast is **larger in uniform than mixed** LR. Register support for a
contrast only if its paired difference has the predicted sign in at least 3/4 seeds and the
four-seed mean magnitude exceeds **0.20 in k**. Report each seed, the difference magnitude, and
uncertainty; this threshold is a practical effect criterion, not a manufactured significance test.
If signs disagree or uncertainty is broad, report INCONCLUSIVE. If either contrast reverses with
similar consistency and magnitude, report the directional prediction refuted.

Fit a pooled predictor of k from type/block and a second with three-way design-by-type terms. Evaluate on a
whole held-out seed using training-only tuning. Require at least **10% lower pooled held-out RMSE**
and improvement in at least **3/4 held-out seeds** to call intervention scope predictively useful.
This is separate from testing whether raw differences are statistically distinguishable from zero.

Interpretation: a reproducible matched difference means response depends on which other matrices
are perturbed, or on the resulting collective state, despite an earlier null aggregate neighbour
coefficient. It does not identify a particular off-diagonal Hessian block or prove that one layer's
curvature itself causes another's. If the difference disappears in this matched implementation,
the old writer/global mismatch is protocol-dependent; retire it as evidence for a network mechanism.
The current power law may remain useful in either outcome, but k must be indexed by intervention
scope if the difference persists.

If the reversal appears only in full-global and not uniform-Muon, attribute the matched contrast to
the accompanying non-Muon LR intervention, not to an established Muon-neighbour curvature channel.

### Deliverables

Commit under `logs/kmaxwell/req052_matched_lr_scope/`: configurations/generator, base/runtime-LR
manifest, raw JSON, per-matrix paired response table, per-seed contrast table, reproducible analysis,
held-out prediction table, and README with explicit supported/refuted/inconclusive outcomes.
Include one figure per metric with all 12 blocks, uniform/mixed curves distinguished consistently,
and the same matrix-type columns. Commit no checkpoints, model/optimizer tensors, or secrets.

## Template

```md
## REQ-NNN: short title

- status: OPEN
- requested: name / date
- priority and dependencies:
- resource limit:

Question and hypothesis.
Exact treatment, controls, seed count, checkpoints, and measurements.
Registered decisions, uncertainty, failure/inconclusive conditions.
Provenance, commands/configuration, cost estimate, and artifact paths.
```
