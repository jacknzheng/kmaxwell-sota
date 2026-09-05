# Learning rate and sharpness: consolidated findings

Updated 2026-09-05. This is the campaign's findings summary; [requests.md](requests.md) contains
the active experiments. Historical discussions and retractions remain in
[the pre-cleanup record](https://github.com/jacknzheng/kmaxwell-sota/blob/28d00746aa80d71caf1fb8cb38b2e336b4c5d2d9/requests.md).

## What we can currently explain

Increasing a matrix's own learning rate generally lowers its measured sharpness.
The size of that response varies by matrix type and experimental setup. We have useful
predictive relationships, but no validated universal formula explaining all layer differences.

There are **12 transformer blocks and six matrix types per block**: q, k, v, attention
projection, MLP expansion, and MLP projection. A matrix's top Hessian eigenvalue measures its
sharpest local direction. This differs from curvature along the optimizer's actual update.

## Formula and notation

For matrix m, a useful empirical starting point is:

```text
lambda_m(new) / lambda_m(old) ≈ (eta_m(new) / eta_m(old))^(-k_m)
```

Here eta is the actual effective LR and k is its response exponent: larger positive k means
a stronger sharpness reduction. REQ-045 estimates a pooled k of **1.161**, reported SE **0.091**.
At that fitted value, a 40% LR increase predicts approximately 32% lower sharpness.

Keep two previously conflated constants separate:

- **Power-law intercept:** K_m in lambda_m(s) = K_m s^(-k_m).
- **Gradient-normalized ratio:** R_m = lambda_m / g_m², called C_gauge in recent requests.

Neither the ratio's definition nor an exponent near one proves a stability law for Muon.
Its nonlinear polar normalization and momentum differ from ordinary gradient descent.

## Evidence that survives the audits

| Finding | Evidence | What it establishes |
|---|---|---|
| Higher own LR lowers sharpness | REQ-023 reports own-LR slopes about −1.29/−1.38 at two forks; REQ-045's crossed fit gives −1.161. | A reproducible average response in the tested regimes. |
| No detected aggregate neighbor-LR response in REQ-045 | Neighbor coefficient +0.143, reported SE 0.126. | That particular aggregate effect was not detected; individual or type-dependent interactions remain possible. |
| q/k receive weaker backward signals than v | REQ-043: q/k-to-v backward RMS ratio 0.667 ± 0.011 across four seeds; q/k/v share the same input activations. | The measured deficit arises on the backward side; softmax attenuation is a plausible contributor. |
| Token alignment contributes to the gradient deficit | REQ-043: additional alignment factor about 0.646; combined gradient deficit roughly 0.43–0.45×. REQ-047 observes greater token-to-token coherence in v. | Gradient magnitude reflects both signal size and how token contributions reinforce or cancel. |
| Uniform gradient rescaling is an ineffective equilibrium intervention under Muon | Corrected REQ-046: clipped norm slope +1.003; raw norm +0.0028, interval [−0.0143,+0.0200]; sharpness +0.0089. | The experiment does not identify whether raw gradient magnitude causes curvature. Polar normalization removes the persistent scale change; transients can still occur. |
| Curvature has reproducible depth structure | REQ-048 and subsequent within-panel analyses associate boundary sharpness with greater spectral concentration than the middle. | A structural association, not the cause of the depth profile or a complete explanation of LR sensitivity. |
| Equalizing per-type curvature did not improve loss | REQ-036 reduced curvature spread from 0.246 to 0.128; that arm worsened validation loss by about 0.024. | Lower curvature spread was not a successful objective in that experiment. |

Spectral concentration was measured using fresh random Hessian-vector probes, separately from
the Lanczos top eigenvalue. The proxy trace(H)²/trace(H²) requires care for an indefinite Hessian:
positive and negative eigenvalues can cancel in the trace. It is not automatically a literal
count of positive curvature directions.

Sources: [REQ-023](logs/kmaxwell/req023_per_matrix_lr/README.md),
[REQ-043](logs/kmaxwell/req043_seeds_probe/README.md),
[REQ-045](logs/kmaxwell/req045_crossed_global_permatrix_lr/README.md),
[REQ-046 correction](logs/kmaxwell/req046_permatrix_clip_instrument/README.md),
[REQ-047](logs/kmaxwell/req047_pertoken_backward/README.md),
[REQ-048](logs/kmaxwell/req048_spectral_participation/README.md),
[REQ-036](logs/kmaxwell/req036_equalized_curvature_lr/README.md).

## Why the layer-response formula is still unresolved

Baseline gradient and signal measurements predict the curvature level better than its LR response.
An exploratory model trained on three REQ-035 seeds and tested on the fourth improved prediction
of log K from R² **0.279 to 0.721** when probe features were added to matrix type. Prediction of k
improved only from **0.217 to 0.278** at best. These are exploratory results on one architecture,
not causal validation or generalization to new architectures.

More importantly, the ordering of sensitivities changes across interventions. “Writers” below
means attention projection and MLP projection; “internal” means q, k, v and MLP expansion.

| Experiment / endpoint | Mean k(writers) − mean k(internal) | k(v) − mean(k(q), k(k)) |
|---|---:|---:|
| REQ-023 mixed per-matrix LR, fork1500 / step2250 | +0.924 | −0.085 |
| REQ-023 mixed per-matrix LR, fork2000 / step2750 | +1.165 | −0.500 |
| REQ-019 global LR, fork1500 / step2250 | −0.345 | +0.325 |
| REQ-019 global LR, fork2000 / step2750 | −0.330 | +0.317 |
| REQ-035 global LR, four seeds / step2250 | −0.199 to −0.006 | +0.356 to +0.455 |

The sign difference persists at matched training steps. But the older runs differ in treatment
scope: **REQ-019's runtime logs show that global LR scaled every optimizer group, including
embeddings, output weights, and other AdamW parameters. REQ-023 kept those rates fixed.**
The old comparisons therefore cannot isolate a Muon-neighbor mechanism.

Also, REQ-045's effective multiplier is **S_arm × m_i**, not m_i alone. Omitting S produced a
spurious “13× projection sensitivity gap”; the corrected descriptive gap is about 3.3×, still
based on noisy three-point curves.

Sources: [REQ-019](logs/kmaxwell/req019_eos_state_dependence/),
[REQ-023 assignments and raw data](logs/kmaxwell/req023_per_matrix_lr/),
[REQ-035 archives](logs/kmaxwell/req035_armA_seed_replication/),
[REQ-045 draws](logs/kmaxwell/req045_crossed_global_permatrix_lr/req045_draws.json).
These contrast audits motivated REQ-052; they are observations rather than new GPU results.

## What is withdrawn or unproven

- **Backward-rank “second channel”: withdrawn.** Bands 73, 74 and 76 joined deep blocks with
  an incorrect +1 shift, yielding 66 instead of 72 matrices per seed. Correct alignment removes
  the claimed backward-rank effect. The within-REQ-048 concentration association survives.
- **Universal type ordering or universal depth effect on k: unproven.** The table above contradicts
  a single ordering across all LR interventions.
- **No cross-layer influence: unproven.** A nonsignificant aggregate coefficient is not an
  equivalence test, and changing other optimizers is a distinct intervention.
- **Gradient mediation or exact LR invariance of R: unproven.** Algebraic identities do not show
  causation; small averages can conceal opposing matrix-level effects.
- **Independent replication from checkpoints: invalid.** Four model seeds are four independent
  network replicates. Five checkpoints or two forks from one lineage are not additional seeds.
- **Early concentration's origin: unresolved.** A profile visible at the first measurement does not
  establish when or why it formed. A nonsignificant time trend does not prove equilibrium.

## Experiments that can resolve the open questions

| Request | Current queue status | Discriminator |
|---|---|---|
| REQ-050 | OPEN | Measure initialization through step1500 to distinguish pre-existing from early-emerging depth structure. |
| REQ-051 | OPEN | Four bases, six LR levels per matrix; measure forward/backward/alignment responses and test whether early changes predict final k. |
| REQ-052 | OPEN | On the same bases, compare mixed LR, uniform Muon LR, and full-global LR; five added controls per seed, 20 continuations. |

REQ-051 decomposes measured responses using:

```text
g = ||a|| * ||d|| * rho
k_lambda = 2*k_g + k_R
k_g = k_a + k_d + k_rho
```

These are accounting checks. Their components must be measured at the same state with compatible
loss normalization. Prediction needs held-out seeds, and mediation needs additional causal evidence.
REQ-052 tests whether k must explicitly depend on which other learning rates change.

No execution handle or new results for these three requests has been verified. The current goal
remains open.

## Related optimizer findings

The paired three-seed [REQ-044 batch study](logs/kmaxwell/req043_paired_kernel_batch_ablation/README.md)
finds Bi-Maxwell's benefit shrinking from about −0.0105 validation loss at 1× batch to +0.0006
at 16×. K-Maxwell retains about −0.0047 at 16×. This supports a batch-sensitive denoising
explanation for Bi-Maxwell while leaving K-Maxwell's additional mechanism unresolved.
It does not establish a curvature-mediated cause.

Earlier results: [REQ-034](logs/kmaxwell/req034_kmaxwell_batch_ladder/),
[REQ-037](logs/kmaxwell/req037_nonlr_instrument/),
[REQ-038/041](logs/kmaxwell/req038_activation_backward_probe/).

## Cross-panel join audit (2026-09-05)

The withdrawn backward-rank result came from a **row-level merge** of two panels on a mis-shifted block
key. Every other two-panel result was audited for the same failure mode. The outcome separates two
structurally different ways of using two panels:

| result | how it combines panels | exposed? |
|---|---|---|
| withdrawn backward-rank channel | **row-level merge** of REQ-047 into REQ-048 on a shifted block key | **yes — retracted** |
| per-matrix LR elasticity (REQ-023 + REQ-045) | separate fits per panel, then estimates compared/pooled | no |
| writer-vs-internal LR contrast | separate per-panel contrasts, compared side by side | no |

**Key sets verified:** REQ-023 and REQ-045 contain the **same 72 matrix names** and the **same block
range 0–11** (identical key sets confirmed), so a name-keyed merge between them would in fact be safe —
but none of those results performs one. **An estimate computed entirely within a single panel cannot be
corrupted by a cross-panel key error**, which is why only the row-level merge failed.

**The distinction worth keeping:** *merging rows across panels is a modelling assumption requiring
validation; comparing independently-computed estimates is not.* The failed case used 66 of 72 matrices
per seed and that discrepancy was visible in every run that used it.

## Validation rules to retain

Validate matrix names and block indices before joining panels: expect blocks 0–11 and 72 matrices
per seed, check identical key sets, and compare shared measurements at compatible states.
Use actual runtime LR, explicit operator/loss scaling, and independently generated probe features.
Separate cross-type, within-type depth, and within-matrix treatment relationships.
Check shared terms, collinearity, group-summary artifacts and selection leakage before interpreting
a correlation. Report seed dependence, effect sizes, uncertainty, and inconclusive outcomes.

