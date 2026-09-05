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

## Cross-panel audit completed (2026-09-05)

The audit begun after the backward-rank retraction is now finished. **Every result that touches more
than one panel has been classified**, and the two remaining REQ-047 results were re-derived from raw
data to confirm they carry no join at all.

| result | panels used | verdict |
|---|---|---|
| withdrawn backward-rank channel | REQ-047 **merged into** REQ-048 | **retracted** — 66 of 72 rows, six blocks misaligned |
| q/k token-incoherence mediation | **REQ-047 only** (`da_cos_mean` and `align_ratio` are both its fields) | **clean** — no join exists |
| q/k step-alignment gap | **curvature probe only** (`curvature_along_polar` and `λ_top`) | **clean** — no join exists |
| per-matrix LR elasticity, writer contrast | REQ-023 and REQ-045, **fitted separately** | **clean** — estimates compared, not rows merged |

**Both clean REQ-047 results were re-derived, not merely inspected.** Token incoherence reproduces at
**8/9/8/8% of the v/proj reference across the four seeds** (recorded: 8–9%) on the **full 72 rows per
seed**. The step-alignment gap reproduces at **−0.321 dex with the same sign in 11 of 11 LR arms**
(recorded: −0.3079). Neither figure moves.

**The failure mode was narrower than it first appeared.** Only one analysis ever merged rows across
panels, and only that one broke. **The distinguishing feature is not "uses two panels" but "merges rows
on a key"** — a result computed inside one panel, or assembled from separately-fitted estimates, has no
key to get wrong.

## The unexplained component, re-tested on a validated join (2026-09-05)

The withdrawn backward-rank result left the ~43% of the layer profile that concentration does not
explain without a candidate. That question needs a **gradient-side** predictor, and **REQ-047 carries no
curvature field at all**, so `lambda/g^2` cannot be formed inside it — the test unavoidably requires a
join. Rather than avoid the operation that failed, the **key was validated first**:

| check | result |
|---|---|
| (seed, matrix-name) key sets, REQ-047 vs REQ-048 | **288 = 288, identical, zero unmatched either way** |
| rows per seed after joining on name | **72** (not 66) |
| distinct blocks after joining | **12** (not 11) |

**The earlier failure was not inherent to joining these panels.** A straight name-keyed join is exact;
the retracted work broke because it **imposed a block shift the data never required**.

**Re-running the test on that validated join gives a different and much weaker answer.** Seven
gradient-side candidates, declared in advance, with type and block dummies and spectral concentration
already in the model:

| candidate | partial correlation with the residual | same sign |
|---|---:|---:|
| **`d_frob`** (backward signal magnitude) | **+0.224** | 4/4 |
| `grad_rank1_frac` | −0.181 | 3/4 |
| `a_frob` | +0.116 | 4/4 |
| `d_eff_rank` | **+0.045** | 4/4 |
| `da_cos_mean` | **−0.068** | 4/4 |
| others | ≤ 0.05 | mixed |

Permutation null over matrix labels, max across all seven: **p = 0.0008**.

**⚠️ But the effect size is small, and that is the operative fact.** Adding `d_frob` on top of
concentration:

| | mean |
|---|---:|
| spectral concentration, beyond type + block | **44.3%** of variance |
| **`d_frob`, on top of that** | **2.8%** |
| per-seed significance | **2 of 4** |

**`d_frob` is a real but minor contributor — it nibbles at the residual rather than explaining it.**
Reporting it as the answer would repeat the overclaiming that produced the retraction.

**Two things this settles.** First, the two quantities the retracted bands named — `d_eff_rank` and
`da_cos_mean` — are **near zero on the corrected join** (+0.045 and −0.068), confirming the retraction
was correct rather than merely cautious. Second, **the residual remains substantially unexplained**: no
available gradient-side field accounts for more than a few percent of it, and REQ-047's field set is
now exhausted.

## Is the backward-magnitude association real? (2026-09-05)

`d_frob` emerged as the leading gradient-side correlate of the unexplained component. **It is a
multiplicative factor of `g`, and `g` sits in `C = lambda/g^2`'s denominator** — precisely the
shared-term structure that has produced several withdrawals in this campaign. Three checks were run
before treating it as real.

**The join is independently validated by a physical quantity.** REQ-047's `grad_frob` and REQ-048's
`gradient_block_norm` measure the same thing in different panels and correlate at **+0.9556** across
the name-keyed join (spread 0.074 dex). **A mis-paired join could not produce that.** The
component identity `g = |a|_F · |d|_F · align_ratio` also holds to **8.9e-16**, confirming the
decomposition is exact rather than approximate.

**The three components of `g` do NOT behave alike**, so the signal is not simply the `g` channel
leaking through the denominator:

| component of `g` | partial correlation with the residual | per seed |
|---|---:|---|
| **`d_frob`** (backward magnitude) | **+0.224** | +0.27, +0.27, +0.16, +0.21 |
| `a_frob` (forward magnitude) | +0.116 | +0.09, +0.16, +0.15, +0.07 |
| `align_ratio` | −0.044 | −0.12, −0.03, +0.05, −0.07 |

**⚠️ But `a_frob` and `d_frob` are negatively collinear (−0.454), and that changes the reading.**
Entering both **raises** each coefficient — `a_frob` +0.635, `d_frob` +0.688 — which is **mutual
suppression, not two independent channels**. `d_frob` is significant in **3 of 4** seeds, `a_frob` in
**1 of 4**.

**What this supports, stated at its actual strength:** the residual is associated with the **backward
signal magnitude** more than with the forward magnitude or the alignment, the association survives the
shared-term check, and the join underlying it is verified against a physically shared measurement.
**What it does not support:** treating `d_frob` as *the* explanation. Its incremental contribution is
**2.8%** of variance against concentration's 44.3%, and the forward and backward magnitudes are
entangled rather than separable.

## The gradient-side association is construction — withdrawn (2026-09-05)

Iteration 213 reported `d_frob` (backward signal magnitude) as the leading gradient-side correlate of
the unexplained component (+0.224, surviving a permutation null at p = 0.0008). **That result is
withdrawn.** Following it one step further exposed the reason.

**Step 1 — it is not the backward half.** Testing the sum and difference of the two magnitudes:

| predictor | partial correlation with the C residual |
|---|---:|
| **`log a + log d`** (the product `\|a\|·\|d\|`) | **+0.333** |
| `log d` alone | +0.224 |
| **`log d − log a`** (the difference) | **+0.076** |

The **product** carries the signal and the **difference is null** — so the effect was never specific to
the backward direction.

**Step 2 — and the product is `g`.** For a bias-free Linear, `g = \|a\|_F · \|d\|_F · align_ratio`
exactly (verified to 8.9e-16), so `log a + log d = log g − log align`. Adding the alignment factor back:

| predictor | partial correlation |
|---|---:|
| `log a + log d` | +0.333 |
| **`log g`** | **+0.341** |

**`log g` alone matches the product.** But **`C = lambda/g²` has `g` in its denominator by
construction**, and REQ-047's `grad_frob` correlates with REQ-048's `gradient_block_norm` at **+0.9556**
— the same physical quantity in both panels. **So this is `g` appearing on both sides of the
regression.** It is a shared-term artifact, not a mechanism.

**What this means for the open question.** The unexplained component is **still unexplained**, and now
for a structural reason worth stating: **every gradient-side quantity REQ-047 measures is a factor of
`g`**, and `g` is already inside `C`. **No decomposition of `g` can serve as an independent predictor of
a residual defined using `g`.** Separating these requires an experiment that varies the gradient side
*causally* rather than observing it — which is what REQ-051's per-matrix LR ladder does.

**Method note.** Three checks in sequence were needed: a permutation null (passed), a component
comparison (redirected the finding from `d` to the product), and the identity check (killed it). **The
permutation null was never going to detect this** — the correlation is real, reproducible and
seed-stable; it is simply between a quantity and itself.

## A g-free predictor of the unexplained component (2026-09-05)

The previous entry withdrew the gradient-side association because every REQ-047 *magnitude* field is a
factor of `g`, and `g` sits in `C = lambda/g^2`. **I then claimed the seam was exhausted. That claim was
wrong, and testing it found the campaign's first admissible predictor of the residual.**

**The distinction that was missed: `g`-family versus `g`-free fields.**

| class | fields | usable for this residual? |
|---|---|---|
| **`g`-family** (multiplicative factors of `g`) | `a_frob`, `d_frob`, `align_ratio`, `grad_frob` | **no — shared-term artifact** |
| **`g`-free** (ratios, ranks, coherences — scale-invariant) | `d_cv`, `d_part`, `a_cv`, `a_part`, eff-ranks, `da_cos_mean`, `grad_rank1_frac` | **yes** |

**`d_cv` — the coefficient of variation of the per-token backward norms — is the leading g-free
predictor**, from eight declared in advance:

| candidate | partial correlation with the residual | same sign |
|---|---:|---:|
| **`d_cv`** | **−0.445** | **4/4** |
| `d_part` | +0.325 | 4/4 |
| `grad_rank1_frac` | −0.181 | 3/4 |
| `a_part` | −0.179 | 4/4 |
| others | ≤ 0.07 | mixed |

Permutation null over matrix labels, max across all eight: **p = 0.0002**.

**It is genuinely `g`-free, formally and empirically.** `d_cv` is a **ratio** (sd ÷ mean of token
norms), so scaling the backward signal by any constant leaves it unchanged — it cannot be a factor of
`g`. **In the data its partial correlation with `log g` is −0.169**, confirming the formal argument
rather than assuming it. **This is exactly the check the withdrawn `d_frob` result failed.**

**`d_cv` and `d_part` are one construct.** They are collinear at **−0.731**, and in a joint fit `d_cv`
survives (mean −0.315) while **`d_part` collapses to exactly 0.000**. Participation ratio and
coefficient of variation are two views of the same token-norm dispersion.

**Effect size, with its variability stated.** Incremental R² of `d_cv` over type + block + concentration:
**21.1% on average**, but **6.9% to 32.3% across seeds**, and per-seed |t| ≥ 2 in only **2 of 4**. **It
is a substantial but unstable contributor** — larger than anything else found for the residual, and not
yet a settled result.

**Interpretation.** C is higher where the backward signal is **evenly spread across tokens** (low
dispersion) and lower where a few tokens dominate. Together with concentration this gives two
scale-invariant descriptors — one of the curvature spectrum, one of the token-level gradient
distribution — neither of which is a factor of `g`.

**Method note.** The productive step was **classifying candidates by their algebraic relationship to the
outcome before testing them**, rather than by which panel they came from. Every previously withdrawn
gradient-side finding was `g`-family; the surviving one is not.

## The g-free predictor is one matrix type, not a general effect (2026-09-05)

The previous entry recorded `d_cv` (token-norm dispersion of the backward signal) as the first
admissible predictor of the unexplained component: −0.445 across four seeds, permutation p = 0.0002,
21.1% incremental variance. **It also flagged the effect as unstable — 6.9% to 32.3% by seed. Chasing
that instability changes the reading.**

**The instability is not in the predictor.** `d_cv`'s own distribution is near-identical across seeds
(sd 0.310–0.325, mean 0.811–0.820) and the outcome's residual spread is likewise uniform (0.103–0.126
dex). So neither side of the regression varies — the *relationship* does.

**Fitted within each matrix type, only one type carries it:**

| type | partial correlation | per-seed values |
|---|---:|---|
| **`mlp.proj`** | **−0.717** | −0.75, −0.65, −0.92, −0.55 — **all four negative** |
| `attn.v` | −0.341 | −0.36, −0.42, **+0.01**, −0.60 |
| `mlp.fc` | +0.212 | **−0.09, −0.47, +0.93, +0.48** — sign flips |
| `attn.q` | +0.145 | +0.07, +0.21, +0.76, **−0.46** |
| `attn.proj` | +0.171 | −0.01, +0.13, +0.05, +0.52 |
| `attn.k` | +0.097 | **+0.47, −0.39,** +0.32, −0.01 |

**`mlp.proj` is consistent and strong; the other five types are unstable or opposite-signed.** The
pooled −0.445 is **one matrix type generalised to six** — the same failure mode as the withdrawn
`mlp.fc` concentration exception, which also looked general until fitted per type.

**What survives, stated narrowly.** For **`mlp.proj` specifically**, C is lower where the backward
signal's token-norm dispersion is higher, consistently across all four seeds. **That is a real
type-specific association.** It is **not** a general account of the unexplained component, and the
21.1% figure should not be quoted as one.

**A specification error worth recording.** The first attempt at this test fit block dummies *within*
each type — 12 matrices against 12 dummies, zero degrees of freedom, all-NaN output. **The NaNs were
the only reason the mistake surfaced**; a design with one more matrix per type would have returned
plausible numbers from an over-specified fit. **Check residual degrees of freedom before reading any
per-group regression.**

## The mlp.proj association survives every guard (2026-09-05)

The previous entry narrowed the `d_cv` result to a single matrix type and warned it might be the
band-37 failure mode: **an apparently special type that is really just the noisiest one.** That warning
was testable, and it is refuted — `mlp.proj` is special in the opposite direction.

**It is the HARDEST type to find a correlation in, and shows the strongest one:**

| type | residual sd | `d_cv` spread | partial correlation |
|---|---:|---:|---:|
| attn.v | 0.0644 | **0.4333** (widest) | −0.341 |
| attn.k | 0.0910 (widest) | 0.1984 | +0.097 |
| attn.q | 0.0807 | 0.1300 | +0.145 |
| mlp.fc | 0.0576 | 0.0580 | +0.212 |
| **mlp.proj** | **0.0588** | **0.0474** (2nd narrowest) | **−0.717** |
| attn.proj | 0.0753 | 0.0412 (narrowest) | +0.171 |

**`attn.v` has 9× `mlp.proj`'s predictor spread and less than half the effect.** Across the six types,
**corr(detection ease, effect size) = −0.090** — **the strongest associations appear where detection is
hardest.** A power artifact would give the opposite sign. **This is the reverse of band 37's mlp.fc
exception, which was the noisiest type rather than a distinctive one.**

**And the six-way selection is priced.** Permuting block labels within each type-seed cell and taking
the max \|r\| across all six types: **p = 0.0002**. `mlp.proj`'s −0.717 is not the best of six noisy
draws.

**What is now established, at its actual scope.** For **`mlp.proj` only**: C is lower where the backward
signal's token-norm dispersion is higher — **all four seeds negative (−0.75, −0.65, −0.92, −0.55)**,
selection-priced, and not attributable to that type being easier to measure in. **This is the first
predictor of the unexplained component to survive every guard this campaign applies** — the shared-term
check (it is `g`-free, a ratio), the per-type check (it is type-specific and stated so), the
power-artifact check, and a selection-priced permutation null.

**What it is not.** It explains the residual for **one of six matrix types**. The other five remain
unexplained, and the campaign's headline decomposition is unchanged: **majority spectral concentration,
with a substantial remainder that has no general account.**

**Why `mlp.proj` plausibly differs.** It is a residual-stream writer whose input is the ReLU²
expansion — the one type whose forward signal is both nonlinear and dimension-reduced before writing.
**That is a structural difference, not a fitted one, but nothing here tests it causally.**

## Validation rules to retain

Validate matrix names and block indices before joining panels: expect blocks 0–11 and 72 matrices
per seed, check identical key sets, and compare shared measurements at compatible states.
Use actual runtime LR, explicit operator/loss scaling, and independently generated probe features.
Separate cross-type, within-type depth, and within-matrix treatment relationships.
Check shared terms, collinearity, group-summary artifacts and selection leakage before interpreting
a correlation. Report seed dependence, effect sizes, uncertainty, and inconclusive outcomes.

