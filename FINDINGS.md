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

## Neither structural explanation for mlp.proj survives (2026-09-05)

The previous entry suggested `mlp.proj` might differ because **its input is the ReLU² expansion** — and
flagged that nothing tested it. **Tested, that explanation fails, and so does the obvious alternative.**

**Test 1 — the expansion.** `mlp.fc` faces the *same* expansion from the other side: its **output**
gradient is the gradient with respect to the expansion, and `d_cv` is a property of the output gradient.
So if the expansion drove the coupling, `mlp.fc` should show it.

| type | relationship to the ReLU² expansion | `d_cv` partial correlation |
|---|---|---:|
| **`mlp.proj`** | input **is** the expansion | **−0.717** (4/4 negative) |
| `mlp.fc` | output **is** the expansion | **+0.212** (sign flips: −0.09, −0.47, +0.93, +0.48) |

**Opposite sign and unstable — the expansion-as-such explanation is refuted.**

**Test 2 — the residual-writer role.** `attn.proj` is the other stream writer, with no expansion
involved:

| type | role | `d_cv` partial correlation |
|---|---|---:|
| **`mlp.proj`** | writer | **−0.717** |
| `attn.proj` | writer | **+0.171** |
| four readers | — | +0.028 mean |

**The two writers differ by 0.888 — more than the writer/non-writer gap itself.** The writer role does
not explain it either.

**Where this leaves the finding.** The `mlp.proj` association is **statistically solid** — all four
seeds negative, selection-priced at p = 0.0002, and shown not to be a power artifact. **But it has no
structural account.** Neither of the two groupings that distinguish `mlp.proj` from the other five types
predicts which types show the effect.

**Stated plainly: this is a validated association without a mechanism.** Recording it that way is more
useful than attaching a structural story that its own discriminating test refutes — the campaign has
already withdrawn several findings that were kept alive by plausible-sounding structural narratives.

**What would settle it.** `mlp.proj` and `attn.proj` differ in input width (3072 vs 768) and in whether
their input passed through a nonlinearity. **REQ-051 measures `k_a`, `k_d` and `k_rho` per matrix under a
per-matrix LR ladder**, which would show whether `mlp.proj`'s backward channel responds differently to
intervention — a causal discriminator that no observational split on this data can provide.

## A third explanation for mlp.proj fails, and the type axis is exhausted (2026-09-05)

Two structural groupings — the ReLU² expansion and the residual-writer role — already failed to predict
which types show the `d_cv` coupling. **The remaining option was that the effect is graded rather than
binary, with the six-type ordering tracking some measured property.**

**Per-type `d_cv` coefficient, ordered, against six pre-declared measured properties:**

| type | `d_cv` coef | `d_cv` | `a_cv` | `d_eff` | `a_eff` | `da_cos` | `rank1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mlp.proj** | **−0.717** | 0.580 | 0.631 | 355 | 142 | 0.091 | 0.227 |
| attn.v | −0.341 | 1.256 | 0.035 | 108 | 31 | 0.417 | 0.543 |
| attn.k | +0.097 | 0.964 | 0.035 | 77 | 31 | −0.018 | 0.183 |
| attn.q | +0.145 | 0.862 | 0.035 | 76 | 31 | 0.060 | 0.238 |
| attn.proj | +0.171 | 0.593 | 0.229 | 383 | 61 | 0.085 | 0.218 |
| mlp.fc | +0.212 | 0.647 | 0.038 | 876 | 19 | 0.036 | 0.270 |

| property | correlation with the coefficient |
|---|---:|
| `a_eff_rank` | **−0.786** |
| `a_cv` | −0.747 |
| `da_cos_mean` | −0.447 |
| others | ≤ 0.33 |

**⚠️ The best correlation is −0.786 and it means nothing.** Permutation over the six type labels, taking
the max across all six properties: **p = 0.3132**. **With six points and six candidates, |r| ≈ 0.79 is an
ordinary draw** — the achievable floor is about 0.003, so a p of 0.31 is uninformative rather than
marginal. *(This is the same arithmetic that made the writer-split p of 0.068 uninterpretable earlier:
six types cannot support a six-candidate search.)*

**⇒ The type axis is exhausted for this question.** Three explanations have now been tested and failed —
the expansion (refuted by its own discriminating prediction), the writer role (refuted by `attn.proj`),
and a graded ordering (unsupported at n=6). **No further structural hypothesis can be distinguished on
six types**, regardless of how plausible it sounds: the data cannot separate a real ordering from a
random one at this size.

**What stands.** The `mlp.proj` association itself is unaffected — it rests on **72 matrices across
4 seeds**, not on the six-type comparison, and remains selection-priced at p = 0.0002 with all four
seeds negative. **What is now firmly established is the boundary: a validated association with no
structural account, and no prospect of one from this data.**

**Method note worth keeping.** Each of the three failed explanations was tested by a criterion chosen
*before* seeing its result — the expansion by whether `mlp.fc` shares it, the writer role by whether
`attn.proj` matches, the ordering by a selection-priced null. **All three were plausible; none
survived.** The alternative — accepting the first plausible story — is how bands 61 and 73 stayed alive
long enough to require retraction.

## The mlp.proj association is withdrawn — d_cv is collinear with depth (2026-09-05)

The `mlp.proj` result was recorded as **the first predictor of the unexplained component to survive
every guard** — all four seeds negative at −0.717, selection-priced at p = 0.0002, and shown not to be a
power artifact. **It is withdrawn.** Extending it along the depth axis produced a sign contradiction that
exposed the reason.

**The contradiction.** Testing whether the effect was uniform across depth returned **+0.67 in all four
seeds** — opposite to the recorded −0.717 on the same rows. The two analyses differed in exactly one
respect: the depth control.

| specification | s0 | s1 | s2 | s3 | mean |
|---|---:|---:|---:|---:|---:|
| no depth term | +0.940 | +0.889 | +0.886 | +0.860 | **+0.894** |
| **linear depth** | +0.747 | +0.668 | +0.669 | +0.573 | **+0.664** |
| **quadratic depth** *(as recorded)* | −0.748 | −0.651 | −0.922 | −0.548 | **−0.717** |
| cubic depth | −0.500 | −0.610 | −0.843 | −0.385 | −0.585 |

**The sign is a property of the depth polynomial, not of the data.**

**The cause.** Within `mlp.proj`, `d_cv` is very nearly a function of depth:

| seed | corr(`d_cv`, block) | corr(`d_cv`, block²) |
|---|---:|---:|
| 0 | +0.882 | **+0.963** |
| 1 | +0.840 | **+0.932** |
| 2 | +0.908 | **+0.975** |
| 3 | +0.839 | **+0.945** |

**With 12 matrices and a predictor correlated at +0.97 with block², `d_cv` and depth cannot be
separated.** Every figure in that analysis — the −0.717, the p = 0.0002, the power-artifact check — was
computed on a fit that is degenerate in a way none of those tests examines.

**Why the earlier guards passed it.** The permutation null shuffled `d_cv` against a *fixed* design
matrix, so it priced the selection of `mlp.proj` from six types but **never questioned the depth
specification**. The power-artifact check compared spreads, not collinearity. **All three guards were
answering a different question from the one that mattered.**

**⇒ The unexplained component has no surviving predictor.** Every candidate has now failed for a
distinct reason: the `g`-family fields by construction, `d_eff_rank` and `da_cos_mean` by a bad join,
the six-type ordering by insufficient points, and `d_cv` by collinearity with depth.

**Rule addition.** *Check a predictor's collinearity with the CONTROLS, not only with other predictors.*
Rule 27 covered predictor-to-predictor collinearity at the finest granularity; this failure was
predictor-to-control, inside a single type where depth has only 12 levels. **A result whose sign depends
on the order of a nuisance polynomial is not a result.**

## The concentration result survives the failure that killed everything else (2026-09-05)

`d_cv` was withdrawn because its sign flipped with the depth polynomial — within `mlp.proj` it
correlated with block² at +0.975, so 12 matrices could not separate it from depth. **The concentration
result is the campaign's headline and is fitted the same way, so it needed the same test. It passes.**

**The coefficient of `log n_eff` does not move across five depth controls:**

| depth control | s0 | s1 | s2 | s3 | mean |
|---|---:|---:|---:|---:|---:|
| none | −0.413 | −0.404 | −0.421 | −0.477 | **−0.429** |
| linear | −0.457 | −0.420 | −0.426 | −0.490 | **−0.448** |
| quadratic | −0.384 | −0.356 | −0.352 | −0.377 | **−0.367** |
| cubic | −0.379 | −0.335 | −0.327 | −0.380 | **−0.355** |
| **full block dummies** | −0.374 | −0.329 | −0.324 | −0.359 | **−0.347** |

**Negative in all 20 specification-seed combinations, spanning only −0.324 to −0.490.** For contrast,
`d_cv` swung from **+0.894 to −0.717** across the same ladder.

**And it holds within every type**, including the one where its depth-collinearity is highest:

| type | corr(`log n_eff`, block²) | linear-depth coef | quadratic-depth coef | sign flip? |
|---|---:|---:|---:|:---:|
| **mlp.fc** | **+0.879** | −1.682 | −1.062 | **no** |
| attn.proj | −0.553 | −0.611 | −0.546 | no |
| attn.q | +0.501 | −0.530 | −0.513 | no |
| attn.v | −0.347 | −0.515 | −0.466 | no |
| mlp.proj | +0.328 | −0.541 | −0.485 | no |
| attn.k | −0.250 | −0.524 | −0.295 | no |

**⇒ And the contrast explains both outcomes.** It is not the raw collinearity that decides — `mlp.fc`
reaches +0.879 and survives. **It is whether the design supplies variation that breaks the collinearity.**

| | `d_cv` within `mlp.proj` | `log n_eff` pooled |
|---|---|---|
| points | 12 | 72 across six types |
| corr with block² | 0.975 | ≤ 0.92 in one type |
| **variance surviving type + quadratic depth** | near zero | **54–58%** |
| outcome | **sign flips** | **stable** |

**`log n_eff` retains over half its variance after depth and type are removed**, so the fit has real
independent variation to work with. `d_cv` inside a single type had almost none. **That is the
structural difference between a result and an artifact here, and it is measurable in advance.**

**What this settles.** The campaign's central claim — **between-layer C is majority spectral
concentration** — has now been tested against every failure mode that has withdrawn a finding in this
work: shared-term construction (it is `g`-free), bad joins (it is computed within one panel),
aggregation artifacts (it holds per matrix and per block), selection (it was not chosen from a search),
and now depth-collinearity. **It is the one result that has survived all of them.**

## The concentration coefficient has no closed form from the committed moments (2026-09-05)

Bands 71/75 established that spectral concentration carries the majority of the between-layer C
profile, and iteration 222 showed the result survives depth-collinearity. What had never been tested
is the coefficient's **value**. The saturated fits give d(log C)/d(log n_eff) = -0.347; controlling
trace and fitting log lam directly gives **-0.5745** (sd 0.0305 across 4 seeds, t vs 0 = -37.6).

Three parameter-free predictions were derived and **all three were refuted by the data.**

**Attempt 1 -- the small-share limit, prediction -0.500.** With lam a small share of trace(H) (median
lam/trace = 0.0040, so the regime assumption holds) and lam dominating trace(H^2), n_eff ~ T^2/lam^2
gives exactly -0.5. Measured -0.5745, t vs -0.5 = **-4.88** across seeds, all four seeds below -0.5.
REFUTED -- consistent overshoot, not noise.

**Attempt 2 -- the exact partial derivative, prediction -1/(2f).** Relaxing "lam dominates trace(H^2)"
to a measured share f = lam^2/trace(H^2) gives d(log lam)/d(log n_eff) = -1/(2f) at fixed bulk.
Measured f = 0.0814 (median 0.0354), so the prediction is **-6.15** against a fitted -0.574.
REFUTED by an order of magnitude.

**The diagnosis, tested rather than asserted.** Attempt 2 assumed the bulk S = trace(H^2) - lam^2 is
held fixed while lam varies. It is not: corr(log lam, log S) = **+0.753** pooled (+0.732 to +0.773
per seed), and the saturated elasticity is d(log S)/d(log lam) = **+1.271** (sd 0.036). The bulk
co-moves strongly with the top eigenvalue, so the observed between-matrix slope is a total
derivative and the fixed-bulk partial was simply the wrong object.

**Attempt 3 -- the co-moving-bulk formula, prediction -1/(k + (2-k)f) with k = 1.271.** Both inputs
measured independently of the -0.5745 regression, so this is arithmetic, not a fit. Predicted -0.730
to -0.773 per seed against fitted -0.539 to -0.605: mean gap **-0.178**, sd 0.049, t = **-7.23**.
REFUTED.

**What this establishes.** The two Hutchinson moments trace(H) and trace(H^2) do **not** determine
the concentration coefficient. Three successive relaxations of the spectral model -- each removing
the assumption the previous one violated -- land at -0.500, -6.15 and -0.73 against a measured
-0.574. The search was stopped here deliberately: a fourth algebraic form chosen to land on -0.574
would be fitting, not deriving, and would carry none of the falsifiability of these three.

An incidental refutation, recorded so it is not re-used: f is **not** a scale-free shape statistic.
corr(log lam, f) = +0.48 and corr(log trace, f) = -0.31 per seed, so the "n_eff is pure shape,
therefore no scale-based derivation can apply" escape was itself wrong and is withdrawn.

**Effect on the standing account: none.** The concentration finding is a claim about sign, share and
robustness -- 57% of the layer profile, 37% incremental beyond type and block, no sign flip in 20
specification-seed combinations. Its coefficient being empirical rather than derived does not weaken
any of that. What is now on record is the sharper statement: the account explains **that** and **how
much**, and cannot yet explain **why that much**. The question relocates to what sets the
bulk-to-top elasticity k = 1.271, which these two moments cannot answer -- it needs the eigenvalue
distribution, which is not in the committed data at any depth.

## The three REQ-048 files per seed are probe repeats, not training steps (2026-09-05)

A structural fact about the only curvature archive, established this iteration and consequential
enough to record before any further use of that data. The twelve REQ-048 files are named
`req048_s{seed}_s{060,100,170}.json`. The `s060/s100/s170` suffix was read throughout this campaign
as a training step. It is not: **every file reports `step = 2750`**. They are three independent
probe repeats (Hutchinson/Lanczos seeds) of one checkpoint, 4 seeds x 72 matrices x 3 repeats = 864
rows. There is only one checkpoint in the archive, so no step-stability guard can be run on it.

**This makes probe reliability measurable for the first time**, since repeats of one fixed quantity
are exactly what a reliability estimate needs.

| quantity | within-matrix sd | between-matrix sd | reliability (m=3) |
|---|---|---|---|
| `log lam` | 0.349 | 0.347 | **0.748** |
| `log n_eff` | 0.181 | 0.556 | **0.966** |
| `log C` | 0.178 | 0.434 | 0.947 |

`log lam` is **noisy: 40.5% of its single-probe variance is measurement error.** This sounds alarming
for a campaign built on lam, and the reason it is not is a point about which side of the regression
the noise sits on. Classical errors-in-variables attenuates a slope only through noise in a
**regressor**; noise in the outcome inflates standard errors but leaves the slope unbiased. In the
headline fit, lam enters only through the outcome `log C = log lam - 2 log g`. The regressor is
`log n_eff = 2 log T - log trace(H^2)`, whose reliability is 0.966.

**Effect on the headline: negligible.** The saturated concentration coefficient is -0.3466 measured,
**-0.3589 disattenuated** -- a 3.5% correction, in the direction of a *larger* effect. Sign, share
and the twenty no-sign-flip specification-seed combinations are untouched.

**Effect on iteration 223: it further undercuts the coefficient-matching exercise**, which was
already refuted three times. The -0.5745 target was itself attenuated by a factor nobody had
measured. Recorded for completeness; it does not rescue any of the three predictions, whose misses
(-0.500, -6.15, -0.730) are all far larger than a 3.5% correction.

**Correction to iteration 223's k.** k was fitted on probe-averaged rows and reported as +1.271; the
correctly-averaged value is **+1.312** (sd 0.048). Pooling the three repeats as independent rows
instead gives +1.44 -- inflated, because probe noise in `log lam` enters `log S = log(trace(H^2) -
lam^2)` on both sides. Averaging repeats before fitting is the correct treatment. Shared-term
contamination (rule 22) is small: the uncontaminated analogue `d(log trace(H^2))/d(log lam)` is
+1.364, a difference of +0.052.

## The bulk-to-top elasticity is not a constant, so there is no k to explain (2026-09-05)

Iteration 223 closed by relocating the open question to "what sets k = 1.271?". That question is
withdrawn: **k varies by matrix type by more than six times its seed uncertainty.**

| type | k | seed sd |
|---|---|---|
| mlp.fc | **+0.466** | 0.099 |
| attn.k | +0.883 | 0.283 |
| attn.q | +0.921 | 0.157 |
| attn.proj | +1.071 | 0.078 |
| mlp.proj | +1.536 | 0.032 |
| attn.v | **+1.587** | 0.488 |

Range 1.121 against a typical seed sd of 0.190. A pooled k of +1.312 is an average over six different
elasticities, not a spectral constant, and "explain k" was the wrong question to hand forward.

## Ritz values cannot reopen the spectral question -- on two independent grounds (2026-09-05)

Iteration 223 asserted that the eigenvalue distribution is absent from committed data. That was
**wrong as stated** and is corrected here: REQ-048 commits the full Lanczos tridiagonal (`alphas[8]`,
`offdiags[7]`), whose 8 Ritz values are partial spectral information. The conclusion nevertheless
holds, for two reasons established by measurement rather than assertion.

**1. The hard rule rejects them, and the data confirms why.** `ritz1` is bit-for-bit identical to
`top_eigenvalue` (max relative difference 7.2e-16) -- the same object, not a correlate. The
subdominant Ritz values are not independent probes of a different spectral region: controlling the
independent trace probe, partial corr(log ritz2, log lam | log trace) = **+0.957** and for ritz3
**+0.923**. All 8 come from one Krylov sequence seeded by one starting vector, so a poor seed or
early convergence moves them together. They are circular under the standing rule.

**2. Even setting circularity aside, they cannot describe the bulk.** The 8 Ritz values account for a
median **9.2%** of trace(H^2) and **1.4%** of trace(H). The ~91% of trace(H^2) that carries the bulk
behaviour is invisible to them.

**Cross-archive replication of any curvature moment is impossible from committed data.** REQ-047,
the only other per-matrix archive, holds activation and gradient fields (`weight_frob`, `a_frob`,
`d_frob`, `d_eff_rank`, `da_cos_mean`, `align_ratio`, `grad_rank1_frac`) and **no** `trace_est`,
`trace_sq_est` or `top_eigenvalue`. REQ-048 is the sole source of curvature moments, at a single
checkpoint. Any spectral question beyond the top eigenvalue requires new measurement -- which is what
REQ-050 and REQ-051 are for.

## "Interior minimum below both ends" is a weak criterion and is retired (2026-09-05)

A guard used repeatedly in the pre-consolidation record (bands 3, 63 and the summary table, all at
commit `28d0074`) is shown here to carry almost no evidential weight, and is retired.

The criterion asks whether free per-block effects put the profile's minimum strictly inside the
network and below both endpoints, reported as "12/12" or "4/4". **Its chance rate is ~0.48.** With
12 exchangeable per-block values the minimum falls in the interior 10 of 12 slots by pure
combinatorics: P = 10/12 = **0.8335**, so four independent seeds agree by chance at
0.8335^4 = **0.4826**. A block-relabelling permutation on the real REQ-048 data returns **0.470**
(940/2000), matching the combinatorial rate. The criterion is weak by construction, for reasons
having nothing to do with this data.

**This does not weaken the bowl.** The consolidation had already dropped the criterion from the
active FINDINGS.md -- it appears nowhere in the current file -- so no standing claim rests on it.
And the bowl's real evidence is far stronger on the same data:

| criterion | observed | chance |
|---|---|---|
| all 4 seed argmins within a span of 1 | argmin **[6, 6, 6, 7]**, span 1 | **0.008** |
| cubic depth fit beats linear, per seed | R² gain **+0.070 to +0.117**, 4/4 | -- |

The location agreement is the load-bearing evidence, not the interiority. Retire the weak criterion;
quote argmin agreement and the cubic-over-linear gain instead.

## Band 58 restated: three directions have depth profiles, one turns around inside (2026-09-05)

Iteration 225 checked whether any past NEGATIVE result was killed by probe noise rather than by
absence of signal, using the reliabilities that iteration 224's probe-repeat discovery made
estimable. Ranking every REQ-048 field:

| field | reliability (m=3) | admissible? |
|---|---|---|
| `curvature_along_random` | **0.651** | yes |
| `curvature_along_gradient` | 0.704 | **no -- circular** |
| `top_eigenvalue` | 0.748 | (the outcome) |
| `curvature_along_polar` | 0.759 | yes |
| `trace_sq_est` | 0.768 | yes |
| `curvature_along_weight` | 0.799 | yes |
| `residual_tail` | 0.820 | **no -- circular** |
| `trace_est` | 0.838 | yes |
| `gradient_block_norm` | 0.917 | yes |
| `participation_ratio` | 0.959 | yes |

`curvature_along_random` at 0.651 was low enough to make band 58's null on it suspect, so band 58
was re-tested. **The concern turned out not to apply, for a reason worth recording:** block index is
measured exactly, so regressing a noisy *outcome* on exact depth is unbiased. Attenuation acts only
through noise in a *regressor*. Probe noise inflates standard errors and deflates R², but cannot
manufacture or hide a depth profile's amplitude.

The re-test nevertheless **corrects band 58's wording.** It is not true that only the top
eigendirection varies with depth. Cubic amplitudes, probe-averaged, type-absorbed:

| direction | amplitude | seed sd | t | free-fit argmin (4 seeds) |
|---|---|---|---|---|
| top eigenvalue | 0.466 | 0.040 | 23.4 | **6, 6, 6, 7** |
| Muon step (polar) | 0.376 | 0.092 | 8.2 | 9, 10, 9, 10 |
| learned weight | 0.810 | 0.090 | 18.0 | 9, 10, 9, 11 |
| random | **0.166** | 0.042 | **7.9** | 2, 11, 3, 11 |

The random direction's profile is significantly nonzero -- "flat" was imprecise. What distinguishes
the top eigendirection is **where** the profile turns: its minimum sits at block 6-7 in all four
seeds, while the weight direction runs monotonically toward the output end and the random direction
gives an unstable argmin (2, 11, 3, 11) consistent with no reproducible location.

Note the internal consistency: the interiority of the Muon-step and weight profiles (both 3-4 of 4
"interior") is exactly the weak criterion retired above, and their argmin **spans** are 1 and 2
against the top eigendirection's 1 at a much more distinctive location. Amplitude and interiority
were never the discriminators; **reproducible argmin location** is.

## "Same sign in 4/4 seeds" is a weak criterion in this design (2026-09-05)

Rule 31 (compute a criterion's chance rate before citing it) was applied to the campaign's most
frequently cited corroboration. It does not survive.

**Calibration.** A pure-noise predictor was drawn 4000 times and run through the campaign's own
specification -- per-seed partial correlation against the residual of `log C` on type and block
dummies -- counting how often the four per-seed signs agreed.

| null model | P(4/4 same sign) | P(3/4 or better) |
|---|---|---|
| independent noise per seed (the idealised assumption) | 0.126 | 0.616 |
| **shared structural predictor** (one value per matrix name, as every real candidate is) | **0.582** | **0.868** |

The nominal rate 2*(1/2)^4 = 0.125 is reproduced exactly by the idealised model, confirming the
simulation. But no predictor this campaign has tested is seed-specific noise: `d_frob`, `d_cv`,
`d_eff_rank`, `grad_rank1_frac`, `a_frob`, `n_eff` are all **structural properties of a matrix**,
identical or near-identical across seeds. For those, **4/4 sign agreement occurs by chance 58% of
the time.**

**The mechanism, measured.** The residual that every candidate is tested against is largely shared
between seeds. Cross-seed correlation of the `log C` residual after type and block are absorbed:

| | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| seed 0 | 1.000 | 0.761 | 0.789 | 0.708 |
| seed 1 | 0.761 | 1.000 | 0.794 | 0.760 |
| seed 2 | 0.789 | 0.794 | 1.000 | 0.734 |
| seed 3 | 0.708 | 0.760 | 0.734 | 1.000 |

Mean off-diagonal **+0.758**. The residual is a reproducible per-matrix structure, not seed noise.
Four seeds therefore supply close to **one** test of a structural hypothesis repeated four times,
not four independent tests. Seed replication in this design tests robustness to **initialisation**
-- genuinely useful, and not nothing -- but it is not independent replication of structure.

**What this does and does not touch.** It does not reverse any finding: a weak criterion passing
does not make a claim false, and every withdrawn claim was withdrawn for a *different* and still
valid reason (shared-term construction, the bad join, aggregation, selection, depth-collinearity).
What it removes is the **corroborative weight** of the "4/4" column wherever it appears in the tables
above -- for `d_frob` (+0.224), `grad_rank1_frac` (-0.181), `a_frob` (+0.116), `d_eff_rank` (+0.045),
`da_cos_mean` (-0.068), `d_part` (+0.325), `a_part` (-0.179) and `d_cv` (-0.445). Those partial
correlations stand or fall on their magnitudes, which is how the small ones (`d_eff_rank` +0.045,
`da_cos_mean` -0.068) should always have been read.

**The headline does not depend on it.** Tested per seed on that seed's own 72 matrices, with
standard errors clustered by block so within-block dependence is not counted as information:

| seed | coefficient | cluster-robust se | t | partial R² |
|---|---|---|---|---|
| 0 | -0.3919 | 0.0390 | **-10.04** | 0.562 |
| 1 | -0.3484 | 0.0205 | **-17.02** | 0.421 |
| 2 | -0.3312 | 0.0352 | **-9.40** | 0.344 |
| 3 | -0.3764 | 0.0329 | **-11.43** | 0.531 |

Every seed is individually significant with a partial R² of 0.34-0.56 **within** that seed. The
cross-seed sharing is irrelevant to these four tests. The concentration result rests on effect size
inside a single seed, which is what it should have rested on all along.

**Consequence for experiment design.** REQ-050 and REQ-051 both specify four seeds. Four seeds
remain correct for testing initialisation-robustness, but their registered criteria should be judged
on **within-seed effect sizes**, not on cross-seed sign agreement. This is now noted in REQ-050.

## The unexplained residual is 69% reproducible structure, not noise (2026-09-05)

Rule 32 found the `log C` residual is largely shared across seeds. Following that through gives a
hard **ceiling** on what any structural predictor could ever explain, computable without any
candidate in hand: the residual's intraclass correlation across seeds. A predictor that is a
property of the *matrix* cannot explain the seed-specific part, no matter how good it is.

| residual | mean cross-seed corr | ICC (variance decomposition) | reproducible | seed-specific |
|---|---|---|---|---|
| after type + block | +0.7578 | +0.7600 | **75.8%** | 24.2% |
| after type + block + `log n_eff` | +0.6896 | +0.6917 | **69.0%** | 31.0% |

Two independent estimators agree to 0.002. **The remaining target is real.** The long search for a
predictor of the unexplained ~43% was not chasing noise: 69% of what is left is reproducible
per-matrix structure, sd 0.099 dex of genuine signal. Every failed candidate failed on its merits.

## A type-by-depth interaction accounts for a third of that structure (2026-09-05)

Averaging the four seeds cancels most of the 31% seed noise and leaves an estimate of the structure
itself (sd 0.104 dex, 72 matrices). Type and block **main** effects are zero by construction here --
they were absorbed -- so anything left in those axes is a genuine interaction.

**Per-type depth slopes of the residual**, continuous depth, no binning:

| type | slope (dex/block) | seed sd | per-seed |
|---|---|---|---|
| `mlp.proj` | **+0.0298** | 0.0064 | +0.038, +0.024, +0.031, +0.026 |
| `attn.q` | +0.0127 | 0.0128 | −0.003, +0.021, +0.026, +0.007 |
| `mlp.fc` | +0.0111 | 0.0058 | +0.016, +0.012, +0.003, +0.014 |
| `attn.k` | −0.0051 | 0.0046 | −0.003, −0.006, −0.000, −0.011 |
| `attn.proj` | −0.0212 | 0.0071 | −0.021, −0.016, −0.031, −0.016 |
| `attn.v` | **−0.0274** | 0.0059 | −0.027, −0.035, −0.028, −0.020 |

Spread 0.057 dex/block. Three checks, chosen because the first presentation of this (a type x
early/late table) was **mechanically antisymmetric** -- within a type the residual is mean-zero, so a
two-bin split forces `late = -early` and proves nothing:

- **Continuous depth** (above) reproduces it without any binning.
- **Permutation null** over type labels: observed variance share 0.310, null mean 0.143, null 95th
  percentile 0.250, **p = 0.0065**.
- **Within-seed** (rule 32, the standard that matters): R² gain from per-type depth slopes is
  **0.468 / 0.339 / 0.408 / 0.288** on each seed's own matrices.
- **Leave-one-type-out**: R² gain stays 0.313-0.414 dropping any single type (full 0.375). Not
  driven by one type.

**The writer/internal split does not explain it.** The campaign's standing structural grouping puts
`attn.proj` and `mlp.proj` together as residual writers, but they sit at **opposite extremes**
(−0.021 and +0.030). Writers-vs-internal and mlp-vs-attn both fail to separate the ordering.

**Registered, not claimed.** The grouping that *does* separate cleanly is the attention output path
(`attn.v`, `attn.proj`) running negative against everything else. That grouping was **read off these
slopes**, so it cannot be tested on the data that produced it -- it is a hypothesis, and reporting it
as a finding would be the selection error rules 24 and 25 exist to prevent. It is registered below
for REQ-050/051 to test on new data.

**What this changes.** For the first time the unexplained residual has a described shape rather than
a list of failed predictors: a third of it is *how each matrix type's curvature drifts with depth*,
above and beyond the main depth bowl and the concentration term. That is a narrower question than
"what predicts the residual", and it is a question new measurement can answer.

## The type-by-depth interaction is a suppressed effect, and the cancellation is exact (2026-09-05)

Iteration 227's interaction was checked against the way `log n_eff` was controlled. The first result
looked like a failure and turned out to be the finding.

**The interaction exists only conditional on the concentration control.** R² gain from per-type depth
slopes on `log C`:

| control on `log n_eff` | mean R² gain | per seed |
|---|---|---|
| **none** | **0.065** | 0.055, 0.074, 0.111, 0.021 |
| linear (as used in 227) | 0.446 | 0.535, 0.399, 0.488, 0.360 |
| linear + quadratic | 0.470 | 0.583, 0.408, 0.501, 0.389 |
| separate slope per type | 0.498 | 0.648, 0.431, 0.496, 0.418 |

An effect that appears only after conditioning is a warning sign with two readings: benign
**suppression**, or a **collider artifact** from conditioning on a descendant of the outcome. The
second is a live concern here because `n_eff = trace(H)²/trace(H²)` and `trace(H²) ≥ lam²`, so
`n_eff` contains `lam` — it passes the letter of the hard rule (Hutchinson, not the Lanczos
tridiagonal) while still sharing a term with the outcome.

**Collider refuted, by construction rather than by argument.** Replacing the control with a
**lam-free** concentration measure, `n_eff_bulk = trace(H)²/(trace(H²) − lam²)`, leaves the result
intact: **0.437** versus 0.446. `log participation_ratio` gives 0.460. Conditioning on `lam` is not
what creates the interaction — and in fact `lam²` is only a median **2.7%** of `trace(H²)`, so
`n_eff` was never `lam`-driven.

**Suppression verified directly, not by elimination.** The effect decomposes exactly:
`total slope = beta x (slope of log n_eff) + direct slope`, with beta the concentration coefficient
(−0.362 mean). The identity closes to 1e-9 for every type:

| type | TOTAL | via `n_eff` | DIRECT |
|---|---|---|---|
| `attn.k` | −0.0017 | +0.0008 | −0.0025 |
| `attn.proj` | −0.0036 | +0.0150 | −0.0186 |
| `attn.q` | +0.0032 | −0.0120 | +0.0153 |
| `attn.v` | −0.0226 | +0.0021 | −0.0248 |
| `mlp.fc` | −0.0036 | −0.0173 | +0.0137 |
| `mlp.proj` | +0.0049 | −0.0275 | **+0.0324** |

Across the six types: sd(TOTAL) **0.0098**, sd(via) 0.0154, sd(DIRECT) **0.0219**, and
**corr(via, direct) = −0.920**. Per seed the correlation is −0.797 to −0.920 with
sd(TOTAL)/sd(DIRECT) = 0.39–0.64.

**The physical statement.** Each matrix type's curvature drifts with depth, and its spectral
concentration drifts the *opposite* way, largely cancelling in `C = lam/g²`. The near-null raw
type-by-depth effect in `C` is a **cancellation, not an absence**. `mlp.proj` is the clearest case: a
direct slope of +0.0324 dex/block almost entirely offset by −0.0275 through concentration, leaving
+0.0049.

**What carries the evidence, stated precisely.** `total = via + direct` is an exact algebraic
identity, so once both components are large and the total is small, cancellation is arithmetically
forced. The per-seed correlations *describe* that cancellation; they do not independently prove it —
six type-slopes per seed is a weak correlation test, and by rule 32 the four seeds are ~76% shared
and are not four independent confirmations. The identity is the load-bearing part.

**Effect on H1.** None of this scores H1, which stays registered for REQ-050/051. It does sharpen
what those runs should measure: the quantity with real between-type structure is the **direct**
component (sd 0.0219) rather than the total (sd 0.0098), so H1's slope criteria should be evaluated
on residuals **after** a concentration control — as specified — and the raw `log C` slopes should be
expected to look near-null even if H1 is true.

## Retraction: iteration 228's −0.920 correlation was an artifact of the decomposition (2026-09-05)

Iteration 228 reported `corr(via n_eff, direct) = −0.920` across the six types as evidence of
cancellation. **That statistic is withdrawn.** It is not evidence of anything.

`direct` is *defined* as `total − via`, so `via` appears on both sides with opposite sign and the
correlation is mechanically negative regardless of the data. Simulating independent components with
the observed standard deviations gives `corr(via, total − via)` a mean of **−0.813**, a median of
−0.874, and **P(≤ −0.920) = 0.32**. The observed value is an unremarkable draw from what the
decomposition produces on its own.

**How it was caught.** Iteration 229 tried to extend the claim from 6 types to 72 individual
matrices and got `corr(via, direct) = 0.0000` in every seed — exactly zero, because at that grain
`direct` is an OLS residual and is orthogonal to the regressor by construction. A test that can only
return one value is vacuous; that prompted an audit of the coarser test, which proved to be a milder
version of the same defect.

Two prior statements are corrected with it. The sd ratio comparison "sd(TOTAL) 0.0098 vs sd(DIRECT)
0.0219" carried no independent weight either — under independence sd(direct) would be 0.0183, so the
numbers were never far from what unrelated components give. And 228's claim that "the algebraic
identity is the load-bearing part" was **wrong in the opposite direction from the usual error**: the
identity `total = via + direct` closing to 1e-9 is arithmetic that holds for any data whatsoever, so
it proves nothing at all.

## The cancellation itself survives a proper test (2026-09-05)

The **conclusion** of iteration 228 stands; only its evidence was bad. The well-posed question is
about the total, not the decomposition: *is the type-by-depth structure of `log C` smaller than it
would be if each type's curvature drift and concentration drift were unrelated?*

**Test.** Hold every type's curvature-depth slope and every type's concentration-depth slope exactly
as observed, but permute **which concentration profile is paired with which type**. This breaks only
the pairing — the claim — and cannot be satisfied by the decomposition's algebra. Using the lam-free
control `n_eff_bulk` so the control shares no term with the outcome (rule 34):

| seed | sd(total) observed | null mean | null 5th pct | p |
|---|---|---|---|---|
| 0 | 0.01245 | 0.02877 | 0.01705 | **0.0059** |
| 1 | 0.01417 | 0.02765 | 0.01573 | **0.0343** |
| 2 | 0.01673 | 0.02913 | 0.01908 | **0.0141** |
| 3 | 0.00722 | 0.02500 | 0.01265 | **0.0036** |

Breaking the pairing roughly **doubles** the type-by-depth spread in `C`, in every seed. The observed
pairing is genuinely special: each type's curvature drift with depth is offset by its concentration
drifting the other way, and this is a property of *which* concentration profile goes with *which*
type, not an artifact of how the components were defined.

Caveat on resolution: with six types there are only 720 permutations, so the attainable p is coarse
and these are not extreme values. The result is consistent across all four seeds, which by rule 32
is robustness to initialisation rather than four independent tests.

**The physical statement is unchanged**, and the per-type numbers from 228 (`mlp.proj` direct
+0.0324 offset by −0.0275 through concentration, leaving +0.0049) remain accurate as a description.
What changed is that the cancellation is now supported by a test of the pairing rather than by a
correlation that any decomposition would have produced.

## Rule 35 sweep: the g-family withdrawal was right, but its sign was never explained (2026-09-05)

Rule 35 (do not treat a relationship between a quantity and its own residual as evidence) was applied
to every surviving claim in this file. **One structural class is affected, and it was already
withdrawn** — but the sweep found something the withdrawal missed.

**The sweep.** Rule 35 fires only when the tested quantity is a component of the residual. REQ-047's
fields (`d_frob`, `d_cv`, `d_eff_rank`, `da_cos_mean`, `grad_rank1_frac`) come from a different
archive and are external to `log C`; rule 35 does not fire on them. The one class where it does is
the **components of `g`**, since `log C = log lam − 2 log g` and
`log g = log|a|_F + log|d|_F + log(align_ratio)` exactly. Simulating a null in which curvature is
**completely independent** of the g-components, and correlating `log|a|_F` with the C residual, gives
**−0.551** — a strong correlation from construction alone. Iteration 214 already withdrew this
family on the equivalent ground ("no decomposition of `g` can serve as an independent predictor of a
residual defined using `g`"), so no standing claim needs revising. Rule 35 is the general form of a
defect this campaign had already identified.

**What the withdrawal never explained: the sign.** Construction predicts **−0.55**; the recorded
values are **positive** (`d_frob` +0.224, `a_frob` +0.116). A sign opposite to the artifact is
informative — it means something real is present and is *fighting* the construction term.

The algebra says where the crossing is. With `log lam = k·log|d|_F + noise`, the residual's
dependence on `log|d|_F` is `(k − 2)`, so the correlation is positive **only if k > 2**. Simulation
confirms the zero-crossing lands at exactly k = 2, and reproduces the observed +0.224 at k ≈ 2.6.

## C rises with g: the between-matrix variation is not a gauge rescaling (2026-09-05)

That prediction is directly measurable from REQ-048 alone — no join, no REQ-047, no g-decomposition —
as `k = d(log lam)/d(log g)`, saturated in type and block, cluster-robust by block.

| specification | k | per seed | t vs k = 2 |
|---|---|---|---|
| type + block | **3.173** | 3.28, 3.27, 2.99, 3.15 | **+17.13** |
| type + block + `log n_eff_bulk` (lam-free) | **2.569** | 2.55, 2.82, 2.64, 2.26 | **+4.88** |

Both are above 2, and the concentration-controlled value (2.57) matches the k ≈ 2.6 that the observed
+0.224 correlation independently implies. Two routes agree.

**Why k = 2 is the meaningful reference.** The campaign's gauge theorem says any scalar multiplying a
matrix's whole contribution cancels exactly in `lam/g²`: rescale a gradient by c and `lam` moves by
c², `g` by c, leaving C invariant — which is **k = 2 exactly**. The measured k > 2 therefore says the
between-matrix variation in this network is **not** a pure gauge rescaling. There is real shape
variation, and it shows up as C co-varying with g:

| specification | d(log C)/d(log g) | robust se | t |
|---|---|---|---|
| type + block | **+1.173** | 0.216 | +17.13 |
| type + block + `n_eff_bulk` | **+0.569** | 0.391 | +4.88 |

The effect is substantial: residual sd(`log g`) is ~0.08 dex, giving a C swing of **0.17–0.21 dex**
across ±1 sd, with **partial R² 0.30–0.39** per seed at fixed type and depth.

**The limit, stated plainly.** `g` is not exogenous. It is measured at the same step as `lam`, and
both respond to the same training dynamics; this says equilibrium C and gradient norm **co-vary
across matrices**, not that raising `g` would raise C. It also does not resurrect any g-derived
predictor — the artifact and the signal remain superimposed in the same number, which is exactly why
`d_frob` and its relatives stay withdrawn. Separating them needs an experiment that varies the
gradient side causally, which is **REQ-051**.

**This sharpens REQ-051's value.** That request was previously justified as removing a join. It now
has a specific quantitative target: measure whether the causal elasticity of `lam` to `g` is also
above 2, or whether the observational k > 2 is confounded by the joint response of both quantities to
training dynamics.

## The gradient norm tracks spectral concentration, not curvature size (2026-09-05)

Iteration 230 measured `k = d(log lam)/d(log g) = 3.173` against a gauge-invariant value of 2, and
concluded the between-matrix variation is not a rescaling. This iteration asks **what breaks the
gauge**, and the answer arrives as an ordered ladder.

**First, the gauge reference re-derived, because the obvious derivation is wrong.** Scaling a
matrix's *contribution to the loss* by c scales the gradient by c and the Hessian block by **c¹**
(the Hessian is a second derivative of a term linear in c), which would leave `C = lam/g²` varying as
1/c — not invariant. The campaign's theorem is about a **reparametrisation**: write `W = c·V`, then
`dL/dV = c·dL/dW` so `g ~ c`, and `d²L/dV² = c²·d²L/dW²` so `lam ~ c²`. That is the gauge under
which C is invariant, and it fixes **every** spectral moment's elasticity wrt `log g` at **+2**.

**All three moments deviate, and they deviate in order.** Saturated in type and block:

| moment | gauge value | measured | deviation | per seed |
|---|---|---|---|---|
| `log trace(H)` (sum of all eigenvalues) | 2.00 | **+1.107** | **−0.893** | 1.12, 1.07, 1.14, 1.10 |
| `log sqrt(trace(H²))` (top-few weighted) | 2.00 | **+2.424** | +0.424 | 2.39, 2.49, 2.28, 2.53 |
| `log lam_top` (the single largest) | 2.00 | **+3.173** | +1.173 | 3.28, 3.27, 2.99, 3.15 |

The elasticity **rises monotonically with how top-weighted the moment is**. The differences are what
the claim asserts, and they are strong within seeds — the right standard under rule 32, since the
three moments are measured on the same matrices and are highly correlated:

| difference | mean | seed sd | t | sign |
|---|---|---|---|---|
| `lam − trace` | **+2.066** | 0.157 | **+26.3** | 4/4 |
| `lam − rms` | +0.749 | 0.113 | +13.3 | 4/4 |
| `rms − trace` | +1.317 | 0.134 | +19.7 | 4/4 |

**Permutation null:** shuffling `log g` within seed and recomputing gives a `lam − trace` spread
centred at +0.002 with sd 0.071 against an observed **+2.066** — **p = 0.0000** over 2000 draws.

**The physical statement.** `trace(H)` is the sum over all eigenvalues, `sqrt(trace(H²))` is dominated
by the largest few, `lam_top` is the single largest. An elasticity that rises across that sequence
means **matrices with larger gradients hold their curvature in fewer directions**. Total curvature
barely responds to `g` (+1.11, well *below* the gauge value), while the top eigenvalue responds
strongly (+3.17). The gradient norm tracks **concentration**, not size.

**Why this matters beyond restating bands 71/75.** Those established that concentration explains the
depth profile of C, using `n_eff` as a predictor. This arrives at concentration from an entirely
different direction — the gauge structure of `C = lam/g²` and the gradient norm — and uses no
concentration statistic as a regressor at all. The three moments are separate Hutchinson estimates;
`lam_top` is the Lanczos quantity, and it appears here as an **outcome**, never as a predictor, so
the hard rule is not engaged.

**The limit, unchanged from iteration 230.** `g` is not exogenous; it is measured at the same step as
the curvature moments and both respond to the same training dynamics. This is co-variation across
matrices, not a demonstration that changing `g` would move the spectrum. **REQ-051** remains the only
queued instrument that can separate them, and this result gives it a second registered target
alongside the causal `k`: whether the **moment ladder** itself survives causal variation of the
gradient side, or collapses toward the gauge value of +2 for all three moments.

## The moment ladder survives independent-probe testing and errors-in-variables (2026-09-05)

Iteration 231's ladder — elasticities wrt `log g` rising from `trace(H)` to `sqrt(trace(H²))` to
`lam_top`, bracketing the gauge value of +2 — was stress-tested against the one failure mode it had
not faced. It survives, and one premise stated in setting up the test was wrong and is corrected.

**Correction: `g` is not probe-free.** Iteration 232 assumed `gradient_block_norm` is a norm of a
stored gradient and therefore measured exactly. It is not: across the three probe repeats of the same
matrix, `log10 g` has a **within-matrix sd of 0.115 dex** against a between-matrix sd of 0.241.
Reliability is **0.786 for a single repeat**, 0.917 for the three-repeat average. So `g` is evaluated
on a probe batch, errors-in-variables applies to the regressor, and the raw elasticities are
attenuated toward zero.

**The decisive test: independent probe sets.** All three moments and `g` come from the same probe
draw within a repeat, so shared probe noise could in principle manufacture an ordering. Taking `g`
from one repeat and the moments from a **different** repeat makes the two error terms independent —
shared noise cannot survive it. All six ordered pairs:

| `g` / moments | trace | rms | lam | spread |
|---|---|---|---|---|
| 060/100 | +1.075 | +2.147 | +2.970 | +1.896 |
| 060/170 | +0.940 | +1.875 | +2.653 | +1.713 |
| 100/060 | +1.094 | +2.429 | +3.201 | +2.107 |
| 100/170 | +0.949 | +2.052 | +2.682 | +1.733 |
| 170/060 | +1.011 | +2.310 | +2.967 | +1.956 |
| 170/100 | +1.028 | +2.229 | +2.853 | +1.825 |
| **mean** | **+1.016** | **+2.174** | **+2.888** | **+1.872** |

The ordering holds in **6 of 6** independent-error configurations. The ladder is not a probe artifact.

**Disattenuated** by the measured single-repeat reliability: trace **+1.292**, rms **+2.765**, lam
**+3.673**, against a gauge value of +2.000 for all three.

**Robustness to the reliability estimate itself.** Disattenuation divides all three elasticities by
the same rho, so:

- the **ordering** trace < rms < lam is **rho-invariant by construction**;
- `lam` (+2.888) and `rms` (+2.174) exceed the gauge value **with no correction at all**, so those
  conclusions hold for any rho ≤ 1;
- `trace` (+1.016) would need rho ≤ **0.508** to reach +2; the measured value is 0.786.

Correction moves slopes *away* from zero, so it **widens** the ladder — the raw numbers were the
conservative ones, and the bracketing of the gauge value is not an artifact of the correction.

**Standing status.** The ladder has now been tested against shared probe noise (independent-probe
cross-repeat, 6/6), errors-in-variables (disattenuated, conclusions rho-invariant), a permutation
null (p = 0.0000, iteration 231), and within-seed effect sizes (t = +13 to +26, iteration 231). The
limit is unchanged and is not statistical: **`g` is not exogenous**, so this remains co-variation
across matrices. REQ-051 is still the only queued instrument that can make it causal.

## The gauge violation survives an actual intervention, but is far smaller than observed (2026-09-05)

The moment ladder's limit has been that `g` is not exogenous. That limit is partially liftable from
committed data, which iterations 230-232 assumed it was not.

**What is and is not testable.** REQ-048 is the **only** archive carrying `trace_est` and
`trace_sq_est` — checked across every local archive (REQ-036, 037, 045, 047, and Arm A all carry
`gradient_block_norm` and none carry the Hutchinson moments). So the **full ladder** cannot be tested
under intervention. But its top rung, `k = d(log lam)/d(log g)`, needs only `top_eigenvalue` and
`gradient_block_norm`, and **REQ-045 has both under a real per-matrix LR intervention** (3 arms,
72 matrices, deliberate multipliers `m_i`).

**Within-matrix estimate, matrix fixed effects, cluster-robust by matrix (n = 216):**

| estimate | k | se | t vs 2 |
|---|---|---|---|
| **intervention (within-matrix, REQ-045)** | **+2.237** | 0.086 | **+2.77** |
| observational (between-matrix, REQ-045) | +2.415 | — | — |
| observational (between-matrix, REQ-048) | +3.173 | — | — |

**The gauge violation survives an intervention** — but at +2.237, not +3.173.

**Where the gap comes from.** Decomposing the difference between the intervention estimate and the
headline observational one: **81% is dataset** (REQ-045's own observational k is +2.415, already well
below REQ-048's +3.173) and only **19% is estimator** (within +2.237 vs between +2.415 on the same
data). The observational excess of +1.17 above gauge is therefore **not** a causal quantity, and
iteration 231's number should not be quoted as one.

**It is not driven by one arm pair.** The three pairwise arm contrasts give **+2.095, +2.390,
+2.101** — consistent.

**It is concentrated, not uniform.** Within-matrix k per type:

| type | k | se | t vs 2 |
|---|---|---|---|
| **`mlp.proj`** | **+2.504** | 0.083 | **+6.06** |
| `attn.v` | +2.421 | 0.287 | +1.47 |
| `attn.q` | +2.337 | 0.209 | +1.61 |
| `mlp.fc` | +2.122 | 0.218 | +0.56 |
| `attn.k` | +1.895 | 0.287 | −0.37 |
| **`attn.proj`** | **+1.397** | 0.258 | **−2.34** |

Only `mlp.proj` violates the gauge decisively; `attn.proj` sits **below** it; four types are
indistinguishable from 2. The broad, uniform gauge violation suggested by the observational estimate
is not what the intervention shows.

**Attenuation: it cannot rescue the observational number.** REQ-045 has one probe per arm, so its
reliability is not estimable internally. Using REQ-048's per-probe noise in `log g` (0.115 dex) as a
guide, the within-matrix signal here is only **0.0948 dex** — a signal-to-noise ratio of **0.82**,
i.e. the noise is *larger* than the signal. If that transfers, the intervention k is heavily
attenuated. But attenuation biases toward **zero**, and 2 is not zero, so a heavily attenuated
+2.237 implies a true value **above** 2, never below.

**No corrected value is claimed.** Clean disattenuation assumes the regressor's error is independent
of the outcome's; here `lam` and `g` are measured on the **same probe batch**, so their errors are
correlated and can bias the slope in either direction. What survives without that assumption: the
sign of the deviation is positive pooled and in 4 of 6 types, the three arm contrasts agree, and
`mlp.proj` is decisively above gauge.

**Consequence for REQ-051.** Its registered target stands and gains a benchmark: the causal k should
be compared against **+2.237** from this pilot, not against the observational +3.173. REQ-051's
design — multiple probe repeats and a proper LR ladder — is exactly what this 3-arm, 1-seed,
1-probe pilot cannot deliver, and the signal-to-noise ratio of 0.82 found here is the concrete reason
its **probe-repeat requirement matters for the gradient side too**, not only for curvature.

## Under a per-type LR intervention the gauge violation largely disappears (2026-09-05)

Rule 39's archive audit continued and found two more archives carrying `top_eigenvalue` and
`gradient_block_norm`: **REQ-036** (5 arms, the per-type LR design this campaign must validate) and
**REQ-035 Arm A** (4 seeds). Both were tested for the intervention `k = d(log lam)/d(log g)`. The
results substantially qualify iterations 231 and 233.

**REQ-036, within-matrix across five arms (n = 360, 72 matrices, cluster-robust by matrix):**

| estimate | k | se | t vs 2 |
|---|---|---|---|
| **REQ-036 (per-type LR intervention)** | **+1.922** | 0.077 | **−1.01** |
| REQ-045 (per-matrix LR intervention) | +2.237 | 0.086 | +2.77 |
| REQ-048 (observational) | +3.173 | — | — |

**On the per-type LR design, k is indistinguishable from the gauge value of 2.** It is not driven by
any single arm — dropping each in turn gives +1.853 to +1.975 — and all four control-vs-treatment
contrasts agree (`a2_pertype` +1.643, `a3_endcap` +2.156, `a4_antirule` +1.790, `a5_polar` +1.875).

**Arm A's apparent n=4 confirmation is withdrawn before it was used.** Arm A's four seeds initially
gave a within-matrix k of **+2.843**, above 2 in 4/4. That is **not an intervention estimate**: Arm
A's "arms" are the `s060/s100/s170` labels, which iteration 224 established are **probe repeats of a
single checkpoint**, not manipulations. Its within-matrix contrast therefore contains no experimental
variation in `g` — the within-matrix sd of `log g` is **0.078 dex** against REQ-048's per-probe noise
of 0.115 dex, i.e. the same magnitude. `lam` and `g` share a probe batch, so correlated errors
inflate the slope. Consistently, Arm A's cross-repeat (independent-error) estimate is **+2.837**,
matching the *observational* +3.17 family rather than any intervention value. Arm A cannot serve as
the n=4 seed check for k.

## `mlp.proj` is the one type that violates the gauge in both interventions (2026-09-05)

The two genuine interventions disagree at the pooled level (+2.237 vs +1.922), but they agree on
where the violation lives. Within-matrix k per type:

| type | REQ-045 (per-matrix LR) | REQ-036 (per-type LR) |
|---|---|---|
| **`mlp.proj`** | **+2.504** (t vs 2 = +6.06) | **+2.287** (t vs 2 = +3.32) |
| `attn.v` | +2.421 (+1.47) | +1.409 (−3.44) |
| `attn.q` | +2.337 (+1.61) | +1.296 (−2.68) |
| `mlp.fc` | +2.122 (+0.56) | +1.737 (−0.43) |
| `attn.k` | +1.895 (−0.37) | +1.453 (−1.24) |
| `attn.proj` | +1.397 (−2.34) | +1.910 (−1.01) |

**`mlp.proj` is the only type above the gauge value in both designs**, and the only one significantly
so in either. Every other type is at or below 2 in REQ-036, and only `attn.proj` is decisively below
in REQ-045. Two independent LR designs, different manipulations, same answer for this one type.

This is the cross-design replication that Arm A could not supply. It is **not** a four-seed check —
REQ-036 and REQ-045 are each single-seed — so it establishes reproducibility across *designs*, not
across initialisations.

**What must now be said about the observational ladder.** Iteration 231's +3.173, and the excess of
+1.17 above gauge it implied, is **observational and largely not causal**. Iteration 233 showed 81%
of the gap to REQ-045 is dataset rather than estimator; REQ-036 now shows that under a per-type LR
manipulation the pooled violation is absent altogether. The surviving causal claim is narrow: **one
matrix type, `mlp.proj`, has curvature that responds to its gradient more steeply than a
reparametrisation would produce.** The broad statement that "matrices with larger gradients hold
curvature in fewer directions" remains supported **observationally** (iteration 232's independent-probe
tests stand) but is not established causally.

**For REQ-051.** Compare its causal k against this pair of benchmarks, not against +3.173: **+2.237**
under per-matrix LR and **+1.922** under per-type LR. The divergence between them is itself a target
— per-matrix multipliers break confounds that per-type rules cannot, since a per-type rule moves all
twelve matrices of a type together. Report per-type k, and specifically whether `mlp.proj` reproduces
above 2.

## A third intervention: `mlp.proj`'s gauge violation does not replicate in levels (2026-09-05)

REQ-037 varies **batch size** (0.5x, 1x, 2x) — a different physical lever from the two LR designs —
and carries `top_eigenvalue` and `gradient_block_norm`. It is an independent test of iteration 234's
`mlp.proj` claim, and the claim **fails in levels**.

| design | lever | pooled k | `mlp.proj` k |
|---|---|---|---|
| REQ-045 | per-matrix LR | +2.237 | **+2.504** (t vs 2 = +6.06) |
| REQ-036 | per-type LR | +1.922 | **+2.287** (t vs 2 = +3.32) |
| **REQ-037** | **batch size** | **+0.557** | **+1.432** (t vs 2 = **−2.61**) |

Under batch variation `mlp.proj` sits **below** the gauge value, not above it. Every type is far
below 2 here, and the three pairwise arm contrasts agree (+0.003, +1.034, +0.585).

**An argument I made and then had to retract, within this iteration.** I proposed dismissing REQ-037
on the ground that batch size moves `g` purely through mini-batch sampling noise — a channel that
cannot move `lam` — which predicts `g ~ 1/sqrt(B)`, i.e. **larger** `g` at **smaller** batch. The data
show the opposite: mean `log10 g` is 3.771 at 0.5x, 3.822 at 1x, 3.873 at 2x. `g` **grows** with
batch; observed/predicted = **−0.34**. The sampling-noise story is wrong and **cannot** be used to set
REQ-037 aside. Recorded because it was the convenient conclusion and it did not survive its own test.

**What can honestly be said about REQ-037's leverage.** Across a 4x batch change, `log g` moves 0.102
dex while `log lam` moves only 0.039 dex. Two readings remain open and this archive cannot separate
them: batch genuinely does not move curvature much (k is low for a real reason), or batch moves `g`
through a channel weakly coupled to curvature (k is attenuated for an instrument reason). REQ-037
lacks `trace_est`/`trace_sq_est`, so the spectral decomposition that would distinguish them is
unavailable. **No internal reliability estimate exists either:** the 8 "rank shards" are shards of one
distributed measurement, not repeats, so within-arm spread is undefined (it computes as NaN).

## What survives is a rank claim, not a level claim (2026-09-05)

The three designs disagree sharply on levels and their overall orderings are nearly unrelated —
Kendall tau between designs is **−0.07, +0.33, +0.33**. Against that background, one fact holds:

| type | REQ-045 | REQ-036 | REQ-037 | rank in each |
|---|---|---|---|---|
| **`mlp.proj`** | +2.504 | +2.287 | +1.432 | **1 / 1 / 1** |
| `mlp.fc` | +2.122 | +1.737 | +0.901 | 4 / 3 / 2 |
| `attn.q` | +2.337 | +1.296 | +0.240 | 3 / 6 / 3 |
| `attn.k` | +1.895 | +1.453 | +0.136 | 5 / 4 / 4 |
| `attn.proj` | +1.397 | +1.910 | −0.021 | 6 / 2 / 5 |
| `attn.v` | +2.421 | +1.409 | −0.607 | 2 / 5 / 6 |

**`mlp.proj` has the highest k of the six types in all three designs**, across two different physical
levers. Priced honestly: `mlp.proj` was selected *because* it topped REQ-045, so the conditional
probability is P(1st in the other two | 1st in the first) = (1/6)² = **0.028** under independent
ranks — and the near-zero tau values make independence a roughly fair, mildly conservative
assumption.

**The claim is therefore downgraded, not withdrawn.** Iteration 234 stated that `mlp.proj` "violates
the gauge in both interventions". The correct statement after three designs is weaker: **`mlp.proj`
is consistently the matrix type whose curvature responds most steeply to its gradient**, p ≈ 0.03
from a single post-hoc selection, with **no seed replication behind any of the three designs**.
Whether it exceeds the gauge value of 2 is design-dependent — yes under both LR levers, no under
batch.

**Status of the n=4 seed check.** It cannot be done for k from committed data. Every archive
carrying `top_eigenvalue` and `gradient_block_norm` under an intervention is **single-seed**
(REQ-036, REQ-045, REQ-037); REQ-035 Arm A has four seeds but its "arms" are probe repeats, not
manipulations (iteration 234). REQ-051 remains the only route to a seed-replicated causal k.

## The k spread across designs is not an attenuation artifact — a fitted explanation withdrawn (2026-09-05)

Three interventions give pooled within-matrix `k` of +2.237 (REQ-045), +1.922 (REQ-036) and +0.557
(REQ-037). This iteration tested a specific, measurable explanation for that spread and **refuted
it**, including the version that initially looked convincing.

**The candidate.** If part of each design's observed `d(log g)` is measurement noise rather than
manipulation, the within-matrix slope is attenuated by `s²/(s²+e²)`. Weaker manipulations would then
give smaller `k`. The manipulation strengths are measurable and the ordering matches:

| design | sd `d(log g)` | k |
|---|---|---|
| REQ-045 | 0.0948 | +2.237 |
| REQ-036 | 0.0823 | +1.922 |
| REQ-037 | 0.0487 | +0.557 |

Fitting `observed_k = true_k · s²/(s²+e²)` gives `true_k = 2.76`, noise 0.0435 dex, and an **RMS
residual of 0.051** — an apparently excellent fit across all three designs.

**Why that fit is not evidence.** With three points and two parameters there is one residual degree
of freedom. A straight line in `sd(dg)` fits nearly as well (RMS 0.064) with the same parameter
count, so the in-sample fit distinguishes nothing: it shows only that `k` rises with manipulation
strength, which the raw ordering already showed.

**The out-of-sample test, and both models fail it.** Arm subsets within each design vary the
manipulation strength *without changing the estimator*, giving 34 within-matrix estimates on the same
scale. The curve was fitted on three pooled points and checked against all of them:

| model | in-sample RMS (3 points) | **out-of-sample RMS (34 subsets)** | max abs error |
|---|---|---|---|
| attenuation | 0.051 | **0.654** | 2.089 |
| straight line | 0.064 | **0.695** | 1.750 |

Against a k range of roughly 0 to 2.4, both are useless out of sample. The attenuation model also
predicts negative signal variance (returning NaN) for four low-`sd` subsets — it is not merely
inaccurate there but inadmissible.

**What actually explains k: the design, not the manipulation.**

| design | subsets | mean k | sd k | sd(dg) range |
|---|---|---|---|---|
| REQ-045 | 4 | +2.206 | 0.139 | 0.0663–0.0983 |
| REQ-036 | 26 | +1.882 | 0.163 | 0.0274–0.1022 |
| REQ-037 | 4 | +0.545 | 0.422 | 0.0317–0.0575 |

**Between designs: 85.0% of the variance. Within design: 15.0%.** REQ-036's 26 subsets stay at
+1.88 ± 0.16 across a **3.7× range** of manipulation strength. Within-design correlations between
`sd(dg)` and `k` are +0.53 (REQ-036, 26 subsets), +0.88 (REQ-045, 4 subsets) and +0.06 (REQ-037,
4 subsets) — not the uniform strong relationship attenuation requires.

**Status: the divergence stands unexplained.** It is a real property of the three designs, not an
artifact of weak manipulation, and it is the reason `k`'s *level* cannot yet be quoted as a physical
constant. The surviving cross-design claim remains the **rank** one from iteration 235 (`mlp.proj`
highest in all three, p ≈ 0.028), which is unaffected: ranks within a design do not depend on the
design's overall level.

**Method note.** This is the second time in this campaign that a model fitting a handful of aggregate
points collapsed when tested at finer grain — the first was iteration 208's join error. A fit with
one residual degree of freedom should be treated as a hypothesis to test, never as a result.

## The `mlp.proj` rank claim repriced: p = 0.053, not 0.028 (2026-09-05)

Iteration 235 reported that `mlp.proj` has the highest per-type `k` in all three intervention designs
and priced it at (1/6)² = **0.028**, conditioning on its post-hoc selection but assuming each design's
ranking is a clean independent draw. **That assumption is wrong, and the p-value is corrected upward.**

**A design's own ranking is not stable.** Recomputing per-type `k` on every arm subset within each
design shows how often `mlp.proj` actually leads:

| design | arm subsets | `mlp.proj` mean rank | sd | % of subsets ranked 1st |
|---|---|---|---|---|
| REQ-045 | 4 | 1.75 | 0.96 | **50%** |
| REQ-036 | 26 | 1.42 | 0.81 | **73%** |
| REQ-037 | 4 | 1.00 | 0.00 | **100%** |

In REQ-045 three different types top at least one subset; in REQ-036, five do. A single fit per design
is therefore noisier than a clean draw, and 1/36 overstated the evidence.

**Repriced with a permutation null.** Shuffling the type labels once and applying that assignment to
all three designs — the correct null, since the six types are the same physical objects in every
design, so a genuinely special type is special everywhere:

| statistic | observed | null mean | p |
|---|---|---|---|
| **A: best mean rank across designs** (pre-specified) | 1.000 | 1.990 | **0.0525** |
| B: best mean z-score of `k` (post-hoc) | +1.358 | 0.810 | 0.0255 |

**Statistic A is the result: p = 0.053.** It is the statistic that follows directly from iteration
235's claim. Statistic B is more powerful — it uses the size of each gap rather than only the
ordering, and z-scoring within design makes it immune to the level divergence of iteration 236 — but
it was **chosen after seeing A's result**, so its p-value is not a valid confirmatory test and is
recorded for completeness only.

**Per-type means across the three designs** (z-scored within design, so levels do not contribute):

| type | mean z | mean rank |
|---|---|---|
| **`mlp.proj`** | **+1.358** | 1.00 |
| `mlp.fc` | +0.314 | 3.00 |
| `attn.q` | −0.214 | 4.00 |
| `attn.v` | −0.438 | 4.33 |
| `attn.k` | −0.477 | 4.33 |
| `attn.proj` | −0.543 | 4.33 |

`mlp.proj` sits well clear of the next type, and the two MLP matrices occupy the top two places in
the mean while the four attention matrices fill the bottom four. That is the substantive picture;
the formal p sits at the margin of conventional significance.

**Current status of the claim.** `mlp.proj` is *plausibly* the matrix type whose curvature responds
most steeply to its gradient — p = 0.053 on the pre-specified statistic, from three single-seed
designs whose internal rankings are unstable. It is **not** established. Iteration 235's 0.028 is
superseded.

**This is what REQ-051 is for.** Four seeds would convert an unstable single fit per design into a
seed-replicated estimate, which is precisely the deficiency identified here. The pre-registered
prediction stands as written: does `mlp.proj` again have the highest per-type `k`, judged within each
seed?

## The ranking churn is rivals moving, not `mlp.proj` — and a stronger test replaces the rank claim (2026-09-05)

Iteration 237 weakened the `mlp.proj` claim to p = 0.053 because a design's own per-type ranking is
unstable across arm subsets. This iteration asks **where that instability lives**, and the answer
inverts the reading.

**`mlp.proj` is the most stable type in the design.** Across REQ-036's 26 arm subsets:

| type | mean k | sd across subsets | mean SE | pairwise heterogeneity Q (9 dof) |
|---|---|---|---|---|
| **`mlp.proj`** | +2.301 | **0.201** | 0.147 | 16.6 — consistent with one value |
| `attn.proj` | +1.797 | 0.386 | 0.235 | 16.0 — consistent |
| `attn.q` | +1.134 | 0.582 | 0.321 | **26.6 — heterogeneous** |
| `attn.v` | +1.405 | 0.586 | 0.313 | **55.6 — heterogeneous** |
| `attn.k` | +1.254 | 0.659 | 0.530 | **22.1 — heterogeneous** |
| `mlp.fc` | +1.741 | 0.745 | 1.122 | 7.2 — consistent |

**`mlp.proj` has the smallest spread of any type**, roughly half the next smallest.

**The churn is entirely rivals spiking.** In the 7 of 26 subsets where `mlp.proj` is not first:

- `mlp.proj`'s k is **+2.287** full-design, **+2.310** when it wins, **+2.275** when it loses — a
  drop of **0.012**.
- The winning rival's k exceeds its own full-design value by **+1.143** on average.

So the ranking instability is not evidence against `mlp.proj`; it is volatility in the attention
types' estimates.

## A precision-weighted interaction replaces the rank statistic (2026-09-05)

The stability asymmetry cuts both ways: a stable estimate can top an average ranking even under a
common true k, because ranks ignore precision. The fix is to abandon ranks and test the hypothesis
directly — fit `d(log lam) = a + b·d(log g) + c·[mlp.proj]·d(log g)` per design, where `c` is
`mlp.proj`'s extra elasticity over the other five types combined.

| design | b (other five) | **c (mlp.proj extra)** | se | t | p |
|---|---|---|---|---|---|
| REQ-045 | +2.037 | **+0.467** | 0.167 | +2.79 | 0.0053 |
| REQ-036 | +1.751 | **+0.536** | 0.134 | +3.99 | 0.0001 |
| REQ-037 | +0.283 | **+1.149** | 0.274 | +4.19 | <0.0001 |

**Significant in each design separately.** Fixed-effect pooling gives **c = +0.591, se 0.098,
t = +6.03**, with heterogeneity Q = 4.85 on 2 dof (consistent). Clustering by **block** rather than
by matrix — 72 matrices sit inside 12 blocks — gives **c = +0.578, se 0.075, t = +7.72**.

**Placebo on every type**, the null distribution made concrete (block-clustered, pooled):

| type | REQ-045 | REQ-036 | REQ-037 | pooled c | t |
|---|---|---|---|---|---|
| **`mlp.proj`** | **+0.467** | **+0.536** | **+1.149** | **+0.578** | **+7.72** |
| `mlp.fc` | −0.143 | −0.187 | +0.454 | −0.002 | −0.01 |
| `attn.proj` | −0.944 | −0.024 | −0.602 | −0.171 | −1.44 |
| `attn.q` | +0.109 | −0.705 | −0.387 | −0.318 | −2.33 |
| `attn.v` | +0.205 | −0.533 | −1.313 | −0.442 | −4.06 |
| `attn.k` | −0.372 | −0.490 | −0.515 | −0.460 | −2.07 |

**`mlp.proj` is the only type positive in all three designs**, and the only one with a large positive
pooled interaction. The specification does not manufacture positive interactions.

**Selection caveat, stated plainly.** The interaction statistic was chosen **after** the rank
statistic returned p = 0.053 (rule 43). It is a better statistic on its merits — it uses magnitudes
and standard errors, it is immune to rank churn and to the stability asymmetry, and the placebo
provides its own null — but it was not pre-specified for this data. Treat it as a strong,
well-controlled description rather than a confirmatory test.

**Status.** `mlp.proj`'s curvature responds to its gradient about **+0.58 dex per dex more steeply**
than the other five matrix types, consistently across three interventions with different levers.
This survives block clustering and a six-way placebo. It still rests on **three single-seed designs**;
REQ-051's four seeds remain the confirmatory test, and its registered prediction should now be scored
on **this interaction**, not on the rank.

## Why `mlp.proj`? One hypothesis refuted, two not separable from committed data (2026-09-05)

The `mlp.proj` interaction (c = +0.578) is a description, not a mechanism. Three structural facts
distinguish `mlp.proj` from the other five matrix types, and they make different predictions.

**H-A — residual writer: REFUTED.** `mlp.proj` writes to the residual stream, but so does
`attn.proj`, so H-A predicts `attn.proj` should also show c > 0. Measured `attn.proj` c = **−0.171**
(pooled, block-clustered). Grouping the two writers together gives c = +0.395 (t = +3.36) — *lower*
than `mlp.proj` alone (+0.578, t = +7.72), i.e. adding `attn.proj` dilutes the effect rather than
strengthening it. The residual-writer role does not explain this, consistent with the earlier
withdrawal of that grouping elsewhere in this file.

**H-B (ReLU² input) and H-C (fan-in shape) cannot be separated by grouping**, because in this
architecture they select exactly the same matrix. Shapes from the committed `shape` field:

| type | shape (out, in) | fan-in / fan-out |
|---|---|---|
| `attn.{q,k,v,proj}` | (768, 768) | 1.00 |
| `mlp.fc` | (3072, 768) | 0.25 |
| **`mlp.proj`** | **(768, 3072)** | **4.00** |

`mlp.proj` is the only matrix that is *wide* (fan-in > fan-out) **and** the only one whose input is
the squared-ReLU expansion. Selecting on either fact selects the same rows. Note `mlp.fc` has the
same 4× dimension ratio transposed and shows c = **−0.002**, so the effect is not about the ratio
alone — but orientation and nonlinearity-position are still confounded.

**A within-`mlp.proj` discriminator was attempted and returned a null.** Shape is identical at every
depth and cannot vary; the ReLU² expansion's activation statistics do change with depth. So a
depth-dependent effect would favour H-B. Measured d(k)/d(block) within `mlp.proj`: +0.0669 (REQ-045),
−0.0144 (REQ-036), −0.1069 (REQ-037), pooled **+0.0182, se 0.0166, p = 0.27**, signs inconsistent
across designs.

**The null is weak, and is reported as weak.** At 80% power this test detects |d(k)/d(block)| ≥ 0.047
per block, i.e. a swing of **0.511 in k** across twelve blocks — **88% of the entire c effect**. It
therefore rules out a depth dependence comparable in size to the effect itself, and rules out nothing
smaller. Since H-B predicts no particular size, this does **not** favour H-C; both hypotheses remain
live. **No mechanism for `mlp.proj` is established.**

**What would separate them** (registered as REQ-053 below, as a follow-on rather than a substitute
for the open queue):

- **Expansion-ratio arm.** An MLP at 2× or 8× instead of 4×, holding ReLU² fixed. H-C predicts c
  scales with the ratio; H-B predicts c is unchanged.
- **Nonlinearity arm.** GELU in place of ReLU² at the same 4× shape. H-B predicts c changes; H-C
  predicts it does not.

## The `mlp.proj` elasticity result cannot carry the depth question (2026-09-05)

Five iterations of work on `k` and `mlp.proj` produced a gradient-side type effect. Before extending
it further, this iteration asks the accounting question the campaign's charter demands: **how much of
C's depth profile can it explain?** The answer is: very little, and this is recorded as a limit on
that whole line of work.

**Upper bound.** A type-specific elasticity difference reaches the depth profile only through the
depth variation of `g` itself, which is small:

| seed | sd of block-mean `log g` | sd of block-mean `log C` | generous bound `0.578 × sd(log g)` | as % of C's depth spread |
|---|---|---|---|---|
| 0 | 0.0526 | 0.1486 | 0.0304 | **20.5%** |
| 1 | 0.0420 | 0.1373 | 0.0243 | 17.7% |
| 2 | 0.0505 | 0.1562 | 0.0292 | 18.7% |
| 3 | 0.0582 | 0.1971 | 0.0336 | 17.1% |

And that bound is generous twice over: it applies to **one type of six**, and only if `mlp.proj`'s
depth pattern aligned with the average (measured corr +0.36 to +0.72).

**Direct test — remove `mlp.proj` entirely.** The depth profile is essentially unchanged: argmin
stays at 6, 6, 6, 7; correlation with the full profile is **+0.92 to +0.96**. Leave-one-type-out for
all six types gives correlations of **0.944 to 0.993** — `mlp.proj` is the most influential type, and
even it does not carry the profile. No single matrix type does, consistent with the bowl being
positional rather than type-driven.

## C's depth profile is what survives a curvature–gradient cancellation (2026-09-05)

Decomposing the block-mean profile exactly (`log C = log lam − 2 log g`) produced a finding not
previously recorded:

| seed | sd `log C` | sd `log lam` | sd `2 log g` | corr(lam, g) across depth |
|---|---|---|---|---|
| 0 | 0.1423 | **0.1773** | 0.1008 | +0.597 |
| 1 | 0.1314 | **0.1866** | 0.0805 | +0.800 |
| 2 | 0.1495 | **0.1953** | 0.0966 | +0.665 |
| 3 | 0.1887 | **0.2332** | 0.1115 | +0.600 |

**`sd(log lam)` exceeds `sd(log C)` in every seed.** The curvature profile is *larger* than the C
profile, and the gradient profile partially cancels it — `g` rises with `lam` across depth and enters
with a minus sign. C's depth profile is the residue of that cancellation.

**Tested with the pairing permutation** (rule 35's instrument, holding each block's `lam` and `g`
fixed and permuting only which `g` profile pairs with which `lam` profile):

| seed | sd(log C) observed | null mean | p |
|---|---|---|---|
| 0 | 0.1423 | 0.2026 | **0.0154** |
| 1 | 0.1314 | 0.2019 | **0.0018** |
| 2 | 0.1495 | 0.2165 | **0.0089** |
| 3 | 0.1887 | 0.2563 | **0.0240** |

Breaking the pairing raises C's depth spread by roughly 40% in every seed. This is the **same
suppression structure** found for the type-by-depth interaction, now at the level of the main depth
profile — the second independent appearance of curvature and gradient moving together and partly
cancelling in `C = lam/g²`.

**A presentation error corrected in passing.** A variance-share framing of this decomposition printed
"lam 155–202%, g 35–50%", which is meaningless: the shares exceed 100% because the cross term is
large and negative. Standard deviations with the cross term shown explicitly are the correct
presentation, and are what appear above.

**Where this leaves the charter question.** C's depth profile is a **curvature** profile, not a
gradient one; within the curvature part, concentration explains 63–73% of the depth variance. The
gradient side contributes mainly by *cancelling* part of the curvature profile rather than by
creating it. The `mlp.proj` elasticity result remains a real type effect (c = +0.578, t = +7.72) but
is **largely orthogonal to the depth question**, and REQ-053 is correctly ranked last.

## Withdrawn before use: the "trace + n_eff" control is a proxy for the outcome (2026-09-05)

Iteration 240 found `log lam` and `log g` co-move across depth (corr +0.60 to +0.80). Testing whether
concentration explains that co-movement, controlling `log n_eff` alone left it intact (+0.51 to
+0.85), but controlling `log trace(H)` **and** `log n_eff` together collapsed it to −0.007, +0.443,
−0.013, −0.221.

**That collapse is not reportable.** Since `log n_eff = 2 log trace(H) − log trace(H²)`, those two
controls span exactly the two Hutchinson moments — and those moments predict `log lam` across depth
at **R² = 0.80 to 0.88** (corr(log lam, log rms) = +0.85 to +0.93). Controlling them removes most of
`lam`'s own depth variation, so the residual correlation with `log g` is computed on what little of
`lam` remains. That is rule 35's defect: a quantity regressed against its own near-complement. The
result is withdrawn before it was used, and the question it was meant to answer — whether the
`lam`–`g` co-movement reduces to concentration — **remains open**.

## The curvature moments predict the gradient norm's depth profile (2026-09-05)

The non-circular version of that question is legitimate and gives a strong result. Ask whether the
**gradient** norm's depth profile is predicted by the two Hutchinson curvature moments, using `g` as
the outcome and never using `lam`. Neither the hard rule nor rule 35 applies: `g` is not built from
either moment, and `lam` appears nowhere.

**Block-mean depth profile** (12 blocks), `log g` on `log trace(H)` and `log sqrt(trace(H²))`:

| seed | R² | adjusted R² | permutation p |
|---|---|---|---|
| 0 | **0.9410** | 0.9278 | **0.0000** |
| 1 | 0.8854 | 0.8600 | 0.0000 |
| 2 | 0.8837 | 0.8579 | 0.0000 |
| 3 | 0.8547 | 0.8225 | 0.0000 |

The permutation null shuffles which block's moments pair with which block's `g`: null mean R² is
**0.18** (95th percentile 0.48) against observed 0.85–0.94, in 2000 draws per seed.

**It is not an aggregation artifact.** At **matrix level** — 72 points per seed, no block averaging,
type absorbed — R² is **0.975 to 0.982**, an incremental **+0.120 to +0.157** over type alone.

**It is not a shared depth trend.** Against a baseline of type plus **cubic depth**, the two moments
still add **+0.100 to +0.146** in every seed.

**Coefficients are stable and both positive**: on `log trace(H)` +0.198 to +0.323, on
`log sqrt(trace(H²))` +0.217 to +0.277, across all four seeds.

**What this says.** Where a matrix's curvature is large — in both total (`trace`) and top-weighted
(`rms`) senses — its gradient norm is large too, and this holds at matrix level after depth and type
are absorbed. Combined with iteration 231's ladder (elasticities wrt `log g` rising from `trace` to
`rms` to `lam`), the two moments are jointly informative about the gradient side rather than either
one alone.

**What it does not say.** This is observational co-variation within a single checkpoint, not a
direction of causation: curvature and gradient norm are measured at the same step and both respond to
training dynamics. It also does **not** close the account — the open question from iteration 240
(*why* `lam` and `g` co-move across depth, and whether that reduces to concentration) is untouched by
this, because answering it with the Hutchinson moments is exactly the circular move withdrawn above.
Settling it needs a design that moves curvature and gradient independently, which is **REQ-051**.

## The curvature/gradient displacement asymmetry is a noise artifact (2026-09-05)

Iteration 241 left open whether the curvature–gradient relationship has a direction. The three
intervention archives can address it without Hutchinson moments: an LR change acts on the optimiser,
so comparing each arm's displacement from control in `log lam` versus `log g` says which side moves.

**The raw result looked like a strong asymmetry.** Under every arm, `sd(d log lam)` exceeded
`sd(d log g)` by **2.3× to 4.9×** — apparently, curvature responds far more than the gradient norm.

**It does not survive noise accounting.** Two errors were made and corrected in sequence.

**First error — a vacuous test.** A "reverse regression" diagnostic was computed, comparing
`b(lam~g)` with `b(g~lam)`. Their product equals `corr²` as an algebraic identity, so the test can
never disagree with itself. Discarded (rule 35).

**Second error — an inapplicable noise constant.** Subtracting REQ-048's measured per-probe noise
(0.349 dex for `log lam`) floored *every* displacement at exactly zero, because that noise exceeds
every observed displacement (0.17–0.30 dex). **That constant does not transfer.** REQ-048's figure is
the spread of an 8-iteration Lanczos estimate under *reseeding*; REQ-036 and REQ-037 each trained
their own arms and probed once. The archives settle it directly: matching REQ-036's control against
REQ-037's control over 72 shared matrices gives **corr(log lam) = +0.865**, whereas noise of 0.349
dex against a between-matrix sd of 0.446 would cap reliability at **0.620**. The transferred constant
is too large.

**Redone with a defensible bound.** The same cross-archive comparison bounds noise from *above*
without assuming anything: `sd` of the control-to-control difference is **0.2234 dex** for `log lam`
and **0.0416 dex** for `log g`. These contain both archives' noise *and* real drift (steps 2250 vs
2750, different runs), so per-measurement noise is at most 0.158 and 0.029 dex — upper bounds, which
make any surviving signal conservative.

| design | arm | raw sd d(log lam) | **signal** | raw sd d(log g) | **signal** | signal ratio |
|---|---|---|---|---|---|---|
| REQ-036 | a2_pertype | 0.2305 | 0.0570 | 0.0734 | 0.0604 | **0.94** |
| REQ-036 | a3_endcap | 0.2658 | 0.1441 | 0.0884 | 0.0780 | **1.85** |
| REQ-036 | a4_antirule | 0.2711 | 0.1537 | 0.1142 | 0.1064 | **1.44** |
| REQ-036 | a5_polar | 0.2992 | 0.1990 | 0.1288 | 0.1219 | **1.63** |
| REQ-037 | a2_batch05x | 0.1716 | **0.000** | 0.0395 | **0.000** | — |
| REQ-037 | a3_batch2x | 0.1849 | **0.000** | 0.0375 | **0.000** | — |

**The pure-noise expectation for the raw ratio is 5.37** — larger than every raw ratio observed. So
the raw 2.3–4.9× was never evidence of asymmetry; it was consistent with noise alone, and the
noise-corrected ratios (**0.94 to 1.85, mean 1.47**) show curvature and gradient displacing by
**comparable** amounts under an LR intervention.

**Both readings the raw number invited are therefore withdrawn:** curvature is not shown to be more
responsive than the gradient norm, and no direction is established. REQ-037's batch arms produce no
detectable signal at all under these bounds, consistent with iteration 235's finding that batch size
is a weak instrument here.

**What this leaves.** The direction of the curvature–gradient relationship remains **undetermined**
from committed data, as it was after iteration 241. The useful by-product is a **transferable
measurement fact**: `log g` is measured far more reproducibly than `log lam` across archives
(control-to-control difference 0.042 vs 0.223 dex; corr +0.988 vs +0.865), which is why every
elasticity in this campaign is limited by the curvature side, not the gradient side.

## `log g` reproduces by **type**, not by depth — a clarification to iteration 242 (2026-09-05)

Iteration 242 reported that `log g` reproduces across archives at **corr +0.988** while `log lam`
manages +0.865, and used this to argue the gradient side is the better-measured one. That number is
correct but **must not be read as saying g's depth structure is reproducible**. It is a matrix-level
correlation over 72 matrices, and it is dominated by the type axis.

**Variance decomposition of `log g` within a panel**, averaged over 11 committed panels
(REQ-036 control, REQ-037 control, REQ-045, REQ-048 × 4 seeds, Arm A × 4 seeds):

| axis | share of `log g` variance |
|---|---|
| **type** | **0.831** |
| **block (depth)** | **0.045** |

**Reproducibility at each grain**, mean pairwise correlation across the 11 panels:

| grain | mean cross-panel correlation |
|---|---|
| matrix level, type structure included | **+0.951** |
| matrix level, type means removed | +0.760 |
| **block-mean depth profile (12 points)** | **+0.688** |

So the headline +0.988 reflects a large, stable *type* pattern. g's **depth** profile reproduces far
less well: the pairwise range is +0.134 to +0.991, and the argmin block splits between **block 1**
(five panels) and **block 10** (four panels), with argmax at block 3 (seven panels) or block 11.

**g's depth profile does have a consistent shape, but a small one.** Z-scoring each panel's profile,
7 of 12 blocks have a mean displacement exceeding their cross-panel sd: block 1 is low (−1.57),
blocks 3 and 4 high (+1.30, +0.88), blocks 9 and 10 low (−0.72, −0.87), block 11 high (+1.06). The
total depth range is only **0.12 to 0.26 dex**, against the type spread that carries 83% of the
variance.

**Consequences.**

- Iteration 242's conclusion that curvature, not gradient, is the limiting measurement stands — but
  the supporting figure should be quoted as **+0.951 at matrix level**, and the depth-specific figure
  is **+0.688**.
- The n=4-style cross-archive check hoped for on the gradient side is **weaker than expected**: with
  depth carrying 4.5% of g's variance, eleven panels agree on the depth profile only moderately.
- This does not touch iteration 241's result (curvature moments predict g's *matrix-level* profile at
  R² = 0.975–0.982 with type absorbed), which was fitted at matrix level after removing type — the
  grain where cross-panel agreement is +0.760, not +0.688.

## Operator cleanup of REQ-051, accepted (2026-09-05)

The branch head moved to `185b68c`, authored by the operator (`jackzengh`), not by Jerry: *"Clean up
requests.md: restate REQ-051's registered targets once, drop the iteration diary."* No Jerry response
and no NEEDS-INFO has been received.

The change is correct and is accepted without amendment. Six chronological "added iteration NNN"
layers had accumulated in REQ-051's header, each partly superseding the last, leaving **three
contradictory scoring instructions** ahead of the spec and pushing the request metadata ~80 lines
down. It also removed a stale `corr(via, direct) = −0.920` still quoted as live evidence in the H1
block — **a figure retracted in iteration 229 that should have been cleared from requests.md at the
time**. FINDINGS.md carried the retraction; requests.md did not, and that inconsistency was mine.

Standing lesson: when a statistic is retracted in FINDINGS.md, search requests.md for it in the same
commit. The 2-node resource constraint is preserved in the rewritten header.

## The depth axis is a minority of C's variation — type carries most of it (2026-09-05)

Iteration 243 found type carries 83% of `log g`'s variance and depth only 4.5%. Applying the same
decomposition to every quantity gives a result that **reframes what this campaign's headline question
is asking**, and it should be stated plainly rather than left implicit.

**Variance shares of each quantity** (REQ-048, mean over 4 seeds, block and type as dummies):

| quantity | type only | block (depth) only | both | residual |
|---|---|---|---|---|
| **`log C`** | **0.755** | **0.116** | 0.871 | 0.129 |
| `log lam` | 0.299 | 0.226 | 0.525 | 0.475 |
| `log g` | 0.847 | 0.041 | 0.888 | 0.112 |
| `log n_eff` | 0.622 | 0.110 | 0.732 | 0.268 |

**In absolute terms** (dex): `log C` total sd 0.452, type-axis sd **0.393**, depth-axis sd **0.153**.

**The decision-relevant version.** A *perfect per-type* learning-rate schedule would remove
**73–78%** of `log C`'s variance; a *perfect per-layer* schedule would remove **9–16%**.

**Three checks before recording, all passed.**

- **Not parameter count.** `mlp.fc`/`mlp.proj` hold 2,359,296 parameters against attention's 589,824,
  so a size effect could masquerade as type. But `log n_params` alone explains only **0.158–0.243**
  against full type dummies at **0.731–0.780** — the bulk is *within*-group type structure, not an
  MLP-vs-attention split.
- **Not a scale effect.** `C = lam/g²` is invariant under the reparametrisation gauge, so a per-type
  scale difference cannot produce a type effect in C. Conditioning on `log g` first, type still adds
  **+0.408 to +0.511** incrementally.
- **Reproduces across archives**, not just REQ-048's four seeds: REQ-036 control (type 0.683, depth
  0.211), REQ-037 control (0.804, 0.079), REQ-045 (0.744, 0.104).

**What this changes, and what it does not.**

It does **not** invalidate any finding in this file. The bowl, the concentration account, the
LR-invariance of C's depth profile and the REQ-036 null are all statements *about the depth axis*,
and they remain exactly as measured. What changes is the **framing**: the depth axis is roughly
**one-eighth** of `log C`'s between-matrix variation, while type is roughly three-quarters. Every
"explains X% of the profile" figure in this file is a share of that one-eighth, not of C's total
spread, and should be read that way.

Note this is consistent with — and sharpens — the campaign's completed REQ-036 validation. That work
showed a **per-type** learning-rate rule cannot change the between-layer spread of C (a per-type
constant cancels exactly in the depth contrast). The present decomposition shows the converse
framing: per-type structure is where most of C's variance actually lives, but it is precisely the
part a per-type LR rule cannot touch *in the depth direction*. The two results address different
axes and do not conflict.

**A caveat on `log lam`.** Curvature alone is far more balanced (type 0.299, depth 0.226) than C is.
C's type dominance is inherited largely from `g`, which is 84.7% type. So the depth question is
relatively *more* prominent in raw curvature than in C — which is why the curvature-side account
(concentration) has been the productive one.

## `log lam`'s large residual is real structure, and concentration explains about half of it (2026-09-05)

Iteration 244 found `log lam` retains a **47.5%** residual after type and block, against `log C`'s
12.9%. The probe repeats settle whether that is signal or noise.

**It is overwhelmingly signal.** Splitting each residual into a part that survives probe reseeding
and a part that does not:

| quantity | residual sd | reproducible sd | noise sd | **reproducible share** |
|---|---|---|---|---|
| `log lam` | 0.2893 | 0.2711 | 0.1009 | **87.8%** |
| `log n_eff` | 0.3042 | 0.2843 | 0.1082 | 87.3% |
| `log g` | 0.0828 | 0.0787 | 0.0255 | 90.5% |
| `log C` | 0.1731 | 0.1411 | 0.1003 | 66.4% |

The `log lam` residual also correlates **+0.880 across seeds** (`log C`: +0.756). So there is a large,
reproducible per-matrix curvature pattern that neither type nor depth captures — **larger than the
depth bowl itself**.

**Concentration explains roughly half of it.** Adding the lam-free `log n_eff_bulk` to additive type
and block dummies removes **47.6% to 62.3%** of that residual's variance per seed (residual sd 0.270–0.294
→ 0.166–0.213 dex), lifting R² for `log lam` from **0.525 to 0.783**, an increment of **+0.257**.

This extends the standing concentration account beyond the depth axis: concentration predicts
per-matrix curvature generally, not only its depth profile.

## Withdrawn before use: an R² of 0.907 that was mostly a mathematical identity (2026-09-05)

Adding `log trace` alongside `n_eff_bulk` raised R² for `log lam` to **0.907**, which looked like a
near-complete account. **It is not reportable**, and rule 47 catches it.

`{n_eff_bulk, trace}` spans the two Hutchinson moments, and those moments predict `log lam` **on
their own**:

| model for `log lam` | R² |
|---|---|
| **moments only (`trace`, `rms`)** | **0.766** |
| type + block only | 0.525 |
| moments + type | 0.868 |
| moments + type + block | 0.921 |

Once the moments are in the model, type and block together add only **+0.156**. Most of the 0.907 is
the mathematical relationship between `lam` and the spectral moments that bound it — `sqrt(trace(H²))`
is an upper bound on `lam` — not an explanation of per-matrix structure.

**Caveat carried to the surviving number.** `n_eff_bulk = trace(H)²/(trace(H²) − lam²)` is built to
exclude `lam` algebraically and is admissible under the hard rule (Hutchinson, not the Lanczos
tridiagonal). But it still contains `trace(H²)`, which bounds `lam` above, so it is not fully
independent of the outcome. **The +0.257 increment should be read as an upper bound** on
concentration's incremental explanatory value for `log lam`, not a point estimate. On its own,
`n_eff_bulk` explains 0.276 of `log lam`.

**Net position.** The reproducible per-matrix curvature structure is real (87.8% reproducible,
cross-seed +0.880) and concentration accounts for at most about half of it. What remains — roughly a
fifth of `log lam`'s total variance — is reproducible structure with **no admissible explanation**,
and it is a larger target than the depth residual this campaign has been chasing.

## The lam–g relationship confirmed across independent runs (2026-09-05)

Iteration 242 left a real worry: `lam` and `g` are measured on the same probe batch in REQ-048, so
their association could be inflated by shared measurement error. REQ-047 settles it, because its
`grad_frob` is the gradient norm measured in a **different run** and cannot share probe error with
REQ-048's `lam`.

Joined on (seed, matrix name) with rule-28 discipline: 288 = 288 rows, identical key sets, 12 blocks
per seed. `corr(log g_REQ048, log grad_frob_REQ047) = +0.9552` — the same physical quantity.

**Partial correlation with `log lam`**, after type, block and `log n_eff_bulk`:

| gradient measurement | provenance | partial r | per seed |
|---|---|---|---|
| `grad_frob` (REQ-047) | **different run — no shared probe error** | **+0.750** | +0.75, +0.76, +0.75, +0.74 |
| `gradient_block_norm` (REQ-048) | same run, same probe batch | +0.786 | +0.82, +0.79, +0.80, +0.74 |

**The relationship survives the clean test at 95% of its same-archive strength.** This is the
strongest available answer to iteration 242's concern: the curvature–gradient association is not an
artifact of shared probe measurement, and it is remarkably stable across seeds (range 0.02).

**This is the known relationship re-measured, not a new predictor.** Iteration 231 established
`d(log lam)/d(log g) = +3.173`; the +0.750 here is that same association reaching `log lam` through
an independent archive. It does **not** address iteration 245's target — the reproducible per-matrix
curvature structure that type, depth and concentration leave unexplained.

## The cross-archive increment is inconclusive — three explanations, none clean (2026-09-05)

`grad_frob` retains a partial correlation of **+0.264** with `log lam` even after REQ-048's own `g` is
controlled. Three readings were tested and **none survives cleanly**; the result is recorded as
inconclusive rather than resolved.

**Attenuation — refuted, twice.** If REQ-048's `g` were merely noisy, a second independent
measurement would pick up the uncontrolled remainder, and the increment should *shrink* as the
control improves. It does the opposite: controlling `g` from one probe repeat gives **+0.189**, two
repeats **+0.259**, all three **+0.264**. Attenuation also predicts symmetry; the relationship is
asymmetric — `g48 | grad_frob` = **+0.432** versus `grad_frob | g48` = **+0.264**.

**Same-run confounding — partially supported but insufficient.** REQ-048's `g` shares both run state
and probe batch with REQ-048's `lam`, which would explain its advantage without either measurement
being better. But if REQ-047 contributed only a generic cross-run gradient signal, its other
gradient-side field should behave similarly. It does not: `d_frob | g48` retains only **+0.105**
against `grad_frob`'s +0.264. Whatever REQ-047 adds is specific to its gradient-norm measurement.

**Structured run disagreement — the most likely account, and a limitation.** The two archives'
gradient measurements disagree by sd **0.0748 dex** around a constant per-seed offset, and that
disagreement is **not** random: **30.9%** of it is explained by matrix type and **20.3%** by block.
The two runs differ systematically, not just noisily. An increment built on a structured
between-run difference cannot be interpreted as either measurement error or new physics without an
experiment that holds the run fixed.

**Status.** No new predictor of `log lam`'s unexplained structure was found. The productive outcome
is the cross-run confirmation above; the increment is left open, and the structured disagreement
between REQ-047 and REQ-048 is now on record as a constraint on any future cross-archive join.

## The type axis is run-independent; the depth axis is low-signal but reliable when averaged (2026-09-05)

Iteration 246 found REQ-047 and REQ-048 differ **systematically** — 30.9% of their gradient
disagreement is explained by type, 20.3% by block. Two systematically different runs are a stronger
replication test than four REQ-048 seeds, which rule 32 showed share ~76% of their structure.

**The type axis is run-independent.** Comparing each archive's own gradient measurements:

| axis | within REQ-048 (seed-to-seed) | within REQ-047 | **across archives** |
|---|---|---|---|
| type effects | +0.9986 | +0.9994 | **+0.9891** |
| depth profile | +0.6490 | +0.5692 | **+0.5431** |

The variance split also reproduces: type carries **0.847** of `log g` in REQ-048 and **0.895** in
REQ-047; depth carries 0.041 and 0.031. The full 6-way type ordering
(`mlp.proj` > `attn.v` > `mlp.fc` > `attn.proj` > `attn.k` > `attn.q`) is **identical in 3 of 4
seeds**, with the fourth swapping only the bottom two.

**The depth axis's weaker reproduction is not caused by the run difference.** Cross-archive depth
correlation (+0.543) is essentially the same as *within*-archive seed-to-seed correlation (+0.649 and
+0.569). Depth reproduces less well because it carries only 3–4% of `log g`'s variance, not because
the two runs differ in depth structure.

## Correction: single-seed profile correlations understate the averaged profile's reliability (2026-09-05)

Those pairwise correlations of +0.54 to +0.65 invite the conclusion that this campaign's depth
profiles are shaky. **They are not**, and the pairwise statistic is the wrong instrument: it measures
a *single-seed* profile, while every depth claim in this file is made on the **4-seed average**.

Intraclass correlation of the 12 block means, treating seeds as replicates — the reliability of the
averaged profile actually analysed:

| quantity | between-block sd | seed-to-seed sd | **ICC of the 4-seed profile** |
|---|---|---|---|
| `log C` | 0.1486 | 0.0630 | **0.957** |
| `log n_eff` | 0.1872 | 0.0781 | **0.958** |
| `log lam` | 0.1874 | 0.0904 | **0.945** |
| `log g` | 0.0406 | 0.0311 | **0.872** |

Restricting to the three probe repeats within a seed — measurement noise only, no training variation
— gives ICC **0.922** for `log g` and **0.918** for `log lam`, so probe noise and seed-to-seed drift
contribute comparably.

**Consequence.** The depth profile of `log C` that this campaign has analysed is a **stable object**
(ICC 0.957), and the bowl, the concentration account and the LR-invariance results rest on a reliable
measurement. What is genuinely limited is any claim about a **single seed's** depth profile, and any
claim about `log g`'s depth profile specifically, which is both the lowest-signal (3–4% of variance)
and the least reliable (0.872) of the four.

This does not disturb iteration 243's finding that `log g` reproduces **by type, not by depth** — that
comparison was at matrix level and stands. It adds that the *averaged* depth profile is nonetheless
reliable enough to analyse, which iteration 243's +0.688 figure understated for the same reason.

## C's depth bowl reproduces across five genuinely different runs (2026-09-05)

Every depth claim in this campaign has rested on REQ-048's four seeds, which rule 32 showed share
~76% of their structure. Five archives carry both `top_eigenvalue` and `gradient_block_norm`, so
`log C`'s depth profile is computable in all of them — **different training runs, different steps,
different configurations**: REQ-048 (4 seeds, step 2750), Arm A (4 seeds, 2750), REQ-036 control
(2250), REQ-037 control (2750), REQ-045 arm s10 (2750).

**The bowl reproduces.** Mean pairwise correlation of the five depth profiles is **+0.8009**, and
**+0.8661 after removing a linear depth trend from each** — the agreement is in the *curvature*, not
a shared slope. Removing the quadratic as well collapses it to +0.333, confirming the bowl **is** the
quadratic component.

**Permutation null**, shuffling block labels independently per panel over 20,000 draws:

| profiles | observed | null mean | null 95th | p |
|---|---|---|---|---|
| raw | +0.8009 | +0.0010 | +0.1784 | **0.00000** |
| linear-detrended | +0.8661 | +0.0001 | +0.1707 | **0.00000** |

**Which blocks carry it**, z-scored across the five panels (sd is across *different runs*):

| block | mean z | sd |
|---|---|---|
| 5 | −0.76 | **0.08** |
| 8 | −0.47 | 0.17 |
| 6 | **−1.07** | 0.25 |
| 10 | +0.42 | 0.26 |
| 7 | −0.78 | 0.32 |
| **11** | **+2.04** | 0.52 |

Blocks 5–8 (the interior minimum) and block 11 (the output end) are tightest. Block 11 is the highest
block in **every** panel (+0.295 to +0.370 dex centred).

**This is much stronger evidence than seed replication.** Four correlated seeds cannot demonstrate
run-independence; five systematically different runs can, and the bowl is present in all of them with
argmin at block **4 or 6** and amplitude 0.49–0.87 dex.

## But the cross-run agreement is not specific to C (2026-09-05)

A guard against over-claiming: the same test on the components gives near-identical numbers.

| quantity | raw cross-run corr | detrended | amplitude range |
|---|---|---|---|
| `log C` | +0.8009 | +0.8661 | 0.492–0.865 dex |
| `log lam` | +0.8009 | **+0.8755** | 0.608–0.852 dex |
| `log g` | +0.7757 | +0.8612 | 0.147–0.263 dex |

So "the depth profile reproduces across runs" is **not** a special property of C — curvature and
gradient norm each reproduce about as well. What *is* specific to C is the **location** of its
minimum:

| panel | argmin `log C` | argmin `log lam` | argmin `log g` |
|---|---|---|---|
| REQ-048 | 6 | 6 | 1 |
| Arm A | 6 | 6 | 1 |
| REQ-036 | 4 | 7 | 10 |
| REQ-037 | 6 | **1** | 10 |
| REQ-045 | 4 | **1** | 1 |

`log C`'s minimum sits at block 4 or 6 in all five panels; `log lam`'s scatters across 1, 6 and 7,
and `log g`'s across 1 and 10. **C's interior minimum is the reproducible object, and it is more
stable than either component's** — consistent with iteration 240's finding that C's profile is the
residue of a partial cancellation between curvature and gradient, with the cancellation stabilising
the location.

**Status.** The bowl is established as run-independent, not merely seed-replicated. The claim that
should be made is about C's minimum *location* (blocks 4–6, 5/5 panels) rather than about profile
correlation, which the components share.

## Why C's minimum is more stable than its components': the mechanism, quantified (2026-09-05)

Iteration 248 found `log C`'s argmin sits at block 4–6 in all five runs while `log lam`'s scatters
(1, 6, 7) and `log g`'s scatters (1, 10). A difference of two unstable quantities being *more* stable
than either is not automatic — it requires their run-to-run fluctuations to be correlated in a
specific way, and that requirement is a sharp, checkable inequality.

**The condition.** With `log C = log lam − 2 log g`, C's profile is more run-stable than lam's exactly
when `cov(dev_lam, dev_g) > var(dev_g)`. Measured over five panels × 12 blocks:

| quantity | value |
|---|---|
| `var(dev_lam)` | 0.00745 |
| `var(dev_g)` | 0.00074 |
| `cov(dev_lam, dev_g)` | **0.00112** |
| corr | +0.476 |

**0.00112 > 0.00074 — the condition is satisfied**, and `var(dev_C) = 0.00594`, a ratio of **0.798**
against lam's.

**A second mechanism points the same way.** C's bowl is *sharper* than lam's — quadratic coefficient
**+0.0136** versus **+0.0112** — and a sharper bowl pins the minimum more tightly against the same
noise.

**A parameter-free prediction.** For a profile `a(x−m)² + noise(s)`, the argmin's sd scales as `s/a`,
so the predicted argmin-sd ratio is `sqrt(0.798) × (0.0112/0.0136) = 0.736`.

| | argmin (quadratic fit) mean | sd across runs |
|---|---|---|
| `log lam` | 5.27 | 0.82 |
| `log g` | 5.36 | 3.19 |
| **`log C`** | **5.35** | **0.53** |

**Observed ratio 0.640 against a predicted 0.736.** This is the first parameter-free prediction in
this campaign that has not been refuted — iteration 223's three attempts missed by factors of 1.15,
10 and 1.3.

**Reported as consistent, not confirmed.** Leave-one-panel-out shows the *prediction* is stable
(0.678–0.842) but the *observation* is not: dropping REQ-037 gives 0.328, dropping REQ-045 gives
0.924. With five panels an sd has 4 dof, whose 90% sampling interval already spans a factor of
0.65–1.87 of the true value, and a ratio of two such sds is wider still. The agreement in magnitude
and direction is real; the point values are not precise.

**An incidental structural finding.** `log g`'s quadratic coefficient across depth is **negative**
(−0.0012, negative in all five panels): **g is an arch, not a bowl.** The quadratic-argmin model
therefore does not apply to `g` at all, and the C-versus-g version of the prediction is meaningless —
recorded so the comparison is not attempted later. It also sharpens iteration 240's cancellation
picture: C's bowl is not curvature's bowl minus a gradient bowl, but curvature's bowl **plus** the
inverse of a gradient arch, with both contributions deepening the same interior minimum.

## Withdrawn: "`g` is an arch" (2026-09-05)

Iteration 249 recorded, as an incidental finding, that `log g`'s quadratic depth coefficient is
negative in all five panels and concluded **g is an arch, not a bowl**, with the arch adding to C's
bowl through the `−2 log g` term. **That is withdrawn.**

**The coefficient is not significant anywhere**: t = −0.31, −0.97, −0.69, −0.31, −1.20 across the five
panels. Pooling gives `a_g = −0.00099`, se 0.00068, **t = −1.46, p = 0.144**; the 5/5 sign agreement
alone gives p = 0.062.

**More decisively, the quadratic does not describe `log g`'s depth profile at all.** Its R² is
**0.034 to 0.244** (mean 0.133). A parabola fitted to a non-parabolic shape returns a coefficient
that absorbs residual structure rather than measuring curvature, so `a_g` is not evidence of an arch.
The raw profile confirms it: `g`'s argmin lands on block 1, 1, 10, 10, 1 and its argmax on 11, 3, 3,
3, 2 — a wiggle, not an arch.

**Consequence for the decomposition.** The identity `a_C = a_lam − 2 a_g` holds by construction, and
the apparent "arch share of C's bowl" (6.2%–41.1%, mean 17.7%) is therefore **not a real
apportionment** — it is a share attributed to a coefficient indistinguishable from zero. The correct
statement is that **the gradient's contribution to C's bowl is not established** by these data.

## The bowl is sharper in C than in either component (2026-09-05)

Asking the same question of every quantity — *does a parabola describe this depth profile?* — gives a
result that does not follow from either component:

| panel | `log lam` R² | `log g` R² | **`log C` R²** |
|---|---|---|---|
| REQ-048 | 0.443 | 0.099 | **0.809** |
| Arm A | 0.498 | 0.105 | **0.877** |
| REQ-036 | 0.602 | 0.244 | **0.790** |
| REQ-037 | 0.256 | 0.034 | **0.743** |
| REQ-045 | 0.368 | 0.183 | **0.878** |
| **mean** | **0.433** | **0.133** | **0.819** |

t-statistics on the quadratic term tell the same story: `log C` reaches **+4.29 to +7.93** in every
panel, `log lam` only **+1.56 to +3.38**, `log g` never differs from zero.

**C is more parabolic than either quantity it is built from.** This rules out the reading that C's
bowl is simply inherited from curvature: if it were, `log lam`'s parabola would fit at least as well.
Something about the combination `lam/g²` produces a cleaner quadratic depth profile than curvature
alone — consistent with iteration 240's cancellation result and iteration 249's finding that C's
minimum is more run-stable than its components', but stronger than either, because it concerns the
*shape* rather than the location or the noise.

**What is not claimed.** The mechanism producing this is unidentified. The gradient's depth profile
is real (it has structure) but not parabolic, so "the arch deepens the bowl" is unavailable as an
explanation, and no substitute has been tested. This is a description of a systematic asymmetry
across five independent runs, awaiting a mechanism.

## Why C is more parabolic than its components: shared non-quadratic structure cancels (2026-09-05)

Iteration 250 established that a parabola describes `log C`'s depth profile far better (R² 0.819)
than it describes either `log lam` (0.433) or `log g` (0.133), and left the mechanism open.
Subtracting a poorly-parabolic profile from a weakly-parabolic one should not *improve* the fit —
unless the two share their non-parabolic structure.

**They do.** Removing a fitted quadratic from each profile leaves a non-quadratic residual shape;
those shapes are strongly correlated between curvature and gradient:

| panel | sd nq(`lam`) | sd nq(`g`) | **corr** |
|---|---|---|---|
| REQ-048 | 0.1387 | 0.0398 | **+0.969** |
| Arm A | 0.1272 | 0.0393 | **+0.960** |
| REQ-036 | 0.1648 | 0.0588 | +0.744 |
| REQ-037 | 0.1582 | 0.0510 | **+0.968** |
| REQ-045 | 0.1317 | 0.0667 | +0.930 |

**And `C`'s coefficient of 2 is close to the cancelling one.** Fitting `k` freely to minimise the
non-quadratic residual of `log lam − k·log g`:

| panel | best `k` | nq sd at best `k` | at `k = 2` | at `k = 0` |
|---|---|---|---|---|
| REQ-048 | 3.38 | 0.0344 | 0.0647 | 0.1387 |
| Arm A | 3.11 | 0.0356 | 0.0562 | 0.1272 |
| REQ-036 | 2.09 | 0.1101 | 0.1102 | 0.1648 |
| REQ-037 | 3.01 | 0.0394 | 0.0647 | 0.1582 |
| REQ-045 | 1.84 | 0.0484 | 0.0496 | 0.1317 |

Best `k` = **2.68 ± 0.68**, straddling the 2 that `C = lam/g²` uses. At `k = 2` the non-quadratic
residual is already less than half its `k = 0` value (0.056 vs 0.132 on average).

**So C's cleaner bowl is not a new phenomenon — it is the λ–g relationship acting on the depth axis.**
The shared non-quadratic depth structure of curvature and gradient largely cancels in `lam/g²`,
leaving the quadratic component exposed. That the exponent 2 happens to sit near the cancelling
coefficient is a property of the EoS constant's algebraic form, not a tuned choice.

**The cancelling `k` belongs to the λ–g elasticity family.** Computed on the same five panels, `k`
(mean 2.68) tracks the between-matrix elasticity (mean 2.83) at **corr +0.966** and the block-mean
slope (mean 2.05) at **+0.898**. Exact permutation nulls over all 120 orderings of five panels give
**p = 0.0083** — the smallest value n = 5 permits — and **p = 0.033** respectively.

**What is not claimed.** That `k` tracks the *between-matrix* elasticity more closely than the
block-mean slope is **not supported**: the gap between +0.966 and +0.898 is well inside the sampling
error of a correlation on five points, and the two predictors are themselves correlated at +0.876. The
supported statement is the weaker one — the cancelling coefficient is a member of the λ–g elasticity
family rather than an independent constant.

**Connection to the standing account.** This links three previously separate results: the λ–g
elasticity (iterations 231–234, ~2–3 depending on design), C's cancellation-stabilised minimum
(iteration 249), and C's parabolic depth profile (iteration 250). They are aspects of one
relationship between curvature and gradient norm, expressed at different grains.

## The bowl is a curvature phenomenon; dividing by g² sharpens it rather than creating it (2026-09-05)

Iteration 251 explained why C's depth profile is *cleanly* parabolic but not why there is a bowl at
all. The decisive test: remove the gradient's influence from curvature at matrix level and ask
whether the bowl survives.

| quantity | mean `a2` | mean t | mean R² | argmins across 5 panels |
|---|---|---|---|---|
| `log lam` | +0.0112 | +2.45 | 0.433 | 6, 6, 7, 1, 1 |
| `log lam \| log g` | +0.0122 | +3.66 | 0.609 | 6, 6, 5, 1, 4 |
| **`log C`** | **+0.0136** | **+6.39** | **0.819** | **6, 6, 4, 6, 4** |

**The bowl survives.** Curvature with the gradient's linear influence regressed out still has a
positive quadratic depth coefficient, significant in its own right. The bowl is a property of
curvature; `lam/g²` does not manufacture it.

**The sharpening is significant and consistent.** Paired across the same five panels, `a2(C) − a2(lam)`
is **+0.0024** (sd 0.0017, t = +3.09) and `a2(C) − a2(lam|g)` is **+0.0013** (sd 0.0008, t = +3.63),
**positive in 5/5 panels** in both cases.

**And it is the same bowl, not a different object.** Within-panel profile correlations: corr(`lam|g`,
`C`) = **+0.867 to +0.958**, corr(`lam`, `lam|g`) = +0.893 to +0.987. The gradient adjustment changes
the bowl's depth and the stability of its location, not its shape.

## A qualification: growth in the quadratic coefficient is not itself evidence (2026-09-05)

Sweeping the adjustment coefficient `k` in `log lam − k·log g` shows why `a2` alone must not be read
as sharpening:

| `k` | mean `a2` | mean t | mean R² | argmins |
|---|---|---|---|---|
| 0 (raw `lam`) | +0.0112 | +2.45 | 0.433 | 6, 6, 7, 1, 1 |
| fitted (~1) | +0.0122 | +3.66 | 0.609 | 6, 6, 5, 1, 4 |
| **2** (`log C`) | +0.0136 | +6.39 | 0.819 | 6, 6, 4, 6, 4 |
| **2.7** (measured elasticity) | +0.0144 | **+7.96** | **0.863** | 6, 6, 4, 6, 4 |
| 4 (over-correction) | **+0.0159** | +6.66 | 0.788 | 6, 6, 3, 6, 6 |

`a2` **keeps growing monotonically** even at `k = 4`, a deliberate over-correction — so a larger
quadratic coefficient can be produced simply by subtracting more of `g`'s own shape, and the rise
from +0.0112 to +0.0136 is not by itself evidence that the gradient sharpens anything.

**What is evidence is that the fit quality peaks near the measured elasticity.** R² and the
t-statistic both rise to `k ≈ 2.7` — the λ–g elasticity measured independently in iterations 231–251
— and **decline** by `k = 4`. The argmins likewise stabilise at [6, 6, 4, 6, 4] for `k` between 2 and
2.7 and destabilise outside that range. The optimum sitting at the independently measured elasticity,
rather than at the boundary of the sweep, is what distinguishes a real cancellation from an artifact
of subtracting more.

**Net position on the charter question.** The bowl in C has three established components: it
originates in **curvature** (survives removing `g`, +0.0122 with t = +3.66), it is **sharpened and
its location stabilised** by dividing by `g²` (5/5 panels, and the optimum coincides with the
measured elasticity), and it is **run-independent** (iteration 248, five different runs). What sets
the curvature bowl itself remains the open question, with concentration explaining 63–73% of its
depth variance (iteration 240).

## Concentration produces about half the bowl, and the half it produces is a bowl (2026-09-05)

Iteration 240 recorded that concentration explains 63–73% of `log lam`'s depth **variance**. That is
not the same as explaining the **bowl**: a predictor can absorb most of a profile's variance while
leaving its quadratic shape intact. Testing the quadratic directly gives a sharper answer.

Controlling `log n_eff_bulk` at matrix level (Hutchinson, `lam` excluded algebraically, so the hard
rule is satisfied), then refitting the quadratic depth coefficient:

| quantity | mean `a2` | mean t | mean R² | argmins (4 seeds) |
|---|---|---|---|---|
| `log lam` | +0.0116 | +2.43 | 0.402 | 7, 6, 9, 6 |
| `log lam \| n_eff_bulk` | **+0.0076** | +2.06 | 0.384 | 10, 2, 9, 1 |
| `log C` | +0.0123 | +5.54 | 0.736 | **6, 6, 6, 7** |
| `log C \| n_eff_bulk` | **+0.0064** | +3.73 | 0.656 | 2, 2, 6, 2 |

**The reduction is significant and consistent.** Paired across four seeds: `log lam` loses **+0.0039**
(sd 0.0011, **t = +7.16**, 4/4 seeds) — **34%** of its bowl; `log C` loses **+0.0059** (sd 0.0017,
**t = +6.93**, 4/4) — **48%** of its bowl.

**The residual is not flat, so the argmin shift is real.** `log C`'s depth amplitude falls from 0.555
to 0.336 dex — a ratio of **0.61**, not a collapse. A profile retaining 61% of its amplitude still has
a well-defined minimum, so the argmin moving from a tight **[6, 6, 6, 7]** to a scattered
**[2, 2, 6, 2]** is a genuine relocation rather than the degeneracy of a flattened curve.
**Concentration explains where the minimum sits, not only how deep it is.**

**And the part concentration explains is itself a bowl.** Extracting the fitted component
`n_eff_bulk`'s contribution to `log C` and taking its depth profile:

| seed | `a2` of the explained component | argmin | amplitude |
|---|---|---|---|
| 0 | +0.0075 | 8 | 0.291 dex |
| 1 | +0.0058 | 6 | 0.289 dex |
| 2 | +0.0035 | 7 | 0.269 dex |
| 3 | +0.0067 | 8 | 0.361 dex |

The explained component is a positive-curvature bowl with its minimum at blocks **6–8**, closely
matching C's own **6–7**. So concentration does not merely correlate with C across depth — the part
of C it accounts for has the same shape and nearly the same minimum location as the phenomenon being
explained.

**Where this leaves the charter question.** The account is now:

1. C's bowl **originates in curvature** and survives removing the gradient (iteration 252).
2. It is **sharpened and location-stabilised** by dividing by g², optimally near the measured λ–g
   elasticity (iteration 252).
3. **About half of it is spectral concentration** — 34% of curvature's bowl, 48% of C's — and the
   concentration-explained component is itself an interior bowl at blocks 6–8.
4. It is **run-independent** across five different runs (iteration 248).

The remaining half of the bowl has no admissible explanation. This is the same target identified in
iteration 245 (reproducible per-matrix curvature structure that type, depth and concentration leave
unexplained), now located specifically in the bowl's quadratic component rather than in variance
generally.

## The unexplained half of the bowl is real, reproducible, and concentrated in `mlp.fc` (2026-09-05)

Iteration 253 left half of C's bowl unexplained after controlling concentration. Characterising that
residual gives a specific target rather than a vague remainder.

**It is a genuine, reproducible bowl.** Across the four REQ-048 seeds the residual depth profile has
mean pairwise correlation **+0.759** (range +0.676 to +0.840), quadratic coefficients of +0.0053,
+0.0036, +0.0083, +0.0086 with **t = +4.20, +2.36, +4.44, +3.92** — significant in 4/4 seeds. Its
shape differs from the concentration-explained half: block 2 is the deepest point (mean z −1.54) and
block 11 the highest (+1.68), against the explained component's minimum at blocks 6–8.

**It lives overwhelmingly in `mlp.fc`.** Per-type quadratic coefficients of the residual: `mlp.fc`
**+0.0162**, `mlp.proj` +0.0083, `attn.v` +0.0068, `attn.proj` +0.0044, `attn.k` +0.0028, `attn.q`
+0.0002. `mlp.fc`'s residual bowl is significant in every seed (t = +4.2, +7.7, +8.7, +5.0).

**The pre-control check is what makes this interesting.** `mlp.fc` ranks only **3rd of 6** on its raw
bowl (a2 = +0.0162) but **1st of 6** after control — because concentration removes **0%** of it:

| type | a2 before | a2 after | **share removed** |
|---|---|---|---|
| `attn.proj` | +0.0219 | +0.0044 | **80%** |
| `attn.k` | +0.0116 | +0.0028 | 76% |
| `attn.q` | +0.0006 | +0.0002 | 65% |
| `mlp.proj` | +0.0193 | +0.0083 | 57% |
| **`mlp.fc`** | +0.0162 | **+0.0162** | **0%** |
| `attn.v` | +0.0043 | +0.0068 | **−58%** |

## Both anomalies have a mechanical explanation (2026-09-05)

A control that removes nothing from one type and *deepens* another's bowl is a warning sign, so both
were diagnosed before the result was recorded. The explanation is in `n_eff_bulk`'s own **depth
profile** per type — not in its matrix-level relationship with C:

| type | a2 of `n_eff` depth profile | corr(C, `n_eff`) across depth | share removed |
|---|---|---|---|
| `attn.proj` | **−0.0305** | −0.792 | 80% |
| `attn.k` | −0.0152 | −0.634 | 76% |
| `mlp.proj` | −0.0192 | −0.831 | 57% |
| `attn.q` | −0.0007 | −0.309 | 65% |
| **`mlp.fc`** | **−0.0000** | −0.107 | **0%** |
| **`attn.v`** | **+0.0044** | +0.082 | **−58%** |

**`mlp.fc`'s concentration has no arch at all** — its `n_eff` depth profile rises almost monotonically
(z from −1.37 at block 0 to +1.57 at block 10), so its quadratic coefficient is zero and controlling
it **cannot** remove a bowl, however strong the matrix-level association. **`attn.v` is the only type
whose `n_eff` has positive depth curvature**, so removing it deepens rather than flattens the bowl.

**The sharpened claim.** Concentration explains C's bowl in the four types where `n_eff` itself has an
arch across depth (57–80% removed), and explains none of it in `mlp.fc`, where `n_eff` rises
monotonically instead. The unexplained half of C's bowl is therefore not a uniform remainder — it is
**almost entirely `mlp.fc`'s bowl**, and it is unexplained for a specific, identified reason.

**Note on the two MLP matrices.** `mlp.fc` now carries the unexplained bowl, while `mlp.proj` carries
the excess λ–g elasticity (iteration 238, c = +0.578). These are different matrices and different
phenomena; the campaign should not conflate them. `mlp.fc`'s residual bowl is a **depth** result and
bears directly on the charter question, whereas `mlp.proj`'s elasticity was found to be largely
orthogonal to depth (iteration 240).

## `mlp.fc`'s unexplained bowl is tracked by activation effective rank (2026-09-05)

Iteration 254 localised the unexplained half of C's bowl to `mlp.fc`, where concentration removes 0%
of it. Testing REQ-047's activation and backward fields against that residual — admissible here, and
with g-family fields flagged separately since `g` sits in `log C`'s denominator — gives one strong
candidate:

| field | g-family? | share of `mlp.fc`'s residual bowl removed |
|---|---|---|
| **`a_eff_rank`** | **no** | **91%** |
| `grad_rank1_frac` | yes | 75% |
| `d_eff_rank` | no | 58% |
| `weight_frob` | no | 45% |
| `da_cos_mean` | no | 43% |
| `grad_frob`, `align_ratio` | yes | 9% |

`a_eff_rank` — the effective rank of the input activations — removes **91%**, consistently across all
four seeds (residual a2 falls to +0.001–+0.002). Against a random depth-varying control the null mean
is **9.1%** (95th percentile 21.3%, max 39.9%): **p = 0.0000**.

**The guard that nearly killed it.** `a_eff_rank`'s own depth profile has a quadratic coefficient of
**+0.0158**, almost exactly matching `mlp.fc`'s residual bowl of **+0.0163**. Controlling one
parabola with another of the same size removes it close to by construction, so the 91% is **not
evidence on its own**. Three tests distinguish coincidence from structure:

1. **The fitted slope is stable and near unity**: +0.816, +1.010, +1.005, +0.938 — mean **+0.942**,
   sd 0.091, coefficient of variation **0.10**. An arbitrary curve-match would not produce a
   consistent coefficient across seeds.
2. **It holds at matrix level**, not only between two 12-point profiles: within `mlp.fc`'s 48 rows the
   partial correlation is **+0.853** raw, **+0.856** with seed dummies, **+0.878** adding linear depth.
3. **They agree beyond the shared parabola.** Removing the quadratic from *both* series and
   correlating the leftovers gives **+0.319, +0.414, +0.317, +0.634** — mean **+0.421**, positive in
   4/4 seeds. Two curves that merely happened to share a bowl size would leave uncorrelated
   remainders.

**The placebo is also informative rather than uniform.** `a_eff_rank` does not remove every type's
bowl: 95% for `attn.v`, 91% for `mlp.fc`, 80% for `attn.k`, but **0%** for `mlp.proj` and *negative*
shares for `attn.proj` and `attn.q`. It is not a generic depth proxy.

**Why this is plausible for `mlp.fc` specifically.** `mlp.fc` reads the residual stream and expands
it; the effective rank of its input activations is a direct measure of how many independent
directions that input occupies. The near-unit slope (+0.942) says a one-dex change in input effective
rank moves the concentration-adjusted `log C` by about one dex.

**What is not established.** Direction: `a_eff_rank` and the curvature are measured at the same step,
so this is co-variation. It is also a within-type result on **one archive** (REQ-047 × REQ-048 joined,
4 seeds, 48 rows), not the five-panel cross-run evidence available for the bowl itself — REQ-047 is
the only archive with activation fields. Registering it as **hypothesis H2** for REQ-051, which
already commits to activation probes.

## Two quantities explain 80% of C's bowl — but only with per-type coefficients (2026-09-05)

Concentration and activation effective rank looked complementary across iterations 254–255.
Testing them jointly gives a result that at first appeared contradictory and turned out to be the
finding.

**Per type, the pair is nearly complete:**

| type | raw `a2` | `\| n_eff` | `\| a_rank` | **`\| both`** | joint share |
|---|---|---|---|---|---|
| `mlp.fc` | +0.0162 | +0.0163 | +0.0019 | **+0.0005** | **97%** |
| `attn.v` | +0.0043 | +0.0036 | +0.0002 | +0.0002 | 95% |
| `attn.proj` | +0.0219 | +0.0050 | +0.0154 | +0.0013 | 94% |
| `attn.k` | +0.0116 | +0.0034 | +0.0026 | +0.0008 | 93% |
| `mlp.proj` | +0.0193 | +0.0110 | +0.0099 | +0.0109 | 44% |
| `attn.q` | +0.0006 | +0.0002 | +0.0003 | +0.0011 | −90% |

Note `attn.q`'s raw bowl is +0.0006 — essentially absent — so its share is arithmetic noise on a
near-zero denominator, not a failure.

**Pooled, the same two controls remove only 47%** — and `a_eff_rank` alone removes **2%**, against 91%
within `mlp.fc`. **The gap is a specification artifact, not a limit on the explanation.** A pooled
regression fits one coefficient per control for all six types; allowing per-type slopes lifts the
share to **80%**:

| model | mean `a2` | share removed |
|---|---|---|
| raw `log C` | +0.0123 | — |
| + common slopes (2 controls) | +0.0065 | 47% |
| **+ per-type slopes (12 controls)** | **+0.0025** | **80%** |

Priced against a null using two *random* controls with the same per-type-slope structure: null mean
**18.0%**, 95th percentile 26.2%, max 35.4%, against an observed 80% — **p = 0.0000**. The extra
degrees of freedom do not account for it.

**The per-type heterogeneity is real, and it is what forces this specification.**

| type | `n_eff` slope | seed sd | `a_eff_rank` slope | seed sd |
|---|---|---|---|---|
| `mlp.fc` | −0.464 | 0.440 | **+1.016** | **0.139** |
| `attn.proj` | −1.160 | 0.527 | +0.541 | 0.431 |
| `attn.k` | −0.380 | 0.119 | +0.414 | 0.422 |
| `attn.v` | −0.207 | 0.335 | +0.411 | 0.113 |
| `mlp.proj` | −0.462 | 0.100 | **+0.024** | **0.075** |
| `attn.q` | −0.197 | 0.304 | −0.137 | 0.456 |

Between-type spread exceeds within-type (seed) spread for both controls: F-like ratios **5.42**
(`n_eff`) and **8.90** (`a_eff_rank`). The sharpest contrast, `mlp.fc` minus `attn.q` on
`a_eff_rank`, is **+1.153** (sd 0.512, **t = +4.51**, same sign 4/4).

**`n_eff`'s slope is negative for every type** — concentration relates to C the same way everywhere,
differing only in magnitude. **`a_eff_rank`'s slope is not**: it ranges from +1.016 (`mlp.fc`) to
+0.024 (`mlp.proj`) to −0.137 (`attn.q`). The two MLP matrices have the *most stable* `a_eff_rank`
slopes across seeds (sd 0.139 and 0.075) and sit at opposite ends of its range — consistent with H2's
registered specificity prediction, which requires the effect to fail for `mlp.proj`.

**What this does not establish.** That slopes differ by type is not a mechanism. It says the two
quantities relate to C differently in different matrices, and it explains why every pooled analysis
in this campaign has understated the explanation. Direction remains unestablished for both controls,
and `a_eff_rank` is available in one archive only.

## The `a_eff_rank` result is architecture-level, not per-run — H2 amended (2026-09-05)

H2 rests on a cross-archive join: `a_eff_rank` lives only in REQ-047, its outcome `log C` in REQ-048,
and iteration 246 showed those runs differ **systematically** (gradient disagreement sd 0.075 dex,
30.9% type-structured, 20.3% block-structured). Rule 52 requires that join to be checked before the
result is relied on.

**The seed-shuffle test.** The join pairs REQ-047 seed *s* with REQ-048 seed *s*. Repeating it with
**mismatched** seeds (all nine derangements) gives:

| pairing | share of `mlp.fc`'s residual bowl removed |
|---|---|
| matched seeds (the actual join) | **91%** |
| mismatched seeds (9 derangements) | **93%** (range 92–94%) |

**Mismatched seeds do as well as matched ones**, and the test had power to detect otherwise:
`a_eff_rank`'s between-block variation is **7.63×** its seed-to-seed variation for `mlp.fc`
(cross-seed profile correlation +0.986).

**Two consequences, one weakening and one strengthening.**

*Weakening.* This is **not** evidence that a run's own activation rank sets its own curvature bowl.
The strictest baseline confirms it: controlling the **seed-averaged** `a_eff_rank` profile — which
carries no seed or run information at all — removes **93%**, matching own-seed's 91%. `a_eff_rank`
contributes a **fixed depth curve**. The "4-seed replication" is therefore not replication in the
usual sense; any seed's profile works, because they are nearly identical.

*Strengthening.* The cross-archive join is **exonerated**. A join that performs identically under
mismatched seeds cannot be manufacturing a spurious per-seed match, so rule 52's concern does not
apply here.

**It is still the right curve, not merely a curve.** Comparing every REQ-047 field at identical
degrees-of-freedom cost, each entered as its seed-averaged depth profile:

| field | share of `mlp.fc`'s residual bowl removed |
|---|---|
| **`a_eff_rank`** | **93%** |
| `grad_rank1_frac` | 75% |
| `d_eff_rank` | 58% |
| `weight_frob` | 45% |
| `da_cos_mean` | 43% |
| `d_frob`, `d_rms` | 16% |
| `a_frob`, `a_rms` | 14% |
| `grad_frob`, `align_ratio` | 9% |

**H2 as registered is amended.** Its share and slope criteria stand, but two of its framings were
wrong:

- **H2-share and H2-slope should be scored on the seed-averaged profile**, not per seed. Per-seed
  scoring implies run-specific information the data do not support.
- **The claim is architecture-level**: the depth profile of input activation effective rank has the
  same shape as `mlp.fc`'s concentration-adjusted curvature bowl, in every seed and both archives.
  Whether a *perturbation* to activation rank moves the bowl is untested and is what REQ-051's
  activation probes should answer.

**Why this still matters for the charter question.** An architecture-level explanation is weaker
causally but is exactly the right kind of object for "what sets the between-layer difference in C":
the bowl's location and shape are properties of the architecture, reproducible across five runs
(iteration 248), and `a_eff_rank`'s depth profile is the closest measured match to the part
concentration cannot explain.

## Concentration is architecture-level too — a correction to the campaign's evidence base (2026-09-05)

Iteration 257 found `a_eff_rank` explains `mlp.fc`'s bowl through a fixed depth curve. Rule 25
requires asking the same of the campaign's own standing account. **Concentration behaves the same
way.**

**Absolute quadratic coefficients** (ratios are unusable here — see the method note below):

| type | raw `a2` | matched `n_eff` | mismatched seeds | seed-averaged |
|---|---|---|---|---|
| `attn.k` | +0.0116 | +0.0034 | +0.0060 | **+0.0001** |
| `attn.proj` | +0.0219 | +0.0050 | +0.0049 | +0.0042 |
| `attn.v` | +0.0043 | +0.0036 | +0.0027 | +0.0024 |
| `mlp.fc` | +0.0162 | +0.0163 | +0.0159 | +0.0162 |
| `mlp.proj` | +0.0193 | +0.0110 | +0.0116 | +0.0109 |
| `attn.q` | +0.0006 | +0.0002 | +0.0007 | +0.0006 |

Across the five types with a usable bowl, **mean(matched − averaged) = +0.0010** — the seed-averaged
profile leaves a *smaller* residual than each seed's own. Using seed *j*'s `n_eff` against seed *i*'s
`log C` works as well as the matched pairing. **Concentration explains the bowl through a depth curve
that is essentially the same in every seed**, exactly as `a_eff_rank` does.

**The root cause, quantified.** The share of each depth profile that is identical across seeds:

| quantity | between-block sd | seed-to-seed sd | **shared** |
|---|---|---|---|
| `log n_eff` | 0.1793 | 0.0611 | **89.6%** |
| `log C` | 0.1519 | 0.0536 | **88.9%** |
| `log lam` | 0.1928 | 0.0824 | 84.6% |
| `log g` | 0.0435 | 0.0302 | 67.5% |

**What this corrects.** Every 4-seed check in this campaign on a **depth-profile** claim was testing
initialisation robustness of a near-constant curve, not replicating a mechanism. This is rule 32's
finding (four seeds share ~76% of their structure) reaching the depth axis, and it applies to the
concentration account as much as to H2. Per-seed agreement on depth claims should not be quoted as
independent replication anywhere in this file.

**What is unaffected, and is now load-bearing.** The **five-panel cross-run evidence** of iteration
248 — C's bowl at argmin 4–6 in 5/5 panels, detrended cross-panel correlation +0.866, permutation
p = 0.0000 — stands untouched, because those panels differ in **run, step and configuration**, not
only in seed. That is now the campaign's strongest replication, and the only one that tests more than
initialisation.

**Method note: a ratio statistic broke, and the fix changed the answer.** "Share removed"
(1 − a2_after/a2_before) produced incoherent values — `attn.k` at matched 71%, mismatched 48%,
averaged 99%; `attn.q` at 65%, −15%, 2% — because the denominator is small for weak bowls. `attn.q`'s
raw bowl is +0.0006, so its share is arithmetic noise. Reporting **absolute** `a2` instead resolved
every inconsistency and reversed the apparent conclusion for `attn.k`.

**What remains genuinely untested.** Both explanations are curves co-varying with C's bowl across
depth, measured at one step in runs sharing an architecture. Nothing in the committed data
distinguishes: (i) activation rank and concentration *set* the curvature bowl; (ii) all three are
downstream of a common architectural cause; (iii) all three are different views of one spectral fact.
Distinguishing them requires perturbing one and observing the others — **REQ-051** for the LR side,
**REQ-053** for an architecture change.

## Validation rules to retain

Validate matrix names and block indices before joining panels: expect blocks 0–11 and 72 matrices
per seed, check identical key sets, and compare shared measurements at compatible states.
Use actual runtime LR, explicit operator/loss scaling, and independently generated probe features.
Separate cross-type, within-type depth, and within-matrix treatment relationships.
Check shared terms, collinearity, group-summary artifacts and selection leakage before interpreting
a correlation. Report seed dependence, effect sizes, uncertainty, and inconclusive outcomes.

A parameter-free prediction must be derived and stated before its target value is looked at; once
several derivations have missed a known number, stop -- a further algebraic form chosen to land on
it is fitting, not deriving.
Verify what a filename suffix means before treating it as an experimental factor; repeated files
may be probe repeats of one state rather than distinct states, which changes both the correct
averaging and what guards are runnable at all.
Compute a criterion's chance rate before citing it as evidence: an 'interior minimum' style test
on k bins passes by pure combinatorics at (k-2)/k per replicate, so agreement counts like 4/4 or
12/12 can be near-worthless. Prefer criteria whose chance rate is small, such as agreement of a
location across seeds.
Cross-seed agreement is not independent replication when the tested residual is shared across
seeds; measure that sharing before treating N seeds as N tests. In this design the log C residual
is 76% shared, so judge claims on within-seed effect size with clustered standard errors.
Before hunting predictors for a residual, compute its reproducible share across replicates: that
share is a ceiling on what any structural predictor can explain, and it says whether the search is
worth running at all.
When an effect appears only after conditioning on a control, distinguish suppression from a
collider artifact by rebuilding the control without the shared term; if the result is unchanged,
the conditioning did not create it. Then verify suppression by exact decomposition, not by
elimination.
Never treat a relationship between a quantity and its own residual as evidence: if B is defined as
A minus C, then corr(C, B) is mechanically negative and corr on an OLS residual is exactly zero.
Test the claim on the composed quantity against a null that breaks only the hypothesised link.
An exact algebraic identity closing to machine precision confirms arithmetic, never a hypothesis.
When a correlation has the OPPOSITE sign to what its construction artifact predicts, the artifact
is not the whole story: compute the artifact's expected value and treat the gap as a measurable
quantity rather than dismissing the correlation.
Re-derive a theoretical reference value before testing against it; the plausible derivation may
give the wrong exponent. Here, scaling a loss contribution scales the Hessian by c, not c^2 -- only
a reparametrisation gives the c^2 that makes lam/g^2 gauge-invariant.
Verify whether a quantity is probe-estimated before assuming it is measured exactly; repeated
probe files make this checkable, and a regressor's reliability decides whether a slope is
attenuated. When several outcomes share one regressor, attenuation scales them all equally and
cannot create an ordering -- so orderings are reliability-invariant while level comparisons are not.
Before concluding a question needs new data, check every committed archive for the specific fields
it requires -- a weaker archive may support part of the test. And when disattenuating, verify the
regressor's and outcome's errors are independent: shared-probe errors can bias a slope either way,
so no corrected value should be quoted.
Confirm that a design's 'arms' are actual manipulations before treating a within-unit contrast as
causal: repeated measurement files can masquerade as arms, and when the outcome and regressor share
a measurement batch their correlated errors inflate the slope.
Test the mechanism you invoke to dismiss inconvenient data: a plausible story for why a result
should be discounted is a hypothesis with its own predictions, and it may fail them. When several
designs disagree on levels, a rank that holds across all of them may still be real -- price it for
post-hoc selection rather than quoting the level agreement.
A model fitted to a handful of aggregate points must be tested at finer grain before it is
believed: subsets that vary the fitted predictor without changing the estimator are usually
available, and a fit with one residual degree of freedom is a hypothesis, not a result.
Before pricing a cross-dataset agreement, measure how stable each dataset's own answer is: an
analytic p-value that treats one fit per dataset as a clean draw overstates the evidence when
resampling within a dataset changes its answer. And when a second statistic is chosen after the
first has been seen, report both and mark the second as non-confirmatory.
When a ranking is unstable, ask which competitor is moving before weakening the claim: the churn
may lie entirely in volatile rivals. Then replace the rank with a precision-weighted contrast and
run the same test on every category as a placebo -- that makes the null concrete instead of assumed.
Report a null with the effect size it could have detected; a null that only excludes effects as
large as the phenomenon itself does not favour either hypothesis. And when two explanations select
the same rows, no grouping can separate them -- say so and design the run that varies one alone.
Before extending a line of work, compute how much of the chartered question it could explain at
best; an effect can be real, well-controlled and replicated while being nearly orthogonal to the
question being asked. Never present variance shares when the cross term is large and negative --
report standard deviations with the covariance shown.
Check what a set of controls SPANS before interpreting a partial correlation: two controls that
jointly reconstruct the outcome will collapse any association, and the collapse means nothing.
Measure how well the controls predict the outcome itself first.
A noise estimate is specific to how a measurement was repeated; do not transfer one archive's
probe-reseeding spread to archives probed once. Bound the noise from within the archives being
compared, and check the ratio a pure-noise model predicts before calling a ratio an asymmetry.
State the GRAIN of a reproducibility figure: a matrix-level correlation dominated by a large
category effect says nothing about whether a within-category profile reproduces. Decompose the
variance by axis before quoting the number. When retracting a statistic, grep every file that
quotes it, not only the findings record.
Report what share of a quantity's total variation the studied axis carries, early and in the
findings record: a result explaining most of one axis may explain a small fraction of the whole,
and readers will assume otherwise unless told.
Benchmark an added predictor against a model containing its own mathematical relatives, not only
against the structural baseline: moments that bound the outcome will carry most of a high R2 by
identity. Report an increment as an upper bound when the predictor still shares a term with the
outcome.
Two measurements of one quantity from different runs are not interchangeable: check whether their
disagreement is structured by the same axes as the analysis before treating one as a cleaner
instrument for the other.
Measure reliability on the estimator you actually use: pairwise correlations between single
replicates understate an averaged profile's reliability, and the intraclass correlation over all
replicates is the right statistic. Systematically different runs test a claim more strongly than
correlated seeds -- use them when the fields allow.
Run the same reproducibility test on a quantity's components before claiming the result is special
to it: if the components agree equally well, the reproducible feature is something else -- find the
statistic that actually distinguishes them.
When a prediction is compared to an observed ratio of standard deviations, check the observation's
own stability by leave-one-out before claiming agreement: with a handful of replicates the
prediction can be far better determined than the quantity it is being tested against.
Check that a fitted functional form describes the data before interpreting its coefficients: a
parabola fitted to a non-parabolic profile returns a curvature coefficient that measures residual
shape, not curvature. Report the fit's R2 alongside the coefficient.
With a handful of panels, two high correlations are not distinguishable from each other: use an
exact permutation null to price each, and do not claim one predictor beats another when the gap is
inside the sampling error and the predictors are themselves correlated.
When an adjustment improves a statistic, sweep the adjustment past its intended value: if the
statistic keeps improving monotonically the gain may be mechanical, and the evidence is an interior
optimum coinciding with an independently measured quantity, not the improvement itself.
Explaining a profile's variance is not explaining its shape: test the shape parameter directly.
And when a control moves a profile's extremum, check the residual amplitude first -- a flattened
profile has no meaningful extremum, so relocation is only a finding if amplitude survives.
A control removes a shape only if the control itself has that shape along the same axis: check the
control's own profile per subgroup before concluding it fails there. A matrix-level association can
be strong while the control is flat along the axis being explained.
When a control removes most of a shape, check whether the control has that same shape at a similar
magnitude: if so the removal is near-automatic. Distinguish coincidence from structure by testing
whether the two agree AFTER the shared shape is removed from both.
A pooled model with one slope per predictor understates an explanation whose slopes differ by
subgroup: check per-subgroup fits before concluding a predictor is weak overall, and price the
extra parameters with a random-control null carrying the same structure.
Test a cross-dataset join by deliberately mismatching the join key: if a mismatched join performs
as well, the effect carries no per-unit information and must be described at the level that does
survive -- and the join itself is exonerated of manufacturing the result.
Prefer absolute changes to ratio-of-effect statistics when the denominator can be small: a
'share removed' explodes on weak effects and can invert a conclusion. Check the denominator's
magnitude before quoting any share.
