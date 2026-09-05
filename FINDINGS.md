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
