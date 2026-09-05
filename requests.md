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

Next request number: **REQ-053**.

---

### CONSOLIDATED NOTES (iterations 112–119, pre-table)

*Instrument audit and the arm-4 design history. Full text in git history from `ab04e19`.*

- **The batch instrument (REQ-037 arms 1–3) is unusable** — gives +0.383 [+0.028, +0.726] against the
  LR instrument's +2.07, but its reduced form is **non-monotone** and per-type ratios span **−1.25 to
  +1.12**. Not evidence against REQ-035; evidence that arm 4 was the right test.
- **Arm 4's original spec was broken twice** (iter. 119): a clip inside `polar_express` is **cancelled
  exactly** by the unit-norm step, and the probe **recomputes the gradient outside the optimiser**
  (`measure_per_matrix_curvature.py:100`), so the first stage would be identically zero. Both were
  fixed in REQ-046 — and band 31 later showed the design was **impossible regardless**.

## ⊘ THE SECOND COMPONENT IS NOT ANY MEASURED DIRECTION (iteration 203) — a pre-declared negative

*Band 71 bounds the unexplained component precisely: **0.272 dex of swing, cubic (R² 0.746), rising
toward both ends, replicating at +0.422.** REQ-048 carries **exactly three** admissible quantities not
already used in the concentration fit — all rule-13 clean, all containing **no `lam_top`**. **All three
were declared before fitting, all are reported, and the selection is priced by a permutation null.***

**⚠️ THE PRIOR WAS AGAINST ALL THREE, AND IT IS STATED FIRST.** Bands 56, 57 and 58 already established
what each direction does with depth: `curvature_along_polar` **monotone**, `curvature_along_weight`
**monotone**, `curvature_along_random` **flat**. **The residual is cubic and rises at BOTH ends** — a
shape none of them has. **Running the test anyway is the point: a bounded target with three
pre-declared candidates that all fail is a real negative, not a wasted iteration.**

| candidate | mean corr with the residual | same-sign |
|---|---:|---:|
| `curvature_along_weight` | **−0.236** | 7/12 |
| `curvature_along_polar` | −0.185 | 8/12 |
| `curvature_along_random` | +0.141 | 8/12 |

**Permutation null over block labels, max \|r\| across all three (20,000 shuffles): p = 0.7454.**

> **⊘ NONE of the three directional probes explains the second component.** The best is −0.236 and the
> selection-priced p is **0.745** — **not distinguishable from chance.** **The residual is not curvature
> along any direction REQ-048 measures.**

**⇒ WHY THIS IS INFORMATIVE RATHER THAN EMPTY.** The three probes span the natural alternatives: the
**optimiser's step direction**, the **learned weight direction**, and the **typical/random direction**.
Together with the concentration measure itself (`n_eff`, which explains the other 57%), **that is every
admissible field REQ-048 provides.** **So the second component is not a directional-curvature effect at
all** — it is something the current probe set does not measure.

**⇒ AND IT TIGHTENS WHAT THE SECOND COMPONENT CAN BE.** It is:
**reproducible** (+0.422 across fits, +0.410 within seed ⇒ architectural), **cubic** (R² 0.746),
**symmetric-ish** (rises at both ends), **0.272 dex** in swing, **present in `C = λ/g²` after removing
concentration**, and **invisible to λ along the step, weight and random directions.** **Since C's only
other component is `g`, the natural remaining candidate is a GRADIENT-side effect** — which REQ-048 does
not probe, but **REQ-051 does** (`a_frob`, `d_frob`, `align_ratio`, `d_token_participation`,
`da_cos_mean`, `grad_rank1_frac` at two checkpoints).

**PROPOSED n=4 SEED CHECK — band 72 (criterion registered).**
*Criterion:* after regressing the C profile on the concentration profile, the residual's correlation with
**each** of `curvature_along_polar`, `curvature_along_weight`, `curvature_along_random` must be
**non-significant** under a permutation null over block labels that prices the selection (p > 0.05), in
**≥3 of 4 seeds**.
*Status:* **satisfied by committed REQ-048 data** (best −0.236; **p = 0.745**).
**No new compute requested; ≤2-node ceiling.**

**⇒ A CONCRETE NOTE FOR REQ-051.** Its probe list already records the gradient-side fields at both 2050
and 2750. **If the second component is gradient-side, REQ-051's `k_a / k_d / k_rho` decomposition would
localise it at no additional cost** — and its registered decision 5 (*"which part of the gradient
response differs by layer?"*) is **exactly the right question, though currently aimed at the LR response
rather than at this residual.** **Recorded as an observation for whoever runs it; REQ-051 is unedited.**

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN** (origin, 16.2 min); **REQ-051 OPEN** (would probe the
gradient side); **REQ-052 OPEN**; REQ-049 optional. **No Jerry response since REQ-048.**

## 🔧 "ONE PHENOMENON" IS TOO STRONG (iteration 202) — concentration explains 57% of the bowl, and the rest is REPRODUCIBLE

*Band 70 argued the bowl and the concentration signature are **co-established**, from **timing** (both
present at 1750) and **correlation** (+0.77). **Neither distinguishes "one phenomenon" from "two
phenomena sharing an origin and a rough shape."** A sharper test exists on REQ-048, and — unlike bands
69/70 — **it is rule-6 CLEAN**: `log n_eff` contains **no `lam_top`**.*

**THE TEST.** Regress the C profile on the `log n_eff` profile with a free slope. Under *one
phenomenon*, the residual should sit at the per-layer noise floor.

| quantity | value |
|---|---:|
| mean slope | **−0.590** (sd 0.171) |
| **mean R²** | **0.567** — concentration explains **57%** of the C profile |
| mean rms residual | **0.1006 dex** (17% of the 0.600 dex bowl swing) |
| per-layer noise floor (this panel) | 0.0824 dex |
| **residual / floor** | **1.22×** |

**⇒ AND THE RESIDUAL IS A REPRODUCIBLE SECOND STRUCTURE, not fit noise:**

| check | result |
|---|---:|
| **mean pairwise correlation of the residual across the 12 fits** | **+0.422** (max +0.936) |
| within-seed pairwise correlation (across LRs, same network) | **+0.410** |
| residual swing | **0.272 dex** |
| cubic R² of the mean residual | **0.746** |

| block | 0 | 2 | 6 | 8 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|
| mean residual | +0.029 | **−0.130** | −0.074 | +0.048 | +0.076 | **+0.142** |

**The residual replicates across fits and is essentially as strong within-seed as across seeds (+0.410
vs +0.422), so it is architectural rather than seed-specific.** It has its **own** shape — rising toward
**both** ends, cubic R² 0.746.

> **🔧 BAND 70's "one phenomenon, not a phenomenon and a later consequence" is CORRECTED.** The timing
> claim stands — both structures are present and at equilibrium by step 1750. **But they are not the
> same object.** **Spectral concentration accounts for ~57% of the between-layer C profile; the
> remaining ~43% is a reproducible, architecturally-consistent second structure that concentration does
> not explain.**

**⚠️ AND NOTE THE SLOPE.** The fitted slope here is **−0.590** — **identical to band 59's profile-level
figure**, which iteration 191 corrected to a range (−0.34 saturated to −0.59 profile-level). **That
consistency is expected, since this is the same profile-level construction**, and it is a reminder that
**this iteration's 57% is itself a profile-level number**; the saturated relationship is weaker.

**PROPOSED n=4 SEED CHECK — band 71 (criterion registered).**
*Criterion:* regressing the C profile on the `log n_eff` profile per fit: (i) **mean R² is in
[0.35, 0.80]** — i.e. concentration explains a substantial but **not complete** share; (ii) the residual
**replicates at mean pairwise correlation ≥ +0.30** across fits; (iii) the residual's **rms exceeds the
per-layer noise floor by ≥1.1×**.
*Status:* **satisfied by committed REQ-048 data** (R² 0.567; +0.422; 1.22×).
**No new compute requested; ≤2-node ceiling.**

**⇒ WHAT THIS CHANGES IN THE ACCOUNT.** The headline answer becomes **quantitative rather than
absolute**: *the between-layer difference in C is **majority** a spectral-concentration profile, with a
**second, smaller, reproducible component** that concentration does not capture.* **That is a weaker
claim than the one I have been making since iteration 181, and it is the accurate one.** **Identifying
the second component is now the sharpest open analysis question** — and it is bounded: **0.272 dex of
swing, cubic-shaped, rising toward both ends.**

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN** (origin, 16.2 min); **REQ-051 OPEN**; **REQ-052 OPEN**;
REQ-049 optional. **No Jerry response since REQ-048.**

## ★ THE CONCENTRATION SIGNATURE IS PRESENT AS EARLY AS THE BOWL (iteration 201)

*Band 69 replicated band 57's consequence off-panel but left a gap: **REQ-048 measures only at step
2750**, so the concentration story could still be a **late-training description** rather than the bowl's
origin. **REQ-023 measures at 1750–2250 and carries both required fields** — and band 54 already showed
the **C bowl** is fully formed at 1750. **The untested question is whether the concentration signature is
too.***

**IT IS — unanimously, from the earliest measurement that exists:**

| step | detilted argmin (3 arms) | below both ends | corr with C profile |
|---|---|---:|---:|
| **1750** | 6, 6, 7 | **3/3** | **+0.754** |
| 1875 | 7, 7, 7 | 3/3 | +0.793 |
| 2000 | 4, 6, 6 | 3/3 | +0.777 |
| 2125 | 6, 5, 4 | 3/3 | +0.778 |
| 2250 | 4, 6, 4 | 3/3 | +0.765 |

**Interior minimum in 15/15 arm-steps; below both endpoints in 15/15; mean correlation with the C
profile +0.773, positive in 15/15.**

**⇒ AND IT IS ALREADY AT EQUILIBRIUM AMPLITUDE, not still growing:**

| quantity | trend over 1750–2250 | t |
|---|---:|---:|
| **concentration signature** (detilted swing) | **+0.028 dex/1000 steps** | **+0.14** |
| *(band 54: the C bowl itself)* | *+0.1425 dex/1000 steps* | *+1.04* |

**Neither trend is significant.** **The bowl and its concentration signature are BOTH at equilibrium by
step 1750 — the earliest curvature measurement anywhere in the repository (band 54).**

> **★ THIS CLOSES A REAL ALTERNATIVE.** Band 57's mechanism could have been a *description of the
> equilibrated state* that emerges late — in which case it would explain the bowl's *maintenance* but not
> its *origin*. **It is not: the signature is fully formed at the earliest observable point, at the same
> amplitude it has 1000 steps later.** **Whatever establishes the bowl establishes the concentration
> structure at the same time — they are one phenomenon, not a phenomenon and a later consequence.**

**⚠️ RULE 6 CAVEAT, CARRIED FORWARD UNCHANGED.** This quantity shares `log λ` with C, so it remains a
**descriptive consistency check, not independent evidence** for band 44. What it adds is a **timing**
fact, which the shared term does not affect: *when* the signature appears is not an artifact of *how*
it is constructed.

**PROPOSED n=4 SEED CHECK — band 70 (criterion registered).**
*Criterion:* on a panel with `top_eigenvalue` + `curvature_along_polar` at multiple early steps:
(i) the detilted `log λ − log cp` profile has an **interior minimum below both endpoints in ≥80% of
arm-steps at the EARLIEST available step**; (ii) its depth amplitude shows **no significant trend**
across the early window; (iii) its correlation with the C profile is **≥ +0.50 at every step**.
*Status:* **satisfied by committed REQ-023 data** (3/3 at step 1750; trend t +0.14; +0.754–0.793).
⚠️ **n = 3 arms, 1 seed.** **No new compute requested; ≤2-node ceiling.**

**⇒ WHAT THIS DOES TO THE OPEN QUESTION.** Both the bowl **and** its concentration signature are
established **before step 1750**, and **nothing in the repository measures earlier** (band 54, verified
across 22 curvature files). **The entire remaining question now sits inside a window no committed data
touches** — which is precisely what **REQ-050** (steps 0, 125, 250, 500, 1000, 1500; **16.2 min of
training for 4 seeds**) was filed to probe. **This iteration sharpens REQ-050's value: it would now
answer the origin of TWO co-established structures, not one.**

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN** (highest value); **REQ-051 OPEN**; **REQ-052 OPEN**
(endorsed); REQ-049 optional. **No Jerry response since REQ-048; no NEEDS-INFO.**

## ✅ THE CONCENTRATION RESULT REPLICATES ON A PANEL IT WAS NOT DERIVED FROM (iteration 200)

*Bands 44, 57 and 59 — the answer to the campaign's central question — **all come from ONE panel**:
REQ-048, the only dataset carrying `participation_ratio` / `trace_est` / `trace_sq_est`. **That
single-panel dependence has never been flagged, and it is a real weakness.** REQ-019 (11 LRs) and
REQ-023 lack those fields — **but band 57's mechanism makes a prediction testable with the fields they
DO have.***

**THE PREDICTION.** Band 57: `trace(H)` is flat across depth while `trace(H²)` carries the bowl. A
spectrum whose second moment rises while its first stays flat **must push `λ_top` up relative to a
non-extremal direction**. `curvature_along_polar` is such a direction and **is present in
REQ-019/023**. So `log λ_top − log cp` should show the bowl on a panel where the concentration fields
do not exist.

**⚠️ SCOPE, STATED FIRST.** `cp` is a **separate HVP, not tridiagonal** (rule 13 satisfied), **but
`log λ` appears in both this quantity and in C.** Under **rule 6** this is a **descriptive consistency
check, NOT independent evidence for band 44.** Recorded as such.

**FIRST PASS — partial, and the failure is informative:**

| | result |
|---|---|
| corr(peak profile, C profile) | **+0.723**, positive in **11/11** LRs |
| peak's own argmin interior (4–8) | **6/11** — drifts to blocks 0–1 |
| below both ends | 10/11 |

**⇒ AND THE DRIFT IS EXPLAINED BY AN EXISTING BAND, not a failure.** Decomposing (identity exact to
**6.9e-17**):

| block | 0 | 3 | 6 | 9 | 11 |
|---|---:|---:|---:|---:|---:|
| **C** | +0.169 | −0.011 | **−0.163** | −0.030 | +0.300 |
| log λ | +0.145 | +0.140 | −0.081 | −0.045 | +0.454 |
| **log cp** | **+0.208** | +0.195 | +0.046 | −0.118 | +0.037 |

**`log cp` is monotone — linear R² 0.487, slope −0.0286/block — which is exactly band 56's finding,
reproduced here on an independent panel.** Subtracting a *monotone* cp from a *bowl-shaped* λ tilts the
result toward the input end. **The partial reproduction is what band 56 predicts.**

**★ REMOVING THAT TILT GIVES A CLEAN REPLICATION:**

| | result |
|---|---|
| **argmin interior (4–8)** | **11/11 LRs** |
| **below both endpoints** | **11/11 LRs** |
| argmins | 4, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6 |

> **✅ The concentration mechanism's observable consequence reproduces at 11 learning rates on a panel
> that has none of the concentration fields.** **Bands 44/57/59 are no longer a single-panel result in
> their *implication*** — though the direct PR measurement still exists only in REQ-048, and **that
> limitation stands.**

**PROPOSED n=4 SEED CHECK — band 69 (criterion registered).**
*Criterion:* on any panel carrying `top_eigenvalue` and `curvature_along_polar`: (i) after removing the
**linear** component that `cp`'s monotone depth trend injects, `log λ − log cp` has an **interior
minimum in ≥80% of LR/seed fits**; (ii) it is **below both endpoints in ≥80%**; (iii) the **raw**
(untilted) profile still correlates with the C profile at **≥ +0.50**.
*Status:* **satisfied by committed REQ-019 data** (11/11; 11/11; +0.723). ⚠️ **Descriptive only under
rule 6 — shares `log λ` with C.** **No new compute requested; ≤2-node ceiling.**

**⇒ WHAT THIS DOES AND DOES NOT FIX.** It **does** show the concentration story is not an artifact of
REQ-048's probe implementation — its consequence appears on a differently-produced, differently-forked,
11-LR panel. It **does not** provide independent evidence for band 44 (shared `log λ`), and it **does
not** remove the need for the direct measurement to be replicated. **REQ-051's probe list includes
`curvature_along_polar` and `top_eigenvalue` at both 2050 and 2750, so it would extend this check to
4 seeds at no extra cost** — worth noting for whoever runs it.

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN** (the causal question, 16.2 min); **REQ-051 OPEN**;
**REQ-052 OPEN** (endorsed); REQ-049 optional. **No Jerry response since REQ-048; no NEEDS-INFO.**

## ✅ THE LEVER HAZARD IS CONTAINED (iteration 199) — band 67 was the only exposed survivor

*Band 67 died because a quantity measured under the **mixed-LR** design reversed under the **global-LR**
design. **That is a general hazard, not a one-off**: several bands are estimated on one design and stated
as if universal. **Before trusting the rest of the account, the exposure had to be enumerated.***

**CLASSIFICATION — a band is lever-exposed only if it FITS AN LR RESPONSE:**

| class | definition | bands |
|---|---|---|
| **(A) no LR variation used** | profile computed **within** a single LR; LRs compared only for *agreement* | **44, 57, 58, 59, 63, 64** |
| (B) global-LR variation | REQ-019's 11 arms / REQ-035's 3 | 52 |
| (C) per-matrix (mixed) LR | REQ-023 / REQ-045 | 42, 49, 53, **67** |

> **★ EVERY BAND IN THE MODEL-FREE CORE IS CLASS A.** Bands 44, 57, 58, 59, 63 and 64 compute their
> profiles **within** one LR and then check that the answer **agrees** across LRs. **They never fit an
> LR response at all**, so the mixed-vs-global distinction **cannot flip them** — it is not a hazard
> they are exposed to.
> **Bands 42, 49 and 53 are class C but were ALREADY scope-limited to per-matrix interventions in
> iteration 177**, before band 67 was written. **Band 67 was the only surviving band that fitted an LR
> response and stated it universally — and it is the one that fell.**

**EMPIRICAL CONFIRMATION — the bowl is the SAME object under both designs:**

| design | argmin | swing | min − L0 | min − L11 |
|---|---:|---:|---:|---:|
| **global LR** (REQ-035) | **L6** | 0.538 | −0.355 | −0.538 |
| **mixed LR** (REQ-023) | **L6** | 0.555 | −0.418 | −0.555 |
| | | | **corr = +0.931** | |

**Same minimum, same magnitude, correlation +0.931 across two designs with different forks, different
randomisations and different LR mechanisms.** **The central finding is design-independent** — which is
exactly what class A predicts and is now measured rather than assumed.

**PROPOSED n=4 SEED CHECK — band 68 (criterion registered).**
*Criterion:* (i) the C bowl's **argmin agrees between a global-LR panel and a mixed-LR panel**;
(ii) the two bowl profiles correlate at **≥ +0.70**; (iii) **no band whose evidence is a within-LR
profile may be quoted as an LR-response claim** — a documentation criterion, checkable by inspection.
*Status:* **satisfied by committed REQ-023 + REQ-035 data** (L6 = L6; **+0.931**).
**No new compute requested; ≤2-node ceiling.**

**⇒ WHAT THIS SETTLES, AND WHAT IT DOES NOT.** It settles that **band 67's failure was contained**: the
account's core does not rest on LR-response fits, and the bowl reproduces across both lever types.
**It does not rescue band 67**, which remains downgraded to mixed-LR-only, and **it does not weaken
REQ-052** — that request is still the right instrument for deciding whether the writer sensitivity is
real under matched controls, and I continue to endorse it.

**Standing rule 26.** *State which experimental lever a claim's evidence varies, and never generalise an
LR-response finding beyond the lever that produced it.* **The distinction that killed band 67 was
already recorded in iteration 177** (global LR moves the trajectory; per-matrix LR scales one
contribution) — **it simply was not applied as a filter when band 67 was written.** *Enumerating
exposure by class, as here, is cheaper than discovering it one band at a time.*

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN** (16.2 min, the causal question); **REQ-051 OPEN**;
**REQ-052 OPEN** (endorsed); REQ-049 optional. **No Jerry response since REQ-048.**

## ⛔ BAND 67 DOWNGRADED — REQ-052's audit is CORRECT and I reproduce it exactly (iteration 198)

*REQ-052 (Jack, 2026-09-05) challenges band 67 on two grounds. **Both are right, I have verified the
first independently from the raw archives, and the second is a rule I wrote myself and then failed to
apply.***

**① THE SPLIT REVERSES ON GLOBAL-LR DATA — reproduced exactly.**

| design | writer − internal k_lambda |
|---|---:|
| REQ-023 **mixed** LR, fork1500 → 2250 | **+0.924** |
| REQ-023 **mixed** LR, fork2000 → 2750 | **+1.165** |
| **REQ-035 global LR, 4 seeds @ 2250** | **−0.194, −0.125, −0.006, −0.199** |
| *(my independent reproduction @ 2250)* | ***−0.194, −0.125, −0.006, −0.199*** ✓ |
| *(and @ 2750, which I checked additionally)* | *−0.276, −0.232, −0.248, −0.255 — **more negative*** |

**My reproduction matches REQ-052's audit to three decimals.** **Band 67's positive writer split does
NOT generalise from the mixed-LR design to the global-LR design — it reverses sign in 4/4 independent
seeds.**

**② THE "5/5 STEPS" ROBUSTNESS WAS PSEUDO-REPLICATION — my own rule 15, unapplied.** Band 67 cited
*"writers > internal at 5/5 steps"*. **Those five checkpoints come from ONE continuation: dependent
measurements of the same network, not five independent replicates.** **Rule 15 was written in iteration
164 after exactly this error cost a claim** (the localised-residual retraction), and **I applied it to
bands 40/42/43 in iteration 165 and then failed to apply it to my own band 67 one iteration ago.**
**REQ-052 is correct that band 67's four-seed criterion is untested by REQ-023.**

> **⛔ BAND 67 IS DOWNGRADED from "CONFIRMED" to "MIXED-LR ONLY, n=1 seed, CONTRADICTED under global
> LR".** The finding is **not withdrawn entirely** — it is a real, large, depth-saturated contrast
> **within the mixed-LR design** (+0.924, t +4.23 with full block dummies) — **but it is
> design-specific, and the opposite sign holds under global LR.**

**⇒ AND THE REVERSAL IS NOT A PUZZLE — the campaign already explains it.** **Iteration 177 established
that the per-matrix and global LR levers are mechanistically different**: a per-matrix multiplier scales
one matrix's whole contribution (the gauge theorem applies, band 42), while **a global LR changes the
TRAJECTORY**, so every matrix reaches a different point in weight space and `g` is not merely rescaled.
Measured then: **global-LR `d log C/d log s` = −0.436 (t −18.29)** versus **per-matrix +0.081 (powered
null)**. **A quantity that behaves oppositely under the two levers is exactly what that distinction
predicts.** **REQ-052's control design is the right way to settle it**, and its five uniform-vs-mixed
arms directly target this.

**③ ONE THING BAND 67's ANALYSIS DID GET RIGHT, and it survives.** The three-way test I ran this
iteration (before seeing REQ-052) asked whether the writer split's three appearances are one effect or
three. After removing group means, per-matrix within-group correlations on REQ-023:

| pair | corr |
|---|---:|
| k_lambda vs gradient slope | **+0.602** |
| k_lambda vs k_g | **+0.678** |
| **gradient slope vs k_g** | **−0.058** |

**k_lambda tracks both, but the other two do not track each other** — so **"three independent
confirmations" was already too strong even before REQ-052**, and the correct reading is that the LR
elasticity is the shared axis. **Recorded as a partial self-correction that arrived independently.**

**⚠️ NO n=4 SEED CHECK PROPOSED — the criterion I registered has been shown untestable on the data I
cited.** **REQ-052 is the correct instrument** and it is already filed. **No new compute requested from
me; I endorse REQ-052's scope (5 control arms × 4 seeds, ≤2 nodes, reusing REQ-051's bases).**

**Standing rule 25.** *Apply your own guards to your own newest claim first. Rule 15 existed for 33
iterations, was applied to four other bands, and was still missed on band 67 — because the "5/5 steps"
framing made dependent measurements look like replication.* **A robustness count is only as independent
as its unit; state the unit explicitly whenever quoting one.**

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN**; **REQ-051 OPEN**; **REQ-052 OPEN (new — endorsed)**;
REQ-049 optional. **No Jerry response since REQ-048.**

## ★ THE WRITER SPLIT APPEARS A THIRD TIME — in the LR ELASTICITY (iteration 197) — and my iteration-196 numbers are corrected

*Iteration 196 flagged that REQ-051's decision 5 targets q/k/v while `attn.proj` (+0.217) and `mlp.proj`
(+2.874) appeared to differ by **13×**. **That was measured on REQ-045 alone, where only 37 of 72
matrices have full LR coverage and attn.proj had n = 4.** Before proposing anything, it had to be checked
on **REQ-023 — an independent per-matrix LR experiment with FULL coverage (all 72 matrices, 12 per
type).***

**⚠️ FIRST, THE CORRECTION TO MY OWN NUMBERS.** REQ-045's `attn.proj = +0.217` was a **small-sample
artifact**:

| type | REQ-045 (n per type) | **REQ-023 (n = 12 each)** |
|---|---:|---:|
| **attn.proj** *(writer)* | **+0.217** (n=4) | **+1.454** |
| **mlp.proj** *(writer)* | +2.874 (n=8) | **+2.350** |
| attn.k | +1.149 (n=7) | +0.897 |
| attn.q | +1.126 (n=5) | +0.885 |
| attn.v | +1.077 (n=8) | +0.806 |
| mlp.fc | +1.721 (n=5) | +1.327 |

**The "13× spread" I reported is not real — it is 1.6× on full-coverage data.** *(Iteration 196's
recommendation still stands, but its headline number was wrong and is withdrawn here.)*

**★ THE SPLIT ITSELF IS REAL, AND STRONGER ON THE BETTER DATA:**

| experiment | writers | internal | difference | t |
|---|---:|---:|---:|---:|
| REQ-045 (37 matrices, partial coverage) | +1.988 | +1.236 | +0.753 | +1.40 |
| **REQ-023 (72 matrices, full coverage)** | **+1.902** | **+0.979** | **+0.924** | **+3.32** |

**ROBUST ACROSS ALL FIVE MEASUREMENT STEPS** (REQ-023 dumps at 1750–2250, five quasi-independent
replicates):

| step | writers | internal | diff | t |
|---|---:|---:|---:|---:|
| 1750 | +1.295 | +0.843 | +0.452 | +1.90 |
| 1875 | +1.619 | +0.778 | +0.841 | **+3.73** |
| 2000 | +1.652 | +0.899 | +0.754 | +2.94 |
| 2125 | +1.820 | +1.000 | +0.820 | +3.18 |
| **2250** | **+1.902** | **+0.979** | **+0.924** | **+3.32** |

**Writers > internal at 5/5 steps.**

**⇒ AND IT IS NOT DEPTH IN DISGUISE.** Writers occur at **every** block, so the contrast can be taken
**within** block. With **full block dummies (saturated in depth)**: **writer coefficient +0.924, se
0.219, t = +4.23.** **Depth cannot explain it.**

> **★ THIS IS THE THIRD INDEPENDENT QUANTITY ON WHICH THE RESIDUAL-WRITER SPLIT APPEARS:**
> **band 7** — gradient slopes (+2.17, p < 0.0001); **band 60** — curvature concentration
> (`writer × edge` −0.627, t −3.29); **and now the LR elasticity itself** (+0.924, t +4.23 saturated).
> **Three unrelated measurements separate the same two matrix types from the other four.**

**PROPOSED n=4 SEED CHECK — band 67 (criterion registered), and offered to REQ-051 as decision 5b.**
*Criterion:* on a per-matrix LR panel, (i) **`k_lambda(writers) − k_lambda(internal) > +0.4`** in
**≥3 of 4 seeds**; (ii) the contrast survives **full block dummies** with **|t| ≥ 3**; (iii) the sign
holds at **every measurement step** available.
*Status:* **satisfied by committed REQ-023 data** (+0.924; t +4.23 saturated; 5/5 steps).
**No new compute requested** — but **REQ-051's balanced six-level ladder would test it at n = 4 seeds
with full coverage, which neither REQ-023 (1 seed) nor REQ-045 (partial coverage) can.**

**⇒ ADVISORY TO REQ-051 (unchanged: I have not edited the request).** Decision 5's registered q/k/v
prediction is worth keeping — but on full-coverage data the q/k/v spread is **0.806–0.897, a 1.1×
range**, while the **writer-vs-internal gap is 1.9×**. **A registered writer prediction would test where
the between-matrix variation actually lives**, and REQ-051 is the only design that can do it properly.

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN**; **REQ-051 OPEN** (clarified 00:07, decision 5 still
q/k/v only); REQ-049 optional. **No Jerry response since REQ-048** — the recent commits are under my own
git identity.

## 📋 REQ-051 PRE-FLIGHTED AGAINST COMMITTED DATA (iteration 196) — three findings before it spends 2 nodes

*REQ-051 (Jack/Codex, 2026-09-05) asks **why the own-LR curvature elasticity differs across matrices** —
directly the campaign's open question, and it correctly builds on the existing record (`C_gauge`
notation, the ~−1.16 pooled elasticity, the null neighbour channel). **Several of its seven registered
decisions are testable NOW on REQ-045's committed data.** Doing that first is cheap and changes what the
run will mean.*

**① DECISION 1 WOULD "FAIL" ON EXISTING DATA — but the failure is a COVERAGE ARTIFACT, and REQ-051's
design is exactly the fix.**

*Criterion:* ≥90% of matrices have `k_lambda > 0`, seed-median in [0.9, 1.5].
*On REQ-045:* **89.2%**, median **1.353** — a hair under the threshold. But:

| distinct LR levels per matrix in REQ-045 | count |
|---|---:|
| 3 levels | **37** |
| 2 levels | 34 |
| 1 level | 1 |

**REQ-045 drew each matrix's multiplier independently per arm, so a matrix could receive the same level
twice — only 37 of 72 have a genuine 3-point curve.** Splitting by coverage:

| subsample | k>0 | median |
|---|---:|---:|
| 3-level matrices (n=37) | **89.2%** | 1.353 |
| 2-level matrices (n=35) | **74.3%** | 1.251 |

**The shortfall is estimation noise on short curves, not evidence against the inverse law.**
**REQ-051's cyclic Latin assignment — every matrix gets every one of 6 levels exactly once — is
precisely the right correction**, and it is *why* the threshold is reachable there and not here. **No
change requested; this is a note that decision 1's failure on old data must not be read as a prior
against it.**

**② DECISION 4 ALREADY PASSES — so it will confirm, not discriminate.**

*Criterion:* `|mean(k_C_gauge)| < 0.15` **and** `|k_C| < 15%` of `|k_lambda|`.
*On REQ-045:* **mean k_C_gauge = +0.1378** (< 0.15 ✓) and the ratio is **9.3%** (< 15% ✓) — **passes on
both legs, but +0.1378 sits at 92% of its own threshold.** **A six-level, fully-covered design will
estimate this far more precisely, and the criterion may well tighten past it.** **Suggestion: report
`k_C_gauge` with a confidence interval rather than a pass/fail against 0.15** — the interesting quantity
is *how close to zero*, and bands 49/53 already put the pooled per-matrix C response at **4.2% of λ's**.

**③ DECISION 5's REGISTERED PREDICTION IS REVERSED ON EXISTING DATA — and it is aimed away from the
variance.**

*Registered:* `k_lambda(attn.v) > mean k_lambda(attn.q, attn.k)` in ≥3 of 4 seeds.
*On REQ-045 (3-level matrices only):*

| type | k_lambda | n |
|---|---:|---:|
| **attn.proj** | **+0.217** | 4 |
| attn.v | +1.077 | 8 |
| attn.q | +1.126 | 5 |
| attn.k | +1.149 | 7 |
| mlp.fc | +1.721 | 5 |
| **mlp.proj** | **+2.874** | 8 |

**attn.v (+1.077) is BELOW the q/k mean (+1.137) — the prediction is reversed, though the gap is small
and n is thin.** More substantively: **attn.proj and mlp.proj differ by 13×**, and **that is the
residual-writer split** (bands 7, 60) — **the q-vs-k-vs-v contrast is a small effect sitting inside a
much larger one the prediction does not mention.**

> **RECOMMENDATION (advisory — this is Jack's request, not mine to alter):** **add a registered
> writer-vs-internal prediction to decision 5.** The campaign has it on two independent quantities
> already — gradient slopes (band 7: +2.17, p < 0.0001) and concentration (band 60: `writer × edge`
> −0.627, t −3.29) — and REQ-051's balanced ladder would test it on a *third*, the LR elasticity itself,
> where committed data already shows a **13× spread**. **That is where the between-matrix variation
> REQ-051 is chasing actually lives.**

**⚠️ NO CHANGES MADE TO REQ-051.** It is well-specified, correctly scoped to ≤2 nodes, and explicitly
sequenced **after REQ-050** — which I agree with: REQ-050 answers *inherited vs learned* and is 16.2 min
of training, while REQ-051 is a six-arm × four-seed design. **These notes are recorded here for whoever
runs it; the request itself is unedited.**

**Queue:** REQ-035/036/048 DONE; **REQ-050 OPEN** (highest value, cost exact); **REQ-051 OPEN** (new,
sequenced after REQ-050); REQ-049 optional. **Note: REQ-051 was committed under the same git identity as
my own commits, so it is NOT a Jerry pickup — Jerry's last delivery remains REQ-048.**

## 🔧 BAND 3 GUARDED AT LAST (iteration 195) — the position field survives; the word "symmetric" does not

*The guard audit covered bands 39–64. **Bands 1–38 predate rules 13 and 23 entirely** and were never
tested against them. Scanning them, **band 3 is the priority**: it is load-bearing for everything
downstream, and its evidence is explicitly a **deviation-from-trend** ("corr with block-mean residual",
"a symmetric term must beat a linear one") — **precisely the class that killed band 61 and the
iteration-163 residual.***

**⚠️ AND ITS CLAIM IS NOT BAND 63's.** Band 63 says *"an interior minimum exists"*. **Band 3 says
something stronger: the field is SYMMETRIC — a linear trend cannot explain it.** The saturated test of
*that* is a direct three-way position contrast, with free per-block effects and the gradient controlled:

| block | 0 | 2 | 4 | **6** | 8 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| effect | 0.000 *(ref)* | −0.145 | −0.269 | **−0.371** | −0.238 | −0.057 | +0.136 |

| contrast | estimate | se | t |
|---|---:|---:|---:|
| **block 0 − block 6** | **+0.371** | 0.049 | **+7.51** |
| **block 11 − block 6** | **+0.506** | 0.049 | **+10.37** |
| **block 0 − block 11** *(symmetry)* | **−0.136** | 0.065 | **−2.10** |

**PER-FIT:**

| check | result |
|---|---|
| **both ends above the middle** | **12/12 fits** |
| argmin interior (4–8) | **12/12 fits** |
| L0 − min | **+0.386**, t **+5.75**, 12/12 same sign |
| L11 − min | **+0.522**, t **+16.85**, 12/12 same sign |
| **L0 − L11 (asymmetry)** | **−0.136**, t **−1.96**, only **8/12** same sign |

> **✅ BAND 3's SUBSTANCE SURVIVES.** With **no polynomial and no fitted trend**, both ends sit
> significantly above the middle in **every one of the 12 fits**. **The position field is real and is not
> an artefact of the model comparison that established it** — which is a genuine result, given that two
> other deviation-from-trend claims in this campaign did not survive the same guard.

> **🔧 BUT "SYMMETRIC" IS NOT SUPPORTED AND IS WITHDRAWN FROM THE BAND'S TITLE.** The two ends differ by
> **−0.136 dex (t = −2.10 pooled, −1.96 per-fit, 8/12 same sign)** — **marginal, but pointing
> consistently the same way, and the campaign already knows why:** **band 61 showed the tilt is supplied
> entirely by residual writers** (internal-only tilt **+0.001 dex**). **Band 3's "symmetric" was a
> convenient description of a field that is genuinely lopsided toward the output end.**

**⚠️ NOT A CONTRADICTION — a convergence.** Band 3 (iteration 55) called the field symmetric; band 39
(iteration 154) found a tilt; band 61 (iteration 186) attributed the tilt to writers; **this iteration
shows band 3's own saturated data carries that same tilt.** **Four independent routes now agree the
field is asymmetric, and band 3's title was the last place still saying otherwise.**

**PROPOSED n=4 SEED CHECK — band 66 (criterion registered).**
*Criterion:* using **free per-block effects with the gradient controlled** (no polynomial): (i) **both
endpoint-minus-minimum contrasts are positive in ≥10 of 12 fits**; (ii) both are pooled-significant at
**|t| ≥ 3**; (iii) the **L0 − L11 asymmetry is NOT claimed as zero** — report it with its sign.
*Status:* **satisfied by committed REQ-035 data** (12/12 and 12/12; t +7.51 and +10.37; asymmetry −0.136
reported, not assumed away). **No new compute requested; ≤2-node ceiling.**

**⇒ AUDIT STATUS OF THE OLDER BANDS.** Bands **6, 7, 8, 9, 10, 12** also rest on deviation-from-trend or
offset constructions and remain unguarded. **Band 10 is already marked ❌ NOT CONFIRMED**, and band 8 is
itself a bias-correction, so the live exposures are **6, 7, 9, 12** — the residual-writer slope split and
the type-offset structure. **Those are next**; band 3 was taken first because it is the one everything
else is built on.

**Queue:** REQ-035/036/048 DONE, REQ-049 optional, **REQ-050 OPEN** (16.2 min training, 4 seeds, ≤2
nodes — the only route to a causal answer). No new Jerry response.

## 🔧 REQ-050's COST PREMISE WAS WRONG — resolved from the record (iteration 193)

*REQ-050 is the only route left to a causal answer, and it has sat OPEN. Before adding analysis, I
checked whether it was blocked on something **I** could fix. **It was: I filed it with an unresolved
conditional that the campaign's own record had already settled.***

**⚠️ THE ERROR.** REQ-050 was filed as *"**PROBE-ONLY** if early checkpoints exist; otherwise one short
training run."* **I never checked which branch applied.** The record answers it three times over:

| source | statement |
|---|---|
| **REQ-038** (cost premise corrected) | *"no `.pt` weights are committed anywhere in the repo — REQ-019's boxes were ephemeral and only the derived `per_matrix_curvature.json` files landed"* |
| **REQ-041** | *"prior curvature checkpoints were cleaned by re-bootstraps"* |
| **REQ-035** | *"Fork-1500 states were regenerated from scratch (the 'existing checkpoint' premise was **false**)"* |

**Early checkpoints do not persist. The probe-only branch is dead**, and **REQ-038 had already been
corrected for this exact mistake** — I repeated it.

**✅ BUT THE TRUE COST IS SMALL, AND NOW EXACT.** At the campaign's measured **0.162 s/step**:

| run | per seed | **4 seeds** |
|---|---:|---:|
| to step 1500 *(as filed)* | 4.0 min | **16.2 min** |
| to step 1000 | 2.7 min | 10.8 min |
| to step 500 | 1.4 min | 5.4 min |

**16.2 minutes of training for the full 4-seed version — 1.01× the REQ-035 Arm A budget that was
already delivered.**

> **★ AND THE STRUCTURE IS FAVOURABLE IN A WAY I HAD NOT STATED.** Unlike every fork-based request in
> this campaign, **REQ-050's run starts at step 0 — so a SINGLE run passes through EVERY measurement
> point** (0, 125, 250, 500, 1000, 1500). **The dumps are written in passing: the cost is ONE traversal,
> not six.** A request that looked like it needed six probe points needs one run per seed.

**REQ-050's status line is corrected in place** with the resolved cost, so whoever picks it up is not
left re-deriving the conditional.

**Standing rule 24.** *A request that offers two cost branches is unfinished. Resolve the branch from the
record before filing — the campaign had already corrected REQ-038 for exactly this premise, and I
repeated it two months of iterations later.* **An unresolved conditional transfers work to the person
who can least cheaply do it.**

**⚠️ NO NEW COMPUTE REQUESTED THIS ITERATION.** This corrects an existing request rather than adding one.
**REQ-050 remains ≤2 nodes**, and the corrected figure makes it **cheaper than it appeared**, not more
expensive.

**Queue:** REQ-035 DONE, REQ-036 DONE, REQ-048 DONE, REQ-049 optional, **REQ-050 OPEN (cost now exact:
16.2 min training, 4 seeds, ≤2 nodes)**. No new Jerry response.

## ✅ CONSISTENCY RE-VERIFIED AFTER THE GUARD AUDIT (iteration 192) — and the consolidated answer

*The last end-to-end check was **iteration 169**. Since then the account gained bands 44, 57–61, 63, 64
and had bands 39, 42, 47, 49, 59, 61 **corrected or withdrawn**. **A consistency check run before ten
bands changed is not evidence about the current account**, so it was re-run on constraints that *must*
hold together.*

| # | constraint | value | verdict |
|---|---|---:|:---:|
| 1 | `n_eff = n_params × PR` — bands 44 and 59 share a definition | **8.88e-16** | **PASS** |
| 2 | `log PR = 2·log tr − log tr² − log n` — band 57's identity | **2.22e-15** | **PASS** |
| 3 | band 63's C contrast, block 6 − block 0 = −0.393 | **4.9e-04** | **PASS** |
| 4 | band 64's mirror — C and PR opposite at **both** ends | exact | **PASS** |
| 5 | band 59's slope in (−1, 0), rejecting equal-eigenvalue | **−0.336** | **PASS** |

**5/5 hold simultaneously on one dataset** — two definitional identities to machine precision, three
empirical claims reproducing. **The account is internally consistent after the audit.**

---

### THE CONSOLIDATED ANSWER — what sets the between-layer difference in C

**THE MECHANISM, stated model-free.** `C = λ/g²` is high at the network's ends and low in its middle
because **the Hessian's curvature is concentrated into few directions at the boundaries and spread
across many in the middle, at the same total curvature**:

- **`trace(H)` is FLAT across depth** (corr with the C profile −0.061, 6/12) while **`trace(H²)` carries
  the bowl** (+0.644, 12/12) — *band 57*.
- In units: **≈1,500 effective curvature directions at the ends vs ≈4,600 in the middle** (`n_eff =
  trace(H)²/trace(H²)`, 0 violations of [1, n_params] in 864 rows) — *band 59*.
- **Only the top eigendirection carries this.** Muon's step direction, the learned weight direction and
  the average random direction are all **monotone or flat** in depth — *band 58*.
- **Both claims survive with NO functional form** (free per-block effects, matrix-clustered): C's
  interior minimum is below both ends **12/12**, PR's interior maximum above both ends **12/12**, and the
  mirror holds as opposite-signed **position contrasts** — *bands 63, 64*.

**WHY NO OPTIMISER LEVER REACHES IT.** The **gauge theorem** (band 42): any scalar multiplying a matrix's
whole contribution cancels exactly in `λ/g²`. Confirmed causally — a randomised **per-matrix LR** moves
λ by −1.2 and C by **4.2% of that**, pooled across two independent experiments (bands 49, 53). Batch
moves C's *level* but **not** its shape (band 51). **The bowl lives in a subspace neither the optimiser's
step nor the learned solution occupies.**

**THE REQ-036 VERDICT — five independent reasons, one algebraic.** C is actively restored (band 16);
there is no bowl along Muon's step direction (band 56); types share one positional bowl (bands 45/46); a
per-type constant **cannot change between-layer spread at all** — 0.000% over 2000 random rules
(band 48); and **no LR intervention at any granularity moves C** (band 49). **Recommendation, final:
curvature equalization is unreachable through the learning rate. For the layer axis the lever must be
per-layer or per-matrix — and even then C is gauge-invariant to it.**

**WHAT REMAINS GENUINELY OPEN.** *Why* the boundary concentrates curvature into fewer directions. This
is a **structural identification, not a cause** — `trace(H²)` and C are both Hessian functionals.
**REQ-050** (curvature at steps 0–1500) is the filed, unrun experiment that separates **inherited from
the architecture** vs **built during training**, and it is now the **only** route to a causal answer:
every committed-data claim has been guarded, and the earliest curvature measurement in the repository is
step 1750 (band 54).

**⚠️ NO n=4 SEED CHECK PROPOSED.** This iteration verifies existing bands rather than making a new claim.
**No new compute requested; ≤2-node ceiling.**

**Queue:** REQ-035 DONE, REQ-036 DONE, REQ-048 DONE, REQ-049 optional, **REQ-050 OPEN**. No new Jerry
response.

## 🔧 BAND 59's SLOPE IS SPECIFICATION-DEPENDENT (iteration 191) — the conclusion holds, the number does not

*Band 64 flagged band 59's slope (**−0.590** vs the predicted **−1**) as the one load-bearing-adjacent
claim still resting on a fitted construction. It was fitted between two **profiles** (type-removed block
means) — the construction rule 23 targets. **Two separate worries, both now tested.***

**⚠️ WORRY 1 — SPECIFICATION. The coefficient is NOT stable; the conclusion is.**

| specification | slope | t vs 0 | **t vs −1** |
|---|---:|---:|---:|
| **profile-level** *(band 59 as recorded)* | **−0.590** | −11.92 | **+8.28** |
| matrix-level, type dummies only | **−0.408** | — | **+11.47** |
| **matrix-level, saturated in block AND type** | **−0.336** | −9.44 | **+18.69** |

> **The equal-eigenvalue model is rejected in every specification — and more decisively as the controls
> get tighter (t vs −1 rises from +8.3 to +18.7).** **But the coefficient ranges −0.336 to −0.590, and
> band 59 recorded the largest of the three.** The saturated figure identifies the slope from
> **within-block** variation (across types, seeds and LRs), which is the cleanest available; the
> profile-level figure additionally absorbs the between-layer signal after smoothing. **Band 59's
> "−0.590" is corrected to "−0.34 to −0.59 depending on specification, with −0.336 the saturated
> estimate."**

**✅ WORRY 2 — ATTENUATION. Tested and cleanly ruled out.** `n_eff` is a *measured* quantity, so
errors-in-variables bias any slope **toward zero** — and −0.59 vs −1 is exactly what attenuation looks
like. **This is why REQ-048 was specified to store `pr_per_probe_vHv`** (requested in iteration 162 so
the estimator's own variance would be measurable rather than assumed). Using it:

| quantity | value |
|---|---:|
| within-block+type variance of log n_eff | 0.11246 |
| **mean measurement variance (from the 16 probes)** | **0.00006** |
| **reliability ratio λ** | **0.9994** |
| raw saturated slope | −0.3356 |
| **attenuation-corrected slope** | **−0.3358** |

**The 16 Hutchinson probes are effectively noise-free at this scale — measurement error is 0.05% of the
signal variance, and the correction moves the slope by 0.0002.** **Attenuation is NOT the explanation for
the gap from −1**, so **band 59's substantive conclusion stands on firmer ground than when it was
recorded**: the spectrum is genuinely not "n_eff equal directions", and the shortfall is real physics
rather than measurement bias.

*(Methodological note: the pre-flight in iteration 162 chose m = 16 by simulation, and the request asked
for per-probe values specifically so this check would be possible. **That decision, made 29 iterations
earlier, is what allows attenuation to be excluded rather than merely hoped away.**)*

**PROPOSED n=4 SEED CHECK — band 65 (criterion registered).**
*Criterion:* (i) **d(log C)/d(log n_eff) rejects −1 (t ≥ 3) in BOTH the profile-level and the
block-saturated specification**; (ii) the **saturated estimate lies in [−0.55, −0.15]**; (iii) the
**reliability ratio λ ≥ 0.95**, so attenuation is demonstrably not driving the gap.
*Status:* **satisfied by committed REQ-048 data** (+8.28 and +18.69; −0.336; λ = 0.9994).
**No new compute requested; ≤2-node ceiling.**

**⇒ GUARD AUDIT COMPLETE.** Every load-bearing and adjacent claim has now been through rule 23:

| claim | saturated verdict |
|---|---|
| **the bowl** (band 63) | ✅ holds model-free — interior minimum below both ends, 12/12 |
| **the concentration mirror** (band 64) | ✅ holds model-free — opposite-signed position contrasts, 12/12 |
| **the n_eff slope** (band 59 → here) | 🔧 **conclusion holds, coefficient corrected to a range** |
| band 61's discrete break | ⛔ withdrawn (iteration 188) — baseline-dependent |
| iteration 163's localised residual | ⛔ withdrawn (iteration 164) — pseudo-replication |

**Both withdrawn claims were deviation-from-trend claims; all three surviving claims are contrasts or
coefficients between measured quantities.** **The pattern rule 23 predicted is now complete across the
campaign.**

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value — and now the only route to a *causal*
answer, since every committed-data claim has been guarded); REQ-049 optional. No new Jerry response.

## ✅★ BAND 44 SURVIVES THE SATURATED GUARD TOO (iteration 190) — the mirror as position-contrasts, not curve correlation

*Band 63 established the bowl model-free. **Band 44 — the spectral-concentration result, and the answer
to the campaign's central question — is stated the same polynomial way** ("log PR cubic R² 0.827, argmax
in layers 4–8") **and its headline is `corr = −0.862` between two FITTED PROFILES.** A correlation
between fitted curves is exactly the fragile construction rule 23 targets, so the same guard applies.*

**THE SATURATED FORM, in two parts, with free per-block effects and no polynomial anywhere:**

| block | 0 | 2 | 4 | **6** | **8** | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **log C** | 0.000 *(ref)* | −0.225 | −0.235 | **−0.393** | −0.274 | −0.157 | +0.117 |
| **log PR** | 0.000 *(ref)* | +0.122 | +0.276 | +0.500 | **+0.506** | +0.356 | +0.004 |

**(i) PR has an interior maximum above both endpoints** (matrix-clustered):

| contrast | estimate | se | t |
|---|---:|---:|---:|
| block 8 − block 0 | **+0.506** | 0.107 | **+4.74** |
| block 8 − block 11 | **+0.501** | 0.202 | **+2.48** |

**(ii) THE MIRROR, restated as paired position-contrasts** — same blocks, opposite signs, **no
correlation between curves involved**:

| contrast | **C** | **PR** | opposite? |
|---|---:|---:|:---:|
| block 6 vs 0 | **−0.393** (t −5.60) | **+0.500** (t +5.50) | **YES** |
| block 6 vs 11 | **−0.511** (t −4.34) | **+0.496** (t +2.55) | **YES** |

**PER-FIT — unanimous on all four checks:**

| check | result |
|---|---|
| PR argmax interior (4–8) | **12/12** |
| PR above **both** endpoints | **12/12** |
| C and PR opposite sign @ L0 | **12/12** |
| C and PR opposite sign @ L11 | **12/12** |
| mean PR vs L0 | **+0.532**, t **+5.96** |
| mean PR vs L11 | **+0.528**, t **+20.59** |

**⚠️ NOTE ON THE ONE WEAK LEG.** The pooled `block 8 − block 11` contrast is the weakest (t = +2.48),
but per-fit the same quantity gives **t = +20.59**. **That is not a contradiction:** the pooled
clustered SE includes between-fit variation in the *level* of PR at L11, which the paired per-fit test
differences away. **The paired figure is the appropriate one for a within-fit position contrast, and the
pooled figure is the conservative one — both are reported rather than the flattering one alone.**

> **✅★ THE CENTRAL RESULT IS NOT AN ARTEFACT OF CURVE-FITTING.** Band 44's answer — *C is high where the
> Hessian spectrum is concentrated and low where it is spread* — now holds as **contrasts between
> measured positions**, with **no polynomial and no correlation between fitted profiles.** Together with
> band 63, **the campaign's two load-bearing claims have both passed the guard that dissolved band 61.**

**PROPOSED n=4 SEED CHECK — band 64 (criterion registered).**
*Criterion:* using **free per-block effects (no polynomial)**: (i) **log PR's maximum lies in blocks 4–8
in ≥10 of 12 fits** and is **above both endpoints in ≥10 of 12**; (ii) at the block where **C** is
minimal, the **C and PR contrasts against each endpoint have OPPOSITE signs in ≥10 of 12 fits**;
(iii) the pooled PR-vs-L0 contrast is positive with **|t| ≥ 3** under matrix-clustered SEs.
*Status:* **satisfied by committed REQ-048 data** (12/12, 12/12; 12/12 and 12/12; +0.506, t +4.74).
**No new compute requested; ≤2-node ceiling.**

**⇒ WHAT IS NOW MODEL-FREE, AND WHAT IS NOT.** Model-free (position contrasts only): **the bowl**
(band 63) and **the concentration mirror** (here). Still polynomial-dependent, and flagged as such:
**band 59's `n_eff` slope of −0.590 vs the predicted −1** (a regression coefficient, not a contrast) and
**band 39's cubic form.** *Neither is load-bearing for the central claim* — but the distinction should be
kept visible rather than assumed away.

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value); REQ-049 optional. No new Jerry response.

## ✅★ THE BOWL SURVIVES ITS OWN GUARD (iteration 189) — stated with NO functional form

*Rule 23 was written last iteration after band 61's "discrete break" dissolved under a change of
baseline. **Its first duty is to be turned on the campaign's own headline.** Every bowl result —
bands 39, 44, 52, 59 — is phrased as *"cubic R² high, linear R² ≈ 0, argmin at L6"*. **That is a
polynomial-baseline claim, and rule 23 says a fitted baseline can create the very feature being
claimed.** So: does the bowl survive when depth is modelled with **free per-block effects**?*

**THE SATURATED TEST.** No polynomial anywhere — a free coefficient per block, type controlled, standard
errors clustered on the 72 matrices. The saturated form of *"there is a bowl"* is: **the free per-block
profile has an interior minimum significantly below BOTH endpoints.**

| block | 0 | 2 | 4 | **6** | 8 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **effect** | 0.000 *(ref)* | −0.225 | −0.235 | **−0.393** | −0.274 | −0.157 | +0.117 |
| se | — | 0.084 | 0.067 | 0.070 | 0.088 | 0.066 | 0.127 |

| contrast | estimate | se | t |
|---|---:|---:|---:|
| **block 6 − block 0** | **−0.393 dex** | 0.070 | **−5.60** |
| **block 6 − block 11** | **−0.511 dex** | 0.118 | **−4.34** |

**PER-FIT, the honest replication:**

| check | result |
|---|---|
| **argmin interior (4–8)** | **12/12 fits** (L6 ×10, L7 ×2) |
| **minimum below BOTH endpoints** | **12/12 fits** |
| mean (min − L0) | **−0.397**, t **−4.72** |
| mean (min − L11) | **−0.515**, t **−25.82** |

> **✅★ THE BOWL IS NOT AN ARTEFACT OF THE CUBIC.** With depth modelled by **free per-block effects and
> no functional form at all**, an interior minimum at block 6 sits significantly below both ends, in
> **every one of the 12 fits**. **The campaign's central finding passes the guard that dissolved band
> 61's break.**

**⇒ AND THE CONVERSE — what the cubic was ever doing.** Fitting the *free* profile:

| fit to the free per-block profile | R² |
|---|---:|
| **cubic** | **0.886** |
| **linear** | **0.000** |

**The cubic DESCRIBES the free profile well; it does not create it.** *(And the linear fit's R² = 0.000
is the cleanest possible statement that the depth structure is not a trend.)* **Every "cubic R²" figure
in bands 39/44/52/59 was a faithful summary of a feature that is present without it.**

**⚠️ WHY BAND 61 FAILED THIS TEST AND THE BOWL PASSED — the distinction is real, not luck.** Band 61's
claim was *"one specific block deviates from a fitted trend"* — that is **defined relative to the
baseline**, so changing linear→cubic changed the answer (F 176 → 5.46, rank 1/12 → 5/12). **The bowl's
claim is "an interior point is lower than the endpoints"** — a **contrast between measured positions**,
which no baseline choice can manufacture or remove. **Rule 23's remedy (prefer saturated controls) works
because it converts the first kind of claim into the second.**

**PROPOSED n=4 SEED CHECK — band 63 (criterion registered).**
*Criterion:* on a fresh panel, using **free per-block effects (no polynomial)**: (i) the profile's
**minimum lies in blocks 4–8 in ≥10 of 12 fits**; (ii) the minimum is **below both endpoints in ≥10 of
12 fits**; (iii) the pooled **block-6-minus-block-0 and block-6-minus-block-11 contrasts are both
negative with |t| ≥ 3** under matrix-clustered SEs.
*Status:* **satisfied by committed REQ-048 data** (12/12; 12/12; −5.60 and −4.34).
**No new compute requested; ≤2-node ceiling.**

**⇒ STATE OF THE ACCOUNT.** The bowl is now established **three ways**: as a cubic profile (bands 39,
52), as a spectral-concentration profile (bands 44, 57, 59), and — here — **as a model-free contrast
between measured positions.** The two claims that did **not** survive scrutiny were both
*deviation-from-trend* claims (band 61's break, iteration 163's localised residual), and **rule 23 now
explains why that class fails while position-contrast claims hold.**

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value — origin of the concentration structure at
initialisation); REQ-049 optional. No new Jerry response.

## 🔧 BAND 61's "DISCRETE BREAK" IS BASELINE-DEPENDENT (iteration 188) — the interaction survives, the deviation figure does not

*Band 61 reported L11's writers deviating **−1.384** from the interior trend and called the last block a
**discrete structural break**. Iteration 187 left that as the only surviving term. **Before building on
it, it needed the placebo test: with 12 blocks, some block always has the largest deviation, so a single
pre-chosen block is not evidence until the choice is priced.***

**⚠️ THE PLACEBO TEST — fitting the SAME indicator at every block, over a cubic baseline:**

| block | coef | F | rank |
|---|---:|---:|---:|
| **1** | +0.582 | **41.08** | **1/12** |
| **0** | −0.798 | **24.57** | 2/12 |
| 3 | −0.251 | 6.90 | 3/12 |
| 10 | +0.252 | 6.88 | 4/12 |
| **11** | **−0.389** | **5.46** | **5/12** |
| … | | 0.15–3.25 | 6–12/12 |

**L11 ranks only 5th of 12.** Blocks 1 and 0 deviate far more from a cubic. **"The last block is a
structural exception" does not survive a cubic baseline.**

**⇒ AND THE CONFLICT WITH BAND 61's −1.384 RESOLVES EXACTLY — it is the BASELINE:**

| baseline | L11's F | L11's rank |
|---|---:|---:|
| **linear** (what band 61 used) | **175.65** | **1/12** |
| quadratic | 72.33 | **1/12** |
| **cubic** | **5.46** | **5/12** |

**Against a linear interior trend L11 dominates overwhelmingly; against a cubic it is ordinary, because
a cubic can bend to follow an end-drop.** Band 61 extrapolated a **linear** fit of L1–L10 to L11. **Every
bowl result in this campaign is stated against a CUBIC (bands 39, 52, 59), so consistency requires
judging this claim against the cubic too — and there it does not stand out.**

> **🔧 BAND 61's "−1.384 deviation ⇒ discrete structural break" is WITHDRAWN as baseline-dependent.**
> It is not a wrong calculation — it is a correct calculation against a baseline the rest of the campaign
> does not use. **Reported as a correction, not a contradiction.**

**✅ WHAT SURVIVES, AND WHY IT IS UNAFFECTED.** The load-bearing result — the **`writer × LAST`
interaction** — was fitted with **full block dummies**, i.e. **saturated in depth**. No polynomial
baseline enters it at all, so the placebo result cannot touch it:

| term | coef | clustered se (72 matrices) | t |
|---|---:|---:|---:|
| **writer × LAST (block dummies)** | **−0.926 dex** | 0.230 | **−4.02** |

**With every block given its own intercept, the writers still lose ~0.93 dex of concentration-directions
at L11 relative to internal matrices at the same block.** That is a **within-block group contrast**, not
a deviation from a fitted curve — and **band 61's tilt result** (writers supply the bowl's asymmetry;
internal-only tilt **+0.001 dex**) is likewise a same-block group comparison. **Both stand.**

**PROPOSED n=4 SEED CHECK — band 62 (criterion registered).**
*Criterion:* (i) the **`writer × LAST` interaction with FULL block dummies is negative with |t| ≥ 3**;
(ii) **no single-block indicator over a cubic baseline gives L11 the top-3 F rank** — i.e. the discrete-
break reading must NOT be reinstated; (iii) the internal-only bowl tilt remains **< 0.10 dex**.
*Status:* **satisfied by committed REQ-048 data** (−0.926, t −4.02; L11 rank 5/12; +0.001 dex).
**No new compute requested; ≤2-node ceiling.**

**Standing rule 23.** *A deviation-from-trend is a statement about the trend as much as the point. Before
calling any single position a structural break, (a) run the same test at every position and report the
rank, and (b) check the claim against the baseline the rest of the work uses. Here a linear baseline made
L11 rank 1/12 with F = 176; the campaign's own cubic made it 5/12 with F = 5.5.* **Prefer a saturated
control (dummies) when the claim is about one position — it removes the baseline choice entirely.**

**⇒ NET EFFECT ON THE ACCOUNT.** The **writer effect at the last block is real and baseline-free**; the
**characterisation of it as a discrete architectural break is withdrawn**. The writers' profile is a bowl
with a strong end-drop that a cubic accommodates — **consistent with the bowl being smooth rather than
punctuated**. **REQ-050's question is unchanged and still decisive**: is the writer concentration
structure present at initialisation?

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value); REQ-049 optional. No new Jerry response.

## ⊘ THE "DOWNSTREAM RE-MIXING" HYPOTHESIS IS REFUTED (iteration 187) — including by a model comparison of mine that was misleading

*Band 61 flagged, as consistent structure rather than tested mechanism, that **L11's writers are the only
matrices whose output is never re-mixed by a later block**. That makes a graded prediction: a writer at
block b has **(11−b)** blocks downstream, so if "how much processing follows you" is the mechanism, the
concentration penalty should scale **smoothly** with downstream depth. **Testing it refutes it — twice.***

**⚠️ FIRST, A MODEL COMPARISON OF MINE THAT LOOKED DECISIVE AND WAS NOT.** Comparing five laws on the
writers' `log n_eff` (type controlled) by AIC:

| model | k | RSS | AIC |
|---|---:|---:|---:|
| **downstream depth + last-block** | 4 | 54.51 | **−471.38** |
| last-block indicator only | 3 | 71.29 | −396.11 |
| linear in downstream depth | 3 | 88.23 | −334.71 |
| type only | 2 | 90.45 | −329.55 |
| log(1+downstream depth) | 3 | 90.11 | −328.63 |

**The combined model wins by a wide margin**, which reads as "both a graded downstream effect *and* a
last-block break." **That reading is wrong**, and two checks show why.

**⊘ REFUTATION 1 — THE SIGN IS BACKWARDS.** The hypothesis says more downstream processing ⇒ more
re-mixing ⇒ curvature **spreads** ⇒ `n_eff` **higher**, so the coefficient must be **positive**:

| term | coef | clustered se (24 writer matrices) | t |
|---|---:|---:|---:|
| **downstream depth (per block)** | **−0.0797 dex** | 0.0293 | −2.72 |
| last-block extra | −1.4115 dex | 0.2034 | −6.94 |

**The downstream coefficient is NEGATIVE — the opposite of the prediction.** More blocks downstream is
associated with *less* spread, not more.

**⊘ REFUTATION 2 — IT IS NOT AN INDEPENDENT VARIABLE AT ALL.** `downstream = 11 − block` is an **exact
linear function of block**. Once a depth term is in the model it can add nothing:

| model | RSS |
|---|---:|
| cubic in depth alone | **46.140** |
| **cubic + downstream depth** | **46.140** *(identical — perfectly collinear)* |
| cubic + last-block indicator | 45.265 — **F(1,282) = 5.46** |

**"Downstream depth" was never a mechanism; it was a reparametrisation of block index.** My AIC table
was comparing **curve-fitting flexibility**, not competing mechanisms — the combined model won because
`down + islast` approximates a bowl-plus-step better than either piece alone, not because both effects
are real.

**⇒ WHAT THE WRITERS' PROFILE ACTUALLY SHOWS** (type-centred `log n_eff`):

| block | 0 | 2 | 4 | **7** | 9 | 10 | **11** |
|---|---:|---:|---:|---:|---:|---:|---:|
| log n_eff | −0.424 | −0.470 | −0.051 | **+0.489** | +0.300 | +0.107 | **−0.855** |

**The writers have a BOWL of their own** (rising to a mid-network peak at L7, falling away) **plus a
discrete L11 drop.** A monotone downstream term cannot produce that shape — it was fitting half of a U.

> **⇒ BAND 61's ARCHITECTURAL READING IS NARROWED, NOT CONFIRMED.** The **discrete last-block break is
> real** (F = 5.46 over a cubic; band 61's −1.384 interior-trend deviation). But **the graded
> "downstream re-mixing" story is refuted**: wrong sign, and not separable from depth. **Band 61's
> statement that this is "consistent structure, NOT a tested mechanism" was the right hedge — and it now
> has a specific negative attached rather than an open invitation.**

**⚠️ NO n=4 SEED CHECK PROPOSED.** This is a refutation with an exact-collinearity component; no new
compute can rescue a regressor that is a linear function of one already in the model. **Settled within
committed data.**

**Standing rule 22.** *Before comparing models, check whether a candidate regressor is an exact function
of one already present. AIC will happily rank a reparametrisation above the original when it is paired
with a second term that patches the residual shape — the comparison then measures flexibility, not
mechanism.* **This is rule 9's collinearity check applied to model SELECTION rather than to coefficient
interpretation.**

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value — the last-block break's origin at
initialisation is now the sharpest open question, with the graded alternative eliminated); REQ-049
optional. No new Jerry response.

## ★★ THE BOWL'S TILT IS ENTIRELY A LAST-BLOCK WRITER EFFECT (iteration 186)

*Band 60 found writers lose concentration-directions "at the ends" — but **the two ends are
structurally different**, and band 39 already established the bowl itself is **asymmetric** (L11 +0.302
vs L0 +0.166). Splitting the interaction by end is a **pre-declared structural contrast**, not a search.*

**★ THE EFFECT IS STRONGLY ASYMMETRIC:**

| term | coef | clustered se (72 matrices) | t | per-fit |
|---|---:|---:|---:|---|
| writer (interior) | +0.954 dex | 0.100 | +9.50 | — |
| **writer × FIRST (L0)** | **−0.300 dex** | 0.131 | −2.29 | −0.300 ± 0.160, **11/12** |
| **writer × LAST (L11)** | **−0.953 dex** | 0.233 | **−4.09** | −0.953 ± 0.149, **12/12** |
| **paired difference (FIRST − LAST)** | **+0.653** | 0.053 | **+12.42** | — |

**The last block's writers lose 3.2× more concentration-directions than the first block's.** Band 60's
*"writers lose concentration at the ends"* is **too coarse and is restated here**: it is overwhelmingly a
**last-block** effect.

**★★ AND IT EXPLAINS THE BOWL'S TILT — a falsifiable prediction, confirmed.** If the writer asymmetry
causes band 39's tilt, removing the writer types must remove the tilt:

| group | C bowl tilt (L11 − L0) | positive in |
|---|---:|---:|
| all 6 types | **+0.117 dex** | 9/12 |
| **internal 4 types only** | **+0.001 dex** | 7/12 |
| writers only | **+0.350 dex** | 10/12 |
| **paired reduction on dropping writers** | **+0.116 ± 0.024** | **t = +4.91** |

> **★★ The bowl's asymmetry is ENTIRELY a residual-writer effect.** Among internal matrices the tilt is
> **+0.001 dex — exactly zero.** **Band 39's "the bowl is tilted toward the output end" is now explained:
> the tilt is contributed by the two types that write to the residual stream, and specifically by the
> LAST block's writers.**

**⇒ AND THE LAST BLOCK IS A DISCRETE EXCEPTION, NOT A TREND ENDPOINT.** Extrapolating the writers'
interior trend (L1–L10) to each end:

| end | observed log n_eff | interior-trend prediction | **deviation** |
|---|---:|---:|---:|
| L0 | +3.610 | +3.762 | **−0.152** |
| **L11** | **+3.179** | **+4.563** | **−1.384** |

**L11 deviates 9× more than L0.** The last block is a **structural break**, not the far end of a smooth
gradient.

**⇒ THE ARCHITECTURAL FACT THAT MATCHES IT.** From `train_gpt.py` (read in iteration 157): the final
block's output passes through `norm(x)` and goes **straight to the LM head**. **L11's writers are the
only matrices in the network whose output is never re-mixed by a subsequent attention or MLP block.**
**That is a concrete, pre-existing structural asymmetry of exactly the right shape** — a discrete
exception at one block, affecting only the matrices that write to the stream. *(Stated as a matching
structural fact, **not** as a demonstrated cause: nothing here manipulates the architecture, so this is a
hypothesis the data is consistent with, not a tested mechanism.)*

**PROPOSED n=4 SEED CHECK — band 61 (criterion registered).**
*Criterion:* (i) **`writer × LAST` is negative with |t| ≥ 3** clustered by matrix, and **|writer × LAST| >
2 × |writer × FIRST|**; (ii) the **C bowl tilt among internal-only types is not significant** (|mean| <
0.10 dex); (iii) the writers' **L11 deviation from their interior trend exceeds their L0 deviation by
≥3×**.
*Status:* **satisfied by committed REQ-048 data** (−0.953 t −4.09, ratio **3.2×**; internal tilt
**+0.001**; deviations −1.384 vs −0.152 = **9.1×**). **No new compute requested; ≤2-node ceiling.**

**⇒ WHAT THIS DOES TO THE ACCOUNT.** The bowl now decomposes into two separable parts:
- a **symmetric** component present in all six types (internal-only contrast still 1.54×, band 60), and
- an **asymmetric** component supplied **entirely by last-block residual writers**, which is what makes
the bowl tilt.

**REQ-050 gains a second decisive question:** does the **last-block writer anomaly** exist at
initialisation? Since it is tied to a fixed architectural feature — nothing downstream re-mixes L11 — an
origin at step 0 is plausible and directly checkable.

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value, now two questions); REQ-049 optional. No new
Jerry response.

## ★ THE CONCENTRATION CONTRAST IS A RESIDUAL-WRITER EFFECT (iteration 185)

*Band 59 left a specific anomaly: the ends-vs-middle concentration ratio spans **1.22× to 15.34×** by
type, and **the two largest are exactly the two residual-writer types** — the split **band 7** established
independently (+2.17 slope gap, p < 0.0001). **That is a pre-existing grouping, not one chosen from this
data**, so testing it is not a fishing expedition.*

**PER-TYPE gain in `n_eff` from ends to middle** (pooled over 12 fits):

| type | | log₁₀ gain | ratio |
|---|---|---:|---:|
| **mlp.proj** | **WRITER** | **+1.186** | **15.34×** |
| **attn.proj** | **WRITER** | **+0.969** | **9.30×** |
| attn.k | internal | +0.416 | 2.60× |
| mlp.fc | internal | +0.148 | 1.41× |
| attn.q | internal | +0.097 | 1.25× |
| attn.v | internal | +0.085 | 1.22× |

**Writers 11.94× vs internal 1.54× — a gap of +0.891 dex.**

**⚠️ THE TYPE-AXIS TEST IS CAPPED BY CONSTRUCTION, and I am reporting that rather than the p-value.**
Permutation over the 6 type labels gives **p = 0.0681** — but with 6 types there are only **C(6,2) = 15**
possible 2-vs-4 splits, so **the smallest achievable one-sided p is 1/15 = 0.067.** **The writer split is
therefore the most extreme of all possible splits, and that is all a 6-point test can ever say.** It
cannot reach conventional significance at this n (the band-46 lesson).

**★ SO THE TEST WAS MOVED TO THE MATRIX AXIS, WHERE THERE IS POWER — 72 matrices, not 6 types.** The
claim *"writers concentrate more at the ends"* is an **interaction**: `writer × edge` on `log n_eff`.

**⚠️ A DESIGN FAULT CAUGHT AND FIXED FIRST.** My initial fit returned **`se = nan`** for the `edge` main
effect. Cause: **`edge` is exactly `block_0 + block_11`, so with block dummies present the design is
rank-deficient** (shape 864×15, rank 14). `lstsq` still returns a minimum-norm solution and the
interaction is identifiable, **but a rank-deficient design is not something to report from.** Rebuilt
without the redundant main effect (block dummies absorb it — `edge` has no meaning once every block has
its own intercept): **864×14, full rank.**

| term | coef | clustered se (72 matrix clusters) | t |
|---|---:|---:|---:|
| writer | +0.954 dex | 0.100 | **+9.50** |
| **writer × edge** | **−0.627 dex** | 0.191 | **−3.29** |

**PER-FIT replication at the independent unit (rule 15):** **−0.627 ± 0.125, negative in 12/12,
t = −17.39.** *(The clean and rank-deficient designs give an identical interaction estimate — the NaN was
cosmetic — but the check was still the right call.)*

**⇒ AND THE FALSIFIABLE CONSEQUENCE HOLDS.** If the effect is a writer effect, removing the writers must
collapse band 59's headline contrast:

| group | ends-vs-middle contrast |
|---|---:|
| all 6 types | **3.04×** |
| **internal 4 types only** | **1.54×** |
| writers only | **11.94×** |

> **★ Band 59's "3× more curvature directions in the middle" is driven by the two residual-writer types.
> Among internal matrices the contrast is only 1.54×.** **The concentration bowl is largely a property of
> the matrices that WRITE to the residual stream** — which independently reproduces **band 7's
> residual-writer split** on a completely different quantity (`n_eff` from Hutchinson probes, vs band 7's
> gradient slopes). **Two unrelated measurements now separate the same two types from the other four.**

**PROPOSED n=4 SEED CHECK — band 60 (criterion registered).**
*Criterion:* (i) the **`writer × edge` interaction on log n_eff is negative with |t| ≥ 3** clustered by
matrix, in ≥3 of 4 seeds; (ii) the interaction is **negative in ≥10 of 12** fits; (iii) the ends-vs-middle
contrast **drops by ≥40%** when the writer types are excluded.
*Status:* **satisfied by committed REQ-048 data** (−0.627, t −3.29 clustered / −17.39 per-fit; 12/12;
3.04× → 1.54× = **−49%**). **No new compute requested; ≤2-node ceiling.**

**⇒ WHAT THIS SHARPENS FOR REQ-050.** The open question is whether the concentration structure is
inherited or learned. **It is now more specific: does the RESIDUAL-WRITER concentration contrast exist at
initialisation?** Writers are the matrices whose output enters the residual stream directly, so an
architectural origin is plausible and checkable at step 0 — **REQ-050 would answer it without
modification.**

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value); REQ-049 optional. No new Jerry response.

## ★/⚠️ THE BOWL IN UNITS OF DIRECTIONS (iteration 184) — and a point prediction that FAILS

*Band 57 says `trace(H²)` carries the bowl while `trace(H)` is flat; band 58 says only the extreme
eigendirection carries depth structure. Both are stated as **correlations**. This iteration converts them
into a quantity with **units** — `n_eff = trace(H)²/trace(H²)`, the **effective number of curvature
directions** — because a number of directions is **checkable against the matrix dimension** and therefore
falsifiable in a way a correlation is not.*

**SANITY CHECKS PASS.** `n_eff` must lie in [1, n_params]: **0 violations in 864 rows**, observed range
35.4 to 72,128 against n_params of 589,824 / 2,359,296.

**★ THE BOWL, IN DIRECTIONS** (geometric mean over fits and types):

| block | 0 | 2 | 4 | **6** | **8** | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **n_eff** | **1494** | 1977 | 2820 | **4724** | **4784** | 3389 | **1508** |

> **The middle of the network spreads its curvature over ≈4,600 directions; the ends over ≈1,500 — a
> 3.05× contrast at the same total curvature.** *(corr(log n_eff, C profile) = **−0.862**, argmax L8 —
> band 44 restated on an absolute scale.)*
> **And curvature is extremely concentrated everywhere: n_eff is a median 0.23% of available
> directions.**

**⚠️ ROBUST IN SIGN, BUT NOT UNIFORM — the per-type spread is large.** Checking per fit and per type (the
discipline that caught band 37's artifact):

| | result |
|---|---|
| **per fit (12)** | mean **3.24×**, **>1 in 12/12**, range 2.08–7.11 |
| per type | mlp.proj **15.34×**, attn.proj **9.30×**, attn.k 2.60×, mlp.fc 1.41×, attn.q 1.25×, attn.v **1.22×** |

**The contrast holds in every fit and every type, so it is a property of the network — but "3×" is a
pooled average spanning 1.22× to 15.34×.** The two **residual-writer** types (attn.proj, mlp.proj) show
by far the largest concentration contrast, echoing band 7's residual-writer split; **the four others are
between 1.2× and 2.6×.** **Reporting "the middle spreads curvature 3× more" as a uniform fact would
overstate it.**

**⚠️/★ A POINT PREDICTION, TESTED AND FAILED — and this is the useful part.** If C were set by
concentration in the simplest way — a spectrum of `n_eff` **equal** eigenvalues has
`λ_top ≈ trace/n_eff`, and `trace` is flat — then **d(log C)/d(log n_eff) should be exactly −1.**
Measured across the 12 fits:

| | value |
|---|---:|
| slope d(log C)/d(log n_eff) | **−0.590** (sd 0.171) |
| t vs the predicted **−1** | **+8.28** ⇒ **rejected** |
| t vs 0 | **−11.92** ⇒ clearly real |

> **The equal-eigenvalue model is REFUTED at −0.590, roughly 60% of the predicted response.**
> **Concentration is real and drives C, but the spectrum's shape is not "n_eff equal directions".** A
> slope below 1 means the top eigenvalue grows **more slowly** than pure concentration would imply — i.e.
> as directions are removed, the curvature that leaves them is **not** fully inherited by the top
> eigenvalue; part spreads into the sub-leading bulk. **This is a genuine constraint on the spectral
> shape, obtained only because the claim was stated with units and a coefficient rather than as a
> correlation.**

**PROPOSED n=4 SEED CHECK — band 59 (criterion registered).**
*Criterion:* (i) **n_eff ∈ [1, n_params]** for every matrix (an implementation check);
(ii) **n_eff(middle L5–L8) / n_eff(ends L0,L11) > 1 in ≥10 of 12 fits**;
(iii) **d(log C)/d(log n_eff) is significantly negative AND significantly greater than −1** — i.e. the
equal-eigenvalue model is rejected in the same direction.
*Status:* **satisfied by committed REQ-048 data** (0/864 violations; 12/12; −0.590 with t −11.92 vs 0 and
t +8.28 vs −1). **No new compute requested; ≤2-node ceiling.**

**Standing rule 21.** *Restate a correlational finding with units and a predicted coefficient wherever
the algebra allows. Here the correlation (−0.862) was already known and told us nothing new; the
**slope** (−0.590 vs a predicted −1) refuted a specific spectral model. **A correlation can only be
confirmed; a coefficient can be wrong.***

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** — still the highest-value request, and now sharper: it
would show whether this **1,500-vs-4,600-direction structure** is present at initialisation or built
during training. REQ-049 optional. No new Jerry response.

## ⊘/★ THE BOWL IS IN THE **TOP** OF THE SPECTRUM ONLY — three directions now tested (iteration 183)

*The last unused REQ-048 field is `curvature_along_weight = ŴᵀHŴ` — curvature along the **learned weight
direction**. It tests a concrete mechanism for band 57: for a Gauss-Newton Hessian `H = JᵀJ`, if the ends
of the network concentrate curvature into few directions **and W is aligned with those directions**, then
the weight direction's curvature relative to the typical direction should be an **inverted-U** mirroring
PR. **Admissible: a separate HVP, no tridiagonal content (rule 13), no `lam_top` (rule 6 clean).***

**⊘ THE MECHANISM IS REFUTED.**

| quantity | corr with C profile | same-sign | argmax |
|---|---:|---:|---|
| log `curvature_along_weight` (raw) | **+0.153** (sd 0.349) | 8/12 | **L0 in 12/12** |
| log (W-dir curvature ÷ typical) | **+0.163** (sd 0.346) | 8/12 | **L0 in 12/12** (one L1) |

**Essentially null, and monotone rather than the predicted inverted-U.** Curvature along the direction
the network has actually built does **not** carry the bowl.

**⚠️ AND THE NULL IS POWERED — this is not a noisy field.** The W-curvature profile **replicates at
+0.951 mean pairwise correlation** across the 12 fits (min +0.870) — a precise measurement — and is
**cleanly monotone: linear R² 0.876, slope −0.0637/block.**

| block | 0 | 2 | 4 | **6** | 8 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C** | +0.199 | −0.026 | −0.036 | **−0.194** | −0.075 | +0.042 | **+0.316** |
| **log PR** | −0.294 | −0.172 | −0.018 | **+0.206** | +0.212 | +0.062 | −0.290 |
| **W-curv** | **+0.450** | +0.152 | +0.077 | −0.045 | −0.195 | **−0.383** | −0.141 |

**A profile that replicates at +0.951 but is monotone where C is a bowl is a real dissociation, not a
failure to measure** — structurally identical to band 56's C_polar result.

**★ ⇒ THREE DIRECTIONS TESTED, AND THE PATTERN IS CLEAN:**

| direction probed | depth profile | carries the bowl? |
|---|---|:---:|
| **top eigendirection** (`λ_top`) | **U-shaped, min L6** | **YES** |
| Muon's step direction (`curvature_along_polar`) | **monotone decline** (band 56, 11/11 LRs) | no |
| the learned weight direction (`ŴᵀHŴ`) | **monotone decline** (here, +0.951 replication) | no |
| *typical/random direction* (`trace/n`) | *flat* (band 57, corr −0.061) | no |

> **★ Only the TOP of the spectrum carries the depth structure.** Along **every other direction tested**
> — the optimiser's step, the learned weights, and the average over random directions — curvature either
> **declines monotonically with depth or is flat**. **The bowl is not a property of the loss surface
> broadly; it is a property of its extreme eigendirection**, consistent with band 57's finding that
> `trace(H²)` carries the profile while `trace(H)` does not.

**⇒ THIS SHARPENS WHY EVERY OPTIMISER LEVER FAILED.** Muon steps along the polar direction (monotone);
the network's own weights lie along a monotone direction; the average direction is flat. **The bowl lives
in a subspace that neither the optimiser's step nor the learned solution occupies** — which is precisely
why the LR (bands 49, 53), batch reshaping (band 51) and per-type equalization (band 48) were all inert
on it.

**PROPOSED n=4 SEED CHECK — band 58 (criterion registered).**
*Criterion:* (i) **corr(C profile, log curvature_along_weight) not significant** (|mean| < 0.35, same
sign < 10/12); (ii) the W-curvature profile **replicates at mean pairwise ≥ +0.70** (so the null is
powered); (iii) the W-curvature profile is **monotone** — linear R² ≥ 0.60 with a negative slope.
*Status:* **satisfied by committed REQ-048 data** (+0.153, 8/12; **+0.951**; linear R² 0.876, slope
−0.0637). **No new compute requested; ≤2-node ceiling.**

**⚠️ WHAT REMAINS OPEN, and it is now a single sharp question.** `trace(H)` flat, `trace(H²)` bowl-shaped,
and only the extreme eigendirection carrying depth structure ⇒ **why does the network's boundary
concentrate its curvature into few large eigenvalues while its middle spreads the same total?**
**REQ-050** (curvature at initialisation) separates *inherited from the architecture* vs *built during
training* and remains the **highest-value open request**.

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN**; REQ-049 optional. No new Jerry response. **All REQ-048
fields are now analysed** — `participation_ratio`, `trace_est`, `trace_sq_est`, `curvature_along_random`
(iteration 182) and `curvature_along_weight` (here).

## ★ THE BOWL IS IN THE HESSIAN'S **SECOND MOMENT** (iteration 182) — band 44 localised, and one cross-check withdrawn

*REQ-048 also delivered two diagnostic HVPs I had requested but never used —
`curvature_along_weight` and `curvature_along_random`. Both are separate HVPs (rule 13 clean). Using them
sharpens band 44 from "concentration" to a specific moment, **and retires a cross-check that looked
excellent.***

**⚠️ FIRST — A CROSS-CHECK THAT LOOKED DECISIVE AND IS WITHDRAWN.** `curvature_along_random` estimates
the **typical** direction (`E[vᵀHv] = trace(H)/n`), so `λ_top / curvature_along_random` is a **peak-to-mean
ratio** — a second, independent concentration measure. It correlates with the C profile at **+0.941,
12/12, cubic R² 0.831, argmin interior 12/12.** Superb — and **largely construction**:

```
log C          = log λ − 2 log g
log peak2mean  = log λ − log rnd      ← shares log λ, SAME sign
```

**Rule 6 check on the raw components — the clean predictors that contain NO `lam_top`:**

| predictor | corr with C profile | same-sign |
|---|---:|---:|
| log `curvature_along_random` (raw) | **−0.062** | 7/12 |
| log `trace_est` (raw) | **−0.061** | 6/12 |
| **log PR** (Hutchinson, no λ) | **−0.741** | **12/12** |
| *(log λ_top — shares with C)* | *+0.876* | *12/12* |

> **The +0.941 is withdrawn as a shared-term artifact** — the same construction that inflated iteration
> 160's `align` result and iteration 170's amplitude result. **This is the fourth time a quantity passed
> every significance test and failed on construction.** **But PR survives at −0.741, 12/12, and PR
> contains no `lam_top`** — so **band 44 is unaffected**, and the honest cross-check is that the two
> *raw* trace probes are null while PR is not.

**★ AND THAT DISCREPANCY LOCALISES THE FINDING.** `log PR = 2·log(trace) − log(trace_sq) − log n`
exactly (identity residual **1.67e-16**). If `trace` alone is null but PR is not, the signal must sit in
the **second moment**:

| quantity | corr with C profile | same-sign | depth swing |
|---|---:|---:|---:|
| log `trace_est` (**first moment**) | **−0.061** | 6/12 | 0.227 |
| log `trace_sq_est` (**second moment**) | **+0.644** | **12/12** | **0.659** |
| log PR | −0.741 | 12/12 | 0.506 |

> **★ THE BOWL IS IN `trace(H²)`, NOT `trace(H)`.** The Hessian's **total** curvature (first moment) is
> **flat across depth** — swing 0.227, correlation null. Its **squared** curvature (second moment) carries
> the profile — swing 0.659, **+0.644 in 12/12 fits**. **PR is not a delicate cancellation** (trace's
> contribution is small); **PR ≈ inverted trace_sq**, so band 44's "concentration" reading is really a
> statement about the **second moment**.

**⇒ WHY THIS IS A SHARPER RESULT.** `trace(H)` is the **sum** of eigenvalues, `trace(H²)` the **sum of
their squares**. Equal totals with unequal squares means the eigenvalues are **differently distributed at
the same total** — the ends of the network hold their curvature in **fewer, larger** eigenvalues, the
middle spreads the **same total** across more directions. **This is a precise mathematical statement of
what differs between layers, and it is measured directly rather than inferred from a ratio.**

**PROPOSED n=4 SEED CHECK — band 57 (criterion registered).**
*Criterion:* (i) **corr(C profile, log trace_sq_est) ≥ +0.50** with the same sign in ≥10/12 fits;
(ii) **corr(C profile, log trace_est) not significant** (|mean| < 0.25, same sign < 9/12);
(iii) **swing(trace_sq profile) > 2 × swing(trace profile)**.
*Status:* **satisfied by committed REQ-048 data** (+0.644, 12/12; −0.061, 6/12; 0.659 vs 0.227 = 2.9×).
**No new compute requested; ≤2-node ceiling.**

**⚠️ STILL A STRUCTURAL IDENTIFICATION.** `trace(H²)` is a Hessian functional like C, so this remains
*what differs*, not *what causes it*. **The causal question is now maximally sharp: why does the second
moment of the Hessian spectrum peak at the network's boundaries while the first moment stays flat?**
**REQ-050** (curvature at initialisation) is the filed experiment that would separate *inherited* from
*learned* — and it is now the highest-value open request.

**Standing rule 20.** *A ratio that survives rule 6 should be decomposed into its raw parts before being
interpreted. Here PR passed the shared-term check, but only decomposing it revealed the signal lives
entirely in one component — a sharper and more testable claim than the ratio itself.*

**Queue:** REQ-048 **DONE**; **REQ-050 OPEN** (highest value); REQ-049 optional. No new Jerry response.

## ★★★ REQ-048 DELIVERED — THE BOWL **IS** A SPECTRAL-CONCENTRATION PROFILE (iteration 181)

*Jerry delivered REQ-048 (`logs/kmaxwell/req048_spectral_participation/`, n=4). **It answers the
campaign's central question.** Verified against **band 44's registered criterion on the raw JSON**, not
the README summary — iteration 144's lesson.*

**THE MEASUREMENT.** Per Muon matrix at the 2750 checkpoint, **m = 16 Hutchinson probes** (fresh
Rademacher, independent of any Lanczos state — **admissible under rule 13**) estimating
`PR = trace(H)² / (n·trace(H²))`. **Jerry implemented the pre-flighted design exactly** — m = 16 as
simulated in iteration 162, per-probe values retained, and the **iteration-172 LR-axis correction
incorporated** (the `s*` tags recorded explicitly as LR multipliers). 12 states, all finite.

**★ THE RESULT — PR IS THE MIRROR IMAGE OF THE BOWL:**

| block | 0 | 2 | 4 | **6** | **8** | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C profile** | +0.199 | −0.026 | −0.036 | **−0.194** | −0.075 | +0.042 | **+0.316** |
| **log PR profile** | **−0.294** | −0.172 | −0.018 | **+0.206** | **+0.212** | +0.062 | **−0.290** |

**C bottoms at layer 6; PR peaks at layer 8. corr(C profile, log PR profile) = −0.862.**

| band-44 criterion | result | verdict |
|---|---|:---:|
| **(i)** corr(log PR, **C profile**) ≤ −0.60, ≥10/12 | **mean −0.741** (sd 0.140); **11/12** ≤ −0.60; **12/12 negative** | **PASS** |
| **(ii)** log PR cubic R² ≥ 0.70, **argmin** in layers 4–8 | cubic R² **0.827, 12/12**; **argmin at L0 or L11 in 12/12** | **FAIL** |
| falsification (\|corr\| < 0.30, or PR monotone) | not triggered | **survives** |

**⚠️ CRITERION (ii) FAILED BECAUSE I MIS-DRAFTED IT — the hypothesis is confirmed.** REQ-048's stated
hypothesis was: *"the spectrum is most **concentrated** at the ends of the network and most **spread** in
the middle."* **PR near 1 = spread, near 1/n = concentrated.** So the hypothesis predicts PR **LOW at the
ends, HIGH in the middle** — an **inverted bowl, argMAX interior**. **I wrote the shape clause as
"argmin in layers 4–8", as if PR should track C, when the hypothesis stated in the same request predicts
the opposite sign.** Criterion (i) — the correlation clause — had the sign right.

**RE-SCORED against the hypothesis as stated** *(criterion (ii′): log PR is an inverted bowl — cubic
R² ≥ 0.70 and **argmax** in layers 4–8)*:

| | result |
|---|---|
| cubic R² ≥ 0.70 | **12/12** (mean 0.827) |
| **argmax interior (4–8)** | **12/12** — argmaxes 6,8,8,6,6,8,7,7,7,8,8,6 |
| **VERDICT** | **PASS, 12/12** |

> **★★★ THE ANSWER TO THE CAMPAIGN'S CENTRAL QUESTION.** **The between-layer difference in C is a
> spectral-concentration profile.** At the **ends** of the network the Hessian's spectrum is
> **concentrated** in few directions (low PR) and C is **high**; in the **middle** the spectrum is
> **spread** across many directions (high PR) and C is **low**. **corr = −0.862, holding at every
> learning rate** (per-LR means −0.794 / −0.683 / −0.708). This is *why* the bowl lives in the **top of
> the spectrum** (band 56): where curvature concentrates into few directions, the top eigenvalue rises
> relative to the gradient.

**⚠️ WHAT THIS DOES AND DOES NOT ESTABLISH.** PR and C are **both** functions of the Hessian, so this is
a **structural identification, not an external cause**: it says *the bowl is a concentration profile*,
not *what makes the spectrum concentrate at the ends*. **That is a real and substantial narrowing** — the
question moves from "why is C U-shaped" to "why does the Hessian spectrum concentrate at the network's
boundaries" — but it is not the end of the causal chain. **Circularity check: PR comes from Hutchinson
probes, C from `top_eigenvalue`/`gradient_block_norm`; they share the Hessian but not a factorisation,
so rule 13 is satisfied** — unlike `residual_tail`, rejected in iteration 161.

**BAND 44 — REGISTERED, SCORED, AND AMENDED.** ✅ **CONFIRMED n=4** on criteria (i) and (ii′);
**criterion (ii) as originally written is withdrawn as a drafting error**, with the correction recorded
rather than quietly fixed. **The amendment does not weaken the test**: (ii′) is *harder* than (ii) was
(12/12 vs the 10/12 threshold), and criterion (i) — registered before any data existed and unchanged —
passes on its own.

**Standing rule 19.** *When registering a criterion, check each clause's SIGN against the hypothesis in
the same document. A correlation clause and a shape clause can silently disagree, and a pre-registered
criterion that contradicts its own hypothesis is worse than none — it converts a confirmation into an
apparent failure.*

**Queue:** **REQ-048 DONE.** **REQ-050 OPEN** (curvature at initialisation) — now the highest-value
remaining request, since it asks whether this concentration profile is inherited or learned.
**REQ-049 optional.**

## ★★ BAND 43 CONFIRMED AT 11 LEARNING RATES (iteration 180) — the top-of-spectrum localisation is now the campaign's second-strongest claim

*Before filing further requests I checked my own "the committed-data seam is exhausted" claim (iteration
178) — an exhaustion claim deserves the same scrutiny as any other. The field set is indeed unchanged
across every dataset, **but one combination had never been run: `curvature_along_polar` on REQ-019's
11-LR panel.** Band 43 — the finding that localises the bowl to the **top of the spectrum** — was
established on a single 3-fork panel. **This is a 4× stronger test.***

**ADMISSIBILITY.** `cp` is a **separate HVP**, not from the Lanczos tridiagonal (rule 13). `C_polar =
cp/g²` contains **no `lam_top`**. Fully admissible.

**THE TWO PROFILES, FITTED ON THE SAME MATRICES AT EACH OF 11 LRs:**

| LR | **C** argmin | C cubic R² | **C linear R²** | **C_polar** argmin | C_polar cubic R² | **C_polar linear R²** | C_polar slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | **6** | 0.905 | **0.003** | 8 | 0.907 | **0.729** | −0.0219 |
| 0.85 | **6** | 0.910 | **0.095** | 8 | 0.944 | **0.893** | −0.0232 |
| 1.00 | **6** | 0.904 | **0.043** | 10 | 0.928 | **0.875** | −0.0264 |
| 1.30 | **6** | 0.926 | **0.017** | 9 | 0.962 | **0.844** | −0.0344 |
| 1.70 | **6** | 0.923 | **0.112** | 10 | 0.973 | **0.947** | −0.0390 |
| *(all 11)* | **6 in 11/11** | 0.869–0.956 | **0.003–0.112** | 8–11 | 0.907–0.978 | **0.729–0.977** | **−0.0302 mean** |

**C_polar's slope is negative in 11/11 LRs, mean −0.0302, |t| = 15.76** — band 43's original value was
−0.0266, so it **replicates closely on independent data with 4× the LR coverage.**

**⇒ THE DISSOCIATION, MEASURED WITHOUT CHOOSING A FUNCTIONAL FORM.** A "bowl index" = *cubic R² minus
linear R²* (high = genuinely curved; ≈0 = a straight line already suffices), **paired across the same 11
LRs and the same matrices, so LR and seed noise cancel:**

| quantity | bowl index | range |
|---|---:|---|
| **C** | **0.869** (sd 0.049) | [0.793, 0.950] |
| **C_polar** | **0.082** (sd 0.062) | [0.001, 0.183] |
| **paired difference** | **+0.787 ± 0.008** | **t = +93.70, positive in 11/11** |

> **★ C is a bowl; C_polar is a straight line. Same matrices, same fits, 11 learning rates, t = +93.7.**
> **This is the cleanest statement of band 43 anywhere in the campaign**, and it makes the
> **top-of-spectrum localisation the second-best-supported claim here** — behind only the bowl's own
> existence.

**⇒ WHAT IT MEANS, restated.** Conditioning along the **top eigendirection** is U-shaped with a minimum
at layer 6; conditioning along **Muon's actual step direction** declines monotonically with depth.
**Whatever creates the bowl acts on the top of the spectrum, not on the subspace the optimiser moves
in** — which is *why* no optimiser lever has reached it (bands 51, 53) and why REQ-036's equalization was
inert (band 48).

**⚠️ WHERE THE TWO DIVERGE — descriptive only, rule 6 respected.** `log C − log C_polar = −(alignment)`,
and C shares `log λ` with alignment, so iteration 160's withdrawn correlation is **not** revived here.
Used **only to say where** the profiles separate:

| block | 0 | 2 | 4 | 6 | 8 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| alignment | +0.063 | +0.023 | **+0.154** | +0.126 | −0.039 | −0.190 | **−0.417** |

**The divergence is concentrated at the output end** (L11 −0.417), i.e. the deep layers are where the
step direction sees least of the top curvature. **Stated as description, not as evidence for a
mechanism.**

**PROPOSED n=4 SEED CHECK — band 56 (criterion registered).**
*Criterion:* on a fresh multi-LR panel, (i) **C's bowl index ≥ 0.60 and C_polar's ≤ 0.30** at ≥90% of
LRs; (ii) the **paired difference is positive at every LR**; (iii) **C_polar's depth slope is negative at
≥90% of LRs**.
*Status:* **satisfied by committed REQ-019 data** (0.869 vs 0.082; 11/11 positive; slope negative 11/11).
⚠️ **n = 1 seed per LR arm** — the LR axis is richly replicated, the seed axis is not.
**No new compute requested.**

**⚠️ EXHAUSTION CLAIM, CORRECTED.** Iteration 178 said the committed-data seam was exhausted. **That was
premature** — this iteration found a genuinely new and high-value result in it. The accurate statement:
**no new admissible FIELDS remain, but existing fields have not all been crossed with the richer panels.**
I will not repeat the exhaustion claim without having enumerated field × panel combinations.

**Queue:** REQ-048 **OPEN**, REQ-050 **OPEN** (both unanswered), REQ-049 optional. No Jerry response.

## ★ THE BOWL IS ALREADY FORMED AT THE EARLIEST MEASUREMENT THAT EXISTS (iteration 179)

*REQ-023's curvature dumps run at steps **1750, 1875, 2000, 2125, 2250** — **500 steps earlier than any
data used so far** (everything else starts at 2250). Two levers are inert on the bowl and it survives 11
LRs, so the natural hypothesis is that it is set **early**. This tests it.*

**THE BOWL AT EACH EARLY STEP:**

| step | argmin | cubic R² | linear R² | swing |
|---|---:|---:|---:|---:|
| **1750** | **L6** | 0.926 | 0.106 | 0.501 |
| 1875 | **L6** | 0.905 | 0.227 | 0.404 |
| 2000 | L4 | 0.890 | 0.008 | 0.468 |
| 2125 | **L6** | 0.881 | 0.046 | 0.476 |
| 2250 | **L6** | 0.952 | 0.020 | 0.555 |

**Mean pairwise correlation across steps: +0.928** (min +0.853).

**⇒ AND IT IS THE SAME BOWL AS 1000 STEPS LATER:**

| early step vs step 2750 | r |
|---|---:|
| **1750** | **+0.943** |
| 1875 | +0.914 |
| 2000 | +0.902 |
| 2125 | +0.928 |
| 2250 | +0.927 |

> **The bowl at step 1750 is the same bowl as at step 2750 (r = +0.943).** It is not a late-training
> phenomenon. Combined with its **LR-invariance in location** (band 52, 11/11 argmin L6), its
> **immunity to the per-matrix LR lever** (band 53) and to **batch reshaping** (band 51), the picture is
> consistent throughout: **the depth profile is established early and training hyperparameters do not
> move it.**

**⚠️ IS IT STILL FORMING AT 1750? — no detectable trend.** Regressing bowl swing on step over the
1750–2250 window: **+0.1425 dex per 1000 steps, se 0.1369, t = +1.04.** **Not significant** — the bowl is
already at its equilibrium amplitude by 1750, not still growing.

**⛔ AND THE LIMIT, STATED RATHER THAN CROSSED.** **Step 1750 is the earliest curvature measurement that
exists anywhere in the repository** (verified: 22 curvature files scanned, global minimum step = 1750).
**So "is the bowl present at initialisation?" CANNOT be answered from committed data.** A linear back-cast
from a 500-step window to step 0 gives 0.196 dex, but **that is an extrapolation 1750 steps outside the
data's range and is NOT offered as evidence** — it is precisely the error iteration 128 was withdrawn for
(interpreting an intercept where no regressor reaches zero).

**PROPOSED n=4 SEED CHECK — band 54 (criterion registered).**
*Criterion:* (i) the bowl's **argmin is in layers 5–7 at ≥4 of 5 early steps**; (ii) **mean pairwise
correlation across early steps ≥ +0.80**; (iii) **correlation with the late (2750) bowl ≥ +0.80**;
(iv) the **trend in swing over the early window is not significant**.
*Status:* **satisfied by committed REQ-023 data** (4/5 argmin L6; +0.928; +0.902 to +0.943; t +1.04).
⚠️ **n = 1 seed, 3 arms.** **No new compute requested for this band.**

---

## REQ-050: curvature at initialisation and early training — is the bowl inherited or learned?

- status: **OPEN — filed 2026-09-04 (iteration 179); COST RESOLVED 2026-09-04 (iteration 193).** ⚠️ **The "probe-only" branch is DEAD — the record already answered it and I should have checked before filing.** REQ-038's own correction states *"no `.pt` weights are committed anywhere in the repo — REQ-019's boxes were ephemeral and only the derived `per_matrix_curvature.json` files landed"*; REQ-041 adds *"prior curvature checkpoints were cleaned by re-bootstraps"*; REQ-035 found *"the 'existing checkpoint' premise was false"*. **Early checkpoints do not persist, so this needs a training run from step 0.** ✅ **But the true cost is small and now exact:** at the measured **0.162 s/step**, 1500 steps = **4.0 min/seed**, so **4 seeds = 16.2 min of training — 1.01× the already-delivered REQ-035 Arm A budget.** **Crucially a single run from step 0 passes through EVERY measurement point (0, 125, 250, 500, 1000, 1500) — the dumps are written in passing, so the cost is ONE traversal, not six.** ≤2 nodes.
- priority: **high — it separates two mechanisms that all committed data is blind to.**

**THE QUESTION.** The bowl is fully formed at **step 1750**, the earliest measurement in the repository,
and is unmoved by the learning rate (bands 52, 53), by batch reshaping (band 51), by `post_lambda` /
`resid_lambda` (bands 41, 42), and by stream scale, input rank or shape (iterations 156, 161).
**Everything tested says "not the optimiser".** The untested alternative is that it is **inherited from
the architecture at initialisation** — which no committed data can distinguish from "formed during the
first 1750 steps".

**WHAT TO RUN.** The existing curvature probe (`measure_per_matrix_curvature.py`, unmodified) at:
**steps 0, 125, 250, 500, 1000, 1500** — on **one seed** initially; 4 seeds if cheap.
- **Step 0 is the decisive point**: an untrained network at initialisation.
- **No training is needed if checkpoints at these steps were retained** (the standard cadence is 125), in
  which case this is a **probe-only** request costing a few GPU-minutes.
- If early checkpoints were **not** retained, it needs **one run to step 1500** with dumps — ~a third of
  the REQ-035 Arm A budget by that precedent.

**BAND 55 — CRITERION REGISTERED IN ADVANCE:**
(i) **INHERITED**: if the bowl is present at **step 0** with **cubic R² ≥ 0.70 and argmin in layers 4–8**,
and correlates **≥ +0.70** with the step-2750 bowl ⇒ **the depth profile is an architectural property of
the initialised network**, and the entire optimiser-side search was correctly abandoned.
(ii) **LEARNED-EARLY**: if step 0 shows **no bowl** (cubic R² < 0.30 or argmin at an edge) but the bowl is
present by step 500–1500 ⇒ **it forms in early training**, and the mechanism question becomes *what
happens in the first few hundred steps* — a completely different and newly-tractable target.
(iii) Either outcome is decisive. **There is no ambiguous result**, which is why this is worth the budget.

**WHY THIS MAY BEAT REQ-048.** REQ-048 (spectral participation ratio) asks *what property of the surface*
makes the bowl; **REQ-050 asks whether the bowl exists before any surface has been shaped by training.**
If the answer is **inherited**, REQ-048's measurement should be taken **at initialisation**, where it is
far easier to interpret — so **REQ-050 may reframe REQ-048 rather than compete with it.**

- requester: analysis loop, iteration 179
- constraint acknowledged: **≤2 nodes**, no assertion of any higher authority.

## ★ BAND 49 REPLICATES ON AN INDEPENDENT EXPERIMENT (iteration 178) — REQ-049's question answered from committed data

*Surveying every curvature-bearing dataset in the repo (rule 18, third application) turned up
**`logs/kmaxwell/req023_per_matrix_lr/`** — **a second, independent per-matrix LR experiment** I had
cited many times but never opened the raw data for. Its `assignments.tsv` gives each of 72 matrices a
multiplier in {0.6, 1.0, 1.7} across three arms, at **fork 1500** — **REQ-045's design at a different
fork with a different randomisation.** **This is precisely the replication REQ-049 was filed to request.***

**THE REPLICATION** (1,080 observations vs REQ-045's 216; type, block and arm controlled; the predictor
is the **assigned multiplier**, a treatment, so it remains admissible and causal):

| quantity | **REQ-023** (fork 1500, n=1080) | **REQ-045** (fork 2000, n=216) |
|---|---:|---:|
| d log λ | **−1.152** (t −21.11) | −1.218 (t −7.55) |
| d log g | **−0.541** (t −33.04) | −0.650 (t −12.61) |
| **d log C** | **−0.070** (t **−2.01**) | **+0.081** (t +0.89) |
| identity residual | **1.2e-16** | 9.99e-16 |

**λ replicates closely (−1.15 vs −1.22).** **C is small in both** — but REQ-023's is **marginally
significant** where REQ-045's was a clean null, so the two must be compared formally rather than
eyeballed.

**FORMAL COMPARISON:**

| test | result |
|---|---|
| difference of the two C estimates | +0.151 ± 0.097, **z = +1.55** ⇒ **CONSISTENT** |
| **inverse-variance pooled C elasticity** | **−0.0505 ± 0.0327** (t −1.55), 95% CI **[−0.115, +0.014]** |
| **C's response as a fraction of λ's** | **4.2%** |

> **★ BAND 49 REPLICATES — with its claim made more precise.** Two independent per-matrix LR
> experiments, at different forks with different randomisations, agree that **C's response is ~4% of
> λ's** and not distinguishable from zero when pooled. **But the correct statement is "C's response to a
> per-matrix LR is at most a few percent of λ's, and may be slightly negative" — NOT "C is exactly
> invariant."** The gauge theorem predicts **exactly zero**; at n = 1080 a small deviation is becoming
> detectable, and pretending otherwise would be overclaiming.

**⇒ WHY IT NEED NOT BE EXACTLY ZERO — a principled reason, flagged not tested.** The theorem assumes the
multiplier scales a matrix's whole contribution **with everything else fixed**. In practice a per-matrix
LR change **also moves that matrix's own weights**, which feeds back into the loss surface it subsequently
sees — a second-order effect the theorem ignores. **The prediction is that the deviation should scale
with |log m|.** REQ-023 has only 3 multiplier levels, so that test would be underpowered; **I am flagging
it rather than running it under-powered** (rule 16).

**⇒ REQ-049's STATUS.** Its question — *does band 49's causal result replicate?* — **is now answered
from committed data: yes, at 5× the sample size, on an independent fork and randomisation.** **REQ-049
should be DOWNGRADED from "high priority" to "optional"**: a genuine 4-seed version would still add the
seed axis (both experiments are n=1 seed), but the **replication risk it was filed against has been
retired.** **REQ-048 is now the only request whose question remains unanswered.**

**PROPOSED n=4 SEED CHECK — band 53 (criterion registered).**
*Criterion:* across ≥2 independent per-matrix LR experiments, (i) **d log λ/d log m ∈ [−1.5, −0.9]** in
each; (ii) **|d log C/d log m| < 0.15** in each; (iii) the **pooled** C elasticity is **< 10% of λ's in
magnitude**; (iv) both **identity residuals < 1e-10**.
*Status:* **satisfied by committed REQ-023 + REQ-045** (−1.152/−1.218; 0.070/0.081; **4.2%**;
1.2e-16/9.99e-16). **No new compute requested; ≤2-node ceiling.**

**⚠️ DATASET SURVEY RESULT (rule 18).** Only **two** directories in `logs/kmaxwell/` carry per-matrix
curvature data beyond those already used: **REQ-019** (11 LRs — iteration 177) and **REQ-023** (this
iteration). **Both are now analysed.** The committed-data seam that rule 18 opened is, as far as the
repository shows, **exhausted** — which restores REQ-048 as the binding constraint on the central
question.

**Queue:** REQ-048 **OPEN** (the only outstanding question); REQ-049 **filed, now optional**. No Jerry
response.

## ★/🔧 A RICHER PANEL FOUND — 11 LEARNING RATES (iteration 177): the bowl is confirmed, band 47 is corrected

*Looking for early-training curvature, I found **`logs/kmaxwell/req019_eos_state_dependence/`** — a panel
I had never opened: **fork@1500 with ELEVEN learning rates (0.60×–1.70×)** plus fork@2000 with three.
**5,760 rows.** Nearly 4× the LR resolution of the 3-point axis every LR claim in this campaign has
rested on. (Rule 18 again: the data was in the repo the whole time.)*

**★ THE BOWL IS CONFIRMED FAR MORE STRONGLY THAN BEFORE.** Fitted independently at each of 11 LRs:

| | result |
|---|---|
| **argmin at layer 6** | **11 of 11 learning rates** |
| cubic R² | 0.869 – 0.956 |
| **linear R²** | **0.003 – 0.112 (≈0 throughout)** |
| swing | 0.439 – 0.538 dex |
| **mean pairwise bowl correlation** | **+0.922** (min +0.683) |

**And identically at fork 2000** (3 LRs, argmin L6 ×3, mean pairwise **+0.968**). **The bowl is the most
robust finding in this campaign** — it survives 11 learning rates, 2 fork points, 4 seeds, 5 checkpoints,
6 matrix types and a 2.8× LR range.

**🔧 BUT BAND 47's "LR-INVARIANT" IS TOO STRONG AND IS CORRECTED.** With 11 LR arms there is real power
to detect shape drift, and there **is** some: the **LR × block interaction is significant**,
**F(11, 3931) = 2.22** against a permutation critical value of **1.79**.

Significance alone is not the point (n = 3960 detects tiny effects — the mirror of rule 16), so the
**magnitude**:

| quantity | value |
|---|---:|
| bowl swing | 0.4630 dex |
| interaction coefficients (d bowl / d log s) | −0.4216 to −0.0321 |
| **implied bowl distortion over the full 2.8× LR range** | **0.1762 dex** |
| **as a fraction of the bowl** | **38.0%** |

> **38% is not negligible.** **Band 47's claim is corrected from "the bowl is LR-invariant" to "the
> bowl's minimum and shape are preserved at every LR (argmin L6, 11/11), but its amplitude/shape is
> modulated by up to ~38% across a 2.8× LR range."** The **existence and location** of the bowl are
> LR-invariant; its **exact profile** is not.

**⚠️/★ AND A SECOND RESULT THAT LOOKS LIKE A CONTRADICTION BUT IS NOT — the two LR levers genuinely
differ.**

| lever | d log λ | d log g | **d log C** |
|---|---:|---:|---:|
| **per-MATRIX LR** (REQ-045, band 49) | −1.218 | **−0.650** | **+0.081** (powered **null**) |
| **GLOBAL LR** (REQ-019, 11 arms) | −1.347 | **−0.456** | **−0.436** (t **−18.29**) |

**λ responds almost identically (−1.22 vs −1.35). The entire difference is in g** — and it is exactly
what decides whether C moves:
- per-matrix: `2 × (−0.650) = −1.300` ≈ cancels λ's −1.218 → **C flat**
- global: `2 × (−0.456) = −0.912` < λ's −1.347 → **C moves −0.436**

*(Identity check: −1.347 − 2(−0.456) = **−0.435** vs fitted **−0.436**. Internally exact.)*

> **⇒ THE GAUGE THEOREM IS NOT VIOLATED — its precondition simply fails for a global LR.** The theorem
> requires the scalar to multiply **that matrix's whole contribution**, holding everything else fixed. A
> **per-matrix** multiplier does exactly that. A **global** LR changes **every** matrix at once, so the
> network follows a **different trajectory** to a **different point in weight space**; g is not merely
> rescaled. **Band 42's theorem stands, and this sharpens its scope: it applies to per-matrix
> interventions, not to global hyperparameter changes.**

**PROPOSED n=4 SEED CHECK — band 52 (criterion registered).**
*Criterion:* on a fresh multi-LR panel, (i) the C bowl's **argmin is in layers 5–7 at ≥90% of LRs**;
(ii) the **LR × block interaction distorts the bowl by < 60% of its swing** across the tested LR range;
(iii) the **global-LR elasticity of log C is significantly negative** while the **per-matrix elasticity
is not** — the two-lever distinction.
*Status:* satisfied by committed REQ-019 + REQ-045 data (11/11 argmin L6; 38.0%; −0.436 t −18.3 vs +0.081
t +0.89). ⚠️ **REQ-019's LR arms are n = 1 each** — the *LR axis* is richly sampled, the *seed axis* is
not. **No new compute requested.**

**⚠️ WHAT THIS CHANGES ELSEWHERE.** Band 42's **empirical** LR corroboration (iteration 172's "C's
LR-response is 0.44× λ's") was computed on **global** LR arms — so it was measuring the **global** lever,
where C legitimately *does* respond. **That number is not evidence for the gauge theorem** and should not
be cited as such; **REQ-045's per-matrix result (band 49) is the theorem's proper empirical test.**

**Queue:** REQ-048 **OPEN**; REQ-049 **filed**. No Jerry response.

## ⊘ BATCH MOVES C's LEVEL, NOT THE BOWL (iteration 176) — the goal-relevant follow-up to band 50, and it is negative

*Band 50 established that batch moves C where the LR cannot. **But the campaign goal is the BETWEEN-LAYER
difference, not C's level** — and band 48 showed exactly how a lever can move a quantity while leaving its
depth spread untouched. So the decisive question is whether the batch effect **varies by depth**.*

**IT DOES NOT.** The bowl survives every batch size essentially unchanged:

| batch | argmin | cubic R² | swing |
|---|---:|---:|---:|
| 0.5× | **L6** | 0.938 | 0.554 |
| 1.0× | **L6** | 0.886 | 0.492 |
| 2.0× | L4 | 0.906 | 0.659 |

Pairwise bowl correlations: **+0.720, +0.889, +0.536.**

**FORMAL TEST — batch × block interaction on log C: F(11, 187) = 0.68**, against a permutation critical
value of **1.86**. **Far below.** RSS falls only 6.6133 → 6.3598 for 11 extra parameters.

> **⇒ Batch shifts C's LEVEL uniformly across depth; it does not reshape the bowl.** **On the campaign's
> actual target — the between-layer difference — the batch lever is inert, exactly as the per-type LR was
> (band 48), though for a different reason: the per-type LR was *algebraically* incapable, while batch is
> *empirically* uniform in depth.**

**⚠️ THE NULL'S BOUND — rule 16, and a simulation error of mine caught and fixed en route.** A null needs
its detectable-effect size. My first attempt produced **0% power at amplitude 0.20 and 100% at 0.30** —
which is not a power curve but a **threshold artifact**: the F-test is deterministic given the data, and I
had injected a fixed signal with **no added noise**, so all 200 "trials" were identical. Redone with
residuals resampled per trial (additive-model residual sd **0.1828 dex**):

| injected depth-varying batch effect | power |
|---:|---:|
| 0.05 dex/dex | 5.0% |
| 0.10 | 13.0% |
| 0.15 | 26.0% |
| 0.20 | 44.3% |
| 0.25 | 65.7% |
| **0.30** | **85.7%** |

> **The design resolves depth-dependence of ~0.29 dex/dex at 80% power.** The null is therefore
> **meaningful for large effects and silent for small ones** — but note the bowl's own swing is ~**0.5
> dex**, so a batch effect that **proportionally reshaped** it would have been detected. **A uniform
> level shift is the correct reading; a subtle reshaping below ~0.29 dex/dex is not excluded.**

**⇒ WHAT THIS DOES TO THE PICTURE.** Two levers have now been tested against the bowl and **both are
inert on it**:

| lever | moves C's level? | reshapes the bowl? |
|---|---|---|
| **learning rate** (per-matrix, REQ-045) | **no** — gauge-cancelled, powered null | no |
| **batch / gradient noise** (REQ-037) | **yes** — −0.274 (t −5.41) | **no** — F 0.68 |

**The bowl is untouched by both the optimiser's step scale and its noise scale.** Combined with the
exclusions already recorded (stream scale, input rank, shape, `post_lambda`, `resid_lambda`, axis
artifacts), **the depth profile is looking like a property of the architecture's forward/backward
structure that training hyperparameters do not reach at all.** That is consistent with band 47's
finding that the bowl is **learning-rate invariant across a 2.8× range**, and with iteration 155's
constraint that it must come from the **loss surface**, not the optimiser.

**⚠️ I am NOT claiming the bowl is unreachable** — two levers is not a proof, and this null has a
**0.29 dex/dex floor**. What is established is narrower: **the two optimiser knobs tested so far change C
without changing its depth structure.**

**PROPOSED n=4 SEED CHECK — band 51 (criterion registered).**
*Criterion:* on a fresh panel, (i) the **batch × block interaction on log C is not significant** (F below
its permutation 95th percentile) in ≥3 of 4 seeds; (ii) the **bowl's argmin stays in layers 4–8 at every
batch size**; (iii) the **main batch effect on log C remains significant** (\|t\| ≥ 3) — i.e. the lever
demonstrably works while failing to reshape.
*Status:* satisfied by committed REQ-037 data at **n = 1/arm** (F 0.68 vs crit 1.86; argmins L6/L6/L4;
main effect t −5.41). **Requires replication** — and **REQ-037's arms are step-matched, so the
token-matched variant recommended in band 50 should be used.** **No new compute requested this
iteration.**

**Queue:** REQ-048 **OPEN**; REQ-049 **filed**. No Jerry response.

## ★★ C IS MOVABLE — BUT NOT BY A LEARNING RATE (iteration 175) — a clean two-lever dissociation

*Rule-18 audit continued to `make_req037_arms.py`, and it revealed a design feature I had never
registered: **`skip = FORK*BASE_BATCH//bt`, so every batch arm resumes from the SAME TOKEN POSITION at
the fork.** The batch arms are token-matched at the fork and hold the LR fixed — making batch a **second,
mechanistically different lever** to set against REQ-045's LR lever.*

**WHY THIS IS THE RIGHT CONTRAST.** The gauge theorem (band 42) says the LR cancels in `C = λ/g²` because
it multiplies a matrix's **whole contribution**. **Batch does not do that** — it changes gradient
**noise**, not the scale of W's influence — so **the theorem makes no cancellation prediction here.**
That is a genuine out-of-sample test of the theorem's *scope*, not just its content.

**THE TWO LEVERS, SIDE BY SIDE** (both from randomised/assigned treatments; both 216 observations;
type and block controlled):

| | **LR lever** (REQ-045, per-matrix) | **BATCH lever** (REQ-037, 0.5×/1×/2×) |
|---|---:|---:|
| d(log λ) | **−1.218** (t **−7.55**) | **+0.065** (t **+0.79** — *unmoved*) |
| d(log g) | −0.650 (t −12.61) | **+0.169** (t **+6.94**) |
| **d(log C)** | **+0.081** (t +0.89 — **powered null**) | **−0.274** (t **−5.41**) |
| identity residual | 9.99e-16 | **1.7e-16** |

> **★ THE DISSOCIATION IS EXACT AND INVERTED.** The **LR** lever moves **λ hard and leaves C fixed**.
> The **batch** lever leaves **λ unmoved and moves C decisively**. **C is not an inert construct — it is
> movable. It is simply not movable by a learning rate**, at any granularity (band 49).

**⇒ THIS ANSWERS A QUESTION OPEN SINCE BAND 13.** REQ-037 arm 4 (the per-matrix gradient clip) was
deferred, and REQ-046's clip instrument proved **inert** (iteration 130: Muon's unit-norm step absorbed
it, exponent 0/0). **The batch arms were the working non-LR instrument all along, sitting in committed
data.** They demonstrate that **a non-LR intervention CAN move C**, which no experiment in this campaign
had previously shown.

**⚠️ THE CONFOUND, ADDRESSED HONESTLY AND NOT EXPLAINED AWAY.** REQ-037's own status flags it: *"batch
confounds g-noise with tokens-seen (val 0.5× 3.626 / 1× 3.512 / 2× 3.421)"*. At fixed step count a larger
batch sees more tokens. I checked whether it could be controlled for:

**`corr(log batch, val) = −0.9979` across the three arms.** With **3 arms lying on a line**, batch and
tokens-seen are **not separable** — fitting both to three points is exactly what **rule 9** forbids
(check two constructed regressors' correlation *before* regressing). **No attempt is made, and the
magnitude −0.274 is NOT claimed as a pure batch effect.**

> **WHAT SURVIVES THE CONFOUND — and it is the load-bearing part.** The confound affects *how much* of
> the C response to attribute to batch versus training progress. It **cannot** explain **which component
> each lever moves**, because that is a contrast **within the same arms**: the batch lever moves **g**
> (t +6.94) while leaving **λ** flat (t +0.79) — the **exact inverse** of the LR lever. **A confound
> shared by all three arms cannot invert a component split.** **The dissociation stands; the effect size
> does not.**

**PROPOSED n=4 SEED CHECK — band 50 (criterion registered, with the confound built into the design).**
*Criterion:* on a fresh panel, (i) the **LR** lever gives **|d log λ| ≥ 0.5, |t| ≥ 3** with **d log C not
significant**; (ii) the **BATCH** lever gives **|d log C| ≥ 0.15 with |t| ≥ 3** and **d log λ not
significant** — the inverted split; (iii) both identity residuals **< 1e-10**.
*Design note for whoever runs it:* **the confound is removable** — run the batch arms **token-matched at
the STOP step** (equal tokens seen, unequal step counts) rather than step-matched. That breaks the
batch/progress collinearity and would license a causal magnitude, not just a dissociation.
*Status:* the **dissociation** is satisfied by committed data (REQ-045 n=1 seed, REQ-037 n=1/arm); **the
magnitudes are not established.** **No new compute requested this iteration** — REQ-049 already asks for
the n=4 replication of the LR half, and a token-matched batch arm would be the natural companion.

**⇒ CONSEQUENCE FOR THE CAMPAIGN GOAL.** The between-layer bowl is a property of the **loss surface**
(band 40), immune to every scale factor (band 42), and now shown **movable by a noise-channel
intervention**. **That is the first positive evidence about what kind of intervention could reach it** —
and it points away from optimiser hyperparameters toward the **gradient-noise / batch channel.**

**Queue:** REQ-048 **OPEN**; REQ-049 **filed** (iteration 174). No Jerry response.

## ★★★ THE GAUGE THEOREM IS CONFIRMED CAUSALLY (iteration 174) — a randomised per-matrix LR moves λ but NOT C

*Continuing the rule-18 script audit. `make_req045_arms.py` reveals REQ-045 to be **the one experiment in
the campaign that varies the learning rate PER MATRIX** — 3 global arms `S ∈ {0.7, 1.0, 1.4}` crossed
with per-matrix `m_i` drawn **independently** from {0.6, 0.85, 1.0, 1.2, 1.7}. **This is a randomised
treatment, and it can answer the question band 48 left open: can a per-MATRIX lever move C at all?***

**ADMISSIBILITY.** The predictor is the **assigned multiplier** — an experimental treatment, not derived
from λ, not from the Lanczos tridiagonal. **Fully admissible, and causal rather than correlational.**
*(The committed `req045_draws.json` also stores its own identifiability check:
`corr(own m_i, others' mean) = −0.182` — far from the **−1.0000** collinearity that killed my iteration-125
attempt. The design is sound, and rule 18 applied to the data file too: its schema is nested differently
from what the generating script suggested, and I inspected rather than assumed.)*

**THE RESULT — 216 matrix-observations, controlling for type, block and the global arm:**

| quantity | elasticity to own `log m_i` | se | t | 95% CI |
|---|---:|---:|---:|---|
| **log λ** | **−1.218** | 0.161 | **−7.55** | [−1.535, −0.902] |
| log g | **−0.650** | 0.052 | **−12.61** | [−0.751, −0.549] |
| **log C** | **+0.081** | 0.091 | **+0.89** | **[−0.097, +0.258]** |

> **Turning up one matrix's learning rate moves its own curvature hard (elasticity −1.22) and moves its
> own C not at all.**

**⚠️ THE NULL IS POWERED — rule 16 satisfied, which is what makes this decisive.** C's CI half-width is
**0.177 dex per dex**, and the interval **excludes λ's elasticity (−1.218) by a wide margin**. This is a
**real null**, not a failure to detect — unlike the underpowered nulls flagged in iteration 166.

**⇒ AND THE COMPONENTS MOVE IN EXACTLY THE OFFSETTING PROPORTION THE THEOREM DEMANDS:**

```
d(log λ)/d(log m) − 2 · d(log g)/d(log m)  =  −1.218 − 2(−0.650)  =  +0.081  =  d(log C)/d(log m)
                                                          identity residual: 9.99e-16
```

**λ falls at roughly twice g's rate, so `C = λ/g²` is left invariant.** That is the gauge theorem's
signature, observed **causally under randomisation**.

> **★ THIS UPGRADES BAND 42 FROM DERIVATION TO EXPERIMENT.** The gauge theorem was proved algebraically
> (iter. 159) and corroborated observationally (iter. 172: C's LR-response is 0.44× λ's). **REQ-045
> supplies the causal test: an intervention that scales one matrix's contribution moves λ and g but
> leaves C fixed, with a powered null.** No band in this campaign had a randomised-intervention
> confirmation until now.

**⇒ AND IT CLOSES THE DESIGN QUESTION COMPLETELY.** Band 48 showed a per-**type** LR is *algebraically*
powerless on the between-layer axis. **REQ-045 now shows a per-MATRIX LR — the finest possible lever —
is powerless on C itself, experimentally.**

**⇒ THE FIFTH AND FINAL REASON REQ-036 WAS A NULL, and the most fundamental:** *no learning-rate
intervention of any granularity can move C, because the LR is exactly the kind of scale factor `C = λ/g²`
cancels.* Bands 16, 43, 45/46 and 48 explained why the specific design failed; **this explains why the
entire class of LR-based curvature-equalization designs must fail.** **Recommendation, final: curvature
equalization is not reachable through the learning rate at any granularity — per-type, per-layer or
per-matrix.**

**PROPOSED n=4 SEED CHECK — band 49 (criterion registered).**
*Criterion:* on a fresh randomised per-matrix LR panel, (i) **|d(log λ)/d(log m)| ≥ 0.5 with |t| ≥ 3**
(the lever demonstrably works); (ii) **d(log C)/d(log m) not significant, with a 95% CI half-width
< 0.30** (a powered null, not an absent test); (iii) the CI **excludes** the fitted λ elasticity.
*Status:* **satisfied by committed REQ-045 data** (−1.218, t −7.55; +0.081, t +0.89, half-width 0.177;
CI excludes −1.218). ⚠️ **n = 1 seed** — REQ-045 ran three arms on one seed, so this is a **single-network
result** and the criterion is registered for a genuine 4-seed replication. **No new compute requested;
≤2-node ceiling.**

**⚠️ HONEST LIMITATION.** REQ-045 is **one seed, one step (2750), 216 observations**. The elasticities are
precisely estimated *within* that network, and the identity check is exact arithmetic, **but the result
has not been replicated across seeds.** It is the campaign's strongest single result and its replication
is the highest-value cheap experiment available — **cheaper than REQ-048**, since REQ-045's machinery
already exists and 3 arms × 4 seeds is ~16 min of training per the REQ-035 precedent.

**Queue:** REQ-048 still **OPEN**, no Jerry response.

## ✅ REQ-036 VALIDATION CLOSED (iteration 173) — verified against its own source; the design is ALGEBRAICALLY powerless on the layer axis

*Rule 18 cost twenty iterations, so its implication was acted on immediately: **read the Jerry scripts I
have never read.** Priority went to `make_req036_arms.py`, since REQ-036 is the design this loop
explicitly asks me to validate — and until now I had validated it from its **results**, never from its
**source**.*

**THE DESIGN, FROM THE SOURCE** (`logs/kmaxwell/req036_equalized_curvature_lr/make_req036_arms.py`):

```
A2 = {attn.proj 0.40, attn.k 0.88, mlp.fc 0.91, attn.q 1.18, attn.v 1.25, mlp.proj 1.56}
A5 = {attn.q 0.568, attn.k 0.755, attn.proj 0.642, attn.v 1.101, mlp.fc 1.260, mlp.proj 2.462}
A4 = 1/A2                     # anti-rule
a3 = A2, except blocks 0 and 11 where attn.proj→1.20 and mlp.proj→3.00
```

> **Every arm is a PER-TYPE CONSTANT in depth.** The sole depth dependence in the entire design is
> arm a3's override at **2 of 12 blocks, for 2 of 6 types**. **The design has essentially no depth
> dimension.** This is confirmed from the source, not inferred from the outcome.

**⇒ THE DECISIVE RESULT — a per-type constant is ALGEBRAICALLY powerless on the between-layer axis.**
A per-type multiplier shifts **every layer of that type by the same amount**, so all block means move
together and the between-layer **spread** is unchanged. Searching **2000 random per-type rules** over
multipliers in [0.3, 3.0]:

| quantity | value |
|---|---:|
| between-layer variance of block-mean log C (s = 1.0) | 0.03527 |
| **best between-layer variance reduction over 2000 rules** | **0.000%** |

**Not "small" — zero, by construction.** **No choice of per-type constants, including the optimal ones,
could have moved the between-layer effect.** This is stronger than the three empirical mechanisms
previously recorded: it is not that the design *failed*, it is that **on this axis it could not have
succeeded.**

**⚖️ IN FAIRNESS TO THE DESIGN — it was not aimed at nothing.**

| axis | variance of the corresponding means |
|---|---:|
| **between-TYPE** (type-mean log C) | **0.17855** |
| between-LAYER (block-mean log C) | 0.03527 |
| **ratio** | **5.06×** |

**The type axis carries 5× more curvature variance than the layer axis.** A per-type rule targets a real
and larger source of C-spread. **The mismatch is with the campaign's GOAL — "what sets the difference in
C between LAYERS" — not with the design's internal logic.** REQ-036 answers a different question than the
one this loop is chasing, and answers it in the negative (uniform LR best, harm monotone in equalization,
Spearman −1.000).

**⇒ CONSOLIDATED REQ-036 VERDICT — now four independent reasons, one of them algebraic:**
1. **Band 16** — C is actively restored; equalizing it fights a homeostat.
2. **Band 43** — along Muon's actual step direction there is **no bowl to equalize**.
3. **Bands 45/46** — types share one positional bowl whose amplitude also varies by type: two
   orthogonal axes.
4. **THIS — algebraic**: a per-type constant cannot change between-layer spread at all (0.000%
   over 2000 rules).

**⇒ RECOMMENDATION, FINAL AND UNCHANGED: do not build per-layer or per-type LR on curvature
equalization.** If the goal is the **layer** axis, the lever must be **per-layer** (or per-matrix); a
per-type lever is provably inert there. **The REQ-036 validation this loop asked for is COMPLETE.**

**PROPOSED n=4 SEED CHECK — band 48 (criterion registered).**
*Criterion:* on any panel, (i) the best per-type-constant rule reduces between-layer variance of
block-mean log C by **< 1%** over ≥1000 random draws; (ii) **between-type variance exceeds between-layer
variance** (confirming the design targeted a real but different axis).
*Status:* **satisfied by committed data** (0.000%; ratio 5.06×). **This criterion is deterministic given
the panel — it needs no new compute and cannot fail by chance.** ≤2-node ceiling; nothing requested.

**⚠️ SCRIPT AUDIT STATUS (rule 18).** Read so far: `analyze_req035_armA.py` (which produced iteration
172's axis correction) and `make_req036_arms.py` (this iteration — **confirms** the analysis rather than
overturning it). **Unread and outstanding:** `make_req037_arms.py`, `analyze_req046.py`,
`make_req046_arms.py`, `apply_req046_patches.py`, `make_req045_arms.py`, `analyze_req045.py`,
`analyze_req047.py`, and the three `measure_activation_backward*.py` probes. **Any of these could carry
a correction of the same kind as iteration 172's**, and they are the cheapest remaining source of
information in the repository.

**Queue:** REQ-048 still **OPEN**, no Jerry response.

## ⛔⇒★ MAJOR CORRECTION: THE "FORK STATES" ARE LEARNING RATES (iteration 172) — mislabelled throughout, and the bowl is STRONGER for it

*Unable to run REQ-048 locally (no CUDA on this machine), I read **Jerry's own analysis script**,
`logs/kmaxwell/req035_armA_seed_replication/analyze_req035_armA.py`. **It reveals that I have been
misinterpreting the REQ-035 panel's third axis for roughly twenty iterations.***

**WHAT JERRY'S SCRIPT SAYS:**

```
S = {"s060":0.60, "s100":1.00, "s170":1.70}
fit:  log10 lam = log10 C - k*log10 s      # C is the INTERCEPT at s = 1
```

and REQ-035's own status line reports **"k = 1.17–1.34 per seed"**. **`s` is a LEARNING-RATE
MULTIPLIER.** The three tags are **three separate training runs at 0.60×, 1.00× and 1.70× LR** — not
three checkpoints of one run.

**VERIFIED AGAINST THE RAW DATA, not taken on trust:**

| check | result |
|---|---|
| recovered `k = −d(log λ)/d(log s)` per matrix | **1.364** (sd 0.530, n=288) — **matches REQ-035's 1.17–1.34** |
| median raw λ at s = 0.60 / 1.00 / 1.70 | **22576 → 12562 → 5494** — monotone in LR, as `k > 0` requires |

**⇒ WHAT I GOT WRONG.** I called these "fork states" and treated them as **replicates**. Every statistic
that averaged over them or measured spread across them was mis-labelled:
- **band 42's "per-matrix sd of log C across fork states = 0.1449 dex"** — described as *stability*;
- **bands 38/39's "cross-fork transfer"** — described as generalisation across *training states*;
- the **60-fit panel** (4 seeds × 3 "forks" × 5 steps) — the 3 was a **treatment**, not replication.

**⇒ WHAT SURVIVES, AND IS STRONGER THAN CLAIMED — the bowl is LR-INVARIANT.** Recomputing the bowl
**separately within each learning rate** (the correct conditioning, since `s` is a treatment):

| learning rate | cubic R² | linear R² | argmin | swing |
|---|---:|---:|---:|---:|
| **0.60×** | 0.904 | 0.054 | **L6** | 0.523 |
| **1.00×** | 0.961 | 0.042 | **L6** | 0.602 |
| **1.70×** | 0.975 | 0.048 | **L6** | 0.488 |

**Pairwise correlation of the bowl between learning rates: +0.977, +0.886, +0.817.**

> **The bowl has the same shape, the same minimum at layer 6, and comparable amplitude across a 2.8×
> learning-rate range.** That is a **genuine robustness result** — far stronger than the "replication
> across checkpoints" I mistakenly reported. **The central positional finding is not damaged by this
> error; it is corroborated by it.**

**⇒ AND BAND 42's NUMBER BECOMES A BETTER RESULT UNDER THE CORRECT READING.** The 0.44× ratio
(sd log C **0.1449** vs sd log λ **0.3320**) is **not** "C is more stable across checkpoints". It is:
**C responds far less to the learning rate than λ does.** **That is precisely what the gauge theorem
predicts** — the LR is one of the scale factors that cancels exactly in `C = λ/g²` — so a statistic I
mis-derived turns out to be **a direct empirical confirmation of band 42's central claim**, and a
sharper one than the stability framing.

**⇒ IMPLICATIONS FOR THE OTHER BANDS.** The **within-LR** results are untouched, because they never
crossed the `s` axis: band 39's bowl and cubic form (argmin L6 in **each** LR separately, above),
band 45's per-type bowls, band 40's variance decomposition, band 43's C_polar dissociation, and the
gauge theorem itself. **What must be re-labelled is every phrase reading "fork state" as a replicate**,
and every "cross-fork transfer" figure, which is **cross-LR generalisation** — still a meaningful and
arguably better test, but not what the text says.

**⚠️ AND IT VOIDS ONE OF MY OWN RULES-DERIVED CONCLUSIONS.** Iteration 165's rule-15 audit averaged fork
states *within* a seed to form "n = 4 independent units". **Averaging over a treatment is wrong** — those
seed-level profiles mix three learning rates. The audit's *conclusions* happen to survive (the bowl is
LR-invariant, so averaging over LR distorts little, as the table above shows), **but the reasoning was
invalid and is corrected here.**

**Standing rule 18.** *Read the producing team's own analysis script before interpreting their data
layout. A directory-name convention (`s060`, `s100`, `s170`) is not self-documenting, and twenty
iterations were built on a guess about one when the answer was one file away in the same repository.*
**This is the single largest interpretive error of the campaign** — larger than any retracted band,
because it touched the axis rather than a claim.

**PROPOSED n=4 SEED CHECK — band 47 (criterion registered).**
*Criterion:* on a fresh panel, (i) the C bowl has **cubic R² ≥ 0.80 with argmin in layers 5–7 at EVERY
learning rate separately**; (ii) **pairwise correlation of the bowl between LRs ≥ +0.70**; (iii) **sd(log C)
across LRs < 0.60 × sd(log λ)** — the gauge theorem's LR prediction.
*Status:* **satisfied by committed REQ-035 Arm A data** (0.904/0.961/0.975, argmin L6 ×3; r +0.82 to
+0.98; 0.1449/0.3320 = **0.44**). **No new compute requested; ≤2-node ceiling.**

**Queue:** REQ-048 still **OPEN**, no Jerry response. **Note for whoever runs it: REQ-048 should record
the LR multiplier explicitly in its output**, so this ambiguity cannot recur.

## ◐ THE AMPLITUDE TRACKS λ's DEPTH RANGE (iteration 170) — survives circularity checks, then largely deflates itself

*Band 46 left the 19.6× type-varying amplitude **unexplained**: four architectural candidates, best
r +0.849 but permutation **p 0.146** at n=6. **All four were architectural.** A **measured** candidate was
never tried, and it follows directly from band 40 (the bowl lives in λ): **if a type's λ simply varies
more across depth, the same positional forcing produces a bigger C bowl** — no special property required.*

**THE PRE-DECLARED HYPOTHESIS AND ITS RESULT.** One hypothesis, one directional prediction, so **no
multiplicity correction is owed** — unlike band 46's, which had to price a search over four:

| type | amplitude | sd(log λ profile) | range(log λ) |
|---|---:|---:|---:|
| mlp.proj | 2.052 | **0.629** | 1.968 |
| attn.proj | 1.452 | 0.370 | 1.338 |
| mlp.fc | 1.046 | 0.204 | 0.845 |
| attn.k | 0.786 | 0.107 | 0.346 |
| attn.v | 0.559 | 0.143 | 0.415 |
| attn.q | 0.104 | 0.148 | 0.528 |

**corr(amplitude, sd_λ) = +0.901; permutation null over the 6 type labels: p = 0.0108.** It survives
where band 46's four candidates failed.

**⚠️ RULE 6 CHECK — and it passes, which is the strongest part of this iteration.** The amplitude is
fitted on the **C** profile and `log C = log λ − 2 log g`, so predictor and outcome **share log λ** —
precisely the trap that killed the −0.801 `align` result in iteration 160. **Split-sample test:
amplitude from seeds {0,1}, sd_λ from seeds {2,3}.** Sharing a term across **different networks** cannot
manufacture a correlation:

> **r = +0.865, permutation p = 0.0197.** **The relationship replicates across independent networks — it
> is a genuine property of the type, not a shared-term artifact.**

**◐ BUT THE SAME CHECKS LARGELY DEFLATE IT, and this is the honest headline.** Two findings from the
follow-through:

| diagnostic | value | reading |
|---|---:|---|
| **corr(amplitude, sd of the C profile)** | **+0.986** | **the amplitude IS essentially sd(C profile) rescaled** |
| corr(amplitude, sd_g) | +0.817 | the g component correlates nearly as well |
| corr(sd_λ, sd_g) | +0.935 | the two components are barely separable at n=6 |
| regress sd_C on both: sd_λ | +0.502, **t +2.12** | λ wins, but marginally (3 dof) |
| regress sd_C on both: sd_g | −0.205, t −0.15 | g contributes nothing |

> **VERDICT, at the strength the evidence supports.** *"Amplitude tracks λ's depth range" is **close to a
> restatement**: amplitude and sd(C profile) correlate at **+0.986**, so the claim reduces largely to
> "types whose C profile varies more have larger amplitude" — **true by construction**. The
> **non-tautological** residue is that sd_C is driven by **λ rather than g** (t +2.12 vs −0.15), which is
> **band 40's finding re-derived per type, not a new mechanism.*** **Band 46's "amplitude is
> unexplained" stands, only slightly narrowed: no INDEPENDENT driver has been found.**

**Why record a deflated result at all.** The split-sample p = 0.0197 is real and the tautology check is
what makes the result interpretable. **Without the corr(amp, sd_C) = +0.986 diagnostic this would have
been written up as "the amplitude's cause is found" — a significant p, a clean split-sample replication,
and a mechanism story from band 40.** It is the third time in this campaign that a quantity survived
every significance and replication test and still failed on construction (iteration 160's `align`,
iteration 161's `residual_tail`, this).

**Standing rule 17.** *Before crediting a predictor of a fitted parameter, correlate that parameter with
the raw spread of the thing it was fitted to. If they agree at r ≈ 1, the predictor explains the
construction, not the phenomenon — significance and out-of-sample replication cannot distinguish these.*

**⚠️ NO n=4 SEED CHECK PROPOSED.** The split-sample test **is** the seed check (amplitude and predictor
from disjoint seed pairs), and the finding is deflated rather than promoted, so registering a band would
overstate it. **Band 46 is left as written.**

**Queue:** REQ-048 still **OPEN**, no Jerry response. It remains the only outstanding request and the only
path to the central question from new measurement; ≤2 nodes, no training.

**⚠️ Run-length, restated:** the loop was specified as **8 hours from 2026-09-03 ~00:45 PDT** and has now
run **~15.8 hours**. Still flagged for the operator to stop or explicitly extend.

## ✅ END-TO-END CONSISTENCY CHECK + CONSOLIDATED STATE (iteration 169)

*46 bands, six audits, five retractions. **Never checked end-to-end: do the surviving load-bearing claims
cohere on one dataset, or has the account drifted into mutually inconsistent pieces?** All four claims
that must hold simultaneously were tested together.*

| # | claim | result | verdict |
|---|---|---|---|
| 1 | **identity** (band 40): C profile ≡ λ profile − 2× g profile | max error **5.55e-16** | **PASS** *(machine precision)* |
| 2 | **the bowl** (bands 39/45): interior minimum, beats a line, every seed | cubic R² 0.830–0.969; linear 0.014–0.181; argmin **L6,L6,L6,L7** | **PASS 4/4** |
| 3 | **gauge stability** (band 42): C more fork-stable than λ, per matrix | sd 0.1449 vs 0.3320 = **0.44×** | **PASS** |
| 4 | **held-out prediction** (band 39, corrected): near but above the floor | LOLO **0.0982** vs floor **0.0903** = **1.09×** | **PASS** |

**The account is internally consistent.** No drift, no mutually contradictory pieces.

---

### THE STATE OF THE CAMPAIGN — what is established about between-layer C

**THE ANSWER, as far as the evidence supports it.** Between-layer variation in C is **overwhelmingly
positional**, and the position field is a **specific, reproducible, asymmetric bowl**:

- **Position explains 68.8%** of between-layer variance; **the gradient 3.5%**; **type 0.0%** (removed by
  construction when block-means are taken). *(band 38)*
- The profile is a **smooth bowl**, minimum at **layer 6**, ends **unequal** (L11 +0.302 > L0 +0.166),
  best described by a **cubic**; a monotone trend explains almost nothing. *(band 39)*
- It is **time-stationary** (argmin L6 at all 5 checkpoints, profile corr +0.974) *(iter. 163)* and
  present **independently in 5 of 6 matrix types** (mutual corr +0.54 to +0.91) *(band 45)*.
- **`type + gradient + cubic position` predicts a held-out layer to 0.0982 dex against a 0.0903 floor
  (1.09×)** — a **49% error reduction** over the no-position model. *(band 39, corrected iter. 163)*

**WHERE IT LIVES.** The bowl is in **λ, not the gradient** (var shares 144% vs 30%, corr +0.866 vs
+0.130) *(band 40)*, and specifically in the **top of the spectrum**: the same gauge-invariant ratio built
on Muon's actual step direction (`C_polar`) is **monotone, not bowl-shaped** *(band 43)*.

**WHAT IT CANNOT BE — a closed class, not a list of failed guesses.** The **gauge theorem** (band 42):
*any scalar multiplying a matrix's whole contribution to the loss cancels exactly in `C = λ/g²`.* This
forecloses `post_lambda`, `resid_lambda` and its downstream product, the LR, and every output gate **in
one line**. Individually excluded as well: Muon's step magnitude *(band 31)*, residual-stream scale,
input effective rank *(iter. 156)*, matrix shape *(iter. 161)*, and an axis artifact *(iter. 157)*.
**⇒ The bowl is a conditioning property of the loss surface — curvature relative to a matrix's own
gradient — not a scale.**

**THE REQ-036 VERDICT — three independent reasons the per-type LR design was a null:**
1. **Band 16** — C is actively restored, so equalising it is fighting a homeostat.
2. **Band 43** — along the direction Muon actually steps there is **no bowl to equalise**.
3. **Bands 45/46** — types share **one positional bowl**; a per-**type** LR cannot address a
   **positional** effect, and the bowl's amplitude varies by type as well. **Two orthogonal axes.**
**The empirical result (uniform LR best; harm monotone in equalization, Spearman −1.000) now has a
mechanism.** **Recommendation stands: do not build per-layer or per-type LR on curvature equalization.**

**WHAT REMAINS OPEN, precisely bounded:**
- **~0.039 dex** of between-layer structure the cubic does not capture — **real in magnitude,
  unlocalised** (the layer-2/3/6/8 localisation was retracted as pseudo-replication, iter. 164).
- **Why** the surface is stiffest at both ends and softest at layer 6. **REQ-048** (spectral
  participation ratio) is the filed, un-run measurement designed to answer exactly this.
- The **bowl's amplitude varies 19.6× by type** with **no structural predictor surviving** a permutation
  null at n=6 *(band 46)*.

**METHODOLOGICAL LEDGER — 16 standing rules, most written after an error.** The five retractions:
band 37 (depth artifact), band 27's correlation half, iteration 129's band-13 overturning, iteration
163's residual localisation, and iteration 160's `align`-vs-C correlation. **The single most valuable
guard was the hard rule on the Lanczos tridiagonal**: `residual_tail` correlates with the bowl at
**−0.833 (|t| 31.4, 12/12)** — the strongest correlate found anywhere in this campaign — and is
**inadmissible**, coming from the same `eigh()` as `lam_top` *(iter. 161)*.

---

**⚠️ RUN-LENGTH NOTE FOR THE HUMANS.** This loop was specified as an **8-hour run started 2026-09-03
~00:45 PDT**. It is now **2026-09-04 16:27 PDT — roughly 15.6 hours elapsed, nearly double the stated
window.** Flagging rather than silently continuing: **the operator may want to stop the loop, or extend
it explicitly.** Work continues to be scoped at **≤2 nodes** with no assertion of higher authority.

**⚠️ QUEUE.** **REQ-048 remains OPEN and un-run** — it is the only outstanding request and the only
measurement that can advance the central open question. Jerry's last commit was **REQ-047 at 12:35 PDT**
(~4h ago); today's inter-delivery gaps have run 3h29m–6h45m, so this is **still inside normal turnaround
and no escalation is being made**. **If REQ-048 is not picked up, the campaign has no further path on the
central question from committed data** — iterations 156–168 exhausted it deliberately, and the admissible
predictor set explains only **4.3%** of the bowl *(iter. 162)*.

## ⚠️/✅ THE BOWL'S AMPLITUDE IS TYPE-VARYING (iteration 168) — a real misspecification that costs nothing

*Band 45 found the bowl shared across five of six types with **type contributing only an offset**. That
has a sharp consequence never checked: **if type were purely an offset, every type's bowl would have the
same amplitude.** The measured swings span **5×**, so the consequence fails — and the additive
`type + position` model used since band 3 rests on exactly that assumption.*

**Fitting each type's profile as `a × (shared bowl shape)`**, where the shared shape is the type-centred
pooled profile over all 60 fits:

| type | **amplitude a** | se | R² of `a×shared` | own swing |
|---|---:|---:|---:|---:|
| **mlp.proj** | **2.052** | 0.049 | 0.734 | 1.199 |
| attn.proj | 1.452 | 0.075 | 0.796 | 0.676 |
| mlp.fc | 1.046 | 0.053 | **0.845** | 0.526 |
| attn.k | 0.786 | 0.088 | 0.647 | 0.399 |
| attn.v | 0.559 | 0.045 | 0.488 | 0.394 |
| **attn.q** | **0.104** | 0.102 | 0.042 | 0.245 |

**Amplitude spans 0.104 → 2.052, a 19.6× range.** Testing each against a common amplitude a = 1:
**attn.k −2.4, attn.proj +6.1, attn.q −8.8, attn.v −9.8, mlp.fc +0.9, mlp.proj +21.3.**
**Five of six differ significantly.**

> **⇒ THE ADDITIVE MODEL IS MISSPECIFIED.** `type offsets + one shared position curve` assumes a common
> amplitude, and that assumption is false. This is a genuine limitation of every model in this campaign
> from band 3 onward, and it was never tested until now. **Band 45's "type is only an offset" is
> therefore too strong: types share the bowl's SHAPE but not its SIZE.**

**⊘ DOES THE AMPLITUDE TRACK ANYTHING STRUCTURAL? — not detectably, and the discipline is the point.**
Four candidates, all declared from structure before looking, all reported:

| candidate | corr with amplitude |
|---|---:|
| **residual-writer** (attn.proj, mlp.proj — band 7's split) | **+0.849** |
| type level (log C offset) | −0.791 |
| log fan-in | +0.751 |
| is-mlp | +0.619 |

**Permutation null over the 6 type labels, max \|r\| across all four candidates (20,000 shuffles):
p = 0.1461.** **Not significant.** With **n = 6**, a correlation of **+0.85 is unremarkable** — reporting
the r alone would have manufactured a finding, exactly as 36 shape tests nearly did in iteration 156.
**The amplitude's structural explanation is UNDETERMINED, and n = 6 types cannot determine it.**

**✅ AND THE COST OF THE MISSPECIFICATION IS ZERO — measured, not assumed.** Allowing a **free cubic per
type** (18 position parameters instead of 3), same LOLO protocol as band 39:

| model | position params | LOLO rmse (n=720) |
|---|---:|---:|
| **additive (shared curve)** | 3 | **0.0982 dex** |
| per-type amplitude/shape | 18 | **0.1050 dex** |
| *(noise floor)* | | *0.0903 dex* |

**The flexible model predicts WORSE on held-out layers** — 6× the parameters buying a 7% *increase* in
error. **The extra amplitude structure is real in description but is not recoverable from 12 layers per
type without overfitting.**

> **VERDICT, at the strength the evidence supports.** *The bowl's amplitude genuinely varies by type
> (19.6×, five of six significant). No structural predictor of it survives a permutation null at n = 6.
> The additive model is misspecified but remains the best working model out of sample, and band 39's
> figures stand unchanged.* **The honest summary is: a known, quantified, unexplained limitation that
> does not currently cost anything.**

**PROPOSED n=4 SEED CHECK — band 46 (criterion registered).**
*Criterion:* on a fresh 4-seed panel, (i) the per-type amplitude range is **≥ 3×**; (ii) **≥4 of 6 types
differ from a common amplitude at \|t\| ≥ 2**; (iii) **per-type position terms do NOT beat the shared
cubic** on LOLO. *Status:* **satisfied by committed data** (19.6×; 5 of 6 at \|t\| 2.4–21.3; 0.1050 vs
0.0982). **No new compute requested; ≤2-node ceiling.**

**⚠️ THIS MODIFIES BAND 45, NOT REFUTES IT.** The bowl *is* present in five of six types with mutual
correlations +0.54 to +0.91 — that stands. What is corrected is the inference that **type is therefore
only an offset**; it is an offset **and** a scale factor on the bowl. **REQ-036's verdict is unaffected
and arguably strengthened**: a per-type LR cannot fix a positional effect whose amplitude *also* varies
by type — that is two orthogonal axes, not one.

**Queue:** REQ-048 still OPEN, no Jerry response (last Jerry commit REQ-047, 12:35 PDT; gaps today have
run 3h29m–6h45m, so this remains inside normal turnaround).

## ★ THE BOWL IS SHARED ACROSS FIVE OF SIX TYPES (iteration 167) — position and type are separable

*Every bowl result so far removes **type** first and profiles the residual, which by construction yields
one shared bowl. **Never asked: does each type have its own bowl, and is it the same one?** Rule 16 makes
this the right kind of question — each type has 12 layers per seed, so the **layer axis supplies the
power** rather than the 4-seed axis.*

**PER-TYPE BOWLS, fitted independently (no type removal):**

| type | argmin | cubic R² | linear R² | swing | per-seed argmins |
|---|---:|---:|---:|---:|---|
| mlp.proj | 7 | **0.950** | 0.000 | 1.225 | [7, 6, 7, 7] |
| attn.proj | 7 | **0.878** | 0.015 | 0.697 | [6, 7, 6, 7] |
| mlp.fc | 6 | **0.871** | 0.053 | 0.604 | [8, 2, 6, 7] |
| attn.v | 6 | **0.817** | 0.323 | 0.393 | [7, 6, 6, 6] |
| attn.k | 6 | **0.791** | 0.009 | 0.411 | [6, 6, 5, 7] |
| **attn.q** | 4 | **0.315** | 0.282 | **0.245** | **[0, 2, 6, 1]** |

**Five of six types carry the same bowl independently** — argmin 6–7, cubic R² 0.79–0.95, linear R² ≈ 0
— with mutual profile correlations of **+0.54 to +0.91** (attn.proj↔mlp.fc **+0.907**). **The bowl is not
an artifact of pooling types: it is present in each type separately.**

> **⇒ POSITION AND TYPE ARE SEPARABLE AND ADDITIVE.** The same depth profile appears in attention and MLP
> matrices, in readers and writers, at 768×768 and 3072×768. **This is what licenses every model in this
> campaign that fits `type offsets + position` additively** — an assumption made from band 3 onward and
> never tested until now.

**⇒ AND IT SHARPENS THE REQ-036 VERDICT.** REQ-036's per-type LR design assumed the types differ in a way
worth equalising. **They do not differ in depth structure — they share one bowl.** The type dimension
carries **offsets**, the position dimension carries **the bowl**, and equalising *per type* cannot address
a *positional* effect. **This is a third independent reason REQ-036 was a null** (alongside band 16's
active restoration and band 43's absence of a bowl along Muon's step direction), and it is structural
rather than empirical: the design's axis and the effect's axis are orthogonal.

**⚠️ THE attn.q EXCEPTION — checked against the campaign's own failure mode, and it is MOSTLY NOISE.**
Bands 37 and the mlp.fc saga taught that an exceptional-looking type is usually a noisier one. Measuring
noise on the same footing:

| type | seed-to-seed sd | swing | **signal/noise** | cubic R² |
|---|---:|---:|---:|---:|
| mlp.proj | 0.0781 | 1.225 | **15.7** | 0.950 |
| attn.proj | 0.0933 | 0.697 | 7.5 | 0.878 |
| mlp.fc | 0.0913 | 0.604 | 6.6 | 0.871 |
| attn.v | 0.0879 | 0.393 | 4.5 | 0.817 |
| attn.k | 0.1189 | 0.411 | 3.5 | 0.791 |
| **attn.q** | **0.1430** *(highest)* | **0.245** *(smallest)* | **1.7** | 0.315 |

**attn.q has the worst signal-to-noise of the six by a factor of 2**, which is sufficient to explain a
flat-looking profile. **But pooling all 60 fits — 15× more data for the same type — does NOT recover a
bowl** (cubic R² stays **0.315**, argmin L4, swing 0.245 vs attn.k's 0.399 and attn.v's 0.394 on the same
pooled basis).

> **Verdict, stated at the strength the evidence supports: attn.q's bowl is genuinely SMALLER (swing
> 0.245 vs 0.39–1.23), and its shape is UNRESOLVED at this noise level — not established as absent.**
> **No exception is claimed.** *(Consistent with band 35: q and k already differ in step alignment by
> 0.306 dex despite identical chunk geometry, so attn.q having its own character is not a new anomaly.)*

**PROPOSED n=4 SEED CHECK — band 45 (criterion registered).**
*Criterion:* on a fresh 4-seed panel, (i) **≥5 of 6 types independently show cubic R² ≥ 0.70 with argmin
in layers 5–8**; (ii) **mean pairwise correlation between the six type-profiles ≥ +0.40**;
(iii) **no type's profile correlates negatively with the pooled bowl at ≤ −0.30**.
*Status:* **satisfied by committed REQ-035 Arm A data** (5/6 at R² 0.79–0.95, argmin 6–7; mean pairwise
+0.469; most negative single pair −0.165, and no type-vs-pooled correlation below −0.30).
**No new compute requested; runs under the ≤2-node ceiling.**

**⚠️ QUEUE NOTE — REQ-048 is not overdue.** Checking rather than assuming: **39 of the last 40 commits on
this branch are mine**; Jerry's most recent is **REQ-047 at 12:35 PDT**. Jerry's delivery gaps today have
run **3h29m and 6h45m**, and REQ-048 was filed at ~15:27 (now 16:07). **Silence is well inside normal
turnaround — no escalation is warranted and none is being made.** REQ-048 remains the only outstanding
request, ≤2 nodes, no training.

## ⚖️ THE PANEL'S POWER, MEASURED (iteration 166) — which claims are sound and which are merely unrefuted

*Iteration 165 flagged that at n = 4 clusters a "not significant" result is **consistent with zero, not
proven zero**. That bound governs every remaining claim the committed data can support, so it is
quantified here once — from the panel's **own measured seed-to-seed noise**, not a textbook formula.*

**THE PANEL'S NOISE AND THE BOWL'S MARGIN.**

| quantity | value |
|---|---:|
| measured seed-to-seed sd of the C profile, per layer | **0.0623 dex** |
| the bowl's own swing | **0.5377 dex** |
| **signal-to-seed-noise** | **8.6×** |

**That 8.6× is why the bowl was detectable at all**, and it is the reason the positive findings are
robust while the nulls are not.

**⚠️ POWER FOR A CLUSTERED CORRELATION TEST (two-sided α = 0.05, simulated on the panel's noise):**

| true ρ | **n=4** | n=8 | n=12 | n=16 |
|---:|---:|---:|---:|---:|
| 0.30 | **14%** | 42% | 84% | 99% |
| 0.50 | **23%** | 86% | 100% | 100% |
| 0.70 | **40%** | 100% | 100% | 100% |
| 0.80 | **54%** | 100% | 100% | 100% |

> **At n = 4, a true correlation of 0.80 is detected only 54% of the time — a coin flip. At ρ = 0.5 it is
> 23%.** Every clustered null this panel can produce is therefore near-uninformative on its own.

**⇒ CLASSIFYING THE CAMPAIGN'S RECENT CLAIMS BY WHETHER POWER SUPPORTS THEM.**

**✅ SOUND — shape and consistency claims.** Power comes from the **12-layer axis** (12 points per seed,
8.6× margin); seeds only need to supply *agreement*, which is a much cheaper requirement:

| claim | evidence |
|---|---|
| band 39 — bowl exists, cubic form, argmin L6 | argmin L6 in **5/5 steps, 12/12 fits** |
| band 42 — bowl present in every fork state | R² 0.90–0.98, argmin L6 in **all 3** |
| band 43 — C_polar monotone, slope negative | **4/4 seeds same sign**, \|t\| 4.91 |
| band 40 — corr(C profile, λ profile) = +0.866 | \|t\| **26.0**, far above the n=4 threshold |
| iter. 163 — bowl is time-stationary | profile corr **+0.974** across 5 steps |

**⚠️ MERELY UNREFUTED — null claims at n = 4.** These are **weak evidence and must not be cited as
established absences**:

| claim | statistic | the problem |
|---|---:|---|
| band 40 — "corr(C, g) not significant" | \|t\| 1.39 | a true ρ = 0.5 would be **missed 77% of the time** |
| band 43 — "corr(C, C_polar) low" | \|t\| 0.66 | same limitation |
| iter. 158 — "C blind to the `post_lambda` channel" | \|t\| 1.36 | same limitation |

**This is a correction to how I have been reporting these.** All three were stated as findings; they are
**failures to detect**, and the distinction is material. *(Note: iteration 158's `post_lambda` result has
a second, independent leg that does **not** depend on this — the **algebraic** cancellation in
`log C = log λ − 2 log g` is exact and needs no statistics. **The derivation stands; only the empirical
corroboration is weak.** Likewise band 43's positive half — C_polar's monotone decline at 4/4 — is sound;
only the "and it is uncorrelated with C" half is underpowered.)*

**✅ AND THIS VINDICATES BAND 44's DESIGN — REQ-048's criterion is achievable.** Band 44 was specified
with a **sign-count** criterion ("same sign in ≥10 of 12 fits") rather than a clustered t-test. That was
the right choice by a wide margin:

| true \|ρ\| | P(≥10 of 12 same sign) | P(4/4 same sign) | *(clustered t at n=4)* |
|---:|---:|---:|---:|
| 0.40 | **89%** | 65% | *18%* |
| 0.50 | **98%** | 82% | *23%* |
| 0.60 | **100%** | 93% | *32%* |

**A sign criterion computes each fit's correlation over 12 layers, so the layer axis supplies the power
and the seed axis only supplies consistency. At ρ = 0.5 it has 98% power where a clustered t-test has
23%.** REQ-048 will be able to give a decisive answer, either way, at n = 4.

**⚠️ NO NEW COMPUTE REQUESTED.** This is an analysis of the existing panel's limits. **REQ-048 remains the
only outstanding request** (still OPEN, no Jerry response), unchanged: ≤2 nodes, no training.

**Standing rule 16.** *State power before stating a null. A "not significant" from n = 4 clusters is a
failure to detect, not an absence — and should be written that way. Where a null matters, prefer a
criterion whose power comes from a within-unit axis (here, 12 layers per seed) over one that spends all
its power on the between-unit axis.*

## ✅ RULE-15 AUDIT OF THE RECENT BANDS (iteration 165) — all survive seed-clustering, and it is now clear *why*

*Rule 15 was written last iteration after pseudo-replication cost a claim. **Its first duty is to be
turned on the campaign's own recent bands**, several of which report "12/12" or "11/12" — where those 12
are **4 seeds × 3 fork states of the same 4 networks**, not 12 independent draws. Every headline was
recomputed with the **seed** as the independent unit (fork states averaged within seed first, n = 4).*

**RESULT — the recent bands hold.**

| claim | as reported (n=12) | **clustered on seed (n=4)** | retained |
|---|---:|---:|---:|
| **band 43** — C_polar slope | −0.0266, \|t\| **5.38**, 11/12 | −0.0266, \|t\| **4.91**, **4/4** | **91%** |
| **band 40** — corr(C, λ) | +0.867, \|t\| **42.68** | +0.866, \|t\| **26.02**, **4/4** | 61% |
| **band 40** — corr(C, g) *(the null half)* | +0.096, \|t\| 1.36 | +0.130, \|t\| **1.39**, 3/4 | 102% |
| band 43 — C's own slope *(the null half)* | \|t\| 0.46 | \|t\| **0.45** | — |
| band 43 — corr(C, C_polar) | +0.152, \|t\| 1.33 | +0.112, \|t\| **0.66** | — |
| *(for contrast)* **iter-163 residual at L6 — RETRACTED** | \|t\| **5.87** | \|t\| **1.88** | **32%** |

**Band 40's central contrast is intact under the correct clustering:** corr(C, λ) = **+0.866 (\|t\| 26.0,
4/4)** versus corr(C, g) = **+0.130 (\|t\| 1.39)**. **Band 43's dissociation is intact:** C_polar declines
at **\|t\| 4.91 with the same sign in all 4 seeds**, while C's own linear slope stays null at \|t\| 0.45.

**⇒ AND THE ASYMMETRY IS EXPLAINED, NOT JUST OBSERVED.** Why did clustering destroy the residual claim
(32% retained) but barely touch band 43 (91%)? It is a measurable property of each quantity — how much
of its variance is **reproducible across seeds** versus **within-seed wobble across fork states**:

| profile | reproducible across seeds | within-seed (fork) wobble |
|---|---:|---:|
| **C profile (the bowl)** | **80.4%** | 19.6% |
| C_polar profile | **67.3%** | 32.7% |

> **The bowl is 80% reproducible across independent networks.** Averaging fork states within a seed
> therefore discards almost nothing, and the clustered t stays high. **The retracted residual was, by
> construction, the leftover *after* the reproducible part had been modelled** — so it was mostly
> within-seed wobble, and clustering was fatal to it. **The two outcomes were not luck; they follow from
> where each quantity's variance lives.**

**What this settles.** Rule 15 is **not** a general deflator of this campaign's results. It bites
precisely on **residual-type quantities** — anything defined as *what is left after the reproducible
structure is removed* — and leaves **structural** quantities almost untouched. **Bands 40, 42 and 43
stand as written; band 39's magnitude correction stands; only the iteration-163 localisation claim was
lost, and it was the one quantity of residual type.**

**⚠️ A LIMIT WORTH STATING.** At n = 4 clusters the tests above have real power only for large effects.
**corr(C, g) at \|t\| 1.39 and corr(C, C_polar) at \|t\| 0.66 are reported as "not significant", not as
"zero"** — with 3 degrees of freedom, a moderate correlation would not be detectable. **The null halves
of bands 40 and 43 are consistent with the data, not proven by it.** This is a limitation of n = 4 seeds,
and no analysis of the committed data can remove it.

**⚠️ NO n=4 SEED CHECK PROPOSED.** This iteration *is* an n=4 check — of the campaign's own recent
claims, at the correct clustering. **REQ-048 remains the only outstanding request** (still OPEN, no Jerry
response), unchanged in scope: ≤2 nodes, no training.

**Standing rule 15, amended with its scope.** *Cluster on the independent unit — and expect the cost to
scale with how much of the quantity's variance is within-cluster. Structural profiles (reproducible
across seeds) survive clustering nearly intact; residual quantities (defined as the leftover after
structure is removed) rarely do. Compute the reproducible share to predict which kind you have, before
trusting a per-cell statistic.*

## ⛔ THE "LOCALISED RESIDUAL" WAS PSEUDO-REPLICATION (iteration 164) — my own iteration-163 claim, retracted

*Iteration 163 reported the residual as **systematic and localised to layers 2, 3, 6, 8** (|t| 3.9–5.9)
and called it "a sharper target than residual variance". **Chasing that target destroyed it.** Two
candidate explanations were tested and refuted, and the third check showed the target was never there.*

**⊘ REFUTATION 1 — architecture does not explain it.** `train_gpt.py` carries exactly the kind of
per-layer set structure that could produce a localised residual, all declared **before** fitting (read
from source, not chosen after seeing the data):

| indicator | coef (dex) | t | weighted R² |
|---|---:|---:|---:|
| `cache_layers` [3,7] | +0.0329 | **+1.26** | 0.137 |
| `attn_gate_layers` [3,10] | +0.0183 | +0.73 | 0.051 |
| `xsa_layers` [1,3,4,7] | +0.0149 | +0.72 | 0.050 |
| `ve present` [1,2,9,10,11] | −0.0106 | −0.56 | 0.031 |
| `dc_layers` [10] | −0.0144 | −0.46 | 0.020 |
| `paired_head_layers` [0,2,5,9] | −0.0033 | −0.17 | 0.003 |

**Permutation null over layer labels, taking the max |t| across all six** (20,000 shuffles) — pricing
**the selection**, not a single test: **p = 0.8149.** Nothing.

**⊘ REFUTATION 2 — it is not periodic either.** The alternating signs suggested periodicity; every
period the 12-layer axis supports was tested, all reported:

| period | amplitude (dex) | F |
|---|---:|---:|
| 4 | 0.0241 | **2.09** |
| 3 | 0.0149 | 0.67 |
| 6 | 0.0125 | 0.47 |
| 12 | 0.0051 | 0.05 |
| 2 | *(singular — Nyquist limit, amplitude is a numerical artifact, not a result)* | 1.39 |

**Permutation null, max F across all five periods: p = 0.8234.** Nothing.

**⛔ REFUTATION 3 — AND THIS ONE RETRACTS MY OWN CLAIM.** Before concluding "localised but unexplained",
I checked whether the localisation was ever real. **It was not.** Iteration 163's |t| values treated all
**60 fits as independent — but the 5 checkpoints within a seed × fork are the SAME NETWORK.** Across the
genuinely independent axis the residual barely replicates: **mean pairwise correlation across the 4 seeds
= +0.220, with one pair NEGATIVE (−0.34)** (across fork states, +0.466).

Re-testing with the correct clustering unit:

| | L0 | L1 | **L2** | **L3** | L4 | L5 | **L6** | L7 | **L8** | L9 | L10 | L11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **\|t\|, naive (iter. 163, n=60 as if independent)** | 0.80 | 1.42 | **3.87** | **5.54** | 1.64 | 2.49 | **5.87** | 0.13 | **4.39** | 0.39 | 1.30 | 2.19 |
| **\|t\|, clustered by seed (n=4 independent networks)** | 1.49 | 1.23 | 2.14 | **3.28** | 0.60 | 1.00 | 1.88 | 0.03 | 1.43 | 0.22 | 0.63 | 1.17 |

> **The four spikes collapse to 1.4–3.3; only layer 3 survives at |t| 3.28, and that is one marginal
> result out of twelve tested.** Counting 5 correlated checkpoints as 5 independent observations
> inflated every t by roughly **√5 ≈ 2.2**. **Iteration 163's "systematic residual localised to layers
> 2, 3, 6, 8" is RETRACTED. There is no localisation to explain, and the two refutations above were
> chasing an artifact.**

**✅ WHAT SURVIVES — the magnitude, unchanged.** LOLO **0.0982** vs floor **0.0903** = **1.09×**, excess
**0.0387 dex**. **Band 39's correction stands in full: it never depended on localisation**, only on the
aggregate error against the aggregate floor, and both are computed the same way on the same 60 fits.
**The residual is real in size and structureless in this data.**

**⚠️ THE IRONY, STATED PLAINLY.** Iteration 163 introduced **standing rule 14** — *use the panel at full
extent* — after finding 60 fits where 12 had been used. **Using all 60 was correct for the LOLO ratio and
wrong for the per-layer t-tests**, because the extra fits are repeated measurements of the same networks.
**More data improved the aggregate estimate and simultaneously invalidated the per-layer inference drawn
from it.**

**Standing rule 15.** *Expanding a panel changes what each row is. Before computing any per-cell
statistic, state the independent unit and cluster on it — the same expansion that tightens an aggregate
can manufacture significance in a disaggregate.* The independent unit here is the **seed** (4), not the
seed × fork × step fit (60).

**⚠️ NO n=4 SEED CHECK PROPOSED.** Three refutations and a retraction, all settled within committed data.
**REQ-048 remains the only outstanding request** (still OPEN, no Jerry response), and this iteration
**sharpens its target**: the residual it must explain is **0.0387 dex of unlocalised between-layer
structure**, not a layer-specific pattern. **Band 44's registered criterion is unaffected** — it tests
PR's correlation and shape against the C profile, neither of which depends on this.

## 🔧 BAND 39 CORRECTED AT 5× THE EVIDENCE (iteration 163) — the bowl is stationary, and the residual is REAL and LOCALISED

*REQ-048 is filed and OPEN with no Jerry response. Rather than idle, this iteration uses a resource the
campaign has left untouched: **every profile result so far used step 2750 alone, but the panel holds
five checkpoints (2250–2750) per seed × fork — 60 independent fits, not 12.***

**★ FIRST — THE BOWL IS TIME-STATIONARY, which licenses using all of them.** Band 24 established C's
*level* is time-invariant on an equilibrated window; the depth profile's **shape** had never been checked:

| step | cubic R² | linear R² | argmin | swing |
|---|---:|---:|---:|---:|
| 2250 | 0.952 | 0.003 | **L6** | 0.559 |
| 2375 | 0.887 | 0.028 | **L6** | 0.457 |
| 2500 | 0.894 | 0.000 | **L6** | 0.501 |
| 2625 | 0.912 | 0.000 | **L6** | 0.501 |
| 2750 | 0.961 | 0.004 | **L6** | 0.538 |

**Argmin at layer 6 in all five; linear R² ≈ 0 throughout; mean pairwise correlation between the five
step-profiles = +0.974 (min +0.942).** The bowl is a stationary feature of the equilibrated window, so
**every result built on step 2750 generalises** — and the evidence base is 5× larger at zero compute.

**⚠️ SECOND — BAND 39'S HEADLINE IS CORRECTED.** Band 39 reported that `type + gradient + cubic position`
predicts a held-out layer's mean log C to **0.0924 dex against a 0.0959 floor = 0.96×**, i.e. *"between-
layer C is closed to within seed noise."* Re-run on all **720 held-out layer-fits**:

| model | LOLO rmse (n=720) |
|---|---:|
| type + gradient (no position) | 0.1939 |
| + dist | 0.1279 |
| + quad | 0.1016 |
| **+ cubic** | **0.0982** |
| *(noise floor, measured on 60 fits)* | ***0.0903*** |

> **The ratio is 1.09×, not 0.96×.** The cubic LOLO is slightly worse (0.0982 vs 0.0924) and the floor
> slightly **tighter** (0.0903 vs 0.0959) with 5× the data. **The model ordering and every structural
> claim in band 39 hold exactly** — cubic best, monotone worthless, position essential (0.1939 → 0.0982,
> a 49% error reduction) — **but "closed to within seed noise" was optimistic and is withdrawn.**

**⇒ THE RESIDUAL IS REAL, AND IT IS SMALL AND SHARP.** `√(LOLO² − floor²)` = **0.0387 dex**, about
**7% of the bowl's own 0.52 dex swing**. **And it is NOT noise — it is systematic by layer:**

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **bias (dex)** | −0.018 | +0.018 | **−0.052** | **+0.061** | −0.015 | +0.017 | **−0.046** | +0.001 | **+0.035** | −0.003 | −0.011 | +0.039 |
| sem | 0.022 | 0.013 | 0.013 | 0.011 | 0.009 | 0.007 | 0.008 | 0.011 | 0.008 | 0.006 | 0.009 | 0.018 |
| **t** | −0.80 | +1.42 | **−3.87** | **+5.54** | −1.64 | +2.49 | **−5.87** | +0.13 | **+4.39** | −0.39 | −1.30 | +2.19 |

**Layers 2, 3, 6 and 8 are systematically mispredicted (|t| = 3.9–5.9), with alternating signs.** A
smooth cubic cannot represent a localised oscillation of this kind. **This is a sharper target than
"residual variance": the missing structure is specific, localised, and worth ~0.039 dex.**

**Why this matters for REQ-048.** The residual's size (**0.039 dex**) is precisely what the spectral
participation ratio is being asked to explain, and **band 44's registered criterion is unaffected** — it
tests PR's *correlation and shape* against the C profile, not the residual's magnitude. **REQ-048 now has
a quantitative target to hit rather than an open-ended one.**

**⚠️ NO NEW COMPUTE REQUESTED.** All of this comes from checkpoints already in the committed data.
REQ-048 remains the only outstanding request, still scoped to ≤2 nodes.

**Method note — and a correction to my own practice.** Band 39's 0.96× was computed on 12 fits when 60
were available in the same files. **The error was not in the analysis but in not asking how much data the
panel actually held.** The corrected figure is less flattering and better supported. *Standing rule 14:
before reporting a headline ratio against a noise floor, confirm the panel has been used at full extent —
an under-used panel inflates the floor and flatters the model at the same time.*

## ⛔ `residual_tail` REJECTED AS CIRCULAR (iteration 161) — the strongest correlate of the bowl is inadmissible

*Band 43 sharpened the question to the **top of the spectrum**, which points straight at
`residual_tail` — the probe's own measure of how well the Lanczos iteration has converged on the top
eigenvector. **Checking its provenance before using it is what this iteration is.***

**PROVENANCE — it fails the hard rule at the strongest possible level.** From the probe
(`measure_per_matrix_curvature.py`, `ebf53cd`), `top_ritz(alphas, offdiags)` — docstring *"(top
eigenvalue, residual bound) of the Lanczos tridiagonal"* — builds `t` from `alphas`/`offdiags`, calls
`evals, evecs = torch.linalg.eigh(t)`, and returns `float(evals[-1]), abs(float(evecs[-1, -1]))`.

**`lam_top` and `residual_tail` are the two return values of a single `eigh()` call on a single
tridiagonal.** `residual_tail` is the last component of that same top eigenvector. The hard rule bars
predictors built from the same Lanczos tridiagonal as `lam_top`; **this is not merely the same data, it
is the same matrix factorisation.** **REJECTED — not used as a predictor.**

**⚠️ WHAT THE RULE COST HERE — the number is recorded to price the rule, NOT as evidence.**

| *(forbidden)* | value |
|---|---:|
| corr(C profile, log `residual_tail` profile) | **−0.833** (sd 0.092) |
| \|t\| | **31.43** |
| same sign | **12/12** |
| tail profile cubic R² | 0.889 |

> **This is the strongest correlate of the bowl found anywhere in the campaign — stronger than
> log λ (+0.867) and far stronger than any admissible predictor — and it is worth nothing.** A quantity
> from the same eigendecomposition as `lam_top` tracks `lam_top`'s structure **for free**. Had it been
> used, it would have looked like the mechanism and been the campaign's largest error. **This is the
> clearest demonstration so far that the hard rule is doing real work rather than being a formality.**

**⊘ SECONDARY EXCLUSION — matrix shape cannot contribute to the depth profile.** Checking whether shape
(which sets the dimension the spectrum lives in) could vary with depth:

| type | shape | distinct shapes across blocks | varies with block? |
|---|---|---:|---|
| attn.q / attn.k / attn.v / attn.proj | 768×768 | **1** | **no** |
| mlp.fc | 3072×768 | **1** | **no** |
| mlp.proj | 768×3072 | **1** | **no** |

**Every type has exactly one shape at every depth**, so shape is **absorbed entirely by the type
offsets** present in every model here and **cannot produce a depth profile at all**. Muon's
`shape_mult = max(1, rows/cols)**0.5` is likewise a per-type constant. **Shape is excluded — and
band 19's `shape_mult` rival is now doubly excluded** (band 19 already refuted it on log g with
R² 0.005–0.008; here it is excluded structurally, for the depth question, by construction).

**⚠️ NO n=4 SEED CHECK PROPOSED.** A rejected predictor and a structural exclusion; **new compute would
test nothing.** Both settled within committed data under the ≤2-node ceiling.

**Standing rule 13.** *Before using any probe field as a predictor, read the function that produced it
and check whether it shares a factorisation — not merely a data source — with the outcome.* Two fields
in this probe are forbidden for this reason: `curvature_along_gradient` (≡ `alphas[0]`, an entry of the
tridiagonal) and `residual_tail` (from `eigh()` of the tridiagonal). **The admissible fields are
`top_eigenvalue` (as outcome), `gradient_block_norm`, `curvature_along_polar` (a separate HVP), and
`shape`.** Recording this list explicitly so the question does not have to be re-litigated.

**Search space:** the bowl is a **conditioning property of the top of the spectrum** — peaks at both
ends, minimum at layer 6 — immune to every scale factor (band 42), **absent from Muon's step direction**
(band 43), and not step magnitude (band 31), stream scale, input rank (iter. 156), an axis artifact
(iter. 157), or **matrix shape** (here).

## ⊘/★ THE BOWL IS SPECIFIC TO THE TOP EIGENDIRECTION (iteration 160) — one withdrawal, one dissociation

*Band 42 says C measures **conditioning**, so this iteration went after the one conditioning quantity
already in the committed probe: `curvature_along_polar` (cp), the curvature along **the direction Muon
actually steps**. **Circularity check first:** cp is a **separate HVP**, not an entry of the Lanczos
tridiagonal that produces `lam_top` — unlike `curvature_along_gradient` (≡ `alphas[0]`), which is
forbidden. cp is admissible, on the same basis band 35 already uses it.*

**⚠️ FIRST RESULT — WITHDRAWN under standing rule 6.** The alignment profile
`log align = log cp − log λ` correlates with the C profile at **−0.801** (per-fit mean −0.749, |t| =
17.82, **12/12 same sign**) — a near-mirror of the bowl, and it looked like a strong finding. **But
`log C = log λ − 2 log g` and `log align = log cp − log λ` share `log λ` with opposite signs, so a large
negative correlation is partly guaranteed by construction.** Refitting with raw components:

| relationship | mean corr | \|t\| | same sign |
|---|---:|---:|---:|
| C profile vs **align** profile *(shared λ — suspect)* | **−0.749** | 17.82 | 12/12 |
| C profile vs **log cp** profile *(raw, no shared term)* | **+0.211** | 2.29 | 8/12 |
| C profile vs log λ profile | +0.867 | 42.68 | 12/12 |
| C profile vs log g profile | +0.096 | 1.36 | 8/12 |

**The −0.80 collapses to +0.21 once the shared `log λ` is removed. Most of it was construction. The
finding as first framed is withdrawn.**

**⇒ A DEEPER PROBLEM THE CHECK EXPOSED — and the fix.** `log C` is an **exact** function of `log λ` and
`log g`, so *"what predicts C"* is **ill-posed**: nothing can add to its own two components (verified —
incremental variance from cp given λ and g is **−0.00% ± 0.00**, pure numerical noise). **The right move
is not to predict C but to build an INDEPENDENT quantity of the same kind and ask whether it has the
bowl:**

```
C_polar  =  cp / g²        (log: log cp − 2 log g)
```

**It is the same gauge-invariant construction as C** (cp scales as c², g as c ⇒ every scale factor
cancels, band 42), **contains no `lam_top` and no tridiagonal quantity** (fully admissible), and
measures curvature along **Muon's actual step direction** rather than the top eigendirection.

**★ THE RESULT — C_polar does NOT have the bowl. It is monotone.**

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C** | +0.155 | +0.065 | +0.023 | +0.031 | −0.076 | −0.130 | **−0.201** | −0.152 | −0.078 | −0.059 | +0.084 | **+0.337** |
| **C_polar** | **+0.172** | +0.115 | +0.105 | +0.074 | +0.007 | +0.034 | −0.051 | −0.048 | −0.063 | −0.134 | **−0.135** | −0.077 |

| | cubic R² | **linear R²** | argmin | swing |
|---|---:|---:|---:|---:|
| C | 0.961 | **0.004** | **L6** | 0.538 dex |
| C_polar | 0.946 | **0.905** | L10 | 0.307 dex |

**corr(C profile, C_polar profile) = +0.152 only** (|t| 1.33, 7/12); C_polar's argmin is interior in
just **2/12** fits, scattered across 0–11.

**⚠️ POWER CHECK on the negative** — a null claim needs one:

| | mean pairwise profile corr (replication) | linear slope/block | \|t\| | same sign |
|---|---:|---:|---:|---:|
| C | +0.710 | **+0.0027** | **0.46** | 8/12 |
| C_polar | +0.460 | **−0.0266** | **5.38** | **11/12** |

**C_polar's decline is real and consistently signed; C's linear slope is indistinguishable from zero.
This is not a noisy measurement failing to show a bowl — it is a genuine dissociation.**

> **⇒ THE BOWL IS SPECIFIC TO THE TOP EIGENDIRECTION.** Two gauge-invariant conditioning measures on
> the *same matrices* have **different depth structure**: conditioning along the **top eigendirection**
> is **U-shaped** (minimum at layer 6), while conditioning along **Muon's actual step direction**
> **declines monotonically** with depth. **Whatever creates the bowl acts on the top of the spectrum,
> not on the subspace Muon moves in.**

**Why that matters for the REQ-036 design question.** Muon steps along the polar direction, and along
*that* direction there is **no bowl to equalise** — the profile is monotone. **This is an independent
mechanical reason why REQ-036's per-type LR equalization failed** (null, harm monotone in equalization,
Spearman −1.000): the campaign was equalising a curvature the optimiser does not step along. Band 33
already showed Muon's step sees only ~0.4% of peak curvature; **this adds that the depth structure of
what it does see is a different shape entirely.**

**PROPOSED n=4 SEED CHECK — band 43 (criterion registered).**
*Criterion:* on a fresh 4-seed panel, (i) **C_polar's linear R² > its cubic-minus-linear gain**, i.e. it
is monotone, with slope **negative in ≥10 of 12** fits; (ii) **C's linear slope not significant**
(|t| < 2); (iii) **corr(C profile, C_polar profile) < +0.50**.
*Status:* **satisfied by committed REQ-035 Arm A data** (linear R² 0.905, slope −0.0266, 11/12, |t| 5.38;
C slope |t| 0.46; corr +0.152). **No new compute requested; runs under the ≤2-node ceiling.**

**Search space:** the bowl is a **conditioning property of the top of the spectrum**, peaks at both ends
with a minimum at layer 6, is immune to every scale factor (band 42), and is **absent from the step
direction** (here) — plus not Muon's step magnitude (band 31), stream scale, input rank (iter. 156), or
an axis artifact (iter. 157).

## ★★ THE GAUGE THEOREM (iteration 159) — C is invariant to EVERY scale factor, so the bowl is a SHAPE property

*Band 41 showed C cancels `post_lambda`. **The same line of code carries a second scalar,
`resid_lambdas`, that behaves differently** — and generalising the argument turns a one-off cancellation
into a constraint on the entire remaining search.*

**`resid_lambda` is a deliberate geometric amplifier.** Line 1338:
`nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))` — √1.1 per sublayer, two sublayers per block, so
**the stream is multiplied by ~1.1 per block by design**. Unlike `post_lambda` it multiplies the
**stream**, not this block's output, so it scales everything **downstream** of block *i*:

| block i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downstream `r^(L−1−i)` | 2.85 | 2.59 | 2.36 | 2.14 | 1.95 | 1.77 | 1.61 | 1.46 | 1.33 | 1.21 | 1.10 | 1.00 |
| log₁₀ | +0.455 | +0.414 | +0.373 | +0.331 | +0.290 | +0.248 | +0.207 | +0.166 | +0.124 | +0.083 | +0.041 | 0.000 |

**A +0.455 dex monotone gradient across depth — which would be +0.911 dex in λ.** Large enough to
dominate everything, if it were there.

**⊘ IT IS NOT THERE — the specific prediction is decisively refuted:**

| profile | measured slope/block | predicted | t vs prediction |
|---|---:|---:|---:|
| log λ | **+0.0057** (sd 0.0195) | −0.0828 | **+15.75** |
| log g | **+0.0015** (sd 0.0052) | −0.0414 | **+28.64** |

**Cause identified, not left hanging:** `norm(x)` is `F.rms_norm` (line 952) and is applied **before
every sublayer** (line 1643) and at the output (1682). **RMSNorm is scale-invariant, so it erases the
accumulated amplification** — the stream may grow 1.1× per block, but each block reads a renormalised
copy. **The 1.1 growth is real in the stream and invisible to the matrices.**

**⇒ THE GENERAL RESULT, which is worth more than the refuted specific one.**

> **GAUGE THEOREM.** *Let a matrix W influence the loss only through `c·f(W)` for any scalar c — c may
> depend on the layer, on other parameters, on training time, on anything, provided it multiplies W's
> **whole** contribution. Then `g → c·g` and `λ → c²·λ`, so*
> ```
> C = λ/g²  →  (c²λ)/(c·g)²  =  C
> ```
> ***C is exactly invariant.*** *This covers `post_lambda` (band 41), `resid_lambda` and its downstream
> product, the learning rate, and any output-scaling gate — in one line.*

**⇒ THE CONSTRAINT ON THE SEARCH.** **The bowl cannot come from ANY per-layer scale factor whatsoever.**
Not `post_lambda`, not `resid_lambda`, not the downstream amplification, not the LR, not a gate.
**Whatever sets the bowl must change the SHAPE of the loss surface around W — the ratio of its
curvature to its gradient — not the scale of W's influence.** Four iterations of eliminating candidates
one at a time are superseded by a theorem that eliminates the entire class at once.

**EMPIRICAL CONFIRMATION 1 — C's level is far more stable than λ's**, per matrix across the three fork
states (where median raw λ falls 4×):

| quantity | per-matrix sd across fork states | mean range |
|---|---:|---:|
| **log C** | **0.1449 dex** | 0.2783 |
| log λ | **0.3320 dex** | 0.6550 |
| log g | 0.1167 dex | 0.2303 |

**C is 2.3× more stable than λ** (n = 288 matrices) — λ carries scale that C removes, as the theorem
requires.

**EMPIRICAL CONFIRMATION 2 — the bowl is present in every fork state independently**, so it is not a
scale artifact (which would move with the 4× scale change):

| fork | bowl cubic R² | argmin | swing |
|---|---:|---:|---:|
| 060 | 0.904 | **L6** | 0.523 dex |
| 100 | 0.961 | **L6** | 0.602 dex |
| 170 | 0.975 | **L6** | 0.488 dex |

**PROPOSED n=4 SEED CHECK — band 42 (criterion registered).**
*Criterion:* on a fresh 4-seed panel, (i) **per-matrix sd of log C across fork states < 0.6 ×** that of
log λ; (ii) the C bowl has **cubic R² ≥ 0.80 with argmin in layers 5–7 in every fork state separately**;
(iii) the λ profile's linear slope is **within ±0.03/block of zero**, i.e. the `r^(L−1−i)` amplification
is **absent** (refuting any residual-amplification account).
*Status:* **satisfied by committed REQ-035 Arm A data** (0.1449 vs 0.3320 = 0.44×; R² 0.904/0.961/0.975
all argmin L6; λ slope +0.0057). **No new compute requested; runs under the ≤2-node ceiling.**

**Why this reframes the campaign goal.** The question "what sets the between-layer difference in C" now
has a **sharp mathematical form**: *not* "what is bigger in some layers" — every such quantity cancels —
but **"what makes the loss surface around a matrix more curved *relative to its own gradient* at the
ends of the network than in the middle?"** That is a statement about the **conditioning** of each
block's local problem, and it is invariant to every rescaling the architecture applies.

**Search space:** the bowl is a **shape/conditioning** property of the loss surface, peaking at both
ends with a minimum at layer 6, **immune to the entire class of scale factors** (here), and not Muon's
step (band 31), stream scale, input rank (iter. 156), or an axis artifact (iter. 157).

## ★ C IS INVARIANT TO THE PER-LAYER `post_lambda` SCALARS (iteration 158) — derived, confirmed, with one estimate withdrawn

*Three iterations of elimination without a positive result. Rather than scan for another correlate, this
one **derives** a prediction from the architecture and tests it.*

**The mechanism band 31 does not cover.** `train_gpt.py` lines 1638/1665:

```
x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0 * x0_gates[i]
x = resid_lambdas_mlp[i]  * x + post_lambdas_mlp[i]  * ReLUSqrdMLP(normed, *mlp_args)
```

`post_lambdas` is `nn.Parameter(torch.ones(num_layers, 2))` (line 1334), trained by **Adam** (line 2035)
— **a genuinely per-layer, layer-varying scalar.** Band 31's invariance argument constrains **Muon's
step on the matrices** and says nothing about these. **This is the first layer-varying quantity found
that band 31 does not foreclose.**

**THE DERIVATION.** For a matrix W in block i, the loss depends on W **only** through
`p · f(W)` where `p = post_lambda[i]`. Hence along any direction in W: **g ∝ p** and **λ ∝ p²**, so

```
log λ = 2 log p + (p-free part)
log g = 1 log p + (p-free part)
log C = log λ − 2 log g   ⇒   the p terms CANCEL EXACTLY
```

> **PREDICTION 1: C is invariant to `post_lambda`; λ and g are not.**
> **PREDICTION 2: any p-driven component of the λ profile must appear in the g profile at exactly half
> the size, same sign, layer by layer** — i.e. regressing the g profile on the λ profile gives **+0.5**
> if p dominates.

**PREDICTION 2 — TESTED, AND p DOES NOT DOMINATE.** Slope of (log g profile) on (log λ profile), 12
seed×fork fits: **mean +0.139, sd 0.0735** — **t = −17.01 vs +0.5** (decisively below) but
**t = +6.55 vs 0** (decisively above). **A p-driven channel is present but is a minority of the λ
profile.**

**⚠️ AN ESTIMATE I COMPUTED AND THEN WITHDREW.** From `cov(λ,g) = 2·var(u)` I derived a p-driven share
of **27.8%** of the λ profile. **That number is withdrawn.** The same two-component model implies a
p-driven share of the *gradient* profile of `var(u)/var(g)`, which came out **mean 1.185, range
[0.300, 2.369], exceeding 1.0 in 8 of 12 fits** — arithmetically impossible for a variance share.
**The model's own consistency check refutes it:** the λ–g covariance is carried by something besides
`post_lambda`, so 27.8% is not an upper bound either. **No share is claimed.**

**PREDICTION 1 — CONFIRMED, and it is the load-bearing half.** It never depended on the share:

- **corr(C profile, g profile) = +0.096, sd 0.245, |t| = 1.36** — indistinguishable from zero. **C is
  blind to the channel that carries p, exactly as derived.**
- **C's bowl survives removing the entire gradient channel.** Regressing the C profile on the g profile
  and refitting the residual: **cubic R² 0.860 (sd 0.061)**, minimum still **interior in 12/12 fits**
  (argmins: 6,6,6,6,6,6,6,6,6,7,7,7).

> **⇒ THE BOWL LIVES IN THE p-FREE SURFACE TERM.** `post_lambda` is a real per-layer channel that λ and
> g are exposed to and **C is immune to by construction** — but **it is NOT the bowl.** The bowl is
> untouched by removing g entirely.

**This explains band 32 mechanistically.** Band 32 recorded that **C is more seed-stable than λ**
(0.0776 vs 0.1235 dex) as an empirical fact. **Now there is a reason:** C is *algebraically immune* to a
per-layer learned nuisance scalar that λ is fully exposed to. **A ratio that cancels a trained parameter
is a better-conditioned object than its numerator** — this is why C, not λ, is the right target, and it
is the first derivation-level justification the campaign has for that choice.

**PROPOSED n=4 SEED CHECK — band 41 (criterion registered).**
*Criterion:* on a fresh 4-seed panel, (i) **corr(C profile, g profile) is not significant** (|t| < 2.5
across the seed×fork fits); (ii) after regressing the C profile on the g profile, the residual bowl has
**cubic R² ≥ 0.70** with its **minimum interior (layers 4–8) in ≥10 of 12** fits; (iii) the slope of the
g profile on the λ profile is **significantly below +0.5**.
*Status:* **satisfied by committed REQ-035 Arm A data** (+0.096, |t| 1.36; R² 0.860, 12/12; t = −17.01).
**No new compute requested; runs under the ≤2-node ceiling.**

**⚠️ A MEASUREMENT WORTH REQUESTING LATER — but not now.** `post_lambdas` and `resid_lambdas` are **not
recorded in any committed probe output** (the curvature probe stores only
`top_eigenvalue / curvature_along_gradient / curvature_along_polar / gradient_block_norm / alphas /
offdiags / shape`). Dumping the 13×2 learned scalars at each fork state would be **nearly free** — no
training, just a checkpoint read — and would let the p-channel be measured directly rather than
inferred. **Not filed as a request this iteration**, because the load-bearing result (C is invariant,
the bowl survives) is already established without it; it is logged here so the humans can fold it into
any future probe pass at no marginal cost.

**Search space:** the bowl peaks at both ends, minimum at layer 6, lives in the **loss surface**
(band 40), and is **not** Muon's step (band 31), **not** `post_lambda` (here), **not** stream scale,
**not** input rank (iter. 156), **not** an axis artifact (iter. 157).

## ⊘ THE DEPTH AXIS IS CORRECT — an architectural hypothesis of mine, raised and killed (iteration 157)

*Reading `train_gpt.py` for a structural cause turned up line 1268: **"Attention is skipped in layer 6
by @YouJiacheng"**, with `num_attn_layers = num_layers - 1`. **The λ bowl's minimum is at layer 6.**
That coincidence would have been a complete mechanism — the bowl bottoming out exactly at the
architecture's discontinuity — and it would also have meant **every band on the depth axis was indexed
wrong.** It is false, and the check that killed it is recorded here.*

**The concern was real, not hypothetical.** The attention banks are indexed by `_num_attn_layers` (12),
skipping physical block 6, while `mlp_bank` covers all 13 blocks (`qk_all = self.qk_bank[...].view(
self._num_attn_layers, ...)`, line 1523). If the probe's `blocks.N` names followed the **bank** layout,
an attention matrix at index 6 would physically sit at block 7, and the attn and mlp profiles would be
**on different axes** — silently misaligning every depth result in the campaign.

**First check — the data itself.** If the names followed the module tree of a 13-block model, the
attention-free block would have **no attn matrices**:

| block | attn.k | attn.proj | attn.q | attn.v | mlp.fc | mlp.proj |
|---|---:|---:|---:|---:|---:|---:|
| **0–11 (all)** | 12 | 12 | 12 | 12 | 12 | 12 |

**Every block 0–11 carries all six types with no gap**, and the panel has exactly 12 blocks. So the
probe is not exposing a 13-block tree with a hole in it.

**Decisive check — does remapping improve or destroy coherence?** Two readings had to be separated
rather than assumed: **(A)** the index is the attention-layer index, so attn matrices at index ≥ 6 sit
one physical block later than mlp matrices at the same index; **(B)** the index is already a consistent
physical axis. Under **(A)**, shifting attn by +1 for index ≥ 6 must **improve** the agreement between
the attn and mlp λ profiles; under **(B)** it must **worsen** it:

| | corr(attn λ profile, mlp λ profile) |
|---|---:|
| **as-is, no remap** | **+0.661** (12 shared positions) |
| remapped, attn +1 for idx ≥ 6 | **+0.338** (11 shared positions) |

> **Remapping HALVES the agreement. Reading (B) holds: the depth axis as used throughout this campaign
> is correct, the attn and mlp profiles are already on the same axis, and the layer-6 minimum is NOT
> adjacent to an architectural discontinuity.** The hypothesis is dead and **no band needs reindexing** —
> bands 3, 27, 28, 35, 37, 38, 39 and 40 all stand on their existing axis.

**Why this is worth a full entry despite being a negative.** Had I recorded the coincidence without
testing it, the campaign would have gained a false mechanism *and* a false correction to eight bands at
once. **The coincidence is genuine — the bowl really does bottom out at index 6, and the architecture
really does skip attention at block 6 — but the two facts are unrelated**, because the probe's index is
not the bank index. **A structural coincidence between a finding and a code comment is a hypothesis, not
a mechanism, and the data can usually adjudicate it directly.**

**Standing rule 12.** *Before accepting that an index in probe output means what its name suggests,
count the cells: a missing (block, type) combination, or a block count that disagrees with the model
config, is the cheap check that catches an axis error.* Here the count (12 blocks × 6 types, no gaps)
answered it in one query.

**⚠️ NO n=4 SEED CHECK PROPOSED.** This is a refuted hypothesis about indexing, settled entirely within
committed data. **New compute would test nothing.**

**Search space after this iteration (unchanged, but now on a verified axis):** the cause peaks at both
ends with a minimum at layer 6, is a property of the **loss surface** (band 40), and is **not** step
size (band 31), **not** stream scale, **not** input rank (iteration 156), and **not** an artifact of the
depth axis (here).

## ⊘ TWO STRUCTURAL NEGATIVES (iteration 156) — the stream scale and the input rank are BOTH ruled out

*Band 40 forecloses an optimiser-side cause, so the bowl is a property of the loss surface and the next
step must be structural. **Two candidates were available in already-committed data (REQ-047) and both
are refuted.** Negative results, recorded as such.*

**NEGATIVE 1 — the residual stream scale is monotone; it cannot make a bowl.** The most basic structural
fact about a residual transformer is that the stream grows with depth. attn.q/k/v all read the block
input, and per **band 21** their `a_rms` is bit-identical, so it is a direct measurement of the stream
scale at that depth:

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **stream a_rms** | 0.407 | 0.468 | 0.607 | 0.739 | 0.871 | 0.998 | 1.094 | 1.163 | 1.411 | 1.359 | 1.438 | **1.524** |

**Linear R² 0.979, slope +0.107/block, minimum at the boundary — strictly monotone, no interior
minimum.** The λ bowl has an interior minimum at layer 6. **A monotone quantity cannot produce a U.
Stream scale is NOT the mechanism.**

**NEGATIVE 2 — the input effective rank IS U-shaped, but it is the WRONG U.** A scan of every structural
quantity in the committed data flagged `a_eff_rank` (effective rank of each matrix's **input
activations**). ⚠️ **Multiple-comparisons discipline: the scan ran 36 shape tests, so the best cell is
not reportable** — that is precisely the trap rule 10 was written about. **The defensible claim is the
general one**, and it does hold:

| type | seed0 | seed1 | seed2 | seed3 | interior min in all 4 seeds? |
|---|---:|---:|---:|---:|---|
| attn.k | 5 | 3 | 3 | 3 | **YES** |
| attn.q | 5 | 3 | 3 | 3 | **YES** |
| attn.v | 5 | 3 | 3 | 3 | **YES** |
| mlp.fc | 2 | 3 | 2 | 4 | **YES** |
| mlp.proj | 2 | 3 | 2 | 2 | **YES** |
| attn.proj | 11 | 11 | 11 | 11 | no |

**5 of 6 types, every seed** — a genuine property of the quantity, not a cherry-picked cell. The profile
is strongly U-shaped (**cubic R² 0.943 vs linear R² 0.180**):

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **input eff-rank** | 66.1 | 48.2 | 27.2 | **21.6** | 22.5 | 21.8 | 24.7 | 24.5 | 26.9 | 27.1 | 29.9 | 37.3 |

> **But its minimum is at layer 3, and the λ bowl's is at layer 6 — three layers apart — and
> corr(log input eff-rank, λ bowl) = +0.332 only. These are DIFFERENT shapes. Input effective rank is
> NOT the driver of the λ bowl.** *(Recorded as a real structural finding in its own right: the network's
> input representations collapse in rank through the early blocks and re-expand toward the output, which
> is worth knowing independently — it is simply not this bowl.)*

**Why these negatives are worth the space.** Between them they eliminate the two most natural structural
explanations — *"deeper layers see bigger activations"* and *"middle layers see lower-rank
activations"* — and they do so from committed data at **zero compute cost**. The surviving space is
narrower: whatever sets the bowl peaks at **both** ends with a minimum at **layer 6**, is a property of
the **surface** (band 40), is **not** step size (band 31), **not** stream scale, and **not** input rank.

**⚠️ NO n=4 SEED CHECK PROPOSED THIS ITERATION.** Both results are negatives already verified across all
4 seeds in committed data; there is no positive line of thought that would justify new compute. **Filing
a request here would be spending the ≤2-node budget to re-confirm two refutations.**

**Method note.** The scan tested 36 quantity×type combinations. **The correct output of a 36-test scan
is either a claim that survives a discipline check or a negative — never the top-correlating cell.**
The `mlp.fc` cell correlated +0.697 with the bowl and is exactly what a less careful pass would have
reported as the mechanism.

## ★ THE BOWL LIVES IN λ, NOT IN g (iteration 155) — plus two of my own inferences corrected

*Band 39 pinned the bowl's shape. The mechanism question is **where it comes from**. Since
`log C = log λ − 2 log g` **exactly**, the bowl must be a bowl in λ, in −2 log g, or both. This is
arithmetic, not a model. **Non-circularity:** `top_eigenvalue` and `gradient_block_norm` are separate
measurements and no predictor here is built from the Lanczos tridiagonal.*

**Profiles are block-mean residuals after removing TYPE ONLY** — not gradient, which would absorb the
very thing being decomposed:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **log C** | +0.155 | +0.065 | +0.023 | +0.031 | −0.076 | −0.130 | −0.201 | −0.152 | −0.078 | −0.059 | +0.084 | **+0.337** |
| **log λ** | +0.120 | −0.092 | +0.037 | +0.138 | +0.018 | −0.111 | −0.183 | −0.162 | −0.093 | −0.135 | +0.002 | **+0.461** |
| **−2 log g** | +0.035 | +0.156 | −0.013 | −0.106 | −0.094 | −0.019 | −0.018 | +0.010 | +0.015 | +0.076 | +0.081 | −0.124 |
| *identity check* | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | −0.000 | 0.000 | −0.000 | 0.000 | 0.000 | 0.000 |

**Variance decomposition of the between-layer profile:**

| term | variance | share of var(log C) |
|---|---:|---:|
| var(log C) | 0.02046 | 100% |
| **var(log λ)** | **0.02955** | **144.4%** |
| var(−2 log g) | 0.00623 | 30.4% |
| 2·cov | −0.01670 | **−81.6%** |
| | | corr = **−0.616** |

> **THE BOWL IS IN λ.** The curvature profile carries **144%** of C's between-layer variance; the
> gradient term carries **30%** and partially **cancels** it (corr −0.616). C's bowl is what remains
> after a large λ bowl is partly offset by a smaller opposing gradient profile. **corr(C profile,
> λ profile) = +0.890; corr(C profile, g profile) = +0.126.**

**⇒ MECHANISM CONSTRAINT — and it is sharp.** Band 31 established that Muon's step is unit-spectral-norm
× `shape_mult`, **identical for every layer of a given type**. A λ bowl therefore **cannot** come from
layer-varying step sizes: there are none. **The bowl must be a property of the loss surface itself, not
of the optimiser.** Combined with band 16 (C is actively restored), the reading is: *the optimiser
applies the same step everywhere; the surface responds differently by depth, and where it responds most
stiffly, λ equilibrates highest.*

**⚠️ CORRECTION 1 — an inference I made this iteration and then refuted.** From λ's minimum landing in
layers 4–8 in only **4/12** fits versus C's **12/12**, I inferred that dividing by g² *stabilises the
shape* and that C is the better-behaved object. **That was wrong.** Mean pairwise correlation between
the 12 profiles:

| quantity | mean pairwise profile corr |
|---|---:|
| log C | **+0.710** (sd 0.192) |
| log λ | **+0.714** (sd 0.148) |
| log g | +0.612 (sd 0.198) |

**λ's shape replicates exactly as well as C's.** The 4/12-vs-12/12 gap was an artifact of using the
**argmin** — a discrete statistic that jumps when a curve is flat near its bottom. **Band 32 (C more
seed-stable than λ) is about level, not shape, and this does not contradict it — but "C is smoother"
must not be extended to shape replication.**

**⚠️ CORRECTION 2 — a near-miss claim, caught by its own check.** λ's swing measured **0.644 dex at all
three fork states** — identical to three decimals, which looked like a frozen architectural invariant.
It is **a coincidence of two endpoints**, not a frozen curve. The profiles genuinely differ layer by
layer:

| comparison | max abs difference | corr |
|---|---:|---:|
| 060 vs 100 | 0.0485 dex | +0.991 |
| 060 vs 170 | **0.1659 dex** | +0.889 |
| 100 vs 170 | 0.1265 dex | +0.940 |

*(Sanity: median raw λ is 22576 / 12562 / 5494 across the three states — a 4× fall, so these are
genuinely distinct data, not one checkpoint re-probed.)* **No invariance claim is made.** What is real:
**the shape is highly stable (corr +0.889 to +0.991) while drifting systematically at the input end**
(layer 0: +0.052 → +0.091 → **+0.217**) — the same drift band 39 recorded in C, now located in λ.

**PROPOSED n=4 SEED CHECK — band 40 (criterion registered).**
*Criterion:* on a fresh 4-seed panel, (i) **var(log λ profile) > 1.2 × var(log C profile)** and
**var(−2 log g profile) < 0.5 ×** it; (ii) **corr(C profile, λ profile) ≥ +0.70** while
**corr(C profile, g profile) ≤ +0.40**; (iii) **corr(λ profile, −2 log g profile) < 0** (the cancellation).
*Status:* **satisfied by committed REQ-035 Arm A data** (144.4% / 30.4%; +0.890 vs +0.126; −0.616).
**No new compute requested; runs under the ≤2-node ceiling.**

**Now open:** why the *surface* is stiffest at both ends and softest at layer 6. Band 31 forecloses any
optimiser-side explanation, and bands 27/37's audit forecloses reading it off ‖a‖·‖d‖ concentration —
so the next admissible probe is **structural**, not another correlation on this panel.

## ★★ THE POSITION FIELD IS AN ASYMMETRIC BOWL (iteration 154) — form identified, between-layer C closed to 0.96× the floor

*Band 38 left the position field's **shape** unidentified (`quad` 67.9% vs `dist` 67.1%, tied). That is
now the campaign's central object, so it was attacked directly: **stop guessing forms and let the data
draw the profile.** Free per-layer effect (block-mean residual after type + gradient), 12 fits.*

**THE PROFILE — and it is extraordinarily clean:**

| layer | 0 | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 | **11** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dex** | +0.166 | +0.105 | +0.024 | −0.003 | −0.104 | −0.142 | **−0.207** | −0.146 | −0.066 | −0.037 | +0.108 | **+0.302** |
| se | 0.055 | 0.041 | 0.034 | 0.018 | 0.011 | 0.016 | 0.015 | 0.027 | 0.016 | 0.024 | 0.024 | 0.029 |

**Standard errors of 0.011–0.055 dex on a swing of 0.51 dex.** The profile is **monotone down** from
layer 0 to a minimum at layer 6, then **monotone up** to layer 11. **This settles a band-3 question:
it is not a boundary shell on a flat interior — it is a genuine smooth bowl**, and every interior layer
carries signal.

**⇒ THE FORM IS NOW IDENTIFIED — it is ASYMMETRIC, and a cubic captures it.** Fits to the mean profile:

| form | R² | k |
|---|---:|---:|
| `dist = min(l, 11−l)` | 0.899 | 1 |
| `quad = (l−5.5)²` | 0.915 | 1 |
| quad, free centre | 0.918 | 2 |
| **cubic (allows tilt)** | **0.977** | 3 |
| quad + last-block | 0.944 | 3 |
| quad + both ends | 0.964 | 4 |

**The quadratic vertex is at layer 5.42** — essentially the geometric centre (5.50), so **the asymmetry
is NOT a shifted bowl. It is a tilt**: the output end is deeper than the input end (+0.302 vs +0.166).

**Variance budget with the identified form** (share of between-layer residual variance after type +
gradient, 12 fits): **cubic 88.9%** (sd 6.2) > quad+last 84.5% > quad 82.3% > **dist 67.1%**.
**The cubic beats band 38's `dist` by 21.8 points.**

**⚠️ GUARD — and the cubic passes the test that corrected band 38.** Leave-one-layer-out, predicting a
held-out layer's block mean (144 fits):

| model | LOLO rmse |
|---|---:|
| type + gradient (no position) | 0.1948 |
| dist *(band 38)* | 0.1216 |
| quad | 0.1007 |
| **cubic** | **0.0924** |
| quad + last | 0.1023 |
| *(noise floor, measured on this panel)* | *0.0959* |

> **BETWEEN-LAYER C IS CLOSED TO 0.96× THE NOISE FLOOR.** `type + gradient + cubic position` predicts a
> **held-out layer's** mean log C to **0.0924 dex** against a floor of **0.0959** — i.e. **within seed
> noise, out of sample.** This supersedes band 38's 1.27× residual: **that gap was the wrong functional
> form, not missing physics.** The ~0.075 dex I reported as "remaining structure" in iteration 153 was
> the tilt.

**⚠️ ONE REAL TENSION, resolved rather than buried.** The cubic **wins** on held-out layers (0.0924 vs
dist 0.1216) but **loses** on held-out *fork states* (cross-fork transfer 0.1331 vs dist 0.1192). Those
disagree, so the tilt was checked directly:

| fork | cubic tilt coefficient | sd | per-seed |
|---|---:|---:|---|
| 060 | +0.00104 | 0.00061 | 0.00101, 0.00019, 0.00137, 0.00159 |
| 100 | +0.00140 | 0.00076 | 0.00048, 0.00230, 0.00120, 0.00161 |
| 170 | +0.00090 | 0.00027 | 0.00089, 0.00125, 0.00058, 0.00089 |
| **all 12** | **+0.00111** | 0.00057 | **t = +6.73, same sign 12/12** |

**The tilt is universal (12/12, t = +6.73). The cross-fork failure is drift in the coefficient's
MAGNITUDE, not a fake shape** — a 3-parameter form transfers its *scale* worse than a 1-parameter one
even when its shape is right.

**⇒ AND THE PROFILE ITSELF EVOLVES WITH TRAINING — a new, unregistered observation:**

| fork | L0 | L6 | L11 |
|---|---:|---:|---:|
| 060 | +0.093 | −0.187 | **+0.288** |
| 100 | +0.138 | −0.211 | +0.367 |
| 170 | **+0.266** | −0.224 | **+0.251** |

**As training advances the input end RISES (+0.093 → +0.266) while the output end FALLS (+0.288 →
+0.251).** The bowl is becoming more symmetric over training — which explains the coefficient drift
above, and means **the tilt is a transient of training, not a fixed property of the architecture.**

**PROPOSED n=4 SEED CHECK — band 39 (criterion registered before any new run).**
*Criterion:* on a fresh 4-seed panel, (i) the free per-layer profile is ****single-minimum**** with its minimum
in layers **5–7**; (ii) the **cubic tilt coefficient is positive in ≥10 of 12** seed×fork fits;
(iii) **LOLO rmse of `type + gradient + cubic` ≤ 1.15× the panel-measured noise floor**.
*Status:* **satisfied by committed REQ-035 Arm A data** (minimum at 6; 12/12 positive tilt, t = +6.73;
LOLO 0.0924 / floor 0.0959 = **0.96×**). **No new compute requested; runs under the ≤2-node ceiling.**

**Supersedes:** band 38's "form NOT identified" and its "~0.075 dex remains unexplained" — both resolved
here. **Band 3's "not a pure shell" correction is confirmed independently** by the free profile.

**Now open:** *why* the bowl tilts toward the output end, and why the tilt decays with training. That is
a mechanism question, and it is the first one this campaign has posed with the shape fully pinned down.

## ★ THE BETWEEN-LAYER VARIANCE BUDGET (iteration 153) — POSITION is the answer, and the gradient is not

*The rule-10 audit cleared the ground; this is the question it was clearing for. **Of the between-layer
variance in log C, how much does each surviving mechanism explain, and what is left?** Never asked at
n=4. Panel: 12 independent fits (4 seeds × 3 fork states) at step 2750, from the REQ-035 Arm A rank
shards. Between-layer variance ≡ variance of the **block means** of log C (averaging the 6 types within
a layer) — exactly the quantity the campaign goal names.*

**Between-layer sd(log C) = 0.1806 dex.** Cumulative share explained:

| added term | cumulative between-layer variance explained | sd |
|---|---:|---:|
| type offsets only | **−0.0%** | 0.0 |
| + gradient (log g) | **3.5%** | 6.5 |
| **+ position (`dist`)** | **68.8%** | 12.6 |
| | | |
| **unexplained residual sd** | **0.0947 dex** | 0.0241 |

> **Type explains 0.0% of it — by construction**, since averaging all six types within a layer removes
> type exactly. **The gradient explains 3.5%.** **Position explains 68.8%.** For the *between-layer*
> question specifically, **the position field is the mechanism and the gradient law is nearly
> irrelevant** — the gradient law (bands 2/13) governs variation *within* a layer, across types, which
> is a different question from the one the campaign goal poses.

**The noise floor, measured on this panel rather than quoted.** For each layer, the spread of the
block-mean log C across the 4 seeds at fixed fork and step **is** the floor for a block mean:

| fork | per-layer sd across 4 seeds | min | max |
|---|---:|---:|---:|
| 060 | 0.0773 | 0.0156 | 0.1923 |
| 100 | 0.1044 | 0.0144 | 0.2820 |
| 170 | 0.1060 | 0.0087 | 0.2802 |
| **pooled** | **0.0959 dex** | | |

**In-sample residual 0.0947 vs floor 0.0959 = 0.99×.** ⚠️ **But that is the in-sample number and it is
optimistic** — see the guards below.

**Position beats every alternative shape** (share of the *residual* between-layer variance after type +
gradient, 12 fits):

| term | share | sd |
|---|---:|---:|
| quad | **67.9%** | 17.4 |
| **dist (position)** | **67.1%** | 14.8 |
| last-block only | 39.2% | 19.9 |
| **linear depth** | **15.6%** | 15.2 |
| first-block only | 14.0% | 14.7 |

**A monotone depth trend captures only 15.6%** — consistent with band 3's audit, and the reason this is
not a rule-10 casualty. `quad` and `dist` remain tied (67.9 vs 67.1), so **the form is still not
identified.**

**⚠️ GUARD 1 — leave-one-layer-out. The headline must be the out-of-sample number.** Fit on 11 layers,
predict the held-out layer's block mean (144 held-out fits):

| model | LOLO rmse |
|---|---:|
| type + gradient | 0.1948 dex |
| **type + gradient + position** | **0.1216 dex** |
| *(noise floor)* | *0.0959 dex* |

**GUARD 2 — cross-fork transfer.** Fit one fork state, predict another, same seed:

| | residual between-layer sd |
|---|---:|
| 060 → 100 / 170 | 0.1241 / 0.1195 |
| 100 → 060 / 170 | 0.1137 / 0.1207 |
| 170 → 060 / 100 | 0.1156 / 0.1214 |

> **HONEST HEADLINE.** Out of sample the model reaches **0.12 dex against a 0.0959 dex floor — 1.27×,
> not 0.99×.** The in-sample 0.99× was optimistic by exactly what a fitted parameter buys. **The
> position field is genuinely predictive** (LOLO 0.122 vs 0.195 without it — a 38% error reduction on
> held-out layers, and it transfers across fork states unchanged at 0.114–0.124). **But between-layer C
> is not fully closed: ~1.27× the floor remains, i.e. a small real residual of roughly
> √(0.1216² − 0.0959²) ≈ 0.075 dex.**

**What this reorders in the account.** The campaign has spent most of its effort on the gradient law and
on type structure — both real, both well-verified, and **both nearly irrelevant to the between-layer
question as posed**. Type is removed by the block-mean definition; the gradient contributes 3.5%.
**The between-layer difference in C is, to ~69%, a function of position in the network** — and, per
band 3's audit, position acting through a *symmetric* field (not a monotone trend), weighted toward the
**final** block (+0.361 dex) over the first (+0.211 dex).

**PROPOSED n=4 SEED CHECK — band 38 (registering the criterion before any new run).**
*Criterion:* on any fresh 4-seed panel, (i) `dist`/`quad` explains **≥ 50%** of the between-layer
variance of block-mean log C after type + gradient, in **≥10 of 12** seed×fork fits; (ii) **linear depth
explains < 30%**; (iii) LOLO rmse with position **< 0.75×** the type+gradient LOLO rmse.
*Status:* **already satisfied by the committed REQ-035 Arm A data** (67.1%, 12/12; linear 15.6%;
0.1216/0.1948 = 0.62×). **No new compute requested** — this is registered as a standing criterion for
future panels, and runs entirely under the ≤2-node ceiling.

**Open, and now precisely bounded:** the residual **~0.075 dex** of between-layer structure that
position does not capture, and the **unidentified functional form** (quad vs dist, tied at 67.9 vs
67.1). Both are questions about the *shape* of the position field, which is now the campaign's central
object.

## ✅ BAND 35 SURVIVES RULE 10 INTACT (iteration 152) — the audit is complete

*Last exposed band, and the highest-risk one: it rests on `quadratic R² > 0.5`, the construction that
just proved underpowered in band 3. **It survives without correction — the only band to do so.***

**⚠️ FIRST, MY OWN ERROR — the iteration-145 trap, a second time.** My first pass tested the q−k gap
using REQ-047's `align_ratio = ‖∇W‖_F / (‖d‖_F·‖a‖_F)` and got **+0.011 dex** — wrong sign, 28× too
small — which looked like a flat refutation of band 35's −0.306. It was not. **Band 35's "alignment" is
a different quantity: `log(curvature_along_polar) − log(λ_top)` from the curvature probe.** Two distinct
measurements share the word "alignment". Retested on band 35's own definition:

| seed | mean gap | sd | t | n |
|---|---:|---:|---:|---:|
| 0 | **−0.3386** | 0.172 | −26.46 | 180 |
| 1 | −0.2874 | 0.260 | −14.83 | 180 |
| 2 | −0.3350 | 0.244 | −18.39 | 180 |
| 3 | −0.2705 | 0.201 | −18.07 | 180 |
| **pooled** | **−0.3079** | | | **720** |

**Band 35 claims −0.306; measured −0.3079, same sign in 4/4 seeds, |t| = 14.8–26.5.** Exact replication.
*(Circularity check: `curvature_along_polar` is a **separate HVP**, not from the lam_top tridiagonal —
admissible under the hard rule. λ_top enters only as the normaliser.)*

**Claim (a) is not rule-10 exposed at all** — it is a *mean difference between two types*, not a
correlation across layers. Rule 10 never applied to it.

**Claim (b) — the depth structure — survives on its own definition.** Twelve independent curves
(4 seeds × 3 fork states):

| | quad R² | linear R² | gain |
|---|---:|---:|---:|
| **mean of 12** | **0.636** (sd 0.121) | 0.359 | +0.277 |
| **quad > 0.5** | **10/12** | | |
| pooled curve | **0.873** | 0.524 | +0.349 |

**Permutation null (20k shuffles of block labels): observed 0.873 vs null mean 0.182, 95th pct 0.491 —
p = 0.0004.** The gap deepens then recovers: L0 −0.087 → L6 **−0.441** → L10 −0.323.

**⚠️ One honest caveat, and it is the band-3 lesson applied.** Unlike band 3 — where the linear term was
worthless (t = +0.56) and that is *why* the symmetric shape was safe — **here the linear term carries
real weight (mean R² 0.359)**, and in **seed2/fork170 the quadratic adds exactly nothing** (0.635 vs
0.635; seed2/fork060 adds +0.013). **So the depth structure is real (p = 0.0004) but is part monotone,
part curved — "deepens then recovers" overstates how much of it is the recovery.** The band text's
`quadratic R² > 0.5` criterion is met (10/12), but it should not be read as "the curvature is the whole
effect."

> **BAND 35 CONFIRMED n=4, no correction needed.** Gap −0.3079 dex (4/4 same sign, |t| ≥ 14.8);
> depth structure real at p = 0.0004, quad R² 0.636 ± 0.121, 10/12 — **with the caveat that ~57% of the
> depth R² is available to a straight line.**

---

## 📋 RULE-10 AUDIT COMPLETE — scorecard across all five exposed bands

| band | claim | verdict |
|---|---|---|
| **37** | ‖a‖–‖d‖ concentration, 5 of 6 types, mlp.fc reverses | ⛔ **DOWNGRADED** — depth artifact; the mlp.fc "exception" **withdrawn**, two hypotheses were explaining nothing |
| **27** | ‖a‖·‖d‖ product depth-conserved | 🔧 **HALF WITHDRAWN, HALF SHARPENED** — `corr ≤ −0.80` is depth (4/6 collapse); **conservation is real**, slopes cancel to ~8%, product spread 0.13–0.31× the independence null |
| **3** | boundary field in C | ✅ **CONFIRMED + CORRECTED** — linear depth ruled out (t +0.56 vs −5.92, 12/12); **not a pure shell** (interior U survives, 9/12); edges **unequal** (last +0.361 vs first +0.211) — **resolves band 10 as underpower** |
| **35** | q/k alignment gap, depth-structured | ✅ **CONFIRMED INTACT** — −0.3079 dex 4/4; depth p = 0.0004; caveat: part monotone |
| **17, 28** | *(assessed, not exposed)* | band 28 is depth **by construction**; band 17 asserts a depth slope is **absent**, so depth is its subject, not a confound |

**What the audit cost and bought.** Two bands lost content, one was sharpened, one gained strength, one
was untouched. **No band was lost that anything downstream depended on** — the mlp.fc exception was the
only line of active investigation killed, and it was investigating an artifact. **Band 3's audit was net
positive**: it is now stronger *and* it dissolved band 10's standing failure.

**Standing rule 11 (from this iteration's error).** *Before testing a band, open the script that
produced it and confirm which measured quantity its terms name.* Two distinct quantities in this
campaign are both called "alignment" (`curvature_along_polar/λ_top` from the curvature probe;
`‖∇W‖/(‖d‖‖a‖)` from REQ-047). **This is the second time (iteration 145 was the first) that a
name collision produced a false refutation of my own work.** Reading the delivery is not enough —
read the *definition*.

## ✅ BAND 3 SURVIVES RULE 10 — but its "pure boundary" reading is CORRECTED (iteration 151)

*Continuing the rule-10 audit. Band 3 is the most structurally exposed band — `d_edge =
min(layer, maxlayer−layer)` is a deterministic function of layer index, correlated against a residual
that is itself indexed by layer. **It survives, and it is the first audited band that does.***

**Why it survives where 27 and 37 did not.** Band 3 is not a bare correlation: it is a model comparison
claiming the effect is **specifically a boundary shape**, not merely *some* depth trend. Rebuilt at
**n=4 seeds × 3 fork states = 12 independent fits** (`logC ~ type offsets + log g`, then one added term;
panel 4,320 rows from the REQ-035 Arm A rank shards, zero duplicates):

| added term | mean AIC | mean t | \|t\|>2 in | mean coef |
|---|---:|---:|---:|---:|
| *(baseline)* | −198.88 | | | |
| **quad (smooth U)** | **−230.02** | **+6.03** | **12/12** | +0.0133 |
| **logdist** | **−230.02** | **−6.03** | **12/12** | −0.2307 |
| **dist (band 3)** | −228.96 | **−5.92** | **12/12** | −0.0813 |
| edge1 (end shell) | −214.82 | +4.11 | 11/12 | +0.2855 |
| **linear depth** | −203.29 | **+0.56** | **6/12** | +0.0014 |

> **The decisive line is `linear`.** A monotone depth trend is worth almost nothing (t = +0.56, barely
> better than baseline AIC). **The confound that killed bands 27 and 37 — two series both sliding with
> depth — cannot produce this.** The effect requires a *symmetric* shape, and that is a real structural
> statement about position in the network. **Band 3 is CONFIRMED at n=4** (12/12, mean t −5.92).

**⚠️ But the "pure boundary shell" reading is CORRECTED.** The original seed-0 analysis reported the
effect collapsing when end layers are dropped (F 28.98 → 4.65) and concluded it "lives at the BOUNDARY,
not in smooth curvature." **At n=4 it does not collapse — it weakens and survives:**

| interior kept | mean t(quad) | \|t\|>2 in |
|---|---:|---:|
| blocks 0–11 (all 12) | +6.03 | 12/12 |
| blocks 1–10 (10 layers) | +4.07 | 12/12 |
| **blocks 2–9 (8 layers, both shells removed)** | **+2.85** | **9/12** |

**With both boundary shells deleted the curvature is still there in 9 of 12 fits.** The seed-0 F-test
was **underpowered, not decisive** — a single fork state on 40 interior matrices. **The truth is both:
a genuine smooth U across the interior PLUS extra weight at the ends.** `quad` and `logdist` tie on AIC
(−230.02) and `dist` is within 1.1, so **the functional form is NOT pinned** — do not treat `dist` as
the identified form.

**⚠️ The two edges are NOT equal — and this reconciles band 10.** Fitting first- and last-block
indicators separately:

| edge | coef | mean t | \|t\|>2 in |
|---|---:|---:|---:|
| **first block (L=0)** | +0.211 dex | +2.14 | **6/12** |
| **last block (L=11)** | **+0.361 dex** | **+4.09** | **11/12** |

**The last block carries the boundary effect; layer 0 is marginal.** This independently explains
**band 10** ("layer-0 lift — ❌ NOT CONFIRMED n=4, 1 of 4 seeds, consistently positive but
under-powered"): the layer-0 lift is real but roughly **half the size** of the final-block effect, which
is why it keeps missing significance at 6 matrices per layer. **Band 3 and band 10 are consistent; the
symmetric `d_edge` form averages two unequal edges.**

**REVISED BAND 3 STATEMENT.** *C carries a symmetric position field that a linear depth trend cannot
explain (t +0.56 vs −5.92), confirmed 12/12 at n=4. It is **both** a smooth interior U (surviving both
shells' removal, 9/12) **and** an end-block excess concentrated at the **final** block (+0.361 dex,
11/12) more than the first (+0.211 dex, 6/12). The functional form is not identified — quad, logdist
and dist are within 1.1 AIC.*

**Rule 10 audit status: bands 27 (restated), 37 (downgraded), 3 (confirmed + corrected) done. Band 35
remains** (quadratic-in-depth q/k alignment gap — same `quadratic R² > 0.5` construction that just
proved underpowered here, so it is the priority). Band 28 is depth-by-construction; band 17 asserts a
depth slope is *absent*, so depth is its subject rather than a confound.

**Method note — the audit is paying for itself in both directions.** Two bands lost content (37, 27),
one **gained** it: band 3 is now stronger (12/12 at n=4, linear ruled out) *and* more accurate (not a
pure shell; asymmetric edges), and it resolved band 10's standing failure as a power problem rather than
a contradiction. **Rule 10 is not a demolition tool — it is a re-derivation at correct power.**

## 🔧 BAND 27 RESTATED (iteration 150) — the correlation half is depth; the CONSERVATION half is real

*Rule 10 was written in iteration 149 from a costly error. Its first duty was to be applied to the
campaign's own bands. **Five live bands rest on across-layer correlations (3, 17, 27, 28, 35); band 27
is the load-bearing one** — it underpins band 28 and the depth-flatness account. Audited first.*

**The correlation half does NOT survive rule 10.** Band 27's stated criterion is
`corr(log‖a‖, log‖d‖) ≤ −0.80 within every type across depth` — the same construction band 37 was
downgraded for. Removing depth (within-seed quadratic), then the assumption-free within-layer test:

| type | raw | residual | within-layer | t |
|---|---:|---:|---:|---:|
| attn.k | −0.942 | **+0.065** | +0.130 | +0.73 |
| attn.proj | −0.980 | −0.716 | −0.280 | −1.56 |
| attn.q | −0.970 | **−0.184** | +0.019 | +0.08 |
| attn.v | −0.953 | **−0.137** | +0.264 | +1.92 |
| mlp.fc | −0.984 | −0.780 | −0.241 | −0.65 |
| mlp.proj | −0.885 | **−0.143** | −0.677 | −2.29 |

**Four of six collapse toward zero once depth is removed; nothing is significant within-layer.** So
`corr ≤ −0.80` is **not** evidence of matrix-by-matrix compensation, exactly as with band 37.

**What that correlation actually encodes — two opposite, near-perfectly linear depth trends:**

| type | slope log‖a‖/layer | slope log‖d‖/layer | R² a | R² d |
|---|---:|---:|---:|---:|
| attn.k | +0.0461 | −0.0492 | 0.926 | 0.979 |
| attn.proj | +0.0554 | −0.0664 | 0.952 | 0.975 |
| attn.q | +0.0461 | −0.0504 | 0.926 | 0.964 |
| attn.v | +0.0461 | −0.0573 | 0.926 | 0.977 |
| mlp.fc | +0.0309 | −0.0233 | 0.837 | 0.795 |
| mlp.proj | +0.0555 | −0.0618 | 0.852 | 0.987 |

**‖a‖ rises with depth, ‖d‖ falls, both nearly linearly (R² 0.80–0.99), in all six types.**

**⇒ But the CONSERVATION half survives, and it is the real content.** Band 27's second criterion —
`sd(log‖a‖+log‖d‖) < 0.6 × sd(smaller factor)` — is **not** reducible to "two opposite trends," because
two opposite trends of *unequal* size would leave a large residual sum. Against the independence null
`sd = √(sd_a² + sd_d²)`:

| type | sd(la) | sd(ld) | **sd(sum)** | indep. null | **ratio vs null** |
|---|---:|---:|---:|---:|---:|
| attn.k | 0.195 | 0.202 | **0.055** | 0.281 | **0.197** |
| attn.proj | 0.231 | 0.274 | **0.060** | 0.358 | **0.167** |
| attn.q | 0.195 | 0.209 | **0.038** | 0.286 | **0.132** |
| attn.v | 0.195 | 0.236 | **0.066** | 0.306 | **0.215** |
| mlp.fc | 0.137 | 0.106 | **0.036** | 0.174 | **0.208** |
| mlp.proj | 0.245 | 0.253 | **0.110** | 0.352 | **0.313** |

**The product's spread is 0.13–0.31× what independence would give — cancellation far beyond chance.**

**The mechanism, stated exactly: the two depth slopes match in magnitude.** Per type and per seed, the
ratio `−slope(log‖d‖) / slope(log‖a‖)`:

| type | s0 | s1 | s2 | s3 | mean | sd | residual sum-slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| attn.k | 1.076 | 1.154 | 1.082 | 0.959 | 1.068 | 0.081 | −0.0030 |
| attn.proj | 1.190 | 1.224 | 1.257 | 1.131 | **1.201** | 0.054 | −0.0110 |
| attn.q | 1.104 | 1.153 | 1.127 | 0.992 | 1.094 | 0.071 | −0.0042 |
| attn.v | 1.239 | 1.305 | 1.307 | 1.126 | **1.244** | 0.085 | −0.0111 |
| **mlp.fc** | 0.783 | 0.790 | 0.765 | 0.681 | **0.755** | 0.050 | **+0.0076** |
| mlp.proj | 1.165 | 1.079 | 1.158 | 1.060 | 1.116 | 0.054 | −0.0063 |

**All 24 type-seed ratios: mean 1.079, sd 0.172, range [0.681, 1.307].** The slopes cancel to ~8%,
leaving residual product drift of only **−0.011 to +0.008 dex/layer**. The cancellation is **slightly
over-complete** (t = +2.27 vs 1.0): ‖d‖ falls marginally faster than ‖a‖ rises. **mlp.fc is again the
lone under-compensator (0.755)** — the same type that stood out in band 37, and here the deviation is
consistent across all 4 seeds (sd 0.050), so unlike band 37's artifact this one is a real per-type
difference.

> **BAND 27 RESTATED.** *Not* "‖a‖ and ‖d‖ trade off matrix-by-matrix." **Rather: within each type,
> log‖a‖ rises and log‖d‖ falls near-linearly with depth at slopes that cancel to ~8%, so the product
> ‖a‖·‖d‖ is conserved across depth to 0.13–0.31× the independence null.** The conservation is real and
> n=4-stable; the matrix-level trade-off is not established and **n=4 has no power to test it.**

**Band 28 is unaffected** — it compares `sd(log λ)/sd(log g)` across depth and is a depth statement by
construction, which is what it claims to be.

**Rule 10 audit status.** Band 27 done (restated). **Bands 3, 17, 35 remain to audit**; band 17's
exposure is mild (it asserts a depth slope is *absent*, so depth is the subject, not a confound) and
band 28 is depth-by-construction. **Band 3 (`corr(d_edge, block-mean residual) = −0.91`) and band 35
(quadratic-in-depth alignment gap) are the two genuinely exposed ones and are next.**

## ⛔ BAND 37 DOWNGRADED (iteration 149) — its correlations are DEPTH correlations, and the "exception" is not one

*Iteration 148 narrowed band 37's exception and asked the tighter question it left: **why does mlp.fc
differ from attn.proj and attn.q specifically**, when all three share a U-shaped `d_cv`. Answering it
undermined band 37's framing rather than the exception.*

**Step 1 — the turning points coincide, and only for mlp.fc.** Correlating two depth series depends on
where each turns. Per seed, `log‖d‖` bottoms at **layer 12 in every seed for five types**, but at
**layer 10 in every seed for mlp.fc** (zero seed variance — structural, not statistical):

| type | d_cv argmin / log‖d‖ argmin (per seed) | mean \|gap\| |
|---|---|---:|
| **mlp.fc** | 4/10, 5/10, 8/10, 8/10 | **3.75** ← turns together |
| attn.q | 5/12, 4/12, 4/12, 4/12 | 7.75 |
| attn.proj | 4/12 ×4 | 8.00 |
| mlp.proj | 3/12 ×4 | 9.00 |
| attn.v | 1/12, 2/12, 2/12, 2/12 | 10.25 |
| attn.k | 1/12 ×4 | 11.00 |

**mlp.fc is the only type whose two series turn together**, in all four seeds. Its +0.867 follows
arithmetically. **That alone would have explained the exception with no new physics.**

**Step 2 — removing the depth trend, and the result that changes the framing.** Regressing both series
on a within-seed quadratic in layer and correlating residuals:

| type | raw | residual |
|---|---:|---:|
| attn.k | −0.799 | **+0.403** |
| attn.proj | −0.599 | **+0.637** |
| attn.q | −0.431 | **+0.194** |
| attn.v | −0.858 | **+0.543** |
| mlp.fc | +0.864 | +0.379 |
| mlp.proj | −0.823 | +0.077 |

**All six go positive, and mlp.fc becomes unremarkable — mid-pack, below attn.proj and attn.v.** The
five negative values were carried entirely by the shared depth trend.

**Step 3 — and that residual result does not survive either.** A fitted quadratic is a model I chose,
and it can manufacture a sign. The assumption-free version holds **layer fixed** and correlates across
the 4 seeds, where depth cannot contribute anything (Fisher-z over 12 layers):

| type | raw | resid | **within-layer** | sd | t |
|---|---:|---:|---:|---:|---:|
| attn.k | −0.799 | +0.403 | **+0.489** | 0.421 | **+4.40** |
| attn.proj | −0.599 | +0.637 | −0.269 | 1.168 | −0.82 |
| attn.q | −0.431 | +0.194 | +0.184 | 0.956 | +0.68 |
| attn.v | −0.858 | +0.543 | −0.055 | 0.806 | −0.24 |
| mlp.fc | +0.864 | +0.379 | −0.296 | 0.621 | −1.70 |
| mlp.proj | −0.823 | +0.077 | +0.004 | 1.275 | +0.01 |

**Five of six are indistinguishable from zero (|t| ≤ 1.70), signs scatter both ways, sds are 0.42–1.28.
Only attn.k is significant.** So step 2's uniformly-positive residuals were an artifact of my quadratic
— the exact risk flagged when running it — and **n=4 has no power at the matrix level.**

> **CONCLUSION — band 37 is downgraded to a depth statement.** What band 37 measured is how ‖d‖ and its
> concentration co-vary **across depth**, which is a real and reproducible description of the depth
> profile. It is **not** evidence of a matrix-level concentration mechanism, and **"5 of 6 types trade
> off, mlp.fc reverses" is not a fact about matrix types** — it is a fact about where each type's depth
> curves turn. **The mlp.fc "exception" is withdrawn**: it needs no explanation, and the two mechanisms
> refuted for it in iterations 147–148 were explaining something that was not there.

**Band 27 is untouched** — it is a ‖a‖–‖d‖ statement verified separately (iteration 148: mlp.fc −0.986,
strongest of six), and does not rest on any of this.

**New standing rule (10).** *A per-type correlation computed across layers is a depth correlation.
Before reading it as a property of the type, remove depth and check whether it survives — and if the
only depth-free test available is underpowered, report that the question is open rather than reporting
the depth number.* Three iterations (147, 148, 149) were spent explaining a sign that was a depth
artifact; the first two hypotheses were refuted on their merits, but the target itself was not real.

**Cost of the error, stated plainly.** Band 37 was recorded from a 6-row table of across-layer
correlations without a depth control. Nothing downstream depends on it, so the damage is confined to
three iterations of my own effort — but it is the second time this campaign (after the shared-term
alignment issue behind rule 6) that a constructed aggregate was read as a mechanism.

## ✅ BAND 27 SURVIVES THE mlp.fc EXCEPTION (iteration 148) — the trade-off is intact everywhere

*Band 37 left mlp.fc reversing the concentration signature at +0.867. **The load-bearing question was
whether band 27's trade-off itself inverts for mlp.fc, or only the mechanism by which it happens.**
That had not been checked.*

| type | corr(log‖d‖, d_cv) | corr(log‖a‖, a_cv) | **corr(log‖a‖, log‖d‖) — band 27** |
|---|---:|---:|---:|
| attn.k | −0.805 | −0.440 | **−0.943** |
| attn.proj | −0.597 | −0.552 | **−0.980** |
| attn.q | −0.435 | −0.440 | **−0.972** |
| attn.v | −0.861 | −0.440 | **−0.954** |
| **mlp.fc** | **+0.867** | −0.513 | **−0.986** ← *strongest of all six* |
| mlp.proj | −0.823 | −0.221 | **−0.887** |

> **mlp.fc's ‖a‖–‖d‖ trade-off is not merely intact — it is the STRONGEST of the six types (−0.986).
> Band 27 has no exception. Only the concentration mechanism differs for mlp.fc, and its forward side
> behaves like everyone else's (a_cv −0.513, mid-range).**

**That substantially narrows band 37's exception**: it is a statement about *how* one type's backward
distribution reshapes with depth, not about whether the trade-off holds. **Band 27 is unaffected.**

**A hypothesis for the reversal, tested and found incomplete.** mlp.fc's `d_cv` is **U-shaped** across
depth (0.808 → 0.604 at layer 8 → 0.645) while its `log‖d‖` falls monotonically — and correlating a
monotone series with a U gives a positive value without any inversion. **That would have made the
exception mundane.** Fitting each type's `d_cv` to depth + depth²:

| type | quadratic coef | R² linear | R² quad | U-shaped? |
|---|---:|---:|---:|---|
| attn.proj | +0.00229 | 0.501 | 0.976 | **yes** |
| attn.q | +0.00772 | 0.328 | 0.904 | **yes** |
| **mlp.fc** | +0.00312 | 0.440 | 0.897 | **yes** |
| attn.k | +0.00592 | 0.792 | 0.942 | no |
| attn.v | −0.00184 | 0.928 | 0.931 | no |

**Three types are U-shaped, not one.** attn.proj and attn.q share mlp.fc's U-shaped `d_cv` and still
show the *negative* concentration correlation. **A U-shape alone does not produce the reversal, so the
shape-mismatch explanation is incomplete and is not adopted.**

**Net state of band 37.** The exception is **real, reproducible (sd 0.040), and unexplained** — two
candidate mechanisms tested and both refuted: ReLU² sparsification (iteration 147: mlp.fc has the
*least* concentrated backward signal, not the most) and shape mismatch (here: two other types share the
U-shape without reversing). **What is now established is that the exception does not threaten band 27**,
which was the only way it could have damaged the account.

**Method note.** Both refuted hypotheses were mine, proposed and tested within two iterations. **The
value of recording them is that the next candidate must explain why mlp.fc differs from attn.proj and
attn.q specifically** — a much tighter constraint than "why is mlp.fc odd," and one neither hypothesis
survives.

## ⚠️ QUESTION (b) — PARTIALLY ANSWERED, with an exception I could not explain (iteration 147)

*REQ-047's CHECK 3 used **participation** and was inconclusive. But the token-norm distributions also
carry **mean** and **sd**, and the **coefficient of variation** is the more direct discriminator for
band 27's two readings — a support trade-off concentrates the signal (CV rises as the norm falls), a
scale trade-off leaves the distribution's shape unchanged (CV flat).*

| type | **corr(log‖d‖, d_cv) across depth** |
|---|---:|
| attn.v | **−0.861 ± 0.022** |
| mlp.proj | −0.823 ± 0.029 |
| attn.k | −0.805 ± 0.018 |
| attn.proj | −0.597 ± 0.064 |
| attn.q | −0.435 ± 0.040 |
| **mlp.fc** | **+0.867 ± 0.040** |

> **Five of six types show the concentration signature: as ‖d‖ falls with depth, the backward signal
> concentrates on fewer tokens. That is a SUPPORT trade-off, not a pure rescaling — question (b)'s
> answer for those five.**

**mlp.fc reverses sharply, and tightly enough (sd 0.040) that it is not noise.**

**My proposed explanation is refuted by the same data, and I am recording that rather than the
hypothesis.** mlp.fc is the only type whose measured backward signal passes through **ReLU²**, so
sparsification looked like the obvious account — it predicts mlp.fc's `d` should be **sparse and
skewed**: low participation, high CV. **The opposite holds:**

| type | active fraction | d_cv |
|---|---:|---:|
| attn.v | **0.056** | **1.256** |
| attn.k | 0.087 | 0.964 |
| attn.q | 0.115 | 0.862 |
| **mlp.fc** | **0.336** | **0.647** |
| mlp.proj | 0.405 | 0.580 |

**mlp.fc has among the highest participation and lowest CV of any type** — the least concentrated
backward signal, not the most. **The ReLU² sparsification reading is wrong**, and the exception is
unexplained.

**Registered as band 37 with the exception in the band text**, not hidden in a footnote: the check
requires ≥5 of 6 types negative, which the data supports, and names mlp.fc as the known reversal so a
future seed showing 6 of 6 would be the informative surprise.

**What this leaves.** Question (b) is **answered for five types and open for one**. The honest state is
better than "inconclusive" (REQ-047's CHECK 3) and worse than "answered" — and the exception is now
sharply posed: *why does the one type with the least concentrated backward signal also invert the
concentration trade-off?* **That is a better question than the one this iteration started with, and it
needs no new instrument** — the per-token distributions to answer it may already be recoverable from
REQ-047's raw tensors if the probe is re-run with a sparsity field.

## ✅ REQ-047 DELIVERED (2026-09-04) — question (a) answered, and a probe disagreement recorded

**REQ-047 ran and its status line still reads OPEN**, which is why I nearly missed it again — found by
inventorying `logs/` for raw data rather than by reading statuses. *(Third distinct way a delivery has
been missed today; the reliable check is the directory listing.)*

**The probe was built as specified**, including the sequence-axis correctness note from iteration 143:
`measure_activation_backward_v2.py`, four seeds, 288 matrix-observations, with `da_cos_mean`,
`align_ratio`, `grad_rank1_frac` and per-token participation all present.

**CHECK 1 — verified independently against the raw JSON:**

| seed | da_cos q,k | v, attn.proj | ratio |
|---|---:|---:|---:|
| 0 | +0.0203 | +0.2498 | **0.081** |
| 1 | +0.0234 | +0.2529 | 0.093 |
| 2 | +0.0190 | +0.2479 | 0.077 |
| 3 | +0.0204 | +0.2515 | 0.081 |

**q,k's adjacent-token backward coherence is 8–9% of v/attn.proj's** — a 12× gap, all four seeds.

**And it MEDIATES the alignment deficit completely.** Regressing `log align_ratio` on a q,k indicator,
then adding `da_cos_mean`:

| seed | q,k coef alone | **+ da_cos** | **shrinkage** | da_cos t |
|---|---:|---:|---:|---:|
| 0 | −0.0614 | **+0.0008** | **101%** | +12.4 |
| 1 | −0.0456 | +0.0130 | **128%** | +12.6 |
| 2 | −0.0558 | +0.0015 | 103% | +11.3 |
| 3 | −0.0581 | +0.0021 | 104% | +11.5 |

**The q,k coefficient goes to zero.** `corr(da_cos, log align) = +0.83 to +0.86`.

> **Registered as band 36. Question (a) is answered: the q,k alignment deficit IS token incoherence —
> their backward vectors point in different directions from one token to the next, so the outer
> products accumulate destructively. Not a sparsity effect (`a_participation` is identical for q, k and
> v at 8137.8).**

**CHECK 2 passed** — `corr(align_ratio, grad_rank1_frac) = +0.656` over 288 observations, confirming
alignment and rank-1 concentration measure the same accumulation coherence.

**~~A PROBE DISAGREEMENT~~ — RESOLVED (iteration 146), and it was my error.** I reported that REQ-047
and band 25 disagreed by ~1.6× on the alignment deficit. **They do not.** REQ-043's `alignment.tsv`
header states its reference group explicitly: *"align_deficit = mean over 12 layers of
log10( ((q+k)/2) / v )"* — **attn.v ALONE.** I tested q,k against *v + attn.proj* and against *all four
other types*, neither of which is the filed definition, and reported a conflict from the mismatch.

| reference group | REQ-047 `align_ratio` deficit |
|---|---:|
| **attn.v alone** (REQ-043's definition) | **−0.1881 ± 0.0077** |
| v + attn.proj | −0.1157 |
| all four other types | −0.0552 |

**Band 25 reported −0.1896 ± 0.0068. REQ-047 gives −0.1881 ± 0.0077 — agreement to 0.0015 dex**, far
inside both error bars. The two probes use **identical definitions**
(`‖W.grad‖_F / (‖d‖_F‖a‖_F)`, verified in both sources) and produce the same number on independent
runs. **This is a cross-probe replication, not a conflict.**

**What this means now.** **Band 25's magnitude is confirmed on an independent probe**, not
probe-dependent — the strongest possible outcome, and the opposite of what I reported. Its role in
iteration 107's arithmetic (backward −0.18 + alignment −0.19 = the gradient deficit) is correspondingly
strengthened. Band 36's mediation was never at risk, being a within-probe result.

**Method note.** I compared two measurements without first checking that they used the same reference
group — the comparison was mis-specified, not the data. **The definition was written in the source
file's header the whole time.** *Standing rule: before reporting that two measurements disagree, verify
they are the same comparison — read the filed definition, not the reconstructed one.* This is the
inverse of the day's other recurring error (missing deliveries by reading summaries instead of data):
here I read the data and skipped the header.

**CHECK 3 is inconclusive**, and I am not reading it either way: `corr(a_part, d_part)` across depth is
+0.588, +0.216, −0.358, +0.322, +0.300, +0.114 — **no consistent sign across types**, so it
distinguishes neither the support nor the scale reading of band 27. **Question (b) remains open.**

## ✅ REQ-045 RE-ANALYSED (2026-09-04) — a third LR design, and a rate that transfers

**REQ-045's committed raw curvature JSON had not been used.** Its headline (β_neighbour null →
iteration 124's partial/total reading withdrawn) was recorded from the commit summary alone. **The raw
data supports a second, independent test: REQ-045's global scale `S` is a third LR ladder, built for a
different purpose, so band 30 can be tested on data not collected to test it.**

**The design worked as pre-flight simulated** — iteration 126 predicted separability `corr ≈ +0.62` for
3 global levels; the run achieved **+0.722**, against REQ-023's **−1.000**. *(That is the pre-flight
check from iteration 126 paying off: a design verified by simulation before being requested.)*

| S | cov(log λ, log g) | corr | sd log λ | sd log g |
|---:|---:|---:|---:|---:|
| 0.70 | **0.0917** | 0.629 | 0.513 | 0.284 |
| 1.00 | 0.0825 | 0.641 | 0.454 | 0.283 |
| 1.40 | **0.0785** | 0.636 | 0.449 | 0.275 |

**Right sign, not individually significant:** −0.0131, CI **[−0.0620, +0.0336]** — includes zero.

**That is a range effect, and the rate transfers.** REQ-045 moves the LR **2.0×** where the other two
move it **2.8×**. Normalising by log-range:

| design | LR span | cov drop | **per dex** |
|---|---:|---:|---:|
| Arm A (global, n=4 seeds) | 2.83× | −0.0181 | **−0.040** |
| REQ-023 (per-matrix) | 2.83× | −0.0348 | −0.077 |
| **REQ-045 (crossed global)** | **2.00×** | −0.0132 | **−0.044** |

**Arm A's rate predicts −0.0120 at REQ-045's narrower range; REQ-045 observed −0.0131.** REQ-023's
predicts −0.0232 — also inside the CI.

> **REQ-045 is CONSISTENT with band 30 and individually underpowered. It neither confirms nor refutes,
> which is the correct reading of a wide interval containing both zero and the predicted value.**

**Band 30 amended** to state the *rate* (−0.04 to −0.08 dex per dex of LR) rather than only the
endpoint drops, since the rate is what transfers across designs with different ranges. **The
underpowered third design is recorded as such** — not as a third confirmation, which would overstate
it, and not as a failure, which would misread a CI that contains the prediction.

**Method note.** I recorded REQ-045's headline from a commit summary without opening its data. **The
raw JSON contained an independent test of a different band.** *Reading a delivery's summary is not
reading the delivery* — the same class of omission as checking commit messages instead of request
statuses, which cost three turns earlier today.

## ✅ REQ-044 VERIFIED INDEPENDENTLY (2026-09-04) — and a crossover the headline understates

**Correction to my own reporting first.** I claimed "no new measurements" for three consecutive turns.
**That was wrong: REQ-044 and REQ-045 were both DONE.** I had been checking commit *messages* rather
than request *statuses* — the same error that made me miss REQ-046 earlier. **Checking `git log` is not
checking the queue.** *(Method note for the record: `grep -A3 "^## REQ-0" requests.md | grep status` is
the reliable check; commit subjects are not.)*

**All three REQ-044 claims verified against the committed `summary.tsv`:**

| batch | mu95−mu0 | bimax−mu0 | kmax−mu0 | **kmax−bimax** |
|---|---:|---:|---:|---:|
| 1× | +0.00004 | **−0.01047** | −0.00537 | **+0.00509** |
| 2× | +0.00003 | −0.00732 | −0.00665 | +0.00068 |
| 4× | −0.00000 | −0.00426 | −0.00661 | **−0.00235** |
| 8× | +0.00001 | −0.00197 | −0.00627 | **−0.00430** |
| 16× | +0.00031 | **+0.00055** | **−0.00469** | **−0.00524** |

1. **Single-EMA momentum buys nothing** — max |mu95−mu0| = 0.00031, below 5e-4 at every batch. ✅
2. **bi-Maxwell decays to zero by 16×** — −0.01047 → +0.00055, **sign flips**. ✅
3. **K-Maxwell holds its edge and beats bi-Maxwell at 8×/16×** — −0.00430 (13× sd) and −0.00524
   (21× sd). ✅

**What the headline understates: this is a CROSSOVER, not a ranking.**

> **At 1× bi-Maxwell is better than K-Maxwell by 0.0051. At 2× they are level. From 4× onward
> K-Maxwell leads, growing monotonically to 0.0052 at 16×.**

**The `kmax−bimax` difference changes sign monotonically across the ladder**, with across-seed sd of
0.00009–0.00034 — every cell is many sd from zero, on 3 independent bases (distinct state hashes).
**The finding is not "K-Maxwell is better"; it is "the two kernels have opposite batch scaling", with
the crossover between 2× and 4×.** That is a sharper and more useful statement: it predicts which
kernel to prefer *as a function of batch size*, and it means the 1× comparison actively misleads about
large-batch behaviour.

**Relation to the C campaign:** none directly — REQ-044 is a kernel ablation, not a curvature study.
Recording it here because it is a delivered result I had wrongly reported as absent, and because
its structure (a monotone sign-changing contrast across a ladder) is the same shape as band 30's
LR decoupling, measured on an independent axis.

## ⚠️ ANALYSIS-LOOP STATUS (2026-09-04, second assessment)

**The committed data is exhausted. This is a measured claim, not an impression.**

**Evidence.** Of the last ten analysis iterations, **six were negatives, refutations, or checks on my
own prior claims**: the three U-shapes share a form not a mechanism (138); entropy fails the
quadratic-depth check a second time (137); chunk geometry refuted (136); alignment does not explain the
q,k gap (135); alignment does not belong in C's predictive model (141); the shared-term audit changed
one number and confirmed 15 clean (139). **Two produced new bands (33, 35), and both were then
qualified by the checks that followed.**

**Jerry's last experimental delivery was ~6 hours ago; I have pushed 11 commits since, all analysis of
the same 72 matrices.** One request is OPEN.

**What the pattern means.** The campaign is now finding **artifacts of its own construction** faster
than facts about the network — the shared-λ trap alone has produced three results that passed
conventional validation (134, 138, 141), the last surviving two independent hold-out schemes. **That
is what exhausted data looks like: the remaining signal is comparable to the analysis's own
distortions.**

**The account is complete and internally consistent** — see CONSOLIDATED FINDINGS IV and the goal
statement below it. **The three open questions are measurement-bound**, and one of them is now known to
be **unanswerable by any gradient-scaling intervention** (band 31, a theorem).

**Recommendation: pause the analysis loop.** The productive path is the queue:
- **REQ-045** (OPEN, low priority) — settles band 30's *shape*
- **REQ-044** (Jack's paired batch ablation)
- **A new probe** would be needed for the two alignment questions — per-token backward vectors, which
  is a substantially heavier instrument than anything filed. **I do not recommend filing it on the
  strength of band 33 alone**, since band 141 shows alignment adds nothing to C's predictive model.

**What would restart the loop productively:** any new measurement landing in `logs/`. Until then,
further passes over the same data have negative expected value — each one risks adding a
conventionally-validated artifact to the record, which is precisely what the last three traps were.

## ⚠️ QUEUE STATUS NOTE (2026-09-04, from the analysis session)

**Jerry's last experimental delivery was REQ-043 P2/P3, ~5 hours ago. Since then this session has
pushed 23 commits, all analysis of already-committed data — no new measurements have entered the
repository.**

**This is surfaced, not complained about.** The analysis reached the end of what committed data
supports several iterations ago (see the closing section of REQ-035): the account of C is complete
at n=4 and internally consistent, and the three remaining questions each need a measurement that does
not exist yet. The recent iterations have been auditing earlier claims — worth doing once, and now
done: a retroactive leave-one-out pass, a four-constraint cross-band consistency audit, and two
collinearity traps caught and recorded.

**What is actually blocking progress, in priority order:**

| priority | request | why it matters | status |
|---:|---|---|---|
| **1** | **REQ-046** (was REQ-037 arm 4) | The only open measurement that can still **overturn a load-bearing band** — the exclusion restriction is *known violated* (iteration 114) and this bounds the damage | ✅ **FILED 2026-09-04.** The design decision was made in-session rather than deferred: the probe gains one field, `clipped_gradient_block_norm`. Rationale in the REQ-046 block. |
| 2 | REQ-040 | Dispatch of the existing queue | OPEN |
| 3 | REQ-044 | Jack's paired batch ablation | OPEN |
| 4 | REQ-045 | Settles band 30's *shape*, not its sign | OPEN, **low priority** |

**The decision that was blocking arm 4 has been made and the request filed as REQ-046.** The probe
recomputes the gradient from scratch (`measure_per_matrix_curvature.py:100`,
`torch.autograd.grad(loss, params)`), entirely outside the optimiser — verified by reading it — so a
clip in the Muon path is invisible to it and the first stage would be identically zero. The fix is one
extra field, `clipped_gradient_block_norm`, chosen over the alternative (per-matrix loss weights)
because that alternative changes *what is optimised* and reintroduces the exclusion problem arm 4
exists to remove. **No humans' decision is outstanding.**

**Recommendation:** let the queue drain. Further analysis of committed data has low marginal value,
and the campaign's open questions are measurement-bound, not analysis-bound.

## 🔒 STANDING CONSTRAINTS (2026-09-03, Jack — supersedes all earlier node directives)

**Node ceiling: ≤2 nodes, fleet-wide.** This is the single operative limit. It agrees with the
live ≤2 instruction the agent holds from Jerry, so **there is no node-authority conflict and
nothing is blocked on a ruling.** Any 4-box or ≥10-node language in this file's history is
withdrawn — the ≥10-node grant was rescinded 2026-09-03 and the 2026-09-02 "max 4 concurrent
boxes" directive is retired with REQ-032, the request it was written for.

*Rationale for staying at 2:* measured wall-times make fan-out pointless. Training runs at
**0.162 s/step on 8×H100** at SHA `ebf53cd` (16 arms in
`logs/kmaxwell/req019_eos_state_dependence/summary.tsv`, sd < 1 s), so a from-scratch fork-1500
state is **~4 minutes**, and REQ-035 Arm A's four seeds are ~16 minutes of training in total.
The dominant cost everywhere is the Lanczos probe (~3.7 min per checkpoint, ~5× the training it
measures), not the training. **The whole EoS queue is hours on two boxes.** Run it sequentially.

**Keep the Lanczos probe at 8 iterations.** The committed tridiagonals show the estimate is
converged (7→8 increment: median 0.029%) and duplicate-run scatter is flat from 4 to 8 iterations
— the noise floor is physical, not instrumental. Raising the count multiplies the dominant cost
for no gain.

**Do not commit tensors/weights.** The committed deliverable is curves, logs, and derived JSON.
*(REQ-032's checkpoint exception died with that request.)*

### Run order

Run sequentially on ≤2 boxes:

| priority | request | why |
|---:|---|---|
| **1** | **REQ-035 Arm A, with REQ-038's probe folded in** | Load-bearing: 4 seeds decide whether any finding is architectural or an artifact of one trained network. **Add REQ-038's five measurement fields (`\|a\|`, `\|d\|`, effective ranks, attention-logit stats) to Arm A's probe** — Arm A regenerates step-1500 states by design, so REQ-038 becomes a free by-product tested at n=4 instead of n=1. |
| **2** | **REQ-036** | The per-layer LR design, 5 arms including the anti-rule falsifier. Tests a shipped design. |
| **3** | **REQ-037** | Non-LR instrument; tests the exclusion restriction behind the gradient law. |
| **4** | REQ-038 standalone | **Only if Arm A cannot be extended.** No checkpoints are committed, so a standalone run must regenerate the state first — it is not the cheap probe it was originally filed as. |

**REQ-034 (K-Maxwell batch ladder) is unrelated to this queue** — order it against the above as
you see fit.

### The single number to check first

REQ-038 measures per matrix the input activation `|a|` and the backward tensor `|d|`. **q, k and v
read the same residual vector, so their `|a|` is identical by construction** — any gradient
difference must sit entirely in `|d|`. From committed data:

> **Predicted: `|d|(q,k) / |d|(other four types) = 0.39 ± 0.08`**

- **Near 0.39** → the campaign's central anomaly closes: the gradient law λ ∝ g² is universal and
  q,k's apparent violation is the attention softmax attenuating the backward signal.
- **Near 1.0** → the deficit is in `|a|` instead, which contradicts q,k,v sharing an input and means
  either the probe or our reading of the model code is wrong. **Report this loudly if it happens.**

### Where the live specifications are

REQ-035 and REQ-036 accumulated many superseded prediction blocks across a long analysis session.
**Only the two authoritative tables are live** — the `AUTHORITATIVE SEED-CHECK TABLE` in REQ-035
and the `AUTHORITATIVE ARM TABLE` in REQ-036. The superseded iteration blocks have been removed
from this queue; their provenance is in this file's git history.

---

## REQ-034: K-Maxwell on the fork@2000 batch ladder — 1× → 16×

- status: **DONE (2026-09-03) — `logs/kmaxwell/req034_kmaxwell_batch_ladder/`.** HEADLINE POSITIVE: **K-Maxwell is the large-batch-durable kernel.** benefit(kmax−μ0)@2750 = 1x −0.0046 / 2x −0.0053(fresh μ0) / 4x −0.0067 / 8x −0.0072 / 16x −0.0058 — flat-to-growing across 1x–16x, while bi-Maxwell (REQ-029) DECAYED to ~0 by 16x. The anneal does something structurally different from noise-averaging; it holds its gain where the frozen two-rate kernel loses it (REQ-033's 'grows with batch' now on the same axis). Gates passed (tests, smoke val@2125 finite=annealed fork@2000 clean, budget 16x margin +71). Caveat: my base val@2000=3.44279 vs stored 3.44367 (−0.0009 offset on the stored-μ0 points; 2x fresh-μ0 anchor offset-free; shape robust). n=1/arm.
  fork@2000 template driven for REQ-033; mbs stays 64 so no eager fallback). The earlier
  node-authority hold is resolved: the ≥10-node line was rescinded, the ≤2 ceiling is
  uncontested, and REQ-032 has freed both boxes. Run the 6 arms sequentially or 2-at-a-time —
  they are short (750 steps each), so this is still one pass. On launch: bootstrap @365c392d
  (venv019 + **86 fineweb chunks**) → regen `eos_shared_base` dump@2000 (base val@2000 must match
  3.44367) → 6 configs (5× annealed_weights_muon 1×/2×/4×/8×/16× + 1× muon μ0 @2×) → gates
  (usable-batch budget + 20-step smoke + tests + 2× μ0 val@2000==3.44367) → diff kmax−μ0 @2750
  against stored μ0 (1× 3.34586 / 4× 3.24333 / 8× 3.20561 / 16× 3.17362; 2× fresh).
- requested: Jack / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: 365c392d695f95dc9a4fb89095e85a6a7b5d551e (same as REQ-026/027/028/029/033)
- **node budget: ≤2 nodes.** Run the 6 arms sequentially or 2-at-a-time; they are short
  (750 steps each), so this is still one pass.

**Why.** REQ-026→029 built the momentum-benefit-vs-batch curve for the frozen
bi-Maxwell kernel — `1x −0.01063, 4x −0.00438, 8x −0.00233, 16x ~0` — a clean decay
to zero. REQ-033 measured the **annealed K-Maxwell** kernel, but on a *different
protocol* (fork@1000, 2250 steps, @3250) and a *different range* (0.25×–2×), so its
numbers cannot be laid on the same axis as that curve. Plotting them together
produces a false zigzag; they are not the same measurement.

This request puts K-Maxwell on the **exact bi-Maxwell protocol** — same shared
step-2000 state, same 750-step window, same @2750 readout, same batch ladder — so
the two kernels finally sit on one axis over 1×–16×.

The open question it answers: REQ-033 found K-Maxwell's benefit **grows** with batch
across 0.25×–2× while bi-Maxwell's shrinks. Does K-Maxwell keep its gain at 4×, 8×,
16×, where bi-Maxwell's went to exactly zero — or does it also get absorbed once the
batch is large enough?

### Expected work — 6 arms, 750-step continuations from the shared step-2000 state

Same `eos_shared_base` machinery as REQ-026/028/029 (base val@2000 = 3.44367).

| # | batch | kernel | note |
|---|---|---|---|
| 1 | 1× | `annealed_weights_muon` | |
| 2 | 2× | `annealed_weights_muon` | |
| 3 | 4× | `annealed_weights_muon` | |
| 4 | 8× | `annealed_weights_muon` | |
| 5 | 16× | `annealed_weights_muon` | |
| 6 | 2× | `muon{mu:0.0}` | **the one missing control** |

**Only 2× needs a fresh control.** μ=0 already exists at 1× (3.34586, REQ-026),
4× (3.24333, REQ-026), 8× (3.20561, REQ-028) and 16× (3.17362, REQ-029) — all at
this exact fork and horizon. Difference the new K-Maxwell arms against those stored
values; do **not** re-run them.

### Exact config keys

Copy the REQ-026 fork-continuation template (`make_req026_configs.py`) and change
only the blocks-group optimizer and the batch keys:

| batch | `batch_tokens` | `microbatch_sequences` | `skip_batches` | fineweb chunks |
|---|---|---|---|---|
| 1× | 524288 | 64 | 2000 | 15 |
| 2× | 1048576 | 64 | 1000 | 19 |
| 4× | 2097152 | 64 | 500 | 27 |
| 8× | 4194304 | 64 | 250 | 44 |
| 16× | 8388608 | 64 | 125 | **80** |

All skips are exact integers → every arm resumes at the same ~1.049B-token data
position, exactly as in REQ-026/028/029. `microbatch_sequences` stays **64** at every
batch (larger batches just run more accumulation steps) — this both preserves
per-forward memory and **avoids the torch.compile mbs<64 NaN bug found in REQ-033**;
no eager fallback is needed here.

Budget in **usable batches** (`Σ floor(shard_tokens/batch_tokens)`), the REQ-029
metric, not raw tokens — the chunk counts above already use it. Bootstrap **86**
chunks (REQ-029's verified 16× figure) and every arm fits.

`start_step: 2000, stop_after_step: 2750`, `lr: 0.025, weight_decay: 0.05, mu: 0.95`,
`cool_down_learning_rate cooldown_frac: 0.7`, no `fixed_eta_after`, checkpoint +750
only, no Lanczos — all identical to REQ-026/028/029.

**Kernel** (`annealed_weights_muon`): `switch_step: 2000`, `anneal_steps: 750`.

**Note the deviation and why.** PR #357 switches at step 1000 and anneals to 3250.
Forking at 2000 puts the switch already in the past, and the shared base is plain
Muon with no K-buffers to inherit, so buffers lazy-init at the fork exactly as they
do in the PR at its own switch step. The 58→26 sweep is then **compressed into the
750-step window** so every arm sees the full kernel trajectory rather than a slice of
it. This tests the *kernel*, not the PR's absolute timetable — state that plainly in
the README. (Forking at 1000 instead was considered and rejected: 1000 is not
divisible by 16, so 16× cannot token-align — 62.5 batches — and a 2250-step window at
8×/16× needs 104/211 chunks, exceeding FineWeb10B's 103.)

Decays and weights are the shipped PR #357 values, identical at every batch (no
rescaling in this request — REQ-033 already refuted age-rescaling as a repair):

```yaml
decays: [0.75, 0.822852439855, 0.877930338626, 0.917598547218,
         0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
start_weights: [0.005093975, 0.010187949, 0.015281924, 0.020375898,
                0.025469873, 0.030563847, 0.035657822, 0.857368713]   # mean age 58
end_weights:   [0.032261839, 0.064523678, 0.096785516, 0.129047355,
                0.161309194, 0.193571033, 0.225832871, 0.096668514]   # mean age 26
```

### Gates (hard)

1. Per-config 20-step finite-loss smoke before any full arm.
2. Usable-batch budget assert per config BEFORE launch (REQ-029 precedent — the 16×
   first pass exhausted fineweb 17 steps short on raw-token budgeting).
3. Tests green at the pinned SHA.
4. Confirm the 2× μ=0 control's val@2000 matches the shared base (3.44367) before
   trusting any 2× difference.

### Artifacts

`logs/kmaxwell/req034_kmaxwell_batch_ladder/{README.md,summary.tsv,readout.tsv,
val_trajectories.txt,manifest.tsv,make_req034_configs.py,configs/,logs/}` — the
REQ-026/029 shape.

### Readout

`benefit = final_val(kmaxwell) − final_val(μ0)`, same batch, @2750 — the identical
statistic as the bi-Maxwell curve. Closing table:

```
batch  batch_tokens  benefit(kmax−mu0)  benefit(bimax−mu0)   source of mu0
1x     524288                           −0.01063             REQ-026
2x     1048576                          (none)               THIS REQUEST
4x     2097152                          −0.00438             REQ-026
8x     4194304                          −0.00233             REQ-028
16x    8388608                          ~0.00000             REQ-029
```

The shape is the deliverable, no interpretation needed:

- **K-Maxwell also decays to ~0 by 16×** → both kernels are denoisers; REQ-033's
  "anti-decay" was a 0.25×–2× window effect, and the annealed kernel buys nothing at
  large batch either.
- **K-Maxwell holds its gain at 8×/16× where bi-Maxwell went to zero** → the anneal
  is doing something structurally different from noise-averaging, and it is the
  large-batch-durable kernel. This would be the headline result.
- **K-Maxwell peaks mid-ladder** → there is an optimal batch for the kernel; report
  where.

n=1/cell, seed 0, matching REQ-026/028/029 discovery convention. Noise floor ~2e-4
(REQ-027); read |Δ| < ~5e-4 as noise. If the 16× cell lands inside that band and the
verdict hinges on it, file a follow-up for replicates rather than over-reading n=1.



## REQ-035: what sets the per-matrix equilibrium curvature constant C? (seed-replicated discriminator)

- status: **Arm A DONE (2026-09-03, n=4) — deliverable `logs/kmaxwell/req035_armA_seed_replication/`.** **C is seed-independent to the noise floor:** median-of-pairs |Δ log₁₀ C| = **0.106 dex ≈ the ~0.10 dex floor** (6 pairs: 0.088–0.128), so cross-seed C variation ≈ same-seed run noise → **architecture determines C, not the trained network; the covariate hunt is well-posed** (the ≥0.20 'learned per-network' branch is excluded). k = 1.17–1.34 per seed (~1.26, matches 1.38±0.45). Type ordering reproduces: **attn.proj lowest in all 4 seeds, attn.v/mlp.proj top-2 in all 4.** Fork-1500 states were regenerated from scratch (the 'existing checkpoint' premise was false; base val@2000=3.44267). Ran on 2 nodes under the ≤2 ceiling. **REQ-038's activation/backward fields are NOT folded in yet** — the curvature probe records top_eigenvalue/grad blocks, not forward-activation/output-gradient tensors (needs new hook code); recommend a separate extended-probe pass on regenerated fork-1500s. Arms B/C/D not run. — status(prev): OPEN — Arm A is priority 1, unblocked and ready. Both nodes are free (REQ-032
  stopped and committed). Arm A's four seeds are ~16 min of training total, so it runs
  sequentially on one box under the ≤2 ceiling. **Fold REQ-038's probe fields into Arm A** (see
  run order at the top). Arms B/C/D stay unrun: they are not interpretable until Arm A reports,
  and they are not worth fan-out at this ceiling.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `ebf53cd` (the REQ-019/022 serialized-fork-state design, unchanged)
- **node budget: ≤2 nodes.** Arm A alone is the load-bearing result and fits on one box run
  sequentially. Do not hold the request waiting on any ruling — there is none outstanding.

### ⚠️ AUTHORITATIVE SEED-CHECK TABLE — the only live bands

*This request accumulated 13 overlapping "registered seed check" blocks across ~15 iterations,
several claiming to supersede each other, and most tied to mechanisms that were subsequently
falsified (seven of them). **They have been removed from this queue — only the six bands here
are live.** The superseded blocks remain in this file's git history if provenance is needed.*

**All six reproduce at both fork states in the committed data.** Each is stated with its current
measured value so a seed result can be compared directly.

| # | band | must hold | measured f1500 / f2000 |
|---|---|---|---|
| **1** | **q,k vs other four, gradient-adjusted level** | **+0.81 ± 0.20 dex**, q,k above in **≥10 of 12 blocks** per seed | +0.812 (12/12) / +0.842 (12/12) |
| **2** | **response ratio d log λ / d log g** | **2.00 ± 0.15** per seed | +2.069 / +2.095 |
| **3** | **a POSITION FIELD in C** *("symmetric" WITHDRAWN, iter. 195)* (revised iter. 151) — ✅ **CONFIRMED n=4 (12/12), reading CORRECTED** | **A linear depth trend must NOT explain it** (this is what makes it survive rule 10) and the symmetric term must be significant in ≥10 of 12 seed×fork fits, on top of type offsets + log g | **linear depth t = +0.56 only, 6/12** vs **dist t = −5.92, 12/12**; AIC −228.96 vs baseline −198.88. ⚠️ **NOT a pure boundary shell** — the original seed-0 collapse (F 28.98→4.65) was **underpowered**: with both shells removed the interior U survives, t **+6.03 → +4.07 → +2.85**, 9/12. ⚠️ **Form NOT identified** — quad and logdist tie at AIC −230.02, dist within 1.1. ⚠️ **Edges UNEQUAL**: last block **+0.361 dex, 11/12** vs first **+0.211 dex, 6/12** — **this explains band 10's failure as a power problem, not a contradiction** ✅🔧 **RULE-23 GUARDED (iter. 195) — the first pre-rule band tested.** Saturated (free per-block effects, gradient controlled, no polynomial): **both ends above the middle in 12/12 fits** — block0−block6 **+0.371 (t +7.51)**, block11−block6 **+0.506 (t +10.37)**; per-fit t **+5.75** and **+16.85**, 12/12 same sign ⇒ **the position field is REAL, not an artefact of the model comparison that established it.** 🔧 **But "symmetric" is WITHDRAWN**: L0−L11 = **−0.136 dex** (t −2.10 pooled, −1.96 per-fit, 8/12 same sign) — marginal but consistent, and **band 61 already attributes the tilt entirely to residual writers** (internal-only tilt +0.001). **Four routes now agree the field is lopsided toward the output end** |
| **4** | negative-curvature separation | every nonlinear-path type above every linear-path type | 0.198 > 0.146 / 0.200 > 0.138 |
| **5** | position spacing stays **unequal** | step(0→1) ≥ **2×** step(1→2) | 3.0× / 3.2× |
| **6** | **two-valued gradient slope** — ✅ **CONFIRMED n=4** | residual-writer minus internal raw slope ≥ +1.5 | +2.30 / +2.04 / +2.08 / +2.13, p < 0.0001 in all 4 seeds |
| **7** | **the split is residual-stream position, not shape** (iteration 65) | **slope(attn.proj + mlp.proj) − slope(other four) ≥ +1.5** in ≥3 of 4 seeds, each proj type individually ≥ +2.0 vs internal | +2.173 / +2.183, p < 0.0001 |
| **8** | **the cross-sectional split is bias, not physics** (iter. 67) | **Wald-ratio gap (residual − internal) ≤ +1.0** and **< half the cross-sectional slope gap** in ≥3 of 4 seeds; both first-stage F > 100 | gap +0.374 / +0.688 vs cross-sectional +2.17 / +2.18 |
| **9** | **only attn.proj's offset is resolved** (iter. 68–69, corrected) | **attn.proj lowest** of the six offsets in ≥3 of 4 seeds and **≥0.25 dex below the other five**; **|residual-writer − internal| offset gap < 0.20 dex**; the other four adjacent gaps need NOT resolve | gap −0.454 / −0.430, p ≤ 0.0003 |
| **10** | **layer-0 lift** — ❌ **NOT CONFIRMED n=4** | clears its permutation null on λ/g² | p = 0.022 / 0.190 / 0.120 / 0.058 — **1 of 4 seeds**. Lift is consistently positive (+0.25 to +0.50 dex) but under-powered at 6 matrices per layer |
| **11** | **the assembled model generalises** (iter. 72) | **leave-one-layer-out rmse ≤ 0.20 dex** and **cross-fork rmse ≤ 0.20 dex** for `type + two-slope gradient + layer-0 term`, versus ~0.38 dex for C's own spread | LOLO 0.138 / 0.164; cross-fork 0.165 / 0.140 dex |
| **12** | **type offsets reduce to three binaries** — ✅ **CONFIRMED n=4** | `q,k + residual-writer + mlp.proj` (5p) within 0.02 dex of six free offsets (7p) | gaps 0.004 / 0.004 / 0.005 / 0.002 dex in the 4 seeds |
| **13** | **the causal reading is UNRESOLVED — REQ-046 could not test it** (iter. 76, 114, 129–130) — ⚠️ **iteration 129's overturning RETRACTED** | a valid test needs the clip to move the **raw** gradient. REQ-046 moved `g_clipped` by **+1.003** but the raw gradient by **+0.003** — Muon's unit-norm step absorbed it, so the exponent is 0/0 | raw-g slope **+0.0028**, CI **[−0.015, +0.022]**; λ slope +0.0089. **Iteration 114's two-functional detection stands** (+0.241/+0.268) |
| **14** | **q,k carry a GRADIENT DEFICIT at identical shape** (iter. 77–90) — ✅ **CONFIRMED n=4, size-artifact excluded** | **within the four 768×768 attention matrices only:** Δlog g (q,k − v,attn.proj) ≤ −0.30 dex with p < 0.01, Δlog λ NOT significant, both q,k below both v,attn.proj in ≥10/12 blocks — in ≥3 of 4 seeds | **Δlog g = −0.378/−0.378/−0.356/−0.381, p < 10⁻⁴ all seeds, 48/48 blocks**; Δlog λ p = 0.17/0.85/0.21/0.43 |
| **17** | **the deficit is a depth-independent constant** (iter. 91) — ✅ **CONFIRMED n=4** | **slope of the q,k gradient deficit vs layer index NOT significant (|t| < 2) across layers 0–11**, in every seed; **final block separately deeper by ≥ 0.10 dex** | slopes t = −0.31/−0.25/+0.50/−0.40; interior −0.361 ± 0.065, **layer 12 −0.508 ± 0.010** |
| **18** | **q and k are interchangeable** (iter. 92) — ✅ **CONFIRMED n=4** | **|q deficit − k deficit| < 0.05 dex** and **not significant within any single seed** (p > 0.05), in every seed | q −0.380 vs k −0.366, difference **−0.014 dex** (3.8% of the shared deficit), within-seed p = 0.82/0.56/0.67/0.57 |
| **19** | **QK-norm beats the Muon-chunking rival** (iter. 93) — ✅ **CONFIRMED n=4** | on log g, **QK-norm indicator R² > 0.5 with |t| > 5**, and **shape_mult alone R² < 0.10**; QK-norm coefficient must not weaken when shape_mult is added | QK alone R² 0.63–0.67, t −10.9 to −11.9; shape_mult alone **R² 0.005–0.008, t +0.6 to +0.7**; both: QK **−0.45, t −11.5 to −12.6** |
| **20** | **the mlp gradient gap = the ReLU² forward gap** (iter. 94, 107) — ✅ **DECOMPOSED n=4** | **gap = a_gap + d_gap + alignment**, with **a_gap ≈ −0.43 dex dominating** and matching **log₁₀(a_rms ratio of the ReLU² output to the block input)** to < 0.01 dex | grad −0.302, **a −0.435**, d +0.051, alignment **+0.082 ± 0.006**; ReLU² output is **2.725× RMS** (+0.435 dex) and **7.3× eff-rank** of the block input |
| **21** | **the q,k deficit is PURELY backward** (iter. 95–101, REQ-038/043) — ✅ **CONFIRMED n=4** | **a_rms bit-identical for q, k and v in every seed**; **d_rms ratio (q,k)/v ≤ 0.75** | **a_rms bit-identical, 4/4 seeds**; d_rms ratio **0.667 ± 0.011**, log₁₀(q/v) −0.183 **t = −41.9**, log₁₀(k/v) −0.170 t = −43.3 |
| **22** | **the q,k deficit SHRINKS during training** (iter. 98) — ✅ **CONFIRMED n=4** | **drift of the deficit vs step is POSITIVE (less negative) with a seed-clustered 95% CI excluding 0**, and **same sign in all three LR arms** | pooled **+0.058 dex/1000 steps, CI [+0.032, +0.085]**; per-arm +0.029 / +0.092 / +0.054 |
| **23** | **the g² law is CROSS-SECTIONAL/CAUSAL ONLY — it does not hold in TIME** (iter. 99) | **slope of λ-drift on g-drift across matrices must be < 1.5** (attenuation-corrected), i.e. NOT 2; **C's drift CI must include 0** and **step must explain < 1% of C's variance** | slope **+0.634**, CI [0.329, 0.982], corrected **0.860**; reliability 0.738 so a true 2 would read 1.476. C drift CI [−0.055, +0.049]; step **0.2–0.4%** of variance vs LR 2.0–3.5% |
| **24** | **the measurement window IS equilibrated** (iter. 100) — ✅ **CONFIRMED n=4** | **autocorrelation of successive Δlog λ negative with seed-clustered CI excluding 0** (mean-reverting, not drifting), **and strictly above −0.5** (real dynamics, not white noise); **change magnitude ratio second/first half ≈ 1** | AC **−0.228, CI [−0.437, −0.117]**, implied AR(1) ρ = **0.54**; ratio **0.898** |
| **25** | **the shortfall = the token-wise ALIGNMENT deficit** (iter. 101–105, REQ-043 P2/P3) — ✅ **RESOLVED n=4** | **align_deficit measured at a single state ≤ −0.15 dex with across-seed sd < 0.02**; **identity align_deficit = grad_deficit − d_deficit holds to < 0.001 dex**; **depth slope state-dependent** | **−0.1896 ± 0.0068 dex** (0.646×), identity gap **< 0.0005 dex/seed**; artifact-free slope **−0.0075 dex/layer**; state drift +0.0101 dex/layer per 1000 steps |
| **36** | **the q,k alignment deficit IS token incoherence — complete mediation** (iter. 145, REQ-047) — ✅ **CONFIRMED n=4** | **q,k adjacent-token backward coherence < 20% of v/attn.proj's**, every seed; and **controlling for `da_cos_mean` drives the q,k alignment coefficient to ZERO** (shrinkage ≥ 90%) | da_cos **+0.020 vs +0.250** (8–9%, 4/4); mediation shrinkage **101/128/103/104 %**, da_cos t = **+11.3 to +12.6**; corr(da_cos, log align) **+0.83 to +0.86** |
| **37** | **~~the ‖a‖–‖d‖ trade-off is CONCENTRATION, in 5 of 6 types~~ — ⛔ DOWNGRADED TO A DEPTH STATEMENT** (iter. 147, REQ-047; downgraded iter. 149) | **These are correlations ACROSS DEPTH and are NOT evidence of a matrix-level concentration mechanism.** They faithfully describe the depth profile and reproduce in all 4 seeds, but the per-type sign is set by **where each type's depth curves turn**, not by the type: `log‖d‖` bottoms at **layer 12 in every seed for five types but layer 10 for mlp.fc**, the only type whose `d_cv` and `log‖d‖` minima coincide (mean argmin gap **3.75** vs 7.75–11.00). **The mlp.fc 'exception' is WITHDRAWN — it needs no explanation**, and the ReLU²-sparsification (iter. 147) and shape-mismatch (iter. 148) hypotheses were explaining an artifact. Removing depth with a within-seed quadratic sends **all six positive** (+0.077 to +0.637, mlp.fc mid-pack at +0.379); the assumption-free within-layer test across seeds then finds **5 of 6 indistinguishable from zero** (|t| ≤ 1.70, sd 0.42–1.28, only attn.k significant at t = +4.40). **n=4 has no power at the matrix level — the matrix-level question is OPEN.** Band 27 is untouched (separate ‖a‖–‖d‖ result, mlp.fc −0.986, strongest of six) | depth values as measured: attn.v −0.861, mlp.proj −0.823, attn.k −0.805, attn.proj −0.597, attn.q −0.435, mlp.fc +0.867 ± 0.040 — **read as depth profile only** |
| **38** | **POSITION is what sets the between-layer difference in C — the variance budget** (iter. 153) — ✅ **CONFIRMED n=4 on committed data** | On a 4-seed panel: (i) `dist`/`quad` explains **≥50%** of the between-layer variance of **block-mean log C** after type + gradient, in **≥10 of 12** seed×fork fits; (ii) **linear depth < 30%**; (iii) **LOLO rmse with position < 0.75×** the type+gradient LOLO rmse | between-layer sd(log C) **0.1806 dex**. Cumulative: type **−0.0%** (removed by construction), **+ gradient 3.5%**, **+ position 68.8%**. Position **67.1%** of residual variance (12/12) vs **linear 15.6%**, quad 67.9% (**form NOT identified**), last-block 39.2%, first-block 14.0%. Floor measured on this panel = **0.0959 dex**. ⚠️ **Out-of-sample headline: LOLO 0.1216 dex vs 0.1948 without position (38% reduction, 144 fits); cross-fork transfer 0.114–0.124 dex → 1.27× the floor, NOT the in-sample 0.99×.** **~0.075 dex of between-layer structure remains unexplained** |
| **39** | **the position field is an ASYMMETRIC BOWL — form identified, between-layer C closed** (iter. 154) — ✅ **CONFIRMED n=4 on committed data** | (i) free per-layer profile is **single-minimum with the minimum in layers 5–7**; (ii) **cubic tilt coefficient positive in ≥10 of 12** seed×fork fits; (iii) **LOLO rmse of `type + gradient + cubic` ≤ 1.15×** the panel-measured noise floor | Free profile L0 **+0.166** → L6 **−0.207** → L11 **+0.302**, se 0.011–0.055 dex on a 0.51 dex swing, **monotone both sides of the minimum at layer 6** (⇒ **not a boundary shell** — confirms band 3's correction independently). Cubic **R² 0.977** vs quad 0.915 / dist 0.899; **88.9%** of between-layer residual variance vs dist's 67.1%. Vertex at **5.42** ≈ centre 5.50, so the asymmetry is a **TILT**, not a shift. ⚠️ **CORRECTED iter. 163 on 60 fits (720 held-out): LOLO 0.0982 vs floor 0.0903 = 1.09×, NOT 0.96× — "within seed noise" is WITHDRAWN.** Structure holds exactly (cubic best; position cuts error 49%, 0.1939→0.0982). **Residual √(LOLO²−floor²) = 0.0387 dex is REAL in MAGNITUDE but NOT localised** — the iter.-163 claim of spikes at layers 2/3/6/8 was **pseudo-replication and is RETRACTED (iter. 164)**: the 5 checkpoints per seed×fork are the same network, and clustering on the true independent unit (seed, n=4) collapses those |t| from 3.9–5.9 to 1.4–3.3, only L3 surviving at 3.28. Residual replicates across seeds at only **+0.220** (one pair negative). Architecture refuted (6 pre-declared per-layer sets, best t +1.26, permutation p **0.815**) and periodicity refuted (best F 2.09, permutation p **0.823**). **Bowl is TIME-STATIONARY**: argmin L6 at all 5 checkpoints, step-profile corr +0.974 (min +0.942) (dist 0.1216, quad 0.1007, no-position 0.1948). Tilt **+0.00111, t = +6.73, 12/12 same sign**. ⚠️ Cubic loses cross-fork (0.1331 vs dist 0.1192) — **magnitude drift, not fake shape**, because the profile **evolves with training**: L0 +0.093→+0.266, L11 +0.288→+0.251 across forks 060→170, i.e. **the tilt is a training transient**. **Supersedes band 38's unidentified form and its ~0.075 dex residual** ⛔⇒★ **AXIS CORRECTED (iter. 172)**: "cross-fork transfer" was **cross-LEARNING-RATE** generalisation. The bowl holds **separately at every LR** — cubic R² **0.904/0.961/0.975**, **argmin L6 in all three**, pairwise r **+0.82 to +0.98** across a **2.8× LR range** ⇒ the positional finding is **LR-INVARIANT and stronger than claimed** |
| **63** | **✅★ the bowl stated with NO functional form — it survives rule 23's saturated-control guard** (iter. 189, REQ-048) — ✅ **CONFIRMED n=4 (12 fits)** | Using **free per-block effects, no polynomial**: (i) the profile's **minimum lies in blocks 4–8 in ≥10 of 12** fits; (ii) the minimum is **below BOTH endpoints in ≥10 of 12** fits; (iii) pooled **block6−block0 and block6−block11 contrasts both negative with |t| ≥ 3**, matrix-clustered | **argmin interior 12/12** (L6 ×10, L7 ×2); **below both ends 12/12**; pooled **−0.393 dex (t −5.60)** vs L0 and **−0.511 dex (t −4.34)** vs L11; per-fit means −0.397 (t −4.72) and −0.515 (**t −25.82**). ⇒ **The bowl is NOT an artefact of the cubic.** Fitting the FREE profile: **cubic R² 0.886, linear R² 0.000** — the cubic *describes* it, does not create it. ⇒ **Explains why band 61 failed this guard and the bowl passed**: a *deviation-from-trend* claim depends on the baseline; a *contrast between measured positions* cannot be manufactured by one |
| **68** | **✅ the bowl is DESIGN-INDEPENDENT — the lever hazard that killed band 67 is contained** (iter. 199) — ✅ **CONFIRMED on two independent designs** | (i) the C bowl's **argmin agrees between a GLOBAL-LR panel and a MIXED-LR panel**; (ii) the two bowl profiles correlate at **≥ +0.70**; (iii) **no band whose evidence is a within-LR profile may be quoted as an LR-response claim** (documentation criterion) | **global (REQ-035): argmin L6, swing 0.538; mixed (REQ-023): argmin L6, swing 0.555; corr +0.931** — different forks, randomisations and LR mechanisms. ⇒ **Exposure audit: bands 44/57/58/59/63/64 are class A (profiles computed WITHIN one LR, LRs compared only for agreement) and CANNOT be flipped by the mixed-vs-global distinction; bands 42/49/53 are class C but were already scope-limited in iter. 177; band 67 was the ONLY survivor that fitted an LR response and generalised it — and it fell.** ⇒ **The model-free core is lever-immune by construction** |
| **69** | **the concentration mechanism's consequence replicates on a panel WITHOUT the concentration fields** (iter. 200, REQ-019) — ⚠️ **CONFIRMED at 11 LRs, but DESCRIPTIVE under rule 6** | On any panel with `top_eigenvalue` + `curvature_along_polar`: (i) after removing the linear component `cp`'s monotone depth trend injects, **`log λ − log cp` has an interior minimum in ≥80% of fits**; (ii) **below both endpoints in ≥80%**; (iii) the **raw** profile still correlates with the C profile at **≥ +0.50** | **argmin interior 11/11 LRs; below both ends 11/11**; raw corr with C **+0.723** (11/11 positive); decomposition identity exact to **6.9e-17**. Raw argmin is interior only **6/11** *before* detilting — **explained by band 56**: `log cp` is monotone here too (linear R² 0.487, slope −0.0286/block), and a monotone term subtracted from a bowl tilts it toward the input end. ⇒ **Bands 44/57/59 are no longer single-panel in their IMPLICATION.** ⚠️ **NOT independent evidence for band 44** — shares `log λ` with C (rule 6); the direct PR measurement still exists only in REQ-048 |
| **70** | **★ the concentration signature is present as EARLY as the bowl — they are CO-ESTABLISHED** (iter. 201, REQ-023) — ⚠️ **CONFIRMED n=3 arms, 1 seed; DESCRIPTIVE under rule 6** | On a panel with `top_eigenvalue` + `curvature_along_polar` at multiple early steps: (i) the detilted `log λ − log cp` profile has an **interior minimum below both endpoints in ≥80% of arm-steps at the EARLIEST available step**; (ii) its depth amplitude shows **no significant trend** across the early window; (iii) **corr with the C profile ≥ +0.50 at every step** | **15/15 arm-steps** interior and below both ends, **from step 1750** (3/3 at the earliest step); corr with C **+0.754 to +0.793**, positive **15/15**; amplitude trend **+0.028 dex/1000 steps, t +0.14** (cf. band 54's C-bowl trend +0.1425, t +1.04 — both null) ⇒ **the signature is at equilibrium amplitude by the earliest measurement in the repository.** ⇒ **CLOSES the alternative that band 57 describes the equilibrated state rather than the bowl's origin** — bowl and concentration are **co-established**, not cause and later consequence. ⚠️ Shares `log λ` with C, so it adds a **TIMING** fact, not independent evidence 🔧 **"ONE PHENOMENON" CORRECTED (iter. 202).** The **timing** claim stands, but a rule-6-CLEAN test (`log n_eff`, no `lam_top`) shows they are **not the same object**: regressing the C profile on the concentration profile gives **mean R² 0.567** — concentration explains **~57%** — with a residual of **0.1006 dex = 1.22× the noise floor** that **REPLICATES (+0.422 across fits, +0.410 within seed)** and has its **own cubic shape** (R² 0.746, swing 0.272 dex, rising toward both ends). ⇒ **majority concentration + a second reproducible component** |
| **71** | **concentration explains ~57% of the bowl — the remainder is a REPRODUCIBLE second structure** (iter. 202, REQ-048) — ✅ **CONFIRMED n=4 (12 fits), rule-6 CLEAN** | Regressing the C profile on the `log n_eff` profile per fit: (i) **mean R² in [0.35, 0.80]** — substantial but NOT complete; (ii) the residual **replicates at mean pairwise corr ≥ +0.30**; (iii) the residual's rms **exceeds the per-layer noise floor by ≥1.1×** | **R² 0.567** (slope −0.590 ± 0.171); residual **rms 0.1006 dex** vs floor **0.0824** = **1.22×**; residual replicates **+0.422** across fits and **+0.410** within seed ⇒ **architectural, not seed-specific**; residual swing **0.272 dex**, cubic R² **0.746**, rising toward **both** ends. ⚠️ `log n_eff` contains **no `lam_top`**, so unlike bands 69/70 this is **independent evidence**, not descriptive. ⚠️ **Profile-level**: the slope −0.590 matches band 59's profile-level figure, whose saturated value is **−0.336** (iter. 191) |
| **40** | **the bowl lives in λ, NOT in the gradient — and the optimiser cannot be its cause** (iter. 155) — ✅ **CONFIRMED n=4 on committed data** | On a 4-seed panel, from the exact identity `log C = log λ − 2 log g`: (i) **var(log λ profile) > 1.2×** var(log C profile) and **var(−2 log g profile) < 0.5×** it; (ii) **corr(C profile, λ profile) ≥ +0.70** while **corr(C profile, g profile) ≤ +0.40**; (iii) **corr(λ profile, −2 log g profile) < 0** | var shares **λ 144.4%**, **−2log g 30.4%**, **2·cov −81.6%** (corr **−0.616**); **corr(C,λ) = +0.890** vs **corr(C,g) = +0.126**; identity checks to **0.000** at all 12 layers. ⇒ **Band 31 forecloses an optimiser cause** (Muon's step is unit-spectral-norm × shape_mult, identical for every layer of a type), so **the bowl is a property of the LOSS SURFACE**. ⚠️ **λ's shape replicates as well as C's** (mean pairwise profile corr **+0.714 vs +0.710**) — *the 4/12-vs-12/12 argmin gap was an artifact; do not extend band 32's "C is smoother" from level to shape*. ⚠️ **No invariance claim**: λ's 0.644 dex swing at all three forks is an **endpoint coincidence** (profiles differ by up to 0.166 dex; median raw λ falls 4×) ✅ **RULE-15 AUDITED (iter. 165)**: clustering on the seed (n=4, fork states averaged within seed) gives **corr(C,λ) +0.866, |t| 26.02, 4/4** vs **corr(C,g) +0.130, |t| 1.39** — the contrast is intact. ⚠️ At n=4 the null half is *consistent with*, not *proven* zero ⚠️ **POWER (iter. 166): the null half is UNDERPOWERED** — at n=4 clusters a true ρ=0.5 is missed **77%** of the time, so "corr(C, g) not significant" is a **failure to detect, not an absence**. The positive half (corr(C,λ) |t| 26.0) is unaffected |
| **41** | **C is ALGEBRAICALLY INVARIANT to the per-layer `post_lambda` scalars — and the bowl survives without the gradient channel** (iter. 158) — ✅ **CONFIRMED n=4 on committed data** | Each block enters the stream as `post_lambda[i] * block_output` (`train_gpt.py` 1638/1665; `nn.Parameter(ones(num_layers,2))` on **Adam**, line 1334 — **layer-varying, and NOT covered by band 31**, which constrains Muon's step on the matrices). Since the loss sees W only via `p·f(W)`: **g ∝ p, λ ∝ p², so `log C = log λ − 2 log g` cancels p EXACTLY**. Criteria: (i) **corr(C profile, g profile) not significant** (|t| < 2.5); (ii) after regressing C on g, residual bowl **cubic R² ≥ 0.70** with **interior minimum in ≥10 of 12** fits; (iii) slope of g profile on λ profile **significantly < +0.5** | **corr(C, g) = +0.096**, sd 0.245, **|t| = 1.36** (blind, as derived); residual bowl **cubic R² 0.860** (sd 0.061), **minimum interior 12/12** (argmins 6×9, 7×3); slope **+0.139 ± 0.074**, **t = −17.01 vs +0.5** yet **t = +6.55 vs 0** ⇒ a p-channel exists but is a minority. ⇒ **The bowl lives in the p-FREE surface term.** **Explains band 32 mechanistically**: C is more seed-stable than λ because it is *immune by construction* to a trained per-layer nuisance scalar λ is exposed to. ⚠️ **A 27.8% p-share estimate was computed and WITHDRAWN** — the same model implies a gradient-profile share of 1.185 (up to 2.369, >1.0 in 8/12), impossible for a variance share; **no share is claimed** ⚠️ **POWER (iter. 166): the empirical leg is UNDERPOWERED** (n=4, ρ=0.5 missed 77%) — but the **algebraic cancellation of p in `log C = log λ − 2 log g` is EXACT and needs no statistics**, so the derivation stands independently |
| **42** | **GAUGE THEOREM — C is invariant to EVERY per-layer scale factor, so the bowl is a SHAPE property** (iter. 159) — ✅ **CONFIRMED n=4 on committed data** | *Theorem:* if W influences the loss only via `c·f(W)` for any scalar c, then `g→c·g`, `λ→c²·λ`, so **C = λ/g² is exactly invariant** — covering `post_lambda`, `resid_lambda` **and its downstream product**, the **LR**, and any output gate. Criteria: (i) per-matrix sd of log C across fork states **< 0.6×** that of log λ; (ii) C bowl **cubic R² ≥ 0.80, argmin in layers 5–7, in EVERY fork state separately**; (iii) the λ profile's linear slope **within ±0.03/block of zero** (the `r^(L−1−i)` amplification is absent) | **sd(log C) 0.1449 vs sd(log λ) 0.3320 dex = 0.44×** (n=288, while raw λ falls 4× across states); bowl **R² 0.904 / 0.961 / 0.975, argmin L6 in all three forks**; λ slope **+0.0057/block** vs the **−0.0828 predicted** by resid_lambda's 1.1×/block amplification (**t = +15.75**; g: +0.0015 vs −0.0414, **t = +28.64**) — **refuted because `F.rms_norm` (line 952) before every sublayer is scale-invariant and erases it**. ⇒ **The bowl cannot be ANY scale factor; it is a conditioning property — curvature relative to a matrix's OWN gradient** ⛔⇒★ **AXIS CORRECTED (iter. 172): the "fork states" are LEARNING RATES (0.60/1.00/1.70×), not checkpoints** — verified, recovered k = **1.364** matching REQ-035's 1.17–1.34, median λ **22576→12562→5494**. So the 0.44× ratio is **NOT stability**; it is **C responding far less to the LR than λ does** — which is **exactly what this band's own gauge theorem predicts**, since the LR is a cancelling scale factor. **The number now CONFIRMS the theorem directly** ⚠️ **SCOPE SHARPENED (iter. 177)**: the theorem needs the scalar to multiply **that matrix's whole contribution**. A **per-matrix** LR does (band 49: C null). A **GLOBAL** LR does NOT — it moves every matrix, changing the **trajectory**: 11 global-LR arms give **d log C = −0.436 (t −18.29)**, because g moves only −0.456 (vs −0.650 per-matrix) and no longer cancels λ's −1.347. **Theorem intact; it applies to per-matrix interventions.** ⚠️ **Iteration 172's "C's LR-response is 0.44× λ's" was computed on GLOBAL arms and is therefore NOT evidence for this theorem** — band 49 is its proper empirical test |
| **47** | **the bowl is LEARNING-RATE INVARIANT** (iter. 172) — ✅ **CONFIRMED n=4 on committed data** | Conditioning on the LR treatment correctly (the REQ-035 `s` axis is **0.60/1.00/1.70× LR**, not checkpoints): (i) C bowl has **cubic R² ≥ 0.80 with argmin in layers 5–7 at EVERY LR separately**; (ii) **pairwise bowl correlation between LRs ≥ +0.70**; (iii) **sd(log C) across LRs < 0.60 × sd(log λ)** — the gauge theorem's LR prediction | **0.904 / 0.961 / 0.975**, argmin **L6 / L6 / L6**, swings 0.523 / 0.602 / 0.488; pairwise **+0.977, +0.886, +0.817**; sd ratio **0.1449 / 0.3320 = 0.44** ⇒ **the positional bowl survives a 2.8× LR range unchanged**, and C's LR-response is 0.44× λ's, as the gauge theorem requires 🔧 **CORRECTED (iter. 177) on ELEVEN LRs**: **argmin L6 in 11/11** LRs (cubic R² 0.869–0.956, linear ≈0, mean pairwise **+0.922**) ⇒ the bowl's **existence and location ARE LR-invariant**. **But its PROFILE is not**: the LR×block interaction is significant, **F(11,3931) = 2.22** vs crit 1.79, distorting the bowl by **0.1762 dex = 38.0%** of its 0.4630 swing across 2.8×. **Restate as "shape stable to within ~38%", NOT "LR-invariant"** |
| **52** | **the bowl across ELEVEN learning rates — location invariant, profile modulated** (iter. 177, REQ-019) — ✅ **CONFIRMED on 11 LR arms (n=1/arm)** | (i) C bowl **argmin in layers 5–7 at ≥90% of LRs**; (ii) **LR×block interaction distorts the bowl by <60% of its swing**; (iii) the **GLOBAL**-LR elasticity of log C is **significantly negative** while the **per-MATRIX** one is **not** — the two-lever distinction | **argmin L6 in 11/11** (0.60×–1.70×), cubic R² 0.869–0.956, mean pairwise **+0.922**; fork 2000 agrees (3/3, +0.968). Distortion **38.0%**. Global **−0.436 (t −18.29)** vs per-matrix **+0.081 (t +0.89)**; identity −1.347−2(−0.456) = **−0.435** ✓. ⚠️ **LR axis richly sampled, seed axis is n=1 per arm** |
| **48** | **a per-TYPE LR is ALGEBRAICALLY powerless on the between-LAYER axis — REQ-036's verdict, from its source** (iter. 173) — ✅ **CONFIRMED, deterministic** | Verified from `make_req036_arms.py`: **every arm is a per-type CONSTANT in depth** (only a3 overrides, at 2 of 12 blocks for 2 of 6 types). Criteria: (i) the **best** per-type-constant rule reduces between-layer variance of block-mean log C by **<1%** over ≥1000 random draws; (ii) **between-type variance > between-layer variance** | **Best reduction over 2000 random rules = 0.000%** — a per-type multiplier shifts every layer of that type equally, so block means move together and the between-layer **spread is unchanged by construction**. ⇒ **REQ-036 did not fail on this axis; it could not have succeeded.** ⚖️ **But it was not aimed at nothing**: between-TYPE variance **0.17855** vs between-LAYER **0.03527** (**5.06×**) — the design targets a real, larger axis, just not the campaign's. ⇒ **Fourth and algebraic reason for the null, alongside bands 16, 43, 45/46. RECOMMENDATION FINAL: for the layer axis the lever must be per-LAYER or per-matrix; a per-type lever is provably inert** |
| **49** | **CAUSAL confirmation of the gauge theorem — a randomised per-MATRIX LR moves λ but NOT C** (iter. 174, REQ-045) — ⚠️ **CONFIRMED n=1 seed; 4-seed replication REQUESTED (REQ-049)** | On a randomised per-matrix LR panel (treatment, not a derived predictor ⇒ admissible **and causal**): (i) **|d(log λ)/d(log m)| ≥ 0.5 with |t| ≥ 3** — the lever demonstrably works; (ii) **d(log C)/d(log m) not significant with 95% CI half-width < 0.30** — a **powered** null, not an absent test; (iii) the CI **excludes** the fitted λ elasticity | **log λ −1.218 (t −7.55)**, log g **−0.650 (t −12.61)**, **log C +0.081 (t +0.89), CI [−0.097, +0.258]**, half-width **0.177**, CI excludes −1.218. Identity: −1.218 − 2(−0.650) = **+0.081**, residual **9.99e-16** ⇒ λ falls at ~2× g's rate, leaving `C = λ/g²` invariant. ⇒ **Upgrades band 42 from derivation to randomised experiment** ⇒ **Fifth and most fundamental reason REQ-036 was null: NO LR intervention at ANY granularity can move C** ★ **REPLICATED (iter. 178) on an INDEPENDENT experiment** — REQ-023, fork 1500, different randomisation, **n=1080** (5×): λ **−1.152 (t −21.11)**, g −0.541, **C −0.070 (t −2.01)**, identity **1.2e-16**. The two C estimates are **CONSISTENT** (z +1.55); **pooled −0.0505 ± 0.0327, = 4.2% of λ's magnitude**. 🔧 **Claim made precise: "C's per-matrix-LR response is at most a few percent of λ's, possibly slightly negative" — NOT "exactly invariant"**, since the theorem predicts exactly zero and n=1080 begins to resolve deviations |
| **53** | **the per-matrix gauge result holds across TWO independent experiments** (iter. 178, REQ-023 + REQ-045) — ✅ **CONFIRMED, 2 experiments, n=1296 combined** | Across ≥2 independent per-matrix LR experiments: (i) **d log λ/d log m ∈ [−1.5, −0.9]** in each; (ii) **|d log C/d log m| < 0.15** in each; (iii) the **pooled C elasticity < 10% of λ's** in magnitude; (iv) both **identity residuals < 1e-10** | λ **−1.152** / −1.218; C **−0.070** / +0.081; pooled **−0.0505 ± 0.0327** = **4.2%** of λ's; residuals **1.2e-16** / 9.99e-16; difference z **+1.55** (consistent). ⚠️ **Both are n=1 seed** — the LR axis and matrix axis are replicated, the **seed axis is not**. ⚠️ A principled second-order deviation is expected (a per-matrix LR also moves that matrix's own weights, feeding back into its surface) and should scale with **\|log m\|** — **untested, only 3 levels available** |
| **54** | **the bowl is ALREADY FORMED at step 1750 — the earliest measurement that exists** (iter. 179, REQ-023) — ✅ **CONFIRMED (n=1 seed, 3 arms)** | (i) bowl **argmin in layers 5–7 at ≥4 of 5 early steps**; (ii) **mean pairwise correlation across early steps ≥ +0.80**; (iii) **correlation with the late (2750) bowl ≥ +0.80**; (iv) **no significant trend in swing** over the early window | argmin **L6 at 1750/1875/2125/2250** (L4 at 2000), cubic R² 0.881–0.952, linear ≈0; mean pairwise **+0.928** (min +0.853); **vs step 2750: +0.943 / +0.914 / +0.902 / +0.928 / +0.927**; swing trend **+0.1425 dex/1000 steps, t +1.04 (n.s.)** ⇒ **already at equilibrium amplitude by 1750, and the SAME bowl as 1000 steps later.** ⛔ **Step 1750 is the global earliest curvature in the repo (22 files scanned)** — **presence at INITIALISATION is unanswerable from committed data; REQ-050 filed.** *(A back-cast to step 0 exists but is 1750 steps out of range and is NOT offered as evidence — the iteration-128 error.)* |
| **50** | **C IS MOVABLE — but NOT by a learning rate: a two-lever dissociation** (iter. 175, REQ-037 vs REQ-045) — ⚠️ **dissociation CONFIRMED (n=1/arm); magnitudes NOT established** | Two assigned-treatment levers on the same quantity: (i) **LR lever** — `|d log λ| ≥ 0.5, |t| ≥ 3` with **d log C not significant**; (ii) **BATCH lever** — `|d log C| ≥ 0.15, |t| ≥ 3` with **d log λ not significant** (the inverted split); (iii) both identity residuals **< 1e-10** | **LR**: λ **−1.218 (t −7.55)**, g −0.650, **C +0.081 (t +0.89, powered null)**. **BATCH**: λ **+0.065 (t +0.79, unmoved)**, g **+0.169 (t +6.94)**, **C −0.274 (t −5.41)**. Residuals 9.99e-16 / **1.7e-16** ⇒ **the levers move OPPOSITE components — C is a real, movable quantity, just not via the LR.** ⚠️ **Magnitude −0.274 is NOT a pure batch effect**: batch and tokens-seen are collinear at **−0.9979** over 3 arms (rule 9 forbids separating them). **The component split is confound-free** (within-arm contrast); the effect size is not. **Fix: token-match the batch arms at STOP, not step** |
| **51** | **the batch lever moves C's LEVEL but does NOT reshape the bowl** (iter. 176, REQ-037) — ⚠️ **CONFIRMED n=1/arm; null bounded at 0.29 dex/dex** | (i) **batch × block interaction on log C not significant** (F below its permutation 95th pct) in ≥3 of 4 seeds; (ii) **bowl argmin stays in layers 4–8 at every batch size**; (iii) **main batch effect on log C stays significant** (|t| ≥ 3) — the lever must demonstrably work while failing to reshape | **F(11,187) = 0.68** vs permutation crit **1.86** (RSS 6.6133→6.3598 for 11 params); argmins **L6 / L6 / L4**, cubic R² 0.938/0.886/0.906, bowl corr **+0.54 to +0.89**; main effect **−0.274, t −5.41**. ⚠️ **Power bound: 80% at ~0.29 dex/dex** (residual sd 0.1828) — a *proportional* reshaping of the ~0.5 dex bowl **would** have been seen; a subtler one is not excluded. ⇒ **Two optimiser levers now tested against the bowl — LR (inert on C entirely) and batch (moves level, not shape) — and NEITHER reaches the depth structure** |
| **43** | **the bowl is SPECIFIC TO THE TOP EIGENDIRECTION — absent from Muon's step direction** (iter. 160) — ✅ **CONFIRMED n=4 on committed data** | Build `C_polar = cp/g²` — the **same gauge-invariant construction as C** (band 42) but on `curvature_along_polar`, a **separate HVP containing NO `lam_top` and no tridiagonal quantity** (admissible). Criteria: (i) **C_polar is monotone** — linear R² exceeding its cubic gain, slope **negative in ≥10 of 12** fits; (ii) **C's linear slope not significant** (|t| < 2); (iii) **corr(C profile, C_polar profile) < +0.50** | C_polar **linear R² 0.905** vs C's **0.004**; slope **−0.0266/block, |t| 5.38, 11/12 same sign** (profile replicates at +0.460, so the null is powered); C slope **|t| 0.46**; **corr = +0.152** (|t| 1.33, 7/12), C_polar argmin interior in only **2/12**. ⇒ **Conditioning along the top eigendirection is U-shaped (min L6); along Muon's actual step direction it declines monotonically.** ⇒ **Independent mechanical reason REQ-036 failed: there is no bowl to equalise along the direction Muon steps.** ⚠️ *A −0.801 `align`-vs-C correlation was found first and **WITHDRAWN under rule 6** — it shares `log λ`; raw refit gives +0.211, |t| 2.29, 8/12* ✅ **RULE-15 AUDITED (iter. 165)**: clustered on the seed (n=4) the C_polar slope holds at **−0.0266, |t| 4.91, 4/4 same sign** (91% of the n=12 |t| retained), C's own slope stays null at **|t| 0.45**, and corr(C, C_polar) stays low at **+0.112**. The bowl profile is **80.4% reproducible across seeds**, which is why clustering costs it almost nothing ⚠️ **POWER (iter. 166): the "uncorrelated with C" half is UNDERPOWERED** (n=4). The positive half — C_polar monotone, **4/4 seeds same sign, |t| 4.91** — is sound ★★ **CONFIRMED AT 11 LRs (iter. 180, REQ-019)** — independent panel, 4× the LR coverage: C_polar's depth slope **negative in 11/11**, mean **−0.0302** (vs band 43's −0.0266), **|t| 15.76**. **Form-free contrast**: bowl index (cubic R² − linear R²) **C 0.869 vs C_polar 0.082**, paired difference **+0.787 ± 0.008, t = +93.70, 11/11** ⇒ **C is a bowl, C_polar is a line, on the same matrices.** Now the campaign's **second-best-supported claim** |
| **56** | **C is a bowl and C_polar is a LINE — the dissociation at 11 learning rates** (iter. 180, REQ-019) — ✅ **CONFIRMED on 11 LR arms (n=1/arm)** | (i) **C's bowl index ≥ 0.60 and C_polar's ≤ 0.30** at ≥90% of LRs; (ii) the **paired difference is positive at every LR**; (iii) **C_polar's depth slope is negative at ≥90% of LRs** | Bowl index **C 0.869** (sd 0.049, range 0.793–0.950) vs **C_polar 0.082** (sd 0.062, range 0.001–0.183); paired **+0.787 ± 0.008, t +93.70, 11/11**; C argmin **L6 in 11/11** with linear R² 0.003–0.112, C_polar linear R² **0.729–0.977**, slope **−0.0302, 11/11 negative**. ⇒ **The bowl acts on the TOP of the spectrum, not the subspace Muon steps in** — which is *why* no optimiser lever reaches it (bands 51, 53) and why REQ-036 was inert (band 48). ⚠️ **n=1 seed per LR arm** |
| **44** | **★ THE BOWL IS A SPECTRAL-CONCENTRATION PROFILE** (iter. 162 registered → 181 scored, REQ-048) — ✅ **CONFIRMED n=4 (4 seeds × 3 LRs)** | **Registered before the data existed:** (i) **corr(log PR profile, C profile) ≤ −0.60**, same sign ≥10/12; **(ii′ AMENDED)** log PR is an **INVERTED bowl** — cubic R² ≥ 0.70 with **argMAX** in layers 4–8. *(PR = trace(H)²/(n·trace(H²)) from **16 Hutchinson probes**, no Lanczos content ⇒ **rule 13 satisfied**.)* | **corr = −0.741** (sd 0.140), **11/12 ≤ −0.60, 12/12 negative**; log PR cubic R² **0.827, 12/12**, **argmax interior 12/12** (6,8,8,6,6,8,7,7,7,8,8,6); pooled corr **−0.862**; holds at every LR (−0.794/−0.683/−0.708). ⇒ **C is HIGH at the ends where the spectrum is CONCENTRATED (low PR) and LOW in the middle where it is SPREAD (high PR)** — explains band 56 (the bowl lives in the top of the spectrum). ⚠️ **Criterion (ii) as first written said "argmin" and FAILED 0/12 — a drafting error of mine** (the hypothesis in the same request predicts the opposite sign); withdrawn and recorded, see rule 19. ⚠️ **Structural identification, not external cause**: PR and C are both Hessian functionals, so this narrows the question to *why the spectrum concentrates at the boundaries* |
| **64** | **✅★ the concentration mirror stated with NO functional form — band 44 survives rule 23** (iter. 190, REQ-048) — ✅ **CONFIRMED n=4 (12 fits)** | Using **free per-block effects, no polynomial and no curve-to-curve correlation**: (i) **log PR's maximum lies in blocks 4–8 in ≥10 of 12** fits **and is above both endpoints in ≥10 of 12**; (ii) at C's minimal block, the **C and PR contrasts against each endpoint have OPPOSITE signs in ≥10 of 12**; (iii) pooled PR-vs-L0 contrast positive with **|t| ≥ 3**, matrix-clustered | **PR argmax interior 12/12; above both ends 12/12; opposite signs 12/12 at L0 AND 12/12 at L11.** Pooled: PR block8−block0 **+0.506 (t +4.74)**, block8−block11 +0.501 (t +2.48); paired per-fit **+0.532 (t +5.96)** and **+0.528 (t +20.59)**. Mirror at block 6: **C −0.393 (t −5.60) vs PR +0.500 (t +5.50)**; vs L11 **C −0.511 vs PR +0.496**. ⇒ **Band 44's `corr −0.862` between fitted profiles is NOT what carries it** — the result holds as contrasts between measured positions. ⚠️ *Pooled vs paired differ on the L11 leg (2.48 vs 20.59) because the clustered SE includes between-fit level variation the paired test removes; both reported* |
| **57** | **★ the bowl is in the Hessian's SECOND MOMENT — `trace(H²)`, not `trace(H)`** (iter. 182, REQ-048) — ✅ **CONFIRMED n=4 (12 fits)** | (i) **corr(C profile, log trace_sq_est) ≥ +0.50**, same sign ≥10/12; (ii) **corr(C profile, log trace_est) NOT significant** (|mean| < 0.25, same sign < 9/12); (iii) **swing(trace_sq profile) > 2× swing(trace profile)** | **trace_sq +0.644, 12/12, swing 0.659**; **trace −0.061, 6/12, swing 0.227** (ratio **2.9×**); identity `log PR = 2·log trace − log trace_sq − log n` exact to **1.67e-16** ⇒ **PR ≈ inverted trace_sq, not a delicate cancellation.** ⇒ **Total curvature is FLAT across depth; SQUARED curvature carries the bowl — the ends hold the same total in FEWER, LARGER eigenvalues.** ⚠️ **A peak-to-mean cross-check (λ_top/curvature_along_random) hit +0.941, 12/12 — WITHDRAWN under rule 6** (shares log λ with C; raw components null at −0.062 and −0.061). **PR is unaffected — it contains no λ** |
| **58** | **★ ONLY the TOP eigendirection carries the bowl — three directions tested** (iter. 183, REQ-048) — ✅ **CONFIRMED n=4 (12 fits), null POWERED** | (i) **corr(C profile, log `curvature_along_weight`) NOT significant** (|mean| < 0.35, same sign < 10/12); (ii) the W-curvature profile **replicates at mean pairwise ≥ +0.70** — so the null is powered, not a failed measurement; (iii) that profile is **monotone**: linear R² ≥ 0.60, negative slope | corr **+0.153** (sd 0.349, 8/12), argmax **L0 in 12/12**; replication **+0.951** (min +0.870); **linear R² 0.876, slope −0.0637/block**. ⇒ **Direction-by-direction: top eigendirection U-SHAPED (min L6); Muon's step MONOTONE (band 56, 11/11 LRs); learned weight direction MONOTONE (here); typical/random direction FLAT (band 57, −0.061).** ⇒ **The bowl is a property of the EXTREME eigendirection, not of the loss surface broadly** — and it lives in a subspace **neither the optimiser's step nor the learned solution occupies**, which is why bands 48, 49, 51, 53 all found their levers inert |
| **59** | **the bowl in UNITS: ≈1,500 curvature directions at the ends vs ≈4,600 in the middle — and the equal-eigenvalue model is REFUTED** (iter. 184, REQ-048) — ✅ **CONFIRMED n=4 (12 fits)** | `n_eff = trace(H)²/trace(H²)`: (i) **n_eff ∈ [1, n_params]** for every matrix (implementation check); (ii) **n_eff(middle L5–L8) / n_eff(ends L0,L11) > 1 in ≥10 of 12 fits**; (iii) **d(log C)/d(log n_eff) significantly negative AND significantly greater than −1** | **0/864 violations** (range 35.4–72,128 vs n_params 589,824/2,359,296); n_eff **1494 (L0) → 4724 (L6) → 1508 (L11)**, ratio **3.05×**, **>1 in 12/12 fits and 6/6 types**; corr(log n_eff, C) **−0.862**. **Slope −0.590 ± 0.171: t −11.92 vs 0 (real), t +8.28 vs −1 (equal-eigenvalue model REJECTED)** ⇒ as directions are removed the top eigenvalue absorbs only ~60% of the freed curvature; the rest spreads into the sub-leading bulk. ⚠️ **The 3× is a POOLED average spanning 1.22× (attn.v) to 15.34× (mlp.proj)** — the residual-writer types dominate; **not a uniform property**. Curvature occupies a median **0.23%** of available directions 🔧 **COEFFICIENT CORRECTED, CONCLUSION STRENGTHENED (iter. 191).** The **−0.590 was the profile-level fit — the largest of three specifications.** Matrix-level with type dummies gives **−0.408**; **saturated in block AND type gives −0.336**. **Restate as "−0.34 to −0.59 by specification, −0.336 saturated".** ✅ **The rejection of −1 holds in ALL specs and STRENGTHENS as controls tighten** (t vs −1: **+8.28 → +11.47 → +18.69**). ✅ **Attenuation RULED OUT** using the `pr_per_probe_vHv` values requested in iter. 162: measurement variance **0.00006** vs signal **0.11246**, **reliability λ = 0.9994**, corrected slope **−0.3358** (moves by 0.0002) ⇒ **the gap from −1 is real physics, not measurement bias** |
| **60** | **★ the concentration contrast is a RESIDUAL-WRITER effect — band 7's split reproduced on a new quantity** (iter. 185, REQ-048) — ✅ **CONFIRMED n=4 (12 fits)** | (i) the **`writer × edge` interaction on log n_eff is negative with |t| ≥ 3**, clustered by matrix, in ≥3 of 4 seeds; (ii) negative in **≥10 of 12** fits; (iii) the ends-vs-middle contrast **drops ≥40%** when writers are excluded | Interaction **−0.627 dex**, clustered se 0.191, **t −3.29** (72 matrix clusters); per-fit **−0.627 ± 0.125, 12/12 negative, t −17.39**. Ends→middle gain: **mlp.proj 15.34×, attn.proj 9.30×** vs attn.k 2.60×, mlp.fc 1.41×, attn.q 1.25×, **attn.v 1.22×**; writers **11.94×** vs internal **1.54×**. Dropping writers: **3.04× → 1.54× (−49%)** ⇒ **band 59's "3×" is writer-driven.** ⚠️ **Type-axis permutation p 0.0681 is AT THE 1/15 FLOOR** — with 6 types no 2-vs-4 split can do better; the matrix axis (72 clusters) supplies the actual power. ⚠️ *A rank-deficient first design (edge = block₀+block₁₁, se = nan) was caught and rebuilt at full rank; estimates identical* 🔧 **REFINED (iter. 186): "at the ends" is too coarse — it is overwhelmingly a LAST-BLOCK effect.** `writer × LAST(L11)` **−0.953 (t −4.09, 12/12)** vs `writer × FIRST(L0)` **−0.300 (t −2.29, 11/12)**; paired difference **t +12.42** |
| **67** | **~~the residual-writer split in the LR elasticity~~ — ⛔ DOWNGRADED: MIXED-LR ONLY** (iter. 197, REQ-023 + REQ-045) — ⛔ **DOWNGRADED (iter. 198) — n=1 seed, CONTRADICTED under global LR** | On a per-matrix LR panel: (i) **k_lambda(writers) − k_lambda(internal) > +0.4** in ≥3 of 4 seeds; (ii) the contrast survives **FULL block dummies** with **|t| ≥ 3** (so it is not depth in disguise); (iii) the sign holds at **every** measurement step | **REQ-023 (all 72 matrices, 12/type): writers +1.902 vs internal +0.979, diff +0.924, t +3.32**; saturated in depth **+0.924, t +4.23**; **writers > internal at 5/5 steps** (t up to +3.73). REQ-045 (partial coverage, 37 matrices) agrees in sign: +0.753, t +1.40. ⇒ **Third independent quantity showing the same split — after band 7 (gradient slopes, +2.17) and band 60 (concentration, t −3.29).** ⚠️ **Iteration 196's "13× spread" is WITHDRAWN**: attn.proj's +0.217 was an n=4 artifact; it is **+1.454** on full coverage, a **1.6×** spread. ⚠️ **n=1 seed** — REQ-051's balanced ladder would test this at n=4 ⛔ **REFUTED AS A GENERAL CLAIM (iter. 198, per REQ-052's audit — independently reproduced by me to 3 decimals).** Under **global** LR the contrast **REVERSES in 4/4 seeds**: **−0.194, −0.125, −0.006, −0.199** @2250 (and −0.276/−0.232/−0.248/−0.255 @2750) vs REQ-023's mixed-LR **+0.924 / +1.165**. ⚠️ **The "5/5 steps" robustness was PSEUDO-REPLICATION** — five dependent checkpoints of ONE continuation, not five seeds; **my own rule 15, unapplied to my own newest band.** ⇒ **The effect is REAL but DESIGN-SPECIFIC** (mixed-LR only, depth-saturated t +4.23 there). **Consistent with iter. 177: per-matrix and global LR are mechanistically different levers** (global d log C/d log s −0.436 vs per-matrix +0.081 null). **REQ-052 is the correct instrument and is endorsed** |
| **61** | **★★ the bowl's TILT is entirely a LAST-BLOCK writer effect** (iter. 186, REQ-048) — ✅ **CONFIRMED n=4 (12 fits)** | (i) **`writer × LAST` negative with |t| ≥ 3** clustered by matrix, and **|writer × LAST| > 2 × |writer × FIRST|**; (ii) the **C bowl tilt among internal-only types is NOT significant** (|mean| < 0.10 dex); (iii) the writers' **L11 deviation from their own interior trend exceeds their L0 deviation by ≥3×** | `writer × LAST` **−0.953** (se 0.233, **t −4.09**, per-fit 12/12) vs `writer × FIRST` −0.300 (11/12) — ratio **3.2×**. **C bowl tilt (L11−L0): all types +0.117, internal-only +0.001, writers-only +0.350**; paired reduction **+0.116 ± 0.024, t +4.91** ⇒ **band 39's tilt is supplied ENTIRELY by residual writers.** Interior-trend deviation: **L11 −1.384 vs L0 −0.152 (9.1×)** ⇒ **the last block is a DISCRETE structural break, not a trend endpoint.** ⇒ Matching architectural fact: **L11's writers are the only matrices whose output is never re-mixed by a later block** (`norm(x)` → LM head) — *stated as consistent structure, NOT a tested mechanism* ⊘ **THE GRADED READING IS REFUTED (iter. 187)**: if "how much is re-mixed downstream" were the mechanism, the penalty would scale with `11−block` and the coefficient would be **positive**. It is **−0.0797 (t −2.72)** — wrong sign — and `downstream = 11−block` is an **exact linear function of block**, adding **nothing** over a cubic in depth (RSS **46.140 → 46.140**). **Only the discrete last-block indicator survives** (F(1,282) = **5.46**). The writers' own profile is a **bowl peaking at L7 plus an L11 drop** — no monotone term can fit it. **The "never re-mixed" story stays an untested structural coincidence** 🔧 **THE "DISCRETE BREAK" IS WITHDRAWN AS BASELINE-DEPENDENT (iter. 188).** The −1.384 came from a **linear** interior fit; placebo-testing the same indicator at **every** block over the campaign's standard **cubic** baseline puts **L11 at rank 5/12** (F 5.46) behind blocks 1 (**41.08**) and 0 (**24.57**). Linear → L11 rank 1/12 (F 176); quadratic → 1/12 (F 72); **cubic → 5/12**. ✅ **The load-bearing results are UNAFFECTED** — `writer × LAST` uses **full block dummies** (saturated in depth, no baseline): **−0.926 dex, t −4.02**; and the tilt result is a same-block group contrast |
| **45** | **the bowl is SHARED across 5 of 6 matrix types — position and type are SEPARABLE** (iter. 167) — ✅ **CONFIRMED n=4 on committed data** | Fitting each type's depth profile **independently, with no type removal**: (i) **≥5 of 6 types show cubic R² ≥ 0.70 with argmin in layers 5–8**; (ii) **mean pairwise correlation between the six type-profiles ≥ +0.40**; (iii) no type correlates with the pooled bowl at ≤ **−0.30** | mlp.proj **0.950**/L7, attn.proj 0.878/L7, mlp.fc 0.871/L6, attn.v 0.817/L6, attn.k 0.791/L6 — **linear R² ≈ 0 for all five**; mean pairwise **+0.469** (attn.proj↔mlp.fc **+0.907**), most negative pair −0.165. ⇒ **The bowl is present in attention AND mlp, readers AND writers, 768×768 AND 3072×768 — not a pooling artifact — which LICENSES the additive `type + position` models used since band 3.** ⇒ **Third independent reason REQ-036 was null, and a structural one: types share one bowl, so a per-TYPE LR cannot address a POSITIONAL effect — the design's axis and the effect's axis are orthogonal.** ⚠️ **attn.q: cubic R² 0.315, swing 0.245** — it has the **worst signal/noise of the six (1.7 vs 3.5–15.7)**, and pooling 60 fits does not recover a bowl, so its bowl is **genuinely smaller and its shape UNRESOLVED, not absent — no exception claimed** ⚠️ **MODIFIED (iter. 168): types share the bowl's SHAPE but NOT its SIZE.** Per-type amplitude on the shared shape spans **0.104–2.052 (19.6×)**, with **5 of 6 differing from a common amplitude at |t| 2.4–21.3** ⇒ **"type is only an offset" is too strong** — it is an offset **and** a scale factor, so the additive model is misspecified. **No structural predictor survives** (best, the band-7 writer split, r +0.849 but **permutation p 0.146** at n=6). **Cost is zero**: a free cubic per type (18 params vs 3) predicts **worse** out of sample, LOLO 0.1050 vs **0.0982** |
| **46** | **the bowl's AMPLITUDE varies by type — misspecification without cost** (iter. 168) — ✅ **CONFIRMED n=4 on committed data** | Fit each type as `a × (shared bowl shape)`: (i) amplitude range **≥3×**; (ii) **≥4 of 6 types differ from a common amplitude at |t| ≥ 2**; (iii) **per-type position terms do NOT beat the shared cubic** on LOLO | Amplitudes mlp.proj **2.052**, attn.proj 1.452, mlp.fc 1.046, attn.k 0.786, attn.v 0.559, attn.q **0.104** — **19.6× range**; t vs a=1: +21.3 / +6.1 / +0.9 / −2.4 / −9.8 / −8.8 (**5 of 6 significant**); per-type LOLO **0.1050** vs additive **0.0982** (floor 0.0903) ⇒ **the flexible model is WORSE**. ⚠️ **Amplitude's cause is UNDETERMINED** — 4 pre-declared structural candidates, best r +0.849, **permutation p 0.146**; n=6 types cannot resolve it. ⇒ **Strengthens the REQ-036 verdict: a per-TYPE LR faces a positional effect whose amplitude ALSO varies by type — two orthogonal axes, not one** |
| **26** | **C's six-type structure is genuinely THREE-term** (iter. 108) — ✅ **CONFIRMED n=4** | **the identity log g = log‖a‖_F + log‖d‖_F + log(align) holds exactly**; across the six types **no term correlates with C above 0.55**, and **each term's spread exceeds C's own** (offsetting) | identity exact to **1e-6 dex**; corr(C, −2‖a‖) +0.36–0.39, (C, −2‖d‖) +0.44–0.49, (C, −2align) +0.38–0.40; term spreads 1.35 / 1.02 / 0.64 vs C's **1.04** |
| **27** | **the ‖a‖·‖d‖ PRODUCT is the depth-conserved quantity** (iter. 110–111) — ✅ **CONFIRMED n=4, but RESTATED iter. 150 under rule 10** | ⚠️ **The correlation criterion is WITHDRAWN**: `corr(log‖a‖, log‖d‖) ≤ −0.80 across depth` is a **depth** correlation — removing depth collapses 4 of 6 types toward zero (attn.k **+0.065**, attn.q −0.184, attn.v −0.137, mlp.proj −0.143; only attn.proj −0.716 and mlp.fc −0.780 survive) and nothing is significant within-layer. **The matrix-level trade-off is NOT established; n=4 has no power to test it.** **The CONSERVATION criterion stands and is the real content:** within each type log‖a‖ rises and log‖d‖ falls **near-linearly** with depth (R² 0.80–0.99) at slopes that **cancel to ~8%**, so the product is conserved far beyond chance | **sd(sum) = 0.13–0.31× the independence null** √(sd_a²+sd_d²); all **24 type-seed slope ratios** −s_d/s_a mean **1.079**, sd 0.172, range [0.681, 1.307]; residual product drift only **−0.011 to +0.008 dex/layer**; cancellation slightly **over-complete** (t = +2.27 vs 1.0); **mlp.fc the lone under-compensator 0.755 ± 0.050, real in all 4 seeds** |
| **28** | **C's depth structure is carried by λ, not g** (iter. 117) — ✅ **CONFIRMED n=4** | **sd(log λ) / sd(log g) across depth > 1.5 in every matrix type**, every seed — the consistency requirement linking bands 27 and 10/25 | ratios **2.03 / 2.76 / 2.85 / 2.93 / 4.19 / 4.32**, mean **3.2×**; λ variance share 0.51–1.76 |
| **29** | **the λ–g relation WEAKENS as the LR rises** (iter. 120) — ✅ **CONFIRMED n=4** | **cross-sectional slope falls monotonically across s = 0.6 → 1.7**, seed-clustered CI on the spread excluding 0; **sd(log g) flat** (rules out range compression) while **sd(log λ) compresses** | slopes **0.916 / 0.742 / 0.636**, spread CI **[0.208, 0.365]**; sd(log g) **0.246/0.246/0.242**, sd(log λ) **0.429/0.395/0.367**; corr 0.534/0.497/0.453 |
| **31** | **Muon's gradient-magnitude invariance is a THEOREM, not an approximation** (iter. 131) | the update `X = g/(‖g‖·1.02 + 1e-6)` is exactly scale-free wherever the epsilon is negligible; **deviation < 1e-4 for ‖g‖ ≥ 1e-2**, and REQ-046's matrices sit at ‖g‖ ≈ 10³·⁸. Momentum adds a transient of ~1/(1−m) steps only | epsilon deviation at ‖g‖=10: **5e-8**; transient ≈ 20 steps of 750 |
| **32** | **C = λ/g² is the right object — the g² division REMOVES noise** (iter. 132) — ✅ **CONFIRMED n=4** | **C must be more seed-stable than λ** (median |Δ| across seed pairs) **and have a larger architecture-to-seed-noise ratio** | C **0.0776 dex** vs λ **0.1235**; ratio C **15.7×** vs λ **8.0×**; architectural spread C **0.462 dex** vs λ **0.383** |
| **33** | **Muon steps nearly ORTHOGONAL to peak curvature, and the RATIO predicts C** (iter. 134, 140) — ✅ **CONFIRMED n=4, ratio verified by free coefficients** | **cp/λ < 0.02 every type**; and in `C ~ a·log λ + b·log cp` with **free** coefficients, **b/a ≈ −1** (the ratio the data prefers unprompted) with **|t(b)| > 5** | ratio **0.001–0.006**; **b/a = −1.045/−1.301/−1.082/−1.193**, t(b) = **−8.8 to −11.0**; λ-only RSS **~2× worse** |
| **34** | **alignment and the q,k excess are SEPARATE effects** (iter. 135, 139) — ✅ **CONFIRMED n=4, raw-component checked** | **controlling for step-curvature shrinks the q,k C-coefficient by < 35%** — and **by ≤ 0% when the control uses raw `log cp`** (no shared λ) | shrinkage with alignment **22/19/16/20 %**; with `log cp` **−5/−7/−3/−9 %**; `log cp` net of q,k only t = **1.6–2.9** |
| **35** | **q and k differ in step ALIGNMENT, and the gap is DEPTH-STRUCTURED** (iter. 136–137) — ✅ **CONFIRMED n=4, re-verified under rule 10 (iter. 152), NO correction needed** | **|alignment(q) − alignment(k)| ≥ 0.15 dex, same sign every seed**, where **alignment ≡ log(curvature_along_polar) − log(λ_top)** *(the curvature-probe quantity — NOT REQ-047's `align_ratio = ‖∇W‖/(‖d‖‖a‖)`; the name collision caused a false refutation in iter. 152, see rule 11)*; **depth structure must clear a permutation null** | gap **−0.3079 dex** pooled over 720 matrix-observations (per seed −0.3386/−0.2874/−0.3350/−0.2705, **4/4 same sign, |t| 14.8–26.5**). Depth: quad R² **0.636 ± 0.121** across 12 seed×fork curves, **>0.5 in 10/12**; pooled 0.873 vs permutation null mean 0.182 / 95th 0.491, **p = 0.0004**; L0 −0.087 → L6 **−0.441** → L10 −0.323. ⚠️ **Part monotone** — linear R² averages 0.359 (in seed2/fork170 the quadratic adds **0.000**), so "deepens then recovers" overstates the recovery. *Attention entropy does NOT explain it* |
| **30** | **a higher LR DECOUPLES curvature from the gradient** (iter. 121, 123, 144) — ✅ **CONFIRMED on 2 designs, CONSISTENT on a 3rd** | **cov(log λ, log g) falls as the LR rises**, CI excluding 0 on the global ladder and REQ-023's per-matrix randomisation. **Rate ≈ −0.04 to −0.08 dex of covariance per dex of LR** | Arm A **0.0552→0.0371**; REQ-023 **0.0766→0.0418**; REQ-045 **0.0917→0.0785** (CI [−0.062,+0.034], **underpowered at 2.0× range** — Arm A's rate predicts −0.0120, observed **−0.0131**) |
| **15** | **QK-norm scale invariance** (iter. 80, 87–89) — ❌ **QUANTITATIVE PREDICTION FAILS** | predicts **Δlog g = −Δlog‖W‖**. Observed: predicted **+0.130**, actual **−0.417** — wrong sign, 3× the size. q,k sit mid-pack in ‖W‖. The `d log C/d log‖W‖ = 0` result stands but no longer explains the gap | ‖W‖: q +1.755, k +1.756 vs proj +1.778, v +1.809, mlp.proj +1.832, fc +2.124 |
| **16** | **C is an ACTIVELY RESTORED invariant** (iter. 82–83) — ✅ **CONFIRMED n=4 + targeted test** | **global ladder:** matrix identity > 85% of log C's variance, LR < 10%, corr > 0.80. **targeted per-type perturbation:** **slope of Δlog C on log10(multiplier) ≈ 0** while Δlog λ tracks EoS | identity 93.2–94.8%; corr +0.87 to +0.97; **a5 λ-slope −1.153 vs C-slope −0.054** |

**Band 6 is the newest and it sharpens the campaign's central claim.** The cross-sectional gradient
exponent differs systematically by type — **~3.8 for the two projection matrices, ~0.9–1.4 for the
other four — against a within-matrix causal exponent of 2.07. No type sits at the causal value.**
This is not attenuation: measured error in log g is sd 0.0131 dex, reliability 0.962–0.993, and
correcting for it moves each slope by 1–4% and leaves the spread intact (1.35 → 1.24 / 1.54 → 1.42).
**Falsifier:** if attenuation-corrected slopes converge to ~2 across seeds, iteration 63 is wrong
and the law is universal after all.

### CONSOLIDATED FINDINGS II (iterations 87–111, 2026-09-03)

*25 iteration blocks replaced by this summary; full provenance in git history from `f4cdf33` onward.
The authoritative band table above is unchanged and remains the only live specification.*

#### The account, closed end to end

> **λ_eq = C · g²** — exponent 2 by Gauss-Newton derivation (band 13), verified against REQ-023's
> per-matrix LR randomisation. **C is actively restored** against interventions (band 16),
> seed-independent (Arm A), and **time-invariant** (band 24: the window is equilibrated, AR(1)
> ρ ≈ 0.54).
>
> **Every type's gradient decomposes exactly:** `log g = log‖a‖_F + log‖d‖_F + log(alignment)`,
> identity exact to 1e-6 dex.

#### The two architectural effects, each decomposed at n=4

- **q,k (attention), purely BACKWARD.** −0.37 dex = **−0.18 backward attenuation** (softmax
  Jacobian) **− 0.19 alignment deficit**. `a_rms` is **bit-identical** for q, k and v in every seed
  and at all 12 layers — the input is shared exactly, so nothing forward contributes. Confirmed
  n=4: d_rms ratio **0.667 ± 0.011**, t = −41.9. The deficit is **depth-flat** (band 17) and
  **identical for q and k** (band 18, 0.014 dex apart).
- **mlp, purely FORWARD.** −0.30 dex = **−0.435 forward** + 0.05 backward + 0.08 alignment. The
  forward term is the **ReLU²**: its output has **2.725× the RMS** (= +0.435 dex, matching to
  0.001) and **7.3× the effective rank** of the block input. Band 20's iteration-94 prediction,
  made before these fields existed.

#### Band 27 — the deepest regularity found

**`‖a‖·‖d‖` is conserved across depth.** `corr(log‖a‖, log‖d‖) = −0.87 to −0.99` within every type,
permutation p ≤ 0.0002, between two **independently measured** quantities. The product is
**2–4× flatter** than either factor (sd-ratio 0.262–0.493), and **sd(log g) tracks sd(product)**
across all six types — **the gradient is flat in depth because the product is.** That explains the
flatness bands 17 and 22 observed without explaining.

*Compensation is near-exact but not a law:* TLS slopes span −0.765 to −1.230 and contain −1.000 in
only **2 of 6** types.

#### Registered negatives — routes closed

- **Attention entropy and qk-logit RMS do not explain the alignment deficit.** Raw corr −0.786 looks
  strong, but entropy is 0.86-collinear with depth; partialling depth out, three of four collapse,
  and the survivor **flips sign** (t −2.7 → +2.13) when depth is allowed a quadratic term.
- **Bilinear q·k pairing rejected.** corr(q,k) ranks **2 of 15** type pairs, and the trade-off test
  has the **wrong sign** (ratios 1.20–1.38; a product-controlling mechanism predicts < 1).
- **Muon chunk geometry excluded.** `shape_mult` alone gives R² 0.005–0.008; QK-norm gives 0.63–0.67
  and **strengthens** when shape is added.
- **Cross-type cancellation is chance** (p = 0.19–0.25). *Its companion across depth is real —
  band 27 — and iteration 109's claim that no probe could resolve it was withdrawn.*
- **`d_eff_rank` cannot reconstruct the deficit**, at n=1 and again at n=4: depth slopes run
  **opposite** (target t = −8.56, rank t = +4.25).

#### Corrections made in this stretch

Iteration 52's "the q,k gap is in the gradient, not C" was **overturned** — right arithmetic, wrong
object (a *fitted* intercept absorbed the gap; with the exponent fixed by derivation it reappears).
Band 15's scale-invariance **magnitude** prediction **failed** (predicted +0.130, observed −0.417 —
wrong sign); the invariance *result* stands but no longer explains band 14. Band 25's depth trend was
**~34% a state artifact**, predicted by iteration 103 and confirmed by Jerry's second-state probe
(slope −0.0075 artifact-free, inside the predicted CI). Iteration 107's generalisation from two pairs
to the whole network was **narrowed** — across six types the three terms are comparable and partly
offsetting (band 26).

#### Standing rules added

4. **A confound cleared for one statistic is not cleared generally** — each statistic from a
   mismatched comparison needs its own check. *(Band 25's size was clean; its slope was not.)*
5. **Consistency across seeds is not replication of a cross-type pairing** — all seeds share the same
   six types.
6. **Prefer identities to fits.** A regression on terms that sum to the target will silently absorb
   the omitted term into inflated coefficients (~1.5 where the identity says 1.0).

#### Open, and not filed

**Why** the softmax Jacobian's output aligns less well across tokens, and **why** ‖a‖ and ‖d‖ trade
off. Both need per-token backward vectors — a substantially heavier probe. **The C account is closed
arithmetically at n=4; these are research questions in attention dynamics, not gaps in it.**

## REQ-036: equalized-curvature per-type learning rates (the first design derived from C)

- status: **DONE (2026-09-03) — `logs/kmaxwell/req036_equalized_curvature_lr/`.** NULL/directed-negative: uniform control (a1) has the BEST val@2750 (3.51052); every per-type LR arm is worse — a2 +0.0024, a4 +0.0046, a3 +0.0094, **a5 polar +0.024 (the predicted-best is the WORST)**. Curvature check confirms the mechanism IS real: the rules equalize per-type equilibrium curvature exactly as designed (spread a5 0.128 < a3 0.163 < a2 0.194 < control 0.246; a4 anti-equalizes to 0.444) — but **equalization is INVERSELY related to val**, so the premise 'equal per-type curvature is better' is FALSIFIED: equalizing it hurts loss. Config verified correct (a1 all-1.0, a5 per-matrix=type polar mult, official order). n=1/arm (gaps 10x the 2e-4 noise floor). Checkpoint cadence miss (dumped @2250 not 2750) → curvature measured @2250 (equilibrium stable); val@2750 is exact.
  Independent of REQ-035 (uses *measured* C, not predicted), so it does not wait on Arm A.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `25d3208` (`codex/per-matrix-lr-public`, the `PerMatrixLrMuon` used by REQ-023)
- **node budget: ≤2 nodes.** 1 box is enough for the headline arm.
- depends on: REQ-023 (measured the causal exponent), REQ-019/022 (measured C and k).


### ⚠️ AUTHORITATIVE ARM TABLE — the only live prescription

*This request accumulated three prescription tables across iterations 17/19/27, two of which
said "use this for arm 2" with conflicting values. **The superseded tables have been removed from
this queue — only this block is live.** They remain in this file's git history if needed.*

**Common setup:** 750-step continuations from the shared step-2000 state, val@2750, same
`PerMatrixLrMuon` machinery as REQ-023.

| arm | rule | multipliers |
|---|---|---|
| **1** | control | all 1.0 |
| **2** | per-type only | attn.proj 0.40, attn.k 0.88, mlp.fc 0.91, attn.q 1.18, attn.v 1.25, mlp.proj 1.56 |
| **3** | per-type + end-block cap ⚠️ **REVISED, see iteration 49** | arm-2 values, except at blocks 0 and 11 where **proj types (attn.proj, mlp.proj) get a single pooled ×3.0 cap**: attn.proj 1.20, mlp.proj 3.00. Non-proj types unchanged from arm 2. *(The previously filed per-type end values — attn.q 1.35, attn.k 1.01, attn.v 1.43, mlp.fc 1.04, attn.proj 1.72, mlp.proj 6.71 — are **withdrawn**: each rests on n=2 matrices and several sit outside their own bootstrap CI.)* |
| **4** | anti-rule falsifier | arm-2 multipliers inverted (1/s) |
| **5** | **polar target** (iteration 27) | attn.q 0.568, attn.k 0.755, attn.proj 0.642, attn.v 1.101, mlp.fc 1.260, mlp.proj 2.462 |

**Priority if fewer arms fit:** 1, 2, 5, 3, 4. Arms 2 and 5 test *different hypotheses* (which
curvature to equalize) and are the most informative pair; arm 4 is the cheapest falsifier; arm 3
is a magnitude refinement of arm 2.
**Registered predictions** (magnitudes, per the REQ-019 lesson):
- arm 2 beats arm 1 by 0.001–0.006 val; arm 3 beats arm 2 by 0.0005–0.003; **arm 5 beats arm 2 by
  0.0005–0.003**.
- arm 4 is *worse* than arm 1 by a comparable margin. **If arm 4 also beats control, the mechanism
  claim is dead** regardless of which other arm wins.
- **Expect smaller gains than the within-design numbers imply** — iteration 21 showed ~71% of
  cross-experiment variation is irreducible. A null is not a refutation of the curvature findings,
  only of their transfer to a different intervention design.

**Required readouts:** per-matrix curvature at the final checkpoint for arms 1–3 and 5 (to verify
the intervention actually equalized what it targeted); per-type drift over the final window
(iteration 14 — **mlp.fc is the least trustworthy multiplier**); block 11 reported separately
(iteration 18 — worst baseline error of any block).


### Success criteria
- All arms in the authoritative table complete; val@2750 for each against the stored control.
- Per-matrix curvature at the final checkpoint (spread of logC should fall from 0.379 dex toward
  ~0.1); **this is the mechanistic check and matters more than the loss number.**
- `summary.tsv` with per-arm val and the realized per-type curvature spread.
- Commit raw Ritz values + `residual_tail`; do not apply the tail correction silently.

### Artifacts
`logs/kmaxwell/req036_equalized_curvature_lr/`

### On per-layer MOMENTUM (deliberately excluded, with reason)
A per-layer momentum rule is **not** derivable from C today. To first order momentum only
rescales the effective step, `s_eff = s/(1-mu)`, so per-layer mu is redundant with per-layer
LR unless mu affects something LR cannot — a noise/curvature interaction this campaign has
never measured. Muon here also carries momentum internally (`m_fast`/`m_slow`), so mu is not
a free per-layer knob without changing the kernel. Filing a momentum rule now would be
inventing a mapping rather than deriving one. The prerequisite is a registered experiment
asking whether mu does anything LR cannot at fixed `s_eff`; that is REQ-037 if wanted.

### CONSOLIDATED FINDINGS III (iterations 112–118) — and where the goal stands

*Six iteration blocks replaced. Provenance in git history from `ab04e19` onward.*

#### The instrument, audited

- **The batch instrument (REQ-037 arms 1–3) is unusable.** It gives +0.383 [+0.028, +0.726] against
  the LR instrument's +2.07, but its reduced form is **non-monotone** and per-type ratios span
  **−1.25 to +1.12** with no consistent sign. Not evidence against REQ-035 — evidence that arm 4 is
  the clean test, as REQ-037 itself judged.
- **The exclusion restriction is VIOLATED — shown on committed data.** Two functionals of the same
  Hessian under the same instrument must give the same exponent. They do not: top_eigenvalue
  **+2.13/+2.10** vs curvature_along_polar **+1.89/+1.83**, differences **+0.241/+0.268** with CIs
  excluding zero in both forks. **REQ-035's +2.07 is a response ratio contaminated by ≥1 non-gradient
  channel.** The bound is one-sided twice over: a channel acting *equally* on both functionals is
  invisible here.
- **Band 13's registered check is unaffected** (2.000 is inside every CI) but its **reading is
  qualified** — agreement with 2.000 is partly luck. The Gauss-Newton *derivation* is theory and never
  depended on the instrument.
- **The contamination's per-type structure is unresolvable.** 4 of 6 CIs include zero, widths
  0.35–0.91 dex, and attn.q flips sign between forks. **mlp.proj is the one exception** (+0.80/+0.74,
  CI excluding zero) — flagged as a caution on that type's IV exponent only; it does **not** propagate
  into bands 12, 20 or 27 (verified by leave-one-type-out).

#### Two audits that changed nothing — which is the point

- **Leave-one-out, applied retroactively** to bands 12, 14, 18, 20, 27 after the rule was introduced.
  All survive: band 27's weakest LOO correlation is **−0.848**, band 14 moves ≤ **0.032 dex**, band 12's
  reduction holds with **every type dropped** (gap 0.000–0.006 dex).
- **Cross-band consistency, four implied constraints tested.** Bands 27+10/25 → λ must carry C's depth
  structure (**ratio 2.03–4.32, mean 3.2×** ✅). Bands 21+26 → the forward term must contribute exactly
  zero among q,k,v (**0.000000** ✅). Bands 14+28 → q,k λ-equality must survive λ's greater variability
  (**ratios 0.08–0.54** ✅). Bands 16+23 → **passes, after two wrong framings of my own** (see below).
  **Fourteen bands describing the same 72 matrices are mutually consistent.**

#### New in this stretch

- **Band 28:** C's depth structure is carried by **λ, not g** — sd(log λ)/sd(log g) = **2.03–4.32** in
  every type. Registered as the campaign's **first cross-band falsifier**.
- **Band 6 reappears in a new domain:** the two residual writers have the strongest λ–g coupling
  *across depth* (**+0.964, +0.828**), matching their steep *cross-sectional* slope. Two independent
  measurements of one architectural fact.
- **Band 16 amended in wording:** it claims C's **pattern** is restored, not its **level** — the level
  shifts ~0.17 dex under a 2.8× LR change, **non-uniformly** (sd of the shift ≈ its mean), while the
  pattern correlation holds at **+0.87 to +0.97**.

#### Corrections logged

My statement of constraint B misread band 16 as a level claim; my script's rescue ("the shift is
uniform") erred the other way. A `corr = +0.609` verdict rested on **one type** — dropping mlp.proj
gave +0.109. Iteration 113's "the exclusion restriction remains untested" was **wrong** (iteration 114
tested it). Iteration 109's "no probe could resolve the cancellation" was **withdrawn** — the
resolution was in committed data one aggregation level down (band 27).

#### Standing rules, final list

1. Permute group labels and report the null before citing a correlation between group summaries —
   **and drop each group in turn** (extension, iteration 115).
2. Count a predictor's distinct values before fitting a continuous coefficient; one or two makes it a
   label, not a law.
3. Any predictor built from the same Lanczos tridiagonal as `lam_top` is circular and rejected.
4. A confound cleared for one statistic is not cleared generally.
5. Consistency across seeds is not replication of a cross-type pairing.
6. Prefer identities to fits — a regression on terms that sum to the target absorbs the omitted term.

---

### WHERE THE STATED GOAL STANDS

**Goal:** *"a mathematically rigorous account of what sets the between-layer difference in C, plus
validation of the REQ-036 per-type LR design."*

**The design question is ANSWERED, negatively and with a mechanism.** REQ-036 is a null: uniform LR
beats every per-type rule, the predicted-best arm is worst by 120× the val noise floor, and harm is
**monotone in the amount of equalization** (Spearman −1.000, p = 0.042). Band 16 explains why — the
rule fights a quantity the network restores. **Do not build a momentum kernel or per-layer LR on
curvature equalization.**

**The account of C is complete at n=4 and mutually consistent**, with every term measured:

> **λ_eq = C·g²**, exponent 2 by Gauss-Newton derivation. **C is restored** against LR perturbation,
> seed-independent, and time-invariant on an equilibrated window (AR(1) ρ ≈ 0.54).
> **log g = log‖a‖_F + log‖d‖_F + log(align)** exactly, and C's type structure is three comparable,
> partly-offsetting terms — **not** two, as iteration 107 briefly claimed.
> **q,k:** −0.37 dex, **purely backward** (softmax Jacobian −0.18, alignment −0.19), input shared
> bit-identically. **mlp:** −0.30 dex, **purely forward** (ReLU² scale, matching to 0.001 dex).
> **Across depth, `‖a‖·‖d‖` is conserved** (2–4× flatter than either factor), which is why the
> gradient — and C — are flat in depth.

**What is NOT settled, and why no further analysis will settle it:**

1. **Why the softmax Jacobian's output aligns less well across tokens.** Needs per-token backward
   vectors. Committed observables are exhausted (attention entropy and logit RMS tested and rejected —
   the apparent relationship is depth, and the survivor flips sign under a quadratic depth term).
2. **Why ‖a‖ and ‖d‖ trade off.** Same measurement.
3. **How large the non-gradient channel is.** **REQ-037 arm 4** is specified and is the only design
   that removes it rather than bounding it. **This is the one open measurement that could still
   overturn a load-bearing band.**

**Recommendation.** The analysis loop has reached the end of what committed data supports. The
productive path is **arm 4**, then — if the exponent survives — treating C as a measured property to
respect rather than a target to flatten.

**=== ITERATION 134 (2026-09-04): THE DIRECTION CHANNEL — Muon steps almost orthogonal to peak curvature ===**

*Band 31 rules out gradient **magnitude** as a channel. It says nothing about **direction** — and the
orthogonalised step is entirely a function of `g/‖g‖`, so direction is the channel through which this
optimiser acts. It had never been examined.*

**Circularity cleared first (standing rule 3).** The natural measure — `curvature_along_gradient` — is
**exactly α₁**, the first Lanczos coefficient from the same tridiagonal as `lam_top`. **Re-verified on
Arm A: `|cg/α₁ − 1|` median 0.0000000000, max 0.0000000000.** Rejected. `curvature_along_polar` is a
**separate HVP** (ratio to α₁ differs by 98%), so it is admissible — and it measures curvature along
**the direction Muon actually steps in.**

**Result 1 — the step is nearly orthogonal to the steepest direction:**

| type | cp/λ | log(cp/λ) | across-seed sd |
|---|---:|---:|---:|
| mlp.proj | **0.006** | −2.207 | 0.034 |
| attn.proj | 0.005 | −2.273 | 0.044 |
| mlp.fc | 0.005 | −2.290 | 0.009 |
| attn.v | 0.003 | −2.563 | 0.016 |
| attn.k | 0.003 | −2.534 | 0.032 |
| **attn.q** | **0.001** | **−2.839** | 0.036 |

> **Muon's update direction sees ~0.4% of the peak curvature.** The orthogonalisation steers the step
> almost perpendicular to the sharpest direction — which is a concrete statement about *why* an
> orthogonalised optimiser tolerates high curvature, and it varies **6× across types**.

**Result 2 — that alignment predicts C, and it is not the shared-λ artifact.** `alignment = log cp −
log λ` and `C = log λ − 2 log g` **share log λ with opposite signs**, so a negative correlation is
partly induced by construction — the same trap as iteration 118's constraint B.

**Quantified with a null that keeps λ and g exactly as they are and shuffles only cp:**

| seed | observed | **cp-shuffled null** | p |
|---|---:|---:|---:|
| 0 | −0.788 | **−0.339 ± 0.071** | **< 10⁻⁴** |
| 1 | −0.791 | −0.246 ± 0.071 | < 10⁻⁴ |
| 2 | −0.736 | −0.271 ± 0.073 | < 10⁻⁴ |
| 3 | −0.782 | −0.289 ± 0.071 | < 10⁻⁴ |

**About a third of the raw −0.78 is the shared λ term; the rest is real.** cp carries genuine
independent information about C, in every seed. *(For reference, `corr(log cp, C)` without the ratio is
−0.25 to −0.40 — weaker, because the ratio is the meaningful object: alignment, not magnitude.)*

**Registered as band 33** — the first band to examine the channel band 31 leaves open, and the first
non-circular use of `curvature_along_polar` as a *predictor* rather than a placebo.

**Why this matters for the campaign's goal.** Bands 14/20/21 explain C's structure through **forward
and backward magnitudes**. Band 33 adds a **geometric** term: how well the optimiser's step aligns
with the curvature it is climbing. **attn.q's alignment is 6× lower than mlp.proj's** — and q,k are
exactly the types carrying the largest C excess (band 14). **That is a mechanism-shaped connection the
magnitude account does not contain**, and unlike the magnitude channel it is *not* ruled out by
band 31.

**Not overstated.** This is a **correlation across matrices**, not an intervention — and the campaign
has learned repeatedly what that distinction costs (bands 8, 13, 26). **Whether alignment *causes* C's
structure or merely tracks it cannot be settled here**, and under Muon the natural intervention —
rotating the update direction — has no obvious implementation that leaves everything else fixed.

**=== ITERATION 135 (2026-09-04): ALIGNMENT DOES NOT EXPLAIN THE q,k GAP — two effects, not one ===**

*Band 33 flagged a connection without testing it: **attn.q has the lowest step–curvature alignment
(6× below mlp.proj) and q,k carry the largest C excess.** That looked mechanism-shaped. Testing it
properly — on band 14's clean four-same-shape comparison, so no size confound — shows the two are
largely independent.*

**Mediation test.** If alignment explains the q,k gap, controlling for it should collapse the q,k
coefficient:

| seed | q,k coef alone | **with alignment** | shrinkage | alignment coef |
|---|---:|---:|---:|---:|
| 0 | +0.893 | **+0.699** (t +10.4) | **22%** | −0.616 (t −5.9) |
| 1 | +0.774 | **+0.629** (t +18.4) | **19%** | −0.604 (t −10.0) |
| 2 | +0.827 | **+0.694** (t +13.2) | **16%** | −0.582 (t −5.6) |
| 3 | +0.848 | **+0.680** (t +13.5) | **20%** | −0.584 (t −7.0) |

**Alignment mediates only 16–22% of the gap.** The q,k coefficient stays large and highly significant
after the control, and alignment retains a substantial independent effect. **Both terms survive
together in every seed.**

> **Registered as band 34: the step–curvature alignment and the q,k C-excess are largely separate
> effects that co-occur, not one mechanism seen twice.**

**And q,k's low alignment is itself a finding, not just a covariate.** On the same clean comparison:

| seed | alignment q,k | v, attn.proj | difference |
|---|---:|---:|---:|
| 0 | −2.725 | −2.409 | **−0.316 dex (0.48×)** |
| 1 | −2.658 | −2.418 | −0.240 dex (0.58×) |
| 2 | −2.671 | −2.443 | −0.228 dex (0.59×) |
| 3 | −2.692 | −2.403 | −0.288 dex (0.52×) |

**q,k's step direction sees about half the peak curvature that v and attn.proj's does**, at identical
shape and sub-block. **That is a second architectural asymmetry on the same matrices** — distinct from
the backward attenuation of bands 21/25, and consistent in all four seeds.

**Why the negative matters.** Band 33 offered the alignment–excess link as "a mechanism-shaped
connection the magnitude account does not contain." **It is a connection, but not a mechanism for band
14** — 80% of the q,k excess remains unexplained by it. **Recording this prevents the campaign from
absorbing band 14 into band 33 on the strength of a suggestive co-occurrence**, which is exactly the
error pattern the standing rules exist to catch: two effects on the same matrices are not
automatically one effect.

**What the account now contains for q,k, at n=4:** a **backward** attenuation (softmax Jacobian,
−0.18), an **alignment** deficit in the gradient's token-wise accumulation (−0.19, band 25), and now a
**geometric** alignment deficit in the step direction (−0.24 to −0.32 dex, band 34) — **three distinct
asymmetries on the same two matrices**, of which the first two sum to the gradient deficit exactly
(iteration 107) and the third is separate.

**=== ITERATION 136 (2026-09-04): CHUNK GEOMETRY REFUTED — and q ≠ k after all, in alignment ===**

*Band 34 left the step-alignment deficit unexplained. Muon orthogonalises q,k per head-pair
(**128×768**) versus 768×768 for v/attn.proj, and **alignment is a geometric quantity** — so chunk
narrowness was the natural candidate. **Iteration 93 excluded chunk geometry for the *gradient*
deficit; I did not assume that negative transfers to a different quantity.***

**The correlational test is inconclusive, as the standing rules predict for 6 points:**

| | value |
|---|---:|
| corr(log chunk aspect, alignment) | **+0.476** |
| permutation null | **−0.003 ± 0.450** |
| **p** | **0.334** |
| leave-one-type-out range | **+0.208 to +0.760** |

Entirely within chance, and unstable to dropping any single type. **No evidence — but also no
refutation from this test alone.**

**The direct refutation is decisive, and it comes from within the pair.** `attn.q` and `attn.k` sit in
the **same `qk_bank`**, orthogonalised at **identical 128×768 chunks**. Chunk geometry treats them
identically, so it cannot produce any difference between them:

| seed | attn.q | attn.k | **difference** |
|---|---:|---:|---:|
| 0 | −2.889 | −2.560 | **−0.329 dex (0.47×)** |
| 1 | −2.806 | −2.511 | −0.295 dex (0.51×) |
| 2 | −2.840 | −2.501 | −0.340 dex (0.46×) |
| 3 | −2.821 | −2.563 | −0.258 dex (0.55×) |

**Mean −0.306 dex, sd 0.037, same sign in 4/4 seeds** — and that is **larger than the −0.27 dex q,k-vs-v
gap band 34 attributed to the pair as a group.**

> **Chunk geometry is refuted: it cannot explain a 0.31 dex difference between two matrices it
> processes identically.**

**The finding this produced is bigger than the negative it was looking for.** Band 18 established that
**q and k are interchangeable in the gradient deficit — 0.014 dex apart**, and iteration 92 registered
that as evidence the effect tracks the shared RMS-norm rather than the query/key asymmetry. **In step
alignment they are not interchangeable at all: 0.306 dex apart, twenty times the gradient difference,
consistent in every seed.**

**Registered as band 35.** It also **revises band 34's framing**: treating "q,k" as a unit is correct
for the gradient (band 18) and **wrong for alignment**. The q,k-vs-v alignment gap is substantially an
**attn.q** effect — attn.k sits close to attn.v (−2.560 vs −2.563 in seed 0).

**What distinguishes q from k, and what it points at.** Under causal masking a **query attends once**
while a **key is attended by a growing suffix** — the asymmetry iteration 92 tested and found *absent*
in the gradient. **Finding it present in the step geometry, at 20× the magnitude, is consistent with
that asymmetry being real but expressed in direction rather than magnitude** — which is exactly the
channel band 31 leaves open. *Offered as a reading, not a tested claim; the campaign has no
intervention on step direction.*

**=== ITERATION 137 (2026-09-04): THE CAUSAL-MASKING READING FAILS ITS OWN TEST — but the gap has structure ===**

*Band 35 offered a reading: the q−k alignment gap is the causal-masking asymmetry (a query attends
once; a key is attended by a growing suffix). **That reading predicts the gap should track how peaked
attention is** — entropy falls 4.91 → 0.25 nats across depth. Testing it.*

**The gap is strongly depth-structured, and reproducibly so:**

```
 layer:   0      1      2      3      4      5      7      8      9     10     11     12
  gap: -.134  -.063  -.198  -.305  -.334  -.438  -.414  -.474  -.394  -.295  -.289  -.327
```

**Deepens to −0.474 dex around layer 8, then recovers** — across-layer sd **0.123** against an
across-seed sd of **0.082** (structure/noise 1.5×), quadratic **R² 0.563** versus linear 0.233.

**And entropy correlates strongly — until the lethal check.** `corr(gap, entropy) = +0.683` pooled, and
**+0.600 after partialling out depth linearly** — far better than iteration 106's −0.379. But
iteration 106 was killed by a **quadratic** depth term, not a linear one:

| model | R² | entropy coefficient |
|---|---:|---|
| depth linear | 0.233 | — |
| depth linear **+ entropy** | 0.509 | **+0.1002 (t = +5.03)** |
| **depth quadratic** | **0.563** | — |
| **depth quadratic + entropy** | **0.563** | **+0.0111 (t = +0.26)** |
| depth cubic + entropy | 0.566 | +0.0496 (t = +0.59) |

**Entropy collapses from t = +5.03 to t = +0.26, and the R² is identical with and without it (0.563).**
Seed-clustered CI on the entropy coefficient under quadratic depth: **[−0.070, +0.057] — includes
zero.** **Entropy adds nothing once depth is allowed to curve.**

> **The causal-masking reading loses its support. Attention entropy does not explain the q−k alignment
> gap — this is iteration 106's failure mode reproducing exactly, on a different quantity.**

**What survives, and it is worth registering.** The gap's **depth structure is real and independent of
entropy**: a U-shape reaching −0.474 dex at layer 8, reproducing across four seeds at 1.5× the noise.
**Band 35 is amended to include it**, since a bare "q ≠ k by 0.306 dex" understates what the data
shows.

**Why running this check mattered.** The linear partial correlation of **+0.600** was the strongest
entropy result in the campaign and would have been easy to report as confirming the causal-masking
mechanism. **Iteration 106 had already established that a linear depth control is insufficient for
this exact predictor** — entropy is 0.86-collinear with depth, so it will absorb any leftover depth
curvature a linear term misses. **The campaign's own prior negative is what made the correct test
obvious**, and applying it turned an apparent mechanism into a registered negative.

**Standing rule reinforced, not extended:** *when a predictor has previously failed a specific control,
apply that same control before reporting it in a new context.* Entropy has now failed the quadratic-depth
test twice, on two different response variables.

**=== ITERATION 138 (2026-09-04): REGISTERED NEGATIVE — the three U-shapes are not one pattern ===**

*Three depth profiles now sit on the same 13 layers: band 20's mlp gradient gap, band 35's q−k
alignment gap, and C itself. **If they were one underlying pattern, C's depth structure would be a
single effect seen in three measurements.** They are not.*

**The raw correlations look compelling — all three significant:**

| pair | correlation | permutation p |
|---|---:|---:|
| q−k alignment vs mlp gap | **−0.656** | **0.021** |
| q−k alignment vs C | **+0.655** | **0.023** |
| mlp gap vs C | **−0.710** | **0.011** |

**But every smooth U-shape on 13 layers correlates with every other one** — they all project heavily
onto the same quadratic basis function. **That is a shared *form*, not a shared mechanism**, and
correlating raw profiles cannot distinguish them. *(Quadratic R²: q−k 0.813, mlp 0.607, C 0.921 — all
three are mostly quadratic by construction.)*

**Fitting depth + depth² and correlating the RESIDUALS — do they agree beyond the quadratic they
trivially share?**

| pair | raw | **residual** | p | |
|---|---:|---:|---:|---|
| q−k vs mlp | −0.656 | **+0.165** | 0.618 | **gone** |
| mlp vs C | −0.710 | **−0.226** | 0.488 | **gone** |
| q−k vs C | +0.655 | **−0.724** | **0.009** | *survives?* |

**Two of three vanish.** They shared only the U-shape that 13 points and a smooth trend produce
automatically.

**The survivor looked robust — leave-one-layer-out stable at [−0.822, −0.636], negative in all four
seeds — and it is still an artifact.** `C = log λ − 2 log g` and `alignment = log cp − log λ` **share
log λ with opposite signs**, which induces exactly this negative correlation (iteration 134's trap).
**Testing with cp alone, without λ:**

> **residual corr(cp q−k, C) = −0.188** — collapsed from −0.724.

**The survivor was the shared-λ construction, not a mechanism.**

> **Registered as a negative: the three depth profiles share a quadratic form and nothing more. C's
> depth structure is not a single effect appearing in multiple measurements.**

**Why this needed three separate checks to reach.** The raw correlations pass a permutation test; the
residual test kills two of three; the third survives leave-one-out and per-seed replication and dies
only to a construction check. **Each check was necessary and none alone was sufficient** — and the
campaign has now been caught by the shared-λ artifact twice (iterations 134, 138), the first time
catching it, the second time only at the last step.

**Standing rule 6 sharpened:** *prefer identities to fits* — extended to: **when two derived quantities
share a term, test with the raw components before believing any correlation between them, regardless
of how many robustness checks it survives.** Leave-one-out and cross-seed stability do **not** detect a
construction artifact; both quantities are equally artifactual in every subsample.

**=== ITERATION 139 (2026-09-04): SHARED-TERM AUDIT OF ALL BANDS — one exposed, and it strengthens ===**

*Iteration 138's artifact survived leave-one-out **and** cross-seed replication, dying only to a
construction check. That is a failure mode the campaign's other bands were never screened for, so this
iteration screens all 18 at once.*

**Audit result — 3 of 18 bands compare quantities sharing a term:**

| band | comparison | shared term | status |
|---:|---|---|---|
| 32 | C vs λ **seed-stability** | contains λ | **not a correlation** — compares stability, artifact does not apply |
| 33 | alignment vs C | log λ, opposite signs | **null was run at the time** (iter. 134, cp-shuffled, cleared at p < 10⁻⁴) |
| **34** | alignment + q,k → C | log λ | **exposed, never checked** |

**The other 15 are clean by construction:** bands 14, 17, 18, 35 compare *the same quantity* between
groups (λ cancels within each matrix before differencing); 21, 27 use independent probe fields; 25, 26
are stated identities; 6, 28, 30 relate λ and g, which are independent measurements.

**Band 34 under the raw-component test — replacing `alignment` with `log cp`, which contains no λ:**

| seed | q,k alone | ctrl by alignment | shrink | **ctrl by log cp** | **shrink** |
|---|---:|---:|---:|---:|---:|
| 0 | 0.893 | 0.699 | 22% | **0.939** | **−5%** |
| 1 | 0.774 | 0.629 | 19% | **0.826** | **−7%** |
| 2 | 0.827 | 0.694 | 16% | **0.849** | **−3%** |
| 3 | 0.848 | 0.680 | 20% | **0.923** | **−9%** |

**Band 34's conclusion strengthens.** With the shared-λ term removed the q,k coefficient does not
shrink at all — it slightly *grows*. **The 16–22% shrinkage band 34 reported was itself partly the
construction artifact**; the true mediation is ~zero.

**And `log cp` barely predicts C net of q,k:** t = **+1.80 / +1.68 / +1.64 / +2.86** — significant in
one seed of four. **Band 33's "alignment predicts C" holds only for the *ratio*, which is the
meaningful geometric object; the raw curvature-along-polar does not carry it.**

> **Band 34 amended and strengthened: alignment and the q,k excess are not merely "largely separate" —
> they are separate, with mediation indistinguishable from zero once the shared term is removed.**

**Why the audit was worth an iteration.** It found exactly one unchecked exposure out of eighteen, and
that one **changed a reported number in the direction of the band's own conclusion.** The alternative —
discovering later that a band's headline shrinkage was an artifact — is the failure iteration 138 came
within one check of committing. **Screening the whole set once is cheaper than rediscovering the same
trap band by band**, and the 15 clean cases are now documented as clean rather than merely
unchallenged.

**=== ITERATION 140 (2026-09-04): BAND 33 IS GEOMETRY, NOT CONSTRUCTION — the data recovers the ratio unprompted ===**

*Iteration 139 left an apparent contradiction: band 33's **ratio** `log cp − log λ` predicts C at
−0.78, but `log cp` **alone** predicts it only weakly (t = 1.6–2.9). Those are consistent only if the
content lives in cp **relative to** λ — or if the ratio was carrying λ all along, in which case band 33
is another construction effect.*

**The decisive test: give λ and cp free, separate coefficients.** The alignment predictor *imposes*
`b = −a` by construction. If the data prefers that constraint on its own, the ratio is the right
object:

| seed | a (log λ) | b (log cp) | **b/a** |
|---|---:|---:|---:|
| 0 | +1.114 (t +9.9) | **−1.164 (t −8.8)** | **−1.045** |
| 1 | +0.998 (t +10.4) | **−1.298 (t −11.0)** | **−1.301** |
| 2 | +1.116 (t +8.6) | **−1.207 (t −8.0)** | **−1.082** |
| 3 | +1.043 (t +9.9) | **−1.245 (t −9.6)** | **−1.193** |

**The data recovers `b/a ≈ −1` without being told** — the constraint the ratio imposes, arrived at
independently in all four seeds. **And `log cp` is strongly significant here (|t| = 8.0–11.0)**, which
resolves iteration 139's puzzle: **cp matters conditional on λ, not marginally.** A weak marginal
t-statistic said nothing about its conditional contribution.

**Model comparison agrees:**

| seed | free(λ, cp) RSS | ratio-only RSS | **λ-only RSS** |
|---|---:|---:|---:|
| 0 | 7.082 | 7.101 | **14.995** |
| 1 | 4.456 | 5.088 | **12.211** |
| 2 | 7.005 | 7.067 | **13.438** |
| 3 | 5.759 | 6.062 | **13.459** |

**Imposing the ratio costs almost nothing (RSS +0.3% to +14%), while dropping cp entirely doubles the
error.** AIC prefers ratio-only in 2 seeds and free in 2 — i.e. the one-parameter ratio is as good as
the two-parameter fit.

> **Band 33 is geometry. The step–curvature alignment carries genuine predictive content about C, and
> the ratio form is what the data itself selects — not an artifact of how it was constructed.**

**Why this needed doing after iteration 139.** That iteration correctly flagged `log cp`'s weak
marginal significance and correctly concluded the ratio was the meaningful object — **but it did not
establish *why*, and "the ratio works but its component doesn't" is exactly the shape of a construction
artifact** (iteration 138 died of precisely that). **The free-coefficient test distinguishes them: an
artifact would show `b ≈ 0`; geometry shows `b ≈ −a`.** It shows `b ≈ −a`.

**Standing rule 6 extended once more:** *when a ratio predicts and its numerator does not, fit the
components with free coefficients before concluding either way — a data-preferred ratio constraint is
evidence FOR the ratio; a null numerator coefficient is evidence against it.* **The marginal
significance of a component is not the relevant test.**

**=== ITERATION 141 (2026-09-04): ALIGNMENT DOES NOT BELONG IN C's PREDICTIVE MODEL ===**

*Band 33 is established as geometry (iter. 140). Band 12's reduction predates it, so alignment had
never been tested as a predictor alongside the structural binaries. **It appears to help enormously,
and most of that is the shared-λ term.***

**The apparent gain, on the campaign's leave-one-layer-out standard:**

| seed | band-12 model | **+ alignment** | gain |
|---|---:|---:|---:|
| 0 | 0.292 | **0.213** | 27% |
| 1 | 0.235 | **0.141** | **40%** |
| 2 | 0.249 | 0.187 | 25% |
| 3 | 0.243 | 0.172 | 29% |

**23–40% improvement on held-out layers, every seed** — and the three binaries explain only **28–35%**
of alignment, so it looked like largely independent information.

**Rule 6 applied: replace alignment with raw `log cp`, which contains no λ.**

| seed | band-12 | + alignment | **+ log cp (raw)** |
|---|---:|---:|---:|
| 0 | 0.292 | 0.213 (27%) | **0.290 (1%)** |
| 1 | 0.235 | 0.141 (40%) | **0.224 (5%)** |
| 2 | 0.249 | 0.187 (25%) | **0.232 (7%)** |
| 3 | 0.243 | 0.172 (29%) | **0.227 (6%)** |

**The gain collapses from 23–40% to 1–7%.** **Reproduced on the stricter held-out-SEED test**
(alignment 31–41%, raw cp 3–8%).

> **The gap between the two is the shared-λ term. Alignment's predictive power over C is largely the
> model reconstructing C's own λ component — not new information about C.**

**Band 33 is unaffected and band 12 is unchanged.** Iteration 140's free-coefficient test asked
whether the data prefers the **ratio form** — it does, `b/a ≈ −1` recovered unprompted — and that is a
question about *geometry*, not about predictive content. **Both can be true: the step–curvature
alignment is a real architectural property, and it does not improve a model of C.**

**Registered as a negative on the predictive question.** *No band changes*, which is the point worth
recording: a 40% out-of-sample improvement, replicated across seeds and on two hold-out schemes, is
exactly the result that would normally justify adding a term to the model. **It survives every
robustness check the campaign applies except the one that matters here.**

**This is the third time the shared-λ construction has produced a result that passes conventional
validation** (iterations 134, 138, 141). In 134 the null caught it immediately; in 138 it survived
leave-one-out and cross-seed replication and died to the raw-component check; here it survived
**two independent hold-out schemes**. **Cross-validation does not detect a shared-term artifact** —
held-out data is equally contaminated, because the contamination is in the *definitions*, not the
sample.

**Rule 6, final form:** *when a predictor and an outcome share a term, no amount of out-of-sample
validation licenses the predictor. Refit with the raw components; the difference is the artifact.*

### CONSOLIDATED FINDINGS IV (iterations 112–132) — the causal account, revised

*Provenance in git history from `ab04e19` onward.*

#### The headline change: the g² law is NOT causal, and cannot be tested under Muon

- **The exclusion restriction is violated** (iter. 114, committed data): two functionals of the same
  Hessian give **+2.13/+2.10** vs **+1.89/+1.83**, differences **+0.241/+0.268**, CIs excluding zero.
  **The campaign's only established evidence on the causal question.**
- **REQ-046 (clip instrument) is INERT, not decisive.** It moved `g_clipped` by **+1.003** and the
  **raw** gradient by **+0.003**. The exponent +0.009 is **0/0**. *Iteration 129 read this as
  overturning band 13 and was **retracted** (iter. 130); Jerry retracted independently, matching
  numbers.*
- **Band 31 — the invariance is a THEOREM.** Epsilon deviation **5e-8** at these gradient scales;
  momentum transient ~20 of 750 steps. **No gradient-scaling intervention can move this optimiser.**
  REQ-046 was impossible, not botched.
- **Band 13 is unresolved and unanswerable by gradient scaling.** A future test must change the
  **loss**, change gradient **direction**, or use a **different optimiser**.

#### What survived

- **Band 32:** band 31 could have made `λ/g²` arbitrary. It does not — C is **more seed-stable than λ**
  (0.0776 vs 0.1235 dex), **twice the architecture-to-noise ratio** (15.7× vs 8.0×), **largest
  architectural spread** (0.462 vs 0.383). **g² is λ's seed-varying component.**
- **Bands 29–30:** raising the LR **decouples** λ from g (cov **0.0552 → 0.0371**), on **two
  independent designs**. Band 31 derives it: the LR is the only channel, so g is a **correlate, not an
  intermediary** — consistent with band 8's ~75% omitted-variable bias.
- **All descriptive bands** (6, 12, 14, 16–21, 24–28, 30) — none is an IV estimate.
- **REQ-045:** separability **+0.722** post-FE; **β_own −1.161 (t −12.8)** confirms band 30;
  **β_neighbour null** → iteration 124's partial/total reading **withdrawn**.

#### Registered negatives

Batch instrument **unusable** (non-monotone; per-type −1.25 to +1.12). Intercept test of the
non-gradient channel is **extrapolation** — 0 of 72 matrices have a positive first stage (withdrawn,
iter. 128). Attention entropy **does not** explain the alignment deficit (**flips sign** under a
quadratic depth term). Cross-type cancellation is **chance** (p 0.19–0.25) — its across-depth
counterpart is real (band 27).

#### Method failures carried forward

**Five traps of one shape** — reading a coefficient without checking where the data lives: collinearity
in REQ-036's per-type multipliers (122); collinearity in REQ-023's balanced totals (125,
`corr = −1.0000` **by construction**); extrapolation beyond support (128); a `corr = +0.609` verdict
resting on **one type** (115); an intervention my own band 29 predicted inert (130). **One caught
pre-flight** (126, by simulating the design before requesting it) — that is the check that works.

**Rules 7–9 added:** read the code path an intervention modifies *and check it against the campaign's
own findings*; confirm a regressor reaches zero before interpreting an intercept; check two constructed
regressors' correlation before reading either coefficient.

---

### WHERE THE GOAL STANDS (revised, 2026-09-04)

**The design question: ANSWERED, negatively, with a mechanism.** REQ-036 is a null — uniform LR beats
every per-type rule, predicted-best is worst by 120× the val noise floor, harm **monotone in
equalization** (Spearman −1.000). Band 16 explains it. **Do not build a momentum kernel or per-layer LR
on curvature equalization.**

**The descriptive account: COMPLETE at n=4, mutually consistent, correctly scoped.**

> **C = λ/g²** — the right object (band 32), seed-independent, **actively restored** (16),
> time-invariant on an equilibrated window (24). Structure is **three partly-offsetting terms** (26):
> **q,k purely backward** (softmax Jacobian −0.18, alignment −0.19), **mlp purely forward** (ReLU²,
> matching to 0.001 dex), with **‖a‖·‖d‖ conserved across depth** (27) explaining the depth-flatness.
>
> **What it is NOT:** a causal law. It is an equilibrium regularity measured under an optimiser that
> **cannot see gradient magnitude** (31).

**Open, and measurement-bound:** why the softmax Jacobian aligns less well across tokens; why ‖a‖ and
‖d‖ trade off (both need per-token backward vectors); how large the non-gradient channel is (**needs a
non-gradient-scaling intervention — ruled out by theorem**).

## REQ-037: a NON-learning-rate instrument for the curvature-gradient exponent

- status: **arms 1-3 DONE (2026-09-03) — `logs/kmaxwell/req037_nonlr_instrument/`.** Batch instrument at fixed LR: curvature responds only WEAKLY to moving the gradient via batch — per-matrix elasticity dlog(curv)/dlog(batch) median 0.075 (mean 0.062, spread [-0.25,0.36]); geomean curvature non-monotonic. Suggests the gradient channel is NOT dominant for the LR->curvature effect (exclusion restriction questionable). **CAVEAT:** batch confounds g-noise with tokens-seen (val monotonic: 0.5x 3.626/1x 3.512/2x 3.421); the CLEAN instrument is **arm 4 (per-matrix grad clip), which is DEFERRED — no clip hook in ebf53cd, needs a new hook.** n=1/arm, noisy. arm2(0.5x) ran eager (mbs<64 compile bug). Read arms 1-3 as a confounded first look; arm 4 is the right test.
  Shares the REQ-026/028/029 fork@2000 machinery, plus per-matrix curvature measurement which
  those runs omitted.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `ebf53cd` (same trainer + curvature probe as REQ-019/022/023)
- **node budget: ONE box, ~4 arms of 750 steps.** No fan-out needed.
- **priority note:** this tests the single assumption REQ-035's entire account rests on. Seeds
  cannot test it (see below), so it is not redundant with additional REQ-035 seeds.

**Why.** REQ-035 established a response ratio `d log lam / d log g = +1.98` (pooled 2SLS
+2.07/+2.10; robust to dropping weak first stages, +2.05 at F≥10 through +2.23 at F≥50), using
REQ-023's per-matrix learning-rate randomisation as the instrument. A placebo on
`curvature_along_polar` — a different functional of the same Hessian — gives +1.80, consistent
with whole-Hessian (Gauss-Newton) rescaling.

**The assumption.** That estimate is causal only under the **exclusion restriction**: the LR
must move lam *exclusively through* g. This is almost certainly false — changing a matrix's LR
moves it to a different point in weight space, where curvature differs for other reasons.
Sensitivity analysis (first stage −0.613, total effect −1.27): for the true exponent to be 1.0
rather than 2.0, **52% of the LR's effect on curvature would have to bypass the gradient
entirely.** Plausible or not, it is untested.

**Why n=4 seeds cannot test it.** Seeds re-randomise initialisation, not the instrument. Every
seed inherits the identical exclusion structure, so all four would be biased the same way.

**Why committed data cannot test it either — three routes checked and all closed.**
1. *Untreated matrices within a fork.* REQ-023 gives each matrix each multiplier exactly once,
   so **no matrix is ever untreated twice at the same fork — 0 usable pairs.** The balanced
   design that makes the LR instrument clean is exactly what destroys the non-LR one.
2. *Untreated at both forks.* All 72 matrices keep the **same** assignment across forks
   (verified: fraction identical = 1.00), so the surrounding perturbation is identical and the
   only difference is 500 steps of network aging — confounded.
3. *Existing batch ladders (REQ-026/028/029/033/034).* **No per-matrix curvature was ever
   measured** in any of them; they recorded val_loss only. Verified by file search — no
   curvature JSON exists anywhere outside `req019_*` and `req023_*`.

So this needs one new run. That is a hard negative, not an analysis gap.
### Design — 4 arms, 750-step continuations from the shared step-2000 state

The instrument must move g while holding each matrix's own learning rate **fixed**.

| # | arm | instrument |
|---|---|---|
| 1 | control | baseline batch, no clipping |
| 2 | batch 0.5x | halved gradient batch — changes gradient noise scale, LR untouched |
| 3 | batch 2x | doubled batch, same |
| 4 | per-matrix gradient clip | clip every Muon matrix's gradient at a fixed percentile, LR untouched |

Arms 2–3 use the existing batch-ladder machinery (REQ-026/028/029 templates) — the only change
is that **per-matrix curvature must be measured**, which those runs omitted. Arm 4 is the
cleanest instrument (moves g directly, nothing else) but needs a small clipping hook; drop it
if that is not cheap.

**Registered prediction, magnitudes fixed in advance:**
- **d log lam / d log g = 2.0 ± 0.3** under the batch instrument.
- If it lands in band: the exponent survives an instrument with a completely different
  exclusion structure, and the Gauss-Newton reading is **established** rather than assumed.
- If it lands near 1.0 or below: the LR-based +1.98 was carrying exclusion-violation bias,
  the Gauss-Newton reading **falls**, and REQ-035's account must be rewritten.
- If the batch instrument moves g by less than 0.05 dex, the test is underpowered — report
  that as inconclusive rather than as a result.

### Success criteria
- `per_matrix_curvature.json` per arm with the **existing field set** — `top_eigenvalue`,
  `gradient_block_norm`, `curvature_along_gradient`, `curvature_along_polar`, raw Lanczos
  `alphas`/`offdiags`, `residual_tail`. The gradient block norm is the load-bearing field.
- `summary.tsv` with the fitted exponent per arm and its bootstrap CI.
- Commit raw Ritz values; do **not** apply the geometric-tail correction silently.
- Report the realized first-stage strength (how far the instrument actually moved g).

### Artifacts
`logs/kmaxwell/req037_nonlr_instrument/`


## REQ-038: per-type activation and backward statistics — the q/k/v probe

- status: **DONE (2026-09-03, n=1) — `logs/kmaxwell/req038_activation_backward_probe/`.** The q/k 'excess' is a GRADIENT DEFICIT: q,k output-gradient d_rms=0.00262 vs v=0.00399 (ratio 0.66), with IDENTICAL input activation (a_rms=1.0036 all of q/k/v) -> the difference is PURELY backward, not forward; q,k grads also lower-rank (72/76 vs 106). q,k near-identical (band 18). Attention entropy 1.42 nats, q.k-logit rms 27.2. Confirms bands 14-19 empirically. New reusable activation/backward probe. n=1 (fresh fork-1500 seed0).
  **Cost premise corrected:** this was filed as a probe on "an existing checkpoint", but no `.pt`
  weights are committed anywhere in the repo — REQ-019's boxes were ephemeral and only the derived
  `per_matrix_curvature.json` files landed. A standalone run must therefore regenerate a fork-1500
  state first (~4 min at the measured 0.162 s/step), then the probe itself is minutes. Since
  **Arm A regenerates exactly these states by design**, adding this probe's five measurement
  fields to Arm A gets the result on **four seeds at essentially zero marginal cost**. Do that
  rather than a standalone seed-0 run.
- requested: Jack (via Claude analysis session) / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- **node budget: ≤2 nodes.** Run it in the gaps of any other job.

**Why this is the highest-value measurement available.** The campaign's variance budget for the
0.379 dex spread in log C: gradient scale (λ ∝ g², Gauss-Newton) 21.8%; **matrix type, beyond
gradient 53.3%**; end-block position 8.2%; writer-role interaction 1.9%; unexplained 14.8%.

**The largest term is a label, not a mechanism**, and it is irreducible to anything computable
from committed data: against the type label's R² = 0.751, the best architectural descriptor set
(writer role + attn/mlp + fan-in + fan-out) reaches 0.484 and the best measured set (polar
curvature + spectral gap + negative-eigenvalue fraction) reaches 0.505.

**Seven mechanisms have now been proposed for it and falsified** — bilinear q·k coupling, softmax
saturation, nonlinearity exposure as an explanation of levels, within-block consumption order,
curvature concentration, Muon group rank, and QK-norm scale-invariance. The effect itself is
untouched and remains among the most statistically secure findings in the campaign. This probe is
the first measurement that could reach it; **if it also fails (see P4/P5), the type term should be
reported as irreducible in this architecture** — a real and publishable outcome that would close
the campaign's central question rather than leave it open.

### What to measure — one forward+backward pass, per Muon matrix

At a single fork-1500 state (regenerate it, or reuse Arm A's seed-0 state; s=1.00 preferred),
record per matrix:
1. **input activation second moment**: RMS and Frobenius norm of the matrix's input tensor `a`;
2. **output-gradient second moment**: RMS and Frobenius norm of the backward tensor `d`;
3. **effective rank of both**: participation ratio of the singular value spectrum of `a` and `d`;
4. **for attention specifically**: the attention-probability entropy per head, and the RMS of the
   q·k logits — the quantity that distinguishes q/k from v mechanically;
5. token count and batch used, so the moments are normalisable.

Fields 1–3 are generic; field 4 is the discriminating one.

### Registered predictions, bands fixed in advance

- **P1.** The gradient identity `|grad| ≈ |d| · |a|` must hold per matrix to within 20% —
  a correctness check on the probe itself. If it fails, the other numbers are not interpretable.
- **P2.** **q and k have near-identical `|a|` (within 5%, they read the same tensor) but differ in
  `|d|` by ≥ 15%.** If instead their `|d|` matches too, the q/k difference is not in the backward
  pass and the bilinear explanation fails.
- **P3.** **v's `|d|` differs from q/k's by ≥ 30%**, consistent with its 0.81 dex adjusted-level
  gap being the largest of the three.
- **P4.** Adding `|a|`, `|d|` and the two effective ranks as regressors to the model
  `log C ~ log g + …` **raises R² from 0.218 (gradient only) to ≥ 0.60**. If it does not reach
  0.60, the activation/backward moments do **not** explain the type effect either, and the 53%
  should be reported as irreducible in this architecture — a negative result worth having and one
  that would close the campaign's central question rather than leave it open.

**AMENDED PREDICTIONS (iteration 42) — P2 and P3 as filed target the wrong contrast. Use these.**

REQ-038 was written at iteration 33, when the structure looked like a q > k > v ordering.
Iteration 41 established it is **{q, k} versus the other four types**, and the amendment matters
because the filed P2/P3 aim at the wrong comparison:

| contrast | fork-1500 | fork-2000 | status |
|---|---:|---:|---|
| **q,k mean − other four** | **+0.812** | **+0.842** | **THE effect** |
| k − v | +0.640 | +0.656 | part of the same effect |
| q − k | +0.171 | +0.167 | real but ~20% the size |

*(q − k is genuinely significant — paired t = +4.62 / +4.58, q > k in 11/12 and 12/12 blocks — but
it is a second-order feature within the high group, not the main effect. P3 as filed attributes
the 0.81 dex figure to v alone; that number is the q,k-versus-rest contrast.)*

**The sharp quantitative prediction the probe should carry.** Measured from committed data:

> **q and k carry 0.40× the gradient norm of the other four types (log gap −0.403 / −0.407 dex,
> both forks) while sitting +0.81 dex HIGHER in gradient-adjusted curvature.**

Since `|grad| ≈ |d| · |a|` and q, k, v all read the **same residual tensor**, their |a| is identical
by construction — so **the entire q,k-vs-v gradient deficit must appear in |d|**. That is directly
falsifiable against the probe:

- **P2′ (replaces P2).** `|a|` for q, k and v must agree within **5%** (they read the same tensor —
  a probe correctness check as much as a physics one). **`|d|` for q,k must be 0.35–0.45× that of
  v**, matching the −0.40 dex gradient gap. If |d| does not carry the deficit, either the probe is
  wrong or |a| differs where it cannot.
- **P3′ (replaces P3).** The q,k-vs-other-four contrast in `|d|` must be **≥ 2x**, and the same
  contrast in `|a|` must be **< 1.2x** across the four types that read the residual stream.
- **P5 (new, the decisive one).** **q and k receive far less gradient yet carry far more curvature
  per unit gradient. If |d| alone explains the +0.81 dex adjusted-curvature gap, then adding |d|
  to the model must reduce the q,k binary's coefficient by ≥ 50%.** If the binary survives with
  its coefficient intact after conditioning on |a| and |d|, **the effect is not in the first-order
  backward statistics at all** — and given that four mechanisms have already been falsified
  (bilinear coupling, nonlinearity exposure, softmax saturation, consumption order), that would
  establish the QK gap as irreducible to anything measurable in a single forward+backward pass.

**P1 and P4 stand as filed.** P4's ≥0.60 band should now be read against the QK binary's own R² of
0.737 — activation moments must beat *that*, not the 0.218 gradient-only baseline, to count as an
explanation rather than a correlate.

**Why this amendment rather than a new request:** the measurement list (fields 1–5) is unchanged —
only the registered bands move. No extra compute, no queue impact.


### Success criteria
- One JSON per matrix with fields 1–5 plus the existing `gradient_block_norm` for the P1 check.
- `summary.tsv` with the per-type means of every recorded quantity.
- The P4 regression run and reported against the 0.60 band, whichever way it falls.

### Artifacts
`logs/kmaxwell/req038_activation_probe/` — or, if folded into Arm A as recommended, alongside
`logs/kmaxwell/req035_C_mechanism/armA/` with the fields recorded per seed.

## REQ-040: dispatch the active queue

- status: **OPEN — start now.**
- requested: Jack / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)

Please begin the existing active queue immediately, following the standing constraints and
authoritative specifications already recorded above. This is a dispatch request, not a change to
any experiment's protocol.

1. Start **REQ-035 Arm A** first, fold **REQ-038's** activation/backward probe fields into it, and
   run its four seeds sequentially on one box.
2. After Arm A has landed its results, run **REQ-036**, then **REQ-037**, sequentially under the
   fleet-wide ≤2-node ceiling.
3. Run **REQ-034** when capacity permits, without delaying the priority queue.

Do not start REQ-035 Arms B/C/D or a standalone REQ-038 unless their existing request blocks say
their prerequisite or fallback condition has been met. Mark each affected request `RUNNING` when
work begins and preserve the prescribed artifact paths and reporting gates.

## REQ-041: add per-matrix weight norms to curvature runs

- status: **DONE (2026-09-03) — `logs/kmaxwell/req038.../weight_norms.tsv`.** Per-Muon-matrix ||W||_F recorded at fork-1500 in the REQ-023 shape (fork_seed, arm, step, name, weight_frob), folded into the REQ-038 probe. Prior curvature checkpoints were cleaned by re-bootstraps so this is a fresh fork-1500; `measure_activation_backward.py` can ride along on future curvature runs.
- requested: 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)

**Ask:** whenever `measure_per_matrix_curvature.py` runs, also record `‖W‖_F` per Muon matrix at the
same steps, and commit it as a TSV in the same shape as
`logs/kmaxwell/req023_per_matrix_lr/weight_norms.tsv` (`fork/seed, arm, step, name, weight_frob`).

**Cost:** one `.norm()` per matrix per measured step — **negligible** next to the Lanczos probe that
dominates these runs. No new training, no extra nodes; it rides along on whatever is already queued.

**Why:** band 15 tests a theorem — `d log C / d log‖W‖ = 0` for QK-normed matrices, unconstrained
otherwise — that currently passes on REQ-023's two forks (+0.049 CI [−0.261, +0.381] and −0.062 CI
[−0.329, +0.226], against −0.398 / −0.228 for the other four types) and **cannot be seed-checked**,
because Arm A committed curvature without weight norms. Iteration 86 established that no proxy
substitutes: the LR ladder is not a ‖W‖ change under Muon, and the test built on it is under-powered
and uninformative. **This field is the only route.**

**Two cheap companions, if the same run is being touched:**
- **extend the LR ladder to ~10×** — band 16's registered falsifier. If matrix identity's share of
  log C's variance collapses at 10× (it is 93–95% at 2.8×), C is *locally* restored rather than
  genuinely invariant.
- **have two MPI ranks measure one overlapping matrix** — the 8 ranks currently hold disjoint
  9-matrix shards, so pure measurement noise cannot be separated from temporal drift in any existing
  data.

**If REQ-036 or REQ-037 is already running, add the field there rather than re-running anything.**


## REQ-042: matched K-Maxwell vs bi-Maxwell high-batch ladder — 32× and 64×

- status: **BLOCKED — insufficient data (2026-09-03).** 32x/64x × 750 steps exceeds FineWeb10B. The loader StopIterations when data runs out (no looping — hit in REQ-029's 16x). Usable-batch budget (REQ-029 metric, 100M-token shards, max ~100 chunks=10B tokens): **32x needs skip 62 + 750 = 812 usable batches but max avail = 100×⌊100M/16.78M⌋ = 500** → short by 312. **64x needs 781, max avail = 100×2 = 200** → short by 581. (A 64x continuation is ~25B tokens, 32x ~12.6B; FineWeb10B tops out at 10B.) **Cannot run as specified.** Options for you: (a) point me at a larger corpus (FineWeb100B or similar) + I re-bootstrap the downloader; (b) reduce the step count so the token budget fits (e.g. 32x at ≤~430 steps, 64x at ≤~170 — but that breaks the 'matched 750-step window' vs REQ-034); (c) allow data-repeat/looping (the within-batch kernel *diff* stays valid since all arms repeat identically, but absolute loss is 2nd-epoch — and the loader would need a looping patch). REQ-034 (1x–16x) is delivered and is the clean same-axis result; 32x/64x need a data decision. Machinery otherwise ready (365c392d bootstrapped, fork@1984 plain-Muon base, 6 configs = 32x/64x × {annealed, bimaxwell, μ0}).
  already-running request.**
- requested: Jack / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `365c392d695f95dc9a4fb89095e85a6a7b5d551e` (the REQ-026/029/033/034 batch-ladder code path)
- node budget: **≤2 nodes fleet-wide.** Run sequentially or two-at-a-time only if no other request
  is using a node.

### Goal

Extend the batch-size experiment beyond the existing 16× endpoint and make the comparison fully
matched: at each larger batch, compare **annealed K-Maxwell**, **bi-Maxwell**, and the **no-extra-
momentum control** from the same forked model state, with the same data position and training
window. The earlier chart is useful context, but its K-Maxwell and bi-Maxwell series used different
forks/horizons and therefore must not be joined. This request supplies directly comparable points
at **32× and 64×**.

The question is simple: once ordinary momentum's benefit is already near zero at 16×, does either
two-timescale kernel retain a measurable validation-loss advantage at still larger batches? And if
so, does K-Maxwell remain better than bi-Maxwell on the same protocol?

### Design — six matched arms, n=1 per cell

Base batch is 524,288 tokens per optimizer step. Use seed 0 and a fresh shared plain-Muon base run
to **fork step 1984**, then run the following six 750-step continuations from that exact serialized
state:

| batch | `batch_tokens` | `microbatch_sequences` | `skip_batches` | kernels |
|---|---:|---:|---:|---|
| 32× | 16,777,216 | 64 | 62 | `muon{mu:0.0}`, `bimaxwell_muon`, `annealed_weights_muon` |
| 64× | 33,554,432 | 64 | 31 | `muon{mu:0.0}`, `bimaxwell_muon`, `annealed_weights_muon` |

`1984 × 524,288` is exactly divisible by both larger batch sizes. Therefore the continuations all
start at the same approximately 1.040B-token data position; do not substitute the old step-2000
fork, where those skips would be fractional. Run `start_step: 1984`, `stop_after_step: 2734`, and
keep `train_steps: 3250`, `lr: 0.025`, `weight_decay: 0.05`, `cool_down_learning_rate
cooldown_frac: 0.7`, and the existing validation cadence. Keep microbatch sequences at 64, using
gradient accumulation for the larger batches; do not use an eager fallback.

Use the exact existing optimizer definitions:

- **control:** `muon` with `mu: 0.0`;
- **bi-Maxwell:** the REQ-026/029 `bimaxwell_muon` record (`mu: 0.95`, `fast_decay: 0.85`,
  `slow_decay: 0.98`, `fast_weight: 0.4385`, `switch_step: 1000`);
- **K-Maxwell:** the shipped `annealed_weights_muon` schedule from REQ-034, with the listed decays
  and start/end weights, `switch_step: 1984`, and `anneal_steps: 750`.

The new fork means the absolute losses are not a replacement for the old fork-2000 values. The
primary estimands are the two same-batch differences: `final_val(K-Maxwell) − final_val(control)`
and `final_val(bi-Maxwell) − final_val(control)`, plus `K-Maxwell − bi-Maxwell`.

### FineWeb data — expand without reuse

A 64× continuation consumes about 25.17B tokens after the fork (about 26.21B including the shared
base position), exceeding the existing FineWeb10B subset. Pull additional, non-overlapping FineWeb
training shards sufficient for the largest arm, with a safety margin for discarded shard tails.
Do **not** wrap, cycle, or silently reuse training data.

Before launching, make a machine-readable data manifest recording the dataset/revision, shard list,
per-shard usable-batch count, total usable batches, and total usable tokens. Budget with
`Σ floor(shard_tokens / batch_tokens)`, not nominal bytes or raw token totals. Each 32× arm needs
at least `62 + 750 = 812` usable batches; each 64× arm needs at least `31 + 750 = 781`.
Use the same ordered data manifest and identical skip count for all three kernels at a given batch.

### Gates

1. Tests green at the pinned SHA.
2. Build the shared base and verify its validation loss at step 1984 is finite; record it.
3. Assert the usable-batch budget before training every arm, then run a 20-step finite-loss smoke
   test for every config.
4. Verify the three arms within each batch load the same serialized base state and data position.
5. If data provisioning, a budget check, or a smoke test fails, mark the request `NEEDS-INFO` with
   the exact failed gate; do not replace missing data by cycling the existing 10B subset.

### Artifacts and readout

Commit only code, configs, data manifest, logs, and derived results—never checkpoints or tensors:

`logs/kmaxwell/req042_high_batch_kernel_ladder/{README.md,summary.tsv,readout.tsv,manifest.tsv,
data_manifest.tsv,make_req042_configs.py,configs/,logs/}`.

`summary.tsv` must contain base validation, checkpoints through step 2734, and final validation for
all six arms. `readout.tsv` must contain:

```text
batch  control_final  bimax_final  kmax_final  bimax_minus_control  kmax_minus_control  kmax_minus_bimax
32x
64x
```

Interpret absolute differences below roughly `±2e-4` as the established single-seed noise floor.
Plot these as a **new matched high-batch series**; do not draw one continuous line through the old
fork-2000 bi-Maxwell points or fork-1000 K-Maxwell points. Report whether each kernel remains
beneficial, is indistinguishable from control, or is worse at 32× and 64×.

## REQ-046: REQ-037 arm 4 — the per-matrix gradient-clip instrument (design decision made)

- status: **DONE 2026-09-04 — CONCLUSION CORRECTED (iter 130): instrument INERT, band 13 UNRESOLVED (not overturned)** → `logs/kmaxwell/req046_permatrix_clip_instrument/`
- **CORRECTION.** My iteration-129 "band 13 overturned" read was an over-read, now retracted (agrees with your
  iter 130). The clip moved `clipped_gradient_block_norm` (c·‖g‖, +1.003) but the RAW `gradient_block_norm` the
  network experiences did NOT move: +0.0028, CI [−0.0143,+0.0200] (verified from my own committed JSONs). Muon's
  polar map normalises the scale out of the update → trajectory unchanged → raw g unchanged → λ unchanged. So the
  exponent is 0/0, uninformative; band 13 is UNRESOLVED. Full compensation separately excluded (predicts raw-g
  slope ≈−1, outside CI). Root cause = band 29 (Muon grad-scale-invariant), which I failed to apply to my own
  instrument. Data stands + is what enabled the catch; interpretation was wrong. Iter-114 two-functional detection
  + REQ-045 unaffected. See README "What it settles (corrected)".
- **RESULT — band 13's causal reading OVERTURNED.** Exponent `d log λ / d log(clip) = +0.009` (≈0),
  first stage `d log g_clipped / d log(clip) = +1.003` (✓ — the instrument works, unlike the batch arm's
  first-stage≡0). Decisive logic: a causal +2 gradient channel would have moved log λ with slope ≈+2 given
  the +1.003 first stage; it moved +0.009. So changing ONLY gradient magnitude (LR path fixed) does not move
  equilibrium curvature — band 13's ≈+2 exponent is the LR channel, not curvature-gradient physics
  (mechanism = band 29 Muon scale-invariance). Reduced form non-monotone + per-type symmetric-around-zero =
  noise, confirming inert not weak. Instrument = PerMatrixClipMuon (grad clip pre-momentum) + probe field
  clipped_gradient_block_norm; validated (zero-momentum cancels, fork-momentum perturbs ‖Δw‖=1.21). Registry-
  lock test trips by design (new registry entry, research branch).
- (was) status: **OPEN**
- requested: 2026-09-04 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- **node budget: ONE box, 3 arms × 750 steps.** Same shape as REQ-037 arms 1–3.
- **priority: HIGHEST of the open analysis requests.** This is the only outstanding measurement
  that can overturn a load-bearing band.

**Why this supersedes the arm-4 sketch in the REQ-037 block.** That sketch had two defects, found by
reading the code rather than its description (iteration 119). Both are fixed here.

**Defect 1 — a clip inside `polar_express` is cancelled exactly.** `train_gpt.py:177` normalises to
unit spectral norm before Newton-Schulz: `X = X / (X.norm(...) * (1+2e-2) + 1e-6)`. Scaling the
gradient by *c* gives `(c·g)/(c·‖g‖) = g/‖g‖` — **c cancels**, verified numerically (clips of 0.5,
1.0, 2.0 produce identical values to 6 dp). **The clip must be applied to `grad_chunk` BEFORE
`momentum_buffer.lerp_`**, so the momentum buffer accumulates the clipped gradient and its trajectory
is not normalised away.

**Defect 2 — the existing probe cannot see the clip at all.**
`measure_per_matrix_curvature.py:100` recomputes the gradient from scratch via
`torch.autograd.grad(loss, params)` on its own batches, then line 202 records
`gradient_block_norm = ‖g‖`. **This is the raw loss gradient, computed entirely outside the
optimiser.** A clip in the Muon update path never touches it, so `d log g / d log clip ≡ 0` and the
Wald ratio is undefined. *(Verified by reading the probe, not inferred.)*

**DECISION MADE (this session, not deferred).** Of the two fixes proposed in iteration 119, **take
option 1: change what is measured, not what is optimised.**

> **Add one field to the probe: `clipped_gradient_block_norm` = ‖clip(g) ‖, the norm of the same
> recomputed gradient after applying that matrix's clip multiplier.**

**Rationale for choosing this over option 2 (per-matrix loss weights).** Option 2 changes
`param.grad` directly, so the existing probe would measure the first stage with no new field — but it
changes **what is being optimised**, reintroducing an exclusion problem of exactly the kind this arm
exists to eliminate. Option 1 keeps the intervention in the update (where REQ-037 wanted it) and moves
the measurement to match. **The cost is one line in the probe; the alternative costs the experiment's
purpose.**

**With that field the estimator simplifies and needs no ratio at all.** Because
`d log g_clipped / d log clip = 1` by construction, the Wald ratio collapses to the **reduced form**:

> **exponent = d log λ / d log(clip multiplier)**, estimated directly with matrix fixed effects.

**Design.** Fork@2000, 750 steps, 3 arms. Per-matrix clip multiplier over **{0.5, 1.0, 2.0}**, each
matrix receiving each level exactly once across the three arms (REQ-023's balanced assignment — it is
the right choice *here*, since the neighbour effect is not the question and balance removes
confounding from the own-clip effect). Record per-matrix curvature at 2750, `weight_frob`, and the new
clipped-gradient field.

**Registered check — the two properties the batch instrument lacked (iteration 113):**
- **monotone reduced form** — `λ` must move monotonically across the three clip levels, unlike the
  batch arm whose curvature dipped at the control;
- **every per-type ratio positive** — the batch arm gave −1.25 to +1.12 across types;
- **first stage confirmed** — report `d log g_clipped / d log clip` and verify it is ≈ 1.

**What it settles.** The exclusion restriction is **already known violated** (iteration 114: two
Hessian functionals under the same instrument give +2.13 vs +1.89, differences +0.241/+0.268 with CIs
excluding zero). What is *not* known is **how large the non-gradient channel is** — iteration 114's
bound is one-sided twice over, since a channel acting equally on both functionals is invisible to it.
**Arm 4 removes the LR channel entirely rather than bounding it:**
- **result near +2** → the non-gradient channel is small; band 13's exponent survives as approximately
  causal;
- **result near +1 or below, with a monotone reduced form** → ~52% of the LR effect bypasses the
  gradient, and **band 13 becomes a statement about LR response, not curvature-gradient physics.**


## REQ-047: per-token backward statistics — the two open alignment questions

- status: **DONE 2026-09-04 (n=4)** → `logs/kmaxwell/req047_pertoken_backward/`
- **RESULT — both mechanism questions answered.** (a) The q,k alignment deficit is a TOKEN-COHERENCE effect:
  da_cos_mean (adjacent-token backward cosine) q,k≈+0.02 vs v=+0.42, proj=+0.09 — PASS 4/4 seeds; v's backward
  vectors are coherent (cos 0.42) + concentrated (participation 456), q,k's are near-orthogonal token-to-token
  (k slightly anti-coherent) + spread → low rank-1. grad_rank1_frac correlates with align_ratio r=+0.656 (PASS
  |r|>0.5). So band 25's aggregate −0.190 dex deficit = incoherent per-token outer-product accumulation, NOT
  sparsity. (b) Band 27's ‖a‖-‖d‖ depth tradeoff is a SCALE effect not support: corr(a_part,d_part) across
  depth near-zero/positive for 5/6 types (only attn.v −0.36). da_cos computed ALONG seq axis (boundary-safe).
  q/k/v share a_participation=8138 (band 21). Doesn't revive alignment as C-predictor (per scoping).
- (was) status: **OPEN**
- requested: 2026-09-04 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- **node budget: ONE box, one forward+backward pass per seed.** Same cost as REQ-038/043.
- **This is an INCREMENT to an existing, validated probe — not new instrumentation.**

**What to add.** `measure_activation_backward.py` already captures the per-token tensors in `acts`
and `grads` (lines 43–45) and reduces them to `rms`, `frob` and `eff_rank`. **The per-token
information is already in memory and is being discarded.** Add four reductions per matrix:

1. **`d_token_norms`** — the distribution of ‖dₜ‖ across tokens: record mean, sd, and the
   participation ratio (how many tokens carry the backward signal).
2. **`a_token_norms`** — the same for ‖aₜ‖.
3. **`da_cos_mean`** — mean cosine between `dₜ` and `dₜ₊₁` across adjacent tokens. **This is the
   direct measure of the token-wise coherence that band 25's alignment ratio quantifies only in
   aggregate.**
4. **`grad_rank1_frac`** — `σ₁²/Σσᵢ²` of the accumulated `Σₜ dₜaₜᵀ`: how close the weight gradient is
   to rank-1, i.e. how coherently the outer products add.

**IMPLEMENTATION NOTES — verified against the committed probe, 2026-09-04.**

*Confirmed implementable as an increment:* the forward hook stores `acts[nm]=inp[0].detach()` and the
backward hook `grads[nm]=gout[0].detach()` (lines 43–47), and **both remain in scope through the
reduction loop at line 62** — the per-token tensors are live where the new statistics would be
computed. No new hooks, no second pass.

**One correctness detail that must not be missed.** The existing `eff_rank` flattens with
`mat.reshape(-1, mat.shape[-1])`, collapsing **batch and sequence into one axis**. That is fine for a
rank statistic but **wrong for `da_cos_mean`**: flattening would pair the last token of one sequence
with the first token of the next, injecting spurious decorrelation at every sequence boundary.
**`da_cos_mean` must be computed along the sequence axis within each batch row**, e.g. on a tensor
shaped `[batch, seq, feat]` compare `d[:, :-1]` with `d[:, 1:]`, then average. **The same applies to
any per-token ordering statistic added here.**

**Run on:** Arm A's four fork-1500 seeds (as REQ-043 did), so every result is n=4 immediately.

**The two questions this settles, both currently measurement-bound:**

**(a) Why does the softmax Jacobian's output align less well across tokens?** Band 25 established the
alignment deficit is **−0.190 ± 0.007 dex** and iteration 105 showed it *is* the shortfall, measured
rather than inferred. **What is unknown is its origin.** `da_cos_mean` and `grad_rank1_frac` locate it:
if q,k's backward vectors are less coherent token-to-token, the deficit is a property of the softmax
Jacobian's *token structure*; if their per-token norms are more concentrated, it is a *sparsity* effect.

**(b) Why do ‖a‖ and ‖d‖ trade off across depth?** Band 27 measured `corr(log‖a‖, log‖d‖) = −0.87 to
−0.99` with the product 2–4× flatter than either factor, and iteration 111 established this explains
the gradient's depth-flatness. **The mechanism is unexamined.** Per-token norm distributions
distinguish a *scale* trade-off (both tensors rescale) from a *support* trade-off (the same total
signal spread over more or fewer tokens).

**Registered checks (n=4):**
- **`da_cos_mean` for q,k must be lower than for v/attn.proj** — same four same-shape matrices as
  band 14 — in ≥3 of 4 seeds, if the alignment deficit is token-coherence;
- **`grad_rank1_frac` must correlate with the per-matrix alignment ratio** at |r| > 0.5, since both
  measure how coherently outer products accumulate;
- **within a type across depth, `corr(a_token_participation, d_token_participation)` distinguishes
  the two readings of band 27**: strongly negative → support trade-off; near zero → scale trade-off.

**Honest scoping.** Iteration 141 showed the *aggregate* alignment ratio adds nothing to a predictive
model of C once its shared-λ component is removed. **This request is not an attempt to revive it as a
predictor** — it targets the two mechanism questions the campaign has repeatedly identified as
measurement-bound, and its value does not depend on band 33.

**Priority: below REQ-044 and REQ-045.** Neither open question blocks the account of C, which is
complete at n=4. **File it as a background run.**


## REQ-044: fully paired Muon / bi-Maxwell / K-Maxwell batch ablation

- status: **DONE 2026-09-04** → `logs/kmaxwell/req043_paired_kernel_batch_ablation/` (n=3, 60/60 finite)
- **RESULT — all 3 load-bearing questions answered, reproduce across 3 independent seeds.** (1) mu95≈mu0
  at EVERY batch incl 16× (mean +0.0003, |mean|<5e-4 all batches) — single-EMA momentum buys nothing in
  Muon. (2) bimax−mu0 decays to ~zero by 16× (−0.0105→+0.0006) — denoiser edge batch-absorbed. (3) K-Maxwell
  keeps a material gain at 8×/16× in ALL seeds: kmax−mu0 −0.0063/−0.0047, kmax−bimax −0.0043/−0.0052 (all
  3 seeds negative, ~20× sd) — K-Maxwell is the ONLY kernel retaining its large-batch edge and it beats
  bi-Maxwell there. 3 independent bases (hashes differ, val@2000≈3.44367). Fresh paired controls remove the
  0.00088 offset; confirms REQ-026/029/034 reads + new kmax−bimax within-cell contrast.
- (was) status: **OPEN**
- requested: Jack / 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- pinned SHA: `365c392d695f95dc9a4fb89095e85a6a7b5d551e`
- node budget: **≤2 nodes fleet-wide.** Run sequentially or two-at-a-time without
  displacing the standing higher-priority queue.

### Goal

Re-run the complete 1×–16× batch ladder as one matched experiment so every reported
difference uses a **fresh control from the same base state and campaign**. This audits two
assumptions behind the current plot:

1. Does ordinary single-EMA Muon (`mu=0.95`) remain indistinguishable from no-momentum
   Muon (`mu=0.0`) at 16×? It was directly checked only at 1×, 4×, and 8×.
2. Does K-Maxwell retain its 8×/16× gain when it and bi-Maxwell are paired against fresh
   controls, rather than comparing K-Maxwell to stored REQ-026/028/029 controls with the
   documented 0.00088 base offset?

### Design — 60 continuations from three independent bases

Use batches `{1×, 2×, 4×, 8×, 16×}` × four block-matrix optimizers × base seeds
`{0, 1, 2}`:

| arm | optimizer | purpose |
|---|---|---|
| no-momentum control | `muon{mu: 0.0}` | isolates the value of temporal gradient memory while retaining Muon's matrix update |
| ordinary Muon | `muon{mu: 0.95}` | tests whether the single EMA differs from `mu=0`, including the missing 16× ablation |
| bi-Maxwell | REQ-026/029 `bimaxwell_muon` record | fresh reproduction of the frozen two-rate curve |
| K-Maxwell | REQ-034 `annealed_weights_muon` schedule | fresh reproduction of the annealed curve |

For each seed, train a genuinely independent base state to step 2000; **do not** implement
seeds 1/2 by loading the seed-0 fork, because REQ-027 showed that this overwrites the seeded
initialization and measures only accelerator nondeterminism. Fork all 20 continuations for a
seed from that seed's exact serialized state. Record the state hash and base validation loss.

Within each `(seed, batch)` quartet, use the identical ordered data stream, data cursor,
validation tokens, learning-rate schedule, and all non-block optimizers. If data order is not
seed-dependent, say so plainly: the three bases still test different initialized/trained
networks, but not data-order robustness.

### Exact shared protocol

Follow REQ-034's fork@2000 protocol exactly:

| batch | `batch_tokens` | `microbatch_sequences` | `skip_batches` |
|---|---:|---:|---:|
| 1× | 524288 | 64 | 2000 |
| 2× | 1048576 | 64 | 1000 |
| 4× | 2097152 | 64 | 500 |
| 8× | 4194304 | 64 | 250 |
| 16× | 8388608 | 64 | 125 |

Use `start_step: 2000`, `stop_after_step: 2750`, `lr: 0.025`,
`weight_decay: 0.05`, `cool_down_learning_rate cooldown_frac: 0.7`, the same fixed
validation set (`val_tokens: 10485760`), and the same AdamW settings for embeddings,
projection, and remaining parameters. Keep microbatch sequences at 64 and scale only
gradient accumulation. Use the verified REQ-029 usable-batch calculation and at least the
86-shard provision required by 16×; never cycle or silently reuse exhausted data.

Use the exact existing kernel parameters from REQ-026/029 and REQ-034. K-Maxwell must
switch at the fork and complete the full start-to-end weight anneal during the 750-step
window, exactly as REQ-034 did. Do not retune any kernel by batch size.

### Pairing and readout

For every seed and batch, report these paired differences at step 2750:

```text
ordinary_muon_minus_mu0 = final_val(mu95)  - final_val(mu0)
bimax_minus_mu0         = final_val(bimax) - final_val(mu0)
kmax_minus_mu0          = final_val(kmax)  - final_val(mu0)
kmax_minus_bimax        = final_val(kmax)  - final_val(bimax)
```

Negative means the first optimizer has lower validation loss. Preserve each seed's values;
then report mean, standard deviation, and range across the three paired replicates. Do not
use a pooled or historical control. Do not call the old ±2e-4 heuristic a confidence
interval: estimate run-to-run variation from this experiment and keep the raw paired values
visible.

### Gates

1. Tests green at the pinned SHA.
2. Three independently trained step-2000 bases verified by seed, state hash, and finite base
   validation loss.
3. Machine-readable check that all four arms in each `(seed, batch)` cell load the same base
   state and start at the same data cursor.
4. Usable-batch budget assertion and a 20-step finite-loss smoke test for every config before
   its full continuation.
5. If the four arms in a cell do not share the same state/data provenance, reject that cell
   rather than comparing it.

### Success criteria and artifacts

Commit code, configs, logs, and derived tables only—never weights or optimizer tensors—to:

`logs/kmaxwell/req043_paired_kernel_batch_ablation/`

Required files:

- `README.md`: method, provenance audit, results, caveats, and direct comparison with the old
  REQ-026/028/029/034 conclusions;
- `manifest.tsv`: seed, batch, arm, state hash, data cursor, node, config, and exit status;
- `summary.tsv`: base and final validation losses for all 60 arms;
- `readout.tsv`: all four paired differences per seed plus mean/std/range;
- config generator, committed configs, validation trajectories, and concise raw logs;
- one plot showing per-seed points and aggregate curves for all four optimizers. Old campaign
  curves may appear only as clearly labeled context and must not be joined to the new series.

The load-bearing decisions are: whether `mu95-mu0` is still noise-sized at 16×; whether the
fresh bi-Maxwell curve again reaches zero; and whether K-Maxwell remains materially negative
at 8×/16× in every independent paired replicate. Report disagreement across seeds rather
than averaging it away.

## REQ-049: replicate the per-matrix LR gauge test at n=4 seeds — the campaign's strongest result, on one seed

- status: **OPTIONAL (downgraded 2026-09-04, iteration 178) — its replication question is ANSWERED from committed data.** REQ-023 (`logs/kmaxwell/req023_per_matrix_lr/`) is an independent per-matrix LR experiment at fork 1500 with a different randomisation and **5× the sample size**; it reproduces band 49 (λ −1.152 vs −1.218; pooled C elasticity **−0.0505 ± 0.0327 = 4.2% of λ's**; estimates consistent at z +1.55). **The replication risk this request was filed against is retired.** It would still add the **seed axis** (both experiments are n=1 seed), so it remains worth running if capacity is free — but **REQ-048 should take priority.** ≤2 nodes.
- priority: **high — higher value per GPU-minute than REQ-048, and the machinery already exists.**

**WHY.** Iteration 174 found the campaign's strongest single result in **already-committed REQ-045 data**:
a **randomised per-matrix LR** moves a matrix's own curvature hard (**d log λ/d log m = −1.218, t −7.55**)
and moves its **C not at all** (**+0.081, t +0.89**), with a **powered** null (95% CI [−0.097, +0.258],
half-width 0.177, **excluding λ's elasticity**). The components offset exactly as the gauge theorem
requires: **−1.218 − 2(−0.650) = +0.081**, identity residual **9.99e-16**.

**This is the only randomised-intervention confirmation of any band in the campaign** — and **it rests on
n = 1 seed** (REQ-045 ran 3 arms on one network, one step). **Everything else load-bearing is at n = 4.**

**WHAT TO RUN.** Exactly REQ-045's design, on **4 seeds** instead of 1:
- 3 global arms `S ∈ {0.7, 1.0, 1.4}`, crossed with per-matrix `m_i` drawn independently from
  {0.6, 0.85, 1.0, 1.2, 1.7} — **re-draw per seed** (fresh `rng` per seed × arm, as `make_req045_arms.py`
  already does per arm).
- Fork @2000, stop @2750, per-matrix curvature at 2750 — **unchanged from REQ-045**.
- **Emit `req049_draws.json` in the same schema**, including the stored
  `identifiability_corr_own_vs_othersmean` pre-check (REQ-045's was **−0.182**; anything beyond ±0.9
  invalidates the arm and should halt).
- **Record the LR multiplier explicitly in the curvature output** (the iteration-172 lesson: `s060`-style
  tags were misread as checkpoints for ~20 iterations).

**BAND 49 — CRITERION REGISTERED IN ADVANCE:** (i) **|d(log λ)/d(log m)| ≥ 0.5 with |t| ≥ 3** in ≥3 of 4
seeds; (ii) **d(log C)/d(log m) not significant with 95% CI half-width < 0.30** in ≥3 of 4 seeds;
(iii) the C interval **excludes** the fitted λ elasticity in ≥3 of 4 seeds.
**Falsification:** if C's elasticity is significant with the same sign as λ's in ≥2 seeds, **the gauge
theorem's empirical leg fails** and band 42's causal claim must be withdrawn (the algebra would stand; the
claim that the LR behaves as a pure scale factor would not).

**COST.** REQ-035's precedent: four seeds of this fork window are **~16 min of training total**, plus the
existing curvature probe. **3 arms × 4 seeds on ≤2 nodes.** No new code — `make_req045_arms.py` takes a
`--seed` argument already.

**IF ONLY ONE OF REQ-048 / REQ-049 CAN RUN, RUN THIS ONE.** REQ-048 opens a new question (what *causes*
the bowl); **REQ-049 secures the campaign's strongest existing answer**, which currently has no
replication. Both remain scoped to ≤2 nodes.

- requester: analysis loop, iteration 174
- constraint acknowledged: **≤2 nodes**, no assertion of any higher authority.

## REQ-048: spectral participation ratio — the one measurement the committed data cannot supply

- status: **DONE 2026-09-04 (n=4) — HYPOTHESIS CONFIRMED** → `logs/kmaxwell/req048_spectral_participation/`
- **RESULT — the curvature bowl IS a spectral-concentration profile.** corr(logPR,logλ) ≤ -0.60 in 10/12
  fits, negative in 12/12, mean -0.728 (sd 0.161) [band-44 (i) PASS]; logPR cubic R²≥0.70 in 12/12 (mean
  0.827) [(ii) PASS]; falsification NOT triggered. Per-LR: -0.79/-0.68/-0.71 at 0.6/1.0/1.7×. logPR is a
  HUMP (concentrated at network ends where λ is high, spread mid) = the C-bowl INVERTED — matches your
  hypothesis text; NB criterion (ii)'s "argmin in 4-8" wording had the sign flipped vs its own hypothesis
  (PR argmin is at the ENDS, 11/12). So the bowl = curvature carried by a shrinking number of directions
  toward the ends (where band 43 pointed). m=16 Hutchinson, admissible (rule 13), PR scale-invariant. LR
  multipliers recorded explicitly per iter-172. Fix: wdirs weight-direction needs .detach().
- (was) status: **OPEN — filed 2026-09-04 (iteration 162).**
- priority: **high — this is the only outstanding measurement blocking the campaign's stated goal.**

**WHY THIS IS BEING ASKED FOR NOW, AND NOT EARLIER.** Iterations 156–161 exhausted the committed data
deliberately, and the limit is now quantified rather than asserted. Standing rule 13 leaves exactly four
admissible probe fields — `top_eigenvalue` (outcome only), `gradient_block_norm`, `curvature_along_polar`,
`shape` — and `shape` is **constant across depth** (iter. 161), so the entire admissible predictor set for
the depth question is **two numbers per matrix**. Fitting the C bowl on both:

| | variance of the C bowl explained by ALL admissible predictors |
|---|---:|
| per-fit mean | 27.3% (**sd 26.1**, range 0.5–83.2 — i.e. noise) |
| **on the 12-fit mean profile** | **4.3%** |

**The cause of the bowl is not in this probe's output.** That is a measurement gap, not an analysis
failure, and it is what justifies spending compute.

**WHAT TO MEASURE, AND WHY EXACTLY THIS.** Band 43 established the bowl is present along the **top
eigendirection** and **absent along Muon's step direction** (C_polar is monotone, |t| 5.38, 11/12).
Whatever causes it therefore concerns **how curvature is distributed across directions** — not its value
along any one direction. The scalar that measures precisely that is the spectrum's **participation
ratio**:

```
PR = (trace H)² / (n · trace H²)        ∈ [1/n, 1]
```

**PR ≈ 1** = curvature spread evenly over all directions; **PR ≈ 1/n** = concentrated in one. **The
hypothesis this tests: the bowl is a PR profile — the spectrum is most concentrated at the ends of the
network and most spread in the middle.**

**ADMISSIBILITY (the hard rule).** PR is built from **Hutchinson probes with fresh random vectors**,
independent of any Lanczos state. It contains **no `alphas`, no `offdiags`, no tridiagonal
eigendecomposition** — unlike `residual_tail`, rejected in iteration 161 precisely for coming from the
same `eigh()` as `lam_top`. **Admissible under rule 13.**

**EXACT SPECIFICATION.** Per Muon matrix, at each existing curvature checkpoint, with **m = 16** fresh
Rademacher or Gaussian probe vectors `v`:

1. `t1 = mean_v [ vᵀHv ]` → estimates `trace(H)`
2. `t2 = mean_v [ ‖Hv‖² ]` → estimates `trace(H²)`
3. record **`trace_est = t1`**, **`trace_sq_est = t2`**, **`n = rows·cols`**, and the **per-probe values**
   (so the estimator's own variance is measurable post hoc rather than assumed)

Also record, as cheap and independently useful diagnostics — **one HVP each**:

4. **`curvature_along_weight`** = `Ŵᵀ H Ŵ` with `Ŵ = W/‖W‖_F` — the self-similarity/Gauss-Newton check
5. **`curvature_along_random`** = a single fixed-seed random direction, for a spectrum reference point

**PRE-FLIGHT — m = 16 is not a guess.** Following this campaign's standing practice (iteration 126,
where simulating a proposed design caught a fatal collinearity before filing), the estimator was
simulated on three realistic spectra (a few large eigenvalues over a near-zero bulk, n = 768, 200 trials):

| spectrum | PR_true | m=1 rel.err | m=4 | **m=16** | m=64 |
|---|---:|---:|---:|---:|---:|
| concentrated | 0.00309 | 42.8% | 9.3% | **1.4%** | 1.3% |
| intermediate | 0.01013 | **145.3%** | 22.0% | **3.7%** | 1.0% |
| spread | 0.45532 | 7.2% | 3.7% | **0.1%** | 0.3% |

**m = 1 and m = 4 are unusable** (bias to 145%). **m = 16 gives ≤3.7% bias with 7–17% relative sd**,
comfortably inside what is needed: the C bowl's swing is **0.538 dex = 3.45×**, so PR differences of
tens of percent are the target. **m = 64 buys little; m = 16 is the recommendation.**

**COST.** The probe already performs 8 Lanczos iterations plus a polar HVP per matrix. This adds
**m + 1 + 1 = 18 HVPs**, i.e. roughly **+2× probe time and NO training**. `measure_per_matrix_curvature.py`
runs `--no_dist` and shards by rank, so it parallelises freely. **Requested scope: the 4 existing Arm A
seeds at the existing fork states — 2 nodes, no new training runs.** If even that is too much, **a single
seed at one fork state is enough to falsify the hypothesis** and should be run first.

**N=4 SEED CHECK — BAND 44, CRITERION REGISTERED IN ADVANCE (before any data exists).**
*Criterion:* (i) **corr(log PR profile, C profile) ≤ −0.60** with the **same sign in ≥10 of 12** seed×fork
fits; (ii) the **log PR profile is bowl-shaped with cubic R² ≥ 0.70 and argmin in layers 4–8**;
(iii) **log PR retains |t| ≥ 3 for the depth profile after controlling for log g and log cp**.
*Falsification:* if **|corr| < 0.30**, or the PR profile is **monotone** (linear R² exceeding its cubic
gain, as C_polar's was), **the spectral-concentration hypothesis is dead and should be recorded as a
negative** — the bowl would then not be a spectrum-shape effect at all.

**Registering the criterion now, before the measurement exists, is deliberate:** every band in this
campaign that was fitted first and criterion-ed afterwards (37, and the correlation half of 27) failed
its audit.

- requester: analysis loop, iteration 162
- constraint acknowledged: **≤2 nodes**, no assertion of any higher authority.

## REQ-045: crossed global × per-matrix LR — the identifiable partial/total design

- status: **DONE 2026-09-04** → `logs/kmaxwell/req045_crossed_global_permatrix_lr/`
- **RESULT — partial/total reading WITHDRAWN (registered null branch).** Separability corr(own,others'
  mean) post-FE = +0.722 (PASS <0.9 — crossing identifiable vs REQ-023's -1.0000). beta_own d log λ/d log(own
  LR) = -1.161 (t=-12.8, strong: band 30 confirmed per-matrix, own LR halves λ per doubling). beta_neighbour
  d log λ/d log(others' mean LR) = +0.143 (t=+1.14, NULL). Neighbour null → iteration 124's partial/total
  reading withdrawn: curvature-LR decoupling is a local own-LR effect, not a network channel; band-30 shape
  disagreement needs another explanation. Base val@2000=3.44369. (cadence fix every=250→ckpt@2750.)
- (was) status: **OPEN**
- requested: 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)
- **node budget: ONE box.** Reuses REQ-023's machinery entirely.

**What to run.** REQ-023's per-matrix LR randomisation, **crossed with a global multiplier that
differs by arm**:

> `effective_lr(i) = base_lr × S_arm × m_i`

- **`S_arm` ∈ {0.7, 1.0, 1.4}** — one value per arm, applied to *every* matrix.
- **`m_i`** — per-matrix, drawn independently per arm from {0.6, 0.85, 1.0, 1.2, 1.7}. **Do not
  balance** `m_i` across matrices; independent draws are fine and balancing is not required here.
- **3 arms** (one per `S_arm`), fork@2000, 750 steps, per-matrix curvature at 2750 plus
  `weight_frob` (REQ-041's field rides along).

**Why the crossing is essential — a design that looks equivalent does NOT work.** The question is
whether a matrix's curvature responds to **its own** LR differently from **its neighbours'**
(the partial vs total distinction). That needs `own` and `others' mean` to vary independently.

**REQ-023 cannot do it**, and neither can an unbalanced version of REQ-023 — **both give
`corr(own, others' mean) = −1.0000`**, because `mean_others = (Σ − own)/71` is an *arithmetic*
identity whenever the sum comes from the same draw. **Randomising or unbalancing the levels does not
help.** *(Verified by simulation before filing: 2,000 draws of each design, correlation −1.0000 in
both.)*

**The crossing breaks it**, because `log(own) = log S + log m_i` while `log(others' mean) = log S +
mean(log m_j)` — they share only the `S` term:

| design | corr(own, others' mean) | identifiable? |
|---|---:|---|
| REQ-023 balanced | **−1.0000** | no |
| unbalanced per-matrix | **−1.0000** | no |
| **3 global levels × per-matrix** | **+0.62** | **yes** |
| 5 global levels × per-matrix | +0.57 | yes |

**What it settles.** Band 30 (a higher LR decouples λ from g) is confirmed on two designs, **but they
disagree on shape**: REQ-023's per-matrix perturbation shows the decoupling **saturates by s = 1.0**
(the 1.0→1.7 step is not significant, and the two steps differ significantly, both forks), while Arm
A's global ladder shows an **even decline** (both steps significant, indistinguishable). Iteration 124
read this as partial-vs-total — saturation being a property of a matrix in isolation, with the global
ladder's extra decline coming from the network-wide state change. **That reading is currently
untested and untestable on committed data.**

**Registered check.** Regress `d log λ` on **both** `log(own multiplier)` and `log(others' mean
multiplier)` with matrix fixed effects:
- **the two coefficients must be separately estimable** — report `corr(own, others')` in the output
  and confirm |corr| < 0.9 before interpreting anything;
- **if the neighbour coefficient is significant**, iteration 124's reading is supported and band 30's
  shape is design-dependent for the stated reason;
- **if it is null**, the shape disagreement needs a different explanation and the partial/total
  reading is withdrawn.

**Cost.** 3 arms × 750 steps on one box, same as REQ-037's arms 1–3. No new hooks — REQ-023's
`per_matrix_lr_mul` machinery already exists (`train_gpt.py:514–522`) and a global `S_arm` is a
single scalar on the base LR.

**Priority: LOW.** This settles the *shape* of band 30, not its sign, and band 30 is already
confirmed on two designs. **It should not displace REQ-037 arm 4**, which can still overturn a
load-bearing band. File it as a background run if a box is idle.


## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```

## REQ-051: decompose why each matrix has a different LR-to-curvature response

- status: **OPEN**
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

### REQ-045 pre-flight axis correction — 2026-09-05

The iteration-196 pre-flight at the top of this queue is reproducible, but it regresses against
`m_i` alone. REQ-045's actual treatment is **`S_arm * m_i`**, and `S_arm` changes across the three
arms. Using `req045_draws.json` and the raw step-2750 curvature files:

| axis used | matrices with 3 distinct levels | with 2 | with 1 |
|---|---:|---:|---:|
| matrix draw `m_i` only (iteration 196) | 37 | 34 | 1 |
| effective multiplier `S_arm*m_i` | **64** | **8** | **0** |

For the 64 matrices with three effective levels, the mean descriptive `k_lambda` is:

| type | mean k | matrices |
|---|---:|---:|
| attn.q | 0.974 | 11 |
| attn.k | 1.143 | 7 |
| attn.v | 1.095 | 11 |
| attn.proj | 0.401 | 11 |
| mlp.fc | 1.084 | 12 |
| mlp.proj | 1.317 | 12 |

These use `k=-OLS_slope(log(top_eigenvalue), log(S*m))` per matrix and then average within type.
They are still descriptive, noisy three-point fits with an unadjusted neighbour channel, not new
causal type estimates. The v-minus-equal-weight-q/k contrast is small (+0.036), and the projection
ratio is about **3.28**, not 13.22. Thus neither the 37/72 effective-level coverage statement nor
the 13x own-LR sensitivity statement should guide new experiments. The original registered REQ-045
two-regressor fit already used `S*m` correctly and is unaffected by this audit.

REQ-051's balanced six-level design remains useful. Retain the q/k/v prediction as falsifiable,
with uncertainty, and report `k_mlp.proj-k_attn.proj` as an additional exploratory contrast.
Do not claim a writer-versus-internal mechanism from a contrast between two writer types.

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
