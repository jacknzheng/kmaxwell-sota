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

Next request number: **REQ-048**.

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
| **3** | **a symmetric POSITION FIELD in C** (revised iter. 151) — ✅ **CONFIRMED n=4 (12/12), reading CORRECTED** | **A linear depth trend must NOT explain it** (this is what makes it survive rule 10) and the symmetric term must be significant in ≥10 of 12 seed×fork fits, on top of type offsets + log g | **linear depth t = +0.56 only, 6/12** vs **dist t = −5.92, 12/12**; AIC −228.96 vs baseline −198.88. ⚠️ **NOT a pure boundary shell** — the original seed-0 collapse (F 28.98→4.65) was **underpowered**: with both shells removed the interior U survives, t **+6.03 → +4.07 → +2.85**, 9/12. ⚠️ **Form NOT identified** — quad and logdist tie at AIC −230.02, dist within 1.1. ⚠️ **Edges UNEQUAL**: last block **+0.361 dex, 11/12** vs first **+0.211 dex, 6/12** — **this explains band 10's failure as a power problem, not a contradiction** |
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
| **26** | **C's six-type structure is genuinely THREE-term** (iter. 108) — ✅ **CONFIRMED n=4** | **the identity log g = log‖a‖_F + log‖d‖_F + log(align) holds exactly**; across the six types **no term correlates with C above 0.55**, and **each term's spread exceeds C's own** (offsetting) | identity exact to **1e-6 dex**; corr(C, −2‖a‖) +0.36–0.39, (C, −2‖d‖) +0.44–0.49, (C, −2align) +0.38–0.40; term spreads 1.35 / 1.02 / 0.64 vs C's **1.04** |
| **27** | **the ‖a‖·‖d‖ PRODUCT is the depth-conserved quantity** (iter. 110–111) — ✅ **CONFIRMED n=4, but RESTATED iter. 150 under rule 10** | ⚠️ **The correlation criterion is WITHDRAWN**: `corr(log‖a‖, log‖d‖) ≤ −0.80 across depth` is a **depth** correlation — removing depth collapses 4 of 6 types toward zero (attn.k **+0.065**, attn.q −0.184, attn.v −0.137, mlp.proj −0.143; only attn.proj −0.716 and mlp.fc −0.780 survive) and nothing is significant within-layer. **The matrix-level trade-off is NOT established; n=4 has no power to test it.** **The CONSERVATION criterion stands and is the real content:** within each type log‖a‖ rises and log‖d‖ falls **near-linearly** with depth (R² 0.80–0.99) at slopes that **cancel to ~8%**, so the product is conserved far beyond chance | **sd(sum) = 0.13–0.31× the independence null** √(sd_a²+sd_d²); all **24 type-seed slope ratios** −s_d/s_a mean **1.079**, sd 0.172, range [0.681, 1.307]; residual product drift only **−0.011 to +0.008 dex/layer**; cancellation slightly **over-complete** (t = +2.27 vs 1.0); **mlp.fc the lone under-compensator 0.755 ± 0.050, real in all 4 seeds** |
| **28** | **C's depth structure is carried by λ, not g** (iter. 117) — ✅ **CONFIRMED n=4** | **sd(log λ) / sd(log g) across depth > 1.5 in every matrix type**, every seed — the consistency requirement linking bands 27 and 10/25 | ratios **2.03 / 2.76 / 2.85 / 2.93 / 4.19 / 4.32**, mean **3.2×**; λ variance share 0.51–1.76 |
| **29** | **the λ–g relation WEAKENS as the LR rises** (iter. 120) — ✅ **CONFIRMED n=4** | **cross-sectional slope falls monotonically across s = 0.6 → 1.7**, seed-clustered CI on the spread excluding 0; **sd(log g) flat** (rules out range compression) while **sd(log λ) compresses** | slopes **0.916 / 0.742 / 0.636**, spread CI **[0.208, 0.365]**; sd(log g) **0.246/0.246/0.242**, sd(log λ) **0.429/0.395/0.367**; corr 0.534/0.497/0.453 |
| **31** | **Muon's gradient-magnitude invariance is a THEOREM, not an approximation** (iter. 131) | the update `X = g/(‖g‖·1.02 + 1e-6)` is exactly scale-free wherever the epsilon is negligible; **deviation < 1e-4 for ‖g‖ ≥ 1e-2**, and REQ-046's matrices sit at ‖g‖ ≈ 10³·⁸. Momentum adds a transient of ~1/(1−m) steps only | epsilon deviation at ‖g‖=10: **5e-8**; transient ≈ 20 steps of 750 |
| **32** | **C = λ/g² is the right object — the g² division REMOVES noise** (iter. 132) — ✅ **CONFIRMED n=4** | **C must be more seed-stable than λ** (median |Δ| across seed pairs) **and have a larger architecture-to-seed-noise ratio** | C **0.0776 dex** vs λ **0.1235**; ratio C **15.7×** vs λ **8.0×**; architectural spread C **0.462 dex** vs λ **0.383** |
| **33** | **Muon steps nearly ORTHOGONAL to peak curvature, and the RATIO predicts C** (iter. 134, 140) — ✅ **CONFIRMED n=4, ratio verified by free coefficients** | **cp/λ < 0.02 every type**; and in `C ~ a·log λ + b·log cp` with **free** coefficients, **b/a ≈ −1** (the ratio the data prefers unprompted) with **|t(b)| > 5** | ratio **0.001–0.006**; **b/a = −1.045/−1.301/−1.082/−1.193**, t(b) = **−8.8 to −11.0**; λ-only RSS **~2× worse** |
| **34** | **alignment and the q,k excess are SEPARATE effects** (iter. 135, 139) — ✅ **CONFIRMED n=4, raw-component checked** | **controlling for step-curvature shrinks the q,k C-coefficient by < 35%** — and **by ≤ 0% when the control uses raw `log cp`** (no shared λ) | shrinkage with alignment **22/19/16/20 %**; with `log cp` **−5/−7/−3/−9 %**; `log cp` net of q,k only t = **1.6–2.9** |
| **35** | **q and k differ in step ALIGNMENT, and the gap is DEPTH-STRUCTURED** (iter. 136–137) — ✅ **CONFIRMED n=4** | **|alignment(q) − alignment(k)| ≥ 0.15 dex, same sign every seed** (band 18's gradient equality is 0.014 dex); **gap deepens with depth then recovers — quadratic R² > 0.5**. *Attention entropy does NOT explain it* | mean **−0.306 dex** (sd 0.037, 4/4); L0 **−0.134** → L8 **−0.474** → L12 −0.327; quadratic R² **0.563**; entropy t **+5.03 → +0.26** under quadratic depth |
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
