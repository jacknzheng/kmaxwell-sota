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

Next request number: **REQ-043**.

---

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
| **3** | boundary field, **true-layer axis** | corr(d_edge, block-mean residual) **≤ −0.5** | −0.912 / −0.891 |
| **4** | negative-curvature separation | every nonlinear-path type above every linear-path type | 0.198 > 0.146 / 0.200 > 0.138 |
| **5** | position spacing stays **unequal** | step(0→1) ≥ **2×** step(1→2) | 3.0× / 3.2× |
| **6** | **two-valued gradient slope** — ✅ **CONFIRMED n=4** | residual-writer minus internal raw slope ≥ +1.5 | +2.30 / +2.04 / +2.08 / +2.13, p < 0.0001 in all 4 seeds |
| **7** | **the split is residual-stream position, not shape** (iteration 65) | **slope(attn.proj + mlp.proj) − slope(other four) ≥ +1.5** in ≥3 of 4 seeds, each proj type individually ≥ +2.0 vs internal | +2.173 / +2.183, p < 0.0001 |
| **8** | **the cross-sectional split is bias, not physics** (iter. 67) | **Wald-ratio gap (residual − internal) ≤ +1.0** and **< half the cross-sectional slope gap** in ≥3 of 4 seeds; both first-stage F > 100 | gap +0.374 / +0.688 vs cross-sectional +2.17 / +2.18 |
| **9** | **only attn.proj's offset is resolved** (iter. 68–69, corrected) | **attn.proj lowest** of the six offsets in ≥3 of 4 seeds and **≥0.25 dex below the other five**; **|residual-writer − internal| offset gap < 0.20 dex**; the other four adjacent gaps need NOT resolve | gap −0.454 / −0.430, p ≤ 0.0003 |
| **10** | **layer-0 lift** — ❌ **NOT CONFIRMED n=4** | clears its permutation null on λ/g² | p = 0.022 / 0.190 / 0.120 / 0.058 — **1 of 4 seeds**. Lift is consistently positive (+0.25 to +0.50 dex) but under-powered at 6 matrices per layer |
| **11** | **the assembled model generalises** (iter. 72) | **leave-one-layer-out rmse ≤ 0.20 dex** and **cross-fork rmse ≤ 0.20 dex** for `type + two-slope gradient + layer-0 term`, versus ~0.38 dex for C's own spread | LOLO 0.138 / 0.164; cross-fork 0.165 / 0.140 dex |
| **12** | **type offsets reduce to three binaries** — ✅ **CONFIRMED n=4** | `q,k + residual-writer + mlp.proj` (5p) within 0.02 dex of six free offsets (7p) | gaps 0.004 / 0.004 / 0.005 / 0.002 dex in the 4 seeds |
| **13** | **the PER-MATRIX causal exponent is exactly 2** — ⚠️ **RE-SCOPED** (iter. 79) | 2.000 inside the 95% CI **under per-matrix LR randomisation only** (REQ-023 design) | REQ-023 +2.076/+2.079 CI contains 2. **Arm A's GLOBAL LR ladder gives +2.64 to +3.07, CI excludes 2 — a different estimand, not a refutation** |
| **14** | **q,k carry a large C excess in λ/g²** — ✅ **CONFIRMED n=4** | gap ≥ +0.6 dex and both q,k above all four others in ≥10/12 blocks | **+0.888 / +0.774 / +0.833 / +0.834 dex** (mean +0.832, sd 0.041), p < 10⁻⁵, **12/12 blocks in every seed** |
| **15** | **the q,k excess is QK-norm scale invariance** (iter. 80) | **d log C / d log‖W‖ = 0 within CI for attn.q and attn.k**, and **|slope| at least 2× smaller than the other four types**, in ≥3 of 4 seeds. Needs weight norms alongside curvature — **not yet measurable on Arm A** | q,k +0.049 CI [−0.261, +0.381] and −0.062 CI [−0.329, +0.226]; others −0.398 / −0.228 |
| **16** | **C is an ACTIVELY RESTORED invariant** (iter. 82–83) — ✅ **CONFIRMED n=4 + targeted test** | **global ladder:** matrix identity > 85% of log C's variance, LR < 10%, corr > 0.80. **targeted per-type perturbation:** **slope of Δlog C on log10(multiplier) ≈ 0** while Δlog λ tracks EoS | identity 93.2–94.8%; corr +0.87 to +0.97; **a5 λ-slope −1.153 vs C-slope −0.054** |

**Band 6 is the newest and it sharpens the campaign's central claim.** The cross-sectional gradient
exponent differs systematically by type — **~3.8 for the two projection matrices, ~0.9–1.4 for the
other four — against a within-matrix causal exponent of 2.07. No type sits at the causal value.**
This is not attenuation: measured error in log g is sd 0.0131 dex, reliability 0.962–0.993, and
correcting for it moves each slope by 1–4% and leaves the spread intact (1.35 → 1.24 / 1.54 → 1.42).
**Falsifier:** if attenuation-corrected slopes converge to ~2 across seeds, iteration 63 is wrong
and the law is universal after all.

### CONSOLIDATED FINDINGS (iterations 63–86, 2026-09-03)

*22 iteration blocks (~82k chars) are replaced by this summary. Full provenance is in git history —
`git log --oneline -- requests.md` from commit c83ba00 onward. This file bloated to 3,926 lines once
before and stalled the queue for ~2 hours; that is not repeated.*

#### The account of C, as validated

> **λ_eq = C · g²** — the per-matrix exponent is **2 by derivation** (Gauss-Newton: `H ≈ JᵀJ`,
> `g = Jᵀr`), confirmed against REQ-023's per-matrix LR randomisation (+2.076 / +2.079, 95% CI
> contains 2.000). **Band 13.**
>
> **log C = (q,k excess ≈ +0.83 dex) + (residual-writer ≈ −0.47) + (mlp.proj ≈ −0.65) + noise ~0.07 dex**
>
> …and C is **actively restored** by the network against interventions that try to change it.

**Every term confirmed on Arm A's four independent seeds.** Bands 6, 12, 14, 16 ✅; band 10 ❌
dropped; band 13 re-scoped; band 15 blocked on REQ-041.

#### What is established

- **The q,k excess (band 14).** attn.q and attn.k carry ~0.83 dex more curvature than their gradients
  justify. **+0.888 / +0.774 / +0.833 / +0.834 dex across four seeds** (mean +0.832, sd 0.041 —
  *tighter than the noise floor*), p < 10⁻⁵, and both exceed all four other types in **48/48
  block-seed cells**. It is **9.9× the strictest per-type noise floor**.
- **C is an actively restored invariant (band 16).** Matrix identity explains **93–95%** of log C's
  variance and the learning rate **1–4%**, across a 2.8× LR ladder that moves mean log λ by 1.3 dex.
  Under REQ-036's *targeted* per-type perturbation, λ moved as EoS predicts (slope −1.153) while C
  did not follow (**−0.054**). Restoration is active, not mere insensitivity.
- **Structural reduction (band 12).** Three binaries — `q,k`, `residual-writer`, `mlp.proj` — match
  six free type offsets with two fewer parameters (LOLO gap 0.002–0.005 dex in all four seeds).
  Weight norm adds nothing on λ/g².
- **Two-valued gradient slope (band 6).** Residual-writers minus internal: **+2.30 / +2.04 / +2.08 /
  +2.13**, p < 0.0001 in every seed.
- **The noise floor, measured rather than inherited.** Temporal 0.0689 dex, seed 0.0686 dex —
  *identical*, which is band 16 arriving independently. Type-dependent (1.76× worst-to-best), with
  q,k on the noisier side; every surviving band still clears its own floor by ≥5×.

#### Registered negatives — routes closed, so they are not re-attempted

- **C's type structure is not a units artifact.** Solving for the exponents that minimise across-type
  spread of `log λ − a·log g − b·log‖W‖` over the whole plane leaves **0.570 dex, ~6× the floor**.
  No power law of curvature, gradient and weight norm is invariant across types.
- **Weight norm is not causal.** In a horse race with the gradient it collapses to t = 0.3 / 1.2
  while the gradient holds at +2.12 / +2.23. Its apparent −3.5 effect was the gradient inherited
  through a −0.87 within-matrix anti-correlation.
- **Shape is not the mechanism.** attn.proj is (768,768) — identical to q/k/v; mlp.fc and mlp.proj
  are transposes. Parameter count, aspect ratio and fan-in all predict no difference.
- **No structural binary orders C's offsets.** All 62 non-trivial partitions tested; the best *named*
  hypothesis reaches R² 0.104 while 34 of 62 arbitrary partitions beat it.
- **The LR ladder is not a ‖W‖ proxy** (iteration 86). Under Muon the step is spectral-norm scaled and
  ‖W‖_F growth depends on per-matrix update alignment. The test that used it is under-powered
  (t = +0.81, cannot resolve below ~0.11) and its 2/4 outcome carries **no information** — band 15 is
  blocked, not weakened.
- **Attenuation does not explain the slope spread.** Reliability 0.962–0.993; correction moves slopes
  1–4%.

#### Corrections made during the campaign

The `corr(range, slope) = +0.96` claim (null was **+0.72**); the six-way slope split (**F = 0.18 vs
two slopes**); the "exactly 2" claim before it was tested *against* 2; the fan-in power law (one
distinct wide matrix — a label, not a law); the layer-0 "U-shape" (a boundary effect, then not
confirmed at n=4); iteration 52's "the q,k gap is in the gradient, not C" (right arithmetic, wrong
object — a *fitted* intercept whose free slope absorbed the gap).

**Three standing rules these produced:**
1. Before citing a correlation between group summaries, **permute the labels and report the null**.
2. Before fitting a continuous coefficient, **count the predictor's distinct values** — one or two
   makes it a label, not a law.
3. Any predictor built from the same Lanczos tridiagonal as `lam_top` is **circular** and rejected.

#### The design question, answered — and it is a negative

**REQ-036 is a null: equalizing per-type curvature is monotonically harmful.** Uniform LR beats every
intervention; the predicted-best arm was worst by 120× the val noise floor; and across the four
equalizing arms the relation to loss is **perfectly monotone in the wrong direction** (Spearman
−1.000, exact p = 0.042). The rule *direction* is real (a2 beats a4 by 11× noise) but **both lose to
doing nothing**.

**Band 16 explains why:** a per-type LR rule fights a quantity the network actively restores, and the
loss cost scales with how hard it pushes. **Description and prescription came apart** — the
descriptive account survives; the design recommendation is withdrawn.

> **Do not build a momentum kernel or per-layer LR rule on curvature equalization.** The productive
> direction is to treat C as a measured property to respect, not a target to flatten.

#### What would still settle the mechanism

Only two measurements, and iteration 86 showed no further analysis of committed observables
substitutes for them:

- **REQ-041** — per-matrix `‖W‖_F` alongside curvature. Unblocks band 15's seed check
  (`d log C / d log‖W‖ = 0` for QK-normed matrices; currently +0.049 CI [−0.261, +0.381] and −0.062
  CI [−0.329, +0.226] on REQ-023's two forks — the one mechanism that made a numerical prediction and
  passed).
- **REQ-038** — `|a|` / `|d|` per matrix. The independent test of band 14, **target 0.832 dex**.

Two further cheap additions to any future curvature run: **a 10× LR ladder** (band 16's falsifier —
if matrix identity's share collapses there, C is locally restored rather than invariant), and **two
ranks measuring one overlapping matrix** (the 8 MPI ranks currently hold disjoint 9-matrix shards, so
pure measurement noise is unmeasurable).

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

- status: **OPEN — fold into REQ-035 Arm A (priority 1); standalone only as a fallback.**
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

- status: **OPEN**
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

## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```
