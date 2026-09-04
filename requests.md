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

Next request number: **REQ-044**.

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
| **15** | **QK-norm scale invariance** (iter. 80, 87–89) — ❌ **QUANTITATIVE PREDICTION FAILS** | predicts **Δlog g = −Δlog‖W‖**. Observed: predicted **+0.130**, actual **−0.417** — wrong sign, 3× the size. q,k sit mid-pack in ‖W‖. The `d log C/d log‖W‖ = 0` result stands but no longer explains the gap | ‖W‖: q +1.755, k +1.756 vs proj +1.778, v +1.809, mlp.proj +1.832, fc +2.124 |
| **16** | **C is an ACTIVELY RESTORED invariant** (iter. 82–83) — ✅ **CONFIRMED n=4 + targeted test** | **global ladder:** matrix identity > 85% of log C's variance, LR < 10%, corr > 0.80. **targeted per-type perturbation:** **slope of Δlog C on log10(multiplier) ≈ 0** while Δlog λ tracks EoS | identity 93.2–94.8%; corr +0.87 to +0.97; **a5 λ-slope −1.153 vs C-slope −0.054** |

**Band 6 is the newest and it sharpens the campaign's central claim.** The cross-sectional gradient
exponent differs systematically by type — **~3.8 for the two projection matrices, ~0.9–1.4 for the
other four — against a within-matrix causal exponent of 2.07. No type sits at the causal value.**
This is not attenuation: measured error in log g is sd 0.0131 dex, reliability 0.962–0.993, and
correcting for it moves each slope by 1–4% and leaves the spread intact (1.35 → 1.24 / 1.54 → 1.42).
**Falsifier:** if attenuation-corrected slopes converge to ~2 across seeds, iteration 63 is wrong
and the law is universal after all.

**=== ITERATION 107: THE mlp GAP DECOMPOSED — the second effect closes too ===**

*The q,k account is closed end to end. Band 20's mlp gap — the network's **other** gradient effect,
and a **forward** one — has never had the same treatment. REQ-043 now supplies every term at n=4.*

**The decomposition.** Unlike q,k, the mlp pair does **not** share an input, so all three terms are
live: `grad_gap = a_gap + d_gap + alignment_gap`.

| seed | grad gap | **a_gap** | d_gap | alignment |
|---|---:|---:|---:|---:|
| 0 | −0.3009 | **−0.4328** | +0.0503 | +0.0816 |
| 1 | −0.2928 | **−0.4280** | +0.0604 | +0.0747 |
| 2 | −0.3074 | **−0.4486** | +0.0528 | +0.0884 |
| 3 | −0.3060 | **−0.4315** | +0.0415 | +0.0840 |
| **mean** | **−0.302** | **−0.435** | **+0.051** | **+0.082 ± 0.006** |

**The forward term dominates and the alignment term runs the *opposite* way to q,k's** — **+0.082 ±
0.006 here versus −0.190 ± 0.007 there**, both with CIs excluding zero. The two effects are not
variations on one mechanism: in q,k alignment *deepens* the deficit, in the mlp pair it *offsets* it.

**And the forward term is the ReLU², quantitatively.** `mlp.proj`'s input is `ReLU²` of `mlp.fc`'s
output, while `mlp.fc` reads the block residual:

| | mlp.fc input | mlp.proj input (ReLU² output) | ratio |
|---|---:|---:|---:|
| `a_rms` | 0.6101 | 1.6623 | **2.725× = +0.435 dex** |
| `a_eff_rank` | 19.51 | 142.80 | **7.32× = +0.864 dex** |

**The ReLU² output's RMS ratio is +0.435 dex — matching the measured forward gap of −0.435 to within
0.001 dex.** Band 20's prediction from iteration 94, made before any of these fields were used, is
confirmed exactly. Squaring a rectified signal produces a larger *and* much higher-rank activation,
so the mechanism is scale **and** sparsity, not scale alone.

**Both of the network's gradient effects are now accounted for arithmetically, at n=4:**

> **q,k (attention):** −0.37 dex = **−0.18 backward attenuation** (softmax Jacobian) **− 0.19
> alignment**. Input shared exactly; the effect is entirely in the backward pass.
>
> **mlp:** −0.30 dex = **−0.435 forward** (ReLU² activation scale) **+ 0.05 backward + 0.08
> alignment**. The effect is entirely in the forward pass, with alignment partially offsetting.

**Two effects, two different mechanisms, opposite loci — and each closes to within measurement error
with no residual.** *(Note the mlp `a_gap` has a real depth slope, −0.0250 dex/layer at t = −4.96,
which is band 20's U-shape appearing in the forward activation. Its cause is not established here;
recorded as observed.)*

**This completes the descriptive account.** What set out as "what is C?" resolves to: **λ_eq = C·g²
with the exponent fixed by Gauss-Newton, C actively restored by the network, and C's type structure
reducible to two measured architectural effects — one backward and one forward — each decomposed to
its constituent terms at n=4.**

**=== ITERATION 106: REGISTERED NEGATIVE — attention entropy does NOT explain the alignment deficit ===**

*Iteration 105 closed the q,k arithmetic and left one question open: **why** do q,k's gradients
accumulate less coherently across tokens? REQ-038/043 recorded per-block **attention entropy** and
**qk-logit RMS** — statistics I had never used, and the natural candidates.*

**The hypothesis.** The softmax Jacobian is `diag(p) − ppᵀ`. Its structure depends on how spread the
attention distribution is, so diffuse attention (high entropy) should mix many tokens, make
successive backward vectors point in different directions, and produce **low alignment**.

**The raw correlations look strong:**

| | vs `\|d\|` deficit | vs `d_eff_rank` deficit |
|---|---:|---:|
| attention entropy | −0.545 | **−0.786** |
| qk-logit RMS | +0.468 | +0.617 |

**But entropy is nearly a function of depth** — it falls monotonically from **4.91 nats at layer 0 to
0.25 at layer 12**, with `corr(entropy, layer) = −0.861`. This campaign has been caught by exactly
this shape of confound before (band 7's collinear pair, band 12's one-value fan-in), so the raw
numbers mean nothing until depth is partialled out.

**Partialling out depth, three of four collapse:**

| | raw | **partial (depth removed)** |
|---|---:|---:|
| `\|d\|` vs entropy | −0.545 | **+0.111** |
| `\|d\|` vs logit RMS | +0.468 | −0.316 |
| `d_eff_rank` vs logit RMS | +0.617 | −0.140 |
| `d_eff_rank` vs entropy | −0.786 | **−0.379** |

One survivor, and in a horse race it holds at `t = −2.7` against depth's `t = +2.1`. **With 12
distinct layers and 0.86 collinearity, that is a fragile regime, so it gets three stability checks
rather than a headline:**

| check | result |
|---|---|
| leave-one-layer-out | entropy `t` ranges **−4.89 to −1.57** — **crosses the threshold** |
| seed-clustered bootstrap | CI [−0.098, −0.085], excludes 0 |
| **depth allowed to be quadratic** | **entropy `t` flips sign: −2.7 → +2.13** |

**The third check is decisive. Allowing depth a quadratic term — a wholly reasonable
respecification — reverses the sign of entropy's coefficient.** A predictor that flips direction
under a benign change of functional form is not measuring anything; it is absorbing curvature in the
depth profile that a linear term left behind. Leave-one-out agrees, dropping to |t| = 1.57 on some
folds.

*(The bootstrap CI is the one check that looks supportive, and it is the least relevant here: it
resamples **seeds**, but the collinearity that threatens this estimate is across **layers**. Recording
that, because a CI excluding zero is easy to quote out of context.)*

> **Attention entropy and qk-logit RMS do not explain the alignment deficit. Their apparent
> relationship is depth, measured differently.**

**Registered as a negative.** The alignment deficit's cause remains open, and the space of committed
observables that could address it is now exhausted — entropy and logit RMS were the last two
unexamined fields in REQ-043. **Answering "why does the softmax Jacobian's output align less well
across tokens" requires a measurement nobody has made**: the per-token backward vectors themselves,
or a decomposition of the alignment ratio by token position. That is a substantially heavier probe
than anything filed so far, and **I am not filing it** — the campaign's account is closed
arithmetically at n=4, and this last question is a research problem in attention dynamics rather than
a gap in the C account.

**=== ITERATION 105: REQ-043 P2/P3 LAND — band 25 resolved, and it is a RENAMING, not a mechanism ===**

**Jerry delivered the alignment ratio and the second training state.** Both were filed against
iteration 102's specification and iteration 103's correction. Both land — and the honest reading of
the result is narrower than the commit title suggests.

**The central point, stated first: the identity is true by construction.**

```
   align_ratio = ‖Σₜ dₜaₜᵀ‖_F / (‖d‖_F‖a‖_F) = ‖W.grad‖_F / (‖d‖_F‖a‖_F)
   log(align)  = log‖grad‖ − log‖d‖ − log‖a‖
   align_deficit = grad_deficit − d_deficit − a_deficit,  and a_deficit = 0 for q,k vs v (band 21)
   ⟹  align_deficit ≡ grad_deficit − d_deficit ≡ the shortfall
```

**Verified: max |align_deficit − shortfall| = 0.0004 dex across the four seeds.** So **the alignment
ratio does not *explain* band 25's missing factor — it *is* that factor, measured directly rather
than inferred.** Recording this plainly because the commit title ("alignment ratio IS band-25's
missing factor") is correct but could be read as a mechanism, and it is not one.

**What it genuinely buys — three real results:**

**1. A single-state measurement, 6× tighter.** Band 25 computed the shortfall by *differencing two
states*: −0.240 ± 0.041. Measured within one state:

> **align_deficit = −0.1896 ± 0.0068 dex (0.646×), n=4** — across-seed sd falls **6.0×**, and the
> cross-state artifact is gone entirely.

**2. Iteration 103's correction is confirmed, quantitatively.** That iteration argued the cross-state
value was inflated and predicted a corrected slope of −0.0117 with CI **[−0.0169, −0.0064]**. The
artifact-free measurement is **−0.0075 dex/layer — inside the predicted interval.** The filed −0.240
mean was inflated by 0.050 dex relative to the artifact-free −0.190, in the direction predicted.

**3. The state-dependence is confirmed directly, not inferred.** Jerry probed seed 0 at fork-2000 as
well as fork-1500:

| state | align_deficit | depth slope |
|---|---:|---:|
| fork-1500 | −0.1845 | **−0.00909** |
| fork-2000 | −0.1905 | **−0.00405** |

**The depth slope flattens by +0.0101 dex/layer per 1000 steps** — so it *is* state-dependent, and a
cross-state comparison manufactures part of the trend, exactly as iteration 103 warned. **The mean is
stable across states (−0.1845 vs −0.1905); only the slope moves.**

**Band 25 is resolved and restated.** The missing factor is the **token-wise alignment between the
backward and forward tensors** — q,k's gradients accumulate less coherently across tokens than v's,
at 0.646× the alignment. **That is a real, measured, seed-stable quantity (sd 0.0068 dex) and it
closes the arithmetic exactly.** What remains open is *why* the softmax Jacobian's output aligns less
well across tokens — a question about attention dynamics, not about the gradient bookkeeping, which
is now complete.

**The campaign's account of the q,k effect is now closed end to end:**

> **q,k gradient deficit (−0.37 dex) = backward attenuation (−0.18, band 21, softmax Jacobian) +
> alignment deficit (−0.19, band 25).** Both terms measured at n=4, both seed-stable, and the two
> sum to the whole with no residual.

**=== ITERATION 104: AUDITING EVERY BAND AGAINST THE NEW RULE — and verifying REQ-043's seeds are real ===**

*Iteration 103 added a standing rule after band 25 was caught by it: **a confound cleared for one
statistic is not cleared generally.** A rule is only worth adding if it is then applied, so this
iteration applies it to all thirteen live bands.*

**Exposure audit.** The two data sources sit at different states — the REQ-038/043 probe at
fork-1500, Arm A's curvature at steps 2250–2750. A band is exposed **only if it combines quantities
from both**:

| band | claim | source | exposed? |
|---:|---|---|---|
| 6, 12, 14, 17, 18, 19, 20, 22, 23, 24 | *(ten bands)* | **Arm A only** | no |
| 16 | C actively restored | Arm A + REQ-036 | no — both measure curvature the same way |
| 21 | deficit purely backward | **REQ-043 only** | no |
| **25** | the magnitude shortfall | **Arm A + REQ-043** | **EXPOSED** |

**Band 25 is the only exposed band, and it is already amended.** Band 21 in particular is
self-contained: `a_rms` and `d_rms` come from the *same* forward/backward pass, so its comparison is
q,k versus v **within** one measurement and no cross-state term can enter. **The rule does not bite
elsewhere — now verified rather than assumed.**

**A second check the audit surfaced, and it was worth running.** The audit printed **identical model
paths** for all four REQ-043 seeds (`eos_shared_state/train_state_model_step001500.pt`, no seed in
the name) and **near-identical losses** — 29288.9 / 29262.7 / 29207.2 / 29262.8, a spread of 0.28%.
Independent networks should differ more than that. **If REQ-043 had probed one network four times,
band 21's n=4 would be n=1 dressed up**, and iteration 101's headline would be wrong.

**They are genuinely distinct networks:**

| check | result |
|---|---|
| `weight_frob` of `blocks.0.attn.q.weight` | 56.870 / 57.834 / 57.512 / 57.358 |
| `d_rms` of the same matrix | 0.005742 / 0.006217 / 0.005666 / 0.005208 |
| **matrices with identical `weight_frob` between any seed pair** | **0 / 72, all six pairs** |

**Not one of 72 matrices shares a weight norm between any two seeds.** The shared path is a naming
convention (each seed's run writing to its own directory), and the tight loss spread reflects a
training recipe that converges reliably — which is itself consistent with Arm A's central finding
that C is seed-independent. **Band 21's n=4 stands.**

**Why this iteration was worth spending on verification rather than a new hypothesis.** Two of the
campaign's most-cited results — band 21's n=4 confirmation and the ten Arm A-only bands' immunity to
the state confound — rested on assumptions that had never been checked. Both hold. **The cost of
checking was one iteration; the cost of not checking would have been a headline claim of n=4
replication that was actually a single network measured four times.**

**=== ITERATION 103: QUALIFYING BAND 25's DEPTH TREND — the state confound's untested half ===**

*Iteration 98 tested whether the fork-1500 vs 2250–2750 state mismatch explains the shortfall's
**size**. It does not — it widens the gap. But the shortfall's most distinctive feature is its
**depth trend**, and that was never tested against the same confound. It should have been.*

**The test.** The shortfall is `(Arm A gradient deficit at 2250–2750) − (REQ-043 |d| deficit at 1500)`.
If the gradient deficit's **depth profile** evolves between those states, comparing across them
manufactures a depth trend. Arm A's five steps let this be measured directly — 60 (seed, arm, step)
cells, each giving a depth slope:

| step | mean depth slope of the gradient deficit |
|---:|---:|
| 2250 | −0.00237 |
| 2375 | −0.00569 |
| 2500 | −0.00538 |
| 2625 | −0.00453 |
| 2750 | −0.00666 |

**The depth slope drifts: −0.00593 dex/layer per 1000 steps, seed-clustered 95% CI [−0.01119,
−0.00071] — excludes zero.**

**Quantifying the damage:**

| | dex/layer |
|---|---:|
| shortfall depth slope (band 25 as filed) | **−0.0176** |
| attributable to the state mismatch (1000-step gap) | **−0.0059** (CI −0.0007 to −0.0112) |
| **remaining after correction** | **−0.0117** (CI −0.0064 to −0.0169) |
| **share that is a state artifact** | **~34%** (CI 4% to 64%) |

**About a third of band 25's depth trend may be an artifact of comparing two training states**, with
wide uncertainty. **The remaining −0.0117 dex/layer is not attributable to it** — the trend is real
but smaller than filed.

**What is unaffected.** The shortfall's **size** (−0.240 dex, across-seed sd 0.041) stands: iteration
98 established the state mismatch *widens* that gap rather than creating it. Only the **slope** is
qualified. Band 25 is amended to separate the two claims explicitly, since they have different
evidential status.

**Why this matters for the specification.** Iteration 102 handed the alignment ratio a precise target
including `−0.0176 dex/layer`. **That target was overstated.** The corrected target is **−0.0117
dex/layer**, and the honest position is that the campaign **cannot pin the depth slope more tightly
than [−0.0064, −0.0169] from committed data**. Handing a candidate mechanism a spuriously precise
number to hit would have been a way to reject it wrongly.

**This raises REQ-043 priority 2 from "resolves a caveat" to "required".** The probe at a second
training state is no longer a nice-to-have that settles an aside — **it is needed to state band 25
correctly.** Without it, the depth trend carries a 4%–64% uncertainty that no amount of further
analysis on committed data can reduce, because both quantities are measured at fixed, different
states.

**Method note worth keeping.** Iteration 98 tested the state confound against the shortfall's *size*
and cleared it, and I treated that as clearing the confound generally. It did not — a confound can be
harmless for one statistic and material for another, and each statistic derived from a mismatched
comparison needs its own check. **Added to the standing rules.**

**=== ITERATION 102: SPECIFYING THE MISSING FACTOR — and re-rejecting d_eff_rank at n=4 ===**

*Band 25 established the shortfall is real and reproducible. Iteration 97 had rejected `d_eff_rank`
as its explanation **using one seed**. With four seeds and a known target profile, that rejection is
worth redoing properly — and the shortfall itself is worth characterising precisely, since that
specification is what any candidate mechanism must meet.*

**`d_eff_rank` re-tested at n=4 — rejected, and now for a clearer reason:**

| | shortfall | `d_eff_rank` ratio |
|---|---:|---:|
| depth slope | **−0.0176 (t = −8.56)** | **+0.0110 (t = +4.25)** |
| correlation with shortfall | — | **−0.346** |
| R² of the best power-law fit | — | **0.120** |

**The two depth trends run in opposite directions** — the shortfall *deepens* toward the output while
the rank ratio *shrinks*. The fitted coefficient is also **negative**, which would require lower rank
to *reduce* the gradient. **No power of `d_eff_rank` can produce the observed profile**, and this now
rests on 48 layer-seed cells rather than 12.

**The shortfall's profile is well-determined, and it is not a boundary artifact:**

| form | R² |
|---|---:|
| constant | 0.000 |
| **linear in depth** | **0.763** |
| quadratic | 0.800 |
| final-layer indicator alone | 0.539 |
| linear + final-layer | **0.900** |

`shortfall = −0.135 − 0.0176 × layer`, slope **t = −5.67**. **Excluding layer 12 entirely the slope is
−0.0137 with t = −5.71 — the trend survives**, so this is a genuine depth dependence with an
*additional* final-layer excess, not a single outlier driving a spurious line. *(Contrast band 20's
mlp gap, which was U-shaped; this one is monotone.)*

**The specification any candidate mechanism must meet:**

| property | value |
|---|---:|
| mean | **−0.240 dex** (factor **0.575×**) |
| at layer 0 | −0.146 dex |
| at layer 12 | −0.432 dex |
| depth slope | −0.0176 dex/layer, **t = −5.67** |
| across-seed reproducibility | **0.041 dex** |
| **not explained by** | `d_rms`, `d_frob`, `a_rms`, `a_frob`, `a_eff_rank`, `d_eff_rank`, weight norm, parameter count, `shape_mult`, or the training-state difference |

**This is the campaign's open problem stated as precisely as the data allows.** Ten candidate
quantities are excluded, four of them at n=4. The remaining candidate — **token-wise alignment
between `d` and `a`** — is the one quantity that could plausibly deepen with depth, since deeper
layers' backward signals are shaped by more intervening blocks, and it is a single extra scalar in a
probe Jerry has already written and validated twice.

**REQ-043's priority 3 is re-filed against this specification** rather than as a general request:
**the alignment ratio must average −0.240 dex, sit at −0.15 at layer 0 and −0.43 at layer 12, and
reproduce across seeds to ~0.04 dex.** If it does not, the factor is something else again — and this
band's numbers are precise enough that the answer will be unambiguous either way.

**=== ITERATION 101: REQ-043 LANDS — band 21 at n=4, and the shortfall is REAL ===**

**Jerry delivered REQ-043.** Band 21 — the campaign's last result stuck at n=1 — is now confirmed on
four independent seeds:

| | result |
|---|---|
| `a_rms` for q, k, v | **bit-identical in all 4 seeds** |
| `d_rms` ratio (q,k)/v | **0.667 ± 0.011** |
| log₁₀(q/v) | −0.183, **t = −41.9** |
| log₁₀(k/v) | −0.170, **t = −43.3** |

**Every confirmed band in this campaign is now n=4.** Jerry attributes the backward deficit to the
**softmax Jacobian**, which is a sharper mechanism than "RMS-norm rescaling" and consistent with every
band 14–21 result: it acts only on q,k (the logit path), is architectural rather than learned
(band 17), and treats q and k identically (band 18).

**The alignment ratio was not included** — REQ-043 carries the same fields as REQ-038 — so the
magnitude gap stays open. **But n=4 makes something new testable: whether the shortfall itself
reproduces.**

| seed | `\|d\|` deficit | gradient deficit | **shortfall** |
|---|---:|---:|---:|
| 0 | −0.180 | −0.393 | **−0.213** |
| 1 | −0.168 | −0.399 | **−0.231** |
| 2 | −0.183 | −0.366 | **−0.184** |
| 3 | −0.176 | −0.401 | **−0.225** |

> **The shortfall is −0.213 ± 0.021 dex — a factor of 0.61× — reproducing across four independent
> networks with a scatter three times *below* the 0.07 dex noise floor.**

**This converts iteration 97's negative into a positive result.** "We cannot reconstruct the
magnitude" becomes **"there is a specific, reproducible physical factor of 0.61× that REQ-038's
fields do not capture."** A missing term that reproduces to 0.02 dex across independent networks is
not measurement scatter — it is a real quantity with a definite value.

**And it has clean depth structure**, growing monotonically toward the output:

```
  L0 −0.146   L1 −0.151   L2 −0.221   L3 −0.125   L4 −0.215   L5 −0.243
  L7 −0.251   L8 −0.259   L9 −0.265   L10 −0.276  L11 −0.302  L12 −0.432
```

Structure-to-noise **2.0×** (across-layer sd 0.082 vs across-seed sd 0.041). **The missing factor is
not a constant — it strengthens with depth**, which is itself a constraint on what it can be: any
candidate mechanism must produce a ~0.6× attenuation that deepens toward the output.

**Registered as band 25.** This is the sharpest statement of the open problem the campaign has
produced: not "the magnitude is unexplained" but **"a reproducible 0.61× factor with a monotone depth
profile is missing from the backward-pass account."** REQ-043's priority-3 alignment ratio remains the
leading candidate and is now better motivated — **the target it must hit is −0.213 dex on average and
−0.43 at the final layer.**

**=== ITERATION 100: VALIDATING THE WORD "EQUILIBRIUM" — the campaign's core assumption, finally tested ===**

*Band 23 concluded the g² law has no time dynamics. That conclusion is only meaningful if the
measurement window is genuinely at **equilibrium** — if matrices were still **relaxing**, band 23
would be describing a transient, or a window placed too late to see one. **Every band in this
campaign rests on `λ_eq` measured over steps 2250–2750, and the word "equilibrium" has been assumed
throughout, never tested.***

**The discriminator.** A relaxing system approaches its fixed point monotonically, with **shrinking**
step-to-step changes and **positive** autocorrelation of successive differences. An equilibrated one
fluctuates around the point: **negative** autocorrelation (mean reversion), flat change magnitudes.

| test | result |
|---|---|
| autocorrelation of successive Δlog λ | **−0.228, 95% CI [−0.437, −0.117]** |
| change magnitude, 2nd half / 1st half | **0.898** (relaxation predicts ≪1) |

**Mean-reverting, with flat change magnitudes across the window. The system is fluctuating around a
fixed point, not approaching one.**

**But mean reversion alone is not enough, because pure noise fakes it.** If `λ` were constant plus
white measurement noise, the differenced autocorrelation would be **exactly −0.5**. The observed
−0.228 sits *between* white noise (−0.5) and a drifting random walk (0.0), and that position is
informative — for an AR(1) around a fixed point with persistence ρ, the differenced autocorrelation
is `(ρ−1)/2`:

> **ρ = 2(−0.228) + 1 = 0.54.** The series retains **54% of a deviation across 125 steps**. That is a
> system with **real persistence** — not white noise — that is nonetheless **mean-reverting rather
> than drifting**.

**Registered as band 24.** The campaign's foundational assumption is now measured rather than
asserted, and it holds.

**It also explains, rather than merely coexisting with, band 23's need for a reliability correction.**
The within-window fluctuation of log λ is **0.040 dex median**, against a **0.07 dex noise floor** —
the real time-variation is *smaller than the measurement noise*. That is precisely why band 23's
regression of λ-drift on g-drift was attenuated (reliability 0.738) and had to be corrected before it
could be interpreted. **The two findings are one fact seen twice: there is little genuine
time-variation to detect, and what exists does not follow the g² law.**

**What this closes.** The campaign's central law is now bounded on all sides: it holds
**cross-sectionally** (band 8), **causally under LR perturbation** (band 13, exponent exactly 2), and
**not in training time** (band 23) — with the time-domain test now confirmed to have been run on a
genuinely equilibrated system (band 24) rather than a transient. **"Equilibrium curvature" means what
it says**: a fixed point the matrices fluctuate around with ρ ≈ 0.54, restored against LR
perturbation (band 16), invariant across seeds (Arm A) and across training time (iteration 99).

**=== ITERATION 99: BAND 16 SURVIVES A TIME TEST — but the g² law does NOT ===**

*Band 22 found the q,k gradient deficit drifts during training. That raised a question about band 16:
its homeostasis claim was tested against the **learning rate**, never against **time**. If C drifts
within the equilibrium window, "invariant" needs qualifying.*

**Band 16 survives cleanly.** Adding `step` to the variance decomposition:

| seed | matrix identity | learning rate | **step** |
|---|---:|---:|---:|
| 0 | 88.3% | 2.5% | **0.4%** |
| 1 | 90.1% | 3.5% | **0.2%** |
| 2 | 86.7% | 3.0% | **0.2%** |
| 3 | 84.3% | 2.0% | **0.3%** |

**Training time explains an order of magnitude less of C's variance than the learning rate does**, and
the pooled per-matrix drift is **−0.013 dex/1000 steps, 95% CI [−0.055, +0.049]** — includes zero.
Over the 500-step window Arm A measures, C moves **0.007 dex** against a 0.07 dex floor. **C is
invariant in training time as well as in learning rate.**

**But that creates a sharp problem, and the answer is a genuine negative.** Since `C = λ/g²`
identically, if **g drifts** (band 22, CI excludes zero) and **C does not**, then **λ must drift at
exactly twice g's rate**. That is band 13's g² law asserting itself in a domain no previous test has
used. It fails:

| | value |
|---|---:|
| slope of λ-drift on g-drift, 864 matrices | **+0.634 ± 0.110** |
| seed-clustered 95% CI | **[0.329, 0.982]** — **2.000 outside** |
| correlation | +0.192 |

**And this is not an under-powered test — I checked before concluding.** Each drift is fitted from 5
points, so regressing one noisy estimate on another attenuates the slope. Decomposing:

| | |
|---|---:|
| g-drift estimate variance | 0.0126 = **0.0093 true** + 0.0033 measurement |
| **reliability ratio** | **0.738** |
| a true slope of 2.000 would be observed as | **1.476** |
| **observed** | **0.634** |
| attenuation-corrected | **0.860** |

**A true exponent of 2 would have shown up as 1.476, more than double what was observed**, and the
corrected estimate is 0.860. **The g² law does not hold in the time domain.**

> **λ ∝ C·g² holds cross-sectionally and causally (bands 13, 8) but NOT as training proceeds. Within a
> matrix over time, curvature and gradient move nearly independently (corr +0.19), and C stays
> constant because both drifts are small — not because they are locked in a 2:1 ratio.**

**Registered as band 23.** This sharpens what the campaign's central law actually claims. It is a
statement about **equilibrium states** — where a matrix settles under a given learning rate — not a
dynamical law governing the path between them. **The distinction was never tested before this
iteration**, and every earlier statement of the law should be read with it: bands 13 and 8 are
untouched in their own domains, but "λ = C·g²" is not a constraint the network obeys moment to
moment.

**A caution on interpretation.** Both drifts are small (mean +0.0055 and −0.0019 dex/1000 steps), so
this is a test of *how* two small quantities co-move, not of a large effect. The reliability
calculation is what makes it admissible — without it, the observed 0.634 would be uninterpretable.
**The finding is that the co-movement is weak (corr +0.19), not that λ moves in some contrary
direction.**

**=== ITERATION 98: THE STATE CONFOUND IS ELIMINATED — and it makes the shortfall worse ===**

*Iteration 97 concluded the missing 0.32 dex is a physical quantity REQ-038 does not measure. Before
accepting that, one mundane alternative was still open and **named in iteration 96's own caveats**:
REQ-038 probes fork-1500, Arm A measures at steps 2250–2750. If the deficit **grows** over those
750–1250 steps, the residual is a state difference, not a missing term.*

**Arm A measures at five steps inside its window, so this tests directly at zero cost.** Pooling all
4 seeds × 3 LR arms × 5 steps = **60 measurements**, with a seed-clustered bootstrap:

| LR arm | mean deficit | drift |
|---|---:|---:|
| s = 0.60 | −0.368 | **+0.029** dex/1000 steps |
| s = 1.00 | −0.374 | **+0.092** |
| s = 1.70 | −0.359 | **+0.054** |
| **pooled** | **−0.367** | **+0.058, 95% CI [+0.032, +0.085]** |

**The drift is real — CI excludes zero, same sign in all three arms — but it points the wrong way to
help.** The deficit **shrinks** as training proceeds, so extrapolating back to step 1500 makes it
**larger** (≈ −0.47 dex), while REQ-038's `|d|` deficit at that same step is **−0.182**. **The state
difference widens the discrepancy from 0.32 to roughly 0.28→0.42 dex rather than closing it.**

> **The state confound is eliminated. Iteration 97's conclusion stands and is strengthened: the
> magnitude of the q,k gradient deficit is not reconstructible from anything REQ-038 measured, and
> not an artifact of comparing two training states.**

**And the drift is itself a new result, registered as band 22.** No previous band tested time
evolution — band 17 established the deficit is flat in *depth*, and this establishes it is *not* flat
in *training time*. The two are independent: **a fixed architectural attenuation is depth-flat, as
observed, but need not be constant across training**, since what the backward signal carries changes
as the model learns.

**A caution on the per-seed evidence, worth recording.** Per-seed slopes were +0.058, +0.052, +0.080
and +0.179 dex/1000 steps, and **only one cleared t = 2 individually** (seed 3, t = 3.63). With n=5
steps per seed the per-seed test is under-powered — the result rests on **pooling 60 measurements and
the consistency of sign across three independent LR arms**, not on any single seed. Band 22's
registered check is written on the pooled CI and the three-arm sign agreement accordingly.

**What this leaves.** Three reconstructions have failed (type-mean, per-layer, rank-augmented) and now
the state confound is excluded. **The missing factor is 0.48×, grows with depth, and is not explained
by `|d|`, `‖d‖_F`, `d_eff_rank`, or the training-state difference.** REQ-043's priority 3 — the
alignment ratio `‖Σₜ dₜaₜᵀ‖_F / (‖d‖_F‖a‖_F)` — remains the one candidate with the right shape, and
this iteration removes the last alternative explanation that could have made it unnecessary.

**=== ITERATION 97: REGISTERED NEGATIVE — REQ-038's fields cannot reconstruct the deficit's size ===**

*Iteration 96 left the q,k deficit's magnitude unexplained. Two of REQ-038's fields were still
unused. Both are now tested, and both fail — which closes this route on committed data and says
precisely what a new measurement must look like.*

**Route 1: the Frobenius fields.** The gradient is `G = dᵀa`, so `‖G‖_F ≤ ‖d‖_F‖a‖_F` — the
Frobenius norms, not the RMS values, are the right scale. But they turn out **exactly proportional**
to the RMS fields:

| field | q,k | v | ratio |
|---|---:|---:|---:|
| `a_rms` / `a_frob` | 1.00364 / 2517.4 | 1.00364 / 2517.4 | **1.0000** (both) |
| `d_rms` / `d_frob` | 0.00262 / 6.580 | 0.00399 / 10.000 | **0.6580** (both) |

**Identical ratios — the Frobenius fields carry no information the RMS fields did not.** Same
−0.182 dex, same shortfall. Route closed.

**Route 2: the rank asymmetry.** One genuine asymmetry remains — `a_eff_rank` is **identical** (30.17
for q, k and v alike, as the shared input requires) while `d_eff_rank` differs (74.3 vs 106.2). Since
`G = Σₜ dₜaₜᵀ`, how the outer products accumulate should matter. Testing `|d| + p·log(d_eff_rank)`
per layer, for several powers p:

| model | mean | **corr with gradient deficit** | depth slope t |
|---|---:|---:|---:|
| `\|d\|` only | −0.168 | +0.351 | +3.46 |
| `\|d\|` + 0.5·log(rank) | −0.194 | **+0.047** | +4.09 |
| `\|d\|` + 1.0·log(rank) | −0.220 | **+0.006** | +3.83 |
| `\|d\|` − 0.5·log(rank) | −0.142 | +0.169 | −2.54 |
| **observed deficit** | **−0.487** | 1.000 | **−0.12** |

**Adding the rank term makes the fit worse, not better** — correlation with the target falls from
+0.351 to +0.006 — and no power fixes the depth slope. The observed deficit is depth-flat (t = −0.12,
per band 17) while every `|d|`-based model trends strongly (t ≈ +3.5 to +4.1). **A rank term would
have to carry a depth trend that exactly cancels `|d|`'s, and it does not have one.**

**What the missing factor must look like — the useful output of this iteration:**

| property | value |
|---|---:|
| size | **−0.319 dex** (a factor of **0.48×**) |
| depth behaviour | **grows with depth**, slope t = −2.09 |
| correlation with `d_eff_rank` | **−0.490** (wrong sign to help) |

**REQ-038 measured nothing with that shape.** The deficit's size is set by something the probe does
not capture — most plausibly the **token-wise alignment** between `d` and `a`, which determines how
`Σₜ dₜaₜᵀ` accumulates and is *not* recoverable from per-tensor norms or spectra. **That is a
measurable quantity** — e.g. `‖Σₜ dₜaₜᵀ‖_F / (‖d‖_F‖a‖_F)`, the alignment ratio — and it is a single
extra scalar per matrix in a probe that already exists.

**Registered as a negative, and REQ-043 amended a second time.** The request now asks for three
things, in priority order: **(1)** the existing probe on seeds 1–3 (band 21 at n=4); **(2)** the
probe at a second training state (iteration 96's caveat); **(3)** **the alignment ratio
`‖Σₜ dₜaₜᵀ‖_F / (‖d‖_F‖a‖_F)` per matrix** — the one quantity that could close the magnitude gap, and
the direct output of iteration 97's negative.

**Campaign status.** The locus of the q,k effect is settled beyond doubt (`|a|` identical at all 12
layers; the effect is purely backward). **Its magnitude is unexplained by every quantity currently
measured**, and this iteration establishes that as a property of the *data*, not of the analysis —
three independent reconstructions have now failed, each for a different reason.

**=== ITERATION 96: THE PER-MATRIX DATA BREAKS THE RECONCILIATION — d_rms does not track the deficit ===**

*Iteration 95 used REQ-038's **type means** and reconstructed the gradient deficit as `|d|` (−0.182)
plus a rank term (−0.078) = −0.260 against an observed −0.373, leaving ~30% unexplained. REQ-038's
JSON has **per-matrix** data. Using it breaks the reconstruction rather than completing it.*

**First, a result that strengthens band 21's locus claim.** `a_rms` for q, k and v is not merely
equal on average — it is **identical at every one of the 12 layers**:

```
  a_rms deficit (q,k vs v), by layer:
  L0 +0.000  L1 +0.000  L2 +0.000  L3 +0.000  L4 +0.000  L5 +0.000
  L7 +0.000  L8 +0.000  L9 +0.000  L10 +0.000  L11 +0.000  L12 +0.000
```

**12/12 layers, exactly zero.** The shared normed-residual input is shared exactly, everywhere. The
q,k effect is purely backward — that part of band 21 is stronger than the type means showed.

**Now the problem.** `d_rms`'s deficit is **not depth-flat**, while the gradient deficit is:

| | mean | depth slope | t |
|---|---:|---:|---:|
| gradient deficit (Arm A, seed 0) | −0.397 | −0.00583 | **−1.18** (flat, per band 17) |
| **`d_rms` deficit (REQ-038, seed 0)** | −0.156 | +0.01174 | **+6.65** (strong trend) |

**And layer-by-layer they are uncorrelated: corr = −0.012.**

| layer | gradient deficit | d_rms deficit | difference |
|---:|---:|---:|---:|
| 0 | −0.354 | −0.230 | −0.124 |
| 5 | −0.478 | −0.173 | −0.305 |
| 12 | −0.508 | −0.091 | **−0.417** |

**`d_rms` shrinks steadily with depth while the gradient deficit does not — the unexplained residual
grows from 0.12 dex at layer 0 to 0.42 dex at layer 12.** Iteration 95's reconciliation held only at
the level of type means, where two opposite depth trends averaged into a plausible-looking number.
**It does not survive disaggregation, and it is withdrawn.**

**What is and is not established.** The **locus** claim is confirmed twice over: `|a|` is identical at
every layer, so the q,k effect is entirely in the backward pass. The **magnitude** claim — that
`|d|` plus a rank term accounts for the gradient deficit — **fails**. `d_rms` explains 40% of the
deficit on average and its layer-wise pattern is orthogonal to the deficit's.

**Two caveats that could explain part of the mismatch, stated rather than assumed away:**
1. **Different states.** REQ-038 probes fork-1500 directly; Arm A measures at steps 2250–2750 after
   forking. These are not the same point in training, and the campaign has no measurement of how
   `d_rms` evolves over 750–1250 steps.
2. **Different objects.** `d_rms` is the backward tensor's RMS; the gradient is `dᵀa` summed over
   tokens. Their relationship depends on token-wise alignment, which is not measured.

**Neither caveat is testable with committed data, and both are cheap to settle** — which is why
REQ-043 (the probe on seeds 1–3) is now worth extending. **Amended: REQ-043 should also run the probe
at a second training state on at least one seed**, which converts caveat 1 from a speculation into a
measurement. That was already flagged as "worth capturing if cheap"; iteration 96 makes it the
substantive part of the request.

**Net position.** Band 21 is amended to record locus-confirmed / reconstruction-failed. **This is the
campaign's central mechanism and its magnitude is now openly unexplained** — the honest state is that
we know *where* the q,k deficit lives (backward, exactly), and we do not know *what sets its size*.

**=== ITERATION 95: REQ-038 LANDS — the two-sided prediction is confirmed, with a 30% shortfall recorded ===**

**Jerry delivered REQ-038 and REQ-041.** This is the measurement iterations 89–94 were built around,
and **iteration 90's two-sided prediction was registered before the data existed.**

**The prediction, and the result:**

> *"If the attenuation reading is right, `|d|` for q,k should be ~0.37 dex below v and attn.proj
> while `|a|` is equal (all four read the same residual)."* — iteration 90

| quantity | q,k | attn.v | ratio | dex |
|---|---:|---:|---:|---:|
| **`a_rms`** (input activation) | **1.0036** | **1.0036** | **1.0000** | **+0.0000** |
| **`d_rms`** (output gradient) | 0.002623 | 0.003987 | **0.658** | **−0.182** |

**`|a|` is identical to four decimal places; the entire difference is in the backward signal.** The
*locus* prediction is confirmed exactly — this is not a forward-pass effect, and the shared
normed-residual input is measurably shared.

**The magnitude falls short, and that is worth stating plainly.** Predicted −0.37 dex, observed
**−0.18** — a factor of two. The weight gradient is `G = dᵀa` summed over tokens, so its norm depends
on `|d|`, `|a|`, **and the alignment/rank structure of the sum** — which REQ-038 also measured:

| | q,k | attn.v | ratio | contribution |
|---|---:|---:|---:|---:|
| `d_rms` | 0.002623 | 0.003987 | 0.658 | **−0.182 dex** |
| `d_eff_rank` | 74.3 | 106.2 | 0.699 | **−0.078 dex** (as √R) |
| | | | **total** | **−0.260 dex** |

**Against the observed −0.373 dex, that leaves +0.113 dex unaccounted — about 30% of the effect.**
*(An intermediate calculation of mine called this "inside the 0.07 dex noise floor". It is not:
0.113 is 1.6× the floor. Corrected here.)* The √R scaling is a heuristic for how a rank-R sum of
outer products accumulates, not a derivation, so the residual may be that approximation rather than a
missing physical term — **but it is not resolved, and the band records the measured quantities rather
than the reconstruction.**

**Band 20's prediction is also confirmed, and it discriminates cleanly.** Iteration 94 predicted that
if the mlp gap is an activation effect it should appear in `|a|`, since fc reads the block input while
proj reads the ReLU² output:

| | `|a|` gap (fc − proj) | `|d|` gap (fc − proj) |
|---|---:|---:|
| mlp pair | **−0.433 dex** | **+0.050 dex** |

**The mlp gap lives entirely in the forward activation; the backward signals are equal.** So the two
gradient effects in this network are now separated *by measurement*, not just by inference:

> **q,k: purely BACKWARD (`|a|` identical, `|d|` down 34%). MLP: purely FORWARD (`|d|` equal, `|a|`
> down 0.43 dex).** Two different mechanisms, confirmed by the one probe.

**Registered as band 21 — but note it is n=1.** REQ-038 ran a single forward/backward pass on one
seed. Every other confirmed band in this campaign is n=4, and this one is not yet. **The natural
next step is REQ-038's probe on Arm A's other three seeds** — it is one forward+backward pass per
seed, negligible against the Lanczos probe, and it would put the campaign's central mechanism on the
same footing as everything else. **Filed as REQ-043.**

## REQ-043: run the REQ-038 activation/backward probe on Arm A's seeds 1–3

- status: **DONE (all 3 priorities, n=4)** — band 21 + alignment ratio (P3) + second state (P2)
- delivered 2026-09-03 PDT → `logs/kmaxwell/req043_seeds_probe/` (summary.tsv + alignment.tsv + README + raw JSON seeds 0–3 + seed0 fork-2000)

**P3 RESULT — the alignment ratio IS band-25's missing factor (measured, not inferred).** Since `a_rms` is
identical for q/k/v (band 21), `align_deficit = grad_deficit − d_deficit` **exactly** (verified numerically to
<0.0005 dex/seed, `alignment.tsv` `identity_gap`). So the alignment ratio `‖Σₜ dₜaₜᵀ‖_F/(‖d‖_F‖a‖_F)` is
algebraically band-25's shortfall. **align_deficit (q,k)/v = −0.190 ± 0.006 dex (0.646×), n=4** (per-seed
−0.185/−0.200/−0.188/−0.186). Measured at a **single consistent state** → no cross-state artifact, hence
tighter (sd 0.006 vs your reconstruction's 0.041) and smaller than the filed −0.240 (the state mismatch
inflates it). Mechanism: q/k receive both ~0.67× smaller raw `d` (band 21) **and** 0.65× less token-aligned
gradient than v; the two multiply to the full ≈0.43× (−0.37 dex) weight-gradient deficit.

**P2 RESULT — the depth slope is genuinely state-dependent (your iter-103 concern confirmed).** Probed seed 0
at fork-2000: align-deficit depth slope **flattens −0.0091 → −0.0041 dex/layer** from fork-1500→2000 (+0.010
per 1000 steps). So cross-state comparison DOES manufacture part of band-25's depth trend. Artifact-free
single-state slope = **−0.0075 dex/layer (n=4, fork-1500)** — real, monotone toward output, but milder than
the filed −0.0176 and near the low end of your state-corrected CI [−0.0169,−0.0064]. Size is state-stable
(−0.184 vs −0.191); only the slope drifts.

---
- (superseded) status: **BAND-21 DONE (n=4)** — priorities 2 + 3 in progress

**RESULT (band 21, the registered check — n=4):** the q/k output-gradient deficit is seed-reproducible.
`(q,k)/v d_rms = 0.667 ± 0.011` across seeds 0–3 (range [0.655, 0.682]); log₁₀(q/v) = −0.183 dex, t = −41.9;
log₁₀(k/v) = −0.170 dex, t = −43.3. **`a_rms` is bit-identical for q/k/v in all 4 seeds** (band 21 ✓), so the
entire deficit is a **backward-pass** effect — q and k receive ≈0.66× v's output-gradient purely via the softmax
Jacobian, not via less forward signal. Architectural, not seed-dependent (consistent with REQ-035's seed-independent
C). Priorities 2 & 3 (below) require probe surgery + a 2nd fork state → running next on the same 2 nodes.

---
- (original) status: **OPEN**
- requested: 2026-09-03 PDT
- repo: https://github.com/jacknzheng/kmaxwell-sota (branch `jerry-agent`)

**Ask:** run the existing `measure_activation_backward.py` probe (REQ-038's deliverable, already
written and validated) on the **fork-1500 states of seeds 1, 2 and 3** — the same states Arm A
already regenerates — and commit the per-type `a_rms`, `d_rms`, `a_eff_rank`, `d_eff_rank` and
`weight_frob` in the same TSV shape as `req038_activation_backward_probe/summary.tsv`.

**Cost:** one forward+backward pass per seed. **Negligible** — REQ-038's own run took a single pass at
8192 tokens. No new training if Arm A's states can be regenerated or reused; no extra nodes.

**Why:** REQ-038 confirmed at n=1 that the q,k deficit is **purely backward** (`a_rms` identical to 4
decimals, `d_rms` ratio 0.658) and the mlp gap **purely forward** (`|a|` −0.433 dex, `|d|` +0.050).
These are the campaign's two central mechanisms and they are the **only** results still at n=1 —
bands 14, 17, 18, 19 and 20 are all confirmed at n=4. **The registered check is band 21:** `a_rms`
identical for q, k and v; `d_rms` ratio ≤ 0.75; and the mlp gap in `a_rms` not `d_rms`, in ≥3 of 4
seeds.

**Priority 3 — the alignment ratio (added iteration 97).** Record, per matrix, **`‖Σₜ dₜaₜᵀ‖_F / (‖d‖_F · ‖a‖_F)`** — how aligned the backward and forward tensors are across tokens. Iteration 97 showed that `d_rms`, `d_frob` and `d_eff_rank` **cannot** reconstruct the 0.373 dex gradient deficit: the missing factor is 0.48×, grows with depth, and correlates −0.490 with the rank term. Alignment is the one unmeasured quantity with the right shape, and it is a single extra scalar in a probe that already computes both tensors.

**Priority 2 — second training state.** Iteration
96 found that `d_rms`'s q,k deficit shrinks with depth (slope t = +6.65) while the gradient deficit
does not (t = −1.18), and that the two are **uncorrelated layer-by-layer (corr = −0.012)**. One
candidate explanation is that the two are measured at different states — REQ-038 probes fork-1500,
Arm A measures at steps 2250–2750. **Running the probe at a second state (e.g. fork-2000, or at step
2250 on a fork) on at least one seed turns that speculation into a measurement** and is the single
most informative addition available.

**=== ITERATION 94: THE mlp GAP IS AN ACTIVATION EFFECT, NOT UPDATE GEOMETRY ===**

*Iteration 93 surfaced a 0.28 dex gradient gap between `mlp.fc` and `mlp.proj` — 75% the size of the
q,k effect, with no RMS-norm involved — and left two readings the data could not then separate:*
**(A)** *`shape_mult` acting only where the aspect ratio differs from 1 (an update-geometry effect,
and the reading favoured by a striking numerical match: shape_mult 2.0 vs 1.0 = 0.301 dex against an
observed 0.28); or* **(B)** *a separate mlp effect, e.g. the ReLU² nonlinearity between them.*

**They differ in one testable way, and it is decisive.** `shape_mult` is a **constant** fixed by the
architecture, so under (A) the gap must be **depth-flat** — exactly as the q,k deficit is (band 17,
across-layer sd 0.065–0.089, at the noise floor). Under (B) it tracks activation statistics, which
change with depth.

| | mlp gap | q,k deficit (band 17) |
|---|---:|---:|
| across-layer sd | **0.151–0.161 dex** | 0.065–0.089 dex |
| vs ~0.07 dex noise floor | **> 2×** | at the floor |

**The mlp gap varies with depth by more than twice the noise floor. Reading (A) is rejected** — a
constant cannot produce depth structure, and the 0.301-dex numerical match was a coincidence, as
iteration 93 suspected when it declined to promote it.

**The depth structure is real and extremely reproducible:**

| | value |
|---|---:|
| median across-**seed** sd at fixed layer | **0.0182 dex** |
| across-**layer** sd of the seed means | **0.1547 dex** |
| **structure / noise** | **8.5×** |

**And the profile is a boundary pattern, not a trend** — quadratic R² **0.607** against linear 0.205:

```
 layer:    0      1      2      3      4      5      7      8      9     10     11     12
  gap:  -.488  -.284  -.574  -.380  -.249  -.136  -.130  -.130  -.141  -.166  -.245  -.431
        ^^^^^^^^^^^ deep at the entrance ^^^^   flat interior −.13 to −.17   ^^^ deep at exit
```

That is why iteration 93's linear slope was only t ≈ 1.5 despite obvious structure — **the pattern is
U-shaped, and a linear test is the wrong instrument for it.**

> **The mlp.fc / mlp.proj gradient gap is an activation effect that varies with depth, flat through
> the interior and roughly 3× deeper at both network boundaries. It is not update geometry.**

**Registered as band 20.** Note this is a *second* boundary phenomenon in this campaign, and unlike
the withdrawn band 10 it is measured on a **raw gradient ratio** — no fitted intercept, no derived C
— and it reproduces across four seeds at 0.018 dex. **Band 10 failed because it was defined on a
fitted quantity; this one is defined on a direct measurement, which is why it survives where band 10
did not.**

**What it means for the account.** The gradient structure has **two distinct components**: a
**depth-flat architectural attenuation** on the QK-normed matrices (bands 14/17/18/19, ~0.37 dex), and
a **depth-structured activation effect** in the MLP pair (this band, 0.13→0.57 dex). They are
different in size, in depth profile, and now in kind. **REQ-038 discriminates both at once** — the
q,k effect should appear in `|d|` with `|a|` equal, while a ReLU²-driven mlp effect should appear in
`|a|`, since fc and proj read *different* activations (fc reads the block input, proj reads the ReLU²
output).

**=== ITERATION 93: EXCLUDING THE CHUNKING RIVAL — and an unexplained mlp gap found in the process ===**

*The RMS-norm reading has passed three predictions, but **all three are consistency checks**. A rival
that makes the same three predictions is untouched by any of them, and there is one.*

**The rival.** From the recovered architecture, `qk_bank` is **(64, 128, 768)** — Muon orthogonalises
q and k **per head-pair at 128×768** — while `vo_bank` is (24, 768, 768). Muon scales updates by
`shape_mult = max(1, rows/cols)**0.5` (`train_gpt.py:510`), so q,k are updated with a different
chunk geometry than v and attn.proj. **This rival predicts all three passed observations**: it
survives at identical parameter count, it is fixed by the architecture (hence depth-independent), and
q = k (both live in the same bank with the same chunk shape). **The three passes do not discriminate.**

**The horse race settles it. On log g, per seed:**

| model | R² | coefficient (t) |
|---|---:|---|
| **shape_mult only** | **0.005–0.008** | **+0.15 to +0.19 (t = +0.6 to +0.7)** |
| **QK-norm only** | **0.63–0.67** | −0.408 to −0.421 (**t = −10.9 to −11.9**) |
| both | 0.66–0.70 | shape_mult −0.39 to −0.45 (t ≈ −2.7); **QK-norm −0.44 to −0.46 (t = −11.5 to −12.6)** |

**Muon's chunk geometry explains essentially none of the gradient variation on its own (R² < 0.01),
and the QK-norm coefficient does not weaken when it is added — it strengthens.** The chunking rival
is excluded. **Registered as band 19.**

**But the test surfaced something I was not looking for, and it does not fit either account.**
The discriminating comparison used the mlp pair — `mlp.fc` (3072,768) and `mlp.proj` (768,3072),
identical parameter counts, opposite aspect ratios, **neither QK-normed**:

| seed | log g mlp.fc | log g mlp.proj | difference | p |
|---|---:|---:|---:|---:|
| 0 | 3.856 | 4.133 | **−0.277** | **< 10⁻⁴** |
| 1 | 3.866 | 4.142 | **−0.276** | **< 10⁻⁴** |
| 2 | 3.846 | 4.128 | **−0.282** | **< 10⁻⁴** |
| 3 | 3.855 | 4.139 | **−0.283** | **< 10⁻⁴** |

**A 0.28 dex gradient gap — 75% the size of the q,k effect — between two matrices with no RMS-norm
anywhere.** Reproducible to 0.007 dex across four seeds.

**And a numerical coincidence worth flagging rather than believing.** `mlp.fc` has `shape_mult` 2.0
against `mlp.proj`'s 1.0 — a factor of 2, or **0.301 dex** — against an observed gap of **0.28 dex**.
That is a close match. **But the horse race says shape_mult has no explanatory power across the full
set of six types (R² < 0.01, wrong sign),** so this cannot be promoted to a mechanism on the strength
of one pair matching. Two readings are consistent with the data — shape_mult acting *only* where the
aspect ratio actually differs from 1 (true only for mlp.fc), or a genuinely separate mlp effect —
**and this data cannot separate them.**

**Recorded as an open anomaly, not a finding.** It does not touch bands 14/17/18/19, which are
defined on the four same-shape attention matrices where no aspect ratio differs. **It does mean the
account of the gradient structure is incomplete**: there is a second, comparably-sized gradient effect
in the mlp pair with no current explanation, and it was invisible until the chunking rival forced a
comparison I had not previously run.

**Status of the RMS-norm reading: four predictions, four passes, one live rival excluded.** REQ-038's
`|d|`/`|a|` split remains the only outstanding direct test — and it now has a second target: whether
the mlp gap also lives in `|d|` (a backward effect) or in `|a|` (a forward one), which would
discriminate the two readings above at no extra cost.

**=== ITERATION 92: q AND k ARE INTERCHANGEABLE — the RMS-norm reading's third prediction ===**

*The RMS-norm reading has now passed two predictions (depth-independence, and the deficit surviving at
identical shape). It makes a third that is sharper because it is a **within-pair** test, removing
every between-type confound at once.*

**The prediction.** If the attenuation is a property of the **normalisation layer**, q and k receive
the *same operation* — RMS-norm applied to each — so their deficits should be **equal**. If instead
anything **role-specific** is at work, they should differ: under causal masking a key is attended to
by a growing suffix of positions while each query attends once, so queries and keys are not
symmetric in the computation.

| seed | q deficit | k deficit | q − k | within-seed perm p |
|---|---:|---:|---:|---:|
| 0 | −0.382 | −0.374 | −0.008 | 0.821 |
| 1 | −0.384 | −0.371 | −0.013 | 0.558 |
| 2 | −0.364 | −0.348 | −0.017 | 0.670 |
| 3 | −0.390 | −0.373 | −0.017 | 0.570 |

**q and k are interchangeable: −0.380 vs −0.366, a difference of 0.014 dex — 3.8% of the 0.373 dex
deficit they share — and not significant in any individual seed.** The role-specific reading is
excluded; the effect tracks the shared normalisation, not the query/key asymmetry.

**One honest complication, reported rather than smoothed over.** The q−k difference has the **same
sign in all four seeds** (−0.008, −0.013, −0.017, −0.017, sd 0.004), and a one-sample test across
seeds gives **t = −6.44, formally significant**. Both statements are true and not in conflict:
averaging over seeds can resolve an effect below the per-measurement floor. But the magnitude is
**0.014 dex — five times *below* the 0.07 dex noise floor** this campaign uses to decide what counts
as resolved.

**Recorded as a possible sub-floor asymmetry, not a finding.** Band 18's registered check is
deliberately written on the *within-seed* test (|difference| < 0.05 dex, p > 0.05 per seed), which is
what the campaign's own standards support. Treating a t-statistic from n=4 seed means as
establishing a 0.014 dex effect would be exactly the kind of claim this campaign has retracted
repeatedly — a real-looking number below the resolution of the instrument. **If it is real, q sits
marginally below k, which is the direction causal masking would predict; that is a question for a
higher-n design, not for this data.**

**Where the RMS-norm reading now stands — three predictions, three passes:**

| prediction | result |
|---|---|
| survives at identical shape (size artifact excluded) | ✅ −0.37 dex, 48/48 blocks, iter. 90 |
| depth-independent (architectural, not learned) | ✅ slopes \|t\| < 0.5, iter. 91 |
| **q and k equal (norm-driven, not role-driven)** | ✅ **0.014 dex apart, iter. 92** |
| `\|d\|` low with `\|a\|` equal | **awaiting REQ-038** |

**Three independent consequences of one architectural feature, all confirmed at n=4 on committed
data.** This is the strongest position any mechanism has reached in this campaign — and notably, it
was reached *after* iteration 89 demoted the original scale-invariance explanation, by identifying a
different consequence of the same feature. **REQ-038's `|d|`/`|a|` split remains the direct test, and
it is now the only one outstanding.**

**=== ITERATION 91: THE DEFICIT IS A FIXED ARCHITECTURAL CONSTANT — depth-independent, n=4 ===**

*Iteration 90's reading is that RMS-norm rescales the backward signal by `1/RMS(q)`. **That factor is
set by the architecture, not learned**, which makes a prediction testable without REQ-038: the
deficit should be **the same at every depth**. A learned or data-dependent attention effect would
vary with depth, since attention statistics do.*

**Measured within the four same-shape 768×768 attention matrices, so no size confound:**

| | slope of deficit vs layer index | t |
|---|---:|---:|
| seed 0 | −0.00177 | **−0.31** |
| seed 1 | −0.00127 | **−0.25** |
| seed 2 | +0.00355 | **+0.50** |
| seed 3 | −0.00217 | **−0.40** |

**Indistinguishable from zero in every seed.** The deficit sits at **−0.361 ± 0.065 dex across layers
0–11**, and that scatter is *at the 0.07 dex noise floor* measured in iteration 85 — i.e. the
layer-to-layer variation is consistent with pure measurement noise.

> **The q,k gradient deficit is a fixed constant, identical at every depth. It is an architectural
> property, not a learned or data-dependent one.**

**This is what RMS-norm rescaling predicts and what a data-dependent attention effect does not**, and
it is the second prediction from the iteration-90 reading to pass. (The first: `|d|` low with `|a|`
equal — still awaiting REQ-038.)

**A separate, highly reproducible boundary fact.** The final block is an outlier, and an unusually
tight one:

| | interior (layers 0–11) | **layer 12 (final block)** |
|---|---:|---:|
| mean deficit | −0.361 | **−0.508** |
| sd | 0.065 | **0.0099** |
| per-seed | — | **−0.502, −0.512, −0.499, −0.520** |

**Layer 12's deficit reproduces to within 0.01 dex across four independent networks** — six times
tighter than the interior scatter — and sits 2.25 sd below it. Including it was what made the depth
slopes look marginally negative (t ≈ −1.1 to −1.3); excluding it, they flatten to |t| < 0.5.

**A plausible and still-architectural reason:** layer 12 is the final block, so its attention output
feeds the **unembedding directly** rather than another transformer block. Its backward signal has a
different provenance from every other layer's. **This is offered as a reading, not a tested claim** —
what is established is that the final block's deficit is larger, reproducible to 0.01 dex, and
distinct from the flat interior.

**Registered as band 17**, covering both halves: the interior depth-independence (the load-bearing
result) and the final-block excess (the reproducible anomaly). Note this is *not* a revival of the
withdrawn band 10 — that was a **layer-0 lift in C** measured on the fitted intercept and failed at
n=4; this is a **final-block excess in the q,k gradient deficit**, a different quantity at the other
end of the network, and it reproduces where band 10 did not.

**=== ITERATION 90: THE DEFICIT SURVIVES AT IDENTICAL SHAPE — the cleanest form of the result ===**

*Iteration 89 reframed band 14 as a **gradient deficit** (−0.42 dex at identical curvature). Before
hunting a physical mechanism, the mundane explanation had to be excluded: `gradient_block_norm` is a
norm over a block of parameters, so if q,k's block is smaller the norm is mechanically smaller — no
physics required.*

**Partial credit to the mundane reading.** A Frobenius norm over N comparable entries scales as √N,
and the mlp types have 4× the parameters (2,359,296 vs 589,824). Predicted from √N alone: −0.199 dex
against an observed −0.417. Normalising each gradient by √(params) shrinks the deficit consistently:

| seed | Δ log g raw | Δ log g per-parameter |
|---|---:|---:|
| 0 | −0.419 | **−0.268** |
| 1 | −0.421 | −0.271 |
| 2 | −0.408 | −0.258 |
| 3 | −0.420 | −0.270 |

**Parameter count explains about a third of the deficit.** That is a real correction and it is now
recorded — but two-thirds survives it.

**The clean test removes the confound entirely.** Restrict to the **four 768×768 attention
matrices**, where parameter count is identical *by construction*:

> **q, k** (QK-normed) **vs v, attn.proj** (not QK-normed) — same shape, same sub-block, same
> residual-stream input, same parameter count.

| seed | **Δ log g** | perm p | Δ log λ | perm p | Δ log C |
|---|---:|---:|---:|---:|---:|
| 0 | **−0.378** | **< 10⁻⁴** | +0.138 | 0.174 | +0.893 |
| 1 | **−0.378** | **< 10⁻⁴** | +0.018 | 0.854 | +0.774 |
| 2 | **−0.356** | **< 10⁻⁴** | +0.115 | 0.211 | +0.827 |
| 3 | **−0.381** | **< 10⁻⁴** | +0.086 | 0.427 | +0.848 |

**Both q and k sit below both v and attn.proj in 48 of 48 block-seed cells.** The curvature
difference remains non-significant in every seed.

> **At identical shape, identical sub-block and identical input, the two QK-normed matrices carry a
> ~0.37 dex smaller gradient than the two that are not — with their curvature unchanged. Parameter
> count is excluded; the only systematic difference remaining is QK-norm.**

**This is the cleanest statement the campaign has produced.** It needs no fitted exponent (band 13's
`g²` law is not invoked), no derived quantity, no cross-type normalisation, and no assumption about
what C *is* — it compares four matrices that differ in exactly one architectural feature. Band 14 is
updated to be defined on this comparison rather than the six-type one.

**What it does and does not say about mechanism.** It restores QK-norm as the *locus* of the effect —
iteration 89 demoted band 15 because scale invariance could not produce the deficit's **magnitude**
from weight norms, and that failure stands. **QK-norm is where the effect lives; scale invariance is
not how it gets there.** The distinction matters: RMS-norm on q,k does more than make them
scale-invariant — it also **rescales the backward signal by 1/RMS(q)**, which directly attenuates the
gradient without touching the curvature along the same direction. That is a *different* consequence
of the same architectural feature, and it predicts precisely what is observed.

**Registered as the sharpened target for REQ-038.** The `|d|` field measures the backward tensor
magnitude per matrix. **If the attenuation reading is right, `|d|` for q,k should be ~0.37 dex below
v and attn.proj while `|a|` is equal** (all four read the same residual). That is a two-sided
prediction on a measurement already specified — **`|d|` low AND `|a|` equal** — and it is now the
single most informative number in the queue.

**=== ITERATION 89: THE EXCESS IS A GRADIENT DEFICIT — band 14 reframed, band 15's magnitude test FAILS ===**

*QK-norm survived iterations 87–88 by elimination. Elimination is weak evidence — it says the
alternatives are worse, not that this one is right. **A mechanism should predict the magnitude.**
Testing that breaks the account open.*

**The decomposition, which is exact by definition** (`log C = log λ − 2 log g`):

| seed | **Δ log λ** | **Δ log g** | −2·Δ log g | = C excess |
|---|---:|---:|---:|---:|
| 0 | +0.050 | **−0.419** | +0.838 | +0.888 |
| 1 | −0.069 | **−0.421** | +0.843 | +0.774 |
| 2 | +0.016 | **−0.408** | +0.817 | +0.833 |
| 3 | −0.006 | **−0.420** | +0.841 | +0.834 |

**Against permutation nulls, all four seeds:**

| quantity | Δ | p-values across seeds |
|---|---:|---|
| **log λ** (curvature) | +0.05 to −0.07 | **0.62 / 0.50 / 0.87 / 0.95 — never significant** |
| **log g** (gradient) | **≈ −0.42** | **< 10⁻⁴ in every seed** |

> **q,k's curvature is statistically indistinguishable from every other matrix type. Their gradient
> is ~0.42 dex smaller. The entire +0.83 dex "C excess" is that gradient deficit, doubled by the g²
> law.**

**Band 14 is reframed, not withdrawn** — the effect is as real and as reproducible as ever, but its
content is the opposite of how it has been described for twelve iterations. The question **"why do
q,k have high C?"** is really **"why is q,k's gradient 0.42 dex smaller at equal curvature?"**

**And that question kills band 15's magnitude test.** Scale invariance says `g ∝ 1/‖W‖`, so a gradient
deficit requires a **weight-norm excess** of the same size:

| | predicted Δ log g | observed |
|---|---:|---:|
| from `Δ log g = −Δ log‖W‖` | **+0.130** | **−0.417** |

**Wrong sign and 3× the magnitude.** The reason is visible directly — q,k are *not* unusual in weight
norm, sitting mid-pack (q +1.755, k +1.756, against proj +1.778, v +1.809, mlp.proj +1.832, mlp.fc
+2.124). **The scale-invariance mechanism cannot produce a 0.42 dex gradient deficit from a 0.13 dex
weight-norm difference in the wrong direction.**

**What survives of band 15.** The invariance *result* — `d log C/d log‖W‖ ≈ 0` for q,k and clearly
non-zero for the others (+0.049 CI [−0.261, +0.381]; −0.062 CI [−0.329, +0.226]) — is unaffected and
still reproduces. **QK-norm does make q,k's C insensitive to their weight norm.** What fails is the
claim that this *explains the gap*: being insensitive to ‖W‖ says nothing about why the gradient is
small. **Band 15 is demoted from "the mechanism" to "a true property of q,k that does not account for
band 14."**

**Where this leaves the campaign.** All three readings of the q,k gap have now failed a test:
attention-input-projection eliminated (iter. 87), bilinear pairing rejected (iter. 88), and
scale-invariance magnitude failed (here). **The open question is sharper and better posed than the one
this campaign started with:**

> **Why is the gradient on the QK-normed matrices ~0.42 dex smaller than on every other matrix type,
> while their curvature is identical?**

**REQ-038's `|a|`/`|d|` fields address exactly this** — a gradient is `|d|·|a|`-scaled, so the deficit
must live in one of those two factors, and REQ-038 measures both per matrix. **Its target changes from
0.832 dex (the C excess) to 0.42 dex (the gradient deficit)**, which is the quantity that actually
needs explaining. REQ-041 remains worth having for band 15's seed check, but it is no longer the
critical path — **REQ-038 is.**

**=== ITERATION 88: BILINEAR PAIRING REJECTED — QK-norm is the last alternative standing ===**

*Iteration 87 eliminated "attention input projection" using attn.v and left one survivor: q and k
enter a bilinear product with each other, which v does not. I claimed only REQ-041 could separate
that from QK-norm. **That claim was wrong** — the same iteration had just shown I give up on
committed data too early, so I tested it.*

**The bilinear reading makes predictions QK-norm does not.** If the logit scale is what matters, the
controlled quantity is the **product** of q and k norms, so a block's q and k are two halves of one
thing. That predicts (i) their C values **couple within a block** beyond ordinary block-sharing, and
(ii) they **trade off** — the sum of their logs held steadier than independence allows.

**Test 1 — coupling, against all 15 type pairs as the reference set:**

| rank | pair | mean corr across 4 seeds |
|---:|---|---:|
| 1 | mlp.proj–mlp.fc | **+0.858** |
| **2** | **attn.k–attn.q** *(the bilinear pair)* | **+0.741** |
| 3 | attn.k–mlp.fc | +0.643 |
| … | *(other 12 pairs)* | +0.084 to +0.639 |

**q-k ranks 2 of 15 at +1.31 sd above the other pairs — it is not even the highest.** `mlp.proj–mlp.fc`,
a pair with no bilinear relationship at all, correlates more strongly. q-k sits **inside** the
ordinary distribution of within-block correlations, which is what two matrices sharing block-level
conditions produce anyway.

**Test 2 — the trade-off, and this one has the wrong sign:**

| seed | sd(logC_q) | sd(logC_k) | sd(q+k) | independence predicts | ratio |
|---|---:|---:|---:|---:|---:|
| 0 | 0.262 | 0.276 | 0.525 | 0.381 | **1.38** |
| 1 | 0.145 | 0.161 | 0.260 | 0.217 | **1.20** |
| 2 | 0.143 | 0.206 | 0.339 | 0.251 | **1.35** |
| 3 | 0.165 | 0.267 | 0.404 | 0.314 | **1.29** |

**A mechanism controlling the product of q and k predicts ANTI-correlation — ratio below 1.** Every
seed gives **above** 1 (1.20–1.38): q and k are *positively* correlated, moving together rather than
trading off. **The bilinear-pairing reading is rejected, not merely unsupported.**

**Where band 15 now stands.** Three readings of the q,k excess have been tested:

| reading | status |
|---|---|
| attention input projection | **eliminated** (iter. 87 — v sits with the non-qk types, 48/48) |
| bilinear pairing | **rejected** (this iteration — rank 2 of 15, and the trade-off has the wrong sign) |
| **QK-norm scale invariance** | **the only survivor**, and the only one that made a numerical prediction (`d log C/d log‖W‖ = 0`) and passed |

**This is as far as the committed data goes, and now for a stated reason rather than a guess.** The
alternatives are gone by elimination; what remains is *confirming* QK-norm on its own terms at n=4,
which requires the invariance test, which requires weight norms. **REQ-041 is the only remaining
route** — and unlike iteration 86's failed proxy, that is now a conclusion from having exhausted the
alternatives rather than an assumption.

**Registered as check (b) inside band 15**, with the trade-off ratio as its falsifier: if a seed shows
sd(q+k) *below* independence, the bilinear reading revives.

**=== ITERATION 87: THE q,k CONFOUND IS BROKEN — attn.v decides it, n=4 ===**

*Band 14's q,k excess (+0.832 dex) has carried an undecidable confound since it was filed: q and k
share **two** properties — they are QK-normed, and they feed the attention logits. Every mechanism
tested so far failed to separate them, and I recorded the confound as needing REQ-038. **It does
not.** Arm A already contains the discriminator.*

**The discriminator is attn.v.** It reads the **same residual stream** as q and k, sits in the
**same attention sub-block**, and is the **same kind of input projection** — but it is **not
QK-normed** and does **not** enter the logits. So:

- if the excess tracks **QK-norm** → v should sit with the **non-qk** types;
- if it tracks **"being an attention input projection"** → v should sit **with q,k**.

**Result, all four seeds:**

| seed | q,k | **attn.v** | non-attn-input types | v − qk | v − others |
|---|---:|---:|---:|---:|---:|
| 0 | −2.864 | **−3.680** | −3.777 | **−0.816** | +0.097 |
| 1 | −2.973 | **−3.660** | −3.775 | **−0.687** | +0.116 |
| 2 | −2.871 | **−3.643** | −3.725 | **−0.772** | +0.082 |
| 3 | −2.919 | **−3.664** | −3.783 | **−0.745** | +0.119 |

**v sits with the non-qk types: 0.103 dex from them, 0.755 dex below q,k — a 7.3× separation.** And
**v is below both q and k in 48 of 48 block-seed cells.**

> **The "attention input projection" reading is eliminated. Whatever produces the q,k excess is a
> property of q and k that v does not share — and QK-norm is exactly that property.**

**What this does and does not settle.** It **breaks the confound** that has stood since band 14 was
filed: the excess is not about reading the residual, not about being an attention projection, and not
about feeding into the attention sub-block. It is **specific to the two QK-normed matrices**.
Combined with band 15's numerical test — `d log C/d log‖W‖ = 0` for q,k (+0.049 and −0.062, zero
inside both CIs) and clearly non-zero for the others — **two independent lines now point at QK-norm
scale invariance**, one eliminating the alternative and one confirming the prediction.

**It does not close the case.** v differs from q,k in one further respect beyond QK-norm: q and k
enter a **bilinear product** with each other, while v does not. That reading survives this test.
Distinguishing "QK-norm" from "bilinear pairing" needs the invariance test at n=4 — **REQ-041** —
because scale invariance is a consequence of the norm, not of the pairing, and only the norm predicts
`d log C/d log‖W‖ = 0`.

**Band 15 split into two registered checks:** (a) the v-discriminator, **confirmed at n=4 here**, and
(b) the invariance test, **still blocked on REQ-041**. Recording them separately so the confirmed
half is not held hostage to the blocked half.

**Note on the wider queue** (not this campaign): Jerry reports **REQ-034 DONE** — K-Maxwell holds its
gain at 16× where bi-Maxwell decayed to zero — and **REQ-042 BLOCKED**, since 32×/64× over 750 steps
needs 812/781 usable batches against a FineWeb10B maximum of 500/200. That is a corpus-size decision
for the humans: bigger corpus, fewer steps, or an explicit looping policy.

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

## REQ-043: fully paired Muon / bi-Maxwell / K-Maxwell batch ablation

- status: **OPEN**
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

## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```
