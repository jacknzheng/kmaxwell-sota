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

Next request number: **REQ-045**.

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
| **13** | **the PER-MATRIX response ratio is 2, but the exclusion restriction is VIOLATED** (iter. 76, 114) — ⚠️ **band holds, interpretation qualified** | **2.000 inside the 95% CI** by OLS-with-matrix-FE and the IV Wald ratio, every seed. **AND:** two Hessian functionals under the same instrument must give the SAME exponent — they do not | top_eigenvalue **+2.13/+2.10** vs curvature_along_polar **+1.89/+1.83**; difference **+0.241/+0.268**, CIs **[+0.033,+0.443]/[+0.068,+0.465]**, both exclude 0 |
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
| **26** | **C's six-type structure is genuinely THREE-term** (iter. 108) — ✅ **CONFIRMED n=4** | **the identity log g = log‖a‖_F + log‖d‖_F + log(align) holds exactly**; across the six types **no term correlates with C above 0.55**, and **each term's spread exceeds C's own** (offsetting) | identity exact to **1e-6 dex**; corr(C, −2‖a‖) +0.36–0.39, (C, −2‖d‖) +0.44–0.49, (C, −2align) +0.38–0.40; term spreads 1.35 / 1.02 / 0.64 vs C's **1.04** |
| **27** | **the ‖a‖·‖d‖ PRODUCT is the depth-conserved quantity** (iter. 110–111) — ✅ **CONFIRMED n=4** | **corr(log‖a‖, log‖d‖) ≤ −0.80 within every type across depth**, perm p < 0.01; **sd(log‖a‖+log‖d‖) < 0.6 × sd of the smaller factor**, every type. *Compensation is near-exact but NOT universal — TLS slope contains −1 in only 2/6 types* | corr −0.875 to −0.986; sd-ratio **0.262–0.493**; TLS slopes −0.765 to −1.230 |
| **28** | **C's depth structure is carried by λ, not g** (iter. 117) — ✅ **CONFIRMED n=4** | **sd(log λ) / sd(log g) across depth > 1.5 in every matrix type**, every seed — the consistency requirement linking bands 27 and 10/25 | ratios **2.03 / 2.76 / 2.85 / 2.93 / 4.19 / 4.32**, mean **3.2×**; λ variance share 0.51–1.76 |
| **29** | **the λ–g relation WEAKENS as the LR rises** (iter. 120) — ✅ **CONFIRMED n=4** | **cross-sectional slope falls monotonically across s = 0.6 → 1.7**, seed-clustered CI on the spread excluding 0; **sd(log g) flat** (rules out range compression) while **sd(log λ) compresses** | slopes **0.916 / 0.742 / 0.636**, spread CI **[0.208, 0.365]**; sd(log g) **0.246/0.246/0.242**, sd(log λ) **0.429/0.395/0.367**; corr 0.534/0.497/0.453 |
| **30** | **a higher LR DECOUPLES curvature from the gradient** (iter. 121, 123) — ✅ **CONFIRMED on TWO INDEPENDENT DESIGNS** | **cov(log λ, log g) falls as the LR rises**, seed/matrix-clustered CI excluding 0, on **both** the global ladder and REQ-023's per-matrix randomisation. *Shape not resolved — the two designs differ in where the drop occurs* | Arm A **0.0552/0.0448/0.0371**; REQ-023 **0.0766/0.0425/0.0418** (f1500) and **0.0784/0.0421/0.0402** (f2000), endpoint CIs **[−0.071,−0.002]** and **[−0.077,−0.005]** |
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

**=== ITERATION 124 (2026-09-03): THE TWO DESIGNS DISAGREE ON SHAPE — and the disagreement is structured ===**

*Iteration 123 confirmed band 30 on two designs but noted they differ in **where** the decoupling
happens, and declined to resolve it. That is now testable directly: is the difference real, or were
the CIs simply too wide?*

**REQ-023 (per-matrix perturbation) shows a clear threshold:**

| fork | step 0.6→1.0 | step 1.0→1.7 | **steps different?** |
|---|---|---|---|
| 1500 | **−0.0341** [−0.060, −0.010] ✅ | −0.0007 [−0.020, +0.018] ❌ | **−0.0334 [−0.064, −0.006]** ✅ |
| 2000 | **−0.0363** [−0.064, −0.012] ✅ | −0.0018 [−0.022, +0.016] ❌ | **−0.0344 [−0.064, −0.007]** ✅ |

**The first step is significant, the second is not, and the two differ significantly — in both forks.
Decoupling saturates by s = 1.0.**

**Arm A (global ladder) shows no threshold:**

| step | value | CI | |
|---|---:|---|---|
| 0.6→1.0 | −0.0104 | [−0.0155, −0.0068] | **significant** |
| 1.0→1.7 | −0.0074 | [−0.0133, −0.0017] | **significant** |
| **difference** | −0.0030 | **[−0.0129, +0.0059]** | **not distinguishable** |

**Both steps significant, indistinguishable from each other — an even decline across the range.**

> **The two designs genuinely disagree on shape, and this is not a width problem: each is
> individually decisive and they point different ways.** Band 30's *sign and magnitude* hold on both
> (iteration 123); its *shape* is design-dependent.

**The disagreement is structured, not random, and the designs differ in exactly one way.** In Arm A
**every matrix moves together** — the whole network shifts to a new operating point, and the
surrounding matrices' curvature changes too. In REQ-023 **one matrix is perturbed while the rest stay
at baseline**, so the perturbed matrix responds against a fixed background. **A threshold in the
per-matrix design and a smooth decline in the global one is what you would expect if the saturation is
a property of a matrix in isolation, and the global ladder's extra decline comes from the network-wide
state change** — the same distinction band 13's re-scoping identified between partial and total
derivatives (iteration 79).

**Recorded as an observation with a mechanism-shaped reading, not a band.** Registering it would
require distinguishing the above from simpler explanations — three LR levels cannot resolve a
saturation curve, and both designs have only one interior point. **A 5-level ladder on either design
would settle it**, and that is a cheap addition to any future curvature run, but it is not worth a
request on its own.

**What this changes about band 30's use.** The band supports "raising the LR decouples λ from g" and
**does not** support "the effect is proportional to the LR change." **Anyone using band 30 to predict
the effect of a specific LR change should note the size is design-dependent and, on the per-matrix
evidence, may already be saturated at the baseline LR.** Band 30's wording already claims sign and
magnitude only; this iteration is the evidence behind that restriction rather than a change to it.

**=== ITERATION 123 (2026-09-03): BAND 30 CONFIRMED ON AN INDEPENDENT DESIGN ===**

*Iteration 122's test of band 30 was invalid because REQ-036's multiplier is a pure function of type.
That iteration named the valid alternative: **REQ-023 varies the LR *within* each type**, which breaks
the collinearity. Running it.*

**The collinearity is verifiably broken.** Each matrix receives each of {0.6, 1.0, 1.7} exactly once,
so **every type contains all three multipliers** (verified: 12 matrices per type, all three levels
present in each). The regressor is no longer type in disguise.

**Band 30's prediction — cov(log λ, log g) falls as the multiplier rises — reproduces:**

| fork | mult 0.60 | mult 1.00 | mult 1.70 | **cov(1.7) − cov(0.6)** | 95% CI |
|---|---:|---:|---:|---:|---|
| 1500 | 0.0766 | 0.0425 | 0.0418 | **−0.0347** | **[−0.0712, −0.0022]** |
| 2000 | 0.0784 | 0.0421 | 0.0402 | **−0.0381** | **[−0.0765, −0.0051]** |

**Both CIs exclude zero, matrix-clustered.** Band 30 now holds on **two independent designs** — Arm A's
global ladder (every matrix moved together, across seeds) and REQ-023's per-matrix randomisation (one
matrix perturbed at a time, within a single run). **Different perturbation geometry, different fork,
different resampling unit, same conclusion.**

**A shape caveat, recorded rather than smoothed:**

| design | 0.6 → 1.0 | 1.0 → 1.7 |
|---|---:|---:|
| Arm A (global) | **−19%** | **−17%** |
| REQ-023 (per-matrix) | **−45%** | **−2% / −5%** |

**Arm A's decline is roughly even; REQ-023's is almost entirely at the low end.** Both are consistent
with band 30 as registered — *cov falls as the LR rises* — but they disagree on **where**. The
REQ-023 endpoint CIs are wide enough ([−0.071, −0.002]) that the shape difference is **not
resolvable**, so **band 30 is confirmed for its sign and magnitude and makes no claim of
monotonicity.** The band's wording is amended to say so.

**Why the confirmation matters beyond band 30.** It is the campaign's first result established on two
designs whose *confounds do not overlap*: the global ladder confounds LR with whole-network state,
while the per-matrix design confounds it with nothing (each matrix's perturbation is independent of
every other's, which is what made REQ-023 the campaign's cleanest instrument in the first place).
**A finding that survives both is not an artifact of either.**

**And it closes iteration 122's loose end properly.** That iteration recorded a negative and named the
test that would settle it. **Naming a test and then running it is the difference between a registered
negative and an abandoned line** — the negative stands as filed, and the line it pointed to has now
produced a positive.

**=== ITERATION 122 (2026-09-03): AN INVALID TEST OF BAND 30 — the design, not the band, was wrong ===**

*Band 30 came from Arm A's global LR ladder. REQ-036 looked like independent confirmation: five arms,
per-type multipliers, a different fork and a different perturbation geometry. **The test I ran on it
was invalid, and the reason is worth recording because it nearly produced a false refutation.***

**What I tested.** Band 30 predicts a higher LR weakens the λ–g coupling, so within each REQ-036 arm,
matrices given a larger multiplier should contribute *less* covariance. Regressing each matrix's
covariance contribution on its log multiplier:

| arm | slope | band 30 predicts |
|---|---:|---|
| a2_pertype | **+0.0722** | negative |
| a3_endcap | **+0.0686** | negative |
| a4_antirule | **+0.0836** | negative |
| a5_polar | **+0.1503** | negative |

**All four positive — an apparent clean refutation on independent data.**

**It is not, because the regressor is type in disguise.** REQ-036 assigns multipliers **per type**, so
within any arm the multiplier is **constant within a type** — verified: *one distinct multiplier per
type in every arm.* Regressing on `log(mult)` across 72 matrices is therefore regressing on **type
with six distinct values**, and the resulting slope is the type structure — the q,k excess, the mlp
gap, the offsets — that bands 14, 20 and 26 spent the campaign measuring. **It contains no LR
information at all.**

> **The +0.07 to +0.15 slopes measure C's type structure, not an LR effect. Band 30 was never testable
> this way. My test design was wrong; the band is untouched.**

**The valid comparison REQ-036 supports is between arms**, each being a whole run with its own LR
profile — the same shape as Arm A's ladder:

| arm | corr(log λ, log g) | mean multiplier |
|---|---:|---:|
| a5_polar | **0.206** | 1.131 |
| a3_endcap | 0.320 | 1.403 |
| a2_pertype | 0.380 | 1.030 |
| a1_control | 0.413 | 1.000 |
| **a4_antirule** | **0.680** | 1.171 |

`corr(arm correlation, mean multiplier) = −0.093` — **the right sign but nowhere near significant at
n=5 arms.** Reported as uninformative, not as support.

**One observation worth keeping.** **a4_antirule has the strongest λ–g coupling (0.680) and is the arm
that deliberately ANTI-equalises curvature** (spread 0.444 vs control's 0.246, per REQ-036's own
mechanism check). That is consistent with band 30 read in reverse — spreading curvature strengthens
the coupling — but with n=1 arm it is an observation, not evidence.

**Why this iteration is a negative worth recording.** A confounded test that *appeared* to refute a
band on independent data would have been a serious error to publish, and the confound is subtle: the
multiplier is a legitimate experimental variable, it just happens to be perfectly collinear with type
**by design**. **This is standing rule 2 in a new guise** — the predictor takes six distinct values
and each identifies a type, so it is a label, not a dose. *Extending rule 2: check not only how many
distinct values a predictor takes, but whether those values are in one-to-one correspondence with a
known grouping.*

**Band 30 stands as filed, on Arm A's ladder alone.** Independent confirmation would need an
experiment where the LR varies **within** a type — which REQ-023's per-matrix randomisation does
provide, and which is the natural next test if this line is pursued.

**=== ITERATION 121 (2026-09-03): C's SPREAD EXPANDS WITH THE LR — my prediction was backwards ===**

*Band 29 found λ's spread compresses under a rising LR while g's stays flat. Since `C = λ/g²`, I
predicted **C's spread must compress too** — which would have meant the campaign's type structure is
itself LR-dependent and every effect size is quoted at one learning rate.*

**The prediction was wrong, and in the informative direction:**

| s | sd(log λ) | sd(log g) | **sd(log C)** | **type-spread of C** |
|---:|---:|---:|---:|---:|
| 0.60 | 0.427 | 0.246 | **0.450** | **0.997** |
| 1.00 | 0.401 | 0.246 | 0.471 | 1.056 |
| 1.70 | 0.379 | 0.242 | **0.478** | **1.079** |

**C's spread EXPANDS: +6.2%, CI [+0.007, +0.061]; type-spread +8.2%, CI [+0.048, +0.110].** Both
exclude zero. λ compresses, g is flat, and C widens anyway.

**The variance identity explains it exactly, and closes to 1×10⁻⁶:**

```
   var(log C) = var(log λ) + 4·var(log g) − 4·cov(log λ, log g)
```

| s | var(log λ) | 4·var(log g) | **−4·cov** | = var(log C) |
|---:|---:|---:|---:|---:|
| 0.60 | 0.1825 | 0.2413 | **−0.2209** | 0.2029 |
| 1.00 | 0.1607 | 0.2414 | **−0.1791** | 0.2230 |
| 1.70 | 0.1440 | 0.2338 | **−0.1483** | 0.2295 |

**The covariance term shrinks faster than λ's variance falls**, so C widens. Reading the mechanism
directly:

| s | **cov(log λ, log g)** | corr |
|---:|---:|---:|
| 0.60 | **0.0552** | 0.526 |
| 1.00 | **0.0448** | 0.455 |
| 1.70 | **0.0371** | 0.406 |

> **Registered as band 30. A higher learning rate DECOUPLES curvature from the gradient. Band 29's
> flattening slope and band 30's widening C-spread are the same fact seen twice** — the slope is
> `cov/var(g)` and C's variance contains `−4·cov`, so a falling covariance flattens one and widens the
> other simultaneously.

**A caveat this establishes for every band, stated plainly.** C's type structure is **~8% narrower at
s = 1.0 than at s = 1.7**. Every effect size the campaign quotes — the q,k excess, the mlp gap, the
type offsets — is measured at s = 1.0 and carries that sensitivity. **They are not
architecture-only constants; they are architecture-at-a-given-learning-rate.** The dependence is
modest (~8% per 2.8× LR) and does not threaten any band's sign or significance, but it belongs in the
record.

**Consistency with band 16, checked and holding.** Band 16 says C's *pattern* is restored under the
ladder (corr +0.87–0.97) while its level shifts. Band 30 adds that its *spread* also widens. All three
are compatible: the ordering of matrices survives while their level and dispersion both move — which
is precisely why band 16's registered check is a **correlation**, not a variance.

**Method note.** I predicted compression from a two-term intuition (λ down, g flat ⇒ C down) and the
third term — the covariance — reversed it. **`C = λ/g²` is not a two-variable relation when both are
random;** the covariance is a first-class term and I treated it as absent. *This is the same class of
error as iteration 118's constraint-B framing: reasoning about a derived quantity without writing out
its full decomposition.*

**=== ITERATION 120 (2026-09-03): MUON'S UPDATE IS GRADIENT-SCALE-INVARIANT — what λ ∝ C·g² can mean ===**

*Iteration 119's defect 1 was found while checking a spec, but it has a consequence for the campaign's
central law that is larger than the spec it came from.*

**The structural fact.** `polar_express` normalises to unit spectral norm *before* Newton-Schulz, and
everything after operates on that unit-norm matrix. The full path is
**momentum → normalise → Newton-Schulz → scale by LR.** So **Muon's update magnitude is set by the
learning rate alone and carries no gradient information** — only the update's *direction* depends on
the gradient.

**That removes the obvious reading of λ ∝ C·g².** Under plain SGD the law has a natural dynamical
story: larger gradients take larger steps into higher-curvature regions. **Under Muon that story is
unavailable by construction.** Two readings remain:

- **(A) equilibrium selection** — the matrix settles where curvature balances a *fixed* step size, so
  the relation is a property of the landscape *as sampled by this optimiser*, and should **weaken when
  the step size changes**;
- **(B) a Gauss-Newton identity** — `H ≈ JᵀJ`, `g = Jᵀr` make it a statement about the loss surface
  alone, **invariant to the optimiser**.

**Both predict the same exponent; only (A) predicts LR-dependence. The ladder discriminates:**

| s (LR multiplier) | cross-sectional slope | corr(λ, g) |
|---:|---:|---:|
| 0.60 | **0.916** | 0.534 |
| 1.00 | **0.742** | 0.497 |
| 1.70 | **0.636** | 0.453 |

**Monotone decline, spread 0.281, seed-clustered 95% CI [0.208, 0.365] — excludes zero.** Reading (A).

**Range compression ruled out — the obvious artifact, checked.** A flatter fit can arise mechanically
if the predictor's spread shrinks. It does not:

| s | **sd(log g)** | sd(log λ) |
|---:|---:|---:|
| 0.60 | **0.246** | 0.429 |
| 1.00 | **0.246** | 0.395 |
| 1.70 | **0.242** | 0.367 |

**The gradient's spread is flat to 0.004 dex across a 2.8× LR change**, while **λ's spread compresses
by 14%.** The slope falls because the *response* compresses, not because the predictor narrows. **This
is a real effect on λ, not a fitting artifact.**

> **Registered as band 29. The λ–g relation is not an optimiser-independent identity: raising the
> learning rate compresses curvature's spread while leaving the gradient's untouched, weakening the
> relation. Reading (A) — equilibrium selection — is supported; reading (B) is not.**

**What this does and does not touch.** Band 13's *derivation* of exponent 2 from Gauss-Newton is
unaffected as **theory**; what iteration 120 shows is that the *measured* relation carries an
optimiser-dependent component the derivation does not predict. **This is a second, independent line
pointing the same way as iteration 114's exclusion violation** — both say the measured exponent is not
a clean structural constant. They arrive from different directions: 114 from two functionals
disagreeing, 120 from the relation weakening under the LR.

**Consistency with band 16, checked.** Band 16 says C's *pattern* is restored under the ladder
(corr +0.87–0.97) while its *level* shifts. Band 29 adds that the λ–g *relation* also weakens. These
are compatible: a compressing λ spread with a fixed g spread moves the level and the slope while
leaving the ordering intact — which is exactly what bands 16 and 29 measure separately.

**=== ITERATION 119 (2026-09-03): ARM 4's SPEC IS BROKEN AS WRITTEN — corrected before filing ===**

*I was about to file arm 4 as a formal request. **Reading the actual code first found two defects that
would have made the arm a silent no-op.** Both are recorded, because a spec that returns the control
result while appearing to run is worse than no spec.*

**DEFECT 1 — a clip inside `polar_express` is cancelled exactly.** `train_gpt.py:177` reads:

```python
    momentum_buffer.lerp_(grad_chunk, 1 - momentum)   # momentum update
    g = grad_chunk.lerp_(momentum_buffer, momentum)   # Nesterov
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2,-1), keepdim=True) * (1 + 2e-2) + 1e-6)   # spectral normalisation
```

**That last line divides X by its own norm.** Scaling the gradient by *c* gives
`(c·g)/‖c·g‖ = (c·g)/(c·‖g‖) = g/‖g‖` — **c cancels exactly.** Verified numerically: clips of 0.5,
1.0 and 2.0 produce **identical** normalised values (0.794719 in all three).

**Iteration 113 specified "clip before `shape_mult` and Newton-Schulz."** That is *inside* the
normalised region, so **the arm would have had no effect on the update and would have silently
returned the control result three times.**

**The correct insertion point is BEFORE the momentum buffer update** — the buffer *accumulates* the
clipped gradient, and its altered trajectory is not normalised away. Concretely: clip `grad_chunk` on
entry to `polar_express`, ahead of `momentum_buffer.lerp_`, or at the reduce-scatter site
(`train_gpt.py:604–619`) where `grad_chunk` is produced.

**DEFECT 2 — the instrument has no measurable first stage, and this is the serious one.** The probe
records `gradient_block_norm = ‖param.grad‖` — the **raw gradient, before the optimiser touches it**.
A clip applied inside the optimiser **does not change `param.grad`**, so:

> **`d log g / d log clip` ≈ 0 — the first stage is empty and the Wald ratio is undefined.**

This is not a coding detail; it is a **flaw in arm 4 as conceived**, mine as much as REQ-037's.
The clip changes the *update*, not the *measured gradient*, so it cannot instrument
`d log λ / d log g` in the form the campaign has been estimating.

**What would actually work — two options, both stated so the humans can choose:**

1. **Clip and measure the same object.** Have the probe record the **post-clip** gradient norm
   alongside `param.grad`. Then the first stage is mechanical (`d log g_clipped/d log clip = 1` by
   construction) and the Wald ratio becomes `d log λ / d log clip` directly — a clean reduced form
   needing no ratio at all. **This is the cheaper fix: one extra field in the existing probe.**
2. **Scale the loss per matrix instead of clipping.** A per-matrix loss weight changes `param.grad`
   itself, so the existing probe measures the first stage with no new field. But it also changes what
   is optimised, which reintroduces an exclusion problem of its own.

**Recommendation: option 1.** It keeps the intervention where REQ-037 wanted it (in the update, not
the objective) and moves the measurement to match, rather than the reverse.

**Status: arm 4 is NOT filed as a request.** Its premise needs the above decision from the humans
first — filing a spec that cannot produce a first stage would waste a run and, worse, produce a
confident-looking null. **REQ-037 remains DEFERRED with its reason now precisely stated** rather than
"needs a new hook."

**Method note.** Iteration 113 specified this arm from the *description* of Muon's update path
without reading `polar_express`. **Two iterations of analysis rested on a spec that would not have
run**, and the error surfaced only when I went to file it. *Standing rule 7: before specifying an
intervention, read the code path it modifies — a spec derived from a description of the algorithm is
not a spec.*

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

## REQ-044: fully paired Muon / bi-Maxwell / K-Maxwell batch ablation

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
