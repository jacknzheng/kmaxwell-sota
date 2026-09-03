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

Next request number: **REQ-041**.

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

- status: **OPEN — unblocked, ready to launch.** Machinery is ready (reuses the REQ-026/029
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

- status: **OPEN — Arm A is priority 1, unblocked and ready.** Both nodes are free (REQ-032
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
| **6** | **two-valued gradient slope** (iter. 63–66) | **Q(5) > 10** across the six raw slopes; **F(6 slopes vs 2 slopes, both with 6 free intercepts) < 2.5**; **residual-writer slope ≥ 2.5**, **internal slope ≤ 1.8** | Q = 66.7 / 68.3; F = 0.18 / 0.10 |
| **7** | **the split is residual-stream position, not shape** (iteration 65) | **slope(attn.proj + mlp.proj) − slope(other four) ≥ +1.5** in ≥3 of 4 seeds, each proj type individually ≥ +2.0 vs internal | +2.173 / +2.183, p < 0.0001 |
| **8** | **the cross-sectional split is bias, not physics** (iter. 67) | **Wald-ratio gap (residual − internal) ≤ +1.0** and **< half the cross-sectional slope gap** in ≥3 of 4 seeds; both first-stage F > 100 | gap +0.374 / +0.688 vs cross-sectional +2.17 / +2.18 |
| **9** | **only attn.proj's offset is resolved** (iter. 68–69, corrected) | **attn.proj lowest** of the six offsets in ≥3 of 4 seeds and **≥0.25 dex below the other five**; **|residual-writer − internal| offset gap < 0.20 dex**; the other four adjacent gaps need NOT resolve | gap −0.454 / −0.430, p ≤ 0.0003 |
| **10** | **C is lifted at layer 0** (iter. 69–72, corrected twice) | **layer-0 indicator coefficient positive, t > 4** over type+gradient; **layer 0 lift ≥ +0.20 dex** above the interior; **layer 12 NOT lifted** (< +0.15 dex) — in ≥3 of 4 seeds | t = 7.3–8.8; layer 0 +0.26/+0.44, layer 11 +0.24/+0.22, layer 12 +0.12/+0.12 dex |
| **11** | **the assembled model generalises** (iter. 72) | **leave-one-layer-out rmse ≤ 0.20 dex** and **cross-fork rmse ≤ 0.20 dex** for `type + two-slope gradient + layer-0 term`, versus ~0.38 dex for C's own spread | LOLO 0.138 / 0.164; cross-fork 0.165 / 0.140 dex |
| **12** | **three physical quantities replace the six type offsets** (iter. 73) | **LOLO rmse of `log‖W‖ + log fan-in + residual-binary` within 0.02 dex of the 6-free-offset model** (2 fewer parameters); **all coefficients same sign in every seed**: log‖W‖ < 0, log fan-in < 0, residual-binary < 0 at centred gradient | 0.168 vs 0.162 / 0.209 vs 0.217 |

**Band 6 is the newest and it sharpens the campaign's central claim.** The cross-sectional gradient
exponent differs systematically by type — **~3.8 for the two projection matrices, ~0.9–1.4 for the
other four — against a within-matrix causal exponent of 2.07. No type sits at the causal value.**
This is not attenuation: measured error in log g is sd 0.0131 dex, reliability 0.962–0.993, and
correcting for it moves each slope by 1–4% and leaves the spread intact (1.35 → 1.24 / 1.54 → 1.42).
**Falsifier:** if attenuation-corrected slopes converge to ~2 across seeds, iteration 63 is wrong
and the law is universal after all.

**=== ITERATION 73: THE SIX TYPE OFFSETS REDUCE TO THREE MEASURED QUANTITIES ===**

*Band 11's model works but its type term is six free constants — curve-fitting with no mechanism.
The campaign's goal is an account of **why** C differs, so this iteration tries to replace them.*

**Constraint:** only quantities measured independently of the Lanczos tridiagonal are admissible.
That allows `log₁₀‖W‖_F` (from `weight_norms.tsv`) and matrix shape (fan-in, fan-out, parameter
count), and excludes everything derived from the probe. Scored by leave-one-layer-out, the same
honest test as band 11.

| offset replacement | params | LOLO f1500 | LOLO f2000 |
|---|---:|---:|---:|
| 6 free type offsets *(band 11)* | 9 | **0.162** | 0.217 |
| **log‖W‖ + log fan-in + residual-binary** | **7** | **0.168** | **0.209** |
| log‖W‖ + residual-binary | 6 | 0.205 | 0.241 |
| residual-binary only | 5 | 0.219 | 0.256 |
| log‖W‖ only | 5 | 0.298 | 0.326 |
| log fan-in / fan-out / param count | 5–6 | 0.301–0.318 | 0.331–0.348 |

**Three measured quantities match six fitted constants with two fewer parameters** — within 0.006 dex
on fork-1500 and *better* on fork-2000. Cross-fork transfer is comparable (0.170/0.158 vs
0.167/0.146). **The type offsets are not irreducible.**

**A coefficient that looked alarming and is not.** The residual-writer binary fits at **−12.95
(t = −12.3)**, which is not a physical dex offset. It is an extrapolation artefact: the binary is an
intercept shift for a group that also carries its own gradient slope, and with `log g ≈ 3.8` a slope
difference of 3.28 forces an intercept of about −3.28 × 3.8 ≈ −12.5. **Centring log g collapses it
to −0.47 dex in both forks**, which is the interpretable quantity — the level difference at the mean
gradient. The −12.9 was an extrapolation to a gradient of 1.0, far outside the observed 3.5–4.1
range, and means nothing. *(Reporting an uncentred interaction intercept as a physical effect is the
same class of error as citing a correlation without its null — recorded so it is not repeated.)*

**The reduced account, with every coefficient stable across both forks:**

| term | f1500 | f2000 | t (f1500 / f2000) |
|---|---:|---:|---:|
| log₁₀‖W‖_F | **−0.60** | **−0.67** | −3.9 / −3.9 |
| log₁₀ fan-in | **−1.08** | **−1.07** | −6.2 / −5.7 |
| residual-writer (at mean g) | **−0.47** | **−0.47** | −9.1 / −8.1 |
| gradient, residual writers | +3.87 | +3.71 | +15.3 / +13.8 |
| gradient, internal | +0.58 | +0.50 | +5.0 / +3.9 |
| layer-0 lift | +0.20 | +0.40 | +3.1 / +5.7 |

**Both structural coefficients are negative and remarkably stable** — fan-in at −1.08/−1.07 is nearly
identical across forks, and the residual-writer level offset is −0.47 in both. **Larger matrices and
wider fan-in carry lower C**, which is a physically meaningful statement rather than a fitted label.

> **log C ≈ −0.6·log‖W‖_F − 1.1·log(fan-in) − 0.47·(residual writer) + (gradient term: ≈3.8 for
> residual writers, ≈0.5 for internal) + 0.2–0.4·(layer 0) + noise**

**Caveat, stated plainly:** fan-in takes only two values here (768 and 3072), so its −1.08 coefficient
is identified by a single contrast — mlp.fc against everything else — and should be read as "the
3072-fan-in matrix sits lower," not as a fitted power law. A third fan-in value would be needed to
call it an exponent. This is the same trap as the withdrawn Muon rank slope, avoided this time by
declaring it.

**Registered as band 12.** The seed check is deliberately about *signs and parity*, not magnitudes:
the reduction must stay within 0.02 dex of the six-offset model, and all three structural
coefficients must keep their sign in every seed.

**=== ITERATION 72: THE ASSEMBLED MODEL, VALIDATED OUT-OF-SAMPLE — and the edge definition corrected ===**

*The campaign's stated goal is a mathematically rigorous account of C. The components existed but had
never been assembled or tested against held-out data. This does both.*

**Why in-sample R² would have been worthless here.** Every component was found *on* this data, so
fitting them together and reporting R² measures nothing. Two honest tests instead:
**leave-one-layer-out** (fit 12 layers, predict a depth the model has never seen) and **cross-fork**
(fit fork-1500, predict fork-2000, and vice versa).

**Every component earns its place, and none is overfitting:**

| model | LOLO rmse (f1500) | LOLO (f2000) | cross-fork 1500→2000 | 2000→1500 |
|---|---:|---:|---:|---:|
| intercept only | 0.381 | 0.393 | 0.387 | 0.377 |
| gradient only | 0.346 | 0.371 | 0.354 | 0.334 |
| type only | 0.346 | 0.365 | 0.336 | 0.319 |
| type + gradient | 0.212 | 0.275 | 0.247 | 0.193 |
| type + 2-slope gradient | 0.164 | 0.221 | 0.197 | 0.149 |
| **type + 2-slope gradient + edge** | **0.147** | **0.182** | **0.165** | **0.140** |

Against **C's own spread of 0.384 dex** and a **~0.10 dex noise floor**. In-sample rmse is 0.123 vs
LOLO 0.147 — a small gap, so the model is not overfitting. **Band 6's two-slope structure earns its
place a third time here** (0.212 → 0.164 in LOLO), having already survived iterations 66 and 68.

**A defect found by breaking LOLO down by layer.** The edge term helps exactly where it should —
layer 0 by +0.056/+0.106 and layer 11 by +0.073/+0.116 — and barely touches the interior
(+0.004/+0.019). **But layer 12 got actively worse: −0.068 and −0.149.** My indicator marked
`{0, 1, 11, 12}` as boundary, while the residual profile shows layer 12 lifted only +0.12 dex against
layer 11's +0.24 — so the model over-corrected it.

**Testing six edge definitions by LOLO:**

| definition | LOLO (f1500) | LOLO (f2000) | layer-12 rmse (f1500 / f2000) |
|---|---:|---:|---:|
| **{0} only** | **0.138** | **0.164** | **0.064 / 0.055** |
| {0, 11} | 0.142 | 0.216 | 0.071 / 0.052 |
| {0, 1, 11} | 0.145 | 0.175 | 0.094 / 0.111 |
| {0, 1, 11, 12} *(what I used)* | 0.147 | 0.182 | 0.143 / 0.242 |

**`{0}` alone wins both forks** — the simplest definition, one free parameter, and it cuts the
layer-12 error by more than half. **Band 10 is corrected a second time: the robust claim is a lift at
layer 0, not a symmetric boundary shell.** Layer 11's lift is real in the profile but does not earn a
parameter; layer 12's is half the size and must not be corrected as if it were an edge.

**The account of C, as it now stands and as it survives held-out data:**

> **log C = (matrix-type offset) + (gradient term, slope ≈2.5 for residual writers and ≈0.3 for
> internal matrices) + (a lift of ~0.2–0.3 dex at layer 0) + noise ~0.10 dex.**
>
> This predicts a never-seen depth to **0.138–0.164 dex** and transfers between training states at
> **0.140–0.165 dex**, against C's own spread of 0.384 dex — roughly **60% of C's variation
> explained out-of-sample**, with the residual within ~1.5× the measurement noise floor.

**Registered as band 11**, the campaign's first out-of-sample band: LOLO and cross-fork rmse must
both stay ≤ 0.20 dex at n=4. **This is the falsifiable form of the whole account** — if Arm A's seeds
push either above 0.20, the model is not capturing C's structure and the component bands need
re-examination regardless of their individual significance.

**=== ITERATION 71: WALKING BACK ITERATION 70'S WARNING TO REQ-036 ===**

*Iteration 70 closed by warning that REQ-036's per-type LR design "leaves the biggest effect on the
table." That warning was overstated. Quantifying it properly changes the recommendation.*

**What I claimed, and what is actually true.** I inferred from band 10's F = 28.98/41.51 that the
boundary term must dominate the per-type term. Measuring both on the same footing:

| | f1500 | f2000 |
|---|---:|---:|
| per-type offset spread | **0.614 dex** | **0.584 dex** |
| boundary coefficient | **0.183 dex** | **0.290 dex** |
| **ratio boundary / type** | **0.30×** | **0.50×** |

**The boundary term is one-third to one-half the size of the per-type spread it would supplement —
smaller, not larger.** And the practical gain from adding it to a per-type rule is modest: R²
0.287 → 0.337 and 0.248 → 0.367, with rms prescription error improving only **0.318 → 0.306 dex**.
**"Leaving the biggest effect on the table" was wrong** and is withdrawn.

**Why band 10's F was so large anyway — both numbers are correct.** F was computed *on top of the
gradient*, which already absorbs most of C's variance:

| model | R² (f1500) | boundary adds |
|---|---:|---:|
| type | 0.287 | — |
| type + boundary | 0.337 | **+5.0 pts** |
| type + gradient | 0.751 | — |
| type + gradient + boundary | 0.864 | **+11.3 pts** |

A large F against a small residual variance is still a large F. It measures *reliability*, not
*magnitude*, and I conflated the two. Band 10 stands exactly as amended in iteration 70 — the
correction is to my reading of its practical weight, not to the finding.

**The number REQ-036 actually needs.** Under the EoS relation `s ∝ λ^(−1/2)`, a boundary coefficient
of +0.183/+0.290 dex in C implies a learning-rate multiplier of:

> **0.81× (f1500) / 0.72× (f2000) on the boundary layers**, against per-type multipliers that span a
> factor of ~3.

**Revised recommendation for REQ-036: ship the per-type design as filed.** The boundary correction is
real, consistent in sign across both forks, and worth roughly a 20–28% LR reduction on the first two
and last two layers — a worthwhile refinement, but clearly second-order next to the per-type
multipliers, and not a reason to hold or redesign the arm. **If Arm A confirms band 10 at n=4, the
natural follow-up is a per-type × boundary arm, filed separately after REQ-036 reports.**

**One genuinely useful diagnostic from this.** The matrices a per-type rule serves worst are
overwhelmingly at the boundary:

| matrix | error of per-type rule (f1500) | |
|---|---:|---|
| mlp.proj layer 11 | **+1.296 dex** | boundary |
| attn.proj layer 12 | +0.997 dex | boundary |
| mlp.proj layer 2 | +0.811 dex | |
| mlp.fc layer 11 | +0.624 dex | boundary |

Four of the six worst-served matrices sit in the boundary shell, in both forks. So while the
*average* gain is small, the correction is concentrated exactly where a per-type rule fails hardest —
which is the right shape for a follow-up arm, and is worth checking in Arm A's per-matrix output.

**=== ITERATION 70: BAND 10 ATTACKED — it survives, but it is a BOUNDARY effect, not a smooth U ===**

*Band 10 was the largest effect in the campaign, so this iteration tried to break it. It survives two
attacks, fails a third, and the failure changes its functional form.*

**Attack 1 — is it the gradient in disguise? NO.** If log g were itself U-shaped in depth, the depth
term would be re-expressing the gradient:

| | depth-quadratic coefficient | t |
|---|---:|---:|
| **log g** | −0.00065 / −0.00088 | **−0.66 / −0.85** (nothing) |
| **log C** | +0.00911 / +0.01260 | **+2.88 / +3.99** |

**The gradient has no depth curvature; C has it strongly.** Not an artifact of the gradient.

**Attack 2 — is it driven by single layers? NO.** Leave-one-layer-out gives F between 21.83 and
29.74 (f1500) and 26.65 and 44.82 (f2000). Robust.

**Attack 3 — is it smooth curvature? NO, and this is the correction.** Dropping two layers at each
end collapses F from **28.98 → 4.65** and **41.51 → 3.38**. A genuine quadratic would survive
trimming; this does not. **The effect lives at the boundary, and the interior is flat.**

**The functional form is NOT identified by this data,** and I am recording that rather than picking a
winner. On top of type + gradient:

| term | AIC f1500 | AIC f2000 | t |
|---|---:|---:|---:|
| smooth quadratic | **−270.92** ← best | −239.90 | +7.56 / +8.19 |
| log distance-from-edge | −270.10 | −243.52 | −7.47 / −8.60 |
| distance-from-edge | −270.02 | −238.42 | −7.46 / −8.03 |
| 2-layer edge shell | −268.42 | **−245.17** ← best | +7.28 / +8.78 |

**Different forks select different winners and all four sit within ~3 AIC.** They agree on the
*substance* — edge layers have higher C, t = 7.3–8.8 — and disagree on the curve. Do not report a
quadratic as established.

**The residual profile after type + gradient (this is the real object):**

```
 layer:    0      1      2      3      4      5      6      7      8      9     10     11     12
f1500:  +.255  +.117  +.033  -.075  -.162  -.068  -.204  -.197  -.073  -.019  -.057  +.236  +.118
f2000:  +.436  +.237  -.047  -.067  -.166  -.117  -.254  -.215  -.140  -.051  -.080  +.218  +.115
```

**Both ends are lifted** — layer 0 at +0.26/+0.44 and layer 11 at +0.24/+0.22 — against a flat,
negative interior with a clean minimum at **layers 6–7** (−0.20/−0.25). The 2-layer both-ends shell
beats a first-layers-only term decisively (AIC −268.42 vs −241.97; −245.17 vs −227.65).

*(Correction within this iteration: an intermediate test contrasted "first layer" against "last
layer" using layer 12 as the last and found the last-layer term unresolved at t≈1.5. The network has
13 layer slots, 0–12, and the lift sits at layer 11. The both-ends form is correct.)*

**Amended band 10** from "U-shape in depth" to "**C is lifted at the boundary layers**", with the
form left open and both ends required to lift. The magnitude is ~0.25–0.44 dex against a ~0.10 dex
noise floor.

**Why this matters for REQ-036.** The per-type LR design assumes C is a per-type constant. Band 10
says the largest structural term is **per-depth**, not per-type, and is concentrated at the first and
last layers. If Arm A confirms it, a per-type-only prescription is leaving the biggest effect on the
table — a per-type × boundary rule would fit the data better. **This is a finding about the shipped
design, and it should be read before REQ-036's results are interpreted.**

**=== ITERATION 69: THE OFFSETS ARE MOSTLY NOT RESOLVED — and what is really there is a U-SHAPE IN DEPTH ===**

*Iteration 68 reported a six-way ordering of C's type offsets. This iteration tested it properly and
it does not hold up; what replaces it is stronger than what it replaces.*

**Correction to iteration 68.** I presented `attn.proj < attn.k < mlp.fc < attn.q < attn.v <
mlp.proj` as an ordering to be explained. **Only one of its five adjacent gaps is statistically
resolved:**

| adjacent pair | gap (f1500) | t | |
|---|---:|---:|---|
| attn.proj → attn.k | +0.298 | **2.45** | **resolved** |
| attn.k → mlp.fc | +0.042 | 0.56 | not resolved |
| mlp.fc → attn.q | +0.128 | 1.66 | not resolved |
| attn.q → attn.v | +0.081 | 1.30 | not resolved |
| attn.v → mlp.proj | +0.065 | 0.35 | not resolved |

Drop attn.proj and the whole spread halves, 0.614 → **0.316 dex**. The supportable statement is
**"attn.proj is low; the other five are barely distinguishable,"** not a six-way ordering. Band 9 is
corrected accordingly — in particular its "mlp.proj highest" clause rested on a t = 0.35 gap and is
withdrawn.

**An exhaustive search for a structural explanation, which failed.** With only six offsets, an
invented binary can fit by luck, so I tested **all 62 non-trivial binary partitions** and scored nine
named structural hypotheses against that full space:

| named hypothesis | R² (f1500) | R² (f2000) |
|---|---:|---:|
| is an MLP matrix / has a 3072 dim | 0.121 | 0.087 |
| writes to residual / second in sub-block | 0.062 | 0.059 |
| q or k (QK-normed) | 0.000 | 0.005 |
| v or proj (value path) | 0.002 | 0.002 |

**The best named hypothesis scores 0.104, and 34 of 62 arbitrary partitions (55%) do as well or
better.** The best partition overall is simply `{attn.proj}` (R² 0.699) — naming the outlier, not a
mechanism. **No binary structural property orders C's offsets.** Registered as a negative.

**What is actually there.** attn.proj's per-layer C values are not uniformly low — they are a **U**:

```
layer:  0     1     2     3     4     5     6     7     8     9    10    11
logC: 4.23  3.84  3.49  3.47  3.47  3.62  3.54  3.35  3.39  3.79  3.90  4.73
      ^^^^ high                    low in the middle                  ^^^^ high
```

**attn.proj's "low mean" is a middle-layers effect, not a type property.** Fitting a quadratic in
depth per type:

| type | quadratic coef | t | R² (quad) |
|---|---:|---:|---:|
| **attn.proj** | **+0.0274** | **5.70** | **0.793** |
| mlp.proj | +0.0327 | 2.16 | 0.349 |
| mlp.fc | +0.0115 | 2.29 | 0.477 |
| attn.q | −0.0063 | −2.27 | 0.378 |
| attn.k / attn.v | ~0 | 0.65 / −1.61 | 0.045 / 0.616 |

Adding **one common quadratic-in-depth term** to the type+gradient model:

| | RSS before | RSS after | **F(2,63)** |
|---|---:|---:|---:|
| fork-1500 | 2.5339 | 1.3198 | **28.98** |
| fork-2000 | 4.2195 | 1.8206 | **41.51** |

**This is the largest effect found in the campaign** — far above the F ≈ 4–8 of every structural
binary tested. It is also consistent with the proj-specific boundary/spatial field noted much earlier,
now visible directly in raw C rather than in a derived quantity, and strongest in exactly the type
(attn.proj, R² 0.83) whose offset was the one resolved result of iteration 68.

**The corrected picture:**

> **C is set by (i) the gradient, at a within-matrix response ratio near 2, (ii) a modest matrix-type
> offset of which only attn.proj's is individually resolved, and (iii) a strong U-shaped dependence
> on depth — high at the first and last layers, low in the middle — which is the dominant structural
> term and is strongest for attn.proj.**

**Registered as band 10.** *Caveat carried forward:* the depth axis here is the corrected layer index
(attention is skipped at layer 6, so probe slot ≥ 6 maps to layer slot+1) — the same mislabelling
that cost 40 iterations earlier in this campaign.

**=== ITERATION 68: C'S LEVEL AND C'S SLOPE ARE SET BY DIFFERENT THINGS ===**

*Band 8 said the productive target is C at fixed g. This iteration goes there, and finds the
band-7 grouping does NOT carry over from the slope to the level.*

**Retraction, immediately.** I first removed the gradient's effect by subtracting `2.07 · log g`
(the causal ratio) and got a **negative** variance share — "the gradient explains −47.9% of C's
variance." I briefly read that as a sign flip. **It is not.** Every relationship in the data is
positive: pooled corr +0.467/+0.408, every within-type slope +0.36 to +3.94, and the six type means
correlate +0.404. The negative share is arithmetic — all fitted slopes are *below* 2.07, so
subtracting 2.07·log g overshoots and inflates variance. **That script's 38%/81% decomposition was
computed on an over-subtracted residual and is discarded.** There is no Simpson reversal here.

**The correct decomposition.** Comparing nested models on C directly:

| | R² (f1500) | R² (f2000) |
|---|---:|---:|
| gradient alone | 0.218 | 0.166 |
| **matrix type alone** | **0.287** | **0.248** |
| type + gradient | 0.751 | 0.607 |

**Type identity alone outpredicts the gradient** (28.7% vs 21.8%; 24.8% vs 16.6%). C's structure is
not mostly a gradient effect.

**The finding: C's six type offsets do NOT group by residual-writer status.**

| rank | type (f1500) | offset (dex) | |
|---:|---|---:|---|
| 1 | **attn.proj** | **3.734** | ← residual writer, *lowest* |
| 2 | attn.k | 4.032 | |
| 3 | mlp.fc | 4.074 | |
| 4 | attn.q | 4.202 | |
| 5 | attn.v | 4.283 | |
| 6 | **mlp.proj** | **4.348** | ← residual writer, *highest* |

**The two residual writers sit at opposite extremes.** Their group gap is **−0.107 / −0.099 dex**,
at the ~0.10 dex noise floor — while the full type spread is **0.614 / 0.584 dex**, six times larger.
Ordering is identical across both forks.

> **The property that governs C's SLOPE (residual-stream position, band 7, p < 0.0001) is not the
> property that governs C's LEVEL. Whatever orders attn.proj below attn.k below mlp.fc … below
> mlp.proj is a different mechanism, and it is the larger effect.**

This is a genuine constraint: any single-mechanism account of C is now excluded by data.

**A hypothesis of mine, tested and rejected in the same iteration.** The within-type common slope
(+2.76 / +2.35) is far closer to the causal 2.07 than the pooled slope (+0.74), and its 95% CI
**contains 2.07** in both forks — suggesting the tidy model `logC = type offset + 2.4·log g`, which
would have retired band 6. **It fails decisively:**

| model | RSS (f1500) | AIC | |
|---|---:|---:|---|
| 6 offsets + 1 common slope | 2.5339 | −226.98 | |
| **6 offsets + 2 slopes (band 6)** | **1.4665** | **−264.35** | ← best |
| 6 offsets + 6 slopes | 1.4490 | −257.22 | |

**F(1,64) = 46.58 and 38.47** against one common slope. **Band 6's two-slope structure survives a
direct attempt to refute it** and is now the best model by both AIC and F in both forks.

**Registered as band 9.** Note it is deliberately a *negative* prediction — the residual-writer gap
must stay **below** 0.20 dex. If seeds show a large offset gap, band 9 is wrong and the level and
slope share a mechanism after all.

**=== ITERATION 67: THE CROSS-SECTIONAL SPLIT IS ~75% OMITTED-VARIABLE BIAS — and the omitted variable is C ===**

*This resolves the anomaly bands 6–7 left open: why neither cross-sectional slope (3.8, 1.2) equals
the within-matrix response ratio of 2.07.*

**The test.** REQ-023 randomises each matrix's LR multiplier over {0.6, 1.0, 1.7}, each matrix
receiving each level exactly once — a within-matrix experiment. Estimating the response ratio
`(d log λ/d log s) / (d log g/d log s)` **separately for the two band-7 groups** distinguishes:

- **(A)** the ratio is ~2 in both groups → the cross-sectional split is omitted-variable bias;
- **(B)** the ratio itself splits → the cross-sectional split is physical.

| fork | group | dλ/ds | dg/ds | **ratio** | first-stage F | n |
|---|---|---:|---:|---:|---:|---:|
| 1500 | internal | −0.956 | −0.502 | **+1.905** | 834 | 144 |
| 1500 | residual writer | −1.785 | −0.783 | **+2.279** | 280 | 72 |
| 2000 | internal | −0.962 | −0.535 | **+1.797** | 994 | 144 |
| 2000 | residual writer | −2.040 | −0.821 | **+2.484** | 330 | 72 |

**Difference, matrix-clustered bootstrap (4,000 draws):** +0.374, 95% CI **[−0.016, +0.743]**
(f1500, not distinguishable from zero); +0.688, 95% CI **[+0.327, +1.080]** (f2000, distinguishable).

**The decomposition, which is the point:**

| | f1500 | f2000 |
|---|---:|---:|
| cross-sectional slope gap | +2.173 | +2.183 |
| causal response-ratio gap | +0.374 | +0.688 |
| **share that is genuine physics** | **17.2%** | **31.5%** |
| **share that is omitted-variable bias** | **82.8%** | **68.5%** |

**Answer: mostly (A).** Perturb a matrix's learning rate and residual writers and internal matrices
respond *almost the same* (2.28 vs 1.91; 2.48 vs 1.80). The 3.8-vs-1.2 cross-sectional chasm is
**not** a difference in gradient→curvature physics. Roughly three-quarters of it is generated by
whatever else differs between matrices that happen to have different gradients — **and that
"whatever else" is C.**

Both first-stage F values (280–994) are far above any weak-instrument threshold, so neither ratio is
biased by instrument weakness; the unequal n (72 vs 144) is absorbed by the clustered bootstrap.

**Why this matters more than it first appears — it reframes the campaign's own method.** This is
Simpson's paradox, and the contamination is not a nuisance here: **it is the object of study.**
The cross-sectional slope is steep for residual writers *precisely because* C varies with gradient
differently in that group. Reading the cross-section as a "law with a broken exponent" was the wrong
frame. The correct statement:

> **The gradient→curvature response is approximately common across matrix types (~1.8–2.5, with at
> most a +0.69 group difference). The large apparent differences in cross-sectional exponent are
> ~75% a projection of C's own type-dependent structure onto the gradient axis.**

**Consequence for the remaining work.** Hunting a mechanism that makes the *exponent* differ by type
is chasing a shadow — bands 6 and 7 measure a projection of C, not a second physical law. The
productive target is C's dependence on structure directly, holding g fixed. **This does not weaken
band 7's grouping** (p < 0.0001) — it reinterprets what that grouping is telling us: residual
writers differ in C, and that difference *masquerades* as a steeper exponent.

**Standing caveat carried forward** (from the earlier IV audit, unchanged): the Wald ratio is a
**response ratio under LR perturbation**, not a structural causal parameter. The exclusion
restriction — that LR affects λ only through g — is an assumption and is probably not exactly true.
Everything above is a statement about how λ and g co-move under an LR nudge.

**Registered as band 8.**

**=== ITERATION 66: THE SLOPE TAKES TWO VALUES, NOT SIX — band 6 partially withdrawn ===**

**What is withdrawn.** Iteration 65 reported `F(8,60) = 8.27 / 4.31` for "six free slopes vs a
proj/non-proj binary" and concluded the six-way slope structure is real. **That test was
mis-specified.** It compared *six slopes and six intercepts* against *two slopes and **two**
intercepts*, so a difference in intercepts alone would produce a large F. The intercepts do differ
six ways — but that is C varying by matrix type, which is this campaign's premise, not a finding
about slopes.

**The correct test** holds intercepts free six ways in both models and asks only whether the slopes
need six values:

| model | RSS (f1500) | params | AIC |
|---|---:|---:|---:|
| pooled: 1 slope, 1 intercept | 7.9635 | 2 | −154.53 |
| 2 groups: 2 slopes, 2 intercepts | 3.0471 | 4 | −219.70 |
| **6 types: 2 slopes, 6 intercepts** | **1.4665** | **8** | **−264.35** ← best |
| 6 types: 6 slopes, 6 intercepts | 1.4490 | 12 | −257.22 |

**F(4,60) = 0.18 and 0.10.** Six slopes buy essentially nothing over two. AIC selects the
2-slope/6-intercept model in both forks.

**Confirmed independently by the within-group heterogeneity**, which is where this started:

| group | slopes (f1500) | Q | verdict |
|---|---|---:|---|
| residual writers | attn.proj 3.76±0.65, mlp.proj 3.93±0.28 | **Q(1) = 0.1** | homogeneous |
| internal four | q 1.34±0.34, k 0.87±0.35, v 1.41±0.51, fc 1.43±0.97 | **Q(3) = 1.3** | homogeneous |

(fork-2000: Q(1) = 0.0, Q(3) = 0.5.) **The 0.87 → 1.43 spread among the internal four is entirely
error bars.** Band 6's per-type thresholds were fitting noise across four types that agree.

**The corrected claim — simpler than what it replaces:**

> **The cross-sectional gradient exponent takes exactly two values: ≈3.8 for the two matrices that
> write to the residual stream, ≈1.2 (f1500) / 0.7 (f2000) for the four that write internally.
> Neither equals the within-matrix causal exponent of 2.07. C's intercept still varies all six ways;
> only the slope collapses to two.**

**Band 6 amended** to test the two-valued structure — including `F < 2.5` as a *requirement*, so a
seed showing genuine six-way slope structure now falsifies this rather than confirming it. Band 7's
group contrast is untouched and still holds at p < 0.0001.

**=== ITERATION 66b: BAND 7'S MECHANISM IS UNDERDETERMINED (declared, not discovered later) ===**

Band 7 attributes the split to **writing into the residual stream**. In this architecture that
property is **perfectly collinear** with a second one:

| type | writes to residual | second in sub-block (consumes a nonlinearity) |
|---|---:|---:|
| attn.q / attn.k / attn.v / mlp.fc | 0 | 0 |
| attn.proj / mlp.proj | 1 | 1 |

`attn.proj` consumes the softmax-weighted mixture; `mlp.proj` consumes ReLU². **No matrix in this
network separates the two properties**, so the observational data cannot distinguish them, and
band 7's *grouping* (p < 0.0001) is far better established than its *mechanism*.

**This is a design question, not an analysis question.** REQ-038's `|a|` and `|d|` fields separate
the readings: "mixed input" predicts residual writers have distinctly larger, slower-varying `|a|`;
"gradient arrives without crossing a nonlinearity" predicts `|a|` is unremarkable and the signal is
in `|d|`. A cleaner future test would need an architecture with a matrix that writes to the residual
stream *without* sitting second in its sub-block — not available here.

**Band 7 identifies what band 6 could not.** Iteration 65 asked what physically separates the two
projection matrices from the other four, and **shape is decisively ruled out by the committed
shapes themselves**:

| type | shape | slope (f1500) |
|---|---|---:|
| attn.q / attn.k / attn.v | (768, 768) | 1.34 / 0.87 / 1.41 |
| **attn.proj** | **(768, 768)** — *identical* | **3.76** |
| mlp.fc | (3072, 768) | 1.43 |
| **mlp.proj** | **(768, 3072)** — *its transpose* | **3.93** |

`attn.proj` has exactly the same shape as q, k and v yet sits 2.4–2.9 above them, and `mlp.fc` and
`mlp.proj` are transposes of each other yet differ by 2.5. **Parameter count, aspect ratio, fan-in
and fan-out all predict no difference and are therefore falsified.**

What remains is **position in the block**: `attn.proj` and `mlp.proj` are the two matrices whose
output is added back into the **residual stream**. The other four write into an activation consumed
inside the block. Grouping on that single structural fact:

| | writes to residual (n=24) | internal (n=48) | difference | permutation p |
|---|---:|---:|---:|---:|
| fork-1500 | +2.542 | +0.369 | **+2.173** | **< 0.0001** |
| fork-2000 | +2.422 | +0.239 | **+2.183** | **< 0.0001** |

Null computed by shuffling the 24/48 labels 20,000 times (mean −0.032, sd ~0.51), per the standing
permutation rule — and each proj type clears it **alone** (+3.39 and +3.57, both p < 0.0001), so
this is not one type dragging a pair.

**Two readings remain open, and REQ-038's `|a|`/`|d|` fields separate them.** Residual writers may
differ because (a) their input is a *mixture* of many block outputs rather than a single clean
activation, or (b) their gradient arrives directly down the residual highway without passing through
a nonlinearity. **`|a|` distinguishes these:** reading (a) predicts residual writers have distinctly
larger and more slowly varying `|a|`; reading (b) predicts `|a|` is unremarkable and the difference
lives in `|d|`. This is a further reason to keep REQ-038's probe folded into REQ-035 Arm A.

**Registered negative (iteration 65).** Four candidate omitted variables were tested as controls on
the six per-type slopes — weight norm, negative-curvature fraction, and layer index. **Every one
made the spread worse, not better** (uncontrolled 3.06 / 3.43; controlled 3.14–4.49 / 3.98–5.21).
None is the omitted variable behind the slope differences. *(A fourth candidate was excluded before
interpretation: it is built from the same Lanczos tridiagonal as λ_top and is circular by the
standing rule.)*

**Do NOT register corr(range, slope).** Its permutation null is **+0.72, not 0**, so the once-cited
+0.96 is uninformative and was retracted in iteration 64.

**Standing methodology rule (alongside the |corr| > 0.99 rename rule):** before citing any
correlation computed across groups, permute the group labels and report the null. A correlation
between two group-level summaries is guilty until the permutation clears it.

**Band 1 is the single most important number in the campaign.** It is the largest unexplained
effect (~50% of the variance in C) and has survived seven mechanism falsifications untouched. If it
reproduces across four independent seeds it is a property of the architecture and worth a dedicated
programme; **if it varies by more than ±0.3 dex across seeds it is a property of this one trained
network and the question largely dissolves.**

**Band 5 is registered as a NEGATIVE and matters as much as the positives.** Iteration 41 rejected
an ordinal "consumption order" reading because the spacing is 3× uneven. If the seeds show *equal*
spacing, that rejection was wrong and the ordinal model revives.

**Two required readouts, both zero-cost, both fixing errors this campaign made:**
- **the matrix-name → network-layer mapping**, emitted explicitly. The probe's `block` index is a
  parameter-bank slot; the model skips attention at layer 6, so for attention matrices the true
  layer is slot+1 for slot ≥ 6. **This campaign ran 40 iterations on a mislabelled depth axis.**
- **raw Ritz values plus `residual_tail`**, uncorrected. The geometric-tail correction is *not*
  applied in the committed data (median tail 0.024) and must stay reversible.

**Retired — do not score these.** Every band registered against a falsified mechanism: bilinear
q·k coupling, softmax saturation, nonlinearity exposure *as an explanation of levels*, within-block
consumption order, curvature concentration, Muon group rank, and QK-norm scale-invariance. Band 4
above survives only in its narrow form — nonlinearity predicts *negative curvature*, not the level.

---

### Arm A — seed replication (n=4). The load-bearing arm. Runs sequentially on 1 box.

Four independent seeds (0,1,2,3), each trained from scratch to step 1500, each forked into
the ladder s in {0.60, 1.00, 1.70}. **Registered question: is C a property of the
architecture, or of the individual trained network?**

- median |delta log C| across seed pairs **<= 0.10 dex** (the noise floor) => C is
  seed-independent; architecture determines it; the covariate hunt is justified.
- **>= 0.20 dex** => C is a learned per-network property; every static covariate model is
  then bounded away from the floor by construction, and the program pivots to state variables.
- in between => report the seed-reproducible fraction; that becomes the true ceiling for any
  covariate model, replacing 0.363 dex of "explainable" signal.

Also report corr(C_seed_i, C_seed_j), and separately whether the **type ordering** (attn.v
highest, attn.proj lowest) reproduces across seeds even if the levels do not.
**This arm is worth running even if every other arm is dropped.**

### Arm B — depth-sweep discriminator. Not scheduled (see Ordering).

Same recipe at 6, 12, and 24 blocks. Distinguishes absolute depth (block 6 of the 24-block
model matches block 6 of the 12-block model) from relative depth d/D (matches block 12).
Registered on attn.v and mlp.fc — the only two types with real depth structure. If neither
matches, depth is a proxy for a local quantity and the architecture hypothesis is dead.

### Arm C — shape / update-geometry sweep. Not scheduled (see Ordering).

The Muon shape factor sqrt(max(1, rows/cols)) currently takes only two values (2.0 for
mlp.fc, 1.0 for everything else), so update geometry has almost no natural variance to
explain anything with — despite REQ-023 showing a strong causal exponent of -1.3. Vary
head_dim and mlp_ratio so the factor spans {0.5, 1, 2, 4}. Registered: does C shift by the
-1.3 exponent REQ-023 measured, or does the shape factor act only through the effective LR
and not through C?

### Arm D — norm-pinning control. Not scheduled (see Ordering).

Project each Muon matrix back to a held Frobenius norm after every optimizer step; half the
matrices pinned +25%, half -25%, balanced by type, untouched matrices as internal control.
Registered band from REQ-023's gauge slope (-0.57): a held +25% norm change moves C by
**-0.055 dex** if the norm channel is causal at equilibrium, **0** if norm is purely a
transient channel. This is the one remaining test of the weight-norm hypothesis after its
cross-sectional death.

### Ordering

**Arm A first, and alone.** B, C and D are only interpretable once A says whether C is
seed-stable. Run Arm A's four seeds sequentially on one box (each seed is a short
from-scratch run to step 1500 — ~4 min — plus three 3-point-ladder forks); this is the single
most informative thing the program can do next and it does not need a fleet. **Do not run
B/C/D on this pass.** Re-file them against Arm A's result: if C is seed-stable they become
worth scheduling, and if it is not, the covariate programme they serve is moot.

### Success criteria

- Arm A reports cross-seed median |delta log C| with an explicit verdict against the two
  registered bands and comparison to the 0.100 dex floor.
- Every arm commits `per_matrix_curvature.json` with the existing field set — critically
  including `curvature_along_gradient`, `curvature_along_polar`, and `gradient_block_norm`,
  which carry the entire result above.
- A `summary.tsv` reporting per matrix: C, k, C_grad, and A.
- Shared-state gate as in REQ-019 (identical sha256, zero abs-diff, LR = base x mult).
- **Do not** apply the geometric-tail correction silently — commit raw Ritz values plus
  `residual_tail`, as REQ-019 does, so the correction stays reversible.

### Artifacts

`logs/kmaxwell/req035_C_mechanism/<arm>/`


## REQ-036: equalized-curvature per-type learning rates (the first design derived from C)

- status: **OPEN — priority 2, unblocked.** Headline arm needs 1 box; both nodes are free.
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

- status: **OPEN — priority 3, unblocked.** 1 box / 4 arms / fork@2000; both nodes are free.
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

## Template

```md
## REQ-<nnn>: <short title>

- status: OPEN
- requested: <name / timestamp>

<Self-contained request. Include repository, branch/SHA, commands, configs,
success criteria, artifact paths, and any ordering constraints. Do not include
secrets; refer to already-provisioned environment variables.>
```
