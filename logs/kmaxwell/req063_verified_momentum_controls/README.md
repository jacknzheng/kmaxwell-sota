# REQ-063 — verify restored optimizer controls & repair the returned evidence

**Harness `365c392d` + committed REQ-058/059 patches. Two-node fleet ceiling; GPU pilot ≤2 node-hours,
≤10 node-hours total.** Repairs the omissions the Sept-15 audit reproduced (nomom parameter overwrite,
missing REQ-058 endpoints, incorrect REQ-059 significance). **Not** a repeat of all REQ-057–060 runs.

Two stages: **A** = recover + verify existing artifacts with no training (CPU); **B** = verify the
optimizer through the actual restore hooks before any production launch, then a verified control pilot.

---

## Stage A — recovery + corrected statistics (CPU, no training) — DONE

`impl/req063_stageA.py` → `stageA_readout.txt`. Self-contained (t-CDF via regularized incomplete beta;
no scipy). Reproducible from the committed REQ-057/058/059 raw files.

### A1. Recovery manifest
- **COMMITTED (verifiable):** impl generators + optimizer patches, raw per-arm selection-loss TSVs /
  spectral JSONs, and the READMEs/readouts for REQ-057/058/059/060.
- **UNVERIFIED (node-local, lost):** stdout logs, resolved run configs, per-rank state/data manifests,
  and the named allocation JSONs that lived on the REQ-058/059/060 execution hosts (nodes stopped after
  delivery). Per the request, this missing runtime evidence is **UNVERIFIED** — not evidence that a
  favorable fix existed.
- **Recovered source:** harness `365c392d695f95dc9a4fb89095e85a6a7b5d551e` + committed REQ-058/059
  patches (`req058_layerwise_momentum_response/impl/apply_req058_opt.py`,
  `req059_sharpness_guided_momentum/impl/apply_req059_opt.py`).

### A2. REQ-059 corrected inference (paired t-tests, n=4 seeds 3–6, **df=3**; Holm cumulative-max)
Holm adjusted p = cumulative max of `(m − rank + 1)·p_sorted`, capped at 1. 95% paired CIs use
t₀.₉₇₅(3)=3.182. Guided − control at the step-2750 endpoint:

| control | mean Δ | 95% CI | t | p (t, df3) | Holm |
|:--|--:|:--|--:|--:|--:|
| global_a05 | **+0.00914** | [+0.00907, +0.00920] | 419.1 | 3.0e-08 | 0.000 |
| global_a1 | +0.00061 | [+0.00044, +0.00078] | 11.5 | 1.4e-03 | 0.004 |
| global_a2 | −0.01293 | [−0.01324, −0.01262] | −130.8 | 9.8e-07 | 0.000 |
| euclidean | −0.00012 | [−0.00035, +0.00011] | −1.70 | 1.9e-01 | 0.188 |
| typedepth | −0.00018 | [−0.00020, −0.00015] | −20.2 | 2.7e-04 | 0.001 |
| shuffled | −0.00057 | [−0.00082, −0.00032] | −7.32 | 5.3e-03 | 0.011 |
| reversed | −0.00169 | [−0.00196, −0.00143] | −20.2 | 2.7e-04 | 0.001 |
| nomom | −0.00480 | [−0.00497, −0.00463] | −90.4 | 3.0e-06 | 0.000 |

**A success decision checks the corrected p:** a practical win requires guided to *beat* the control
(mean ≤ −5e-4, CI upper < 0, Holm p < 0.05). Guided **loses** to `global_a05` (+0.00914) and to
`global_a1`, is beaten by `global_a2`/`nomom`/`reversed`/`shuffled`, and ties `euclidean`/`typedepth`.
**PRACTICAL WIN = NO.** The negative REQ-059 policy outcome is preserved after correct inference.
(Caveat: the tiny CI widths reflect only 4 same-machine seeds; they certify the *sign*, not a
deployment-grade effect size. `nomom` here is the REQ-059 arm and inherits the Stage-B overwrite caveat.)

### A3. REQ-058 endpoint relabel
All **53/53** seed-0 / fork-1500 files end at step **1560**, not 1564 → **60-update** observations.
The original analysis substituted that last row for the absent fork+64. Corrected: fork-1500 rows are
labeled 60-update and **excluded** from the 64-update analysis. Endpoints could not be recovered (the
run state is node-local and gone). Original raw files retained; corrected derived tables published here.

### A4. REQ-058 retrospective feature comparison (clean 64-update fork-2000 data; **retrospective**, not a gate)
Spearman(feature, a=2 selection-loss penalty at fork+64=2064), fit on seed-0 (DEV), reported per test
seed 1/2. Correlation sign is **not** the old gate.

| feature | seed0 (DEV) | seed1 (TEST) | seed2 (TEST) |
|:--|--:|--:|--:|
| raw S_i | +0.796 | +0.474 | +0.736 |
| S_i / G_i | +0.305 | −0.036 | +0.271 |
| type/depth prior | +0.796 | +0.470 | +0.742 |

Raw S_i (and its near-identical type/depth prior) replicate a positive rank correlation with the
long-memory penalty on held-out seeds; S_i/G_i does not. **Missing feature columns** (not invented,
not joined by label): **Euclidean λ_i** and **actual-update-direction curvature** were never computed
for REQ-058 seeds 0–2. Their prospective comparison is deferred to **REQ-064**. A positive
retrospective correlation is not a demonstrated allocation gain (REQ-059 already showed none).

---

## Stage B — verify the optimizer after loading the checkpoint

Uses the **actual training setup + restore hooks** (`harness/hooks.py:load_training_state` and the real
`Muon`/patched classes at `365c392d`), not a bare constructor or standalone scalar solver. CPU
verifications (items 1–5) done against the real load path; item 6 + the pilot need a real training step.

### B1. nomom overwrite reproduced + fix + zero-momentum independence — **PASS** (`impl/req063_stageB_restore.py`)
The restore path `for (_,built),sd in zip(...): built.load_state_dict(sd)` (`hooks.py:613-614`) uses
torch's `Optimizer.load_state_dict`, which **overwrites the current param-group hyperparameters from the
saved group**. Reproduced on the real `Muon`: a nomom arm (`mu=0.0`) loading a mixture checkpoint's group
(`mu=0.95`) has its **`mu` silently overwritten 0.0 → 0.95** — so the returned REQ-058/059 `nomom` arm
was *not* no-momentum. The audited fix: after `load_state_dict`, **reapply only the declared treatment
hyperparameters** (here `mu=0`), keeping restored buffers/counters. Verified zero momentum then reaches
the post-polar update **bit-independent of the inherited momentum buffer** at fixed gradient
(max|Δ|=0.0 across two different inherited buffers; mu=0.95 differs by 1.48), equal to
`zeropower(g)·scale` to one bf16 ULP.

### B2/B5. exact-age matched EMA — **PASS** (`impl/req063_stageB_exactage.py`, float64)
Independent explicit lag-histograms vs the committed closed-form recurrence, after the real warm/clone
convention (pre-switch plain Muon `mu=nu`, cloned to all 8 streams at the switch): single-EMA
mass/mean-age agree to **2.4e-15 / 2.5e-13**; the mixture `target_age` to **1.6e-13**; and the solver's
matching invariant is the **post-update ν-blended increment age** = target_age to **1.4e-14** (not
`mom/mass`, not the pre-blend). Age **variance necessarily differs** (a single EMA cannot span 8
timescales) — correct for an age-matched control. Realized buffer values verified under impulse / constant
/ alternating gradients (max|Δ| 0 / 2.3e-15 / 9.7e-17). Per-step mass/mean-age/variance/beta through 750
updates in the readout.

### B3. parameter-name resolution — **PASS** (`impl/req063_stageB_names.py`, real 12-layer GPT)
The real model has exactly **72 Muon matrices** (12 blocks × {attn.q,k,v,proj, mlp.fc,proj}, 12/type).
Balanced allocation resolves **72/72**, unique keys, bins **24/24/24**, per-type **4/4/4**; `_orig_mod.`
prefix strips and resolves 72/72; selective target resolves **1/72**; the tag hook **asserts** the target
∈ model (aborts on a bogus name), never silently leaving a param at default.

### B4. a=1 reproduces the mixture — **PASS** (same script)
`PerturbedAnnealedWeightsMuon(perturb_a=1.0)`: `_perturbed_decays == decays` (bₖ^(1/1)=bₖ) and
`_is_perturbed` is always False → the base annealed kernel is used. Numeric `compute_polar_input(a=1)` vs
`AnnealedWeightsMuon` on identical fresh grad+streams: **max|Δ| = 0.0**.

### B6. no-probe/probe replay + verified control pilot — **GPU (≤2 node-hours), pending**
Identical full state + batches, hashed weights/buffers, preserved RNG & data cursor; a probe must not
change the trajectory; central differences restore saved tensors exactly. Then the verified control pilot
with the **fixed restore path** and an explicitly named **ordinary Muon, mu=0.95** control (distinct from
zero momentum, the eight-stream mixture, and the exact-age single-EMA control): `train_steps=3250`, 1×
batch 524288, microbatch 64, original LR schedule, REQ-054 eight-stream decays/weight schedule.

## Files
- `impl/req063_stageA.py` — Stage A repairs (recovery manifest, corrected REQ-059 stats, REQ-058
  endpoint relabel + retrospective features). `stageA_readout.txt` — captured output.
- Stage B impl + raw land here as it runs. No secrets/weights/tensor checkpoints committed.
