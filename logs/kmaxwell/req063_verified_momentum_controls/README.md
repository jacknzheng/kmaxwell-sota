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

## Stage B — verify the optimizer after loading the checkpoint (in progress)

Uses the **actual training setup + restore hooks**, not a bare constructor or standalone scalar solver.
Checks, most CPU-reproducible against the real load path with a tiny model, GPU only where a real
training step is required (probe replay, verified control pilot; ≤2 node-hours):

1. **nomom overwrite** — does `load_state_dict` change the no-momentum arm's `mu` 0→0.95? Restore model /
   inherited buffers / counters first, then reapply only declared treatment hyperparameters; assert &
   log resolved groups before the first affected update; verify zero momentum reaches the post-polar
   update independent of an inherited momentum buffer at fixed gradient.
2. **ordinary Muon, mu=0.95** — an explicitly named control, distinct from zero momentum, the
   eight-stream mixture, and the exact-age single-EMA control.
3. **name resolution** — 72/72 global, 1/72 selective, exact per-type counts for old allocations; abort
   on missing/extra/empty/duplicate/`_orig_mod` mismatch; never silently leave a param at default.
4. **a=1 reproduces the mixture**; changing a reaches the post-polar update; cloned-buffer transition,
   counters, first affected update, LR schedule, weight decay, auxiliary optimizers all checked; common
   baseline update preserved at the switch for every arm incl. zero momentum; treatments activate at the
   same subsequent update; inherited buffers not reset in just one arm.
5. **exact-age EMA** — implemented schedule / inherited mass / outer blend vs the *actual* mixture
   recurrence after restoring counters; impulse / constant / alternating sequences; per-step mass, mean
   age, age variance, beta through 750 updates; tol 1e-10 in CPU float64 + production precision.
6. **no-probe/probe replay** — identical full state + batches, hashed weights/buffers, preserved RNG &
   data cursor; a probe must not change the trajectory; central differences restore saved tensors
   exactly rather than assuming add/subtract reverses FP rounding.

Then a verified control pilot: `train_steps=3250`, 1× batch 524288, microbatch 64, original LR schedule,
REQ-054 eight-stream decays/weight schedule.

## Files
- `impl/req063_stageA.py` — Stage A repairs (recovery manifest, corrected REQ-059 stats, REQ-058
  endpoint relabel + retrospective features). `stageA_readout.txt` — captured output.
- Stage B impl + raw land here as it runs. No secrets/weights/tensor checkpoints committed.
