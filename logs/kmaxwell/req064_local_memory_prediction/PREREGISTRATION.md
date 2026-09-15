# REQ-064 pre-registration — predict local memory improvements beyond a strong global setting

**Committed BEFORE any prospective test-seed (7/8) outcomes**, per the request ("registered prospective
prediction pass"; "no policy trial on a correlation-only pass"). Harness `365c392d` + the committed
REQ-058/059 optimizer patches + the **REQ-063 restore fix** (`apply_req063_restorefix.py`, so declared
per-arm treatments survive `load_training_state`). Two-node ceiling; pilot ≤4 node-hours, ≤24 total.

## 1. Frozen global reference a_star

REQ-063 verified the controls but did not emit a single a_star; it is frozen here from the **clean**
global a-sweep (the a05/a1/a2 mixture arms declare mu=0.95 = the checkpoint's, so they are NOT affected by
the nomom restore-overwrite the B6 pilot exposed). REQ-059 global endpoints (step 2750, seeds 3–6, n=4):

| global a | val_loss@2750 |
|:--:|--:|
| **a=0.5** | **3.32943** ← best |
| a=1.0 (base mixture) | 3.33795 |
| a=2.0 | 3.35149 |
| (nomom, mislabeled mu=0.95 — excluded) | 3.34336 |

**a_star = 0.5.** Local treatments per sentinel: **a_star/2 = 0.25** (shorter) and **2·a_star = 1.0**
(longer). The all-reference global is a=0.5 everywhere; all-shorter = 0.25, all-longer = 1.0.

## 2. States, seeds, treatments (frozen)

- **Development bases:** seeds 0, 1, fork 2000. **Prospective test bases:** seeds 7, 8, fork 2000 —
  verified unused for this prediction task (REQ-058 used 0–2, REQ-059 used 3–6; seed-7 files elsewhere are
  the unrelated anneal-ablation / muonh351 experiments). Reserved here; **no test-seed outcome is inspected
  before this document is committed.**
- **18 sentinels** = six matrix types {attn.q,k,v,proj, mlp.fc, mlp.proj} × blocks {0, 6, 11}. Each sentinel
  is changed alone to a_star/2 or 2·a_star; the other 71 matrices keep a_star. Weight schedule, LR, weight
  decay, auxiliary updates, inherited state fixed.
- **Per base: 39 arms** = 36 selective (18×2) + 3 global (all-shorter/all-reference/all-longer). Each runs
  **256 completed updates** after the fork; selection loss at offsets **0, 64, 128, 256**. Primary response
  = paired loss difference at **256** vs the all-reference (a_star) global; 64/128 secondary. Missing 256
  endpoints are NOT backfilled with earlier offsets — such arms are reported with their actual update count
  and excluded from the 256 analysis.
- **Pilot (≤4 node-hours):** seed 0, block-6's six types × two changes + three globals = **15 arms**. If the
  intervention/measurement/budget checks below pass, expand to 39×4 = **156 continuations** (pilot included).

## 3. Pretreatment features (measured at the shared post-switch state, before the first affected update)

All features use the **same weights + diagnostic token ranges** at that state (log its completed-update
count). Cheap features for all 72 matrices; actual-direction (M3) for all 72 only if M3 is selected.

- **M2 sharpness (primary):** isolated shape-weighted spectral **S_i**, **G_i = r_i·‖g_i‖_nuclear**, and
  **S_i/G_i**; retain raw components, signs, failure flags. r_i includes the actual shape factor and
  relative LR (log the current base LR separately). FW budget validated on the 18 sentinels: **K20/K50,
  5 starts**; on each of 3 diagnostic subsets ≥90% of positively-resolved sentinel estimates change ≤5%
  under increased budget, and median pairwise rank-corr of S_i/G_i across subsets ≥0.8; nested-token
  sensitivity reported as rank vs absolute changes. No claim that Z=1 is a proven stability threshold.
- **Euclidean comparator:** **lambda_i** (8 Lanczos iters + convergence diagnostics) and lambda_i/‖g_i‖_F²,
  at that exact state on the same ranges. No nearby-checkpoint or differently-normalized substitution.
- **M1 type/depth prior:** mean dev S_i per (type, block) — a measurement-free baseline.
- **M3 actual-update direction (selected):** capture the actual post-polar, shape-scaled updates with
  inherited buffers at the common pre-first-affected-update state (preserve the raw gradient before Muon's
  in-place mutation); compare candidate selective directions with cloned buffers at identical W, g, data,
  including unchanged auxiliary optimizers.

## 4. Pre-registered prediction & success criterion (committed before test outcomes)

- **Fit** each feature model on the **18 dev sentinels** (seeds 0, 1) to the primary 256-update response
  (benefit of a_star/2 vs 2·a_star per sentinel). Report a single held-out prediction on **seeds 7, 8
  individually** (not pooled), with a **per-seed 10% RMSE gate** on the predicted vs realized response.
- **Primary hypothesis (H1):** M2 (S_i/G_i sharpness) predicts the held-out per-matrix benefit **with lower
  per-seed RMSE than both M1 (type/depth) and the Euclidean comparator**, on both test seeds, AND recovers
  the correct sign of the a_star/2−2a_star benefit on ≥13/18 sentinels per seed. Correlation sign alone is
  **not** sufficient (that was REQ-058's exposed test).
- **Decision:** only if H1 passes on both prospective seeds does REQ-065's unconstrained frozen policy trial
  proceed; otherwise REQ-064 returns a negative/again-inconclusive prediction result and REQ-065 is not run
  on a correlation-only pass. The negative-is-a-valid-outcome stance is retained.

## 5. Budget & safety

Two nodes max; pilot ≤4 node-hours (15 arms), total ≤24 node-hours (156 continuations). Independent
base-state hashes, data cursors, code SHA, actual LR traces, exact checkpoint steps, and per-probe operator/
loss normalization recorded. Never commit weights, optimizer tensors, checkpoints, secrets, or env dumps.

## Status
- **Frozen 2026-09-15 before any test-seed outcome.** Next: GPU base runs (seeds 0,1,7,8 → dump at 2000) +
  pretreatment feature probes, then the 15-arm pilot, then (gates permitting) the 156-continuation expansion.
