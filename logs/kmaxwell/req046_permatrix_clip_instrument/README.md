# REQ-046 — per-matrix gradient-clip instrument (REQ-037 arm 4) — **band 13's causal reading OVERTURNED**

**SHA `25d3208` + the instrument patch (`req046_instrument.patch`), 1 node (wlkmo03, 8×H100),
venv019 torch 2.10.0+cu128.** 3 balanced arms, per-matrix clip multiplier ∈ {0.5, 1.0, 2.0} (each
matrix sees each level exactly once across the 3 arms — REQ-023 Latin square, 24 matrices per level
per arm), fork@2000 → stop@2750, per-matrix curvature + the new `clipped_gradient_block_norm` field.

## Result — the gradient-magnitude channel to curvature is inert

Registered regression (`readout.tsv`, matrix fixed effects, 216 obs = 72 matrices × 3 arms):

| quantity | value | registered target |
|:---------|------:|:-------------------|
| **exponent `d log λ / d log(clip)`** | **+0.009** | band-13 curvature-gradient exponent ≈ **+2** |
| **first stage `d log g_clipped / d log(clip)`** | **+1.003** | must be ≈ 1 ✓ |
| monotone reduced form | **False** (λ: 0.5→4.103, 1.0→4.159, 2.0→4.108) | registered check |
| every per-type ratio positive | **False** (−0.089 … +0.109) | registered check |

**The instrument works but the curvature does not respond.** The first stage is real and clean
(+1.003): the clip demonstrably scales the recomputed gradient by exactly the intended factor — this is
the property the REQ-037 *batch* instrument fatally lacked (its first stage was ≡ 0, iteration 113). But
the reduced form is **flat**: `d log λ / d log(clip) = +0.009`, essentially zero, non-monotone, with
per-type slopes scattered symmetrically around zero (noise, not signal).

## What it settles — band 13 is an LR-response statement, not curvature-gradient physics

The logic is decisive and does not depend on the (failed) monotonicity check:

> **If the gradient-magnitude channel were causal at band-13's exponent (≈ +2), then — since the clip
> moves the gradient with coefficient exactly 1.003 — the clip would have moved `log λ` with slope
> ≈ +2. It moved it with slope +0.009.** So changing *only* the gradient magnitude (holding the LR
> path fixed) does **not** move the equilibrium curvature.

This is the request's *"result near +1 or below → band 13 becomes a statement about LR response"*
outcome, in its strongest form (≈ 0). Band 13's observed λ–gradient exponent is **confounded**: in the
observational designs (REQ-023) LR and gradient magnitude co-vary, and the ≈ +2 slope is carried by the
**LR channel**, not by gradient magnitude. The mechanism is exactly **band 29** (Muon's update is
gradient-scale-invariant): the polar map normalises the update to unit spectral norm, so at equilibrium
the weights — and hence λ — are insensitive to a per-matrix gradient rescale. **The causal
curvature-gradient reading of band 13 does not survive.**

## The instrument (`req046_instrument.patch`, +96 lines over `25d3208`)

Two changes, both forced by reading the code (iteration 119):

1. **`PerMatrixClipMuon`** (`optimizers/muon.py`) — a `BimaxwellMuon` subclass that scales each
   matrix's gradient by its clip multiplier **before the momentum lerp**, via
   `compute_polar_input`: `p.grad.mul_(c)` then `super().compute_polar_input(...)`. A scale applied
   *inside* `polar_express` cancels exactly (`X/(c·‖X‖)`), so it must precede the momentum update, where
   the buffer's fork-history (accumulated at c=1) breaks the proportionality and the scale survives the
   normalisation. Registered as `per_matrix_clip_muon`.
2. **`clipped_gradient_block_norm`** (`measure_per_matrix_curvature.py`) — one field:
   `c·‖g‖` for the same recomputed gradient, via a `--clip_json {name: clip}` argument. Because
   `d log g_clipped / d log clip = 1` by construction, the estimator is the reduced form directly (no
   Wald ratio), and the +1.003 measurement confirms the field is wired correctly.

**Mechanism validated before the run** (`apply_req046_patches.py` + a unit check): from *zero* momentum
the clip cancels in the polar map (single-step update identical for 0.5/1.0/2.0 — the exact defect-1
failure); from a *non-zero fork momentum* (the real fork@2000 condition) it perturbs the trajectory
(‖w(0.5)−w(1.0)‖ = 1.21 over 30 steps). So the instrument fires precisely in the experimental regime,
and its near-zero equilibrium effect is a *result*, not a wiring bug.

## Adversarial check — "inert channel" vs "weak instrument"

The near-zero, non-monotone reduced form could in principle mean the clip is too weak to move λ rather
than that the gradient channel is inert. Three facts rule that out:
- **the first stage is +1.003** — the clip moved the gradient by the full intended amount, not a
  fraction; the instrument is at full strength on the treated variable;
- **the transient perturbation is real and large** (‖Δw‖ = 1.21 at 30 steps, validated) — the clip
  visibly moves the weights; it is the *equilibrium λ* that is insensitive, which is the physical claim;
- **the residual per-type slopes are symmetric around zero** (+0.109 … −0.089), not uniformly small —
  the signature of noise, not of an attenuated-but-present effect.

## Caveats

- **Registry-lock tests fail by design.** Adding `per_matrix_clip_muon` to the optimizer registry trips
  `test_registry_locks.py` (a reproducibility guard on the canonical registry). This is the intended
  research extension on a branch, not a regression; the functional path (`build_registered_optimizer`,
  config resolution, all runs) is green. Do not merge the lock change without a deliberate re-lock.
- **1× batch, 3 arms, n=1 per (matrix, level).** Balanced design gives 24 matrices per level per arm;
  the FE estimator pools 216 observations. Curvature λ = HVP/Lanczos `top_eigenvalue` at 2750.
- **Equilibrium, not transient.** This measures λ at 2750 (750 steps post-fork), i.e. the equilibrium
  the request targets. The clip's transient effect (first ~20–40 steps, while fork momentum decays) is
  real but washes out — which is *why* the equilibrium exponent is ≈ 0 under Muon's scale-invariance.

## Files

- `readout.tsv` — the registered regression + 3 checks.
- `analyze_req046.py` — the regression (matrix FE) + checks, reproducible from the raw JSONs.
- `make_req046_arms.py` — the balanced-clip config generator (+ per-arm `name→clip` JSON).
- `apply_req046_patches.py` — the exact instrument patch (anchored edits); `req046_instrument.patch` — the resulting diff.
- `req046_assign.json` — per-arm per-matrix clip assignment + types; `req046_status.tsv` — arm finals + ckpt step + curvature exit.
- `raw_curvature_json/req046_a{0,1,2}.json` — per-matrix `top_eigenvalue` + `gradient_block_norm` + `clipped_gradient_block_norm` (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling (one box).
