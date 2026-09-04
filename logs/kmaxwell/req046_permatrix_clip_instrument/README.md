# REQ-046 — per-matrix gradient-clip instrument (REQ-037 arm 4) — **INSTRUMENT INERT; band 13 UNRESOLVED**

> **⚠️ CORRECTION (2026-09-04, after client iteration 130).** The original conclusion here — "band 13's
> causal reading overturned" — was an **over-read and is retracted.** The clip scaled `clipped_gradient_block_norm`
> (`c·‖g‖`, slope +1.003) but the **raw** `gradient_block_norm` the network actually experiences did **not**
> move: slope **+0.0028, CI [−0.0143, +0.0200]** (verified from this deliverable's own JSONs). Muon's
> polar map normalises the scale out of the update, so the clip never reached the loss surface — exactly
> **band 29** (Muon's update is gradient-scale-invariant), which I failed to apply to my own instrument
> design. **The exponent is therefore 0/0 — uninformative about curvature–gradient physics, not evidence
> against it. Band 13's causal reading is UNRESOLVED, not overturned.** The data below is correct and
> complete (it is what made the correction possible); only the interpretation was wrong. The corrected
> reading is in "## What it settles (corrected)" below.

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

## What it settles (corrected) — nothing about band 13; the instrument is inert under Muon

The original claim assumed the clip changed *the gradient the network experiences*. It did not. The
decisive quantity is the **raw** gradient block norm, and it did not move:

| quantity | slope vs log(clip) | CI |
|:---------|-------------------:|:---|
| `clipped_gradient_block_norm` (`c·‖g‖`) | **+1.003** | — (registered "first stage", but *mechanical*) |
| **`gradient_block_norm` (RAW, what the network sees)** | **+0.0028** | **[−0.014, +0.020]** (includes 0) |
| `top_eigenvalue` (λ) | +0.009 | — |

The causal chain (client iteration 130, confirmed on this data):

```
clip on grad_chunk → momentum buffer → polar_express normalises to unit spectral norm
  ⟹ orthogonalised update unchanged ⟹ weight trajectory unchanged
  ⟹ raw gradient unchanged (+0.003) ⟹ curvature unchanged (+0.009)
```

**The clip never reached the loss surface.** So the exponent `d log λ / d log(clip) = +0.009` is **0/0**:
both the numerator (λ response) and the *real* denominator (raw-gradient response) are zero. It is
**uninformative about curvature–gradient physics, not evidence against it.** Band 13's causal reading is
**UNRESOLVED, not overturned.** Full network compensation is separately excluded (it predicts a raw-gradient
slope near −1, far outside the CI) — so the finding is specifically "the intervention was inert," which is
exactly **band 29** (Muon's update carries no gradient-magnitude information): moving the clip pre-momentum
(defect 1) stops `polar_express` cancelling it *within a step*, but the normalisation still removes the scale
from the update, so the trajectory barely changes.

**Why the original read was wrong.** I treated the +1.003 `clipped_gradient_block_norm` slope as "the first
stage," but that field is `c·‖g‖` by construction — it measures that the probe *sees* the clip, not that the
*network* does. The load-bearing first stage is the raw-gradient slope (+0.003 ≈ 0). A valid instrument must
move the raw gradient; under Muon a magnitude intervention cannot (the normalisation removes that channel by
construction), so the question "does gradient magnitude cause curvature" may be **unanswerable by
intervention under this optimiser** — itself a real finding, and where band 29 already pointed. The remaining
options (per-matrix loss weights) reintroduce iteration 119's exclusion problem.

**Lesson (recorded):** adversarially validate an intervention against the campaign's *own established
findings*, not only against the code path it edits — band 29 was ten iterations old and made this instrument
inert; I did not connect it. (Consistent with [[adversarial-validate-every-lead]].)

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
