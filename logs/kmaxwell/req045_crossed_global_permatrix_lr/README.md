# REQ-045 — crossed global × per-matrix LR — **partial/total reading WITHDRAWN (registered negative)**

**SHA `25d3208` (PerMatrixLrMuon), 1 node (wlkmo03, 8×H100), venv019 torch 2.10.0+cu128.**
3 arms, `effective_lr(i) = base_lr × S_arm × m_i`: global `S_arm ∈ {0.7, 1.0, 1.4}` (one per arm,
every matrix), per-matrix `m_i` drawn independently per arm from {0.6, 0.85, 1.0, 1.2, 1.7}. Fork@2000
→ stop@2750, per-matrix curvature (`top_eigenvalue`) + `weight_frob` at 2750. Base val@2000 = **3.44369**
(target 3.44367).

## The crossing is identifiable (the design's whole point)

Regressing `d log λ` on **both** `log(own multiplier)` and `log(others' mean multiplier)` with matrix
fixed effects requires the two regressors to be separable — REQ-023 and any unbalanced variant give
`corr = −1.0000` (an arithmetic identity). The 3-global-level crossing breaks it:

| quantity | value |
|:---------|------:|
| **separability `corr(own, others' mean)` post-FE** | **+0.722** (PASS, < 0.9) |
| identifiability pre-check (draws, m only) | −0.182 |

Both regressors are separately estimable — so the two coefficients below are interpretable.

## Result — own-LR effect strong, neighbour effect null

`readout.tsv` (matrix FE, 216 obs = 72 matrices × 3 arms):

| coefficient | estimate | SE | t |
|:------------|---------:|---:|--:|
| **β_own** `d log λ / d log(own mult)` | **−1.161** | 0.091 | **−12.78** |
| **β_neighbour** `d log λ / d log(others' mean mult)` | **+0.143** | 0.126 | **+1.14** |

- **β_own = −1.16 (t = −12.8)** — a matrix's equilibrium curvature drops strongly with **its own** LR.
  This is **band 30** (a higher LR decouples λ from the gradient), now measured per-matrix with the
  neighbour channel held separate: doubling a matrix's LR multiplies its λ by ≈ 2⁻¹·¹⁶ ≈ 0.45×.
- **β_neighbour = +0.14 (t = +1.14) — NULL.** A matrix's curvature does **not** respond to its
  neighbours' learning rates.

## What it settles — iteration 124's partial/total reading is withdrawn

Iteration 124 read band 30's *shape* disagreement — REQ-023's per-matrix perturbation saturates by
s = 1.0, while Arm A's global ladder shows an even decline — as **partial vs total**: saturation being a
property of a matrix *in isolation*, the global ladder's extra decline coming from the network-wide
state change. That requires a real **neighbour (network) channel**: a matrix's λ responding to *others'*
LR. The registered test:

> **neighbour coefficient significant** → iteration 124 supported, band 30's shape design-dependent;
> **neighbour coefficient null** → the reading is **withdrawn**; the shape disagreement needs another
> explanation.

**The neighbour coefficient is null (t = 1.14).** So **the partial/total reading is withdrawn.** The
curvature-LR decoupling is a **purely local, own-LR effect** — a matrix responds to its own learning
rate, not to the network's. The band-30 shape disagreement between the two designs is therefore *not*
partial-vs-total; it requires a different explanation. (This is the request's `null` branch, pre-registered.)

## Caveats

- **n = 1 per (matrix, arm); independent draws per arm** (not balanced — the request explicitly allows
  this, since the neighbour term is estimated from the crossing, not from balance). β_own is far from
  noise (t = −12.8); β_neighbour's null is a genuine null, not low power on the own effect.
- **Curvature-cadence fix.** `checkpoint_model_at_cadence every=750` fires at 2250, not 2750 (2750 is
  not a multiple of 750) — the same cadence miss as REQ-036. Fixed to `every=250` so the model is
  checkpointed at **2750** and the curvature probe reads the intended step (verified `ckpt_step=002750`).
- λ = HVP/Lanczos `top_eigenvalue` at 2750, merged across 8 ranks.

## Files

- `readout.tsv` — separability + β_own + β_neighbour + significance.
- `analyze_req045.py` — the 2-regressor FE regression, reproducible from the raw JSONs.
- `make_req045_arms.py` — the crossed-design config generator (+ identifiability pre-check).
- `req045_draws.json` — per-arm per-matrix `m_i` draws + S_arm + types + identifiability corr.
- `req045_status.tsv` — per-arm final val + ckpt step + curvature exit.
- `raw_curvature_json/req045_s{07,10,14}.json` — per-matrix `top_eigenvalue` (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling (one box).
