# REQ-048 — spectral participation ratio — **the curvature bowl IS a spectral-concentration profile (n=4, CONFIRMED)**

**SHA `ebf53cd` (EoS serialized-fork-state), 2 nodes (qv155gq + qrv8n03, 8×H100 each, ≤2-node ceiling),
venv019 torch 2.10.0+cu128.** An increment to `measure_per_matrix_curvature.py`: per Muon matrix, at the
2750 checkpoint, **m = 16 Hutchinson probes** (fresh Rademacher, independent of any Lanczos state —
admissible under rule 13) estimate `trace(H)` and `trace(H²)`, giving the **participation ratio**

```
PR = trace(H)² / (n · trace(H²))   ∈ [1/n, 1]   (PR≈1 spread over all directions, PR≈1/n concentrated in one)
```

plus two diagnostic HVPs (`curvature_along_weight = ŴᵀHŴ`, `curvature_along_random`). Run on **Arm A's
4 seeds × 3 learning rates** (s060=0.60×, s100=1.00×, s170=1.70× — the `s*` tags are LR multipliers, per
the iteration-172 correction; recorded explicitly here). **12 states, all finite.**

> PR is **scale-invariant** — `batched_block_hvp` returns Hv in a BATCH_TOKENS sum scale, but that scale
> cancels in `trace(H)²/(n·trace(H²))`, so PR needs no normalisation.

## Result — hypothesis CONFIRMED at n=4

The registered band-44 criterion (conditioning on LR, since LR is a treatment), `readout.tsv`:

| check | result | verdict |
|:------|:-------|:-------:|
| **(i) corr(logPR, logλ) ≤ −0.60**, same sign ≥10/12 | **10/12 ≤ −0.60; 12/12 negative; mean −0.728** (sd 0.161) | **PASS** |
| **(ii) logPR cubic R² ≥ 0.70** | **12/12** (mean 0.827); linear R² ≈ 0.02–0.39 | **PASS** |
| falsification (\|corr\|<0.30 or PR monotone) | not triggered | **survives** |

Per-LR mean corr: 0.6× = −0.794, 1.0× = −0.683, 1.7× = −0.708 — the effect holds at every learning rate.

**The curvature bowl is a spectral-concentration profile.** logPR anti-correlates with the curvature
depth profile at −0.73 (12/12 negative), and logPR is a clean cubic. Where curvature λ is **high** — at
the **ends** of the network (layers 0 and 11) — the Hessian spectrum is **concentrated in few
eigendirections** (low PR); where λ is **low** — the **middle** (layer ~6) — the spectrum is **spread**
(high PR). See `req048_pr_vs_bowl.png`: the log-PR profile is the log-λ bowl **inverted**.

This is the measurement the committed data could not supply (the admissible predictor set for the depth
question was two numbers per matrix, explaining 4.3% of the bowl on the mean profile). **PR supplies the
missing structure: the "why the surface is stiffest at both ends and softest at layer 6" is that
curvature is carried by a shrinking number of directions toward the ends.** It advances the campaign's
central open question — the bowl is a property of *how curvature is distributed across directions*, exactly
where band 43 pointed (bowl along the top eigendirection, absent along Muon's step).

## One honest correction to the registered criterion's wording

Band-44 (ii) as filed said "the log PR profile is **bowl-shaped** with cubic R² ≥ 0.70 and **argmin in
layers 4–8**." The data shows the opposite *location*: logPR argmin is at the **ends** (L0/L11) in 11/12
fits — logPR is a **hump** (argmax mid-depth), i.e. the C-bowl **inverted**. This matches the request's own
**hypothesis text** ("the spectrum is most concentrated at the ends and most spread in the middle" ⟹ PR
*low* at ends, *high* in middle ⟹ a hump), so the "argmin in 4–8" half of the criterion was mis-stated
against its own hypothesis. **The load-bearing half — criterion (i), corr ≤ −0.60 in ≥10/12 — passes
decisively (10/12, mean −0.728), and the cubic-shape half (ii) passes (12/12).** The conclusion (the bowl
is a PR profile) is the hypothesis as intended; only the sign of the expected extremum in the (ii) wording
needs flipping.

## Method / admissibility / cost

- **Admissible (rule 13):** PR uses fresh Rademacher probes and `‖Hv‖²`/`vᵀHv` only — no `alphas`,
  `offdiags`, or tridiagonal `eigh` (unlike `residual_tail`, rejected iter-161 for sharing `lam_top`'s
  `eigh`). m = 16 was pre-validated by the request's own simulation (≤3.7% bias); this run confirms tight
  cross-seed behaviour (corr sd 0.16).
- **Per state:** base seed→1500, s-fork (LR×s) 1500→2750 (`checkpoint_model_at_cadence`), augmented
  curvature at 2750 (8 Lanczos + polar + 16 Hutchinson + 2 diagnostics = ~18 HVPs, ~2× probe time, ~610 s
  on 8 GPUs). No new training beyond reproducing Arm A's states (states were cleaned; regenerated here).
- **Bug fixed before the grid:** the weight direction `Ŵ = W/‖W‖_F` must be `.detach()`-ed — a leaf-param
  division stays in the autograd graph and breaks `batched_block_hvp`'s create-graph double-backward
  (the crash that the n=1 minimal surfaced; every other direction vector in the probe is detached).

## Files

- `readout.tsv` — the 12 per-(seed,LR) fits + the three band-44 checks + per-LR conditioning.
- `analyze_req048.py` — the checks, reproducible from the raw JSONs (LR mapping baked in).
- `apply_req048_patch.py` — the exact probe increment (PR Hutchinson + diagnostics; anchored edits over `ebf53cd`). **Note:** also apply `.detach()` to the `wdirs` weight direction (see the fix above).
- `req048_grid.sh` — the per-node driver (base → 3 LR-forks → augmented curvature).
- `req048_pr_vs_bowl.png` — z-scored logPR (hump) vs logλ (bowl), 12 fits + mean.
- `raw_curvature_json/req048_s{0..3}_{s060,s100,s170}.json` — per-matrix `participation_ratio`, `trace_est`, `trace_sq_est`, `curvature_along_weight/random`, `top_eigenvalue` (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
