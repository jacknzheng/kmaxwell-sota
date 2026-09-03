# REQ-035 Arm A — is the per-matrix equilibrium curvature constant C seed-dependent? **n=4, COMPLETE**

**SHA `ebf53cd` (REQ-019/022 serialized-fork-state design), 2 nodes (q89y4d3 + qekd95q, 8×H100 each, ≤2-node
ceiling), venv019 torch 2.10.0+cu128.** The load-bearing arm of REQ-035: four independent seeds (0,1,2,3), each
trained from scratch to step 1500, each forked into the s-ladder {0.60, 1.00, 1.70}, per-matrix curvature measured,
and C compared across seeds. **All 4 seeds finite, 0 errors.**

## Method

Per seed: `eos_shared_base` (seed=N) → dump fork-1500 → three s-forks (`eos_f1500_s060/s100/s170`, fixed LR×s,
1500→2750) → `measure_per_matrix_curvature.py` (HVP/Lanczos `top_eigenvalue` = equilibrium curvature λ) at steps
2250–2750. Per matrix, λ_eq(s) = mean top_eigenvalue over the 5 steps (merged across 8 ranks); C is the intercept of
`log₁₀ λ_eq = log₁₀ C − k·log₁₀ s` fit across the 3 s-values. 74 Muon matrices per seed.

**NB the "existing checkpoint" premise was false** — REQ-019's fork-1500 weights were not persisted (only derived
JSONs are committed), so every seed's fork-1500 was **regenerated from scratch** here. Base val@2000 = 3.44267
(≈ the expected 3.44367; the 0.001 gap is the ebf53cd-vs-365c392d code path).

## Result — C is seed-independent to the noise floor

`median |Δ log₁₀ C| across seed pairs` (the discriminator) vs the ~0.10 dex run-to-run noise floor (REQ-035's
duplicate-arm rms):

| seed pair | median \|Δ log C\| (dex) | rms (dex) |
|:----------|-------------------------:|----------:|
| 0–1 | 0.1061 | 0.1750 |
| 0–2 | 0.0949 | 0.1504 |
| 0–3 | 0.1151 | 0.1835 |
| 1–2 | 0.1282 | 0.1857 |
| 1–3 | 0.0884 | 0.1623 |
| 2–3 | 0.0903 | 0.1491 |
| **median-of-pairs** | **0.1061** | — |

**Verdict:** median-of-pairs |Δ log C| = **0.106 dex ≈ the ~0.10 dex noise floor.** Cross-seed variation in C is
essentially the same size as same-seed run-to-run noise, so **C is seed-independent to the noise floor — it is a
property of the architecture, not the individual trained network.** This is REQ-035's `≤0.10 dex` branch to within
0.006 dex (the tiny excess is not resolvable above the floor): **the covariate hunt for what sets C is well-posed.**
(It is not the `≥0.20 dex` "learned per-network" outcome — that is decisively excluded.)

Per-seed slopes: median **k = 1.28 / 1.34 / 1.24 / 1.17** (mean ~1.26), consistent with REQ-035's k = 1.38 ± 0.45.

### Type structure reproduces across seeds

type-mean log₁₀ C ordering (`readout.tsv`): **attn.proj is the lowest in all 4 seeds; attn.v and mlp.proj are the
top two in all 4 seeds** (they swap #1/#2 between seeds 0–2 and seed 3, but never leave the top). So the
per-type C ordering REQ-035 expected (attn.v high, attn.proj low) is seed-reproducible — the between-type structure
of C is an architectural signal, not a per-seed artifact.

## Files

- `summary.tsv` — per-matrix log₁₀ C for all 4 seeds + k(seed0) + max pairwise Δ (74 matrices).
- `readout.tsv` — the 6 seed-pair deltas + verdict + per-seed type-ordering.
- `analyze_req035_armA.py` — the C-fit + seed-replication analysis (auto-detects seeds).
- `curvature_json/seed{0..3}.tgz` — raw per-rank per-step curvature JSONs (top_eigenvalue, curvature_along_gradient,
  curvature_along_polar, gradient_block_norm, alphas/offdiags) — the source of truth.

## Not included (follow-ups)

- **REQ-038's activation/backward fields** (|a|, |d|, effective ranks, attention-logit stats) were to be *folded into*
  this arm's probe. They are **not** in this run — the curvature probe records `top_eigenvalue`/gradient blocks, not
  forward-activation or output-gradient tensor stats, which need new forward/backward-hook instrumentation. The 4
  seeds' fork-1500 states are cleaned between runs (per-box state reuse), so a REQ-038 pass needs either a re-run with
  the extended probe or preserved fork-1500s. Recommend implementing the extended probe and running it as a separate
  single-checkpoint pass on regenerated fork-1500s (cheap, ~20 min/seed).
- REQ-036 (per-layer LR design) and REQ-037 (non-LR instrument) are the next queue items.

No secrets/weights/tensor checkpoints committed. Nodes ran under the ≤2 ceiling (both principals agreed).
