# REQ-051 — why each matrix has a different LR-to-curvature response — **both targets confirmed (n=4)**

**SHA `25d3208` (per_matrix_lr_muon), 2 nodes (wdkrxgq + q4zk6y3, 8×H100 each, ≤2-node ceiling),
venv019 torch 2.10.0+cu128.** Four independent seeds, each forked from its own serialized step-2000 base
(distinct hashes `3dc4/2b92/b8cd/0dba`, base val@2000 = 3.4428–3.4450), into **6 arms** carrying a
per-matrix LR multiplier from {0.50, 0.65, 0.85, 1.00, 1.30, 1.70} in a **6-level Latin square**
(`mult = L[(block+arm+seed)%6]`: every matrix sees every level once across arms; each arm assigns each
level to exactly 12 matrices — 2 of each of the 6 types; block→level mapping rotated per seed;
network-wide LR histogram identical across arms). fork@2000 → 2750, **combined probe** (curvature
`top_eigenvalue`/`trace_est`/`gradient_block_norm` + activation `a_frob`/`d_frob`/`align_ratio`) at
**2050 and 2750** on one fixed minibatch shared across a seed's arms. 24 continuations, 96 probe files.

## Registered targets (`readout.tsv`), matrix fixed effects, at equilibrium (2750)

Regression per seed: `d log λ = a + b·d log g + c·[mlp.proj]·d log g` (matrix-demeaned across the 6 LR arms).

| target | result @2750 | prediction | verdict |
|:-------|:-------------|:-----------|:-------:|
| **T1 — `mlp.proj` interaction `c`** | **+0.514 ± 0.085; c>0 in 4/4 seeds** (+0.507/+0.572/+0.378/+0.599) | c>0 each seed, pooled ~+0.6 | **CONFIRMED** |
| **T2 — causal elasticity `k` (= b)** | **+1.681 ± 0.088** | compare to +2.237 (REQ-045) / +1.922 (REQ-036), *not* the observational +3.173 | **gauge violation confirmed** |

- **T1:** the excess curvature-to-gradient elasticity of `mlp.proj` over the other five types replicates
  causally at n=4 — positive in every seed, pooled +0.51 (near the predicted +0.6). It is not a
  ranking artifact; `mlp.proj` genuinely moves.
- **T2:** the causal elasticity is **+1.68**, below the gauge-invariant value 2 (a pure reparametrisation
  `W=c·V` gives k=2, leaving `C=λ/g²` untouched), consistent with the causal benchmarks +2.24/+1.92 and
  **far below the observational +3.17**. So most of the observational λ–g coupling is causal LR response,
  and the residual below 2 is a real gauge violation — reproduced with tight cross-seed sd (0.088).

At the **early** checkpoint (2050) the picture is directionally the same but not yet settled: k=+1.36±0.28,
c=+0.99±0.25 (c>0 4/4). The elasticity rises toward equilibrium (1.36→1.68) and the mlp.proj excess
shrinks (0.99→0.51) between 2050 and 2750.

## Target 3 — the gradient's LR-response is almost entirely the activation channel

Accounting identity `g = ‖a‖_F ‖d‖_F ρ` (bias-free Linear) ⟹ `k_g = k_a + k_d + k_ρ`, with
`k_x = −d log x / d log(LR mult)` over the 6 arms (activation probe). At 2750 (per-matrix mean):

| k_a (activation) | k_d (backward) | k_ρ (alignment) | k_g (sum) |
|-----------------:|---------------:|----------------:|----------:|
| **+0.733** | +0.035 | +0.010 | **+0.778** |

**The gradient shrinks under higher LR almost entirely because the input activations shrink** (`k_a`
carries ~94% of `k_g`); the backward magnitude and token-alignment barely respond. So the between-matrix
variation in LR→curvature response is routed through the forward activations, not the backward path.

## Caveats / method

- n=4 independent bases (distinct hashes); within-arm all six arms load the same base and data cursor
  (Latin square changes only per-matrix LR, not the network-wide LR histogram). Judged with matrix fixed
  effects; per-seed effect sizes reported (readout).
- Probes at 32k tokens (slopes over 6 LR levels are robust at this size); λ = HVP/Lanczos top_eigenvalue
  (iters 8); g = same-minibatch `gradient_block_norm`.
- REQ-045 already showed the neighbour-LR channel is null; this balanced ladder is not re-interpreted as a
  neighbour test.

## Files

- `readout.tsv` — T1/T2 per-seed + pooled at 2050 & 2750, T3 decomposition.
- `analyze_req051.py` — the regressions + decomposition, reproducible from the raw JSONs + assignments.
- `make_req051_arms.py` — the 6-level Latin-square generator (per_matrix_lr_muon); `req051_run.sh` — driver.
- `assignments_s{0..3}.tsv` — per-(seed,arm,matrix) LR multiplier + type + block. `provenance.tsv` — 4 base hashes + val@2000.
- `raw_json/req051_s{0..3}_arm{0..5}_step{2050,2750}_{curv,act}.json` — 96 combined-probe files (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
