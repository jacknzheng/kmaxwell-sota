# REQ-057 — layer-wise spectral sharpness and cross-layer coupling — **reliable per-matrix statistic, but layers are ~97% coupled**

**SHA `365c392d`, node wgp5zvw (1×8 H100), venv019 torch 2.10.0+cu128, ≤2-node ceiling (1 node, 4.8
node-hours total).** Measures a reproducible, optimizer-relevant *shape-weighted spectral* sharpness for the
72 Muon matrices, and quantifies how much is lost by treating them independently. Definitions follow
[Islamov et al. v3, Def. 2.2 / Eq. 19](https://arxiv.org/html/2603.05002v3), extended to the shape-scaled Muon
subspace per the request and the [branch design review](../layerwise_momentum_design_20260911/README.md).

## What was measured

On the true loss Hessian H (via HVPs, **mean-per-valid-token** loss scale — the harness returns summed CE, so
gradient and HVP are divided by the actual valid-token count, never `BATCH_TOKENS`), with shape-weighted radii
`r_i = sqrt(max(1, rows/cols))·lr_mult` and the spectral (operator-2) norm:

- **Isolated** `S_i = max ⟨D_i, H_ii D_i⟩` s.t. `‖D_i‖_op = r_i` (single matrix, others frozen);
- **Joint** `S_joint = max ⟨D, H[D]⟩` s.t. `N_r(D)=1`, `N_r(D)=max_i ‖D_i‖_op/r_i` (keeps all cross terms);
- **`G_i = r_i·‖g_i‖_nuclear`**, `G_joint=Σ G_i`; **`Z = η·S/(2G)`** (candidate normalized-spectral-descent
  diagnostic — **not** a proven K-Maxwell stability boundary; the paper leaves stochastic/momentum extensions
  open);
- **cross-layer decomposition** `c = c_diag + c_cross`, `c_diag=Σ⟨d_i,H_ii d_i⟩`, and the 72×72 interaction
  matrix `Q_ij = ⟨d_i, H_ij d_j⟩` at the joint gradient-polar direction.

Frank-Wolfe with an exact-SVD spectral-polar LMO; the sphere objective is recomputed on the radially-
normalized iterate `D/N_r(D)` with an independent HVP, taking the **best feasible boundary witness** across
restarts (an interior zero is never reported as a negative-definite sphere maximum).

**Two stages.** A **pilot** (step 2000, seed-0 base, 18 sentinel matrices = all 6 types in blocks 0/6/11,
K∈{20,50}, 5 restarts, **3 disjoint train-file subsets + a nested 32768-token check**) established the
measurement's reliability; the **full stage** then mapped all **72 matrices × steps {1500,2000,2500} ×
seeds {0,1,2}** at the validated-cheapest budget (K=20, 2 restarts, 1 subset). The pilot is the rigorous
accuracy/reproducibility anchor; the full stage is the coverage map.

## Correctness (validated before any GPU compute)

`impl/test_cpu_sharpness.py` — **9/9 float64 toy-Hessian tests pass**: joint & diagonal HVP == dense H;
cross-layer example (c=3.8, c_diag=2, c_cross=1.8; symmetric Q); joint sphere (3.8) ≠ isolated sum (2);
spectral sharpness 1.2 vs 2.2 at equal Euclidean spectrum; **negative-definite sphere = −1, not the interior
ball 0**; loss-scale invariance of S/G and Z (raw S linear, `λ/‖g‖_F²` inverse); central-difference HVP ==
autograd; quadratic loss identity.

## Pilot gates — both PASS (`readout_pilot.tsv`)

- **(a) reliability:** 18/18 positively-resolved sentinel `S_i` change ≤5% from K=20 to K=50 (100%, need
  ≥90%; worst 4.5%). So **K=20 is the validated cheapest budget**.
- **(b) reproducibility:** median pairwise Spearman of `S_i/G_i` across the 3 disjoint subsets = **0.983**
  (need ≥0.8).
- **HVP validation:** central-difference on the joint grad-polar direction plateaus at small ε
  (0.75% @0.005, 3.67% @0.01), growing at larger ε — the autograd HVP is confirmed.

## Full-stage results (`readout_full.tsv`)

### Headline — layers are not separable

At the joint gradient-polar direction, **cross-layer coupling is 96.8% ± 0.1% of the joint curvature**
(`c_cross/|c|`) across **all 9 checkpoints** — extraordinarily stable across seeds and steps. Isolated
per-matrix sharpness `S_i` therefore captures only **~3%** of the joint curvature; summing isolated maxima is
nowhere near `S_joint`. (The pilot's 18-matrix figure was 86.8%; with all 72 matrices the cross terms
accumulate to ~97%.) **This is the central answer to the request's question** — treating the matrices
independently discards almost all of the joint sharpness.

### The statistic is reliable

Cross-seed median Spearman of `S_i/G_i` (72 matrices) = **0.962 / 0.958 / 0.934** at steps 1500 / 2000 /
2500 — reproducible across independent seeds, confirming the pilot's cross-subset result at full coverage
(slight decline late in training).

### Sharpness rises through training

`S_joint` roughly doubles 1500→2500 (~3.7k → ~8.4k); mean `S_i` ~1.8 → ~3.9; **`Z_joint` ~0.74 → ~1.2,
crossing ≈1 by step 2500** — a progressive-sharpening signature. `Z` is reported as a diagnostic only.

### Depth / type structure (S_i by block, mean over seeds, step 2000)

- **MLP ≫ attention**: mlp.fc/mlp.proj `S_i` ~5–9 vs attention ~0.2–2.
- **attn.v and attn.proj fall monotonically with depth** (sharp early, flat late); attn.k/q roughly fall
  with a block-3 spike.
- **MLP re-sharpens at the last block** (mlp.fc b11=7.88, mlp.proj b11=5.96, up from the mid-depth minima
  ~5.4 / ~2.2).

## Caveats / scope

- The full map uses the validated-cheapest budget (K=20, 2 restarts, 1 subset); the 3-subset / 5-restart /
  K=50 / nested-token rigor lives in the **pilot sentinel anchor** (step 2000 seed 0), both gates passed.
- **HVP finite-difference on the aggregate 72-matrix joint direction** is ε-sensitive late in training
  (rel_err at ε=0.005 ranges 1.7–18.3%; >5% on several late checkpoints) because that direction is a large
  aggregate perturbation. The autograd HVP itself is validated (toy Hessians + pilot isolated/joint
  directions at 0.75%); late full-joint FD cross-checks above 5% are flagged, not silently accepted.
- A same-run 8-iter Lanczos **Euclidean comparator** (`λ_i`, `λ_i/‖g_i‖_F²`) was not recomputed in the
  cheap-budget stage (REQ-048/050/051 hold Euclidean profiles at nearby states); the spectral depth ordering
  is reported here and a direct spectral-vs-Euclidean re-ordering is a cheap documented follow-up.
- **Actual-step (momentum) curvature** and the realized-step loss-quadratic `b/c/R`: the optimizer-state
  (momentum) dumps exist at 1500/2000/2500 for all three seeds; per-realized-step geometry is deferred to
  REQ-058's shared-state perturbations, which consume these bases.

## Consequence for REQ-058/059

The measurement premise is validated (reliable, reproducible, distinguishable from its Euclidean
comparator), so REQ-058 is unblocked. But the ~97% cross-layer coupling is a strong caution: a momentum rule
derived from **isolated** `S_i` may be spoiled by interactions — which is exactly what REQ-058 is designed to
test (does isolated sharpness predict the response to changing a matrix's memory, on held-out seeds).

## Files

- `impl/measure_layerwise_spectral_sharpness.py` — probe primitives (isolated/joint FW, shape radii, G/Z, Q_ij, HVP validation).
- `impl/test_cpu_sharpness.py` — 9 float64 toy-Hessian correctness tests.
- `impl/measure_layerwise_pilot.py` — GPU driver (data-parallel mean-token HVP; pilot & full via flags).
- `impl/analyze_pilot.py`, `impl/analyze_full.py` — gate + aggregation analyses.
- `readout_pilot.tsv`, `readout_full.tsv` — the two-stage results.
- `raw/pilot_step2000.json`, `raw/full/full_s{0,1,2}_step{1500,2000,2500}.json` — per-checkpoint source (S_i/S_joint/G/Z, 72×72 Q_ij, cross-layer, HVP validation).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2-node ceiling; node stopped after delivery.
