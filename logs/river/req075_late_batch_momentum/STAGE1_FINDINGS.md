# REQ-075 Stage 1 — frozen-state gradient-noise (DONE)

Frozen common state = REQ-073 **B-mom fork @ step 2500** (x_2500, pre-update). 64 independently sampled
B-sized probe batches from a probe region (`fineweb_train_00010*.bin`, chunks 100–103) **disjoint** from the
baseline (chunks ~1–14) and the Stage-2/3 continuation region. Each probe reproduces the training gradient
exactly (per-rank 64-seq micro-batch split into 8-seq sub-chunks for memory, token-SUM CE backward,
`all_reduce(SUM)` over 8 ranks, ÷B for mean-per-token units). No optimizer, no parameter change. Driver
`impl/req075_stage1_noise.py`; raw per-probe gradients off-Git on durable FS
(`/root/.cache/user_artifacts/req075/stage1/raw`); summary `stage1/noise.json` committed.

**Predeclared low-noise criterion (fixed before labelling): NSR = RMS gradient noise / bias-corrected signal
norm < 0.10, bootstrap 16–84% interval reported.**

## Results (block-5 Q, K, MLP-proj)

| matrix | signal‖ḡ‖ (bias-corr) | NSR_B | NSR_4B | NSR_16B [boot 16–84] | 16B dir-agreement cos | low-noise 16B? |
|---|---|---|---|---|---|---|
| attn.q | 7.40e-4 | 1.68 | 0.82 | **0.368** [0.27–0.35] | 0.848 ± 0.017 | **No** |
| attn.k | 6.34e-4 | 1.85 | 0.90 | **0.401** [0.29–0.37] | 0.824 ± 0.011 | **No** |
| mlp.proj | 2.52e-3 | 1.65 | 0.80 | **0.359** [0.26–0.33] | 0.853 ± 0.003 | **No** |

### Findings
1. **At B, the late-training gradient is noise-dominated** (NSR_B ≈ 1.7–1.8: the single-batch sampling noise
   *exceeds* the signal norm). Momentum's denoising role therefore has large headroom to help at B.
2. **Noise scales ≈ 1/√B, as expected** — measured vs ideal: Q 4B 6.07e-4 vs 6.21e-4, 16B 2.72e-4 vs
   3.11e-4; K and MLP the same pattern. 16B noise sits slightly *below* the 1/√B line (noise falls a touch
   faster than ideal), within pilot uncertainty. **No anomalous departure from inverse-batch scaling.**
3. **16B is NOT a low-noise (≈noise-free) state.** NSR_16B ≈ 0.36–0.40 ≫ 0.10 for all three matrices, with
   bootstrap upper bounds ≤ 0.37. Reaching NSR < 0.10 would need batch ≈ 16B·(0.37/0.10)² ≈ **~220B
   tokens/update** — far beyond the registered 16B. **The 16B arm must be read as a *reduced-noise* regime,
   not an infinite-batch / noise-free limit** (exactly the caution the request raised).
4. **Large-batch directions agree only moderately** (16B pairwise cosine ≈ 0.82–0.85), consistent with the
   residual NSR ≈ 0.37 still rotating the direction by ~30°. Selected-matrix agreement does **not** establish
   whole-model agreement (not claimed).

### Caveats / limitations
- 4 nested 16B groups are a **pilot** variance estimate; bootstrap intervals reported, not a precise variance.
  Optional 128-probe refinement would tighten the estimate but cannot move the verdict (upper bound 0.37 ≪
  the 0.10 line is nowhere near crossing).
- Bias in the signal norm removed via E‖mean‖² = ‖true‖² + tr(Cov)/n (correction small here: 7.56e-4→7.40e-4).
- Sampling-noise-only diagnostic: the model has no dropout and parameters are fixed, so the sole stochastic
  source is data order (documented); no other stochasticity to separate.

### Consequence for Stages 2–3
Because 16B is still meaningfully noisy, any late-training oscillation change under 16B is a *partial* noise
reduction, not elimination — so a "smoother 16B curve" cannot by itself be attributed to removing noise, and
momentum's benefit at B (where NSR≈1.7) is expected to be large. Stage 2/3 measure the actual mechanism and
useful-descent, with this noise floor as context.

Resource use: 1×8-H100, ~33 s compute for 64 probes + stats; negligible vs the forecast.
