# REQ-068 — gradient history + whole-model river geometry

**Scope amendment (Jack, 2026-10-02):** REQ-068 was narrowed mid-run to three 64-step windows
(500-563, 1500-1563, 2500-2563) requiring TWO streams — raw gradient g_t AND actual displacement
delta_t=theta_(t+1)-theta_t — plus a 64x64 temporal K/C Gram/cosine analysis per window and probe
states at offsets 32/48/63. **Status: the gradient stream below is DONE as a superset** (full
3250-step history, covers all three windows + forks; preserved as historical data per the
amendment). **Remaining:** the displacement stream delta_t for the 192 window steps + the K/C Gram
analysis + window probe states — to be captured by forking from the retained theta_500/1500/2500.

---

## Full gradient history (superset, DELIVERED)

**Harness `365c392d`, node wngkx23 (1×8 H100, venv019 torch 2.10.0+cu128). Raw tensors stay off Git on the
durable shared FS; this dir holds the manifest, schema, reader, analysis, and README.** Complete capture of
the gradient tensor for **every trainable parameter at every optimizer step** of a full 3250-step baseline.

## Baseline (pinned)
Standard **Muon (mu=0.95)** on the 72 hidden matrices + auxiliary **AdamW** (embed lr0.7 / head lr0.004 /
scalars lr0.015, all wd0.001; blocks lr0.025 wd0.05), GPT vocab 50304 / 12 layers / dim 768, fineweb10B,
batch 524288 tokens, microbatch 64, 3250 updates, original cooldown LR schedule. Config:
`impl/make_req068_config.py`. Seed 0. Final val_loss **3.278**.

## Capture (conventions)
- **Point:** `capture_full_gradients` pre_optimizer hook — after microbatch accumulation + `all_reduce(SUM)`,
  **before** `apply_updates()` (Muon's `grad.lerp_` mutates `.grad` in place). Gradients are **replicated**
  across ranks (all_reduce SUM), so rank-0 capture is complete; no shard reconstruction needed (documented).
- **Values:** raw token-**SUM** gradient, **native precision, lossless** (census: 172/173 params fp32, 1 bf16
  — grads accumulate in fp32 even though matrix params are bf16; this is why actual size 1.86 TB > the
  1.06 TB bf16-param forecast). `tokens_per_update=524288` recorded for mean-per-token conversion.
- **One observation = one optimizer update** (global index 0..3249). No fabricated zeros (None→explicit flag;
  0 inactive at step 0).

## Acceptance / coverage (`manifest.json`, `analysis_summary.json`)
| metric | value |
|:--|:--|
| steps captured | **3250** (coverage **0..3249 contiguous**, no gaps/dupes) |
| chunks | 65 × 50 steps, SHA-256 each; spot re-hash (chunks 0/32/64) **OK** |
| total size | **1,859,672,450,945 B ≈ 1.86 TB** (lossless native) |
| params/step | 173 (162,354,816 numel); fork checkpoints | 27 files @ steps 500/1500/2500 |
| wall time | capture run ~2.34 h (step_avg ~2.59 s incl. capture; baseline ~0.16 s/step → capture is the dominant cost: ~170 synchronous D2H fp32 copies/step + 16 GB chunk flushes) |
| storage | durable shared FS `csi-storage-sfs` (137 TB, 54 TB free, 622 MB/s write) |

**Smoke-tested** before the full run (`impl/run_req068_smoke.sh`): reader round-trip (reconstruct, native
dtype), logging ON-vs-OFF training parity at the CUDA-nondeterminism floor (steps 0,5 bit-identical; capture
is read-only), fork dumps. CPU unit tests (`impl/test_req068_capture.py`, 6/6): round-trip, **no-aliasing**
(survives in-place Muon mutation), None flags, chunking, manifest verify, corruption/gap detection.

## Retrieval (off-Git raw artifacts)
- Gradient chunks + manifest + schema: `/root/.cache/user_artifacts/req068_s0/grads/` on `csi-storage-sfs`
  (durable; survives node stop). Fork states: `/root/.cache/user_artifacts/req068_s0/forks/`.
- Reconstruct any step / stream per-matrix series: `impl/req068_capture.py::GradientHistoryReader`
  (`.grad(step,name)`, `.full_step(step)`, `.series(name)`, `.verify(expected_steps)`).

## Initial descriptive finding — **clear period-two oscillation on a slow drift**
Mean-per-token gradient direction, signed cosine at lags (global per matrix type):

| type | lag1 | lag2 | lag4 | lag8 | lag16 | lag32 |
|:--|--:|--:|--:|--:|--:|--:|
| attn.q | −0.420 | +0.323 | +0.157 | +0.027 | +0.004 | −0.017 |
| attn.v | **−0.586** | +0.368 | +0.140 | +0.003 | −0.001 | −0.003 |
| mlp.fc | −0.409 | +0.300 | +0.128 | +0.021 | +0.004 | −0.009 |
| mlp.proj | −0.502 | +0.331 | +0.138 | +0.030 | +0.006 | −0.009 |
| embed | −0.392 | +0.255 | +0.077 | −0.011 | −0.003 | −0.001 |
| head | −0.238 | +0.182 | +0.055 | +0.003 | +0.003 | −0.003 |

**lag-1 strongly negative, lag-2 positive, lag-4 weakly positive, →0 by lag-8/16**: consecutive gradients
anti-align while every-other-step gradients re-align — a damped period-two bounce superimposed on a
slowly-varying component. Strongest in attn.v and mlp.proj, weakest in the output head. Global gradient norm
decays 0.45→0.11→0.07 (peak 2.12 early). **This is a candidate slow "river" signal under the oscillation,
not proof of a valley floor** — curvature probes at the retained fork states belong to REQ-070.

## Files
`manifest.json`, `param_schema.json`, `analysis_summary.json` (committed); `impl/req068_capture.py`
(writer+reader, 6/6 CPU tests), `impl/apply_req068_capture.py` (hook patch), `impl/make_req068_config.py`,
`impl/run_req068_smoke.sh`, `impl/req068_verify_analyze.py`. Raw 1.86 TB tensors + 27 forks: off-Git on the
durable volume (paths above). No secrets/weights/tensors committed.

---

## Amended-scope results: whole-model temporal geometry (3 windows × 2 streams) — **DELIVERED**

Fork-captured from the retained θ_500/1500/2500 (node q8rn683): 64 updates per window, **both streams**
(raw gradient g_t + actual displacement δ_t=θ_{t+1}−θ_t fp32) + probe states at offsets 32/48/63 (27
files/window). 64×64 K/C Gram per window/stream (`analysis/win{500,1500,2500}.json`; `req068_window_analyze.py`).

| window | stream | lag1 | lag2 | lag4 | lag8 | top-4 eig frac | OOS r1/2/4/8 | past-mean | dominant family |
|:--|:--|--:|--:|--:|--:|:--|:--|--:|:--|
| 500 | gradient | **−0.723** | +0.516 | +0.159 | +0.015 | .13/.07/.07/.06 | .03/.03/.04/.07 | .004 | attn.v .40, mlp.proj .32 |
| 500 | displacement | **+0.793** | +0.634 | +0.394 | +0.130 | .12/.12/.10/.09 | .00/.01/.02/.04 | .004 | **embed 1.0** |
| 1500 | gradient | −0.630 | +0.449 | +0.203 | −0.028 | .15/.08/.07/.07 | .06/.07/.10/.13 | .010 | mlp.proj .40, attn.v .33 |
| 1500 | displacement | +0.785 | +0.627 | +0.396 | +0.143 | .12/.12/.10/.09 | .00/.01/.03/.04 | .004 | **embed 1.0** |
| 2500 | gradient | −0.249 | +0.075 | +0.038 | −0.009 | .13/.06/.05/.04 | .00/.01/.04/.07 | .001 | mlp.proj .45, attn.v .28 |
| 2500 | displacement | +0.745 | +0.586 | +0.369 | +0.143 | .12/.11/.10/.08 | .00/.01/.02/.04 | .004 | **embed 1.0** |

**1. The optimizer converts an oscillating gradient into persistent motion (the river).** Raw gradients
**anti-align** step-to-step (lag1 −0.72/−0.63/−0.25; period-two, +lag2), weakening markedly over training.
Actual displacements **align** step-to-step (lag1 **+0.75 to +0.79**, smooth positive decay through lag8,
no alternation) at every window. So the net parameter motion flows smoothly along a direction while the
gradient bounces across it — direct evidence for a candidate local slow component, **separable from the
fast alternating gradient**. This answers REQ-068's question in the affirmative for the *existence* of a
slow component.

**2. Where the simple "river" picture fails (honest limits).** (a) **Whole-model Euclidean displacement is
entirely embedding-dominated** (family energy share = 1.0), because AdamW lr 0.7 on `embed` dwarfs Muon
lr 0.025 on the hidden matrices in raw δ norm — exactly the "keep embedding contributions visible" caveat.
The hidden-matrix river needs the **separately-labeled layer-reweighted** view (follow-up). (b) The slow
component is **not low-rank-constant over a 64-step window**: the Gram spectrum is flat (top-4 eigen fracs
~0.12 each, no dominant mode) and **out-of-sample low-rank capture is low** (δ r8 ≈ 0.04; even past-mean/
exp-avg baselines ≈ 0.00–0.02). High adjacent-step persistence (lag1 0.79) but a direction that drifts
enough over 32 steps that an early rank-≤8 fit does not predict the later half — a *bending* river, not a
straight line or a fixed valley floor. (c) Gradient geometry is attn.v/mlp.proj-dominated, distinct from
the embedding-dominated displacement.

**Verdict:** evidence **supports a candidate local slowly-changing component** (strong adjacent-step
displacement persistence vs anti-aligned gradients), but **not** a low-dimensional constant river direction
over these windows, and whole-model Euclidean geometry is dominated by the embedding. Independent-branch
(REQ-069) and loss/curvature tests (REQ-070) are required before any "valley floor / useful descent" claim.

**Artifacts (off-Git on durable FS):** `req068_s0/win{500,1500,2500}/{grads,disp,probes}` (both streams +
27 probe states/window), fork states `req068_s0/forks`. Committed: `analysis/win*.json` (K/C 64×64 + all
derived), `impl/{apply_req068_displacement,make_req068_window_config,req068_gram,req068_window_analyze}.py`
(+ CPU tests). Node q8rn683 stopped after delivery.
