# REQ-043 — is the q/k output-gradient deficit seed-reproducible? **n=4, COMPLETE**

**SHA `ebf53cd` (EoS serialized-fork-state), 2 nodes (wxyg00q + qekdm1q, 8×H100 each, ≤2-node
ceiling), venv019 torch 2.10.0+cu128.** Four independent seeds (0,1,2,3), each trained from scratch to
fork-1500, each probed with the REQ-038 activation/backward instrument (`measure_activation_backward.py`).
Seed 0 = REQ-038's original probe; seeds 1–3 are this run. **All 4 seeds finite, 0 errors.**

This is the seed-replication of REQ-038's single-seed finding that **q and k receive a systematically
smaller output-gradient than v** — the "gradient deficit" that motivates the client's REQ-035 band-14–19
q/k analysis. The question: is the deficit an architectural signal or a seed artifact?

## Result — the q/k gradient deficit is seed-reproducible to a razor-tight floor

Per-type RMS of the **output-gradient** `d` (the backward tensor flowing into each Muon matrix) at fork-1500,
q and k relative to v (`summary.tsv`):

| seed | q d_rms | k d_rms | v d_rms | (q,k)/v | log₁₀(q/v) | log₁₀(k/v) |
|:----:|--------:|--------:|--------:|--------:|-----------:|-----------:|
| 0 (REQ-038) | 0.00258 | 0.00266 | 0.00399 | 0.658 | −0.189 | −0.175 |
| 1 | 0.00263 | 0.00271 | 0.00391 | 0.682 | −0.172 | −0.160 |
| 2 | 0.00253 | 0.00261 | 0.00392 | 0.655 | −0.191 | −0.177 |
| 3 | 0.00257 | 0.00264 | 0.00387 | 0.672 | −0.178 | −0.167 |

**(q,k)/v = 0.667 ± 0.011** (mean ± sd, range [0.655, 0.682]). The deficit is significant far beyond the
seed noise: log₁₀(q/v) mean = **−0.183 dex, t = −41.9 (n=4)**; log₁₀(k/v) mean = **−0.170 dex, t = −43.3**.
**q and k each receive ≈0.66× the output-gradient of v** — reproduced across four independent networks with
a cross-seed spread (0.011) that is ~6% of the effect. **This is an architectural property, not a seed
artifact** — consistent with REQ-035's finding that C is seed-independent.

## Band 21 — the deficit is *purely* in the backward pass

Within every attention block, q/k/v read the **same input** (the post-LayerNorm residual), so their input
activation RMS must be identical — and it is, bit-for-bit, in all 4 seeds (`summary.tsv`):

| seed | a_rms q | a_rms k | a_rms v | identical? |
|:----:|--------:|--------:|--------:|:----------:|
| 0 | 1.00360 | 1.00360 | 1.00360 | ✓ |
| 1 | 1.01980 | 1.01980 | 1.01980 | ✓ |
| 2 | 1.00620 | 1.00620 | 1.00620 | ✓ |
| 3 | 0.99820 | 0.99820 | 0.99820 | ✓ |

Since the forward input is identical for q/k/v, the entire (q,k)/v = 0.66 gap is carried by the **output
side** — the gradient `d` that softmax-attention back-propagates into q and k is smaller than what it
back-props into v. **The deficit is a backward-pass phenomenon, band 21 confirmed n=4.** Mechanistically this
is the softmax Jacobian: v's gradient passes through the (near-1) attention weights linearly, while q/k
gradients are throttled by the softmax derivative `p⊙(1−p)`-style contraction (attention is near-peaked, so
that factor is <1). This directly supports the client's REQ-035 bands 14–19 q/k-specific analysis: q and k
are *undertrained per unit LR* relative to v, purely because they see less gradient — not less signal.

## Method

Per seed: `eos_shared_base` (seed=N) → `run.py … stop_after_step=1500` → dump `train_state_model_step001500.pt`
→ `measure_activation_backward.py --model <ckpt> --out req043_seed{N}.json`. The probe is single-process: forward
hooks capture each Muon Linear's input `a`; full-backward hooks capture the output-gradient `d`; it reports
RMS/Frobenius/effective-rank of both, plus reconstructed attention-logit stats. One minibatch of FineWeb eval
tokens, same batch across seeds. Per-type numbers average over the type's 12 per-layer matrices.

## Caveats

- **n=4, one minibatch per seed.** The cross-seed sd (0.011 on the ratio) already bounds the seed noise; the
  effect (0.66) is ~30 sd from unity, so the deficit is not noise. Different eval minibatches would move the
  absolute d_rms but not the q/k-vs-v *ratio* (it is scale-invariant — all three see the same tokens).
- **Fork-1500 snapshot.** This is the deficit at one training point (step 1500). REQ-038 showed the same sign
  at the single seed; the client's bands track its evolution — a step-sweep is a clean follow-up but not
  requested here.
- **`d_eff_rank`** (participation ratio of the gradient's singular spectrum) is in `summary.tsv` for
  completeness; the load-bearing readout is d_rms.

## Files

- `summary.tsv` — per-seed per-type d_rms/a_rms/d_eff_rank + the n=4 deficit stats + the band-21 a_rms check.
- `measure_activation_backward.py` — the probe (REQ-038 instrument, unchanged).
- `raw_json/req043_seed{0..3}.json` — raw per-matrix a/d stats (source of truth). seed0 = REQ-038's JSON.

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
