# REQ-038 — per-type activation & backward statistics (the q/k/v probe) + REQ-041 (weight norms) — **DONE, n=1**

**SHA `ebf53cd`, 1 node (8×H100, single-process probe), ≤2-node ceiling.** One forward+backward pass at a regenerated
**fork-1500** (seed 0, s=1.00), recording per Muon matrix via forward/backward hooks: input activation `a` (RMS,
Frobenius, effective rank), output-gradient `d` (RMS, Frobenius, effective rank), attention q·k-logit RMS + per-head
attention entropy, and ‖W‖_F. This directly measures the quantity REQ-035's bands 14–19 converged on analytically.

## Result — the q/k "excess" is mechanically a **gradient deficit**

Per-type means (12 matrices each), fork-1500 (`summary.tsv`):

| type | a_rms | **d_rms (output-grad)** | a_eff_rank | d_eff_rank | ‖W‖_F |
|:-----|------:|------------------------:|-----------:|-----------:|------:|
| attn.q | 1.0036 | **0.00258** | 30.2 | 72.5 | 54.4 |
| attn.k | 1.0036 | **0.00266** | 30.2 | 76.1 | 54.6 |
| attn.v | 1.0036 | **0.00399** | 30.2 | 106.2 | 63.1 |
| attn.proj | 1.1202 | 0.00395 | 58.6 | 389.0 | 58.6 |
| mlp.fc | 0.6087 | 0.00388 | 20.0 | 901.6 | 127.3 |
| mlp.proj | 1.6491 | 0.00345 | 161.0 | 359.3 | 65.1 |

**q,k output-gradient (d_rms) = 0.00262 vs v = 0.00399 → ratio 0.66.** q and k receive **~2/3 of v's output-gradient
magnitude**, with **identical input activation** (all a_rms = 1.0036 — q/k/v share the same normed-residual input). So
the q↔v and k↔v difference is **purely in the backward signal, not the forward** — a gradient deficit, exactly as the
analysis (bands 14–19) predicted. q,k gradients are also **lower-rank** (eff_rank 72/76 vs v's 106). q and k are
near-identical to each other (d_rms 0.00258 vs 0.00266), consistent with band 18's "q and k are interchangeable".
Attention: mean prob-entropy 1.417 nats, mean q·k-logit RMS 27.2 across 12 blocks.

## Caveats
- **n=1** (one fork-1500, seed 0). The q,k/v deficit (0.66, a 34% effect) is large and unlikely to be noise, but a
  same-probe pass on Arm A's other seeds would confirm robustness.
- `d_rms` is from a **summed-loss** backward (the model returns a token-summed loss), so its absolute scale is arbitrary;
  all **ratios across types** (the q/k/v comparison) are scale-invariant and unaffected.

## REQ-041 (folded in) — weight norms
`weight_norms.tsv` records ‖W‖_F per Muon matrix at fork-1500 in the REQ-023 shape (`fork_seed, arm, step, name,
weight_frob`), as requested. (The prior curvature runs' checkpoints were cleaned by re-bootstraps, so this is a fresh
fork-1500 rather than a ride-along; future curvature runs can call `measure_activation_backward.py` alongside the
Lanczos probe to record norms + a/d together.)

## Files
- `summary.tsv` — per-type a/d/rank/‖W‖ + the q/k-deficit finding + attention stats.
- `weight_norms.tsv` — REQ-041 per-matrix ‖W‖_F. `measure_activation_backward.py` — the probe (reusable).
- `req038_actback_fork1500_seed0.json` — the full per-matrix + per-block raw record.

No secrets/weights/tensors committed. Ran under the ≤2 ceiling.
