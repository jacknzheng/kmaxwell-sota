# REQ-033 — does the ANNEALED K-Maxwell kernel survive a change of batch size? **COMPLETE**

**SHA `365c392d` (same as REQ-026/027/028/029), node `wox8gkw` (8×H100), venv019 torch 2.10.0+cu128.**
12 arms = batch {0.25×, 0.5×, 2×} × kernel {A, B, C, D}, seed 0, n=1/cell, 2250-step continuations
from a shared **step-1000** state. **All 12 arms finite, 0 errors.** Node released after collection.

## The curve — within-batch benefit vs the frozen bi-Maxwell reference

`benefit = final_val(kernel) − final_val(arm A single-EMA)`, **within batch size**, @ step 3250
(negative = the kernel beats single-EMA μ0.95). Absolute loss is not comparable across batch sizes
(2× sees 8× the tokens of 0.25×), so only the within-batch contrast is read.

| batch | batch_tokens | **B − A** bimaxwell | **C − A** anneal *shipped* | **D − A** anneal *rescaled* | bi-Maxwell (REQ-029, ctx) |
|:------|-------------:|--------------------:|---------------------------:|----------------------------:|:--------------------------|
| 0.25× | 131072 | −0.00492 | +0.00063 | **+0.10553** | (1×: −0.01063) |
| 0.5×  | 262144 | −0.00380 | −0.00368 | +0.01003 | (4×: −0.00438) |
| 2×    | 1048576 | −0.00084 | **−0.00513** | −0.00209 | (8×: −0.00233 / 16×: ~0) |

Raw finals: A/B/C/D = 0.25× 3.45397/3.44905/3.45460/3.55950 · 0.5× 3.35723/3.35343/3.35355/3.36726 ·
2× 3.21181/3.21097/3.20668/3.20972. Base val@1000 = **3.65169**.

### Shape (the deliverable — noise floor ~2e-4 from REQ-027; read |Δ| < ~5e-4 as noise)

- **B (bimaxwell)** — benefit **decays toward zero as batch grows** (−0.0049 → −0.0038 → −0.0008). Same
  denoiser shape as REQ-029's frozen bi-Maxwell curve: the two-rate edge is batch-specific and absorbed by
  large-batch gradient averaging. Confirms the fresh-B arm reproduces the known curve on this fork/node.
- **C (annealed, shipped ages)** — the **opposite** trend of B. At 0.25× C ≈ A (+0.0006, at noise); the
  single-EMA-relative benefit **grows** with batch (−0.0037 @0.5×, −0.0051 @2×). So within 0.25×–2× the
  shipped annealed kernel does **not** reproduce bi-Maxwell's decay-to-zero — its benefit is flat-to-growing,
  smallest at the *smallest* batch. The 1×-tuned anneal gives **no** small-batch denoising and helps most at ≥0.5×.
  (This is neither of the request's two clean scenarios — not "C decays like bi-Maxwell", not "C holds flat where
  bi-Maxwell decayed" — it *anti*-decays within this window.)
- **D (annealed, ages × (1×/batch))** — the "obvious repair" **backfires at small batch.** D is dramatically
  worse than A at 0.25× (**+0.1055**) and 0.5× (+0.0100), only reaching ~A at 2× (−0.0021). Lengthening the
  memory to compensate small-batch noise (mean age 58→26 becomes 232→104 at 0.25×) **cripples** the kernel; it
  does not restore or improve on the shipped ages. Age-rescaling as a token-reparameterisation is **refuted as a
  repair** here — the request's "C decays but D holds → reparameterise in tokens" scenario did not occur; D holds
  only at 2× where nothing was broken.

**One line:** within 0.25×–2×, bi-Maxwell (B) denoises and decays like REQ-029; the shipped annealed kernel (C)
does the opposite (benefit ~0 at 0.25×, best at 2×); rescaling its ages to the batch (D) makes it far worse at
small batch, not better.

## ⚠️ torch.compile `microbatch_sequences < 64` NaN bug (found, isolated, worked around)

The **first** 12-arm pass came back **all-NaN** — every arm, including the plain-muon control A, diverged to NaN
within **10 steps** of the fork (val@1010 = nan; val@1000 = the finite base 3.65169). It was isolated with 6
controls:

| control | result | rules out |
|:--|:--|:--|
| 1× fork@1000 (compiled) | descends 3.652→3.612 | base state / fork mechanism |
| 2× fork@1000 (mbs=64, compiled) | descends 3.652→3.538 | fork / large batch |
| 0.25× at 0.25× LR | still NaN in 10 steps | LR magnitude |
| 0.25× frozen (LR≈0) | still NaN | optimizer step & data → the **gradient** is NaN |
| **1× batch @ mbs=16 and @ mbs=32** | **NaN** | batch_tokens (identical to stable base; only mbs changed) |
| 0.25× with `TORCHDYNAMO_DISABLE=1` | **finite** (val ~3.82) | → it is a **compile** defect |

**Root cause: a torch.compile shape-specialization defect at `microbatch_sequences < 64`.** mbs=64 (1×, 2×) is
stable; mbs=16 and mbs=32 produce NaN gradients regardless of `batch_tokens`; eager computes them correctly.
Because 0.25×/0.5× *require* mbs ≤ 16/32 (per-GPU-seq cap — 64 is illegal there), the **8 small-batch arms were
run eager** (`TORCHDYNAMO_DISABLE=1`, identical math, ~25× slower but the batches are tiny so wall-clock was fine)
and the **4 2× arms compiled** (mbs=64). The readout is a **within-batch** contrast, so the eager/compiled split
is orthogonal to every reported number. `summary.tsv` records `engine` per row. *(Follow-up for the maintainer:
the compile path at mbs<64 is a latent correctness bug for anyone training with gradient-microbatching below 64
sequences — worth a minimal repro + fix upstream; eager is the trustworthy path meanwhile.)*

## Gates (all passed before the full arms)

1. **Tests green** at the pin (`test_registry_locks`, `test_newton_muon`).
2. **Usable-batch budget** (REQ-029 metric, real 100M-token shard headers): 2× tightest at margin **+100**
   (30 shards × 95 = 2850 ≥ needed 2750) — your 29-of-30 binding case. All 12 ≥ needed.
3. **microbatch_sequences divides batch_tokens/(8·1024)**: 16|16, 32|32, 64|128 — all OK.
4. **Launch-time resolution** of all 13 configs (optimizers/hooks/regexes bind).
5. Per-arm smoke (stop@1150 so the step-1125 coarse val fires) — finite before every full arm.

## Method / faithfulness

- Shared base = the Track-3 `bimaxwell_muon switch_step:1000` run, which behaves as **plain Muon through step
  1000**; dumping training state at step 1000 (a `pre_optimizer` hook, *before* the switch fires) yields a
  single-momentum state with **no K-buffers**. Arms B/C/D lazy-init their streams from that momentum at
  `switch_step:1000` exactly as PR #357 does; A continues as muon. Every arm resumes `eos_shared_state/step 1000`
  and token-aligns its data skip (`1000·524288/batch_tokens` = 4000/2000/500) to the same ~0.524B-token position.
- Arms C/D: `annealed_weights_muon`, `switch_step:1000`, `anneal_end_step:3250` (= your "anneal_steps 2250"),
  `warm_streams_before_switch:False`; start/end weights scale-invariant (identical for C and every D), only
  `decays` differ per your spec.
- lr 0.025 / wd 0.05 / mu 0.95 fixed across all arms; `cool_down cooldown_frac:0.7`; no `fixed_eta_after`;
  checkpoint@3250 only; dense val [3000,3250]/10 + coarse /125.

## Known limitations (not papered over)

- **n=1 per cell.** Cross-seed spread ~2e-4 (REQ-027). The 0.5× C−A (−0.0037) and B−A (−0.0038) sit just above the
  ~5e-4 read-as-noise band; D's small-batch penalty (+0.106, +0.010) and the 2× C−A (−0.0051) are well above it and
  robust. If the C-vs-B ordering at 0.5× is load-bearing for a claim, replicate those two cells.
- **No 1× cell** (as specified) — the curve is read as a trend across 0.25×/0.5×/2× only; the n=8 1× runs in
  `logs/kmaxwell/{bimaxwell339_n8,ablation_anneal_n8}` are out-of-band context, never differenced here.
- **Update-noise floor (recorded per your note):** age-scaling (arm D) does not restore constant update noise;
  the Nesterov term `h0=(1−mu)+mu·Σw(1−β)` has a floor `(1−mu)²=0.0025` no memory-lengthening removes. The empirics
  are stronger than "removes most of it": at small batch D is *worse*, so the residual noise is not the whole story
  — over-long memory (bias) dominates. True noise-matching (co-scaling `mu`) remains the untested lever.

## Files

- `summary.tsv` — finals + within-batch benefit + REQ-029 context curve + the compile-NaN note.
- `readout.tsv` — the closing benefit table + shape.
- `val_trajectories.txt` — every arm's raw per-step val_loss (+ base @1000).
- `manifest.tsv` / `make_req033_configs.py` — the 13-config design + generator.
- `configs/` — the 13 YAMLs (base + 12 arms). `logs/` — 12 full logs (gz) + base (gz) + 12 smoke logs.

Checkpoints (step 3250 per arm), `eos_shared_state`, FineWeb data, env: on-box, **not** committed. Node released.
