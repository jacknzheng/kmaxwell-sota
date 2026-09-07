# REQ-050 — curvature at initialisation and early training — **LEARNED-EARLY (n=4)**

**SHA `ebf53cd`, 2 nodes (wdkrxgq + q4zk6y3, 8×H100 each, ≤2-node ceiling), venv019 torch 2.10.0+cu128.**
Four independent seeds, each trained from step 0 to 1500 (checkpoint every 125), then the curvature
probe (`measure_per_matrix_curvature.py`, patched to also emit `trace_est`/`trace_sq_est` Hutchinson
moments — REQ-048) at steps **0, 125, 250, 500, 1000, 1500**, **3 probe repeats per step**
(independent Hutchinson seeds via `--probe_rep`, each committed separately). 6 steps × 3 reps × 4 seeds
= 72 merged profiles. Measured cost ≈ **450–630 s per probe**.

## Result — the depth-curvature bowl is absent at init and learned within ~125 steps

`readout.tsv`, judged **within each seed** (band-55 criteria):

| step | cubic R² (bowl fit) | argmin | corr with step-1500 profile |
|-----:|:--------------------|:------:|:----------------------------|
| **0** | **— (λ ≡ 0, all 72 matrices, all seeds)** | — | — |
| 125 | 0.91 / 0.95 / 0.97 / 0.99 | 6–9 (mid-depth) | −0.12 … +0.89 |
| 250 | 0.91 – 0.96 | 6–9 | +0.18 … +0.93 |
| 500 | 0.78 – 0.92 | 7–9 | +0.54 … +0.83 |
| 1000 | 0.69 – 0.87 | 7–9 | **+0.70 … +0.95** |
| 1500 | 0.59 – 0.86 | (ref) | +1.00 |

**Verdict: LEARNED-EARLY in all four seeds.** The registered INHERITED criterion (profile present at
init, cubic R² ≥ 0.70, corr ≥ 0.70 with late) is **decisively false** — there is no profile at init at
all. The LEARNED-EARLY criterion (step 0 has R² < 0.30 / no minimum, profile present by 500–1500) holds
4/4.

## Why step 0 is exactly zero — the zero-initialised output projection

At step 0 the probe returns `top_eigenvalue = trace_est = gradient_block_norm = 0.0` for **every** Muon
matrix, in every seed. This is not a measurement failure — the step-0 checkpoint has real, correctly
initialised block weights (e.g. `blocks.0.attn.q.weight` ‖·‖ = 15.9). The cause is mechanistic:

- **`proj.weight` (the output/unembed projection) is zero-initialised** — ‖proj.weight‖ = 0.0, max 0.0
  (confirmed from `model_step000000.pt`), the standard modded-nanogpt output-head init.
- So at init the logits are identically zero → uniform softmax → **loss = ln(50304) = 10.826** (exactly
  the observed step-0 val loss).
- The backward pass multiplies by `proj.weightᵀ = 0`, so the gradient reaching **every** block matrix is
  identically zero — and therefore so is the Hessian curvature along any block-matrix direction.

**The depth-curvature profile is thus structurally undefined at initialisation**, not merely weak: the
loss surface is flat in every Muon-matrix direction until the output projection moves off zero after the
first optimiser step. This is a clean answer to REQ-050's question — the profile *cannot* be inherited
from init; it is created by training.

## What emerges, and when

By **step 125** the bowl is already strong and mid-depth (cubic R² 0.91–0.99, argmin in blocks 6–9), so
the depth structure forms within the first ~125 optimiser steps. Its *specific* per-layer shape then
converges toward the step-1500 profile progressively: cross-step correlation with 1500 rises from a
seed-dependent +(−0.12…0.89) at step 125 to **+0.70…+0.95 by step 1000**. So a bowl-shaped profile
appears almost immediately, but the exact profile that persists to 1500 stabilises around step 500–1000.

## Caveats / method

- **Within-seed judgement** (band-55): seeds test robustness to init; cross-seed sign agreement is not
  treated as independent structural evidence. All four seeds independently give LEARNED-EARLY.
- λ = HVP/Lanczos `top_eigenvalue` (iters 8); profile = mean over 3 Hutchinson repeats of the per-layer
  mean-over-6-types of log λ. `trace_est`/`trace_sq_est` committed per matrix (for the concentration
  replication REQ-050/051 asked for), independently sampled — not Lanczos-derived.
- step-0 λ ≡ 0 makes cubic R²/argmin/corr undefined there; reported as ZERO(init), not forced.

## Files

- `readout.tsv` — per-seed per-step cubic R² / argmin / corr-with-1500 + the band-55 verdict + the zero-init note.
- `analyze_req050.py` — the analysis, reproducible from the raw JSONs.
- `req050_run.sh` — the driver (train 0→1500 checkpoint every 125; probe 6 steps × 3 reps).
- `raw_curvature_json/req050_s{0..3}_step{000000..001500}_rep{0,1,2}.json` — 72 merged per-matrix profiles (source of truth; `top_eigenvalue`, `trace_est`, `trace_sq_est`, `gradient_block_norm`, participation_ratio).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
