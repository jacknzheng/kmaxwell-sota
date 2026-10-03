# REQ-072 — six optimizer ablations: Q/K-free attn.proj spectra through AdamW/SGD/Muon ×momentum

**Status: RUNNING (planning/build) 2026-10-03.** Harness `365c392d`. Independent descriptive optimizer
study (does NOT depend on the river outcome); reuses REQ-068 capture conventions.

## Selected matrices (frozen before outcomes)
`blocks.0.attn.proj.weight`, `blocks.5.attn.proj.weight`, `blocks.11.attn.proj.weight` (attention
output-projection at first/middle/last block; three depths, one type). 768×768 each. Verify names against
the pinned harness at build time.

## Six arms (3250 updates each, identical init/data/eval, one paired seed)
| arm | optimizer (hidden matrices) | first-moment momentum | registry |
|:--|:--|:--|:--|
| adamw-mom | AdamW β1=0.9 β2=0.95 | on | CaptureAdamW |
| adamw-nomom | AdamW β1=0 β2=0.95 | off (keep √v scale) | CaptureAdamW |
| sgd-mom | SGD momentum 0.9, dampening 0, Nesterov off | on | SgdBlocks(momentum=0.9) |
| sgd-nomom | SGD momentum 0 | off | SgdBlocks(momentum=0) |
| muon-mom | standard Muon mu=0.95 | on | Muon(mu=0.95) |
| muon-nomom | same Muon conditioning, mu=0 | off | Muon(mu=0) |
Decoupled weight decay in all arms (SGD coupled wd=0, apply declared decay separately — labeled). AdamW/SGD
over the FULL model; Muon on hidden matrices + aux AdamW on embed/head/scalars (aux fixed, labeled as AdamW).
LR scales per optimizer frozen after a short stability pilot; not auto-scaled.

## Three streams per selected matrix, every step (REQ-068 conventions)
1. raw g_t (after accum/sync/unscale, before clip/momentum/mutation) — restricted `capture_full_gradients`.
2. **conditioned direction u_t** (after momentum/bias-correction/adaptive-scale/Muon-orthogonalization +
   shape scaling, BEFORE lr × and decoupled decay) — NEW per-optimizer capture: Muon stashes
   `compute_polar_input` output; SgdBlocks stashes its `update`; **CaptureAdamW** (transparent non-fused
   AdamW exposing m̂/(√v̂+eps)) stashes that — captured before `p.add_(update,-lr)` / decoupled decay.
   NOT reconstructed from rounded weight-diff/LR.
3. actual displacement δ_t = θ_{t+1}−θ_t (fp32) — `snapshot_weights_pre_update` + `capture_displacement`.

## Pre-launch storage/IO forecast (REQUIRED)
Three 768×768 matrices = 1,769,472 params. Per stream per step fp32 = 7.08 MB; ×3250 = 23.0 GB/stream/arm
(11.5 GB at bf16). **3 streams × 6 arms ≈ 414 GB fp32 (207 GB bf16)** + ~2 GB fork checkpoints @500/1500/2500.
Fits the durable shared FS (`csi-storage-sfs`, 54 TB free, 622 MB/s). Native-precision lossless; census dtypes
at build. Capture overhead tiny (3 small matrices), so the 6 runs are compute-bound (~6×≈baseline, ~1h each).

## Analysis (temporal STFT spectrograms)
Per coordinate of each selected matrix, temporal STFT over optimizer steps of the u_t stream (and g_t);
aggregate power P_m(τ,f)=Σ_i |Σ_j h_j u_{τ+j,i} e^{-2πi f j}|² (sum squared magnitudes across coordinates —
NOT norms/abs/signed-avg). Window-energy normalized, total + mean-per-coordinate power. Compare the fast
(period-two, f≈0.5 cyc/step) band across arms: does momentum / Muon's transform / batch size move power from
high to low f? Primary at equal optimizer-update count.

## Build order
1. (this) selection + 6-arm plan + forecast + RUNNING.
2. CaptureAdamW + u_t stash in Muon/SgdBlocks + restricted 3-stream capture hooks (CPU-validated).
3. 6 configs + LR stability pilot.
4. 6 runs (2-node ceiling, sequential) + STFT spectrogram analysis.
