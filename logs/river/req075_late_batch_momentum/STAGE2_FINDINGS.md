# REQ-075 Stage 2 — 256-update mechanism pilot (DONE; spectra pending)

5 arms from the identical REQ-073 **B-mom @2500** fork, one shared data-order seed (all start at the same
token, seq 1,286,144), **LR held constant** at the checkpoint value (eta(2500)=0.3297 ⇒ Muon 0.00824; half-LR
0.00412), 256 optimizer updates each. 4-stage capture (raw grad / post-momentum / post-polar / displacement)
for block-5 Q/K/MLP every update. **Not compute-matched**: 16B arms consumed 16× the tokens of B arms (the
registered design). All arms resumed from val 3.35099 at step 0 (identical state confirmed).

## Loss (val at end of 256 updates; Δ from the 3.35099 fork)
| arm | val_end | Δ | note |
|---|---|---|---|
| B-mom | 3.32051 | −0.0305 | |
| B-nomom | 3.32411 | −0.0269 | momentum edge at B = 0.0036 |
| **16B-mom** | **3.28377** | **−0.0672** | best (16× tokens) |
| 16B-nomom | 3.30154 | −0.0495 | momentum edge at 16B = 0.0177 |
| 16B-nomom-halfLR | 3.30619 | −0.0443 | |

- **16B beats B per update** (saw 16× tokens; not compute-matched — Stage 3 does the equal-token test).
- **Momentum helps MORE at 16B (Δ-gap 0.0177) than at B (0.0036).** If momentum acted *purely* by averaging
  batch noise, it should help more at noisy B (NSR_B≈1.7) and less at reduced-noise 16B (NSR≈0.37) — the
  observed pattern is the opposite, hinting momentum's late-state benefit is not only denoising (conditioning/
  dynamics). Confounded by the 16× token difference; Stage 3 (equal token) is the real test.
- Half-LR (16B-nomom) < full-LR 16B-nomom: smaller steps, less progress per update.

## Mechanism: lag-1/lag-2 direction agreement (tail, after the 8-update switch transient)
Per-update vector cosine; lag-1 < 0 == period-two (alternating) motion. (attn.q / mlp.proj shown.)

| arm | grad lag-1 | post-mom lag-1 | **disp lag-1** | disp lag-2 |
|---|---|---|---|---|
| B-mom | +0.03 / +0.03 | +0.76 / +0.76 | +0.65 / +0.76 | +0.61 / +0.72 |
| B-nomom | −0.04 / −0.04 | (=grad) | +0.00 / +0.00 | +0.00 / +0.00 |
| 16B-mom | **−0.36 / −0.33** | +0.54 / +0.55 | +0.71 / +0.81 | +0.68 / +0.78 |
| 16B-nomom | **−0.75 / −0.66** | (=grad) | −0.02 / −0.01 | +0.04 / +0.05 |
| 16B-nomom-halfLR | −0.17 / −0.15 | (=grad) | +0.01 / +0.03 | +0.02 / +0.03 |

### Findings
1. **Period-two in the RAW gradient is a large-batch / low-noise phenomenon.** grad lag-1 ≈ 0 at B (masked by
   sampling noise — Stage 1 found NSR_B≈1.7), but strongly **negative at 16B** (−0.75 for 16B-nomom). Reducing
   noise by 16× *reveals* a deterministic period-two alternation in the gradient that is invisible at B. This
   **reframes REQ-073**: the small-batch update-space period-two reported there was a momentum-dynamics effect;
   the raw-gradient period-two only surfaces once large batch removes the masking noise.
2. **Momentum is a low-pass that cancels the oscillation.** For mom arms, consecutive *displacements* are
   strongly aligned (lag-1 +0.65…+0.81) even when the underlying gradient is period-two (16B-mom grad −0.36 →
   disp +0.71). nomom arms (mu=0 ⇒ s2=s1) have no such smoothing.
3. **Half-LR sharply weakens the gradient period-two** (16B-nomom −0.75 → half-LR −0.17). Since halving the
   step (not the noise) suppresses the oscillation, this **supports an overshooting / optimizer-dynamics
   origin** for the period-two, not a sampling-noise origin (the spec's registered "falls with half LR" case).
4. **Orthogonalization does not pass the gradient's period-two through as a step oscillation**: 16B-nomom has
   grad lag-1 −0.75 but displacement lag-1 ≈ 0 (near-white steps), not −0.75. The period-two lives in the raw
   gradient; Newton-Schulz orthogonalization of the mu=0 update produces near-uncorrelated steps. (The
   companion absolute period-two spectral power — req074 time-resolved, window 128 hop 1 + 32 sensitivity —
   quantifies this; see `stage2_spectra/`.)
5. **Step size is LR-controlled** (Muon normalization): displacement norms ≈ 0.091 for all full-LR arms,
   ≈ 0.046 for half-LR — independent of batch/momentum.

### Caveats
- Equal-updates, NOT equal-tokens: the 16B-vs-B loss gap is mostly the 16× token difference; Stage 3 resolves
  useful-descent per token/compute. The momentum-helps-more-at-16B observation is likewise token-confounded.
- Switch transient (first 8 updates) reported separately in the per-arm JSON, not discarded.
- Selected-matrix (Q/K/MLP) agreement does not establish whole-model agreement (not claimed).

Raw 4-stage captures off-Git on durable FS; `stage2_analysis/*.json` (norms + lag series) and
`stage2_spectra/*.npz` (absolute/normalized period-two power) committed.
