# REQ-075 — late-training large batch vs momentum: final report

**Status: complete (pending equal-token period-2 spectra figures).** All training from the REQ-073 **B-mom
fork @ step 2500** (complete model + 8-rank optimizer + muon_steps_seen; provenance in `PROVENANCE.md`). One
node (1×8 H100, `wd8vemq`), within the 2-node limit. Raw tensors/checkpoints off-Git on durable FS; derived
arrays, configs, readers, figures committed. Measured findings are kept separate from interpretation.

## The question
Would a sufficiently large batch reproduce the benefit of suppressing late-training oscillations, and can
small-batch momentum achieve comparable-or-better loss for the same data/compute budget? Separate random
gradient noise from optimizer-dynamics oscillation; measure useful learning, not just smoother curves.

## Answer
**No, and yes.** A 16× batch does **not** reproduce momentum's loss benefit — it is **worse per token** in
every condition (16B−B = +0.019…+0.027 nats at 736B tokens, 3 seeds). **Small-batch + momentum is the most
token-efficient recipe.** Momentum's own contribution is **denoising**: its benefit falls monotonically with
batch size and **vanishes at 16B once the LR is tuned** (nomom−mom: +0.0114 @B → −0.0002 @16B). The
late-training period-two oscillation is **optimizer-dynamics / overshoot driven, not sampling noise**, and is
**revealed, not removed, by large batch**.

## Evidence by stage
### Stage 1 — frozen-state noise (`STAGE1_FINDINGS.md`)
At x_2500, over 64 B-probes: the gradient is **noise-dominated at B** (NSR_B ≈ 1.7, noise > signal); noise
scales ≈ 1/√B; **16B is NOT low-noise** (NSR ≈ 0.37, far above the predeclared 0.10; reaching 0.10 needs
~220B tokens/update). So 16B is a *reduced-noise*, not noise-free, regime — every later "16B" statement is
read that way.

### Stage 2 — 256-update mechanism pilot, constant LR (`STAGE2_FINDINGS.md`)
Equal-updates (16B saw 16× tokens). **Period-two in the raw gradient is a large-batch phenomenon**: lag-1
cosine ≈ 0 and period-2 fraction ≈ 3% at B (masked by noise) → lag-1 −0.75 and period-2 fraction **49–83% at
16B**. **Half-LR collapses it** back to 6–8% (same tokens/noise, half the step) ⇒ overshoot/dynamics origin,
not noise. **Momentum low-passes**: it turns the (near-white or period-two) gradient into strongly aligned
displacements (lag-1 +0.65…+0.81) and removes period-two from the step (16B-mom disp p2-frac 0.016 vs
16B-nomom 0.073). Orthogonalization only partially dilutes the gradient's period-two into the step.

### Stage 3 — equal-token performance, 4 configs × 3 seeds + LR search (`STAGE3_FINDINGS.md`)
Equal 736B tokens, token-indexed schedule. **Momentum benefit is denoising-dominated**: nomom−mom = +0.0114
(B) → +0.0016 (16B) inherited, and B +0.0072 → **16B −0.0002** after per-config LR tuning (all select ×2). At
B the benefit survives LR tuning (not a mere step-size proxy at high noise); at reduced-noise 16B it vanishes
(no conditioning benefit beyond denoising demonstrated). **16B worse per token** throughout: more accurate
updates (Stage 2) but only 46 vs 736 of them — accuracy does not repay the token cost.

## Mapping to the registered interpretations
- #1 *larger batch reduces noise/oscillation and the benefit of momentum ⇒ momentum chiefly denoises* —
  **supported** (benefit monotone↓ in batch; →0 at 16B+tuned).
- #2 *oscillation survives noise reduction but falls with half LR ⇒ overshoot/dynamics* — **supported** (16B
  raw-grad period-2 is large despite reduced noise; half-LR collapses it; absolute power examined, not
  fractions alone).
- #3 *momentum still improves loss at low noise ⇒ benefit beyond averaging* — **not supported at low noise**
  (vanishes at 16B+tuned); at B it persists but B is high-noise, so still denoising-consistent.
- #5 *better per update but worse per token/time ⇒ accuracy doesn't repay* — **supported** (16B).

## Limits (not overclaimed)
3-seed pilot (tiny variance, limited power — no equivalence claimed from non-significance). 16B is reduced-,
not zero-, noise (no infinite-batch claim). Selected-matrix (block-5 Q/K/MLP) results do not establish
whole-model behaviour. A causal *period-two-cancellation-causes-gains* claim needs the registered
matched-noise/matched-age filter follow-up (REQ-073 pair-average vs EMA), not run here. No additional late
checkpoint confirmed (equal-token result is single-checkpoint); the registered second-checkpoint confirmation
is a scoped follow-up.

## Resource use
1×8-H100 `wd8vemq`: Stage 1 ~0.5 node-h; Stage 2 (5 arms) ~0.7 node-h; Stage 3 (24 inherited + LR + 12 tuned
= 36 runs) ~3 node-h; analysis/spectra ~0.5 node-h. ~14.5B training tokens total, within forecast. Corpus:
103 fineweb10B chunks (10.3B tokens); all training windows + probe region disjoint, no cycling (data module
does not wrap). Node stopped after delivery.
