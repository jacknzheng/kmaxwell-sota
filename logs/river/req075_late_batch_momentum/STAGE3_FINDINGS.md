# REQ-075 Stage 3 — equal-token performance (DONE)

4 configs (B/16B × Muon momentum on/off) from the identical **B-mom @2500** fork, **3 paired continuation
data-order seeds**, **equal token budget 736B = 385,875,968 tokens** (736 B-updates == 46 16B-updates; token
endpoint 3236/3250 truncated — recorded, not stretched). TOKEN-indexed LR schedule (`cool_down_by_tokens`) so
B and 16B get identical eta at equal tokens. wd=0 retained (no decay-shrinkage mismatch). Reporting-validation
= fineweb_val; LR-search tuning-validation = held-out chunk 99. 4-stage capture every update. Seed variance is
tiny (≤0.0004), so all contrasts below are far above seed noise but reported with 3-seed std.

## Equal-token final val_loss (mean ± std over 3 seeds; fork = 3.35099)
| config | inherited LR (Muon 0.00824) | tuned LR (×2 = 0.01648) |
|---|---|---|
| B-mom | 3.29611 ± 0.00030 | 3.29584 ± 0.00039 |
| B-nomom | 3.30752 ± 0.00023 | 3.30306 ± 0.00025 |
| 16B-mom | 3.32377 ± 0.00010 | 3.32243 ± 0.00014 |
| 16B-nomom | 3.32539 ± 0.00007 | 3.32221 ± 0.00008 |

## Paired contrasts
**Momentum effect (nomom − mom; >0 ⇒ momentum helps):**
| batch | inherited | tuned (selected ×2) |
|---|---|---|
| B | **+0.01141 ± 0.00010** | **+0.00722 ± 0.00023** |
| 16B | +0.00162 ± 0.00003 | **−0.00022 ± 0.00006** |

**Batch effect (16B − B; >0 ⇒ 16B worse per token):** inherited +0.0277 (mom) / +0.0179 (nomom); tuned
+0.0266 (mom) / +0.0192 (nomom).

**LR search (tuning-val, chunk 99):** every config selects the bounded-max multiplier ×2 (no broader sweep).
LR gain (0.5→2): B-mom +0.0047, B-nomom +0.0085, 16B-mom +0.0043, 16B-nomom +0.0061 — **nomom arms gain more
from higher LR** (higher LR partly substitutes for momentum's effective-step contribution).

## Findings
1. **Momentum's benefit tracks the noise level — it is denoising-dominated.** nomom−mom falls monotonically:
   B +0.0114 → 16B +0.0016 (inherited), and with the per-config-selected LR it goes B +0.0072 → **16B −0.0002
   (vanishes)**. At the reduced-noise 16B state (Stage-1 NSR 0.37) with a tuned step, removing momentum costs
   nothing. This is the registered interpretation #1 (*larger batch reduces the benefit of momentum ⇒ momentum
   chiefly denoises*).
2. **At small batch, momentum's benefit is NOT merely a learning-rate proxy.** It survives per-config LR tuning
   (+0.0072 at B even after giving nomom its best ×2 LR). So at high noise (NSR_B≈1.7) momentum does something
   raising the step cannot replace — but since the benefit *disappears* once noise is low (16B), the evidence
   points to denoising, **not** a separate conditioning/cancellation benefit (registered interpretation #3:
   no demonstrated useful-learning benefit beyond batch-noise averaging at low noise).
3. **Large batch does NOT reproduce the benefit — it is worse per token.** 16B loses +0.019…+0.027 nats to B
   at equal tokens in every condition. 16B makes *more accurate* updates (Stage 2: better per-update loss,
   displacement smoothed by orthogonalization) but only 46 of them vs 736 — the accuracy does not repay the
   token cost (registered interpretation #5: *better per update but worse per token ⇒ accuracy doesn't repay*).
4. **Answer to the registered question.** "Would a sufficiently large batch reproduce the benefit of
   suppressing late oscillation, and can small-batch momentum match/beat it for the same budget?" — **No and
   yes**: 16B does not reproduce the loss benefit (worse per token); **small-batch + momentum is the best
   recipe per token**, and its advantage over large batch is large and seed-robust. Momentum's own
   contribution is denoising, which large batch supplies differently (fewer, cleaner updates) but less
   token-efficiently.

### Caveats / scope
- Pilot statistics: 3 seeds (tiny variance here, but limited power — not claiming equivalence from
  non-significance). 16B groups have 46 updates; "comparable/better" claims rest on the large, consistent
  loss gaps, not on fine significance.
- 16B is a *reduced-noise* regime (NSR≈0.37), not noise-free (Stage 1) — the "momentum vanishes at 16B"
  result is at reduced, not zero, noise; a true infinite-batch claim is not made.
- A causal claim specifically about *period-two cancellation* driving the gains would need the registered
  matched-noise/matched-age filter follow-up (REQ-073 pair-average vs EMA control) — not run here.
- Period-two spectra at equal token (`stage3_spectra/`) corroborate Stage 2: see REPORT.md.
