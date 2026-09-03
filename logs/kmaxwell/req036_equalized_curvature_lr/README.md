# REQ-036 — equalized-curvature per-type learning rates — **NULL RESULT (val), n=1/arm**

**SHA `25d3208` (`PerMatrixLrMuon`), 2 nodes (q89y4d3 + qekd95q, 8×H100), venv019 torch 2.10.0+cu128, ≤2-node
ceiling.** Five 750-step continuations from a regenerated shared **step-2000** state (base val@2000 = 3.4444/3.4443,
matching the expected 3.44367), read out at val@2750. The design derived per-type LR multipliers from the measured
curvature C (REQ-019/023) to put every matrix type at the same equilibrium curvature; this tests whether that
improves training vs uniform LR.

## Result — uniform control wins; every per-type intervention is worse

| arm | rule | val@2750 | Δ vs control (lower = better) |
|:----|:-----|---------:|------------------------------:|
| **a1 control** | all 1.0 | **3.51052** | 0 (best) |
| a2 per-type | measured-C rule | 3.51295 | +0.00243 |
| a4 anti-rule | 1/arm2 | 3.51515 | +0.00463 |
| a3 per-type + end-cap | blocks 0,11 proj capped | 3.51996 | +0.00944 |
| a5 polar target | iteration-27 rule | 3.53460 | **+0.02408 (worst)** |

**The equalized-curvature per-type LR design does not beat uniform LR at this fork/horizon** — every variant
*increases* val@2750. Notably the **predicted-best arm (a5 polar) is the worst** (+0.024), and the per-type rule (a2)
does **not** reproduce the expected "beats control by 0.001–0.006" — it is worse than control by +0.0024.

**One real but net-harmful signal:** a2 (rule, 3.51295) < a4 (anti-rule, 3.51515) by 0.0022, so equalizing curvature
*is* better than anti-equalizing — the mechanism has a direction. But **both are worse than doing nothing** (uniform
LR), so the per-type intervention provides no net gain here. The request's mechanism-death test ("if the anti-rule a4
also beats control") is not triggered (a4 is worse than control), but neither does the rule a2 beat control.

## Caveats

- **n=1 per arm.** The ≥0.002 dex... (val) gaps are ~10× the ~2×10⁻⁴ val seed-noise floor (REQ-027), so "control beats
  all" is robust; the a2-vs-a4 gap (0.0022) is real but small. Replicates would tighten the small ones.
- **Config verified correct:** a1 has all-1.0 multipliers; a5's per-matrix `lr_multipliers` equal each matrix's type
  polar value for all 72 matrices; built via the official `sorted_matrix_names()` + `common()` (REQ-023's proven
  optimizer order). So the null is not a mis-mapping.
- **Curvature-spread (mechanism verification):** the arms dumped their model checkpoint at step **2250** (my
  `checkpoint_model_at_cadence every=750` fired at 2250, not 2750 — a cadence miss). The per-type curvature spread is
  therefore measured at 2250 (equilibrium curvature is stable by then under fixed LR); see `curvature_2250/` (appended
  when the probes finish). The val@2750 above is exact (read from the run logs, independent of the checkpoint).

## Files

- `summary.tsv` — the 5 arms' val@2750 + benefit vs control + the multiplier tables + the caveat.
- `make_req036_arms.py` — the arm-config generator (per-type → per-matrix `lr_multipliers`).
- `curvature_2250/` — per-arm per-matrix curvature at step 2250 (mechanism check), when available.

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.

## Curvature-spread mechanism check (appended) — the intervention works, the premise is falsified

Per-type curvature spread @2250 (std of log₁₀ top_eigenvalue across the 6 types; smaller = more equalized) — `curvature_2250/spread.tsv`:

| arm | across-type curvature spread | val@2750 |
|:----|-----------------------------:|---------:|
| a1 control | 0.2457 | 3.51052 (best) |
| a2 per-type | 0.1941 | 3.51295 |
| a3 end-cap | 0.1630 | 3.51996 |
| **a5 polar** | **0.1281 (most equalized)** | **3.53460 (worst)** |
| a4 anti-rule | 0.4441 (anti-equalized) | 3.51515 |

**The interventions do exactly what they target:** a5 polar equalizes per-type equilibrium curvature the *most* (0.128 vs
control's 0.246), and the anti-rule a4 *anti*-equalizes it (0.444) — the per-type LR knob controls curvature spread as
designed. **But equalization is inversely related to val:** the arm that equalized curvature most (a5) has the *worst*
loss; control (least equalized) has the *best*. So the mechanism is real, but **the design's premise — that equal
per-type equilibrium curvature improves training — is falsified: equalizing it HURTS.** REQ-036 is therefore a clean
directed-negative, not merely a null: the LR rule achieves its stated target and the target turns out to be the wrong
objective.
