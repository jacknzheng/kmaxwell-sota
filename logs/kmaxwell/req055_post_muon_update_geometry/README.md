# REQ-055 — geometry of the *realized Muon update*: K-Maxwell vs the age-matched single EMA — **no per-step advantage (n=1 seed, 4 checkpoints)**

**SHA `365c392d` + `AgeMatchedEmaMuon` (`apply_req054_opt.py`), 2 nodes (wdkrxgq + q4zk6y3, 8×H100,
≤2-node ceiling), venv019 torch 2.10.0+cu128.** One seed forked from its step-2000 base into two paired
arms (fork@2000 → 2750, same base/data cursor, only the block optimizer differs), with **consecutive model
dumps** at 2050/2051, 2250/2251, 2500/2501, 2749/2750 (each pair labeled `S → S+1`; no step past the stop
was trained):

- **kmax** — `annealed_weights_muon` (K-Maxwell): 8 EMA streams, weights annealed SW→EW.
- **agema** — `age_matched_ema_muon`: single EMA tracking K-Maxwell's scheduled memory age `A(t)`=58→26
  (the REQ-054 control that K-Maxwell beats by ~0.010 val).

## Displacement decomposition (the request's central separation)

Muon applies `p ← p·(1−lr·wd) − lr·update`, so for each of the 72 Muon matrices the realized displacement
splits exactly as `δ_full = δ_wd + δ_muon`, with `δ_wd = −(lr·wd)·x_S` (`lr = 0.025·η(S)`, `wd = 0.05`;
`η` from the linear `cool_down` schedule, `cooldown_frac 0.7`, `train_steps 3250`: `η` = 0.528 / 0.440 /
0.330 / 0.220 at 2050 / 2250 / 2500 / 2749). **`δ_muon`** — the polar/shape/LR update itself — is the
analysis direction. Non-Muon displacement = `x_{S+1}−x_S` on the AdamW params (embed/proj).

| step | ‖δ_full‖ | ‖δ_wd‖ | ‖δ_muon‖ | ‖δ_nonMuon‖ | wd fraction | recon residual |
|:----:|---------:|-------:|---------:|------------:|:-----------:|---------------:|
| 2050 | 3.74 | 0.39 | 3.78 | **713.9** | 10.4% | 3.4e-08 |
| 2250 | 3.14 | 0.34 | 3.15 | 593.4 | 10.8% | 2.8e-08 |
| 2500 | 2.36 | 0.25 | 2.37 | 452.4 | 10.5% | 2.0e-08 |
| 2749 | 1.58 | 0.16 | 1.58 | 305.9 | 10.0% | 1.3e-08 |

`δ_wd + δ_muon` reconstructs `δ_full` to **~1e-8** (fp32 roundoff). Weight decay is a steady **~10–11%** of
the Muon step. The **non-Muon (embed/unembed) displacement dwarfs the Muon step by ~200×** — the realized
*full* model step is dominated by the embedding/unembedding update, not the blocks. (Values shown for kmax;
agema matches within <0.2%.)

## Charter result — K-Maxwell's realized Muon step has no per-step geometric advantage

On 3 fixed held-out probe minibatches (paired across arms), for `v = δ_muon/‖δ_muon‖`:

| metric | kmax − agema (mean) | per-step 2050/2250/2500/2749 | reading |
|:-------|--------------------:|:-----------------------------|:--------|
| **downhill alignment** `−g·δ/(‖g‖‖δ‖)` | **−0.0011** | −0.0036 / 0.0000 / −0.0011 / +0.0002 | **NULL** |
| **directional curvature** `vᵀHv` (FD) | **−0.79** | −15 / +8.6 / −0.65 / +3.6 | **NULL** (scale ~30–66, sign flips) |
| **uphill-matrix fraction** | **+0.0000** | +0.04 / 0.00 / −0.03 / −0.01 | **NULL** |

**The realized Muon step of K-Maxwell is geometrically indistinguishable from the age-matched single EMA**
at all four checkpoints — same downhill alignment, same directional curvature, same fraction of matrices
stepping uphill. The `vᵀHv` difference flips sign across steps, so the −0.79 mean is noise on a ~30–66
scale. K-Maxwell's ~0.010 val advantage over the age-matched EMA (REQ-054) is therefore **not** delivered by
a better-aligned or lower-curvature *individual* step; it is a **trajectory-level / accumulated** effect —
consistent with REQ-054's reading that the benefit is the multi-timescale mixture's expressivity built over
many steps. (Sanity: kmax's held-out `L0` is lower than agema's at every checkpoint — the REQ-054 gap
reproduces in this fork.)

Both arms take steps that are only weakly aligned with the *instantaneous* held-out gradient (cosine
~0.003–0.016) — expected, since Muon's orthogonalised momentum is not the current minibatch gradient — yet
the along-direction descent is real: the linear term `α·g·δ_muon` is large (`g·δ_muon` ≈ −350…−4800) because
`‖g‖` is large.

## Directional-derivative validation (HVP-vs-finite-difference, and the α-scan)

Curvature is measured two independent ways (an autograd HVP is **impossible** here —
`aten::_scaled_dot_product_flash_attention_backward` has no derivative — so both are finite-difference):

- **`vHv_fd`** — gradient central finite-difference HVP, `(g(x_S+εv)−g(x_S−εv))·δ_muon/(2ε‖δ_muon‖²)`,
  `ε=0.01`, **local at `x_S`, cross-matrix terms retained** (full-δ perturbation). This is the "curvature at
  the same state as `g`" the request asked for.
- **`vHv_scan`** — symmetric 2nd difference of the loss scan, `(ΔL(+0.5)+ΔL(−0.5))/(0.25‖δ_muon‖²)` (the
  effective curvature averaged over `±0.5δ`).

They **agree to <6%** at 2050/2250/2500 (kmax), e.g. 2050 agema 65.6 vs 65.6, 2250 kmax 49.3 vs 50.6 —
validating both estimators. At **2749 they diverge ~45%** (FD 30 vs scan 44): `‖δ_muon‖` is smallest and
the `±0.5δ` scan reaches an **anharmonic** region where the local FD and the wider scan legitimately differ
— which is precisely what the larger-α scan is for.

The **loss scan vs quadratic prediction** `α·(g·δ)+½α²·(δᵀHδ)` tracks the observed curve closely to α≈1
(2050 kmax α=0.5: obs −86.7 vs pred −84.0; α=1: obs +6.7 vs pred +14.2) and deviates mildly by α=2 (obs 738
vs pred 757) — the quadratic model is accurate well past the realized step.

## Shared finding (both arms) — the Muon step overshoots its own direction ≈2× (edge of stability)

The loss-scan minimum along `δ_muon` sits at **α ≈ 0.25–0.5, not the realized α = 1**, for *both* arms
(token-summed loss, `L0 ≈ 80k`):

| step | arm | α=0.25 | α=0.5 | **α=1 (realized)** | α=2 |
|:----:|:----|-------:|------:|-------------------:|----:|
| 2050 | kmax | −64.0 | **−86.7** | +6.7 | +738 |
| 2050 | agema | −87.9 | **−121.2** | −13.1 | +907 |
| 2749 | kmax | +0.1 | −4.7 | +6.4 | +84 |

A half-step lowers held-out loss by ~87–121 while the full realized step lands near baseline — the step is
~**2× its along-direction optimum**, the model landing on the far wall at similar loss. With the steep
quadratic (`vᵀHv ~ 30–66`) and the α=2 blow-up, this is the edge-of-stability signature: step size is
calibrated to the *training-distribution* geometry (`λ_max·lr ≈ 2`), not to any single held-out direction.
The overshoot is identical across arms (a Muon-at-EoS property, not a kernel property) and shrinks toward
2749. The **full realized displacement** (`δ_full` + non-Muon) at α=1 is more positive than the Muon-only
scan (2050 kmax: full +86 vs Muon +6.7), i.e. the large embed/unembed update *also* overshoots on held-out.

## Caveats / method

- **n=1 seed**, 4 checkpoints. The three geometry contrasts are null at every checkpoint individually, so the
  "no per-step advantage" verdict is robust to n; a small *advantage* would need multiple seeds to exclude,
  and this run excludes only effects above the observed ~0.001 alignment / ~10-unit curvature scatter.
- **Held-out probe only** (3 fixed minibatches paired across arms). The training-batch alignment the request
  also lists was not collected — reconstructing the exact step-`S` training minibatch from the data cursor is
  fragile and not needed for the arm contrast, which the request itself centres on the fixed held-out probes.
- **Instrumentation is non-perturbing:** the only added hook is `checkpoint_model_at_cadence`, a pure read +
  save of `x_t` in the `pre_optimizer` slot; it mutates no state, so the instrumented trajectory equals the
  uninstrumented one by construction (no separate match run was needed).
- **`dL @ realized (α=1)` is not used for the arm verdict:** the arms sit at different `x_S`/`L0` and `ΔL` is
  a token-sum, so absolute realized-step loss changes are not comparable; only the scale-invariant geometry
  (alignment, curvature per unit δ, uphill fraction, overshoot factor) is compared across arms.
- **Probe hardening (adversarial validation), three defects caught and fixed:** (1) an autograd HVP crashed
  on flash-attention's missing double-backward → replaced with two finite-difference estimators; (2) `L(x_S)`
  was not reproducible (`ΔL(0.0)≈+4…+6`) because the model ran in `train()` mode with add/sub param
  round-trips → fixed with `model.eval()` + exact reset `copy_(x_S+α·δ)`, `ΔL(0.0)=0.0000` verified on all 8
  probes; (3) a double-base-run corruption (two concurrent step-2000 bases per node, ~1080 ms/step, about to
  clobber dumps) → killed, cleaned, one clean arm/node relaunched at healthy 157 ms/step.

## Files

- `readout.tsv` — per-(step,arm) decomposition / alignment / dual-method curvature / scan + the across-step verdict.
- `analyze_req055.py` — the comparison, curvature cross-check, decomposition, and quadratic check, reproducible from the raw JSONs.
- `measure_step_geometry.py` — the probe (δ_muon separation, FD-HVP + loss-scan curvature, quadratic prediction, full-displacement α=1, eval-mode exact reset).
- `req055_run.sh` — driver (base→2000, then arm→2750 with dense consecutive model dumps).
- `raw_json/geom_{kmax,agema}_s0_step{2050,2250,2500,2749}.json` — 8 profiles (source of truth; each carries per-matrix alignment + per-matrix FD curvature, grouped by type/depth via the block index in the name).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
