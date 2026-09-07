# REQ-056 — K-Maxwell memory inside standard Adam — **does not reliably help Adam (n=3)**

**SHA `365c392d` + `KMaxwellAdam` (new optimizer, `apply_req056_opt.py`), 2 nodes (wdkrxgq + q4zk6y3,
8×H100, ≤2-node ceiling), venv019 torch 2.10.0+cu128.** Replaces **only Adam's first moment** with the
scheduled K-Maxwell mixture; the second moment, denominator, ε, and LR schedule are identical across arms.
This is standard Adam — **no** Muon polar transform, **no** Nesterov blend, **no** AdamH hyperball/scale-
invariant projection.

Settings recorded explicitly: **β1 = 0.9, β2 = 0.999, ε = 1e-8, zero weight decay** in all arms. The
memory intervention is applied to the **72 `blocks.*.weight` transformer matrices**; ordinary Adam (the same
`kmaxwell_adam` in `adam` mode, no shadow) runs on `embed`/`proj`/rest with identical group settings across
arms. Three arms:

1. **adam** — single first-moment EMA, β1 = 0.9.
2. **kmaxwell** — the REQ-054 scheduled mixture: 8 fixed-decay streams, weights annealed start→end over
   [1000, 3250], each stream bias-corrected `1 − decayᵢ^t`, mixed in Adam's numerator.
3. **agema** — a single first-moment EMA whose decay follows K-Maxwell's scheduled average age
   (`A(t)` = 58 → 26), with **variable-decay bias-correction mass `M = 1 − ∏ₜ β(τ)`** (not `1 − β^t`).

## Protocol

1× batch (**524,288 tokens**), microbatch 64, **3 independent seeds**, step-**1000 fork**, endpoint **3250**.
Each base trains 0→1000 with ordinary Adam while **accumulating the K-Maxwell streams and the age EMA in
shadow** (the update reads only the adam numerator, so the base trajectory is unchanged); at the fork every
arm loads that shadow history, sharing identical weights, Adam second moments, scheduler, RNG, and data
cursor. This is a standard-Adam continuation, **not** a Muon-trained base.

**LR pilot** (baseline Adam, separate seed 100, same horizon/schedule): endpoint val **4.637 / 3.842 /
3.726** at LR **1e-4 / 3e-4 / 1e-3** → **lr = 1e-3 frozen** for the paired comparison (all runs stable).

**Parity / correctness checks (all pass):**
- `KMaxwellAdam` in `adam` mode **and** with a single constant-decay 0.9 stream reproduces
  `torch.optim.Adam` to **4.4e-16** (machine ε) over 200 steps — the one-stream ≡ Adam check the request
  requires (`req056_parity.py`).
- Scheduled age **58.0 → 26.0** exactly over [1000, 3250]; the age-matched EMA's **realized (finite-history)**
  age tracks the mixture's realized age within **+0.4…0.8** at every checkpoint (`req056_age_trace.py`,
  `age_trace.txt`) — so the single EMA is genuinely age-matched and the **KM−agema** contrast isolates
  kernel *shape* (multiple timescales) from average memory age.
- No nonfinite values anywhere (all 3 pilots + 3 bases + 9 continuations exit 0 with finite val throughout).

## Result — K-Maxwell does not reliably improve standard Adam

Endpoint val_loss @ 3250 (`readout.tsv`):

| seed | adam | kmaxwell | agema |
|:----:|-----:|---------:|------:|
| 0 | 3.58041 | 3.57390 | 3.58160 |
| 1 | 3.74537 | 3.75555 | 3.76390 |
| 2 | 3.75034 | 3.76601 | 3.77290 |

Per-seed paired diffs and the across-seed verdict (practical-equivalence margin **0.0005**):

| contrast | per seed (0/1/2) | mean ± std | verdict |
|:---------|:-----------------|-----------:|:--------|
| **K-Maxwell − Adam** | −0.0065 / +0.0102 / +0.0157 | **+0.0065 ± 0.0094** | **INCONCLUSIVE / no improvement** (sign flips; \|mean\|<std; fails margin) |
| **age-matched − Adam** | +0.0012 / +0.0185 / +0.0226 | **+0.0141 ± 0.0093** | **REGRESSION** (all 3 seeds worse) |
| **K-Maxwell − agema** | −0.0077 / −0.0084 / −0.0069 | **−0.0077 ± 0.0006** | **IMPROVEMENT** (all 3 seeds) |

**K-Maxwell does not reliably help standard Adam at the frozen LR.** The KM−Adam difference flips sign
across seeds (K-Maxwell wins only on seed 0) and its across-seed spread swamps the mean, so it is
inconclusive and, if anything, a slight regression — nowhere near a 0.0005 improvement. Meanwhile the
**age-matched single EMA regresses Adam in all three seeds** (+0.014): a slow scheduled first moment (age up
to 58, β≈0.983) is too sluggish for Adam's numerator versus β1 = 0.9 (age 9). The **multi-timescale mixture
consistently beats that single age-matched EMA** (KM−agema −0.008 in all three seeds, tight std) — so kernel
*shape* still matters, echoing REQ-054's Muon finding — but it only recovers the ground the slow memory
loses; it does not beat plain β1 = 0.9 Adam.

**Bottom line:** the K-Maxwell benefit that is real and consistent for Muon (REQ-034/054) **does not transfer
to Adam's first moment** in this setting. Standard β1 = 0.9 Adam is already competitive; adding the scheduled
multi-timescale first moment neither reliably helps nor is necessary. A fixed-LR result at n=3 — this
establishes behavior at the shared LR, not a claim about each arm's best-tuned Adam.

## Realized Adam-step geometry (seed 0, reuse of the REQ-055 probe)

_Filled from `raw_json/geom_{adam,kmaxwell,agema}_s0_step{1000,2050,3248}.json` — the realized Adam
displacement `δ` (weight decay = 0, so `δ = δ_muon = δ_full`), on 3 fixed held-out probe minibatches:
downhill alignment `−g·δ/(‖g‖‖δ‖)`, directional curvature `vᵀHv` (gradient central-FD HVP + loss-scan 2nd
diff), update norm, and the loss scan along the step._

| step | metric | adam | kmaxwell | agema |
|:----:|:-------|-----:|---------:|------:|
| 1000 | downhill align | **+0.0229** | +0.0115 | +0.0070 |
|      | curvature `vᵀHv` (FD) | +371 | +164 | +112 |
|      | update norm ‖δ‖ | 1.731 | 0.607 | 0.566 |
| 2050 | downhill align | +0.0161 | +0.0148 | +0.0025 |
|      | curvature `vᵀHv` (FD) | +299 | +621 | +291 |
|      | update norm ‖δ‖ | 0.963 | 0.517 | 0.452 |
| 3248 | — | **undefined** (‖δ‖ ≈ 0.001 — the Adam step at the LR→0 cooldown tail is a zero-norm direction; `vᵀHv` = 0/0 blows up, flagged undefined per the request) | | |

At the two well-defined checkpoints the geometry **corroborates the null loss result**: plain Adam's step is
the *most* downhill (+0.0229 at 1000, vs K-Maxwell +0.0115 and agema +0.0070) and the largest (‖δ‖ 1.73 vs
0.61 vs 0.57), and K-Maxwell does not hit consistently lower curvature (lower than Adam at 1000, higher at
2050). K-Maxwell's Adam step is **not** better-aligned or flatter than plain Adam's — consistent with it not
improving the loss. (FD and loss-scan curvature agree to a few % at 1000/2050; both are meaningless at 3248
where ‖δ‖→0. Weight decay = 0, so ‖δ_wd‖ = 0 and δ = the pure Adam update, verified in every profile.)

## Benchmark

Step time (`step_avg`): **adam 168–208 ms | kmaxwell 177–225 ms (+~5–8%, the 8 shadow streams) | agema
170–210 ms**. So the K-Maxwell first moment costs ~5–8% throughput for no reliable loss benefit in Adam.
Memory: the 8 shadow streams add ~8× the first-moment buffer on the 72 blocks matrices; all arms fit
comfortably on 8×H100 (no OOM).

## Caveats

- **n = 3 seeds**, one frozen LR (1e-3, pilot-selected). The KM−Adam result is INCONCLUSIVE, not a proven
  equivalence: the seed spread (±0.009) is larger than the margin, so more seeds could resolve the sign.
  The age−Adam regression and the KM−agema improvement are consistent across all three seeds.
- Fixed-LR comparison: establishes behavior at the shared setting. A best-tuned-Adam claim would require
  equal-budget LR tuning per arm and fresh eval seeds (not done).
- The standard-Adam endpoints (~3.58–3.77) are well above the Muon Track-3 record (~3.28) — expected, this
  is plain Adam on a Muon-designed model; the experiment is the arm contrast at equal LR, not absolute SOTA.

## Files

- `readout.tsv` — endpoints, per-seed diffs, across-seed verdict, step time.
- `kmaxwell_adam.py` — the `KMaxwellAdam` optimizer (shadow K/age buffers, variable-decay age mass, 3 numerator modes, telemetry).
- `apply_req056_opt.py` — installs + registers `kmaxwell_adam`; patches the fork dump/load hooks to tolerate configs with no Muon-family group.
- `make_req056_configs.py` — pilot + base + 3-arm config generator. `req056_driver.sh` — per-node sequential runner.
- `req056_parity.py` — one-stream ≡ torch.optim.Adam parity check. `req056_age_trace.py` / `age_trace.txt` — 58→26 schedule + finite-history realized ages.
- `analyze_req056.py` — endpoint diffs + verdict, reproducible from `raw_logs/val_{arm}_s{seed}.tsv`.
- `raw_logs/` — per-arm val curves (step, val_loss, step_avg). `raw_json/` — seed-0 realized-Adam-step geometry profiles.

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
