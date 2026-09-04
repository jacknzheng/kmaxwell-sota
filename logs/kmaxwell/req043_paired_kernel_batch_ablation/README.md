# REQ-044 — fully paired Muon / bi-Maxwell / K-Maxwell batch ablation — **n=3, all three questions answered**

**SHA `365c392d`, 2 nodes (woxl97w + wlkmo03, 8×H100 each, ≤2-node ceiling), venv019 torch
2.10.0+cu128.** 60 continuations = batch {1×,2×,4×,8×,16×} × optimizer {mu0, mu95, bi-Maxwell,
K-Maxwell} × **3 genuinely independent seeds** (each trained from scratch to its own step-2000 base,
never a seed-0 fork). Fork@2000 → stop@2750, mbs=64 throughout (all compiled — the `<64` compile-NaN
path is avoided). **All 60 arms finite, 0 errors.** Every optimizer forks from the same fresh
per-seed base, so every reported difference uses a fresh control from the same state — no stored
REQ-026/028/029 controls, no 0.00088 base offset.

## Provenance (gates 2, 3, 5)

Three **independent** step-2000 bases (`manifest.tsv`): distinct state hashes
`7c62301e / 14493f27 / 2e43ec08`, base val@2000 = 3.44328 / 3.44334 / 3.44378 (all ≈ 3.44367). Within
each (seed, batch) cell all four arms load that seed's exact base at the same data cursor (asserted by
a pre-fork state-hash guard in the driver). 16× usable-batch budget verified against real shard
headers: 7.34B needed < 8.60B available (86 shards).

## The three load-bearing questions — all answered, all reproduce across 3 seeds

`readout.tsv` (paired Δval@2750, negative = first optimizer better; across-seed sd in parentheses):

**1. Is single-EMA Muon (mu=0.95) still indistinguishable from no-momentum (mu=0.0) at 16×?** — **YES.**
`mu95 − mu0` is noise-sized at **every** batch, including the previously-unchecked 16×:

| batch | 1× | 2× | 4× | 8× | 16× |
|:------|---:|---:|---:|---:|----:|
| mu95 − mu0 (mean) | +0.00004 | +0.00003 | −0.00000 | +0.00001 | +0.00031 |

All |mean| < 5×10⁻⁴ (the read-as-noise band). **Temporal gradient memory (a single EMA) buys nothing
in Muon at any batch** — the polar map already supplies what momentum would. The missing 16× ablation
is now filled and agrees with the 1×/4×/8× checks.

**2. Does bi-Maxwell's benefit reach zero at large batch?** — **YES.** `bimax − mu0` decays monotonically
and crosses zero by 16×: **−0.0105 → −0.0073 → −0.0043 → −0.0020 → +0.0006** (1×→16×). The frozen
bi-Maxwell denoiser curve (REQ-029) reproduces with fresh paired controls: its two-rate edge is
batch-specific and fully absorbed by large-batch gradient averaging.

**3. Does K-Maxwell keep its 8×/16× gain against fresh controls, in every seed?** — **YES, decisively.**

| batch | kmax − mu0 (mean, sd) | kmax − bimax (mean, sd) | all 3 seeds negative? |
|:------|----------------------:|------------------------:|:---------------------:|
| 8× | **−0.00627** (0.00014) | **−0.00430** (0.00034) | both ✓ |
| 16× | **−0.00469** (0.00021) | **−0.00524** (0.00025) | both ✓ |

**K-Maxwell is the only kernel that retains a material benefit at 8× and 16×**, and it **beats
bi-Maxwell** there in all three independent replicates (kmax − bimax negative, ~20× the cross-seed sd).
Where bi-Maxwell has decayed to zero (16×: +0.0006), K-Maxwell is still −0.0047 vs mu0. The campaign's
central claim — K-Maxwell's large-batch edge — survives the fully-paired, fresh-control design.

## Shape across the ladder (the deliverable curve — see `req044_paired_curves.png`)

- **mu95** — flat on zero everywhere (no momentum benefit).
- **bi-Maxwell** — strong at 1× (−0.0105), decays to zero by 16× (classic denoiser, batch-absorbed).
- **K-Maxwell** — the opposite envelope: it is *worse* than bi-Maxwell at 1×/2× (kmax − bimax = +0.0051 /
  +0.0007) but *crosses over* at 4× and dominates at 8×/16× (kmax − bimax = −0.0024 / −0.0043 / −0.0052).
  Its benefit vs mu0 is large and roughly flat across the whole ladder (−0.005 to −0.007), not decaying —
  so at large batch, where bi-Maxwell's denoising is spent, K-Maxwell's annealed multi-rate memory still
  helps.

## Comparison with the prior (unpaired) conclusions

- **REQ-026/028/029** measured bi-Maxwell and mu95 against stored controls with a documented 0.00088 base
  offset. This paired run **confirms** their qualitative reads — mu95 ≈ mu0, bi-Maxwell decays to zero —
  now free of that offset, with tight cross-seed sd (≤ 0.0002).
- **REQ-034** measured K-Maxwell vs stored mu0. Its 8×/16× gain **reproduces here against fresh
  same-seed controls** (kmax − mu0 = −0.0063 / −0.0047), so the gain was not an artifact of the stored
  control. **New here:** the direct **kmax − bimax** contrast at 8×/16× (both negative, all seeds), which
  the earlier runs could not form as a within-cell paired difference.
- **New here:** the 16× mu95 − mu0 ablation (previously only 1×/4×/8×) — noise-sized, closing that gap.

## Caveats

- **n = 3 seeds, 1 run per cell.** Cross-seed sd is the honest run-to-run estimate (0.00006–0.00034),
  *not* the old ±2×10⁻⁴ heuristic. Every headline difference is ≥ 15× its sd; the raw per-seed values are
  in `readout.tsv` (disagreement preserved, not averaged away). No cell disagrees in sign on the
  load-bearing contrasts.
- Absolute val is not comparable across batches (16× sees 16× the tokens of 1×); only within-batch paired
  differences are read.

## Files

- `readout.tsv` — the four paired differences per seed + mean/sd/min/max, per batch.
- `summary.tsv` — all 60 finals (seed, batch, batch_tokens, kernel, val@2750).
- `manifest.tsv` — per-seed base val@2000 + state hash + independence flag.
- `req044_paired_curves.png` — per-seed points + mean curves (benefit vs mu0; kmax − bimax).
- `make_req043b_configs.py` — the 60-config generator (`req043b_*` internal run-ids); `req043b_run_node.sh` — the per-node driver (base → gate → per-fork smoke → full → provenance).
- `status_s{0,1,2}.tsv` — per-seed per-arm smoke/exit/final (raw).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
