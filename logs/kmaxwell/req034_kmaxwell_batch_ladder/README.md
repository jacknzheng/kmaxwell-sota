# REQ-034 — K-Maxwell on the fork@2000 batch ladder (bi-Maxwell protocol, 1×–16×) — **COMPLETE, headline positive**

**SHA `365c392d`, 2 nodes (8×H100), ≤2-node ceiling, seed 0, n=1/arm.** K-Maxwell (annealed_weights_muon, PR #357
shipped decays/weights) on the *exact* bi-Maxwell protocol — same step-2000 fork, 750-step window, @2750 readout — so
the two kernels finally sit on one axis. 6 arms: 5 annealed (1×/2×/4×/8×/16×) + a fresh μ0 @2× control (1×/4×/8×/16× μ0
taken from REQ-026/028/029 stored values, as the request directs).

## Result — K-Maxwell is the large-batch-durable kernel

`benefit = final_val(K-Maxwell) − final_val(μ0)`, same batch, @2750:

| batch | **K-Maxwell − μ0** | bi-Maxwell − μ0 (REQ-029) |
|:------|-------------------:|:--------------------------|
| 1× | −0.00462 | −0.01063 |
| 2× | **−0.00534** (fresh μ0) | — |
| 4× | −0.00668 | −0.00438 |
| 8× | **−0.00719** (peak) | −0.00233 |
| 16× | **−0.00576** | ~0.00000 |

**K-Maxwell's benefit is flat-to-growing across 1×–16×** (peaks ~8× at −0.0072, still −0.0058 at 16×), while
**bi-Maxwell's decayed monotonically to zero** by 16× — fully absorbed by large-batch gradient averaging, i.e. a pure
denoiser. So **the anneal does something structurally different from noise-averaging: it *retains* its advantage at
large batch where the frozen two-rate kernel loses it.** This is the request's predicted headline ("K-Maxwell holds its
gain at 8×/16× where bi-Maxwell went to zero → the large-batch-durable kernel"), and it puts REQ-033's "benefit grows
with batch" on the same axis as the bi-Maxwell curve.

## Gates (all passed before the arms)

Tests 6/6 · **smoke val@2125 = 3.46147 finite** (the annealed fork@2000 loads cleanly — the base@2000's momentum
lazy-inits the 8 anneal streams, no stream-mismatch NaN) · usable-batch budget OK (16× binding at margin +71, 86 chunks).

## Caveats

- **Base offset:** my base val@2000 = 3.44279 vs REQ-026's 3.44367 (−0.00088; likely FineWeb data-order/nondeterminism).
  My annealed vals sit on this slightly-lower base while the *stored* μ0 are on REQ-026's base, so the 1×/4×/8×/16×
  benefits are ~0.0009 more negative than a same-base comparison. Offset-corrected: 1× −0.0037 / 4× −0.0058 / 8× −0.0063
  / 16× −0.0049 — still clearly negative and still holding at 16×. **The 2× point uses a FRESH μ0 on my base (offset-free,
  −0.0053) and anchors the curve.** The shape (holds at large batch) is robust to 0.0009 (bi-Maxwell spans −0.0106→0).
- **n=1/arm.** The −0.005…−0.007 benefits are 25–35× the ~2×10⁻⁴ seed-noise floor (REQ-027); the 8×→16× dip is a few σ.

## Files
- `summary.tsv` — the 6 arms' val@2750 + the K-Maxwell−μ0 curve laid against bi-Maxwell + the offset caveat.
- `make_req034_configs.py` — the 6-config generator (fork@2000, ladder, annealed + μ0).

No secrets/weights/tensors committed. Ran under the ≤2 ceiling.
