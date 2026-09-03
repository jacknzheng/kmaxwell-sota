# REQ-037 — a non-LR (gradient) instrument for the curvature-gradient exponent — **arms 1-3 (batch), n=1; arm 4 deferred**

**SHA `ebf53cd`, 2 nodes (8×H100), ≤2-node ceiling.** The batch instrument moves gradient noise while holding each
matrix's LR fixed; the exclusion-restriction test asks whether curvature responds to the gradient (via batch) — if the
LR→curvature effect really flows through g, moving g should move curvature.

## Result — the gradient (batch) instrument moves curvature only weakly

| arm | batch | val@2750 | geomean curvature@2750 | engine |
|:----|:------|---------:|-----------------------:|:-------|
| a2 | 0.5× | 3.62601 | 12551 | eager |
| a1 | 1.0× (control) | 3.51229 | 11614 | compiled |
| a3 | 2.0× | 3.42097 | 13684 | compiled |

**Per-matrix elasticity `dlog(curvature)/dlog(batch)` = median 0.075, mean 0.062, spread [−0.25, 0.36].** Curvature
responds only **weakly and noisily** to moving the gradient via batch at fixed LR; geomean curvature is non-monotonic.
Suggestively, a weak batch→curvature response is consistent with **the gradient channel not being the dominant path for
the LR→curvature effect** — i.e. the exclusion restriction behind the "causal exponent = 2" estimate is questionable.

## Two load-bearing caveats

1. **The batch instrument is confounded by tokens-seen.** Over 750 steps, 0.5× batch trains on half the tokens and 2×
   on double, so the curvature difference mixes gradient-noise with *training amount* (val@2750 is monotonic in tokens,
   as expected). The clean instrument is a **per-matrix gradient clip** (moves g magnitude at fixed tokens/batch) — that
   is **arm 4**, and it is **deferred**: no clip hook exists in `ebf53cd`; it needs a new forward/backward hook that
   clips each Muon matrix's gradient at a fixed percentile. Read arms 1–3 as a confounded first look, not the clean test.
2. **n=1/arm, 3 batch points** → the elasticity is noisy; median 0.075 means "small / near zero", not a precise value.
   arm 2 (0.5×) also ran eager (compile mbs<64 NaN bug) while 1/3 compiled — a code-path difference on the leftmost point.

## Files
- `summary.tsv` — the 3 arms + elasticity + caveats. `curvature_json/` — raw per-rank curvature JSONs @2750.
- **Follow-up:** implement the per-matrix gradient-clip hook and run arm 4 for the clean, tokens-controlled instrument.

No secrets/weights/tensors committed. Ran under the ≤2 ceiling.
