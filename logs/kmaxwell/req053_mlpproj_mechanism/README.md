# REQ-053 — what makes `mlp.proj` different: expansion ratio vs nonlinearity — **both hypotheses fail (n=2/arm)**

**SHA `25d3208` + a one-line model patch per arm, 2 nodes (wdkrxgq + q4zk6y3, 8×H100, ≤2-node ceiling),
venv019 torch 2.10.0+cu128.** `mlp.proj` has an excess curvature-to-gradient elasticity `c` over the
other five matrix types (+0.578 pooled on committed data; +0.514 at n=4 from REQ-051). Two surviving
hypotheses select the same rows in the 4× ReLU² architecture and are separated here by changing the
architecture and re-running the REQ-051 per-matrix-LR intervention (6-level Latin square), fitting
`c` from `d log λ = a + b·d log g + c·[mlp.proj]·d log g` (matrix FE) at equilibrium (2750):

- **H-C — fan-in shape.** `mlp.proj` (dim, hdim) is the only matrix with fan-in > fan-out; predicts `c`
  scales with the fan-in ratio (larger at 8×, smaller at 2×).
- **H-B — ReLU² input.** `mlp.proj` is the only matrix reading the squared-ReLU expansion; predicts `c`
  changes materially when the nonlinearity is swapped.

## Result — `c` is stable across every architectural variation

| architecture | fan-in | `c` (mean, n=2) | per seed |
|:-------------|-------:|----------------:|:---------|
| 2× width, ReLU² | 1536 | **+0.446** | 0.39 / 0.50 |
| **4× width, ReLU² (REQ-051, n=4)** | 3072 | **+0.514** | (reference) |
| 8× width, ReLU² | 6144 | **+0.506** | 0.52 / 0.50 |
| GELU, 4× width | 3072 | **+0.419** | 0.34 / 0.50 |

**Arm 1 (expansion ratio) — H-C refuted.** Across a **4× fan-in range** (1536 → 6144) `c` is flat:
0.446 → 0.514 → 0.506. If the effect were the fan-in shape, `c` would scale with fan-in; it does not.

**Arm 2 (nonlinearity) — H-B not supported.** Replacing ReLU² with GELU leaves `c = +0.419`, only ~18%
below the ReLU² +0.514 and within the n=2 seed scatter (per-seed 0.34/0.50 vs 0.52 baseline). Swapping
the nonlinearity does **not** eliminate the effect, as H-B requires.

**Conclusion: neither surviving hypothesis explains `mlp.proj`'s excess elasticity.** `c` is remarkably
stable (~0.42–0.51) across 2×/4×/8× MLP width and across ReLU²/GELU. Together with the already-refuted
residual-writer hypothesis (`attn.proj` is also a writer with c=−0.171), all three proposed mechanisms
fail. Whatever singles out `mlp.proj` is invariant to its expansion ratio and its input nonlinearity —
the mechanism remains unidentified.

## Caveats

- **n=2 per arm** (vs REQ-051's n=4 baseline) — the binding limitation is the ~0.82 within-matrix
  gradient S/N noted in the request. The GELU−ReLU² gap (−0.10) is within n=2 noise, so arm 2 is "not
  eliminated," not "provably unchanged"; the arm-1 flatness across a 4× fan-in range is the stronger
  refutation. A follow-up at n=4 would tighten both, but the qualitative verdict (c does not track fan-in
  or nonlinearity) is clear.
- Each architecture trains its own step-2000 base (width/nonlinearity change the model), then the 6-level
  per-matrix LR Latin square; `c` estimated with matrix FE over 72 matrices × 6 arms per seed. Curvature
  probe at 32k tokens, top_eigenvalue (iters 8), g = same-minibatch gradient_block_norm. Model patch
  reverted after each arm.

## Files

- `readout.tsv` — `c` per architecture + the two-arm verdict.
- `analyze_req053.py` — reproducible from the raw JSONs + assignments.
- `req053_run.sh` (width arm), `req053_gelu_run.sh` (nonlinearity arm) — drivers (one-line model patch + REQ-051 intervention).
- `configs/req053_{w2,w8,gelu}/assignments_s{0,1}.tsv` — per-(seed,arm,matrix) LR multipliers. `provenance_arm1.tsv` / `provenance_gelu.tsv` — base hashes.
- `raw_json/req053_{w2,w8,gelu}_s{0,1}_arm{0..5}_curv.json` — 36 curvature profiles (source of truth).

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
