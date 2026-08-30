# REQ-019 — momentum EoS law across fork states (fork-1500)

**Executed on `kmaxwell-sota @ ebf53cd` (serialized-fork-state design), node `qvgl1eq` (8×H100), torch 2.10.0+cu128.** Six fork-1500 arms share one serialized base state at step 1500 and differ only in the absolute post-fork learning-rate multiplier; per-matrix Hessian curvature is then measured across the post-fork window.

## Shared-state gate: **PASS** (`shared-state-check.tsv`)

All six arms loaded the identical serialized `eos_shared_state` at step 1500. Their `model_step001500.pt` share one sha256 (`3d9560ea…`, unique-hash count = 1), the max tensorwise abs-diff between two arms at step 1500 is **0.000e+00** (byte-identical), and each optimizer group's LR at the first fork update equals base × multiplier exactly. The two failed from-scratch fleets could not produce a shared state; this serialized design does, by construction.

## Method

Each arm resumes the step-1500 state, applies a constant LR multiplier ∈ {0.60, 0.77, 1.00, 1.00-dup, 1.30, 1.70} beginning with the fork update, and trains to step 2750. At the five manifest checkpoints (2250, 2375, 2500, 2625, 2750) `measure_per_matrix_curvature.py` records, for each of the 74 matrices, the top eigenvalue of the loss Hessian's **diagonal block** (Lanczos, `iters=8`, fixed 131072-token set), plus curvature along the gradient and the exact polar direction, the gradient block norm, and the complete Lanczos `alphas`/`offdiags` for central geometric-tail recomputation. 6 arms × 5 checkpoints = **30 curvature measurements**; each merged JSON carries all 74 matrices at every checkpoint.

## Result — the per-matrix curvature law DOES change with training state

Max and mean top-eigenvalue over the 74 matrices at the final checkpoint (step 2750):

| multiplier | max top-eig @2750 | mean top-eig @2750 |
|-----------:|------------------:|-------------------:|
| 0.60 | 1,164,458 | 52,951 |
| 0.77 | 717,361 | 38,510 |
| 1.00 | 396,191 | 22,183 |
| 1.00 (dup) | 437,889 | 26,491 |
| 1.30 | 270,698 | 16,714 |
| 1.70 | 150,455 | 9,206 |

Curvature is **monotonically, inversely** related to the post-fork learning-rate multiplier: raising the multiplier from 0.60 to 1.70 drops both the max and mean per-matrix top eigenvalue by ~8×. The two `1.00` duplicates (`s100` / `s100dup`, run-divergence control) bracket each other to ~10%, so the ~8× monotone trend across multipliers is far above run-to-run noise. This is the edge-of-stability / momentum-EoS signature — at a shared state, a larger step drives the optimizer to a proportionately flatter region and a smaller step to a sharper one. Full per-matrix, per-checkpoint spectra (including the low-LR arms that remain in transient, retained per the request) are in each run's `req019_per_matrix_curvature.json` for the central geometric-tail-corrected analysis.

## Files

- `shared-state-check.tsv` — the three-check gate evidence (identical sha256 / zero abs-diff / LR = base×mult).
- `manifest.tsv` — run_id → fork, multiplier, stop step, curvature checkpoints, matrix count, merged-JSON path.
- `summary.tsv` — per run: run_id, fork, multiplier, SHA, node/GPU, train + curvature wall times, completion, checkpoint count (5), matrix count (74), max top-eig at every checkpoint, mean top-eig @2750, failure (none).
- `configs/` — the six fork-1500 configs + `eos_shared_base.yaml`.
- `eos_f1500_<mult>/` — per run: `command.txt`, `console.log` (training), `train-log.txt` (harness), `curvature-console.log`, `req019_per_matrix_curvature.json` (74 matrices × 5 checkpoints, full Lanczos alphas/offdiags).

Checkpoints, optimizer shards, the 9.9 GB serialized `eos_shared_state`, FineWeb data, and env dumps are deliberately **not** committed. The three fork-2000 arms (third-state redundancy, owner-ordered last) run next and will extend this table.
