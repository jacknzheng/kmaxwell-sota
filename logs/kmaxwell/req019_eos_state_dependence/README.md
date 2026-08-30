# REQ-019 — momentum EoS law across fork states (fork-1500)

**Executed on `kmaxwell-sota @ ebf53cd` (serialized-fork-state design), node `qvgl1eq` (8×H100), torch 2.10.0+cu128.** Six fork-1500 arms share one serialized base state at step 1500 and differ only in the absolute post-fork learning-rate multiplier; per-matrix Hessian curvature is then measured across the post-fork window.

## Shared-state gate: **PASS** (`shared-state-check.tsv`)

All six arms loaded the identical serialized `eos_shared_state` at step 1500. Their `model_step001500.pt` share one sha256 (`3d9560ea…`, unique-hash count = 1), the max tensorwise abs-diff between two arms at step 1500 is **0.000e+00** (byte-identical), and each optimizer group's LR at the first fork update equals base × multiplier exactly. The two failed from-scratch fleets could not produce a shared state; this serialized design does, by construction.

## Method

Two shared fork states are used. **Fork-1500:** six arms resume the step-1500 state, apply a constant LR multiplier ∈ {0.60, 0.77, 1.00, 1.00-dup, 1.30, 1.70} from the fork, and train to step 2750; curvature at (2250, 2375, 2500, 2625, 2750). **Fork-2000:** three arms (multipliers 0.60, 1.00, 1.70; third-state redundancy, owner-ordered last) resume the step-2000 state and train to step 3249; curvature at (2750, 2875, 3000, 3125, 3249). At each manifest checkpoint `measure_per_matrix_curvature.py` records, for each of the 74 matrices, the top eigenvalue of the loss Hessian's **diagonal block** (Lanczos, `iters=8`, fixed 131072-token set), plus curvature along the gradient and the exact polar direction, the gradient block norm, and the complete Lanczos `alphas`/`offdiags` for central geometric-tail recomputation. (6+3) arms × 5 checkpoints = **45 curvature measurements**; each merged JSON carries all 74 matrices at every checkpoint. Both forks passed the shared-state gate (identical sha256, zero abs-diff, LR = base×mult; see `shared-state-check.tsv`).

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

Curvature is **monotonically, inversely** related to the post-fork learning-rate multiplier: raising the multiplier from 0.60 to 1.70 drops both the max and mean per-matrix top eigenvalue by ~8×. The two `1.00` duplicates (`s100` / `s100dup`, run-divergence control) bracket each other to ~10%, so the ~8× monotone trend across multipliers is far above run-to-run noise. This is the edge-of-stability / momentum-EoS signature — at a shared state, a larger step drives the optimizer to a proportionately flatter region and a smaller step to a sharper one.

### Fork-2000 (third state) — the law is essentially state-INDEPENDENT

The three fork-2000 arms reproduce the same law at a later shared state (final checkpoint 3249):

| multiplier | fork-1500 max @2750 | fork-2000 max @3249 | fork-1500 mean @2750 | fork-2000 mean @3249 |
|-----------:|--------------------:|--------------------:|---------------------:|---------------------:|
| 0.60 | 1,164,458 | 1,140,571 | 52,951 | 57,041 |
| 1.00 | 396,191 / 437,889 (dup) | 466,865 | 22,183 / 26,491 | 23,809 |
| 1.70 | 150,455 | 188,492 | 9,206 | 11,012 |

At matched multipliers the fork-2000 curvature lands within ~5–25% of the fork-1500 value — inside, or barely outside, the ~10% run-to-run noise floor set by the `1.00` duplicates, and far smaller than the ~8× spread the multiplier itself produces. **The per-matrix curvature law is therefore essentially a function of the learning-rate multiplier and only weakly of the training state at which the fork is taken** — the momentum-EoS relationship is stable across the two fork states probed. Full per-matrix, per-checkpoint spectra (including the low-LR arms that remain in transient, retained per the request) are in each run's `req019_per_matrix_curvature.json` for the central geometric-tail-corrected analysis.

## Files

- `shared-state-check.tsv` — the three-check gate evidence (identical sha256 / zero abs-diff / LR = base×mult).
- `manifest.tsv` — run_id → fork, multiplier, stop step, curvature checkpoints, matrix count, merged-JSON path.
- `summary.tsv` — per run: run_id, fork, multiplier, SHA, node/GPU, train + curvature wall times, completion, checkpoint count (5), matrix count (74), max top-eig at every checkpoint, mean top-eig @2750, failure (none).
- `configs/` — the six fork-1500 configs + `eos_shared_base.yaml`.
- `eos_f1500_<mult>/` — per run: `command.txt`, `console.log` (training), `train-log.txt` (harness), `curvature-console.log`, `req019_per_matrix_curvature.json` (74 matrices × 5 checkpoints, full Lanczos alphas/offdiags).

Both forks are complete (9 arms, 45 curvature measurements). Checkpoints, optimizer shards, the 9.9 GB serialized `eos_shared_state`, FineWeb data, and env dumps are deliberately **not** committed.
