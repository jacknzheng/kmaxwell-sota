# REQ-019 — seed-twins paired noise fleet (n=4)

**Authorized 4-seed paired fleet of the three endpoint runs, executed on `kmaxwell-sota @ ebf53cd`, 8×H100 (box `q4glryq`, now released), torch 2.10.0+cu128.** This estimates run-to-run noise for the current decomposition; it is **NOT** the official n=8 Track-3 significance test.

## What ran

For each seed 0,1,2,3: one `pl_anneal_base` (to step 1001, dumps `warmstart_pl_anneal` = 1 model + 8 optimizer shards at step 1000) followed by three arms that each resume that serialized state at step 1000 (`skipped 1000 batches`) and run to step 3250:

1. `plann_bimaxwell_control` — the exact bi-Maxwell control (**reference**)
2. `expann_b0p982_b0p944` — scheduled single EMA, decay 0.982→0.944 from step 1000→3249
3. `plann_pr357_kernel_control` — the exact PR357 K8 kernel control

All 16 runs completed cleanly (12 ARM_EXIT=0, 4 BASE_EXIT=0). Each seed ran in an **isolated working directory** (`/root/twins_seed_<n>`, code copy + shared-data symlink) so the four `warmstart_pl_anneal` state dirs cannot collide; configs were produced by the frozen generators verbatim (`make_powerlaw_anneal_configs.py` + `make_exponential_anneal_configs.py --start 0.982 --end 0.944`) with `seed=<n>` applied as a top-level CLI override. Before each seed's arms, the step-1000 model + all 8 optimizer shards were verified present.

## Paired result (diff = bi-Maxwell val − candidate val; positive ⇒ candidate has lower loss)

Final val_loss @ step 3250, all four seeds:

| seed | bi-Maxwell | expann 0.982→0.944 | pr357 K8 |
|-----:|-----------:|-------------------:|---------:|
| 0 | 3.27511 | 3.27497 | 3.27228 |
| 1 | 3.27524 | 3.27482 | 3.27250 |
| 2 | 3.27579 | 3.27544 | 3.27289 |
| 3 | 3.27625 | 3.27602 | 3.27365 |

**Paired final diffs (bi-Maxwell − candidate), n=4:**

| candidate | seed0 | seed1 | seed2 | seed3 | mean | std | stderr |
|-----------|------:|------:|------:|------:|-----:|----:|-------:|
| expann 0.982→0.944 | +0.000140 | +0.000420 | +0.000350 | +0.000230 | **+0.000285** | 0.000124 | 0.000062 |
| pr357 K8 | +0.002830 | +0.002740 | +0.002900 | +0.002600 | **+0.002768** | 0.000129 | 0.000065 |

Both candidates beat the bi-Maxwell control on **every** seed. The PR357 K8 kernel is ahead by ~0.0028 (mean/se ≈ 43), a large and consistent margin; the exponential-anneal single EMA is marginally ahead by ~0.0003 (mean/se ≈ 4.6). The dense-cooldown-window (steps 2900–3250) mean paired diffs are reported per seed and aggregated in `summary.tsv` and track the final-step ordering.

Caveat (owner's instruction): this n=4 fleet estimates run-to-run noise for the current decomposition; it must **not** be presented as the official n=8 Track-3 significance test.

## Files

- `summary.tsv` — per-seed final val_loss for all three arms; per-seed and aggregated (mean/std/stderr) paired final diffs; dense-cooldown-window mean diffs.
- `paired_trajectories.tsv` — `candidate, step, seed, bimaxwell_val, candidate_val, diff` at all 59 shared validation checkpoints.
- `configs/` — the generated fork configs (base + 3 arms; seed-invariant except the `seed=` CLI override).
- `seed_<n>/command.txt` — exact generate + torchrun commands for that seed.
- `seed_<n>/console.log` — concatenated base + 3-arm stdout (val_loss trajectories, resume confirmation).
- `seed_<n>/train-logs/` — the harness's own per-run train logs.

Checkpoints, optimizer shards, FineWeb data, and env dumps are deliberately **not** committed.
