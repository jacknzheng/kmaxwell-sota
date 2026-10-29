# RUN_NOTES — blog-six ζ measurements (saved job)

Run by jerry-agent, session 235ffde4-b726-488a-ba27-e1db4ecc95f5, 2026-10-29. Scope per Jack's in-repo
follow-up: **`setup.sh` → smoke solve → the `saved` job only**; the other jobs (`maxent_1250/1750/2250`,
`higher_zeta`, `lrhalf`) were **held** — the owner was asked to retrain and recompute those points.

## What ran
- Machine: Baseten training workstation, **8 × H200** (job `q8m24yw`), not RunPod. ~22 TB free disk.
- Env: `scripts/setup.sh --env-only` → venv, torch 2.11.0+cu128, numpy 2.4.3 / scipy 1.18.1 / wandb / pyyaml;
  `torch.cuda.device_count() == 8` verified. (python3.12-venv apt-installed on the base image first.)
- Downloads: `download_inputs.py --workers 8` → the one-file links only (~25 GB: saved snapshots, warm-start
  vectors, train shards 16–22, validation shard). **The 373 GB restart step was intentionally skipped** — the
  `saved` job does not use restarts, and those links were 401 (owner recomputing those points anyway).
- Smoke: `run_job.sh saved --only lr2x_2220` → `solver_status: dimension_agreement`.
- `run_job.sh saved` → **12 / 12 points, all `dimension_agreement`.** ~7 min/solve.

## Results (ζ = leading eigenvalue of the coupled Muon response operator; fixed solver contract unchanged)
| lineage | update | ζ |
|---|---|---|
| blog_seed0_lr2x_replay2009_20261027 | 2220 | 0.0033205294 |
| blog_seed0_lr2x_replay2009_20261027 | 2240 | 0.0012930571 |
| blog_seed0_lr2x_replay2009_20261027 | 2280 | 0.0015878317 |
| blog_seed0_lr2x_replay2009_20261027 | 2340 | 0.0014744486 |
| blog_seed0_lr2x_replay2009_20261027 | 2360 | 0.0014039585 |
| blog_seed0_lr2x_replay2009_20261027 | 2380 | 0.0011892919 |
| blog_seed0_lr2x_replay2009_20261027 | 2400 | 0.0011721303 |
| blog_seed0_lr2x_replay2009_20261027 | 2420 | 0.0012973900 |
| blog_seed0_lr2x_replay2009_20261027 | 2440 | 0.0011926205 |
| blog_seed0_lr2x_replay2009_20261027 | 2460 | 0.0010142434 |
| blog_seed0_lr2x_replay2009_20261027 | 2480 | 0.0011133950 |
| original_maxentslop_seed0_replay2500_20261028 | 2675 | 0.0053186816 |

(11 LR×2 + 1 MaxEntSlop = the 12 `saved` points. Per-point `result.json`/`prediction.json`/`solve.log`/
`progress.json`/`convergence.csv` and `points/point_NNNNNN.json` are committed alongside this file.)

## Deviations / notes
- **No solver-contract changes.** 16,777,216 fixed validation tokens on 8 ranks, microbatch 16, warm starts,
  stop on dimension agreement — all as supplied.
- **Excluded from git (not from the run):** each point's 8 `leading_N16777216_o0_rank*.pt` eigenvector shards
  (~340 MB/point, ~4 GB total) and the per-point `wandb/` dirs. These are large derived tensors (the ζ
  eigenvector warm vectors); per the raw-tensor-off-Git policy they stay on the box. The ζ values + solver
  status + convergence traces are all in the committed JSONs. The full `outputs/` (incl. eigenvectors, 7.7 GB)
  is on `q8m24yw` under `/root/blog_six/outputs/` — say the word for a zip/upload before the node is reaped.
- **Held jobs:** `maxent_1250/1750/2250`, `higher_zeta`, `lrhalf` not run (owner recomputing). The replay
  drivers were also flagged in `access/restart-downloads/README.md` as "must not be launched as-is" (machine-
  specific paths/locks/upload destinations) — another reason those were out of scope here.
- Node `q8m24yw` stopped after collection.
