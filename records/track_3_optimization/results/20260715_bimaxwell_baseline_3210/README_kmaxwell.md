# K-Maxwell Muon (bi-Maxwell generalization)

Fork of `train_gpt_bimaxwell_baseline.py`. Muon momentum becomes a mix of K
log-spaced EMA buffers. Mix weights are a frozen bell-curve score on `log(tau)`,
not random samples. Mean age is derived and logged, not pinned at 30.

The original 3210-step bi-Maxwell record writeup is in `README.md`.

## Stage 0 — identity with the 3210 bi-Maxwell recipe

CPU check (no GPU / no data):

```bash
python3 records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/check_kmaxwell_identity.py
```

Full train, should match the bi-Maxwell seed-0 curve:

```bash
torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py \
  --seed 0 --k 2 --bimaxwell-exact
```

Or: `scripts/kmaxwell_sweep.sh --stage 0`

## Sweeps

From repo root. Logs: `logs/kmaxwell/stageN/`.

| stage | what | default grid |
| --- | --- | --- |
| 0 | exact K=2 identity | 1 run |
| 1 | does K matter? window frozen at [5.67, 49] | K = 2,3,4,6,8,12,16 |
| 2 | more history | tau_max = 25,49,100,200,400 (K=4 unless `--k`) |
| 3 | follow-up: blend width | sigma = 0.4,0.8,1.2,1.6 |
| 4 | follow-up: switch step | start = 0,500,1000 |
| 5 | follow-up: multi-seed | `--seeds 0,1,2` then `0,1,2,3,4,5,6,7` |

```bash
scripts/kmaxwell_sweep.sh --stage 1
scripts/kmaxwell_sweep.sh --stage 2 --k 4
scripts/kmaxwell_sweep.sh --stage 3 --k 4 --tau-max 200
scripts/kmaxwell_sweep.sh --stage 5 --k 4 --tau-max 200 --seeds 0,1,2
```

`--dry-run` prints commands only. `NPROC_PER_NODE` overrides GPU count.
