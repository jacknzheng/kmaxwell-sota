# K-Maxwell on the #46 SOTA trainer

Port of the #36 K-Maxwell isolation onto [train_gpt_cwd_SOTA.py](../20260619_cwd_rowfloor_tailema/train_gpt_cwd_SOTA.py).
The #46 artifact is untouched. This is **not** an n=8 record.

K-Maxwell replaces only the hidden-matrix first-moment mix
(`m_eff = Σ w_k m_k`) before SOAP, Newton–Schulz, RowFloor, radial brake, and CWD.
Arch, data, and batch stay Track-3 legal.

PR [#339](https://github.com/KellerJordan/modded-nanogpt/pull/339) already did K=2
(bi-Maxwell) on this stack: **2690 → 2635**. `--k 2 --bimaxwell-exact` recovers that
recipe. A #36 stage-1/2 kernel ranking is a first try here, not a guaranteed SOTA ranking
(SOAP preconditions `m_eff`; the run is shorter).

## Stage 0 — identity with PR #339

CPU check (no GPU / no data):

```bash
python3 records/track_3_optimization/results/20260821_kmaxwell_sota/check_kmaxwell_sota_identity.py
```

Full train (should match the #339 seed-0 curve, up to the denser val grid):

```bash
torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20260821_kmaxwell_sota/train_gpt_kmaxwell_sota.py \
  --seed 0 --k 2 --bimaxwell-exact
```

Or: `scripts/kmaxwell_sota_sweep.sh --stage 0`

Bare `--seed 0` is Gaussian K=2 in the bi-Maxwell window (50/50 weights), **not** the
0.4385/0.5615 #339 kernel.

## Sweeps

From repo root. Logs: `logs/kmaxwell_sota/stageN/`.

| stage | what | default grid |
| --- | --- | --- |
| 0 | exact K=2 identity (PR #339) | 1 run |
| 1 | does K matter? window frozen at [5.67, 49] | K = 2,3,4,6,8,12,16 |
| 2 | more history | tau_max = 25,49,100,200,400 (K=4 unless `--k`) |
| 3 | follow-up: blend width | sigma = 0.4,0.8,1.2,1.6 |
| 4 | follow-up: switch step | start = 0,500,1000 |
| 5 | follow-up: multi-seed | `--seeds 0,1,2` then `0,1,2,3,4,5,6,7` |

```bash
scripts/kmaxwell_sota_sweep.sh --stage 1
scripts/kmaxwell_sota_sweep.sh --stage 2 --k 4
scripts/kmaxwell_sota_sweep.sh --stage 5 --k 4 --tau-max 200 --seeds 0,1,2
```

`--dry-run` prints commands only. `NPROC_PER_NODE` overrides GPU count.
`STOP_STEP` still early-stops a run without changing the schedule.

Val is densified from step 2500 every 5 steps (PR #339 grid) so a win vs 2690 / 2635
is measurable. Confirm a winner with n=8 vs #46 and vs #339; this folder does not
claim a step count.

## Retuning notes

- Kernel filter stats log `mu=0.95`. SOTA warms 0.85→0.95 over 300 steps and cools
  after 2700. K-Maxwell defaults to `--start 1000`, so the mix window is already on
  the 0.95 plateau until the accepted #46 / #339 steps.
- SOAP sees `m_eff` instead of a single EMA.
- `--start 1000` was the #339 peak on this 2690-step run; stage 4 can move it.
- Memory: original momentum buffer plus K stacked hidden-matrix buffers.

## Files

- `train_gpt_kmaxwell_sota.py` — #46 clone + K-Maxwell graft
- `kmaxwell_kernel.py` — torch-free ages / betas / weights
- `kmaxwell_momentum.py` — non-inplace first-moment mix
- `check_kmaxwell_sota_identity.py` — CPU checks vs PR #339 named buffers
- `kmaxwell_sota_sweep.py` — staged launcher
