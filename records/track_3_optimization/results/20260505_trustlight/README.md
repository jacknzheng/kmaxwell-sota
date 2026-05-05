
# Bounded Trust Attention-SOAP results

This is a candidate Track 3 optimization record built directly on PR 278:
[`New record: Track_3_optimization: Add SOAP preconditioning to MLPs (3150, -75 steps)`](https://github.com/KellerJordan/modded-nanogpt/pull/278).

## Credits

Starting point and core optimizer: PR 278 by
[@samacqua](https://github.com/samacqua), which added SOAP/Shampoo-style
preconditioning to the MLP matrices before the Contra-NorMuon update. This run
keeps that MLP-SOAP pipeline intact and should be read as a small extension of
that idea, not an independent optimizer family.

Additional inherited components:

- Contra Muon: https://github.com/nilin/contra-muon
- PR 274 NorMuon-lite row/column variance normalization and u/w-floor
- The NanoGPT speedrun / Track 3 training setup in this repository

## Result

Three seeds on 4x H100 all reached the `3.28` target by step 3125. Mean
first target step was 3116.67.

| Seed | 3100 val | 3125 val | First target step | Target val | Train time | Step avg |
| -: | -: | -: | -: | -: | -: | -: |
| 0 | 3.28060 | 3.27880 | 3125 | 3.27880 | 1082.641s | 346.44ms |
| 1 | 3.27941 | - | 3100 | 3.27941 | 1081.587s | 348.90ms |
| 2 | 3.28135 | 3.27947 | 3125 | 3.27947 | 1072.496s | 343.20ms |
| **Mean** | **3.28045** | **3.27914** | **3116.67** | **3.27923** | **1078.908s** | **346.18ms** |

The 3125 mean uses seeds 0 and 2 only, because seed 1 reached target at step
3100 and stopped before evaluating step 3125. Target-step sample standard
deviation is 14.43 steps; target-val sample standard deviation is 0.00037.

Artifacts:

- Seed 0 log: `records/track_3_optimization/20260505_trustlight/runs_seed_0/20260505_170619_seed_0/idea_10e_attnproj_bounded_trust_floor_soapish/seed_0/logs/0996c54e-8e37-4207-96b0-e86550776a64.txt`
- Seed 1 log: `records/track_3_optimization/20260505_trustlight/run_seed_1/idea_10e_attnproj_bounded_trust_floor_soapish/seed_1/logs/1d7d5166-3a25-4fdd-89eb-1d8b35478ee6.txt`
- Seed 2 log: `records/track_3_optimization/20260505_trustlight/run_seed_2/idea_10e_attnproj_bounded_trust_floor_soapish/seed_2/logs/37c85394-091e-4214-965a-a1f1ca4666da.txt`
- Script: `records/track_3_optimization/20260505_trustlight/idea_10e_attnproj_bounded_trust_floor_soapish.py`
- Seed 0 figure: `records/track_3_optimization/20260505_trustlight/runs_seed_0/20260505_170619_seed_0/figure.png`
- Seed 1 figure: `records/track_3_optimization/20260505_trustlight/run_seed_1/figure.png`
- Seed 2 figure: `records/track_3_optimization/20260505_trustlight/run_seed_2/figure.png`

This is still a small seed count compared with the 16-seed Contra Muon record,
but it is now a 3-seed check rather than a single run. The encouraging point is
that all three seeds reach target by 3125 while testing a more controlled
attention-preconditioning variant.

## What changed from PR 278

PR 278 applies SOAP-style preconditioning only to:

- `block.mlp.fc.weight`
- `block.mlp.proj.weight`

Attention matrices continue through the original Contra-NorMuon path.

This experiment keeps PR 278's MLP handling unchanged, then adds a bounded,
trust-gated SOAP contribution only for:

- `block.attn.proj.weight`

The attention projection is the matrix that mixes attention output back into the
residual stream. Full attention SOAP gave a faster early loss drop in prior
idea-10 runs, but it later drifted into a worse trajectory. The hypothesis here
is that attention preconditioning is useful, but full whitening is too much
authority for a nonstationary attention-output distribution.

## Optimizer intuition

For a matrix update `M`, SOAP rotates into a learned row/column basis and divides
by a second-moment estimate:

```text
Z = Q_row.T @ M @ Q_col
Z_soap = Z / sqrt(V)
S = Q_row @ Z_soap @ Q_col.T
```

For MLP matrices, those curvature directions are stable enough that full SOAP is
useful. For attention output projection, the input distribution keeps shifting
because Q/K/V and the token routing pattern keep changing. Late in training,
full whitening can over-amplify low-variance directions that are stale or noisy.

This run replaces full attention whitening with:

```text
M_used = (1 - lambda) * M + lambda * S_safe
```

where `S_safe` is SOAP with a denominator floor, and `lambda` is bounded by both
geometry and schedule:

```text
lambda = clamp(geometry_trust, early_floor(t), early_cap)
```

The geometry trust increases when the SOAP direction agrees with raw momentum
and with the current gradient. The early floor lets attention SOAP contribute
enough to recover some of the fast-start behavior, while the cap prevents full
attention whitening from dominating the trajectory.

## Key hyperparameters

Shared PR 278 / Contra-NorMuon values:

| Parameter | Value |
| - | -: |
| `CONTRA_MUON` | 0.4 |
| operator-normalized subtraction | 0.2 |
| Muon momentum `mu` | 0.95 |
| Muon LR | 0.0375 |
| u/w floor | 0.35 |
| NorMuon beta2 | 0.95 |
| SOAP beta2 | 0.90 |
| SOAP basis refresh | every 10 optimizer steps |
| Train steps | 3175 |

New attention-projection controls:

| Parameter | Value |
| - | -: |
| `ATTN_SOAP_DENOM_FLOOR` | 0.20 |
| `ATTN_EARLY_TRUST_FLOOR` | 0.45 |
| `ATTN_EARLY_TRUST_CAP` | 0.85 |
| `ATTN_TRUST_FLOOR_END_STEP` | 1375 |
| `ATTN_TRUST_FLOOR_FADE_END_STEP` | 1625 |
| `ATTN_TRUST_MIN_AGREE` | 0.20 |
| `ATTN_TRUST_MIN_GRAD_ALIGN` | 0.00 |
| `ATTN_TRUST_POWER` | 1.00 |

## Validation curve

The comparison column uses the PR 278 v18 seed-0 reference curve that the
20260505_trustlight scripts use for exploratory early stopping. The idea10e column is
the mean across seeds 0, 1, and 2 at every common evaluated step. Negative delta
means this run is lower/better at that step.

| Step | Mean val (3 seeds) | PR 278 reference val | Delta |
| -: | -: | -: | -: |
| 125 | 4.50983 | 4.51874 | -0.00891 |
| 250 | 4.05485 | 4.05652 | -0.00167 |
| 375 | 3.89815 | 3.90008 | -0.00193 |
| 500 | 3.81482 | 3.81916 | -0.00434 |
| 625 | 3.76072 | 3.76341 | -0.00269 |
| 750 | 3.72597 | 3.72655 | -0.00058 |
| 875 | 3.69403 | 3.69846 | -0.00443 |
| 1000 | 3.66419 | 3.66557 | -0.00138 |
| 1125 | 3.63745 | 3.63985 | -0.00240 |
| 1250 | 3.60528 | 3.60786 | -0.00258 |
| 1375 | 3.57736 | 3.58225 | -0.00489 |
| 1500 | 3.54735 | 3.54816 | -0.00081 |
| 1625 | 3.52604 | 3.52684 | -0.00080 |
| 1750 | 3.49901 | 3.49950 | -0.00049 |
| 1875 | 3.47509 | 3.47572 | -0.00063 |
| 2000 | 3.45101 | 3.45172 | -0.00071 |
| 2125 | 3.42909 | 3.42949 | -0.00040 |
| 2250 | 3.40709 | 3.40805 | -0.00096 |
| 2375 | 3.38651 | 3.38724 | -0.00073 |
| 2500 | 3.36597 | 3.36632 | -0.00035 |
| 2625 | 3.34471 | 3.34535 | -0.00064 |
| 2750 | 3.32548 | 3.32580 | -0.00032 |
| 2875 | 3.30732 | 3.30741 | -0.00009 |
| 3000 | 3.29105 | 3.29107 | -0.00002 |
| 3100 | 3.28045 | 3.28049 | -0.00004 |

## Reproduction command

```bash
cd /workspace/modded-nanogpt
unset CUDA_VISIBLE_DEVICES

DEVICE=h100 MODE=sequential GPUS_PER_RUN=4 PARALLEL_JOBS=1 \
EXPERIMENTS="idea_10e_attnproj_bounded_trust_floor_soapish.py" \
SEEDS="0 1 2" \
TARGET_VAL_LOSS=3.28 \
CHECK_REQS=1 INSTALL_REQS=0 CHECK_DATA=1 DOWNLOAD_DATA=0 DATA_TOKENS=40 \
EARLY_STOP=1 TRAIN_PROGRESS_INTERVAL=0 \
./records/track_3_optimization/20260505_trustlight/run.sh
```

Because the global batch is fixed by the trainer, running on 4 GPUs changes
wallclock time but should not materially change the step-to-target trajectory
relative to 8 GPUs.





