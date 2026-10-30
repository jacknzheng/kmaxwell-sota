# Blog-six follow-up: new ζ measurements

**Current plan: [`ROUND2.md`](ROUND2.md)** (two nodes, run continuously, push points every 30 minutes). Round 1's `saved` job (12 points) is done; its results are in `outputs/`. Where this file and `ROUND2.md` differ, `ROUND2.md` wins.

## ⚠ Before starting

**One-file links** (`access/downloads.json`) expire **2026-11-05 05:46 UTC**. They cover:

- the saved snapshots
- the warm-start vectors
- train shards 16–22
- the validation shard

**The owner's restart links** (`access/restart-downloads/`) returned HTTP 401 (unauthorized) when tested on 2026-10-28. They cover `boundary_2009`, the MaxEnt restarts and train shards 1–15. They're no longer needed: `access/downloads.json` now has train shards 1–15 and a shared baseline restart at update 2001, from which `boundary_2009` is rebuilt (see `ROUND2.md`). The MaxEnt native restarts are still unavailable.

`OWNER_REQUEST.md` is the note sent to the owner. It asks them to retrain from scratch and recompute the LR ×0.5, increased-momentum and MaxEnt points themselves. If they agree, only the `saved` job is needed here.

## What we are measuring

ζ is the leading eigenvalue of the coupled Muon response operator. It's measured with the fixed solver contract used for every blog point:

- 16,777,216 fixed validation tokens on 8 ranks
- microbatch 16, bf16/fp32
- warm starts, stopping on dimension agreement

The predicted equilibrium comes from the saved momentum kernel, with nothing fitted. A schedule has **equilibrated** from update H when ζ sits at or above the predicted line from H onward: more than half of all points at or after H have ζ ÷ prediction ≥ 1.00.

Every update below is net new: none of them is already measured in any lineage. All new points get merged into each figure's single curve.

Graphs of the plan for all six figures are in `plots/`. PR323 original and decreased momentum get no new points. Each dotted line is one planned point. The red line is the equilibration hypothesis H.

## Sampling approach for LR ×0.5 and increased momentum: hypothesis, dense, check

1. **Hypothesis.** Start from a guess for where ζ equilibrates: LR ×0.5 somewhere in 2750–3250 (fallback H = 3400), increased momentum H = 2060.
2. **Dense.** Measure densely around H (round 1, below).
3. **Check.** `scripts/check_equilibrium.py JOB H` passes if more than half of all points at or after H (existing and new) are at or above the line.
4. **If it fails, form a new hypothesis.** The script prints a few sparse points to measure from the snapshots the replay already saved. Pick the new H as the first update from which most later points sit at or above the line. Then measure densely around the new H and run the check again.

The replay saves a snapshot every 10 updates (LR ×0.5, 2750–3800) or every 5 (momentum, 2045–2195), so later rounds need no new training. Send each check printout back before starting another round.

## Points to measure

| Figure | Phase | Updates | Points | Snapshot source |
|---|---|---|---|---|
| LR ×2 | — | 2220, 2240, 2280, 2340, 2360, 2380, 2400, 2420, 2440, 2460, 2480 | 11 | Saved snapshots, lineage `blog_seed0_lr2x_replay2009_20261027` |
| LR ×0.5 | round 1 · does it equilibrate in 2750–3250? | 2750–3250, every 20 | 26 | New replay from `boundary_2009`, LR factor 0.5 (saves every 10 from 2750 to 3800) |
| LR ×0.5 | only if round 1 fails · dense around H = 3400 | 3300–3500, every 10 (3440 already measured) | 20 | Same replay's saved snapshots |
| LR ×0.5 | only if that fails too | sparse points, then dense around a new H | — | Same replay's saved snapshots |
| Increased momentum | round 1 · dense around H = 2060 | 2045, 2055, …, 2095 (every 5, gaps only), plus 2125, 2150, 2175 | 9 | New replay from `boundary_2009`, momentum-up arm (saves every 5 from 2045 to 2195) |
| Increased momentum | only if the check fails | sparse points, then dense around a new H | — | Same replay's saved snapshots |
| MaxEntSlop | — | 1625, 1675, 1725 | 3 | New replay from native `step_01250` |
| MaxEntSlop | — | 1775, 1825, …, 2175 (every 50) | 9 | New replay from native `step_01750` |
| MaxEntSlop | — | 2570, 2590 (nothing after 2600) | 2 | New replay from native `step_02250` |
| LR ×2 · round 2 | — | 2210, 2230, …, 2490 (fills 2200–2500 to every 10), plus 2660, 2661, 2662, 2663, 2680, 2700 | 19 | Saved snapshots, lineage `blog_seed0_lr2x_replay2009_20261027` |
| **Total still to run** | | | **19 + 9 + 26 runnable now; 14 MaxEnt blocked; more LR ×0.5 / momentum rounds only if a check fails** | |

Round 1's `saved` job (LR ×2 2220–2480 every 20, MaxEntSlop 2675) is done. The three MaxEnt replay jobs stay blocked: the native restarts `step_01250`, `step_01750` and `step_02250` aren't available.

### Why these points

**LR ×0.5.**
- ζ approaches the line from below: the median of ζ ÷ prediction rises from 0.82 to 0.95 between 2010 and 2720.
- The only late points (3417–3447) sit just above the line, at a median of 1.03×.
- Nothing has been measured between 2720 and 3400.
- The timescale is about 1 ÷ (LR × weight decay) = 1 ÷ (0.0125 × 0.05) ≈ 1600 updates, so equilibration should be slow.
- Round 1 checks whether ζ already equilibrates within 2750–3250, the range where nothing is measured: `check_equilibrium.py lrhalf 2750 3250`.
- Only if it doesn't: fallback H = 3400 (6 of the 10 existing points from 3400 on are at or above the line), sampled every 10 from 3300 to 3500.

**Increased momentum.**
- Existing points at 2070 (1.04×), 2080 (1.05×), 2090 (1.04×) and 2200 (1.09×) sit above the line.
- The points at 2060 (0.99×) and 2100 (0.95×) sit just below it.
- Hypothesis H = 2060: 4 of the 6 existing points from 2060 on are above. The every-5 points fill 2045–2095, and 2125/2150/2175 test whether ζ stays above where nothing has been measured.

**LR ×2.** One point every 20 updates from 2220 to 2480, where only a handful of points exist.

**MaxEntSlop.** These fill the gaps between the existing every-50 points from 1600 to 2200, plus 2570 and 2590. No MaxEntSlop point after 2600 is to be measured.

## Jobs

| Job | Restart | Train to | Snapshots | Solves | Lineage written |
|---|---|---|---|---|---|
| `saved` | — | — | downloaded | 12 | original lineages (above) |
| `maxent_1250` | `step_01250` | 1726 | 3 | 3 | `original_maxentslop_seed0_replay1250_density2_20261029` |
| `maxent_1750` | `step_01750` | 2176 | 9 | 9 | `original_maxentslop_seed0_replay1750_density2_20261029` |
| `maxent_2250` | `step_02250` | 2591 | 2 | 2 | `original_maxentslop_seed0_replay2250_density2_20261029` |
| `higher_zeta` | `boundary_2009` (rebuilt from 2001) | 2196 | 25 (~48 GB) | 9, more only if the check fails | `blog_seed0_higher_zeta_eqcheck_replay2001_20261029` |
| `lrhalf` | `boundary_2009` (rebuilt from 2001) | 3801 | 105 (~200 GB) | 26, then 20 only if the check fails | `blog_seed0_lrhalf_eqcheck_replay2001_20261029` |

- Replays use the owner's drivers unchanged: `run_gaps.py` for MaxEnt and `run_arm.py` for the 2009 arms.
- Training microbatch is 8 for the 2009 arms and native for MaxEnt. The longest replay is about 1,800 updates, roughly 4 minutes.
- In a job, the first solve warm-starts from the supplied vector. Each later solve warm-starts from the nearest finished solve.
- Snapshot sizes are about 1.9 GB per update.

## Machine

- 8 × H200, about 1 TB free disk.
- Python 3.12, CUDA 12.8 driver.
- Time: about 7 minutes per solve. About 57 solves is about 6.7 hours, plus about 1 hour of replays, so about 8 hours, or roughly $230 on RunPod community 8×H200. Each extra round after a failed check adds about 7 minutes per point.

## Steps

```bash
scripts/setup.sh                                    # venv + all downloads (resumable; rerun if interrupted)
scripts/run_job.sh saved --only lr2x_2220           # smoke test: one solve, ~7 min
```

Check that `outputs/saved/lr2x_2220/result.json` reports `"solver_status": "dimension_agreement"`. Then:

```bash
scripts/run_job.sh saved
scripts/run_job.sh maxent_1250
scripts/run_job.sh maxent_1750
scripts/run_job.sh maxent_2250
scripts/run_job.sh higher_zeta --phase 1
scripts/run_job.sh lrhalf --phase 1
python3 scripts/check_equilibrium.py higher_zeta 2060   # send both printouts back; wait for confirmation
python3 scripts/check_equilibrium.py lrhalf 2750 3250   # FAIL: run the 3300–3500 round in ROUND2.md, then check lrhalf 3400
# only on a FAIL: the script prints the exact next command, e.g.
# scripts/run_job.sh lrhalf --phase 2 --updates 3600,3700,3800
```

Every command resumes where it stopped if rerun. Finished points are skipped.

## What to send back

`outputs/`, zipped. It contains no snapshots and no restarts. It holds:

- `outputs/saved/<name>/`: `result.json`, `prediction.json`, `solve.log`
- `outputs/saved/points/point_NNNNNN.json`
- `outputs/<job>/<lineage>/results/`: `point_NNNNNN.json`, `measure/…`, solve logs
- `outputs/<job>/<lineage>/`: training log, `training.yaml`, `CAPTURE_COMPLETE.json` and the other receipts

Also include `RUN_NOTES.md`, listing anything that failed or was changed.

## Folder contents

| Path | What |
|---|---|
| `REQUEST.md`, `plots/` | This request and the planned-sampling graphs (six figures) |
| `OWNER_REQUEST.md` | The note to the owner asking for a from-scratch retrain |
| `download_inputs.py`, `access/downloads.json` | One-file 7-day links (160 files, 25.8 GB; size checked, sha256 checked where known) |
| `access/restart-downloads/` | The owner's restart plan and block downloader (needs refreshed links) |
| `jobs/` | `saved.txt`, run_arm queues (`lrhalf.json`, `higher_zeta.json`), run_gaps plans (`maxent_*.json`) |
| `code/replay/` | The owner's patched replay drivers and trainers |
| `code/solver/` | The production solver launcher |
| `scripts/` | `setup.sh`, `run_job.sh`, `check_equilibrium.py`, `make_points.py` |
| `data/links/saved/*/captures/index.jsonl` | Snapshot indexes (the snapshot files themselves download next to them) |

## Constraints

- Don't change the solver contract.
- Don't measure any update not listed here, apart from later rounds once a failed check has been reviewed.
- Keep each replay's lineage name on its points.
- Links are bearer URLs: anyone holding one can download the file, so keep this folder private.
