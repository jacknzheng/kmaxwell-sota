# Round 3 (2026-10-30): keep both nodes busy until each arm has clearly equilibrated

**Keep both 8×H200 nodes running at all times.** Restart Node A now. When a node's own list runs out, it takes the next item from the other node's list. Stop only when every arm below has a verified onset (or you hit a blocker you can't fix; post it).

## What "equilibrated" means now

ζ must sit **consistently above the predicted line and close to it**. `scripts/check_equilibrium.py ARM [START [END]]` finds the onset U: the earliest update from which at least 80% of points (at least 6 of them) are at or above the line, with median ζ ÷ prediction ≤ 1.15. Existing points count; it reads every new `point_*.json` under `outputs/` and `work/`.

By this rule, at the start of round 3:

- `higher_zeta 2030`: **FAIL** (2150 and 2175 sit at 0.88 and 0.87). The round-2 "PASS" used the old majority rule; it no longer counts.
- `lrhalf 2700 3250`: **FAIL** so far (hovering 0.92–1.0 with a few high points).
- `lr2x 2200`: **PASS**, onset 2450. It needs verifying past 2700.

## How to sample (same approach for every arm)

1. **Fill in sparsely first.** Measure points halfway between existing points where ζ hovers near the line. This tells you whether low points were bad luck or real.
2. **If most are still below the line or oscillating, go further along in training**, sparsely, until a stretch looks consistently above.
3. **Then sample densely there to verify**, and run the check after each batch.

## Node A

1. **Increased momentum, in-between points:** 2140, 2160, 2185. If the round-2 `higher_zeta` replay's snapshots are still on disk, use `scripts/run_job.sh higher_zeta --phase 2 --updates 2140,2160,2185` (same lineage). Otherwise run `scripts/run_job.sh higher_zeta_ext --phase 1` (rebuild `boundary_2009` first; snapshots every 5 from 2125 to 2195 and every 10 to 2400), then `scripts/run_job.sh higher_zeta_ext --phase 2 --updates 2140,2160,2185`.
2. Then `python3 scripts/check_equilibrium.py higher_zeta 2030`. If most points from 2100 on are still below or oscillating, measure further along from `higher_zeta_ext`: 2250, 2300, 2350, 2400 first, then densely (every 10) where they're consistently above.
3. **LR ×2, past 2700:** `scripts/run_job.sh saved` runs the 10 new lines (every 40 from 2720 to 3080; finished points are skipped). Then `python3 scripts/check_equilibrium.py lr2x 2200`. Links for every 20 from 2720 to 3100 are in `access/downloads.json`. To sample in between, append lines to `jobs/saved.txt` in the same format.

## Node B

1. **Change of plan: stop the every-20 sampling after 2910** (finish any point already in progress). From 2950 on, sample **every 50 to 3800** to see whether LR ×0.5 is near equilibrium at all: `scripts/run_job.sh lrhalf --phase 2 --updates 2950,3000,3050,3100,3150,3200,3250,3300,3350,3400,3450,3500,3550,3600,3650,3700,3750,3800`. If phase 1 has already exited, this uses its saved snapshots; if it's still running, stop it after the current point first. Then run `python3 scripts/check_equilibrium.py lrhalf 2700`.
2. Sample densely (every 10) **only** where points are consistently above the line and close to it, to verify.
3. If there's still no onset by 3800: `scripts/run_job.sh lrhalf_ext --phase 1` (replay to 4800, snapshots every 10 from 3810), then sparse every 100 from 3900, then dense where consistently above.

## Not in scope

No MaxEntSlop sampling: its replays are still blocked, and nothing after 2600.

## Every 30 minutes

Push new `point_*.json` files and results (no `.pt` tensors), plus one status line per node under the R3 section of `requests.md`: job, points done, latest check result.

## Priority order (Jack, 2026-10-30 07:00Z) — overrides the order above

1. **Increased momentum and LR ×0.5**: one node each, always. Search until each has a verified onset where ζ lies consistently above the line.
2. **MaxEntSlop: 30 new points in its sparsest gaps** (all before 2600): 815, 875, then every 50 from 1025 to 2175 (with 1280 instead of 1275), then 2375, 2425, 2475, 2525. Job file: `jobs/maxent_scratch.json`. No MaxEnt restart exists, so **retrain native MaxEntSlop from scratch**: `train_gpt_muon_maxentslop.py` starts from step 0 (seed 0) when `RESUME_CHECKPOINT` is unset. Use `run_gaps.py` with anchor 0 and without the `input/maxentslop` resume (adapt `run_job.sh`/`run_gaps.py` as needed; note it in `RUN_NOTES.md`), capture these 30 updates, stop at 2526, and measure them. It's a fresh realization; keep its lineage name. Run it on whichever node is waiting (for example during the other arm's replay) or once an arm's onset is verified.
3. ~~LR ×2 past 2700~~: **stopped (Jack, 2026-10-30 15:05Z): no more LR ×2 sampling.**

## Equilibrium rule and switch condition (Jack, 2026-10-30 07:00Z) — overrides the rule above

- **Equilibrated** = from an onset U on, **at least 80% of points are at or above the line** (ζ ÷ prediction ≥ 0.97 counts as "at"), with at least 6 points. The old "median ≤ 1.15" condition is dropped. `scripts/check_equilibrium.py` (updated) applies this, and also reports sampling density around U.
- **Densely plotted** = no gap wider than 50 updates between measured points from U−150 to U+250. Fill any gaps the check lists.
- **A node switches to MaxEntSlop (priority 2) only after its arm PASSes and its density is OK.** Until then it keeps searching: in-between points first; if most are below or oscillating, further along in training (sparse, every 50), extending the replay if snapshots run out (momentum beyond 2400, LR ×0.5 beyond 4800).
- At 07:00Z, with all points so far: increased momentum **FAIL**, LR ×0.5 **FAIL**.
