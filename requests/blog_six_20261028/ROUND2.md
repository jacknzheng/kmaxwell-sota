# Round 2 (2026-10-29): two nodes, run continuously, push points every 30 minutes

Use **two 8×H200 nodes** and keep going through every job below without stopping. The only jobs still blocked are the MaxEnt replays (see the end).

`access/downloads.json` has been regenerated with fresh 7-day links (see its `expires_utc`). It now also has train shards 1–15 (at `data/restarts/datasets/fineweb10B/`, where `run_job.sh` reads them) and the shared baseline restart at update 2001 (`data/links/restart/boundary_2001/`). Rerun `python3 -I download_inputs.py --workers 8` on each node; finished files are skipped. Skip `access/restart-downloads/`: those links are dead and no longer needed.

## Node A: saved snapshots, then increased momentum

1. `scripts/run_job.sh saved`: **19 new points** (finished ones are skipped).
   - LR ×2, filling 2200–2500 to every 10: 2210, 2230, 2270, 2290, 2310, 2330, 2350, 2370, 2390, 2410, 2430, 2470, 2490.
   - LR ×2, dense 2650–2700: 2660, 2661, 2662, 2663, 2680, 2700.
   - Points are now written per arm to `outputs/saved/points/<arm>/`.
2. Build `boundary_2009` (below), then `scripts/run_job.sh higher_zeta --phase 1`, then `python3 scripts/check_equilibrium.py higher_zeta 2060`. On a FAIL, run the commands it prints and keep going until a hypothesis passes.

## Node B: LR ×0.5

1. Build `boundary_2009` (below).
2. `scripts/run_job.sh lrhalf --phase 1`: replay to 3800 (snapshots every 10 from 2750) and measure **every 20 from 2750 to 3250** (26 points).
3. `python3 scripts/check_equilibrium.py lrhalf 2750 3250`. On a PASS, it equilibrated in 2750–3250: report the check and stop LR ×0.5 there.
4. **Only on a FAIL:** measure every 10 from 3300 to 3500, `scripts/run_job.sh lrhalf --phase 2 --updates 3300,3310,3320,3330,3340,3350,3360,3370,3380,3390,3400,3410,3420,3430,3450,3460,3470,3480,3490,3500`, then `python3 scripts/check_equilibrium.py lrhalf 3400`. On another FAIL, run the commands it prints and keep going until a hypothesis passes.

## Building boundary_2009 (both nodes; about 8 training updates)

The original 2009 restart links have expired. Rebuild it from the shared baseline restart at 2001:

- Continue the **shared baseline** (original PR323 schedule, no intervention, training microbatch 8) from `data/links/restart/boundary_2001/` for 8 updates, to update 2009. `code/replay/continuous20_train.py` with LR factor 1.0 from that parent is the intended route.
- Save a full restart at 2009 with `COMPLETE.json` = `{"boundary": 2009, "world_size": 8}` and the same rank-file size (11,908,782,730 bytes), at `data/restarts/checkpoints/critroll/pr323_lr_snap_portable_20261019/continuous20/shared/restart/boundary_2009/`. That's where `run_job.sh` links `restart/shared_2009`, and `run_arm.py` checks both.
- This is a fresh realization. The job files' lineages are already named `*_replay2001_20261029`. Note the build in `RUN_NOTES.md`.

## Push every 30 minutes

On each node, every 30 minutes, and again whenever a job finishes:

- commit and push all new `point_*.json` files, `result.json`, `prediction.json` and solve logs under `outputs/` (no `.pt` tensors);
- add or replace one status line under this section in `requests.md`: node, job, points done / remaining, ETA, and any check printout.

## Still blocked: MaxEnt replays

`maxent_1250`, `maxent_1750` and `maxent_2250` need the native MaxEnt restarts `step_01250`, `step_01750` and `step_02250`. No working copy exists, and the owner has been asked to recompute. Don't run them; say so in the status line. Don't measure any MaxEntSlop update after 2600.
