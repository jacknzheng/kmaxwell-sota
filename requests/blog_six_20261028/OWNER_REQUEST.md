Hi, the restart-download links have expired (every group returns HTTP 401). Rather than refreshing them, could you restart training from scratch and recompute these points on your side? That avoids moving ~373 GB of checkpoints.

Runs needed (fresh lineages are fine, please label them as such):

| Run | Train to | Measure (16M-token solver contract, unchanged) |
|---|---|---|
| LR ×0.5 arm (baseline to 2009, then LR × 0.5) | 3800 | Every 20 from 2750 to 3250, then check; only if ζ hasn't equilibrated there, every 10 from 3300 to 3500 around H = 3400 |
| Increased-momentum arm (baseline to 2009, then momentum up) | 2200 | Dense every 5 from 2045 to 2095 plus 2125, 2150, 2175 around hypothesis H = 2060, then check |
| Native seed0 MaxEntSlop | 2600 | 1625, 1675, 1725; 1775–2175 every 50; 2570, 2590 (nothing after 2600) |

Check for the two arms: equilibrated from H if more than half of all points at or after H sit at or above the predicted line (existing points count). If the check fails, measure a few sparse points, form a new hypothesis H, sample densely around it and check again. Saving a snapshot every 10 (LR ×0.5, from 2750) or every 5 (momentum, from 2045) during the run means later rounds need no second replay.

The LR ×2 points (2220–2480) and MaxEnt 2675 come from saved snapshots, so they don't need retraining. We're running those ourselves.

Thanks!
