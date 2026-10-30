#!/usr/bin/env python3
"""Find where an arm has equilibrated: ζ consistently above the predicted line and close to it.
   check_equilibrium.py ARM [START [END]]      ARM = lr2x | lrhalf | higher_zeta

Rule: the onset U is the earliest measured update such that, counting every point from U to END (at least 6 points),
  - at least 80% have ζ / prediction >= 1.00 (consistently above the line), and
  - their median ζ / prediction is <= 1.15 (close to the line).
Existing measurements (jobs/existing_ratios.json) and every new point_*.json under outputs/ and work/ count.
PASS = an onset exists. Then verify it: sample between the points after U (and a little past the last one) and rerun.
FAIL = no onset yet. Look at the printout: sample between points where ζ hovers at or just above the line; if most
points are below or oscillating, sample further along in training."""
import json, pathlib, statistics, sys

P = pathlib.Path(__file__).resolve().parent.parent
ARM = sys.argv[1]
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
END = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
spec = json.load(open(P / 'jobs/existing_ratios.json'))[ARM]
ratio = {u: x for u, x in spec['points']}
new = set()
for f in [*(P / 'outputs').rglob('point_*.json'), *(P / 'work').rglob('point_*.json')]:
    d = json.load(open(f))
    if (d.get('arm') or d.get('run')) == ARM:
        u = int(d.get('update', d.get('timestamp')))
        ratio[u] = d['zeta'] / spec['pred']; new.add(u)
pts = sorted((u, x) for u, x in ratio.items() if START <= u <= END)
for u, x in pts:
    print(f"{u:5d}  {x:.3f}  {'above' if x >= 1 else 'below'}  {'new' if u in new else 'existing'}")
onset = None
for i, (u, _) in enumerate(pts):
    tail = [x for _, x in pts[i:]]
    if len(tail) >= 6 and sum(x >= 1 for x in tail) >= 0.8 * len(tail) and statistics.median(tail) <= 1.15:
        onset = u
        break
if onset is None:
    print(f'\nFAIL: no update in {START}-{END if END < 10**9 else "end"} from which >=80% of points are at or above the line with median <= 1.15.')
    sys.exit(1)
tail = [x for u, x in pts if u >= onset]
print(f'\nPASS: onset {onset}: {sum(x >= 1 for x in tail)} of {len(tail)} points from {onset} on are at or above the line, '
      f'median {statistics.median(tail):.3f}. Verify by sampling between these points and a little beyond the last one.')
