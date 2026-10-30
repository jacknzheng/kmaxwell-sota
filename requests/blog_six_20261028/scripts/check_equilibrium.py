#!/usr/bin/env python3
"""Find where an arm has equilibrated: most points clearly at or above the predicted line.
   check_equilibrium.py ARM [START [END]]      ARM = lr2x | lrhalf | higher_zeta

Rule (Jack, 2026-10-30): the onset U is the earliest measured update such that, counting every point from U to END
(at least 6 points), at least 80% have ζ / prediction >= 0.97 (at or above the line; 0.97 allows "at the line").
Existing measurements (jobs/existing_ratios.json) and every new point_*.json under outputs/ and work/ count.
PASS = an onset exists. Before moving on, the arm must also be densely sampled around U: no gap wider than 50 updates
between measured points from U-150 to U+250 (or to the last snapshot). The script lists any such gaps to fill.
FAIL = no onset yet: sample between points where ζ hovers near the line; if most are below or oscillating, go
further along in training, sparse first."""
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
    print(f"{u:5d}  {x:.3f}  {'at/above' if x >= 0.97 else 'below'}  {'new' if u in new else 'existing'}")
onset = None
for i, (u, _) in enumerate(pts):
    tail = [x for _, x in pts[i:]]
    if len(tail) >= 6 and sum(x >= 0.97 for x in tail) >= 0.8 * len(tail):
        onset = u
        break
if onset is None:
    print(f'\nFAIL: no update in {START}-{END if END < 10**9 else "end"} from which >=80% of points are at or above the line (ratio >= 0.97).')
    sys.exit(1)
tail = [x for u, x in pts if u >= onset]
us = [u for u, _ in pts if onset - 150 <= u <= onset + 250]
gaps = [(a, b) for a, b in zip(us, us[1:]) if b - a > 50]
print(f'\nPASS: onset {onset}: {sum(x >= 0.97 for x in tail)} of {len(tail)} points from {onset} on are at or above the line. '
      f"Density {onset - 150}-{onset + 250}: {'OK (no gap > 50)' if not gaps else 'fill gaps ' + ', '.join(f'{a}-{b}' for a, b in gaps)}"
      f"{'' if us and us[-1] >= onset + 250 else f' and extend past {us[-1] if us else onset}'}.")
