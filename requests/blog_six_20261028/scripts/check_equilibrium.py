#!/usr/bin/env python3
"""Check an equilibration hypothesis for higher_zeta or lrhalf.
   check_equilibrium.py JOB H [END]   (H = hypothesised equilibration update; END = optional last update to count)

Equilibrated from H = more than half of all points at or after H sit at or above the predicted line (zeta / prediction >= 1.00).
Existing measurements count. On a fail, the script prints sparse points to measure next (from the snapshots the replay
already saved) so a new hypothesis can be formed; then sample densely around the new H and run this check again."""
import json, pathlib, sys

P = pathlib.Path(__file__).resolve().parent.parent
JOB, H = sys.argv[1], int(sys.argv[2])
END = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
# Flat predicted equilibrium after the 2009 intervention (saved kernel, no fit), existing points as (update, zeta / prediction),
# dense half-width and step, and the sparse step used to look for a new hypothesis.
SPEC = {
    'lrhalf': dict(pred=0.004102903720889337, half=100, step=10, sparse=100,
                   existing=[(2700, 1.134), (2715, 0.936), (2716, 0.873), (2717, 0.889), (2718, 0.956), (3417, 0.894),
                             (3439, 1.061), (3440, 1.26), (3441, 1.162), (3442, 0.942), (3443, 1.156), (3444, 1.162),
                             (3445, 1.004), (3446, 0.891), (3447, 0.835)]),
    'higher_zeta': dict(pred=0.004101621231676908, half=30, step=5, sparse=25,
                        existing=[(2030, 0.997), (2031, 0.975), (2032, 1.088), (2033, 1.072), (2034, 1.022), (2035, 1.21),
                                  (2036, 0.957), (2037, 0.999), (2038, 0.988), (2039, 0.861), (2040, 0.919), (2050, 1.216),
                                  (2060, 0.991), (2070, 1.04), (2080, 1.051), (2090, 1.039), (2100, 0.947), (2200, 1.087)]),
}[JOB]
q = json.load(open(P / f'jobs/{JOB}.json'))
new = {}
for f in (P / f"work/{JOB}/blog_outputs/{q['run_id']}/results").glob('point_*.json'):
    d = json.load(open(f))
    new[d['update']] = d['zeta'] / SPEC['pred']
points = sorted((dict(SPEC['existing']) | new).items())
after = [(u, x) for u, x in points if H <= u <= END]
for u, x in points:
    print(f"{u:5d}  {x:.3f}  {'above' if x >= 1 else 'below'}  {'new' if u in new else 'existing'}{'' if u >= H else '  (before H)'}")
above = sum(x >= 1 for _, x in after)
print(f"\nH = {H}{f' to {END}' if END < 10**9 else ''}: {above} of {len(after)} points are at or above the line")
if above * 2 > len(after):
    print('PASS: equilibrated from H. Send this printout back.')
    sys.exit(0)
sparse = [t for t in q['capture_updates'] if t > H and t % SPEC['sparse'] == 0 and t not in new]
print(f'FAIL: not equilibrated from {H}. Measure these sparse points to form a new hypothesis:')
print(f"  scripts/run_job.sh {JOB} --phase 2 --updates {','.join(map(str, sparse))}")
print(f"Then pick the new H (the first update from which most later points sit at or above the line), measure densely around it\n"
      f"(every {SPEC['step']} from H-{SPEC['half']} to H+{SPEC['half']}) with --phase 2 --updates, and rerun this check with the new H.")
