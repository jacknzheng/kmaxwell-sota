#!/usr/bin/env python3
"""Write owner-format point_NNNNNN.json files from launcher output dirs (result.json + prediction.json).
   make_points.py RUN LINEAGE DEST_DIR OUT_DIR [OUT_DIR ...]"""
import json, sys, pathlib
run, lineage, dest, *outs = sys.argv[1:]
dest = pathlib.Path(dest); dest.mkdir(parents=True, exist_ok=True)
N = 524288
for out in map(pathlib.Path, outs):
    r = json.loads((out / 'result.json').read_text()); p = json.loads((out / 'prediction.json').read_text())
    u, z, lr, wd = r['update'], r['zeta'], p['learning_rate'], p['weight_decay']
    point = dict(
        update=u, zeta=z, prediction=p['prediction'], relative_gap=(z - p['prediction']) / p['prediction'],
        tokens=r['tokens'], tier='16m', dimension=r['dimension'], residual=r['residual'],
        dimension_qualified=r['dimension_qualified'], seconds=r['seconds'], precision=r['precision'],
        include_weight_decay=r['include_weight_decay'], analysis_gpu=r['analysis_gpu'],
        analysis_microbatch=r['analysis_microbatch'], output=r['output'], solver_status=r['solver_status'],
        independent_start_checked=r['independent_start_checked'], start_relative_spread=r['start_relative_spread'],
        lineage=lineage, run=run, seed=0, learning_rate=lr, weight_decay=wd, qualified=r['dimension_qualified'],
        control_audit={k: p[k] for k in ['kappa_star', 'crossing_angle', 'prediction', 'learning_rate', 'weight_decay',
                                         'train_tokens', 'kernel_mass', 'convention']},
        timestamp=u, kappa_star=p['kappa_star'], crossing_angle=p['crossing_angle'],
        kappa_measured=lr * N * (z - wd / N), source_hashes_match=True, publication_role='primary',
        warm_start=r['warm_start'], torch=r['torch'], producer='colleague request 2026-10-29, 8xH200')
    if run in ('lr2x', 'lrhalf', 'higher_zeta', 'lower_zeta'):  # intervention arms are keyed by arm + run_id
        point.update(arm=run, run_id=lineage, run=None)
    (dest / f'point_{u:06d}.json').write_text(json.dumps(point, indent=2))
    print(u, z, point['relative_gap'])
