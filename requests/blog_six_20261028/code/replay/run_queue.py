"""Capture t=1500..1510 inclusive, then warm-start eleven fixed-16M measurements."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import write

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'plan.json').read_text())
# The user revised the interval/token count while the node was provisioning.
# Apply the same explicit revision even if the provisioner had the old plan in memory.
PLAN.update(updates=list(range(1500, 1511)), stop_boundary=1511,
            protocol_revision=2,
            token_protocol=dict(tokens=16777216, eigenvalue_relative_tolerance=.001,
                                independent_start_relative_tolerance=.001,
                                audit_relative_jump=.25,
                                audit_reason='extreme adjacent-checkpoint jump or unresolved solver'))
OUT = Path(PLAN['output'])
DURABLE = OUT.parent
CAPTURE = DURABLE / 'captures'
ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'source') + ':' + str(ROOT),
       'OMP_NUM_THREADS': '2', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
       'WANDB_MODE': 'offline', 'PYTHONUNBUFFERED': '1'}
TORCHRUN = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8']


def status(stage, **kwargs):
    write(OUT / 'queue_status.json', dict(stage=stage, time=time.time(), **kwargs))


def child(argv, log, *, env=ENV, cwd=ROOT):
    with log.open('ab') as stream:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write(OUT / 'active_process.json', dict(pid=p.pid, argv=argv, started=time.time(), log=str(log)))
        code = p.wait()
    if code:
        raise RuntimeError(f'Child exit {code}; see {log}')


def read_result(path, tier):
    progress = json.loads((path / 'progress.json').read_text())
    assert progress['stage'] == 'fixed_probes_complete'
    probe = progress['probes'][-1]
    rows = [s['history'][-1] for s in probe['searches']]
    winner = max(rows, key=lambda r: r['mu'] or float('-inf'))
    return dict(update=progress['update'], zeta=probe['mu'], prediction=PLAN['prediction'],
                relative_gap=probe['mu'] / PLAN['prediction'] - 1, tokens=probe['tokens'], tier=tier,
                dimension=winner['dimension'], residual=winner['residual'],
                dimension_qualified=probe['qualified'], seconds=probe['seconds'],
                precision=progress['precision'], include_weight_decay=progress['include_weight_decay'],
                analysis_gpu=progress.get('gpu'), analysis_microbatch=progress.get('microbatch'),
                output=str(path), solver_status='dimension_agreement' if probe['qualified'] else 'unresolved_dimension',
                independent_start_checked=probe['independent_start_checked'],
                start_relative_spread=probe['seed_relative_spread'],
                lineage='seed0_replay_from_1000')


def measure(t, tier, warm):
    path = OUT / 'measure' / f't{t:06d}_{tier}'
    progress = path / 'progress.json'
    if progress.exists() and json.loads(progress.read_text()).get('stage') == 'fixed_probes_complete':
        return read_result(path, tier)
    # Leave time to sync and release. A partially finished solve remains resumable.
    if time.time() > PLAN['deadline'] - 600:
        raise TimeoutError('Budget deadline; completed checkpoints and partial solver state preserved')
    assert tier in ('16m', '16m_audit')
    tokens = 16777216
    status('measuring', update=t, tier=tier, tokens=tokens, output=str(path))
    path.mkdir(parents=True, exist_ok=True)
    pins = {}
    for line in (CAPTURE / 'index.jsonl').read_text().splitlines():
        entry = json.loads(line)
        if entry.get('kind') == 'boundary' and entry.get('update') == t:
            pins[entry['file']] = entry['sha256']
            assert (CAPTURE / entry['file']).stat().st_size == entry['bytes']
    assert len(pins) == 8
    pins['validation'] = PLAN['validation_sha256']
    manifest = dict(source_head='19724805153e59fe16118399bcc4c3e130ff4424', pins=pins,
                    tokens=tokens, update=t, lineage=PLAN['lineage'])
    if (path / 'input_manifest.json').exists():
        assert json.loads((path / 'input_manifest.json').read_text()) == manifest
    write(path / 'input_manifest.json', manifest)
    if progress.exists() and all((path / f'current_vector_rank{r}.pt').is_file() for r in range(8)):
        warm = path
    argv = TORCHRUN + [str(ROOT / 'single_solve.py'), '--capture', str(CAPTURE),
        '--validation', str(ROOT / 'data/fineweb10B/fineweb_val_000000.bin'),
        '--output', str(path), '--update', str(t), '--token-sizes', str(tokens),
        '--dimensions', '8,16,24,32,48,64,96,128', '--microbatches', str(PLAN['analysis_microbatch']),
        '--fixed-probes', '--hvp-mode', 'compile', '--head-backend', 'cached',
        '--matmul-precision', 'high', '--gemm-precision', 'bf16_fp32']
    if warm is not None and all((Path(warm) / f'current_vector_rank{r}.pt').is_file() for r in range(8)):
        argv += ['--warm-start', str(warm)]
    if progress.exists():
        argv += ['--resume-from', str(path)]
    began = time.time()
    child(argv, OUT / f'solve_{t}_{tier}.log', env={**ENV, 'DENSE_INDEPENDENT': str(int(tier == '16m_audit'))})
    row = read_result(path, tier)
    row['wall_seconds'] = time.time() - began
    write(path / 'queue_result.json', row)
    return row


def publish(rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    path = OUT / 'zeta.csv'
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)
    subprocess.run([sys.executable, str(ROOT / 'report.py'), '--output', str(OUT)], env=ENV, check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write(ROOT / 'plan.json', PLAN)
    write(DURABLE / 'code/plan.json', PLAN)
    write(OUT / 'measurement_protocol.json', PLAN)
    status('waiting_for_inputs')
    while not (ROOT / 'STAGED').exists():
        if (ROOT / 'STAGING_FAILED').exists():
            raise RuntimeError('Input staging failed')
        time.sleep(15)
    digest = hashlib.file_digest((ROOT / 'data/fineweb10B/fineweb_val_000000.bin').open('rb'), 'sha256').hexdigest() if sys.version_info >= (3, 11) else None
    if digest is None:
        h = hashlib.sha256()
        with (ROOT / 'data/fineweb10B/fineweb_val_000000.bin').open('rb') as f:
            for b in iter(lambda: f.read(8388608), b''): h.update(b)
        digest = h.hexdigest()
    assert digest == PLAN['validation_sha256']
    if not (OUT / 'replay_complete.json').exists():
        status('replaying', start=1000, stop=1511, dense_updates=PLAN['updates'])
        all_present = all((CAPTURE / f'rank{r}/boundary_{t:06d}.pt').is_file()
                          for t in PLAN['updates'] for r in range(8))
        if not all_present:
            # Preserve a partial attempt and restart this short replay coherently.
            # Never touch the original saved capture roots.
            if CAPTURE.exists():
                CAPTURE.rename(DURABLE / f'captures_partial_{int(time.time())}')
            child([sys.executable, str(ROOT / 'prepare_training.py')], OUT / 'prepare_training.log')
            child(TORCHRUN + ['--module', 'records.track_3_optimization.run', str(ROOT / 'replay.yaml')],
                  OUT / 'replay.log')
        for t in PLAN['updates']:
            for rank in range(8):
                assert (CAPTURE / f'rank{rank}/boundary_{t:06d}.pt').is_file()
        write(OUT / 'replay_complete.json', dict(updates=PLAN['updates'], time=time.time()))
    if not (OUT / 'replay_comparison.json').exists():
        child([sys.executable, str(ROOT / 'compare_overlap.py')], OUT / 'replay_comparison.log')
    # Explicit queue-level auditing replaces the production default's two starts.
    (OUT / 'measure').mkdir(exist_ok=True)
    write(OUT / 'measure/mu_start_policy.json', dict(format=1, mode='sparse_fresh_checks',
          audit_updates=[], jump_relative_threshold=.99))
    warm = DURABLE / 'warm_original_t1500'
    rows = []
    for t in PLAN['updates']:
        point = OUT / f'point_{t:06d}.json'
        if point.exists():
            result = json.loads(point.read_text())
        else:
            began = time.time()
            result = measure(t, '16m', warm)
            reference = rows[-1]['zeta'] if rows else PLAN['original_zeta_t1500']
            jump = abs(result['zeta'] / reference - 1)
            if (jump > PLAN['token_protocol']['audit_relative_jump'] or
                    not result['dimension_qualified']) and not result['independent_start_checked']:
                first = result
                audit = measure(t, '16m_audit', Path(first['output']))
                spread = abs(first['zeta'] - audit['zeta']) / max(first['zeta'], audit['zeta'])
                result = dict(max((first, audit), key=lambda r: r['zeta']))
                agreed = bool(first['dimension_qualified'] and audit['dimension_qualified'] and spread <= .001)
                result.update(independent_start_checked=True, start_relative_spread=spread,
                    solver_status='independent_start_agreement' if agreed else 'solver_disagreement')
                write(OUT / f'start_comparison_{t}.json', dict(first=first, independent=audit,
                      relative_spread=spread, agreement_tolerance=.001, agreed=agreed, adjacent_relative_jump=jump))
            result['total_point_wall_seconds'] = time.time() - began
            write(point, result)
        rows.append(result)
        warm = Path(result['output'])
        publish(rows)
        status('checkpoint_complete', update=t, zeta=result['zeta'], completed=len(rows), total=len(PLAN['updates']),
               solver_status=result['solver_status'])
    status('complete', completed=len(rows), results=str(OUT / 'zeta.csv'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        status('budget_stop' if isinstance(e, TimeoutError) else 'failed', error=repr(e))
        raise
