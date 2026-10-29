"""One sequential small-token search/refinement comparison on a saved checkpoint.

The outer torchrun workers have not imported the model or initialized CUDA yet.
Rank zero launches a separate torchrun; the other workers wait without reserving
GPU memory. After it exits, the original dense queue continues normally.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import write


ROOT = Path(__file__).resolve().parent


def vector_comparison(config, out):
    # CPU only. Angles compare saved numerical directions, not exact eigenspaces.
    import torch
    torch.set_num_threads(2)
    choices = dict(previous=(Path(config['warm']), 'current_vector_rank{rank}.pt'),
        final=(out, 'leading_N16777216_o0_rank{rank}.pt'))
    if config.get('mode', 'small_tokens') == 'small_tokens':
        choices['small'] = (out, 'leading_N1048576_o0_rank{rank}.pt')
    totals = {name: [0., 0., 0.] for name in choices}
    for rank in range(8):
        reference = torch.load(Path(config['baseline']) / f'current_vector_rank{rank}.pt',
                               map_location='cpu', weights_only=True).double()
        reference_norm2 = reference.dot(reference).item()
        for name, (directory, pattern) in choices.items():
            candidate = torch.load(directory / pattern.format(rank=rank),
                                   map_location='cpu', weights_only=True).double()
            totals[name][0] += reference.dot(candidate).item()
            totals[name][1] += reference_norm2
            totals[name][2] += candidate.dot(candidate).item()
    return {name: math.degrees(math.acos(min(1., abs(dot) / math.sqrt(a * b))))
            for name, (dot, a, b) in totals.items()}


def benchmark(config):
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((Path(config['baseline']) / 'progress.json').read_text())
    baseline_probe = baseline['probes'][-1]
    base_wall = json.loads((Path(config['baseline']) / 'queue_result.json').read_text())['wall_seconds']
    plan = json.loads((ROOT / 'plan.json').read_text())
    argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
        str(ROOT / 'single_solve.py'), '--capture', str(out.parents[1] / 'captures'),
        '--validation', str(ROOT / 'data/fineweb10B/fineweb_val_000000.bin'),
        '--output', str(out), '--update', str(config['update']),
        '--token-sizes', ('16777216' if config.get('mode') == 'prefix' else '1048576,16777216'),
        '--dimensions', '8,16,24,32,48,64,96,128',
        '--microbatches', str(plan['analysis_microbatch']), '--fixed-probes', '--hvp-mode', 'compile',
        '--head-backend', 'cached', '--matmul-precision', 'high', '--gemm-precision', 'bf16_fp32',
        '--warm-start', str(config['warm'])]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('TORCHELASTIC_') and k not in
           ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'GROUP_WORLD_SIZE',
            'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')}
    env.update(DENSE_SPEEDUP_BENCHMARK='1', DENSE_INDEPENDENT='0')
    if config.get('mode') == 'prefix':
        env['DENSE_PREFIX_WINDOW'] = str(config.get('window', 9))
    start = time.time()
    write(out / 'benchmark_status.json', dict(stage='running', started=start, argv=argv,
        baseline=config['baseline'], note='Additional benchmark; published dense points are unchanged.'))
    with (out / 'benchmark.log').open('ab') as stream:
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
        write(out / 'active_process.json', dict(pid=p.pid, started=start, argv=argv))
        try:
            code = p.wait(timeout=config['timeout_seconds'])
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    if code:
        raise RuntimeError(f'Speedup benchmark exited {code}')
    candidate = json.loads((out / 'progress.json').read_text())
    assert candidate['stage'] == 'fixed_probes_complete'
    final = candidate['probes'][-1]
    assert [p['tokens'] for p in candidate['probes']] == ([16777216] if config.get('mode') == 'prefix' else [1048576, 16777216])
    assert final['batch_hashes'] == baseline_probe['batch_hashes']
    assert candidate['operator_hashes'] == baseline['operator_hashes']
    assert candidate['precision'] == baseline['precision']
    assert candidate['include_weight_decay'] == baseline['include_weight_decay']
    wall = time.time() - start
    difference = abs(final['mu'] / baseline_probe['mu'] - 1)
    summary = dict(stage='complete', update=config['update'], started=start, completed=time.time(),
        baseline_zeta=baseline_probe['mu'], candidate_zeta=final['mu'], relative_difference=difference,
        baseline_qualified=baseline_probe['qualified'], candidate_qualified=final['qualified'],
        matches_reference_within_0p1_percent=bool(final['qualified'] and baseline_probe['qualified'] and difference <= .001),
        baseline_wall_seconds=base_wall, candidate_wall_seconds=wall, wall_speedup=base_wall / wall,
        baseline_operator_calls=baseline_probe['operator_calls'],
        small_operator_calls=sum(p['operator_calls'] for p in candidate['probes'][:-1]), final_operator_calls=final['operator_calls'],
        baseline_solver_seconds=baseline_probe['seconds'],
        candidate_solver_seconds=sum(p['seconds'] for p in candidate['probes']),
        mode=config.get('mode', 'small_tokens'),
        baseline=config['baseline'], candidate=str(out),
        qualification='One-checkpoint comparison against an empirical baseline; no claim of guaranteed convergence.')
    try:
        summary['acute_angle_to_baseline_degrees'] = vector_comparison(config, out)
    except Exception as error:
        summary['vector_comparison_error'] = repr(error)
    write(out / 'comparison.json', summary)
    write(out / 'benchmark_status.json', summary)
    return summary


def before_solve(argv):
    if os.environ.get('DENSE_SPEEDUP_BENCHMARK') == '1':
        return
    path = ROOT / 'speedup_benchmark_plan.json'
    if not path.is_file() or '--update' not in argv:
        return
    config = json.loads(path.read_text())
    update = int(argv[argv.index('--update') + 1])
    if update != config['before_update']:
        return
    if os.environ.get('DENSE_INDEPENDENT') == '1':
        return
    output = Path(config['output'])
    receipt = output / 'hook_complete.json'
    parent_output = Path(argv[argv.index('--output') + 1])
    rank = int(os.environ['RANK'])
    if rank == 0 and not receipt.is_file():
        began = time.time()
        try:
            result = benchmark(config)
        except Exception as error:
            result = dict(stage='failed', error=repr(error), completed=time.time())
            write(output / 'benchmark_status.json', result)
        # A failed optional benchmark must not stop the requested dense measurements.
        write(parent_output / 'benchmark_overhead.json', dict(seconds=time.time() - began,
              benchmark=str(output), included_in_outer_queue_wall_time=True, result=result['stage']))
        write(receipt, result)
    deadline = time.time() + config['timeout_seconds'] + 180
    while not receipt.is_file():
        if time.time() > deadline:
            raise TimeoutError('Benchmark coordinator failed to release waiting workers')
        time.sleep(2)
