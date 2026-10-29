"""Fresh-process batch sweep: compilation and allocator state never cross trials.

The two supervisor workers test the same token shard on separate H100s. OOM is
a measured capacity result, not grounds to abandon smaller successful batches.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def summarize(rows):
    successful = [row for row in rows if row.get('stage') == 'complete']
    best = min(successful, key=lambda row: row['candidate']['median_seconds']) if successful else None
    return dict(successful_batches=[row['microbatch'] for row in successful],
                fastest_microbatch=best['microbatch'] if best else None,
                fastest_seconds=best['candidate']['median_seconds'] if best else None,
                production_operator_replaced=False, eigenpair_qualified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['timing_compile_tf32','timing_head_compile_tf32'], required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[8,12,16,24,32])
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if not args.batches or any(batch < 1 for batch in args.batches): parser.error('Positive batches required')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []

    def save(stage, current=None):
        report = dict(stage=stage, variant=args.variant, elapsed_seconds=time.time()-started,
                      current_microbatch=current, batches=args.batches, trials=rows, **summarize(rows))
        temporary = args.output/'benchmark.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(args.output/'benchmark.json')

    for batch in args.batches:
        target = args.output/f'batch_{batch}'
        target.mkdir(exist_ok=True)
        save('batch_running', batch)
        # Unique cache per shape/variant; fresh subprocess releases all GPU
        # memory even after an OOM. Non-divisible batches include remainder work.
        env = {**os.environ, 'TORCHINDUCTOR_CACHE_DIR':f'/workspace/hvp_batch_cache/{args.variant}/{batch}'}
        command = [sys.executable,'-m','records.track_3_optimization.analysis.hvp_timing_benchmark',
                   '--capture',args.capture,'--validation',args.validation,'--output',str(target),
                   '--variant',args.variant,'--rank',str(args.rank),'--microbatch',str(batch),
                   '--repeats',str(args.repeats)]
        with (target/'worker.log').open('w') as stream:
            child = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, env=env)
        try:
            row = json.loads((target/'benchmark.json').read_text())
        except (OSError, ValueError):
            row = dict(stage='failed', error='Worker exited without a readable report')
        row.update(microbatch=batch, process_return_code=child.returncode)
        row['out_of_memory'] = 'out of memory' in row.get('error','').lower()
        rows.append(row)
        save('batch_complete', batch)
    save('complete' if summarize(rows)['successful_batches'] else 'failed')


if __name__ == '__main__': main()
