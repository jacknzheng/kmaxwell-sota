"""Attribute Kineto GPU kernels to ATen operators and recorded tensor shapes.

Durations are summed GPU event times, not exclusive module wall times and not
hardware bandwidth measurements. Parent scope attribution uses CPU containment;
the backward module name is recovered through its forward sequence number.
"""
import argparse
import collections
import json
from pathlib import Path


def summarize_trace(trace):
    events = trace['traceEvents']
    cpu = [e for e in events if e.get('cat') == 'cpu_op']
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and 'dur' in e]
    by_thread = collections.defaultdict(list)
    for e in scopes:
        by_thread[e['tid']].append(e)
    for rows in by_thread.values():
        rows.sort(key=lambda e:e['ts'])
    def scope_of(e):
        candidates = by_thread[e['tid']]
        # A module forward cannot overlap another forward on the same CPU thread.
        enclosing = [s for s in candidates if s['ts'] <= e['ts'] and
                     e['ts'] + e['dur'] <= s['ts'] + s['dur'] + 1]
        return min(enclosing, key=lambda s:s['dur'])['name'] if enclosing else 'unattributed'
    sequence = {}
    for e in cpu:
        a = e.get('args', {})
        if a.get('Fwd thread id', 0) == 0 and a.get('Sequence number') is not None:
            scope = scope_of(e)
            if scope != 'unattributed':
                sequence[(e['tid'], a['Sequence number'])] = scope
    # Backward ATen calls are nested in a named autograd node on worker threads.
    backward = collections.defaultdict(list)
    for e in cpu:
        a = e.get('args', {})
        if a.get('Fwd thread id', 0) and a.get('Sequence number') is not None:
            scope = sequence.get((a['Fwd thread id'], a['Sequence number']))
            if scope:
                backward[e['tid']].append((e['ts'], e['ts']+e['dur'], scope))
    operators = {e.get('args', {}).get('External id'):e for e in cpu}
    grouped = collections.defaultdict(lambda:dict(gpu_us=0., calls=0))
    categories = collections.defaultdict(float)
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
    for k in kernels:
        e = operators.get(k.get('args', {}).get('External id'), {})
        name = e.get('name', 'unknown')
        shape = e.get('args', {}).get('Input Dims', [])
        scope = scope_of(e) if e else 'unattributed'
        if not scope.startswith('module::') and e:
            matches = [(b-a,s) for a,b,s in backward[e['tid']] if a <= e['ts'] <= b]
            if matches:
                scope = min(matches)[1]
        key = (scope, name, json.dumps(shape))
        grouped[key]['gpu_us'] += k['dur']
        grouped[key]['calls'] += 1
        category = 'gemm' if any(s in k['name'].lower() for s in ('gemm','sgemm')) else 'other'
        categories[category] += k['dur']
    total = sum(k['dur'] for k in kernels)
    rows = [dict(scope=s, operator=n, input_shapes=json.loads(sh), **value,
                 fraction=value['gpu_us']/max(total,1)) for (s,n,sh),value in grouped.items()]
    rows.sort(key=lambda r:-r['gpu_us'])
    return dict(kernel_count=len(kernels), summed_gpu_us=total,
                categories={k:dict(gpu_us=v,fraction=v/max(total,1)) for k,v in categories.items()},
                operators=rows,
                caveat='Instrumented microbatch, summed GPU durations; no DRAM/compute roofline counters')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize_trace(json.loads(args.trace.read_text())), indent=2))


if __name__ == '__main__':
    main()
