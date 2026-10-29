"""Independent per-GPU tests of exact whole-Muon HVP systems alternatives.

Every production-sized HVP includes all 72 matrices and 131072 local validation
tokens, normalized by 1048576 global tokens. This is one of eight token shards,
NOT an eight-GPU coupled operator timing; distributed validation follows later.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from .coupled_muon import FrozenMuonHessian
from .hvp_systems import StagedHessian, graph_cost_inventory, relative_error
from .hvp_megakernel import mlp_loss, mlp_hvp_megakernel
from .local_muon import frozen_ns_jvp
from .local_muon_capture import fixed_validation_batch
from ..harness.model_gpt import GPT


def timed(call, repeats=3):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        del result
    return dict(seconds=times, median_seconds=statistics.median(times),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7621)
    start = time.time()
    report = dict(variant=args.variant, rank=args.rank, started=start,
                  stage='loading', torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), precision='FP32/highest HVP',
                  global_tokens=1048576, local_tokens=131072,
                  timing_scope='one full-model token shard; excludes NCCL and Arnoldi',
                  production_operator_replaced=False)
    def save():
        report['elapsed_seconds'] = time.time() - start
        tmp = args.output / 'benchmark.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(args.output / 'benchmark.json')
        print(json.dumps({k:report[k] for k in ('variant','stage','elapsed_seconds')}), flush=True)
    save()
    try:
        checkpoint = torch.load(args.capture / 'rank0/boundary_001250.pt',
                                map_location='cpu', weights_only=True)
        model = GPT(50304, 12, 768).cuda()
        model.load_state_dict({**checkpoint['model_before'], **checkpoint['buffers']}, strict=True)
        momenta, owned = {}, {}
        for rank in range(8):
            state = checkpoint if rank == 0 else torch.load(
                args.capture / f'rank{rank}/boundary_001250.pt', map_location='cpu', weights_only=True)
            current = state['post_insertion_prepolar_momentum']
            momenta.update(current)
            if rank == args.rank:
                owned = {n:m.cuda() for n,m in current.items()}
        names = sorted(momenta)
        if len(names) != 72:
            raise ValueError('72 matrix checkpoint required')
        params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        del checkpoint, state, current, momenta
        gc.collect()
        batch = fixed_validation_batch({'val_data':args.validation}, 'cuda',131072, args.rank*131072)
        reference = FrozenMuonHessian(model, params, buffers, names, batch, 8, global_tokens=1048576)
        gen = torch.Generator(device='cuda').manual_seed(7621)
        directions = tuple(torch.randn(p.shape, device='cuda', generator=gen) * .001 for p in reference.p)
        report['matrix_count'] = len(names)
        report['stage'] = 'reference'
        save()
        expected = reference(directions)
        report['reference'] = timed(lambda:reference(directions), args.repeats)
        save()
        if args.variant in ('baseline','micro12'):
            for micro in ([8] if args.variant == 'baseline' else [10,12]):
                candidate = FrozenMuonHessian(model, params, buffers, names, batch, micro, global_tokens=1048576)
                try:
                    torch.cuda.reset_peak_memory_stats()
                    actual = candidate(directions)
                    error = relative_error(actual, expected)
                    del actual
                    measurement = timed(lambda:candidate(directions), args.repeats)
                    measurement.update(microbatch=micro, relative_error=error,
                                       qualified=error <= 1e-4)
                    report.setdefault('candidates', []).append(measurement)
                except torch.cuda.OutOfMemoryError as exc:
                    report.setdefault('candidates', []).append(dict(microbatch=micro, qualified=False, error=str(exc)))
                finally:
                    candidate.close(); del candidate
                    gc.collect(); torch.cuda.empty_cache()
                save()
        elif args.variant in ('trace','compile','graph','forward_compile'):
            mode = {'graph':'cuda_graph','forward_compile':'compile'}.get(args.variant,args.variant)
            candidate = StagedHessian(model, params, buffers, names, batch, 8,
                global_tokens=1048576, mode=mode,
                backend='forward' if args.variant == 'forward_compile' else 'reverse')
            report['stage'] = 'preparing_candidate'
            save()
            tick = time.perf_counter()
            actual = candidate(directions)
            torch.cuda.synchronize()
            report['setup_and_first_seconds'] = time.perf_counter() - tick
            report['error'] = relative_error(actual,expected)
            report['per_matrix_errors'] = {n:relative_error([a],[e]) for n,a,e in zip(names,actual,expected)}
            del actual
            report['stage'] = 'candidate_correctness'
            save()
            other = tuple(torch.randn(p.shape, device='cuda', generator=gen) * .001 for p in reference.p)
            oracle = reference(other)
            actual = candidate(other)
            report['second_direction_error'] = relative_error(actual,oracle)
            combined = candidate(tuple(v+w for v,w in zip(directions,other)))
            report['additivity_error'] = relative_error(combined,tuple(e+o for e,o in zip(expected,oracle)))
            del combined
            del oracle, actual
            scale = candidate(tuple(2*v for v in directions))
            report['homogeneity_error'] = relative_error(scale,tuple(2*e for e in expected))
            del scale
            # Changed runtime tokens: must not replay a frozen validation batch.
            original_batch = candidate.batch
            candidate.batch = tuple(t.flip(1) for t in batch)
            reference.batch = candidate.batch
            actual, oracle = candidate(directions), reference(directions)
            report['changed_tokens_error'] = relative_error(actual, oracle)
            del actual, oracle
            candidate.batch = reference.batch = original_batch
            report['qualified'] = all(report[k] <= 1e-4 for k in
                ('error','second_direction_error','homogeneity_error','additivity_error','changed_tokens_error'))
            if report['qualified']:
                report['stage'] = 'candidate_timing'
                save()
                torch.cuda.reset_peak_memory_stats()
                report['candidate'] = timed(lambda:candidate(directions), args.repeats)
                report['speedup'] = report['reference']['median_seconds'] / report['candidate']['median_seconds']
            report['graph_inventory'] = graph_cost_inventory(candidate.graph_metadata)
            candidate.close()
        elif args.variant == 'ns_compile':
            jvps = {n:frozen_ns_jvp(m) for n,m in sorted(owned.items())}
            h = dict(zip(names,expected))
            compiled = {n:torch.compile(fn, fullgraph=True, dynamic=False,
                         options={'triton.cudagraphs':False}) for n,fn in jvps.items()}
            call = lambda:tuple(fn(h[n]) for n,fn in jvps.items())
            fast = lambda:tuple(fn(h[n]) for n,fn in compiled.items())
            oracle = call()
            tick=time.perf_counter(); actual=fast();torch.cuda.synchronize()
            report['setup_and_first_seconds']=time.perf_counter()-tick
            report['error']=relative_error(actual,oracle)
            report['qualified']=report['error'] <= 1e-4
            report['ns_reference']=timed(call,20)
            if report['qualified']:
                report['ns_candidate']=timed(fast,20)
                report['ns_speedup']=report['ns_reference']['median_seconds']/report['ns_candidate']['median_seconds']
        elif args.variant == 'megakernel_profile':
            from torch.profiler import profile, ProfilerActivity
            micro_reference = FrozenMuonHessian(model, params, buffers, names,
                tuple(t[:8] for t in batch), 8, global_tokens=1048576)
            micro_reference(directions)
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                micro_reference(directions)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output/'hvp_trace.json'))
            events=json.loads((args.output/'hvp_trace.json').read_text())['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e]
            intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
            merged=[]
            for a,b in intervals:
                if merged and a <= merged[-1][1]: merged[-1][1]=max(merged[-1][1],b)
                else: merged.append([a,b])
            busy=sum(b-a for a,b in merged)
            span=merged[-1][1]-merged[0][0] if merged else 0
            report['profile']=dict(kernel_count=len(kernels),gpu_busy_us=busy,gpu_span_us=span,
                gap_fraction=(span-busy)/max(span,1),
                caveat='Profiler overhead affects launch gaps; not a production speedup bound')
            report['megakernel']=[]
            for size in (16,32,64):
                torch.manual_seed(13)
                x=torch.randn(size,size,device='cuda')/size**.5
                w1=torch.randn_like(x).requires_grad_(True)
                w2=torch.randn_like(x).requires_grad_(True)
                v1,v2=torch.randn_like(x),torch.randn_like(x)
                def oracle_call():
                    g=torch.autograd.grad(mlp_loss(x,w1,w2),(w1,w2),create_graph=True)
                    return torch.autograd.grad(g,(w1,w2),grad_outputs=(v1,v2))
                fast=lambda:mlp_hvp_megakernel(x,w1.detach(),w2.detach(),v1,v2)
                oracle,actual=oracle_call(),fast()
                error=relative_error(actual,oracle)
                e_time,f_time=timed(oracle_call,100),timed(fast,100)
                report['megakernel'].append(dict(size=size,error=error,qualified=error<=1e-4,
                    eager=e_time,one_kernel=f_time,speedup=e_time['median_seconds']/f_time['median_seconds'],
                    scope='small two-layer MLP, NOT production GPT'))
                save()
        else:
            raise ValueError('unknown benchmark variant')
        report['stage']='complete'
        reference.close()
    except Exception as exc:
        report.update(stage='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        save()


if __name__ == '__main__':
    main()
