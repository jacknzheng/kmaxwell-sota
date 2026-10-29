"""Warm full-HVP timings, with numerical qualification reported separately.

Reference computations finish and release their allocator pool BEFORE graph
capture. Reference outputs live on CPU, so graph and eager graphs never coexist.
Every variant uses the identical real checkpoint, tokens, and random directions.
Compilation/capture/setup and steady-state execution are separate measurements.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import torch

from .coupled_muon import FrozenMuonHessian
from .hvp_systems import StagedHessian, relative_error
from .hvp_systems_benchmark import timed
from .local_muon_capture import fixed_validation_batch
from ..harness.model_gpt import GPT


def oracle_error(actual, oracle):
    # Only the small output oracle moves to GPU, never an eager AD graph.
    return relative_error(actual, tuple(e.to(actual[0].device) for e in oracle))


def numerical_gate(row, tolerance=1e-4):
    return all(row[k] <= tolerance for k in ('relative_error','second_direction_error',
        'changed_tokens_error','additivity_error','homogeneity_error'))


def candidate_configuration(variant):
    configurations = {
        'timing_graph': ('cuda_graph', 'highest'),
        'timing_compile': ('compile', 'highest'),
        'timing_tf32': ('eager', 'high'),
        'timing_compile_tf32': ('compile', 'high'),
        'timing_head': ('head', 'highest'),
        'timing_head_compile_tf32': ('head_compile', 'high'),
        'timing_cached_compile_tf32': ('cached_compile', 'high'),
    }
    if variant not in configurations:
        raise ValueError('unknown timing variant')
    return configurations[variant]


def run(args, report, save):
    if args.variant.startswith(('timing_head', 'timing_cached')):
        from .frozen_head_loss import cuda_correctness_report
        report['stage']='head_gpu_correctness';save()
        report['head_gpu_correctness']=cuda_correctness_report(
            'cached' if args.variant.startswith('timing_cached') else 'triton');save()
    checkpoint=torch.load(args.capture/'rank0/boundary_001250.pt',map_location='cpu',weights_only=True)
    model=GPT(50304,12,768).cuda()
    model.load_state_dict({**checkpoint['model_before'],**checkpoint['buffers']},strict=True)
    names=[]
    for rank in range(8):
        state=checkpoint if rank==0 else torch.load(args.capture/f'rank{rank}/boundary_001250.pt',map_location='cpu',weights_only=True)
        names.extend(state['post_insertion_prepolar_momentum'])
    names=sorted(names)
    if len(names)!=72 or len(set(names))!=72:raise ValueError('Unique72 matrix checkpoint required')
    del checkpoint,state;gc.collect()
    params,buffers=dict(model.named_parameters()),dict(model.named_buffers())
    # Both workers use EXACTLY the same token shard, not rank-specific data.
    batch=fixed_validation_batch({'val_data':args.validation},'cuda',131072,0)
    changed=tuple(t.flip(1) for t in batch)
    reference=FrozenMuonHessian(model,params,buffers,names,batch,8,global_tokens=1048576)
    gen=torch.Generator(device='cuda').manual_seed(7621)
    direction=tuple(torch.randn(p.shape,device='cuda',generator=gen)*.001 for p in reference.p)
    other=tuple(torch.randn(p.shape,device='cuda',generator=gen)*.001 for p in reference.p)
    report['stage']='baseline_warmup';save()
    for _ in range(2):
        value=reference(direction);del value
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    report['baseline']=timed(lambda:reference(direction),args.repeats)
    report['stage']='baseline_oracles';save()
    expected=tuple(v.cpu() for v in reference(direction))
    other_oracle=tuple(v.cpu() for v in reference(other))
    reference.batch=changed
    changed_oracle=tuple(v.cpu() for v in reference(direction))
    reference.close();del reference
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    report['before_candidate_allocated_bytes']=torch.cuda.memory_allocated()
    report['before_candidate_reserved_bytes']=torch.cuda.memory_reserved()
    report['stage']='candidate_setup';save()
    mode, precision = candidate_configuration(args.variant)
    torch.set_float32_matmul_precision(precision)
    report['candidate_matmul_precision'] = torch.get_float32_matmul_precision()
    report['candidate_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
    report['candidate_mode'] = mode
    compile_options={'triton.cudagraphs':True} if args.compiled_cuda_graphs else {}
    report['compiled_cuda_graphs_requested']=args.compiled_cuda_graphs
    if mode.startswith(('head', 'cached')):
        from .frozen_head_loss import FrozenHeadFusionGPT
        fast=FrozenHeadFusionGPT(50304,12,768,
            head_backend='cached' if mode.startswith('cached') else 'triton',
            head_chunk_size=args.head_chunk_size).cuda()
        fast.load_state_dict(model.state_dict())
        params,buffers=dict(fast.named_parameters()),dict(fast.named_buffers())
        constructor=StagedHessian if mode.endswith('compile') else FrozenMuonHessian
        extra={'mode':'compile','compile_options':compile_options} if mode.endswith('compile') else {}
        candidate=constructor(fast,params,buffers,names,batch,args.microbatch,global_tokens=1048576,**extra)
    elif mode == 'eager':
        candidate=FrozenMuonHessian(model,params,buffers,names,batch,args.microbatch,global_tokens=1048576)
    else:
        candidate=StagedHessian(model,params,buffers,names,batch,args.microbatch,global_tokens=1048576,
                                mode=mode,compile_options=compile_options)
    tick=time.perf_counter();value=candidate(direction);torch.cuda.synchronize()
    report['setup_and_first_seconds']=time.perf_counter()-tick
    report['first_relative_error']=oracle_error(value,expected);del value
    report['stage']='candidate_warmup';save()
    for _ in range(3):
        value=candidate(direction);del value
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    report['stage']='candidate_timing';save()
    # ALWAYS TIME successful candidates, even if their numerical gate fails.
    report['candidate']=timed(lambda:candidate(direction),args.repeats)
    report['diagnostic_speedup']=report['baseline']['median_seconds']/report['candidate']['median_seconds']
    if args.profile:
        report['stage']='candidate_profile';save()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            value=candidate(direction)
            torch.cuda.synchronize()
            del value
        events=prof.key_averages(group_by_input_shape=True)
        report['profile']=[dict(name=event.key,shapes=event.input_shapes,
            calls=event.count,self_device_us=event.self_device_time_total,
            total_device_us=event.device_time_total)
            for event in sorted(events,key=lambda e:e.self_device_time_total,reverse=True)[:40]]
    report['stage']='candidate_accuracy';save()
    actual=candidate(direction)
    report['relative_error']=oracle_error(actual,expected)
    report['per_matrix_errors']={n:oracle_error((a,),(e,)) for n,a,e in zip(names,actual,expected)}
    del actual
    actual=candidate(other);report['second_direction_error']=oracle_error(actual,other_oracle)
    candidate_other=tuple(v.cpu() for v in actual);del actual
    actual=candidate(direction);candidate_direction=tuple(v.cpu() for v in actual);del actual
    actual=candidate(tuple(v+w for v,w in zip(direction,other)))
    report['additivity_error']=oracle_error(actual,tuple(e+o for e,o in zip(expected,other_oracle)))
    report['candidate_self_additivity_error']=oracle_error(actual,
        tuple(e+o for e,o in zip(candidate_direction,candidate_other)));del actual
    actual=candidate(tuple(2*v for v in direction))
    report['homogeneity_error']=oracle_error(actual,tuple(2*e for e in expected))
    report['candidate_self_homogeneity_error']=oracle_error(actual,tuple(2*e for e in candidate_direction));del actual
    candidate.batch=changed
    actual=candidate(direction);report['changed_tokens_error']=oracle_error(actual,changed_oracle);del actual
    candidate.batch=batch
    report['qualified']=numerical_gate(report)
    report['numerical_screen_passed']=report['qualified']
    report['eigenpair_qualified']=False
    report['qualification_scope']='operator discrepancy diagnostic only; eigenpair convergence not tested here'
    candidate.close()
    torch.set_float32_matmul_precision('highest')


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--validation',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['timing_compile','timing_graph','timing_tf32','timing_compile_tf32',
        'timing_head','timing_head_compile_tf32','timing_cached_compile_tf32'],required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--head-chunk-size',type=int,default=512)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--compiled-cuda-graphs',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if min(args.microbatch,args.head_chunk_size,args.repeats)<1:p.error('Batch, chunk and repeats must be positive')
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    start=time.time()
    report=dict(variant=args.variant,rank=args.rank,stage='loading',torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),precision='FP32/highest',matrix_count=72,
        global_tokens=1048576,local_tokens=131072,local_token_offset=0,microbatch=args.microbatch,
        head_chunk_size=args.head_chunk_size,
        timing_scope='Full72 matrix HVP on one of eight token shards; excludes NCCL and Arnoldi',
        numerical_tolerance=1e-4,production_operator_replaced=False,
        oracle_memory_policy='CPU outputs; no eager reference graph after capture')
    def save():
        report['elapsed_seconds']=time.time()-start
        tmp=args.output/'benchmark.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(args.output/'benchmark.json')
        print(json.dumps(dict(stage=report['stage'],seconds=report['elapsed_seconds'])),flush=True)
    save()
    try:run(args,report,save);report['stage']='complete'
    except Exception as exc:
        chain=[];seen=set();current=exc
        while current is not None and id(current) not in seen:
            seen.add(id(current));chain.append(dict(type=type(current).__name__,message=str(current)))
            current=current.__cause__ or current.__context__
        report.update(stage='failed',error=repr(exc),exception_chain=chain,traceback=traceback.format_exc())
    finally:save()


if __name__=='__main__':main()
