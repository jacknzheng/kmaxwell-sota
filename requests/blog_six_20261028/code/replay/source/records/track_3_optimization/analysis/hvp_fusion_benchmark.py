"""Detailed HVP attribution plus targeted softcap/derivative compilation tests.

Independent GPU shards use the real t1250 checkpoint, all 72 matrices, the same
FP32/highest arithmetic policy and 131072 local tokens. Baseline convergence
study runs unchanged elsewhere. Timings exclude NCCL and Arnoldi.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import torch
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend

from .coupled_muon import FrozenMuonHessian
from .hvp_fusions import SoftcapFusionGPT, ProfileGPT
from .hvp_systems import StagedHessian, relative_error
from .hvp_systems_benchmark import timed
from .local_muon_capture import fixed_validation_batch
from .profile_hvp_trace import summarize_trace


def run(args, report, save):
    checkpoint = torch.load(args.capture/'rank0/boundary_001250.pt',map_location='cpu',weights_only=True)
    model = ProfileGPT(50304,12,768).cuda()
    model.load_state_dict({**checkpoint['model_before'],**checkpoint['buffers']},strict=True)
    names = []
    for rank in range(8):
        state = checkpoint if rank == 0 else torch.load(args.capture/f'rank{rank}/boundary_001250.pt',map_location='cpu',weights_only=True)
        names.extend(state['post_insertion_prepolar_momentum'])
    names = sorted(names)
    if len(names) != 72 or len(set(names)) != 72:
        raise ValueError('Unique 72-matrix checkpoint required')
    del checkpoint,state
    gc.collect()
    params,buffers = dict(model.named_parameters()),dict(model.named_buffers())
    batch = fixed_validation_batch({'val_data':args.validation},'cuda',131072,args.rank*131072)
    reference = FrozenMuonHessian(model,params,buffers,names,batch,8,global_tokens=1048576)
    gen = torch.Generator(device='cuda').manual_seed(7621)
    directions = tuple(torch.randn(p.shape,device='cuda',generator=gen)*.001 for p in reference.p)
    other = tuple(torch.randn(p.shape,device='cuda',generator=gen)*.001 for p in reference.p)
    report.update(stage='reference',matrix_count=len(names)); save()
    expected = reference(directions)
    oracle = reference(other)
    changed = tuple(t.flip(1) for t in batch)
    reference.batch = changed
    changed_oracle = reference(directions)
    reference.batch = batch
    report['reference'] = timed(lambda:reference(directions),5)
    save()
    if args.variant == 'profile_compile':
        from torch.profiler import profile, ProfilerActivity,record_function
        scopes = []
        handles = []
        for name, module in model.named_modules():
            if not name: continue
            def pre(module, inputs, name=name):
                r=record_function('module::'+name); r.__enter__(); scopes.append(r)
            def post(module, inputs, output): scopes.pop().__exit__(None,None,None)
            handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
        micro = FrozenMuonHessian(model,params,buffers,names,tuple(t[:8] for t in batch),8,global_tokens=1048576)
        micro(directions)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            micro(directions); torch.cuda.synchronize()
        path=args.output/'hvp_shapes_trace.json';prof.export_chrome_trace(str(path))
        report['profile'] = summarize_trace(json.loads(path.read_text()))
        for h in handles:h.remove()
        del micro,prof
        gc.collect();torch.cuda.empty_cache()
        modes=[('compile','reverse'),('forward_compile','forward')]
    else:
        modes=[('softcap_eager',False),('softcap_compiled',True)]
    for label, setting in modes:
        row={'stage':'starting'}
        report.setdefault('candidates',{})[label]=row
        report['stage']=label;save()
        candidate = None
        try:
            if args.variant == 'profile_compile':
                candidate = StagedHessian(model,params,buffers,names,batch,8,global_tokens=1048576,
                    mode='compile',backend=setting)
            else:
                fast_model=SoftcapFusionGPT(50304,12,768,compiled=setting).cuda()
                fast_model.load_state_dict(model.state_dict())
                candidate=FrozenMuonHessian(fast_model,dict(fast_model.named_parameters()),dict(fast_model.named_buffers()),
                    names,batch,8,global_tokens=1048576)
            start=time.perf_counter(); actual=candidate(directions);torch.cuda.synchronize()
            row['setup_and_first_seconds']=time.perf_counter()-start
            row['relative_error']=relative_error(actual,expected)
            row['per_matrix_errors']={n:relative_error([a],[e]) for n,a,e in zip(names,actual,expected)}
            del actual
            actual=candidate(other)
            row['second_direction_error']=relative_error(actual,oracle);del actual
            actual=candidate(tuple(v+w for v,w in zip(directions,other)))
            row['additivity_error']=relative_error(actual,tuple(e+o for e,o in zip(expected,oracle)));del actual
            actual=candidate(tuple(2*v for v in directions))
            row['homogeneity_error']=relative_error(actual,tuple(2*e for e in expected));del actual
            candidate.batch=changed
            actual=candidate(directions)
            row['changed_tokens_error']=relative_error(actual,changed_oracle);del actual
            candidate.batch=batch
            row['qualified']=all(row[k]<=1e-4 for k in ('relative_error','second_direction_error',
                'additivity_error','homogeneity_error','changed_tokens_error'))
            # Timing failed-numerics candidates is diagnostic, never a deployment qualification.
            torch.cuda.reset_peak_memory_stats()
            row['timing']=timed(lambda:candidate(directions),5)
            row['diagnostic_speedup']=report['reference']['median_seconds']/row['timing']['median_seconds']
            row['production_operator_replaced']=False
            row['stage']='complete'
        except Exception as exc:
            row.update(stage='failed',qualified=False,error=repr(exc),traceback=traceback.format_exc())
        finally:
            if candidate is not None:candidate.close()
            del candidate
            if args.variant != 'profile_compile' and 'fast_model' in locals():del fast_model
            gc.collect();torch.cuda.empty_cache();save()
    reference.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',required=True)
    p.add_argument('--rank',type=int,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    start=time.time();report=dict(variant=args.variant,rank=args.rank,stage='loading',
        timing_scope='One full-network HVP token shard; excludes NCCL and Arnoldi',
        global_tokens=1048576,local_tokens=131072,precision='FP32/highest',torch=torch.__version__)
    def save():
        report['elapsed_seconds']=time.time()-start
        tmp=args.output/'benchmark.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.output/'benchmark.json')
        print(json.dumps(dict(stage=report['stage'],seconds=report['elapsed_seconds'])),flush=True)
    save()
    try:run(args,report,save);report['stage']='complete'
    except Exception as exc:report.update(stage='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:save()


if __name__=='__main__':main()
