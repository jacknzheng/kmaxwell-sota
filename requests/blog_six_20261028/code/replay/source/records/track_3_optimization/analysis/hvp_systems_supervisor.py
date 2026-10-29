"""Run eight independent systems variants, preserve evidence, then terminate.

No lease is created here. Host owner collects small files and releases it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',required=True)
    p.add_argument('--validation',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch-output',type=Path,required=True)
    p.add_argument('--fusion-study',action='store_true')
    p.add_argument('--timing-study',action='store_true')
    p.add_argument('--precision-study',action='store_true')
    p.add_argument('--head-study',action='store_true')
    p.add_argument('--batch-study',action='store_true')
    p.add_argument('--runtime-study',action='store_true')
    p.add_argument('--attention-library',type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    telemetry=(args.output/'telemetry.csv').open('w')
    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits','--loop=5'],stdout=telemetry,stderr=subprocess.STDOUT)
    started=time.time()
    processes=[]
    if sum([args.fusion_study,args.timing_study,args.precision_study,args.head_study,args.batch_study,args.runtime_study])>1:raise ValueError('One study mode required')
    variants=(['runtime_baseline','runtime_cached'] if args.runtime_study else
              ['timing_compile_tf32','timing_head_compile_tf32'] if args.batch_study else
              ['timing_head','timing_head_compile_tf32'] if args.head_study else
              ['timing_tf32','timing_compile_tf32'] if args.precision_study else
              ['timing_compile','timing_graph'] if args.timing_study else ['profile_compile','softcap'] if args.fusion_study else
              ['baseline','micro12','trace','compile','graph','forward_compile','ns_compile','megakernel_profile'])
    import wandb
    run=wandb.init(project='critroll-hvp-systems',mode='offline',dir=str(args.output))
    (args.output/'wandb_dir.txt').write_text(run.dir.rsplit('/files',1)[0]+'\n')
    if args.precision_study and args.attention_library is not None:
        # Complete the small attention compatibility preflight BEFORE any warmed
        # timing starts, so it cannot contaminate the timing GPUs.
        (args.output/'progress.json').write_text(json.dumps(dict(stage='attention_hvp_preflight')))
        try:
            if args.attention_library is None:raise ValueError('precision study requires pinned attention library')
            with (args.output/'attention_hvp.log').open('w') as log:
                subprocess.run([sys.executable,'-m','records.track_3_optimization.analysis.attention_hvp_check',
                    '--library',str(args.attention_library),'--output',str(args.output)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},timeout=600,check=True)
        except Exception as error:
            (args.output/'attention_hvp.json').write_text(json.dumps(dict(stage='preflight_failed',error=repr(error))))
    for rank,variant in enumerate(variants):
        target=args.output/variant;target.mkdir(exist_ok=True)
        stream=(target/'worker.log').open('w')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(rank),'OMP_NUM_THREADS':'2',
             'TORCHINDUCTOR_COMPILE_THREADS':'1','TORCHINDUCTOR_CACHE_DIR':'/workspace/hvp_inductor_cache'}
        module=('records.track_3_optimization.analysis.hvp_runtime_sweep' if args.runtime_study else
                'records.track_3_optimization.analysis.hvp_batch_sweep' if args.batch_study else
                'records.track_3_optimization.analysis.hvp_timing_benchmark' if args.timing_study or args.precision_study or args.head_study else
                'records.track_3_optimization.analysis.hvp_fusion_benchmark' if args.fusion_study else
                'records.track_3_optimization.analysis.hvp_systems_benchmark')
        cmd=[sys.executable,'-m',module,
             '--capture',args.capture,'--validation',args.validation,'--output',str(target),
             '--variant',variant,'--rank',str(rank)]
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env)
        processes.append((variant,child,stream))
    while any(child.poll() is None for _,child,_ in processes):
        rows={}
        for variant,child,_ in processes:
            path=args.output/variant/'benchmark.json'
            try: row=json.loads(path.read_text()) if path.exists() else {'stage':'starting'}
            except json.JSONDecodeError: row={'stage':'writing'}
            rows[variant]=dict(stage=row.get('stage'),seconds=row.get('elapsed_seconds'),return_code=child.poll())
        (args.output/'progress.json').write_text(json.dumps(dict(stage='systems_benchmark',variants=rows),indent=2))
        time.sleep(10)
    for _,child,stream in processes: child.wait();stream.close()
    sampler.terminate();sampler.wait();telemetry.close()
    report={}
    for variant,_,_ in processes:
        path=args.output/variant/'benchmark.json'
        report[variant]=json.loads(path.read_text()) if path.exists() else {'stage':'worker_failed'}
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    for variant,row in report.items():
        if row.get('fastest_seconds') is not None:
            run.log({'variant':variant,'objective':1 / row['fastest_seconds'],
                     'hvp_seconds':row['fastest_seconds']})
        if row.get('speedup') is not None: run.log({'variant':variant,'objective':row['speedup']})
        if row.get('diagnostic_speedup') is not None:
            run.log({'variant':variant,'objective':row['diagnostic_speedup'],
                     'numerically_qualified':int(row.get('qualified',False))})
        for name,candidate in row.get('candidates',{}).items() if isinstance(row.get('candidates'),dict) else []:
            if candidate.get('qualified') and candidate.get('diagnostic_speedup') is not None:
                run.log({'variant':name,'objective':candidate['diagnostic_speedup']})
    run.finish()
    manifest={}
    for path in args.output.iterdir():
        if path.is_file() and path.suffix in ('.json','.csv','.txt','.log'):
            manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'SHA256SUMS.small.json').write_text(json.dumps(manifest,indent=2))
    shutil.copytree(args.output,args.scratch_output,dirs_exist_ok=True)
    terminal=dict(started=started,ended=time.time(),scratch_output=str(args.scratch_output),
                  preservation_complete=True,return_code=0)
    (args.output/'terminal.json').write_text(json.dumps(terminal,indent=2))
    shutil.copy2(args.output/'terminal.json',args.scratch_output/'terminal.json')
    subprocess.run(['sync',str(args.scratch_output)],check=True)


if __name__=='__main__': main()
