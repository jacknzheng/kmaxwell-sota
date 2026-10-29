"""Compare runtime-first cached head against the fastest unfused configuration.

Trials use fresh processes, identical checkpoint/tokens/directions and the same
full-network timing/accuracy checks. Peak memory is reported, not optimized as
the objective. No production convergence job is modified.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--validation',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['runtime_baseline','runtime_cached'],required=True)
    parser.add_argument('--rank',type=int,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.time();rows=[]
    def save(stage,batch=None):
        successful=[r for r in rows if r.get('stage')=='complete']
        best=min(successful,key=lambda r:r['candidate']['median_seconds']) if successful else None
        result=dict(stage=stage,elapsed_seconds=time.time()-started,current_microbatch=batch,
                    trials=rows,fastest_seconds=best['candidate']['median_seconds'] if best else None,
                    fastest_microbatch=best['microbatch'] if best else None,
                    eigenpair_qualified=False,production_operator_replaced=False)
        temp=args.output/'benchmark.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(args.output/'benchmark.json')
    trials=[16] if args.variant=='runtime_baseline' else [8,16,24]
    variant='timing_compile_tf32' if args.variant=='runtime_baseline' else 'timing_cached_compile_tf32'
    for batch in trials:
        save('runtime_trial',batch)
        target=args.output/f'batch_{batch}';target.mkdir(exist_ok=True)
        command=[sys.executable,'-m','records.track_3_optimization.analysis.hvp_timing_benchmark',
                 '--capture',args.capture,'--validation',args.validation,'--output',str(target),
                 '--variant',variant,'--rank',str(args.rank),'--microbatch',str(batch),
                 '--head-chunk-size','32768','--repeats','7','--profile']
        with (target/'worker.log').open('w') as stream:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'TORCHINDUCTOR_CACHE_DIR':f'/workspace/hvp_runtime_cache/{variant}/{batch}'})
        path=target/'benchmark.json'
        row=json.loads(path.read_text()) if path.exists() else dict(stage='failed',error='No worker report')
        row.update(microbatch=batch,process_return_code=result.returncode)
        rows.append(row);save('runtime_trial_complete',batch)
    save('complete' if any(r.get('stage')=='complete' for r in rows) else 'failed')


if __name__=='__main__':main()
