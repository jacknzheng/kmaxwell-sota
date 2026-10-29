"""Parallel compiled+TF32 graph replay trial with explicit terminal dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True)
    p.add_argument('--validation',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch-output',type=Path,required=True)
    args=p.parse_args();target=args.output/'graph_followup';target.mkdir(exist_ok=True)
    import wandb
    run=wandb.init(project='critroll-hvp-systems',mode='offline',dir=str(target))
    (args.output/'graph_wandb_dir.txt').write_text(run.dir.rsplit('/files',1)[0]+'\n')
    command=[sys.executable,'-m','records.track_3_optimization.analysis.hvp_timing_benchmark',
        '--capture',args.capture,'--validation',args.validation,'--output',str(target),
        '--variant','timing_compile_tf32','--rank','0','--microbatch','8',
        '--repeats','7','--compiled-cuda-graphs']
    with (target/'worker.log').open('w') as stream:
        child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    report=json.loads((target/'benchmark.json').read_text()) if (target/'benchmark.json').exists() else dict(stage='failed',return_code=child.returncode)
    if report.get('candidate'):
        run.log({'objective':1/report['candidate']['median_seconds'],
                 'hvp_seconds':report['candidate']['median_seconds']})
    run.finish()
    shutil.copytree(target,args.scratch_output,dirs_exist_ok=True)
    subprocess.run(['sync',str(args.scratch_output)],check=True)
    result=args.output/'graph_result.json';result.write_text(json.dumps(report,indent=2))
    (args.output/'graph_SHA256SUMS.json').write_text(json.dumps({result.name:hashlib.sha256(result.read_bytes()).hexdigest()}))
    (args.output/'graph_terminal.json').write_text(json.dumps(dict(stage=report.get('stage'),preservation_complete=True)))


if __name__=='__main__':main()
