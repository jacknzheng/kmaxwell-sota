"""Durable run/telemetry wrapper. Does not release a lease itself."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from .response_decay import add_decay_flag


def main():
    parser = argparse.ArgumentParser()
    add_decay_flag(parser)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch-output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--pod-expires-at', type=float)
    parser.add_argument('--study-args', nargs=argparse.REMAINDER, default=[],
        help='Explicit study options, e.g. --hvp-mode compile --head-backend cached')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stop_sampling = threading.Event()
    def sample_host():
        with (args.output / 'host_memory.jsonl').open('w') as stream:
            while not stop_sampling.is_set():
                values = {}
                for name in ('memory.current', 'memory.max', 'memory.events',
                             'memory/memory.usage_in_bytes', 'memory/memory.limit_in_bytes',
                             'memory/memory.failcnt', 'memory/memory.oom_control'):
                    path = Path('/sys/fs/cgroup') / name
                    if path.exists():
                        values[name] = path.read_text().strip()
                meminfo = {line.split(':')[0]: line.split(':')[1].strip()
                           for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.split(':')[0] in ('MemTotal','MemAvailable','MemFree','SwapTotal','SwapFree')}
                # Only memory counters; do not publish arbitrary process argv.
                rss = []
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        status = proc.joinpath('status').read_text()
                        if 'python' in status.splitlines()[0]:
                            rss.append(dict(pid=int(proc.name), rss=next(
                                (s.split(':')[1].strip() for s in status.splitlines() if s.startswith('VmRSS:')), None)))
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        pass
                stream.write(json.dumps(dict(time=time.time(), cgroup=values, host=meminfo, python_rss=rss)) + '\n')
                stream.flush()
                stop_sampling.wait(2)
    sampler_host = threading.Thread(target=sample_host, daemon=True)
    sampler_host.start()
    log = (args.output / 'solver.log').open('w')
    telemetry = (args.output / 'telemetry.csv').open('w')
    sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm',
        '--format=csv,noheader,nounits', '--loop=5'], stdout=telemetry, stderr=subprocess.STDOUT)
    started = time.time()
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
        '-m', 'records.track_3_optimization.analysis.coupled_muon_study',
        '--capture', args.capture, '--validation', args.validation, '--output', str(args.output)]
    command += ['--include-weight-decay' if args.include_weight_decay else '--no-include-weight-decay']
    if args.resume_from:
        command += ['--resume-from', str(args.resume_from)]
    if args.pod_expires_at:
        command += ['--pod-expires-at', str(args.pod_expires_at)]
    if any(k in args.study_args for k in ('--output', '--capture', '--validation', '--resume-from')):
        raise ValueError('Study options cannot override supervisor-owned input/output paths')
    command += args.study_args
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    (args.output / 'process.json').write_text(json.dumps(dict(pid=process.pid, started=started, command=command)))
    rc = process.wait()
    stop_sampling.set(); sampler_host.join()
    sampler.terminate()
    sampler.wait()
    log.close(); telemetry.close()
    manifest = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'):
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'SHA256SUMS.small.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Saved Krylov/eigenvector shards are bulk evidence, not EC2 downloads.
    shutil.copytree(args.output, args.scratch_output, dirs_exist_ok=True)
    terminal = dict(return_code=rc, started=started, ended=time.time(),
                    scratch_output=str(args.scratch_output), preservation_complete=True)
    (args.output / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    shutil.copy2(args.output / 'terminal.json', args.scratch_output / 'terminal.json')
    subprocess.run(['sync', str(args.scratch_output)], check=True)
    print(json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
