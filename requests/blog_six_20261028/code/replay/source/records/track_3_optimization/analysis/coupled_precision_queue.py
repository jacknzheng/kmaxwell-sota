"""Run checkpoint/precision comparisons with durable progress and preservation.

Variants on one lease are sequential to reuse inputs and seed the next comparison;
independent checkpoint studies can run on separate leases. Each child retains its
own telemetry, wandb log, convergence CSV and eigenvector shards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch-output', type=Path, required=True)
    args = parser.parse_args()
    configs = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    names = [case['name'] for case in configs]
    if len(set(names)) != len(names) or any(Path(n).name != n for n in names):
        raise ValueError('Each comparison needs a unique safe name')
    root = args.output
    results, runs, launched, started = {}, {}, [], time.time()
    try:
        for case in configs:
            name = case['name']
            output = root / name
            output.mkdir(exist_ok=False)
            shutil.copy2(root / 'input_manifest.json', output / 'input_manifest.json')
            command = [sys.executable, '-m', 'records.track_3_optimization.analysis.coupled_muon_supervisor',
                '--capture', args.capture, '--validation', args.validation,
                '--output', str(output), '--scratch-output', str(args.scratch_output / name),
                '--no-include-weight-decay', '--study-args', *case['args']]
            if case.get('warm_from'):
                command += ['--warm-start', str(root / case['warm_from'])]
            log = (root / f'{name}.supervisor.log').open('x')
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            launched.append(dict(name=name, pid=child.pid, command=command))
            atomic_json(root / 'children.json', launched)
            while True:
                wandb = output / 'wandb_dir.txt'
                if wandb.exists():
                    runs[case['run_name']] = wandb.read_text().strip()
                    atomic_json(root / 'runs.json', runs)
                progress = output / 'progress.json'
                if progress.exists():
                    data = json.loads(progress.read_text())
                    atomic_json(root / 'progress.json', {**data, 'comparison_case': name,
                                                        'completed_cases': results})
                rc = child.poll()
                if rc is not None:
                    break
                time.sleep(10)
            log.close()
            data = json.loads(progress.read_text()) if progress.exists() else {}
            results[name] = dict(return_code=rc, progress=data)
            atomic_json(root / 'comparison_results.json', results)
            # Flat immutable snapshots are the checksum-verifiable small copyback.
            for source in output.iterdir():
                if source.is_file() and source.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'):
                    shutil.copy2(source, root / f'{name}__{source.name}')
    except Exception as exc:
        atomic_json(root / 'queue_error.json', dict(type=type(exc).__name__, error=str(exc)))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt')}
    atomic_json(root / 'SHA256SUMS.small.json', manifest)
    # Children already preserve themselves; copy also binds the queue's snapshots.
    shutil.copytree(root, args.scratch_output, dirs_exist_ok=True)
    subprocess.run(['sync', str(args.scratch_output)], check=True)
    terminal = dict(started=started, ended=time.time(), cases=names,
        return_code=0 if len(results) == len(names) and all(r['return_code'] == 0 for r in results.values()) else 1,
        preservation_complete=True, scratch_output=str(args.scratch_output))
    atomic_json(root / 'terminal.json', terminal)
    shutil.copy2(root / 'terminal.json', args.scratch_output / 'terminal.json')
    subprocess.run(['sync', str(args.scratch_output)], check=True)


if __name__ == '__main__':
    main()
