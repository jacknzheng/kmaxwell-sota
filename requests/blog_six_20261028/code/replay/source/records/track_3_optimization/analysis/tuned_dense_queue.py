"""One-node overnight owner: native dense resume, selected joint mu, posthoc.

Small results live in /persist and are checksum-copyable; raw streams and
restart inputs live on /scratch. The EC2 owner releases on success OR failure.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import yaml

from .tuned_dense_capture import measurement_steps


def atomic(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def analysis_worker(capture, output, settings, rank):
    import torch
    import shutil
    from .autocovariance import analyze
    from .tuned_dense_capture import StreamCapture
    torch.cuda.set_device(rank)
    cap = StreamCapture(capture)
    names = [n for n in sorted(cap.owners) if cap.owners[n] == rank]
    settings = {**settings, 'device': f'cuda:{rank}'}
    output.mkdir(parents=True, exist_ok=False)
    cache = Path('/workspace')/f'tuned_covariance_cache_rank{rank}'
    cache.mkdir(exist_ok=True)
    receipts = []
    for name in names:
        for kind in settings['kinds']:
            # One matrix/stream at a time, staged with a sequential read. Avoid
            # strided mmap I/O on the network volume and unbounded RAM residency.
            source = capture/'raw'/f'{name}.{kind}.npy'
            local = cache/'active.npy'
            if source.stat().st_size + 1024**3 > shutil.disk_usage(cache).free:
                raise RuntimeError('Local disk cannot hold the next exact analysis stream')
            shutil.copyfile(source,local)
            view = StreamCapture(capture)
            view.shapes={name:cap.shapes[name]};view.owners={name:cap.owners[name]}
            view.local_sources[name,kind]=local
            # Boundary readers request all three fields, even for one kind;
            # avoid opening unused streams by serving harmless missing mappings.
            def load(rank_arg, record_kind, step, view=view, kind=kind):
                if record_kind == 'gradient':
                    return view.mapping('gradient',step)
                field={'momentum':'post_insertion_prepolar_momentum',
                       'muon_direction':'polar_output','executed_update':'executed_update'}[kind]
                return {field:view.mapping(kind,step)}
            view.load=load
            unit=output/f'{name}_{kind}';unit.mkdir(exist_ok=False)
            receipt=analyze(view,{**settings,'kinds':[kind]},unit,[name],plots=True)
            receipts.append(receipt)
            for path in unit.iterdir():
                if path.suffix=='.png':
                    shutil.copy2(path,output/path.name)
                elif path.suffix=='.csv':
                    target=output/path.name
                    with target.open('a') as stream,path.open() as source_stream:
                        header=source_stream.readline()
                        if target.stat().st_size==0:stream.write(header)
                        shutil.copyfileobj(source_stream,stream)
            view.arrays.clear()
            del view
            # Only the disposable, exact staged input is removed; original raw
            # streams and per-matrix analysis outputs remain on durable stores.
            local.unlink()
    atomic(output/'receipt.json',dict(complete=True,matrices=names,settings=settings,
        units=len(receipts),input_streams='staged exact copies; original raw retained'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--trainer-dir', type=Path, required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--source-head', required=True)
    parser.add_argument('--analysis-rank', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.analysis_rank is not None:
        analysis_worker(args.capture, args.output, config['analysis'], args.analysis_rank)
        return
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    children, results, runs = [], [], {}
    active = []
    error = None
    phase = 'starting'
    # Terminal metadata must remain reachable even if a child crashes.
    atomic(root/'process.json',dict(pid=os.getpid(),started=started,deadline=args.deadline))
    atomic(root / 'plan.json', dict(config=config, source_head=args.source_head,
        measurement_steps=measurement_steps(), started=started, deadline=args.deadline))
    def progress(**values):
        atomic(root / 'progress.json', dict(stage=phase, started=started,
            elapsed_seconds=time.time()-started, completed_mu=len(results), total_mu=len(measurement_steps()), **values))
    def launch(command, name, env=None):
        if time.time() >= args.deadline:
            raise RuntimeError('User 24-hour/resource deadline reached; preserving partial results')
        log = (root / f'{name}.log').open('x')
        p = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active.append((p, log))
        children.append(dict(name=name, pid=p.pid, command=command, started=time.time()))
        atomic(root / 'children.json', children)
        return p
    def wait(p, child_root=None, name=None):
        while p.poll() is None:
            if time.time() >= args.deadline:
                raise RuntimeError('User 24-hour/resource deadline reached; preserving partial results')
            values = dict(active_pid=p.pid, active_name=name)
            if child_root is not None and (child_root/'progress.json').exists():
                data = json.loads((child_root/'progress.json').read_text())
                values.update(child_progress=data)
            if phase == 'training':
                path = args.capture / 'raw_progress_rank0.json'
                if path.exists():
                    values['raw_progress'] = json.loads(path.read_text())
            progress(**values)
            if child_root is not None and (child_root/'wandb_dir.txt').exists():
                runs[name] = (child_root/'wandb_dir.txt').read_text().strip()
                atomic(root/'runs.json', runs)
            time.sleep(10)
        for proc, log in active:
            if proc is p:
                log.close()
        if p.returncode:
            raise RuntimeError(f'{name} failed, exit={p.returncode}; see durable log')
    def flatten_metrics():
        paths = sorted(args.capture.glob('momentum_restoring_rank*.csv'))
        if not paths:
            return
        with (root/'momentum_restoring_metrics.csv').open('w', newline='') as stream:
            writer = None
            for path in paths:
                with path.open() as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)
    def render_summary():
        import html
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        valid=[r for r in results if r['mu'] is not None]
        if valid:
            fig,ax=plt.subplots(figsize=(10,5))
            ax.plot([r['update'] for r in valid],[r['mu'] for r in valid],'.-')
            ax.set(xlabel='Checkpoint t',ylabel='Joint Muon response mu (8M token mean, decay included)',
                   title='Tuned PR369 trajectory; Adam frozen, all 72 Muon matrices coupled')
            ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'mu_vs_checkpoint.png',dpi=150);plt.close(fig)
        sections=[]
        if (root/'mu_vs_checkpoint.png').exists():
            sections.append('<h2>Joint response</h2><img src="mu_vs_checkpoint.png"><p><a href="mu.csv">mu.csv</a></p>')
        for path in sorted(root.glob('covariance_rank*__*.png')):
            sections.append('<figure><figcaption>'+html.escape(path.name)+'</figcaption><img loading="lazy" src="'+html.escape(path.name)+'"></figure>')
        (root/'index.html').write_text('<!doctype html><meta name="viewport" content="width=device-width"><title>Tuned dense capture</title><style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:12px}img{max-width:100%}figure{margin:30px 0}</style><h1>Tuned Muon dense capture</h1><p>Raw updates 780–3059. Per-matrix centered covariance, windows 128/256/512; no matrix averaging. Mu uses fixed 8M validation tokens, BF16 GEMMs with FP32 accumulation. Residuals are diagnostics. Plots are empirical estimates, not proof of stationarity.</p>'+''.join(sections))
    try:
        import wandb
        run = wandb.init(project='critroll-dense-tuned', mode='offline', dir=str(root), config=config)
        (root/'wandb_dir.txt').write_text(run.dir.rsplit('/files',1)[0]+'\n')
        trainer = args.trainer_dir/'train_gpt_muon_maxentslop.py'
        env = {**os.environ, 'DENSE_CAPTURE_ROOT':str(args.capture),
            'TUNED_TRAINER_SOURCE':str(trainer), 'TUNED_TRAINER_SHA256':digest(trainer),
            'RESUME_CHECKPOINT':config['restart'], 'STOP_STEP':str(config['stop']),
            'OMP_NUM_THREADS':'2', 'TORCHINDUCTOR_COMPILE_THREADS':'1', 'PYTHONUNBUFFERED':'1'}
        for key in ('CHECKPOINT_ROOT', 'CHECKPOINT_STEPS', 'WAYPOINT_SCHEDULE_PATH',
                    'CONSTRAINED_SCHEDULE_PATH', 'TRIAL_BATCH_PATH', 'COMPARE_CHECKPOINT'):
            env.pop(key, None)
        training_command = [str(Path(sys.executable).with_name('torchrun')), '--standalone',
            '--nproc_per_node=8', '--module',
            'records.track_3_optimization.analysis.tuned_dense_capture',
            '--seed',str(config['seed']), '--train_steps',str(config['stop'])]
        # Actual eight-rank resume/observer pilot, on the same intended node.
        # Compare every final model tensor after two updates, not just scalars.
        phase = 'resume_pilot'
        pilot_models = []
        for name in ('reference', 'recorded', 'reference_repeat'):
            reference = name != 'recorded'
            final = root/f'tuned_dense_pilot_{name}.pt'
            pilot_env = {**env, 'DENSE_PILOT':'1', 'STOP_STEP':'782',
                'DENSE_PILOT_REFERENCE':'1' if reference else '0',
                'DENSE_FINAL_MODEL':str(final), 'DENSE_CAPTURE_ROOT':str(root/f'pilot_{name}')}
            p = launch(training_command, f'pilot_{name}', pilot_env)
            wait(p, name=f'resume-pilot-{name}')
            pilot_models.append(final)
        import torch
        expected, actual, repeat = [torch.load(p,map_location='cpu',weights_only=True,mmap=True) for p in pilot_models]
        if expected.keys() != actual.keys() or expected.keys() != repeat.keys():
            raise RuntimeError('Pilot model layouts differ')
        integrity=[]
        for rank in range(8):
            rows=[json.loads(line) for line in (root/f'pilot_recorded/observer_integrity_rank{rank}.jsonl').read_text().splitlines()]
            if len(rows)!=2 or any(row['matrices']!=9 for row in rows):
                raise RuntimeError('Missing actual eight-rank observer integrity evidence')
            integrity.extend(rows)
        comparisons={}
        for n in expected:
            base=expected[n].double()
            comparisons[n]=dict(observer_max_abs=float((actual[n].double()-base).abs().max()),
                reference_repeat_max_abs=float((repeat[n].double()-base).abs().max()),
                observer_bitwise_equal=torch.equal(expected[n],actual[n]),
                reference_repeat_bitwise_equal=torch.equal(expected[n],repeat[n]))
        # Separately compiled distributed runs need not be bitwise identical.
        # The pilot directly verifies that logging mutates NO live tensor or RNG;
        # independent-run differences stay visible alongside baseline variation.
        atomic(root/'resume_pilot.json',dict(start=780,updates=2,world_size=8,
            observer_inputs_and_rng_unchanged=True,integrity_rows=len(integrity),
            per_tensor_run_comparison=comparisons,tensors=len(expected),
            source_sha256=digest(trainer), model_sha256=[digest(p) for p in pilot_models]))
        del expected, actual, repeat
        phase = 'training'
        p = launch(training_command, 'dense_training', env)
        wait(p, name='tuned-dense-training')
        flatten_metrics()
        subprocess.run(['sync',str(args.capture)],check=True)
        atomic(root/'raw_complete.json',dict(start=780, stop=3060, capture=str(args.capture)))
        phase = 'mu'
        warm = None
        for update in measurement_steps():
            output = root/f'mu_t{update:06d}'
            output.mkdir(exist_ok=False)
            pins = {f'rank{r}/boundary_{update:06d}.pt':digest(args.capture/f'rank{r}/boundary_{update:06d}.pt') for r in range(8)}
            pins['validation'] = digest(args.validation)
            atomic(output/'input_manifest.json',dict(source_head=args.source_head,pins=pins))
            command = [str(Path(sys.executable).with_name('torchrun')), '--standalone',
                '--nproc_per_node=8','--module','records.track_3_optimization.analysis.coupled_muon_study',
                '--capture',str(args.capture),'--validation',args.validation,'--output',str(output),
                '--update',str(update),'--token-sizes',str(config['validation_tokens']),
                '--dimensions',','.join(map(str,config['mu']['dimensions'])),
                '--microbatches',str(config['mu']['microbatch']), '--fixed-probes',
                '--hvp-mode',config['mu']['hvp_mode'], '--head-backend',config['mu']['head_backend'],
                '--matmul-precision',config['mu']['matmul_precision'],
                '--gemm-precision',config['mu']['gemm_precision']]
            if warm is not None:
                command += ['--warm-start',str(warm)]
            name = f'tuned-dense-mu-{update}'
            p = launch(command,f'mu_t{update:06d}',env)
            wait(p,output,name)
            data = json.loads((output/'progress.json').read_text())
            if not data.get('fixed_probes_complete') or len(data['probes']) != 1:
                raise RuntimeError(f'Missing requested fixed 8M probe at {update}')
            result = data['probes'][0]
            searches = result['searches']
            record = dict(update=update,tokens=result['tokens'],mu=result['mu'],
                qualified=result['qualified'], residual=max(s['history'][-1]['residual'] or 0 for s in searches),
                seed_relative_spread=result['seed_relative_spread'],
                include_weight_decay=result['include_weight_decay'],
                mu_polar_component=result['mu_polar_component'],weight_decay_shift=result['weight_decay_shift'],
                seconds=result['seconds'],operator_calls=result['operator_calls'],
                operator_seconds=result['operator_seconds'],residual_gating=False)
            results.append(record)
            with (root/'mu.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(record));writer.writeheader();writer.writerows(results)
            atomic(root/'mu_results.json',results)
            if result['mu'] is not None:
                run.log(dict(objective=result['mu'],mu=result['mu'],checkpoint=update,residual=record['residual']))
            if all((output/f'current_vector_rank{r}.pt').exists() for r in range(8)):
                warm=output
            # A failed value-agreement readout is retained, not a gate on later checkpoints.
            progress(latest=record)
            subprocess.run(['sync',str(root)],check=True)
        phase='posthoc'
        import shutil
        from .tuned_dense_capture import StreamCapture
        cap=StreamCapture(args.capture)
        peak=sum(max((args.capture/'raw'/f'{n}.{kind}.npy').stat().st_size
                     for n in cap.shapes if cap.owners[n]==rank
                     for kind in config['analysis']['kinds']) for rank in range(8))
        if peak+8*1024**3 > shutil.disk_usage('/workspace').free:
            raise RuntimeError('Local disk cannot stage eight exact covariance input streams')
        jobs=[]
        for rank in range(8):
            command=[sys.executable,'-m','records.track_3_optimization.analysis.tuned_dense_queue',
                '--output',str(root/f'covariance_rank{rank}'),'--capture',str(args.capture),
                '--trainer-dir',str(args.trainer_dir),'--validation',args.validation,
                '--config',str(args.config),'--deadline',str(args.deadline),
                '--source-head',args.source_head,'--analysis-rank',str(rank)]
            jobs.append(launch(command,f'posthoc_rank{rank}',env))
        for rank,p in enumerate(jobs):
            wait(p,name=f'posthoc-rank{rank}')
        run.finish()
        phase='complete'
        progress()
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),phase=phase)
        atomic(root/'error.json',error)
        phase='failed'
        progress(error=error)
    finally:
        for p,log in active:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        flatten_metrics()
        # Do not copy terabytes of raw streams: they already live durably on scratch.
        # Flatten only small artifacts, with rank prefixes and independent hashes.
        for folder in sorted(root.glob('covariance_rank*')):
            for path in folder.iterdir():
                if path.suffix in ('.csv','.png','.json'):
                    import shutil
                    shutil.copy2(path,root/f'{folder.name}__{path.name}')
        render_summary()
        subprocess.run(['sync',str(args.capture)],check=False)
        atomic(root/'terminal.json',dict(return_code=1 if error else 0,started=started,
            ended=time.time(),completed_mu=len(results),total_mu=len(measurement_steps()),
            raw_capture=str(args.capture),error=error))
        manifest={p.name:digest(p) for p in root.iterdir() if p.is_file() and p.suffix in
                  ('.csv','.json','.log','.txt','.png','.html') and p.name!='SHA256SUMS.small.json'}
        atomic(root/'SHA256SUMS.small.json',manifest)
        subprocess.run(['sync',str(root)],check=True)


if __name__=='__main__':
    main()
