"""Read-only instrumentation of the pinned PR369 trainer and its native resume.

Do not translate its EMA/history state into a different optimizer. AST insertion
only brackets the actual training step and observes its post-insertion momentum.
Raw per-matrix streams avoid rereading whole-model files for every matrix.
"""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from ..harness.capture_store import CaptureWriter
from ..harness.capture_hooks import momentum_restoring_metrics


def measurement_steps():
    return sorted(set([780, 850, 900, 916, 925, 950, 975, 1000, 1050, 1100,
        1150, 1200, *range(1250, 1260), *range(1350, 2751, 100),
        *range(2800, 2810), 2850, 2900, 2925, 2950, 2975, 3000, 3010,
        3020, 3030, 3040, 3050, 3059]))


def instrument(source):
    tree = ast.parse(source)
    counts = dict(setup=0, begin=0, observe=0, end=0)
    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if any(isinstance(t, ast.Name) and t.id == 'step_avg' for t in node.targets):
                # Upstream restart logging divides by zero at start_step>0.
                # Only the timing guard changes, never training math/state.
                if isinstance(node.value, ast.IfExp):
                    node.value.test = ast.parse('step > last_val_step', mode='eval').body
            if any(isinstance(t, ast.Name) and t.id == 'train_loader' for t in node.targets):
                counts['setup'] += 1
                return ast.parse('_dense_recorder = DenseRecorder(model, optimizer2, batch_size, start_step, stop_step)').body + [node]
            if (any(isinstance(t, ast.Name) and t.id == 'update' for t in node.targets)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'muon_update_maxentslop'):
                counts['observe'] += 1
                return [node] + ast.parse('_dense_recorder.observe(p, p.grad, momentum_direction.view_as(p), update, momentum_kernel, self._step, group, zeropower_via_newtonschulz5)').body
            return node
        def visit_Expr(self, node):
            self.generic_visit(node)
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'set_hparams':
                counts['begin'] += 1
                return [node] + ast.parse('_dense_recorder.begin(step)').body
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'model'
                    and call.func.attr == 'zero_grad'):
                counts['end'] += 1
                return ast.parse('_dense_recorder.end(step)').body + [node]
            return node
    tree = Insert().visit(tree)
    if counts != dict(setup=1, begin=1, observe=1, end=1):
        raise ValueError(f'Pinned trainer instrumentation sites changed: {counts}')
    return ast.fix_missing_locations(tree)


def base_model(model):
    return getattr(model, '_orig_mod', model)


class DenseRecorder:
    def __init__(self, model, optimizer, batch_tokens, start, stop):
        allowed_stop = 782 if os.environ.get('DENSE_PILOT') == '1' else 3060
        if start != 780 or stop != allowed_stop or dist.get_world_size() != 8:
            raise ValueError('This capture requires tuned seed0 checkpoint780, stop3060, eight ranks')
        self.model = base_model(model)
        self.optimizer = optimizer
        self.root = Path(os.environ['DENSE_CAPTURE_ROOT'])
        self.rank = dist.get_rank()
        self.writer = CaptureWriter(self.root, self.rank)
        self.start, self.stop, self.tokens = start, stop, batch_tokens
        self.names = {id(p): n for n, p in self.model.named_parameters()}
        self.params = dict(self.model.named_parameters())
        ordered = optimizer.param_groups[0]['params']
        self.owned = [p for i, p in enumerate(ordered) if i % 8 == self.rank]
        self.steps = measurement_steps()
        self.streams = {}
        layout = {self.names[id(p)]: dict(shape=list(p.shape), owner=self.rank) for p in self.owned}
        all_layout = [None] * 8
        dist.all_gather_object(all_layout, layout)
        layout = {k: v for group in all_layout for k, v in group.items()}
        if len(layout) != 72:
            raise ValueError('Dense capture must contain all 72 Muon matrices')
        raw = self.root / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        for p in self.owned:
            name = self.names[id(p)]
            for kind in ('gradient', 'momentum', 'muon_direction', 'executed_update'):
                path = raw / f'{name}.{kind}.npy'
                if path.exists():
                    raise ValueError('Refuse to overwrite captured raw streams')
                self.streams[name, kind] = np.lib.format.open_memmap(path, mode='w+',
                    dtype=np.uint16 if kind == 'muon_direction' else np.float32,
                    shape=(stop-start, *p.shape))
        self.scalar_path = self.root / f'momentum_restoring_rank{self.rank}.csv'
        self.scalar_file = self.scalar_path.open('x', newline='')
        self.scalar_writer = None
        if self.rank == 0:
            (self.root / 'stream_config.json').write_text(json.dumps(dict(format=1,
                start=start, stop=stop, matrices=layout, measurement_steps=self.steps,
                raw_dtype='FP32 gradient/momentum/displacement; BF16 direction stored as uint16 bits',
                direction='actual shape-scaled BF16 polar output, lossless bit storage',
                executed_update='before-after, including decay and rounding',
                recipe='PR369 seed0, start780, anneal780..3120, train_steps3060',
                training_batch_tokens=batch_tokens, residual_gating=False), indent=2))
            (self.root / 'capture_config.json').write_text(json.dumps(dict(
                windows=[[start, stop-1]], anchors=[], ring_length=256,
                storage='per-matrix streams; full model snapshots at mu measurement steps only')))
        dist.barrier()

    def begin(self, step):
        if self.optimizer._step != step:
            raise ValueError('Dense observer and original optimizer clocks differ')
        self.before_owned = {self.names[id(p)]: p.detach().clone() for p in self.owned}
        self.before = {n: p.detach().cpu().clone() for n, p in self.params.items()} if step in self.steps and self.rank == 0 else None
        self.momenta, self.directions, self.metrics = {}, {}, {}
        self.settings = [dict(pattern=r'^blocks\..*\.weight$', optimizer='muon',
            groups=[{k: float(g[k]) for k in ('lr', 'weight_decay')}])
            for g in self.optimizer.param_groups]

    def observe(self, p, g, m, q, taps, step, group, polar):
        pilot = os.environ.get('DENSE_PILOT') == '1'
        if pilot:
            inputs = [p, g, m, q, taps]
            pristine = [x.detach().clone() for x in inputs]
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
        name = self.names[id(p)]
        class Kernel:
            weights = taps
            normalize_available = False
        # The record scales its polar output in-place; undo that scale for the
        # requested raw msign metrics without changing the live update tensor.
        scale = max(1., p.shape[0] / p.shape[1]) ** .5
        unscaled = q.clone() / scale
        # BF16 multiply then divide is not an exact inverse for rectangular
        # matrices: recompute the unscaled trainer map for faithful metrics.
        if scale != 1.:
            unscaled = polar(m)
        metrics = momentum_restoring_metrics(p, g, m, unscaled, Kernel(), 256, polar)
        self.metrics[name] = metrics
        i = step-self.start
        for kind, value in (('gradient', g), ('momentum', m), ('muon_direction', q)):
            saved = (value.detach().contiguous().view(torch.uint16) if kind == 'muon_direction'
                     else value.detach().float())
            self.streams[name, kind][i] = saved.cpu().numpy()
        if step in self.steps:
            self.momenta[name] = m.detach().cpu().clone()
            self.directions[name] = q.detach().cpu().clone()
        if self.scalar_writer is None:
            self.scalar_writer = csv.DictWriter(self.scalar_file,
                fieldnames=['update', 'matrix', 'lr', 'weight_decay', *metrics])
            self.scalar_writer.writeheader()
        self.scalar_writer.writerow(dict(update=step, matrix=name, lr=group['lr'],
                                          weight_decay=group['weight_decay'], **metrics))
        if pilot:
            if any(not torch.equal(x, saved) for x, saved in zip(inputs, pristine)):
                raise RuntimeError('Dense observer mutated an actual optimizer input')
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                raise RuntimeError('Dense observer advanced the trainer RNG')

    def end(self, step):
        if len(self.metrics) != len(self.owned):
            raise ValueError('Dense capture missed an owned Muon matrix')
        for p in self.owned:
            name = self.names[id(p)]
            self.streams[name, 'executed_update'][step-self.start] = (
                self.before_owned[name] - p.detach()).float().cpu().numpy()
        if step in self.steps:
            payload = dict(update=step, rank=self.rank, format=2,
                post_insertion_prepolar_momentum=self.momenta, polar_output=self.directions,
                batch_receipts={'used': {'batch_tokens': self.tokens}}, settings=self.settings)
            if self.rank == 0:
                payload.update(model_before=self.before,
                    buffers={n: b.detach().cpu().clone() for n, b in self.model.named_buffers()})
            self.writer.save('boundary', step, payload)
        if os.environ.get('DENSE_PILOT') == '1':
            with (self.root / f'observer_integrity_rank{self.rank}.jsonl').open('a') as log:
                log.write(json.dumps(dict(update=step, matrices=len(self.metrics),
                    weight_gradient_momentum_direction_kernel_unchanged=True,
                    cpu_cuda_rng_unchanged=True)) + '\n')
        self.scalar_file.flush()
        if (step+1) % 25 == 0 or step+1 == self.stop:
            for array in self.streams.values():
                array.flush()
        # Per-rank completion receipt never advertises unwritten rows.
        receipt = self.root / f'raw_progress_rank{self.rank}.json'
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(last_update=step, completed=step-self.start+1,
                                             total=self.stop-self.start, time=time.time())))
        temporary.replace(receipt)
        self.before = self.before_owned = None
        if self.rank == 0:
            print('DENSE_PROGRESS ' + json.dumps(dict(update=step, total=self.stop-self.start,
                completed=step-self.start+1)), flush=True)


class StreamCapture:
    """Bounded-memory adapter for existing per-matrix covariance estimators."""
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / 'stream_config.json'
        self.config = json.loads(path.read_text())
        self.start, self.stop = self.config['start'], self.config['stop']
        self.windows = [(self.start, self.stop-1)]
        self.shapes = {n: tuple(v['shape']) for n, v in self.config['matrices'].items()}
        self.owners = {n: v['owner'] for n, v in self.config['matrices'].items()}
        self.manifest_hashes = {'stream_config.json': hashlib.sha256(path.read_bytes()).hexdigest()}
        for rank in set(self.owners.values()):
            receipt = json.loads((self.root / f'raw_progress_rank{rank}.json').read_text())
            if receipt['last_update'] != self.stop-1 or receipt['completed'] != self.stop-self.start:
                raise ValueError('Raw capture is incomplete; refuse unwritten rows')
        self.arrays = {}
        self.local_sources = {}

    def mapping(self, kind, step, rank=None):
        result = {}
        for name in self.shapes:
            if rank is not None and self.owners[name] != rank:
                continue
            key = name, kind
            if key not in self.arrays:
                path = self.local_sources.get(key, self.root / 'raw' / f'{name}.{kind}.npy')
                a = np.load(path, mmap_mode='c')
                expected = np.uint16 if kind == 'muon_direction' else np.float32
                if a.shape != (self.stop-self.start, *self.shapes[name]) or a.dtype != expected:
                    raise ValueError('Raw stream shape/dtype differs')
                self.arrays[key] = a
            value = torch.from_numpy(self.arrays[key][step-self.start])
            result[name] = value.view(torch.bfloat16) if kind == 'muon_direction' else value
        return result

    def load(self, rank, kind, step):
        if not self.start <= step < self.stop:
            return None
        if kind == 'gradient':
            return self.mapping('gradient', step)
        return dict(update=step, rank=rank,
            post_insertion_prepolar_momentum=self.mapping('momentum', step, rank),
            polar_output=self.mapping('muon_direction', step, rank),
            executed_update=self.mapping('executed_update', step))


class NoopRecorder:
    """Reference pilot: execute exactly the same trainer without observations."""
    def __init__(self, *args):
        pass
    def begin(self, step):
        pass
    def observe(self, *args):
        pass
    def end(self, step):
        pass


def main():
    source = Path(os.environ['TUNED_TRAINER_SOURCE'])
    expected = os.environ['TUNED_TRAINER_SHA256']
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Tuned trainer source hash changed')
    sys.path.insert(0, str(source.parent))
    sys.argv[0] = str(source)
    recorder = NoopRecorder if os.environ.get('DENSE_PILOT_REFERENCE') == '1' else DenseRecorder
    namespace = dict(__name__='__main__', __file__=str(source), DenseRecorder=recorder)
    exec(compile(instrument(source.read_text()), str(source), 'exec'), namespace)
    if os.environ.get('DENSE_FINAL_MODEL') and int(os.environ['LOCAL_RANK']) == 0:
        torch.save({n: v.detach().cpu() for n, v in base_model(namespace['model']).state_dict().items()},
                   os.environ['DENSE_FINAL_MODEL'])


if __name__ == '__main__':
    main()
