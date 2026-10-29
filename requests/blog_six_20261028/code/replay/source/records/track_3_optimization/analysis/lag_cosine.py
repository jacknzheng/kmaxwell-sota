import argparse
import base64
import gc
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


KINDS = ('gradient', 'momentum', 'muon_update')
BOUNDARY_KEYS = {'momentum': 'post_insertion_prepolar_momentum',
                 'muon_update': 'polar_output'}


def lag_cosines(steps, vectors, targets, max_lag):
    if max_lag < 0 or len(set(steps)) != len(steps):
        raise ValueError('negative lag or duplicate step')
    if len(steps) != len(vectors):
        raise ValueError('step/vector count differs')
    out = np.full((max_lag + 1, len(targets)), np.nan, dtype=np.float32)
    if not steps:
        return out
    shapes = {tuple(v.shape) for v in vectors}
    if len(shapes) != 1:
        raise ValueError('matrix shape changed')
    # Float64 reductions, without changing captured FP32/BF16 tensor values.
    x = torch.stack([v.detach().reshape(-1).double() for v in vectors])
    if not bool(torch.isfinite(x).all()):
        raise ValueError('nonfinite captured tensor')
    norms = torch.linalg.vector_norm(x, dim=1)
    good = norms > 0
    x /= torch.where(good, norms, torch.ones_like(norms))[:, None]
    indices = {t: i for i, t in enumerate(steps)}
    valid_targets = [(j, indices[t]) for j, t in enumerate(targets) if t in indices]
    if not valid_targets:
        return out
    # Only target-to-history dots, not a parameter-by-parameter covariance matrix.
    gram = (x[[i for _, i in valid_targets]] @ x.T).numpy()
    for row, (j, i) in enumerate(valid_targets):
        t = targets[j]
        for k in range(max_lag + 1):
            earlier = indices.get(t - k)
            if earlier is not None and good[i] and good[earlier]:
                out[k, j] = np.clip(gram[row, earlier], -1, 1)
    return out


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / 'capture_config.json').read_text())
        self.snapshot_kind = 'metrics_snapshot' if self.config.get('format') == 3 else 'boundary'
        if self.snapshot_kind == 'metrics_snapshot':
            policy = self.config['metrics_snapshot'] or {}
            intervals = [(i['start'], i['end']) for i in policy.get('dense_intervals', [])]
            intervals.extend((t,t) for t in policy.get('steps', []))
            # Include coarse-cadence observations too, without inventing missing steps.
            for line in (self.root/'rank0/manifest.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row['kind'] == 'metrics_snapshot':
                    intervals.append((row['step'],row['step']))
            self.dense_intervals = []
            for start,end in sorted(intervals):
                if self.dense_intervals and start <= self.dense_intervals[-1][1]+1:
                    a,b = self.dense_intervals[-1]
                    self.dense_intervals[-1] = (a,max(b,end))
                else:
                    self.dense_intervals.append((start,end))
        else:
            self.dense_intervals = [tuple(w) for w in self.config['windows']]
        if (not self.dense_intervals or any(a < 0 or b < a for a, b in self.dense_intervals)
                or any(self.dense_intervals[i][1] >= self.dense_intervals[i + 1][0]
                       for i in range(len(self.dense_intervals) - 1))):
            raise ValueError('dense intervals must be nonnegative, sorted and disjoint')
        self.records = {}
        self.registry_ready = False
        self.manifest_hashes = {}
        for folder in sorted(self.root.glob('rank[0-9]*')):
            rank = int(folder.name[4:])
            manifest = folder / 'manifest.jsonl'
            self.manifest_hashes[str(manifest.relative_to(self.root))] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            for line in manifest.read_text().splitlines():
                r = json.loads(line)
                if Path(r['file']).name != r['file']:
                    raise ValueError('unsafe manifest filename')
                if (not isinstance(r['step'], int) or isinstance(r['step'], bool)
                        or r['step'] < 0 or r['file'] != f"{r['kind']}_{r['step']:06d}.pt"):
                    raise ValueError('manifest clock/filename differs')
                key = (rank, r['kind'], r['step'])
                if key in self.records:
                    raise ValueError('duplicate manifest record')
                self.records[key] = (folder / r['file'], r)
        self.owners = {}
        self.shapes = {}
        first = self.dense_intervals[0][0]
        ranks = sorted({rank for rank, kind, step in self.records if kind == self.snapshot_kind})
        for rank in ranks:
            payload = self.load(rank, 'metrics_snapshot', first)
            if payload is None:
                raise ValueError(f'rank {rank} missing first boundary')
            m, q = (payload[key] for key in BOUNDARY_KEYS.values())
            if set(m) != set(q):
                raise ValueError('momentum/update registry differs')
            for name, value in m.items():
                if name in self.owners or value.ndim != 2 or q[name].shape != value.shape:
                    raise ValueError('duplicate matrix ownership or invalid shape')
                self.owners[name] = rank
                self.shapes[name] = tuple(value.shape)
        if not self.owners:
            raise ValueError('no captured matrices')
        self.registry_ready = True

    def load(self, rank, kind, step):
        if kind in ('boundary', 'metrics_snapshot'):
            kind = self.snapshot_kind
        entry = self.records.get((rank, kind, step))
        if entry is None:
            return None
        path, record = entry
        if path.stat().st_size != record['bytes']:
            raise ValueError(f'file size differs: {path}')
        # mmap avoids reading model/optimizer payloads that this analysis never uses.
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if kind == self.snapshot_kind and (payload['update'] != step or payload['rank'] != rank):
            raise ValueError('boundary clock/rank differs')
        if self.registry_ready:
            mappings = [payload] if kind == 'gradient' else [payload[k] for k in BOUNDARY_KEYS.values()]
            for mapping in mappings:
                for name, value in mapping.items():
                    if kind == 'gradient' and (name == 'aux_settings' or name.startswith('aux::')):
                        continue
                    if name not in self.owners:
                        raise ValueError('unknown matrix in captured tensor registry')
                    if kind == self.snapshot_kind and self.owners[name] != rank:
                        raise ValueError('captured matrix belongs to another rank')
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != self.shapes[name]:
                        raise ValueError('captured matrix shape differs from registry')
        return payload


def analyze(capture, max_lag, output):
    names = sorted(capture.owners)
    result = {kind: [] for kind in KINDS}
    for a, b in capture.dense_intervals:
        targets = list(range(a, b + 1))
        shape = (len(names), max_lag + 1, len(targets))
        arrays = {kind: np.full(shape, np.nan, dtype=np.float32) for kind in KINDS}
        # Gradient files are shared across all matrix owners; map each once per window.
        gradients = {}
        for t in range(max(0, a - max_lag), b + 1):
            payload = capture.load(0, 'gradient', t)
            if payload is not None:
                gradients[t] = payload
        for i, name in enumerate(names):
            steps = sorted(t for t, payload in gradients.items() if name in payload)
            arrays['gradient'][i] = lag_cosines(steps, [gradients[t][name] for t in steps], targets, max_lag)
        del gradients
        gc.collect()
        for rank in sorted(set(capture.owners.values())):
            boundaries = {}
            # Earlier dense windows may supply exact t-k; never bridge missing steps.
            for t in range(max(0, a - max_lag), b + 1):
                payload = capture.load(rank, 'metrics_snapshot', t)
                if payload is not None:
                    boundaries[t] = payload
            for i, name in enumerate(names):
                if capture.owners[name] != rank:
                    continue
                for kind, key in BOUNDARY_KEYS.items():
                    steps = sorted(t for t, p in boundaries.items() if name in p[key])
                    arrays[kind][i] = lag_cosines(steps, [boundaries[t][key][name] for t in steps], targets, max_lag)
            del boundaries
            gc.collect()
            print(json.dumps({'window': [a, b], 'rank_complete': rank}), flush=True)
        for kind in KINDS:
            result[kind].append(arrays[kind])
        np.savez_compressed(output / f'window_{a}_{b}.npz', matrices=np.array(names),
                            checkpoints=np.array(targets), lags=np.arange(max_lag + 1), **arrays)
        print(json.dumps({'window_complete': [a, b]}), flush=True)
    return names, result


def combined(arrays):
    gap = np.full((*arrays[0].shape[:-1], 3), np.nan, dtype=np.float32)
    pieces = []
    for i, array in enumerate(arrays):
        if i:
            pieces.append(gap)
        pieces.append(array)
    return np.concatenate(pieces, axis=-1)


def render_atlases(names, result, windows, output, max_lag):
    cmap = matplotlib.colormaps['coolwarm'].copy()
    cmap.set_bad('#dedede')
    columns = 6
    rows = math.ceil(len(names) / columns)
    ticks, labels, offset = [], [], 0
    for a, b in windows:
        ticks.extend([offset, offset + b - a])
        labels.extend([str(a), str(b)])
        offset += b - a + 4
    for kind in KINDS:
        values = combined(result[kind])
        fig, axes = plt.subplots(rows, columns, figsize=(24, max(4, rows * 2.6)), squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= len(names):
                ax.set_visible(False)
                continue
            image = ax.imshow(values[i], origin='lower', aspect='auto', interpolation='nearest',
                              cmap=cmap, vmin=-1, vmax=1, extent=(-.5, values.shape[-1] - .5, -.5, max_lag + .5))
            ax.set_title(names[i], fontsize=7)
            ax.set_xticks(ticks, labels, fontsize=7, rotation=90)
            ax.set_xlabel('Checkpoint t (separate captured windows)', fontsize=7)
            ax.set_ylabel('Lag k', fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
        fig.suptitle(f'{kind}: per-matrix cosine(X_t, X_(t-k)); gray = unavailable/zero norm', fontsize=14)
        fig.tight_layout(rect=(0, 0, .96, .98))
        cax = fig.add_axes((.975, .2, .008, .6))
        fig.colorbar(image, cax=cax, label='Cosine [-1, 1]')
        fig.savefig(output / f'{kind}_lag_cosine.png', dpi=150)
        plt.close(fig)


def render_browser(names, result, windows, output, max_lag):
    data = {kind: base64.b64encode(combined(result[kind]).astype('<f4').tobytes()).decode() for kind in KINDS}
    width = sum(b - a + 1 for a, b in windows) + 3 * (len(windows) - 1)
    metadata = json.dumps(dict(names=names, data=data, windows=windows, width=width,
                               height=max_lag + 1)).replace('<', chr(92) + 'u003c')
    ticks, offset = [], 0
    for a, b in windows:
        for index, label in [(offset, a), (offset + b - a, b)]:
            ticks.append(f'<span style="left:{100 * (index + .5) / width}%">{label}</span>')
        offset += b - a + 4
    axes = ('<div class="plot"><div class="yaxis"><span class="ylab">Lag k</span>'
            f'<span class="ytop">{max_lag}</span><span class="ymid">{max_lag // 2}</span>'
            '<span class="ybottom">0</span></div><div><canvas id="__KIND__"></canvas>'
            '<div class="xaxis">' + ''.join(ticks) + '</div>'
            '<div class="xlabel">Checkpoint t (separate captured windows)</div></div></div>')
    document = '''<!doctype html><html><meta charset="utf-8"><title>Captured lag cosines</title>
<style>body{font:16px system-ui;margin:24px;max-width:1200px}canvas{width:100%;height:290px;image-rendering:pixelated}section{margin:28px 0}.bar{height:16px;background:linear-gradient(90deg,#3b4cc0,#f7f7f7,#b40426)}.ends{display:flex;justify-content:space-between}.plot{display:grid;grid-template-columns:48px minmax(0,1fr)}.yaxis{position:relative;height:290px;font-size:12px}.ytop,.ymid,.ybottom{position:absolute;right:5px}.ytop{top:0}.ymid{top:50%}.ybottom{bottom:0}.ylab{position:absolute;top:50%;left:-12px;transform:rotate(-90deg)}.xaxis{position:relative;height:44px;font-size:12px}.xaxis span{position:absolute;top:10px;transform:translateX(-50%) rotate(-90deg)}.xlabel{text-align:center;font-size:12px}#readout{position:sticky;top:0;background:white;padding:8px;min-height:24px;z-index:1}</style>
<h1>Gradient, momentum and Muon-update lag cosines</h1>
<p>Select one matrix. No matrices or layers are averaged. X-axis: checkpoint t; Y-axis: lag k (0 at bottom). Gray: unavailable pair or zero norm. Each captured window has its own segment; the intervening checkpoints are not sampled.</p>
<label>Matrix <select id="matrix"></select></label><p id="windows"></p>
<p id="readout">Hover over a cell to read its exact checkpoint, lag and cosine.</p>
<div class="bar"></div><div class="ends"><span>−1</span><span>cosine 0</span><span>+1</span></div>
<section><h2>Gradients</h2><canvas id="gradient"></canvas></section>
<section><h2>Momentum (actual post-insertion input)</h2><canvas id="momentum"></canvas></section>
<section><h2>Muon update (saved polar output, including shape scale)</h2><canvas id="muon_update"></canvas></section>
<p>Cosines are uncentered directional correlations, not centered covariance estimates. Muon update here excludes learning rate, decay and parameter-rounding effects.</p>
<script>const D=__DATA__; const arrays={};
for(const [kind,encoded] of Object.entries(D.data)){const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));arrays[kind]=new Float32Array(bytes.buffer);}
const select=document.getElementById('matrix');D.names.forEach((name,i)=>{const opt=document.createElement('option');opt.value=i;opt.textContent=name;select.append(opt)});
document.getElementById('windows').textContent='Checkpoint t: '+D.windows.map(w=>w[0]+'–'+w[1]).join(' | ')+'; lag k: 0–'+(D.height-1);
function draw(){for(const kind of Object.keys(arrays)){const c=document.getElementById(kind);c.width=D.width;c.height=D.height;const ctx=c.getContext('2d'),im=ctx.createImageData(c.width,c.height),a=arrays[kind],offset=Number(select.value)*D.width*D.height;
for(let k=0;k<D.height;k++)for(let t=0;t<D.width;t++){const v=a[offset+k*D.width+t],p=4*((D.height-1-k)*D.width+t);let rgb=[222,222,222];if(Number.isFinite(v)){const f=Math.min(1,Math.abs(v)),end=v<0?[59,76,192]:[180,4,38];rgb=end.map(e=>Math.round(247+(e-247)*f))}im.data.set([...rgb,255],p)}ctx.putImageData(im,0,0);}}
select.onchange=draw;draw();
for(const kind of Object.keys(arrays)){document.getElementById(kind).onmousemove=event=>{
 const c=document.getElementById(kind),r=c.getBoundingClientRect(),j=Math.min(D.width-1,Math.max(0,Math.floor((event.clientX-r.left)/r.width*D.width))),k=D.height-1-Math.min(D.height-1,Math.max(0,Math.floor((event.clientY-r.top)/r.height*D.height)));
 let t=null,start=0;for(const [a,b] of D.windows){if(j>=start&&j<start+b-a+1)t=a+j-start;start+=b-a+4;}
 const v=arrays[kind][Number(select.value)*D.width*D.height+k*D.width+j];
 document.getElementById('readout').textContent=kind+': '+D.names[Number(select.value)]+'; t='+t+', k='+k+', t−k='+(t===null?'—':t-k)+', cosine='+(Number.isFinite(v)?v.toFixed(6):'unavailable');
};}</script></html>'''
    for kind in KINDS:
        document = document.replace(f'<canvas id="{kind}"></canvas>', axes.replace('__KIND__', kind))
    (output / 'lag_cosines.html').write_text(document.replace('__DATA__', metadata))


def main():
    parser = argparse.ArgumentParser(description='Read captured tensors; plot per-matrix temporal lag cosines on CPU.')
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-lag', type=int, default=64)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.max_lag < 0 or args.threads < 1:
        parser.error('max-lag must be nonnegative and threads positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    capture = Capture(args.capture)
    names, result = analyze(capture, args.max_lag, args.output)
    render_atlases(names, result, capture.dense_intervals, args.output, args.max_lag)
    render_browser(names, result, capture.dense_intervals, args.output, args.max_lag)
    receipt = dict(complete=True, capture=str(capture.root), windows=capture.dense_intervals,
                   max_lag=args.max_lag, matrices=names, shapes=capture.shapes,
                   manifest_hashes=capture.manifest_hashes,
                   input_checks='manifest size + safe weights-only load + rank/update; tensor-file SHA256 not recomputed',
                   definitions={'gradient': 'globally reduced training gradient',
                                'momentum': 'actual post-insertion pre-polar momentum',
                                'muon_update': 'saved polar_output with rectangular scale; excludes lr/decay/rounding'},
                   finite_cells={k: int(sum(np.isfinite(a).sum() for a in arrays)) for k, arrays in result.items()},
                   files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in args.output.iterdir() if p.is_file()})
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'complete': True, 'output': str(args.output), 'finite_cells': receipt['finite_cells']}), flush=True)


if __name__ == '__main__':
    main()
