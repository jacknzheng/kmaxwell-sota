"""Centered temporal covariance, separately for each captured parameter matrix.

Exact sample-space Gram algebra avoids materializing parameter-by-parameter
covariance. No cosine normalization, projections, or averaging across matrices.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from records.track_3_optimization.analysis.lag_cosine import Capture


FIELDS = {'momentum': 'post_insertion_prepolar_momentum',
          'muon_direction': 'polar_output', 'executed_update': 'executed_update'}


def gram(vectors, feature_chunk=16384, device="cpu"):
    """FP64 dots of unnormalized saved tensors, bounded feature workspace."""
    if not vectors or feature_chunk < 1:
        raise ValueError('empty vectors or invalid feature chunk')
    if not all(isinstance(v, torch.Tensor) for v in vectors):
        raise ValueError('non-tensor observation')
    shape = vectors[0].shape
    if any(v.shape != shape for v in vectors):
        raise ValueError('tensor shapes differ')
    flat = [v.detach().reshape(-1) for v in vectors]
    out = torch.zeros((len(flat), len(flat)), dtype=torch.float64, device=device)
    for start in range(0, flat[0].numel(), feature_chunk):
        x = torch.stack([v[start:start + feature_chunk] for v in flat]).to(device=device, dtype=torch.float64)
        if not bool(torch.isfinite(x).all()):
            raise ValueError('nonfinite captured tensor')
        out.add_(x @ x.T)
    return out.cpu().numpy()


def center(g):
    return g - g.mean(axis=0)[None, :] - g.mean(axis=1)[:, None] + g.mean()


def resample_weights(n, block, rng):
    """Noncircular moving blocks; never joins samples to create lag pairs."""
    if not 1 <= block <= n:
        raise ValueError('bootstrap block must lie inside the estimation block')
    indices = np.concatenate([np.arange(s, s + block) for s in
                              rng.integers(0, n - block + 1, size=(n + block - 1) // block)])[:n]
    return np.bincount(indices, minlength=n).astype(float) / n


def bootstrap_profiles(g, max_lag, block, replicates, rng):
    """Reweight original temporal pairs and reestimate the window mean.

Intervals are exploratory weighted moving-block bootstrap intervals, not an
IID confidence guarantee or a test of convergence. No artificial lag pairs are
made across resampled block joins. The original clock/pair mask is retained.
"""
    n = len(g)
    out = np.full((replicates, max_lag + 1), np.nan)
    for b in range(replicates):
        w = resample_weights(n, block, rng)
        gw = g @ w
        mu2 = w @ gw
        for k in range(max_lag + 1):
            t = np.arange(k, n)
            mass = w[t].sum()
            if mass > 0:
                out[b, k] = np.sum(w[t] * (g[t, t-k] - gw[t] - gw[t-k] + mu2)) / mass
    return out


def summarize(g, max_lag, top_rank=5, bootstrap_block=16, replicates=200, seed=0):
    n = len(g)
    if (g.shape != (n, n) or n < 2 or not np.isfinite(g).all()
            or not 0 <= max_lag < n or top_rank < 1 or replicates < 2):
        raise ValueError('invalid Gram matrix or estimator settings')
    z = center(g)
    pairs = n - np.arange(max_lag + 1)
    trace = np.array([np.diag(z, -k).mean() for k in range(max_lag + 1)])
    # C(k) = Z_current.T @ Z_past / (n-k); exact Frobenius norm.
    cov_norm = np.array([np.sqrt(max(0., np.sum(z[k:, k:] * z[:n-k, :n-k]))) / (n-k)
                         for k in range(max_lag + 1)])
    eigenvalues, u = np.linalg.eigh((z + z.T) * .5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, u = np.maximum(eigenvalues[order], 0.), u[:, order]
    threshold = eigenvalues[0] * 1e-10 if eigenvalues[0] > 0 else 0.
    rank = min(top_rank, int(np.count_nonzero(eigenvalues > threshold)))
    # X_centered.T @ basis is an orthonormal basis in parameter space.
    basis = u[:, :rank] / np.sqrt(eigenvalues[:rank])[None, :] if rank else np.empty((n, 0))
    boot = bootstrap_profiles(g, max_lag, bootstrap_block, replicates, np.random.default_rng(seed))
    lo, hi = np.full(max_lag + 1, np.nan), np.full(max_lag + 1, np.nan)
    for k in range(max_lag + 1):
        finite = boot[:, k][np.isfinite(boot[:, k])]
        if len(finite) >= 2:
            lo[k], hi[k] = np.quantile(finite, [.025, .975])
    return dict(centered_gram=z, trace=trace, pair_counts=pairs, covariance_norm=cov_norm,
                mean_norm=np.sqrt(max(0., g.mean())), variance=trace[0],
                magnitude=np.sqrt(np.maximum(np.diag(g), 0.)),
                eigenvalues=eigenvalues / n, basis=basis, numerical_rank=int(np.count_nonzero(eigenvalues > threshold)),
                trace_low=lo, trace_high=hi, bootstrap_valid=np.isfinite(boot).sum(axis=0))


def covariance_drift(a, b, cross, max_lag):
    """Exact ||C_b(k)-C_a(k)||_F, without constructing C in parameter space."""
    h = cross - cross.mean(axis=0)[None, :] - cross.mean(axis=1)[:, None] + cross.mean()
    na, nb = len(a['centered_gram']), len(b['centered_gram'])
    if cross.shape != (na, nb):
        raise ValueError('cross Gram shape differs')
    values = []
    for k in range(max_lag + 1):
        overlap = np.sum(h[k:, k:] * h[:na-k, :nb-k]) / ((na-k) * (nb-k))
        square = a['covariance_norm'][k]**2 + b['covariance_norm'][k]**2 - 2 * overlap
        values.append(np.sqrt(max(0., square)))
    r = min(a['basis'].shape[1], b['basis'].shape[1])
    if r:
        overlap = a['basis'][:, :r].T @ h @ b['basis'][:, :r]
        singular = np.linalg.svd(overlap, compute_uv=False)
        angles = np.degrees(np.arccos(np.clip(singular, 0., 1.)))
    else:
        angles = np.array([])
    return np.array(values), angles


def within_block_noise(g, block, replicates, rng):
    """Bootstrap lag-zero covariance drift from its empirical reference.

Means are reestimated for each weighted replicate. Exact Gram identities keep
this O(replicates*n^2), with no dense parameter covariance or random projection.
"""
    z = center(g)
    n = len(z)
    base2 = np.sum(z * z) / n**2
    z2 = z * z
    rows = []
    for _ in range(replicates):
        w = resample_weights(n, block, rng)
        zw = z @ w
        mu2 = w @ zw
        boot2 = w @ z2 @ w - 2 * np.sum(w * zw**2) + mu2**2
        inner = np.sum(w * np.sum(z2, axis=1)) / n - np.sum(zw**2) / n
        rows.append(np.sqrt(max(0., boot2 + base2 - 2 * inner)))
    return np.quantile(rows, [.5, .95])


def complete_blocks(windows, length):
    if length < 2:
        raise ValueError('block length must be at least two')
    return [(start, start + length - 1) for a, b in windows
            for start in range(a, b - length + 2, length)]


def series(capture, name, kind, start, stop):
    rank = 0 if kind in ('gradient', 'executed_update') else capture.owners[name]
    record_kind = 'gradient' if kind == 'gradient' else 'boundary'
    vectors = []
    for t in range(start, stop + 1):
        payload = capture.load(rank, record_kind, t)
        mapping = payload if kind == 'gradient' else payload.get(FIELDS[kind], {}) if payload is not None else {}
        if payload is None or name not in mapping:
            raise ValueError(f'missing {kind} tensor {name} at update {t}; not joining across gaps')
        value = mapping[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != capture.shapes[name]:
            raise ValueError('analysis tensor shape differs from capture registry')
        # executed_update was saved as before-after; negate to report after-before.
        vectors.append(-value if kind == 'executed_update' else value)
    return vectors


def write_csv(path, rows):
    if rows:
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def render(output, name, kind, length, blocks, summaries, drifts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for (start, stop), s in zip(blocks, summaries):
        lag = np.arange(len(s['trace']))
        line, = axes[0, 0].plot(lag, s['trace'], label=f'{start}–{stop}; pairs {len(s["magnitude"])}→{s["pair_counts"][-1]}')
        axes[0, 0].fill_between(lag, s['trace_low'], s['trace_high'], alpha=.12, color=line.get_color())
    axes[0, 0].set(xlabel='Lag k', ylabel='Centered lag covariance trace (unnormalized)')
    axes[0, 0].legend(fontsize=7)
    time = [b for a, b in blocks]
    axes[0, 1].plot(time, [s['mean_norm']**2 for s in summaries], label='Squared mean-vector norm')
    axes[0, 1].plot(time, [s['variance'] for s in summaries], label='Total centered variance')
    axes[0, 1].set(xlabel='Block end checkpoint', ylabel='Squared tensor units')
    axes[0, 1].legend(fontsize=7)
    for (a, b), s in zip(blocks, summaries):
        axes[0, 2].plot(np.arange(a, b+1), s['magnitude'])
    axes[0, 2].set(xlabel='Checkpoint t', ylabel='Unnormalized tensor magnitude')
    if drifts:
        axes[1, 0].plot(time[1:], [d['distance'][0] for d in drifts], label='Successive C(0) drift')
        axes[1, 0].plot(time[1:], [d['noise95'] for d in drifts], label='Within-block bootstrap 95% reference')
        axes[1, 0].legend(fontsize=7)
    axes[1, 0].set(xlabel='Block end checkpoint', ylabel='Covariance Frobenius norm (not a significance test)')
    for j in range(min(5, len(summaries[0]['eigenvalues']))):
        axes[1, 1].plot(time, [s['eigenvalues'][j] for s in summaries], label=f'Eigenvalue {j+1}')
    axes[1, 1].set(xlabel='Block end checkpoint', ylabel='Lag-zero covariance eigenvalues')
    axes[1, 1].legend(fontsize=7)
    for j in range(max((len(d['angles']) for d in drifts), default=0)):
        axes[1, 2].plot(time[1:], [d['angles'][j] if j < len(d['angles']) else np.nan for d in drifts],
                        label=f'Principal angle {j+1}')
    axes[1, 2].set(xlabel='Block end checkpoint', ylabel='Leading-subspace angle (degrees)', ylim=(0, 90))
    if any(len(d['angles']) for d in drifts):
        axes[1, 2].legend(fontsize=7)
    fig.suptitle(f'{name} | {kind} | L={length}; per-matrix exploratory temporal estimates')
    fig.tight_layout()
    fig.savefig(output / f'{name}_{kind}_L{length}.png', dpi=130)
    plt.close(fig)


def analyze(capture, settings, output, names=None, plots=True):
    names = sorted(capture.owners) if names is None else names
    if not names or len(set(names)) != len(names) or any(
            n not in capture.owners or not re.fullmatch(r'[A-Za-z0-9_.]+', n) for n in names):
        raise ValueError('unknown or empty matrix selection')
    lengths = settings['block_lengths']
    kmax = settings['max_lag']
    if (not lengths or any(not isinstance(l, int) or isinstance(l, bool) or l <= kmax for l in lengths)
            or kmax < 0 or settings['bootstrap_replicates'] < 2 or settings['top_rank'] < 1
            or not 1 <= settings['bootstrap_block'] <= min(lengths)
            or settings['feature_chunk'] < 1 or settings['threads'] < 1
            or not settings['kinds'] or any(k not in ('gradient', *FIELDS) for k in settings['kinds'])):
        raise ValueError('invalid analysis configuration')
    if any(not complete_blocks(capture.dense_intervals, l) for l in lengths):
        raise ValueError('no complete estimation block; choose shorter block lengths')
    torch.set_num_threads(settings['threads'])
    profiles, traces, drift_rows, eigen_rows, angle_rows = [], [], [], [], []
    for name in names:
        for kind in settings['kinds']:
            # Compute each interval's exact Gram ONCE. All block lengths,
            # adjacent-block drift and angles are slices of the same Gram.
            # This avoids rereading terabytes for every length and cross block.
            interval_grams = {}
            for start, stop in capture.dense_intervals:
                vectors = series(capture, name, kind, start, stop)
                interval_grams[(start, stop)] = gram(vectors, settings['feature_chunk'],
                                                      settings.get('device', 'cpu'))
                del vectors
            for length in lengths:
                blocks = complete_blocks(capture.dense_intervals, length)
                summaries, drifts = [], []
                previous = None
                for index, (a, b) in enumerate(blocks):
                    origin, end = next(w for w in capture.dense_intervals if w[0] <= a and b <= w[1])
                    whole = interval_grams[(origin, end)]
                    raw = whole[a-origin:b-origin+1, a-origin:b-origin+1]
                    s = summarize(raw, kmax, settings['top_rank'], settings['bootstrap_block'],
                                  settings['bootstrap_replicates'], settings['seed'] + index)
                    np.savez_compressed(output / f'{name}_{kind}_L{length}_{a}_{b}.npz',
                                        raw_gram=raw, centered_gram=s['centered_gram'],
                                        eigenvalues=s['eigenvalues'], basis_coefficients=s['basis'],
                                        checkpoints=np.arange(a, b+1))
                    base = dict(matrix=name, kind=kind, block_length=length, start=a, stop=b)
                    for k in range(kmax + 1):
                        profiles.append(dict(**base, lag=k, pairs=int(s['pair_counts'][k]),
                                             covariance_trace=float(s['trace'][k]),
                                             covariance_frobenius=float(s['covariance_norm'][k]),
                                             bootstrap_low=float(s['trace_low'][k]), bootstrap_high=float(s['trace_high'][k]),
                                             bootstrap_valid=int(s['bootstrap_valid'][k])))
                    for t, norm in zip(range(a, b+1), s['magnitude']):
                        traces.append(dict(**base, checkpoint=t, magnitude=float(norm), mean_norm=float(s['mean_norm']),
                                           variance=float(s['variance'])))
                    for j, eig in enumerate(s['eigenvalues']):
                        eigen_rows.append(dict(**base, eigen_index=j+1, eigenvalue=float(eig),
                                               numerical_rank=s['numerical_rank']))
                    if previous is not None and blocks[index-1][1] + 1 == a:
                        pa, pb = blocks[index-1]
                        cross = whole[pa-origin:pb-origin+1, a-origin:b-origin+1]
                        distance, angles = covariance_drift(previous, s, cross, kmax)
                        noise = within_block_noise(raw, settings['bootstrap_block'], settings['bootstrap_replicates'],
                                                   np.random.default_rng(settings['seed'] + index))
                        drifts.append(dict(distance=distance, noise95=float(noise[1]), angles=angles))
                        for k, value in enumerate(distance):
                            drift_rows.append(dict(**base, previous_start=blocks[index-1][0], lag=k,
                                                   covariance_distance=float(value),
                                                   lag0_bootstrap_noise95=float(noise[1]) if k == 0 else None))
                        for j, value in enumerate(angles):
                            angle_rows.append(dict(**base, angle_index=j+1, angle_degrees=float(value),
                                                   compared_rank=len(angles)))
                    else:
                        drifts.append(None)
                    summaries.append(s)
                    previous = s
                if plots:
                    render(output, name, kind, length, blocks, summaries,
                           [d or dict(distance=np.full(kmax+1, np.nan), noise95=np.nan, angles=[]) for d in drifts[1:]])
                print(json.dumps(dict(matrix=name, kind=kind, block_length=length, blocks=len(blocks))), flush=True)
    for filename, rows in [('lag_profiles.csv', profiles), ('mean_magnitude.csv', traces),
                           ('covariance_drift.csv', drift_rows), ('eigenvalues.csv', eigen_rows),
                           ('subspace_angles.csv', angle_rows)]:
        write_csv(output / filename, rows)
    return dict(matrices=names, settings=settings, windows=capture.dense_intervals,
                estimand='per-matrix temporal covariance of flattened saved tensors; window mean subtracted',
                normalization='1/(L-k) lagged covariance, 1/L lag-zero population sample covariance',
                uncertainty='exploratory weighted moving-block bootstrap; original lag pairs, reestimated mean',
                noise_reference='lag-zero within-block bootstrap deviation, not a calibrated two-block hypothesis test',
                limits='trajectory drift and stochastic variation mixed; scheduled kernel/LR not held fixed; no convergence verdict',
                update_fields='muon_direction = saved shape-scaled polar output; executed_update = after-before including decay/rounding',
                input_checks='inherited Capture clock/shape/owner/size validation; tensor SHA256 not recomputed',
                manifest_hashes=capture.manifest_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/analysis_autocovariance.yaml')
    parser.add_argument('--matrix', action='append', help='Exact matrix name; repeat to select several. Default: each separately.')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--device', default='cpu', help='FP64 Gram device, e.g. cuda:0; estimators unchanged')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load(args.config.read_text())
    settings['device'] = args.device
    receipt = analyze(Capture(args.capture), settings, args.output, args.matrix, not args.no_plots)
    receipt.update(complete=True, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in args.output.iterdir() if p.is_file()})
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
