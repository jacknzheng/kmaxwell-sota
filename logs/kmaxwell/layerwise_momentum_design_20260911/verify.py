"""CPU-only audit of committed results and proposed measurement identities.

Run with Python + PyTorch. No training, network access, or checkpoint loading.
Writes verification.json beside this script. Does not verify a GPU experiment.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
torch.set_default_dtype(torch.float64)


def table(path):
    return list(csv.DictReader(
        [line for line in path.read_text().splitlines() if not line.startswith('#')],
        delimiter='\t'))


def paired(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = {3: 4.302652729911275, 4: 3.182446305284263}[n] * sd / math.sqrt(n)
    return dict(n=n, differences=values, mean=mean, sample_sd=sd,
                t95=[mean-half, mean+half])


def polar(x):
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


def main():
    out = {'audited_branch_sha': '98c92ea718ba15a7d11c1f6079c9cdca2e583c21',
           'torch_version': torch.__version__, 'scope': 'CPU arithmetic and toy checks only'}
    rows = table(LOGS / 'req054_agematched_ema/readout.tsv')
    out['req054_kmax_minus_scheduled_ema'] = paired([
        float(r['kmax_2750'])-float(r['agema_2750']) for r in rows])
    values = {}
    for seed in range(3):
        values[seed] = {}
        for arm in ('adam', 'kmaxwell', 'agema'):
            p = LOGS / 'req056_adam_kmaxwell/raw_logs' / f'val_{arm}_s{seed}.tsv'
            step, loss, *_ = p.read_text().splitlines()[-1].split()
            assert step == '3250'
            values[seed][arm] = float(loss)
    out['req056_kmax_minus_adam'] = paired([
        v['kmaxwell']-v['adam'] for v in values.values()])
    mismatches = []
    for p in sorted((LOGS / 'req051_lr_curvature_decomp/raw_json').glob('*_curv.json')):
        curv = json.loads(p.read_text())
        act = json.loads(p.with_name(p.name.replace('_curv', '_act')).read_text())
        for step, state in curv.items():
            mismatches.append([p.name, step, state['tokens'], act['tokens']])
    out['req051_probe_pairs'] = len(mismatches)
    out['req051_token_count_pairs'] = sorted(set((r[2], r[3]) for r in mismatches))
    out['req051_mismatched_pairs'] = sum(r[2] != r[3] for r in mismatches)
    p51 = table(LOGS / 'req051_lr_curvature_decomp/provenance.tsv')
    p52 = table(LOGS / 'req052_uniform_lr_controls/provenance.tsv')
    out['req051_provenance'] = p51
    out['req052_provenance'] = p52
    out['req055_late_fd_disagreement'] = []
    for p in sorted((LOGS / 'req055_post_muon_update_geometry/raw_json').glob('*.json')):
        d = json.loads(p.read_text())
        if d['step'] == 2749:
            out['req055_late_fd_disagreement'].append(dict(
                file=p.name, relative_difference=abs(d['vHv_fd']-d['vHv_scan'])/abs(d['vHv_fd'])))

    # Independently execute every existing fixture-free CPU test (pytest optional).
    test_dir = LOGS / 'req019_fw_calibration/impl'
    sys.path.insert(0, str(test_dir))
    spec = importlib.util.spec_from_file_location('fw_tests', test_dir/'test_generalized_sharpness_fw.py')
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    out['existing_fw_tests'] = {}
    for name in sorted(dir(tests)):
        if name.startswith('test_'):
            getattr(tests, name)()
            out['existing_fw_tests'][name] = 'PASS'

    # Same Euclidean spectrum, different matrix-spectral sharpness.
    spectra, sharpness = [], []
    for a in (torch.diag(torch.tensor([1., 0.])), torch.eye(2)/math.sqrt(2)):
        h = torch.outer(a.flatten(), a.flatten()) + 0.1*torch.eye(4)
        spectra.append(torch.linalg.eigvalsh(h))
        witness = torch.eye(2).flatten()
        attained = float(witness @ h @ witness)
        upper = float(torch.linalg.matrix_norm(a, 'nuc')**2 + 0.2)
        assert abs(attained-upper) < 1e-12
        sharpness.append(attained)
    assert torch.allclose(spectra[0], spectra[1])
    assert abs(sharpness[0]-1.2) < 1e-12 and abs(sharpness[1]-2.2) < 1e-12
    out['same_euclidean_spectrum'] = spectra[0].tolist()
    out['different_spectral_sharpness'] = sharpness

    # Two scalar blocks: isolated maxima sum to 2, joint maximum is 3.8.
    h = torch.tensor([[1., .9], [.9, 1.]])
    d = torch.ones(2)
    out['cross_layer_example'] = dict(isolated_sum=2., joint=float(d@h@d))
    assert abs(float(d@h@d)-3.8) < 1e-12
    # Ball and sphere are different on a negative-definite Hessian.
    out['negative_identity_example'] = dict(euclidean_sphere_max=-1., ball_max=0.)

    # Loss scaling: S/G invariant, lambda/g^2 is not.
    g = torch.tensor([[2., .2], [.1, 1.]])
    s = 2.2
    baseline_ratio = s/float(torch.linalg.matrix_norm(g, 'nuc'))
    scale_rows = []
    for scale in (.1, 1., 10.):
        cg = scale*g
        ratio = scale*s/float(torch.linalg.matrix_norm(cg, 'nuc'))
        assert torch.allclose(polar(cg), polar(g), atol=1e-12)
        assert abs(ratio-baseline_ratio) < 1e-12
        gauge = scale*1.1/float((cg*cg).sum())
        scale_rows.append(dict(scale=scale, spectral_ratio=ratio, old_ratio=gauge))
    out['loss_scaling'] = scale_rows
    # Memory can point uphill even though the nuclear norm is positive.
    g = torch.diag(torch.tensor([2., 1.]))
    q = torch.diag(torch.tensor([-2., 1.]))
    alignment = float((g*polar(q)).sum())
    assert alignment == -1.
    out['momentum_alignment_counterexample'] = dict(true_gradient_nuclear=3., downhill_alignment=alignment)

    # Clone-history mean-age control, including the outer direct-gradient blend.
    source = LOGS/'req054_agematched_ema/make_req054_configs.py'
    constants = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    betas = constants['DECAYS']; sw = constants['SW']; ew = constants['EW']
    nu = .95
    mass = moment = 0.
    for _ in range(2001):  # production baseline update at switch, then clone
        moment, mass = .95*(moment+mass), .95*mass+.05
    masses = [mass]*8; moments = [moment]*8
    scheduled_mass, scheduled_moment = mass, moment
    max_error, max_old_error, max_mass_difference = 0., 0., 0.
    solved = []
    for t in range(2001, 2750):
        alpha = (t-2000)/750
        weights = [a+alpha*(b-a) for a,b in zip(sw,ew)]
        weights = [w/sum(weights) for w in weights]
        for k,b in enumerate(betas):
            moments[k], masses[k] = b*(moments[k]+masses[k]), b*masses[k]+1-b
        target_mass = (1-nu)+nu*sum(w*m for w,m in zip(weights,masses))
        target_age = nu*sum(w*p for w,p in zip(weights,moments))/target_mass
        b = target_age/(nu*(moment+mass+target_age*(1-mass)))
        assert 0 <= b < 1
        solved.append(b)
        moment, mass = b*(moment+mass), b*mass+1-b
        actual_mass = (1-nu)+nu*mass
        actual_age = nu*moment/actual_mass
        max_error = max(max_error, abs(actual_age-target_age))
        max_mass_difference = max(max_mass_difference, abs(actual_mass-target_mass))
        stationary_age = sum(w*b/(1-b) for w,b in zip(weights,betas))
        old_b = stationary_age/(1+stationary_age)
        scheduled_moment, scheduled_mass = old_b*(scheduled_moment+scheduled_mass), old_b*scheduled_mass+1-old_b
        old_age = nu*scheduled_moment/((1-nu)+nu*scheduled_mass)
        max_old_error = max(max_old_error, abs(old_age-target_age))
    assert max_error < 1e-10 and max_mass_difference < 1e-12
    out['cloned_muon_age_toy'] = dict(max_exact_age_error=max_error,
        max_mass_difference=max_mass_difference, beta_range=[min(solved),max(solved)],
        max_stationary_control_age_error=max_old_error,
        note='Scalar clone-history replay, normalized published weights; not an inspection of saved optimizer tensors.')

    # Exact quadratic loss accounting along an arbitrary step with interactions.
    h = torch.tensor([[2., .5], [.5, 1.]])
    w = torch.tensor([.3, -.7]); delta = torch.tensor([-.1, .2])
    grad = h@w
    observed = .5*((w+delta)@h@(w+delta)-w@h@w)
    predicted = grad@delta + .5*delta@h@delta
    assert abs(float(observed-predicted)) < 1e-12
    out['quadratic_loss_identity_error'] = abs(float(observed-predicted))
    out['status'] = 'PASS'
    (HERE/'verification.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
