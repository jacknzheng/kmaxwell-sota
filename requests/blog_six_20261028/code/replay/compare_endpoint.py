"""Compare the saved October 27 endpoint before measuring the extension."""
import argparse, fcntl, hashlib, json, math, pathlib, subprocess, time
import torch

p = argparse.ArgumentParser()
p.add_argument('--runtime', required=True)
p.add_argument('--arm', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
R, O = pathlib.Path(a.runtime), pathlib.Path(a.output)
torch.set_num_threads(2)
old_node = {'lower_zeta': '57964111', 'higher_zeta': '57964046'}[a.arm]
remote = 'muoff-r2:muoff-research/runs/blog_six_figures_20261027/node_' + old_node
base = f'blog_outputs/blog_seed0_{a.arm}_replay2009_20261027/captures'
D = O / 'endpoint_comparison_inputs'
D.mkdir(exist_ok=True)
def fetch(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rclone', 'copyto', src, str(dst), '--disable-http2'], check=True)
receipt = D / 'reference_receipt.json'
fetch(remote + '/' + base + '/R2_VERIFIED_2039.json', receipt)
refs = json.loads(receipt.read_text())['files']
def equal_control(x, y):
    if torch.is_tensor(x):
        return torch.is_tensor(y) and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and set(x)==set(y) and all(equal_control(x[k], y[k]) for k in x if k!='shard')
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x)==len(y) and all(equal_control(u,v) for u,v in zip(x,y))
    return x==y
actual = {d['file']: d for d in map(json.loads, (O/'captures/index.jsonl').read_text().splitlines()) if d.get('kind') == 'boundary' and d['update'] == 2039}
rows = []
lock = (O/'captures/.point_2039.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
for rank in range(8):
    rel = f'rank{rank}/boundary_002039.pt'
    ref = next(x for x in refs if x['file'].endswith('/' + rel))
    oldpath, newpath = D/f'old_rank{rank}.pt', O/'captures'/rel
    fetch(remote + '/' + ref['file'], oldpath)
    if not newpath.exists():
        archived = json.loads((O/'captures/R2_VERIFIED_2039.json').read_text())
        fetch(archived['remote'] + '/' + str(newpath.relative_to(R)), newpath)
    for path, pin in [(oldpath, ref), (newpath, actual[rel])]:
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == pin['sha256']
    old = torch.load(oldpath, map_location='cpu', weights_only=False, mmap=True)
    new = torch.load(newpath, map_location='cpu', weights_only=False, mmap=True)
    comparisons = {}
    for key in ['model_before', 'post_insertion_prepolar_momentum', 'polar_output', 'executed_update']:
        if key not in old and key not in new:
            continue
        x, y = old[key], new[key]
        assert set(x) == set(y)
        norm = delta_norm = maximum = identical = 0
        for name in x:
            u, v = x[name].double(), y[name].double()
            delta = v-u
            norm += float(u.square().sum())
            delta_norm += float(delta.square().sum())
            maximum = max(maximum, float(delta.abs().max()))
            identical += int(torch.equal(x[name], y[name]))
        comparisons[key] = dict(relative_l2=math.sqrt(delta_norm/norm) if norm else None, difference_l2=math.sqrt(delta_norm), max_abs=maximum, identical_tensors=identical, tensors=len(x))
    controls = {}
    for key in ['settings','kernel_weights_used','muon_steps_seen_before','anneal_frac_used','anneal_frac_next','batch_receipts']:
        if key in old and key in new:
            controls[key] = equal_control(old[key],new[key])
        elif key not in old and key not in new:
            controls[key] = 'not_stored_in_either_shard'
        else:
            controls[key] = dict(reference_present=key in old, replay_present=key in new)
    rows.append(dict(rank=rank, file_identical=ref['sha256']==actual[rel]['sha256'], tensors=comparisons, equal_controls=controls, batch_comparison_ignores_only_absolute_shard_path=True))
    del old, new
    oldpath.unlink()
report = dict(update=2039, time=time.time(), reference=remote+'/'+base, scope='All eight capture shards, including global model and all owned Muon actions', comparisons=rows, strict_veto=False)
(O/'OVERLAP_COMPARISON_TENSORS_2039.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(stage='endpoint_compared', update=2039, ranks=len(rows))), flush=True)
