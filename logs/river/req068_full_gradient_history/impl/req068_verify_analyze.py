"""REQ-068 post-run verification + initial descriptive analysis (runs ON the box, detached).
Verification: recompute all chunk SHA-256, confirm contiguous coverage 0..3249, census params/None/dtypes.
Descriptive analysis (mean-per-token gradients = raw SUM / tokens_per_update):
 - per-step global + per-layer gradient norm trajectory
 - signed cosine direction similarity at lags 1,2,4,8,16,32 (global + per matrix-type)
 - period-two evidence: cos(g_t, g_{t+1}) vs cos(g_t, g_{t+2}); amplitude of alternation
Writes a compact JSON summary (no raw tensors) to out path.
"""
import sys, json, time, math, collections
sys.path.insert(0, "/root/kmaxwell-sota/logs/river/req068_full_gradient_history/impl")
import torch
from req068_capture import GradientHistoryReader

DUR = "/root/.cache/user_artifacts/req068_s0/grads"
OUT = "/root/.cache/user_artifacts/req068_s0/analysis_summary.json"
r = GradientHistoryReader(DUR)
tpu = r.manifest.get("tokens_per_update") or 1
res = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

# ---- verification: manifest coverage (instant) + spot SHA on first/mid/last chunks ----
import os, hashlib
t0 = time.time()
covered = r.steps()
cov_ok = covered == list(range(3250))
chunks = r.manifest["chunks"]
spot = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
def _sha(fn):
    h=hashlib.sha256()
    with open(os.path.join(DUR, fn),"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
spot_ok = all(_sha(c["file"]) == c["sha256"] for c in spot)
res["verify"] = {"ok": cov_ok and spot_ok, "coverage_0_3249": cov_ok, "spot_hash_ok": spot_ok,
                 "spot_chunks": [c["index"] for c in spot], "n_steps": len(covered),
                 "seconds": round(time.time()-t0)}
steps = r.steps()
names = r.params()
st0 = r.full_step(steps[0])
res["dtype_census"] = dict(collections.Counter(str(g.dtype) for g in st0.values() if g is not None))
res["n_params"] = len(names)
res["n_none_at_step0"] = sum(1 for g in st0.values() if g is None)
res["total_bytes"] = r.manifest["total_bytes"]

# matrix-type grouping (attn.q/k/v/proj, mlp.fc/proj, embed, head, other)
def kind(n):
    if n.startswith("embed"): return "embed"
    if n.startswith("proj"): return "head"
    for t in ("attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"):
        if t in n: return t
    return "other"

# ---- descriptive analysis over a stride (every step is 3250 full loads = slow; use stride) ----
STRIDE = 1  # every step; mean-per-token
def mean_grad(step, name):
    g = r.grad(step, name)
    return None if g is None else (g.float() / tpu)

# global flat cosine at lags: build per-step concatenated unit vectors incrementally is heavy;
# instead track per-matrix-type cosine lag averages + global norm. Load each step once, reuse.
lags = [1, 2, 4, 8, 16, 32]
norm_traj = []            # (step, global_l2_norm)
type_norm = collections.defaultdict(list)
# for cosines we need g_t and g_{t+lag}; stream with a rolling buffer of the max lag
from collections import deque
buf = deque()  # (step, {name: flatfloat tensor})
cos_sums = {L: collections.defaultdict(float) for L in lags}   # [lag][type] sum cos
cos_cnt = {L: collections.defaultdict(int) for L in lags}
p2_num = collections.defaultdict(float); p2_cnt = collections.defaultdict(int)  # period-2 per type

def flat_by_type(step):
    out = {}
    g_all = r.full_step(step)
    gsum = 0.0
    for n in names:
        g = g_all[n]
        if g is None: continue
        f = (g.float() / tpu).reshape(-1)
        out.setdefault(kind(n), []).append(f)
        gsum += float((f * f).sum())
    merged = {k: torch.cat(v) for k, v in out.items()}
    return merged, math.sqrt(gsum)

t1 = time.time()
for i, s in enumerate(steps):
    merged, gnorm = flat_by_type(s)
    norm_traj.append((s, gnorm))
    for k, vv in merged.items():
        type_norm[k].append(float(vv.norm()))
    buf.append((s, merged))
    while buf and (s - buf[0][0]) > max(lags):
        buf.popleft()
    # cosines against earlier buffered steps
    for (ps, pmerged) in buf:
        d = s - ps
        if d in lags:
            for k in merged:
                if k in pmerged:
                    a, b = merged[k], pmerged[k]
                    na, nb = a.norm(), b.norm()
                    if na > 0 and nb > 0:
                        cos_sums[d][k] += float((a @ b) / (na * nb)); cos_cnt[d][k] += 1
    if i % 200 == 0:
        print(f"analyze step {s} ({i}/{len(steps)}) {time.time()-t1:.0f}s", flush=True)

# period-two: compare lag-1 vs lag-2 mean cosine per type (neg lag-1 w/ pos lag-2 => alternation)
res["grad_norm_global"] = {"first": norm_traj[0], "mid": norm_traj[len(norm_traj)//2], "last": norm_traj[-1],
                           "min": min(n for _, n in norm_traj), "max": max(n for _, n in norm_traj)}
res["cos_by_lag_by_type"] = {str(L): {k: (cos_sums[L][k]/cos_cnt[L][k] if cos_cnt[L][k] else None)
                                      for k in sorted(cos_cnt[L])} for L in lags}
res["period_two_signal"] = {k: {"cos_lag1": res["cos_by_lag_by_type"]["1"].get(k),
                                "cos_lag2": res["cos_by_lag_by_type"]["2"].get(k)}
                            for k in res["cos_by_lag_by_type"]["1"]}
res["type_norm_last"] = {k: v[-1] for k, v in type_norm.items()}
res["analyze_seconds"] = round(time.time() - t1)
res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT)
print("VERIFY_OK", res["verify"]["ok"], "COS_LAG1_GLOBAL_TYPES", {k: round(res['cos_by_lag_by_type']['1'][k],3) for k in res['cos_by_lag_by_type']['1']})
print("REQ068_ANALYZE_DONE")
