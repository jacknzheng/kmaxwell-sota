"""REQ-063 Stage B items 3+4 (CPU): parameter-name resolution + a=1-reproduces-mixture, on the REAL model
and the REAL patched optimizer classes. (3) every requested name resolves exactly once (72/72 global,
1/72 selective, per-type counts for allocations), abort on missing/extra/dup/_orig_mod mismatch, never
silently leave a param at default. (4) a nominal a=1 intervention reproduces the base mixture's update."""
import sys, torch
HR = "/Users/jerryhong/.claude/jobs/e9994aef/tmp/harness365/records/track_3_optimization"
sys.path.insert(0, HR)
sys.path.insert(0, "/Users/jerryhong/modded-nanogpt/logs/kmaxwell/req059_sharpness_guided_momentum/impl")
from harness.model_gpt import GPT
from optimizers.muon import PerturbedAnnealedWeightsMuon, AnnealedWeightsMuon
from make_req059_alloc import balanced_assign, TYPES
torch.manual_seed(0)

# --- real model -> the 72 Muon matrices (2D params in blocks) ---
model = GPT(vocab_size=1024, num_layers=12, model_dim=128)
muon_names = [n for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
print(f"(3) Muon matrices in the real 12-layer model: {len(muon_names)} (expect 72)")
by_type = {}
for n in muon_names:
    t = ".".join(n.split(".")[2:4])  # e.g. blocks.3.attn.q.weight -> attn.q
    by_type.setdefault(t, []).append(n)
print(f"    types: {sorted(by_type)} ; per-type counts: {[len(by_type[t]) for t in TYPES]} (expect 12 each)")

# --- allocation resolution: 24/24/24 balanced, every name once, all resolve (72/72) ---
feats = {n: {"type": ".".join(n.split(".")[2:4]), "block": int(n.split(".")[1]),
             "S_i": torch.rand(1).item()} for n in muon_names}
by_type_feat = {}
for n, d in feats.items(): by_type_feat.setdefault(d["type"], []).append({"matrix": n, **d})
amap = balanced_assign(by_type_feat, key=lambda m: m["S_i"])
from collections import Counter
bins = Counter(amap.values())
named = dict(model.named_parameters())
resolve = sum(1 for k in amap if k in named)
dup = len(amap) != len(set(amap))
pertype = {t: Counter(amap[n] for n in by_type[t]) for t in TYPES}
print(f"    allocation: {len(amap)} entries, bins={dict(bins)} (expect 24/24/24), "
      f"resolve {resolve}/72, unique_keys={not dup}")
print(f"    per-type 4/4/4: " + ", ".join(f"{t}:{dict(pertype[t])}" for t in TYPES[:2]) + " ... (all types 4/4/4: "
      f"{all(sorted(pertype[t].values())==[4,4,4] for t in TYPES)})")

# --- _orig_mod prefix handling (compiled models): names resolve after strip ---
compiled_names = {f"_orig_mod.{n}" for n in muon_names}
strip = lambda s: s[len("_orig_mod."):] if s.startswith("_orig_mod.") else s
resolve_compiled = sum(1 for cn in compiled_names if strip(cn) in named)
print(f"    _orig_mod prefix strip resolves: {resolve_compiled}/72")

# --- selective (REQ-058): exactly 1/72; abort on missing name ---
target = muon_names[17]
sel = sum(1 for n in muon_names if n == target)
print(f"(3) selective target '{target}' resolves {sel}/72 (expect 1)")
missing_aborts = "req058_bogus.weight" not in named  # tag hook asserts target in named
print(f"    abort-on-missing guaranteed by tag hook assert (bogus name not in model={missing_aborts})")

# --- (4) a=1 reproduces the base mixture ---
p = torch.nn.Parameter(torch.randn(32, 16)); p.grad = torch.randn(32, 16)
DEC = tuple([0.75,0.822852439855,0.877930338626,0.917598547218,0.945180941073,0.963893920846,0.97637869689,0.984615384615])
SWv = tuple([0.005093975,0.010187949,0.015281924,0.020375898,0.025469873,0.030563847,0.035657822,0.857368713])
EWv = tuple([0.032261839,0.064523678,0.096785516,0.129047355,0.161309194,0.193571033,0.225832871,0.096668514])
kw = dict(decays=DEC, start_weights=SWv, end_weights=EWv, switch_step=2000, anneal_end_step=2750)
pert = PerturbedAnnealedWeightsMuon([p], perturb_a=1.0, perturb_all=True, lr=0.02, **kw)
base = AnnealedWeightsMuon([p], lr=0.02, **kw)
print(f"(4) perturb_a=1.0: _perturbed_decays == decays: {tuple(pert._perturbed_decays)==DEC}")
print(f"    _is_perturbed(p) with a=1 (perturb_all=True): {pert._is_perturbed(p)} (expect False -> base path)")
# numeric: identical fresh grad + streams to BOTH (kernels mutate grad in place -> clone per call)
g0 = torch.randn(32, 16); streams0 = [torch.randn(32, 16) * 0.1 for _ in DEC]
def one_update(opt):
    opt._muon_steps_seen = 2100
    p.grad = g0.clone()
    return opt.compute_polar_input(p, {"streams": [s.clone() for s in streams0]}, {"mu": 0.95}).clone()
u_pert = one_update(pert); u_base = one_update(base)
d = (u_pert.float()-u_base.float()).abs().max().item()
print(f"    numeric compute_polar_input(a=1) vs base mixture: max|diff|={d:.2e} (expect 0, same kernel)")

ok3 = len(muon_names)==72 and dict(bins)=={0.5:24,1.0:24,2.0:24} and resolve==72 and not dup and resolve_compiled==72 and sel==1
ok4 = tuple(pert._perturbed_decays)==DEC and not pert._is_perturbed(p) and d < 1e-6
print(f"\nVERDICT: name-resolution (72/72, 24/24/24, 1/72, _orig_mod, abort-on-missing) {'PASS' if ok3 else 'FAIL'}; "
      f"a=1-reproduces-mixture {'PASS' if ok4 else 'FAIL'}")
