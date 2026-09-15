"""REQ-063 Stage B: reproduce the nomom mu-overwrite through the REAL Muon.load_state_dict path,
demonstrate the fix (reapply declared treatment after load), and verify zero momentum reaches the
post-polar update independent of an inherited momentum buffer at fixed gradient. CPU, real optimizer class."""
import sys, torch
HC = "/Users/jerryhong/.claude/jobs/e9994aef/tmp/harness365/records/track_3_optimization"
sys.path.insert(0, HC)
from optimizers.muon import Muon, muon_update, zeropower_via_newtonschulz5
torch.manual_seed(0)

def mk(mu):
    p = torch.nn.Parameter(torch.randn(8, 4))
    return Muon([p], lr=0.02, mu=mu), p

# --- 1. base mixture-like arm (mu=0.95) with a populated momentum buffer, then dump its state_dict
base, pb = mk(0.95)
base.state[pb]["momentum"] = torch.randn(8, 4)  # inherited nonzero buffer
base._muon_steps_seen = 1500
sd = base.state_dict()
print(f"[dump] base mu={base.param_groups[0]['mu']}, has momentum buffer, steps_seen={base._muon_steps_seen}")

# --- 2. nomom arm (mu=0.0) BEFORE restore
nomom, pn = mk(0.0)
print(f"[pre-load] nomom declared mu={nomom.param_groups[0]['mu']}")
assert nomom.param_groups[0]["mu"] == 0.0

# --- 3. THE REAL RESTORE PATH: built.load_state_dict(sd)  (hooks.py:614)
nomom.load_state_dict(sd)
mu_after = nomom.param_groups[0]["mu"]
print(f"[post-load] nomom mu={mu_after}  <-- BUG: overwritten from 0.0 to {mu_after}" if mu_after != 0.0
      else f"[post-load] nomom mu={mu_after} (unchanged)")
bug_reproduced = (mu_after == 0.95)
print(f"  => nomom mu-overwrite reproduced: {bug_reproduced}")
print(f"  inherited momentum buffer present after load: {'momentum' in nomom.state[pn]}")

# --- 4. THE FIX: reapply declared treatment hyperparameters AFTER load (buffers/counters kept)
DECLARED = {"mu": 0.0}
for g in nomom.param_groups: g.update(DECLARED)
print(f"[fixed] nomom mu={nomom.param_groups[0]['mu']} (declared treatment reapplied; momentum buffer kept="
      f"{'momentum' in nomom.state[pn]}, steps_seen={nomom._muon_steps_seen} inherited)")
assert nomom.param_groups[0]["mu"] == 0.0

# --- 5. zero momentum reaches the post-polar update, independent of the inherited buffer at fixed grad
g = torch.randn(8, 4)
def polar_update(mu, momentum_init):
    st = {"momentum": momentum_init.clone()}
    grp = {"mu": mu}
    # replicate compute_polar_input exactly (muon_update mutates grad in place -> clone)
    return muon_update(g.clone(), st["momentum"], mu=grp["mu"]).clone()

buf_a, buf_b = torch.randn(8, 4), torch.randn(8, 4)  # two different inherited buffers
u0_a, u0_b = polar_update(0.0, buf_a), polar_update(0.0, buf_b)
u95_a, u95_b = polar_update(0.95, buf_a), polar_update(0.95, buf_b)
ref0 = zeropower_via_newtonschulz5(g.clone()) * max(1, 8/4)**0.5
d0 = (u0_a - u0_b).abs().max().item()
d95 = (u95_a - u95_b).abs().max().item()
dref = (u0_a - ref0).abs().max().item()
print(f"[mu=0]   update independent of inherited buffer: max|u_a-u_b|={d0:.2e} (expect ~0)")
print(f"[mu=0]   update == zeropower(g)*scale to bf16: max diff={dref:.2e} (expect <=1 bf16 ULP=7.8e-3)")
print(f"[mu=.95] update DEPENDS on inherited buffer: max|u_a-u_b|={d95:.2e} (expect >0)")
ok = bug_reproduced and d0 < 1e-6 and dref <= 8e-3 and d95 > 1e-6  # dref: 1 bf16 ULP (zeropower is bf16)
print(f"\nVERDICT: overwrite-reproduced={bug_reproduced}, fix-restores-mu0=True, "
      f"zero-mom-post-polar-independent-of-buffer={d0<1e-6 and d95>1e-6}  => {'PASS' if ok else 'FAIL'}")
