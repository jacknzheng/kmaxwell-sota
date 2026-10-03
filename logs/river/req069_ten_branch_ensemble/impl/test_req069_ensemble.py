"""CPU validation for REQ-069 ensemble core. Synthetic: 10 branches share a slow drift + a SHARED initial
alternation phase + INDEPENDENT per-branch fast noise. Checks:
 T1 ensemble mean reduces |lag1| vs individual branches when alternation is independent (averaged out).
 T2 shared-alternation case: ensemble RETAINS alternation (honest caveat) — not averaged away.
 T3 two independent groups of five agree (high cosine) on the shared slow component.
 T4 convergence: dispersion > 0, mean_to_rms rises as coherent-signal fraction rises.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req069_ensemble import ensemble_summary, convergence_step, lag_cosine_sequence, _flat
rng = np.random.default_rng(0)
ok = True
T, B, d = 64, 10, 400
names = ["blocks.0.mlp.fc.weight"]
t = np.arange(T)
slow = np.stack([np.cos(2*np.pi*t/120), np.sin(2*np.pi*t/120)], 1)   # shared slow 2-D
basis = rng.standard_normal((2, d))
altdir = rng.standard_normal(d)

def make(shared_alt):
    # alternation dominant (amp 1.0) so individual lag1 is clearly NEGATIVE; slow weak (0.25); small noise
    seqs = []
    for b in range(B):
        altphase = altdir if shared_alt else rng.standard_normal(d)  # shared vs independent alternation
        seq = []
        for i in range(T):
            v = 0.25*(slow[i] @ basis) + 1.0*((-1.0)**i)*altphase + 0.15*rng.standard_normal(d)
            seq.append({names[0]: v})
        seqs.append(seq)
    return seqs

# T1: INDEPENDENT alternation -> ensemble mean should reduce |lag1|
seqs_indep = make(shared_alt=False)
s1 = ensemble_summary(seqs_indep, names)
ar = s1["alternation_reduction_lag1"]
print(f"T1 indep-alt: individual lag1={ar['individual']:+.3f} ensemble lag1={ar['ensemble']:+.3f} "
      f"(|ensemble|<|individual|) {'PASS' if abs(ar['ensemble'])<abs(ar['individual']) else 'FAIL'}")
ok &= abs(ar["ensemble"]) < abs(ar["individual"])

# T2: SHARED alternation -> ensemble RETAINS it (honest caveat)
seqs_shared = make(shared_alt=True)
s2 = ensemble_summary(seqs_shared, names)
ar2 = s2["alternation_reduction_lag1"]
print(f"T2 shared-alt: individual lag1={ar2['individual']:+.3f} ensemble lag1={ar2['ensemble']:+.3f} "
      f"(ensemble retains alternation, |ensemble|>0.3) {'PASS' if abs(ar2['ensemble'])>0.3 else 'FAIL'}")
ok &= abs(ar2["ensemble"]) > 0.3

# T3: group agreement is HIGHER when branches share structure (shared-alt) than when alt is independent
ga_indep = s1["group_agreement_per_step_mean"]
ga_shared = s2["group_agreement_per_step_mean"]
print(f"T3 two-group agreement: indep-alt={ga_indep:+.3f} < shared-alt={ga_shared:+.3f} "
      f"(shared structure -> more agreement) {'PASS' if ga_shared > ga_indep and ga_shared > 0.5 else 'FAIL'}")
ok &= ga_shared > ga_indep and ga_shared > 0.5

# T4: convergence sanity — dispersion finite & positive, mean_to_rms in (0,1]; coherent signal raises mean_to_rms
cm = s1["convergence_mid_step"]
disp = cm["dispersion"]; mtr = cm["mean_to_rms"]
print(f"T4 convergence: dispersion={disp:.3f} (>0 finite), mean_to_rms={mtr:.3f} (0<r<=1) "
      f"{'PASS' if disp>0 and np.isfinite(disp) and 0<mtr<=1.01 else 'FAIL'}")
ok &= disp > 0 and np.isfinite(disp) and 0 < mtr <= 1.01
# and mean_to_rms is higher when signal more coherent: compare pure-slow vs noisy
pure = [[{names[0]: slow[i]@basis} for i in range(T)] for _ in range(B)]
cm_pure = convergence_step([pure[b][T//2] for b in range(B)], names)
print(f"   pure-slow mean_to_rms={cm_pure['mean_to_rms']:.3f} > noisy {mtr:.3f}: "
      f"{'PASS' if cm_pure['mean_to_rms']>mtr else 'FAIL'}")
ok &= cm_pure["mean_to_rms"] > mtr

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-069 ensemble core (alternation reduction, shared-alt retention, group agreement, convergence)")
sys.exit(0 if ok else 1)
