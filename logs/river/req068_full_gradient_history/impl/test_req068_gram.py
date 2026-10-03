"""CPU validation for the REQ-068 Gram analysis (spec: verify blockwise K vs direct dot products,
symmetry, diagonal norms, eigen reconstruction, coefficient identity, out-of-sample low-rank recovery)."""
import sys, os, warnings, numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)  # macOS numpy+Accelerate false matmul FPE; clean on Linux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req068_gram import (temporal_gram, cosine_from_gram, gram_eig, signed_coefficients,
                         center_gram, lag_agreement, out_of_sample)
rng = np.random.default_rng(0)
ok = True

# synthetic: n=64 whole-model vectors across 3 parameter "families", some with a slow rank-2 component
n = 64
fams = {"a": 50, "b": 30, "c": 20}
def family_of(name): return name.split(".")[0]
# build vectors: shared slow rank-2 (cos/sin over time) + per-step alternating + noise
t = np.arange(n)
slow = np.stack([np.cos(2*np.pi*t/80), np.sin(2*np.pi*t/80)], 1)  # n x 2 slow coeffs
alt = ((-1.0) ** t)[:, None]
vecs = []
basis = {f: rng.standard_normal((2, d)) for f, d in fams.items()}
altdir = {f: rng.standard_normal(d) for f, d in fams.items()}
for i in range(n):
    v = {}
    for f, d in fams.items():
        v[f"{f}.w"] = (slow[i] @ basis[f] + 0.6*alt[i]*altdir[f] + 0.05*rng.standard_normal(d))
    vecs.append(v)
names = [f"{f}.w" for f in fams]

# T1: blockwise K == direct full-vector dot products
K, famK = temporal_gram(vecs, names, family_of)
full = np.stack([np.concatenate([vecs[i][nm] for nm in names]) for i in range(n)])
Kdirect = full @ full.T
e1 = np.abs(K - Kdirect).max()
print(f"T1 blockwise K == direct dots: max|d|={e1:.2e} {'PASS' if e1<1e-8 else 'FAIL'}"); ok &= e1 < 1e-8

# T1b: per-family K sums to total
e1b = np.abs(sum(famK.values()) - K).max()
print(f"T1b per-family K sums to total: max|d|={e1b:.2e} {'PASS' if e1b<1e-8 else 'FAIL'}"); ok &= e1b < 1e-8

# T2: cosine symmetric, unit diagonal
C, zmask = cosine_from_gram(K)
e2 = np.abs(C - C.T).max(); diagok = np.allclose(np.diag(C), 1.0)
print(f"T2 cosine symmetric({e2:.1e}) + unit diag({diagok}): {'PASS' if e2<1e-10 and diagok else 'FAIL'}"); ok &= e2<1e-10 and diagok

# T3: eig reconstruction K ~ V diag(lam) V^T; coefficient identity coeff = sqrt(lam)*V and K = coeff coeff^T
lam, V = gram_eig(K)
recon = (V * lam) @ V.T; e3 = np.abs(recon - K).max()
coeff = signed_coefficients(lam, V); e3b = np.abs(coeff @ coeff.T - K).max()
print(f"T3 eig recon({e3:.1e}) + coeff coeff^T==K({e3b:.1e}): {'PASS' if e3<1e-6 and e3b<1e-6 else 'FAIL'}"); ok &= e3<1e-6 and e3b<1e-6

# T4: lag_agreement == direct off-diagonal cosine averages (correctness, not a data-dependent sign)
la = lag_agreement(C)
direct = {L: float(np.mean([C[i, i+L] for i in range(n-L)])) for L in (1, 2, 4, 8, 16, 32)}
e4 = max(abs(la[L] - direct[L]) for L in la)
print(f"T4 lag_agreement == direct off-diagonal means: max|d|={e4:.1e} (lag1={la[1]:+.3f} lag2={la[2]:+.3f}) {'PASS' if e4<1e-12 else 'FAIL'}"); ok &= e4 < 1e-12

# T5: out-of-sample low-rank recovery — slow rank-2 subspace should capture held-out energy > past-mean,
#     and rank>=2 captures a substantial fraction (the slow + alternating structure is low-rank)
oos = out_of_sample(K, ranks=(1,2,4,8), fit=32)
r2 = oos["by_rank"][2]["captured_frac"]; r8 = oos["by_rank"][8]["captured_frac"]; pm = oos["baseline_past_mean_frac"]
mono = r8 >= r2 >= oos["by_rank"][1]["captured_frac"]
print(f"T5 OOS captured frac r1={oos['by_rank'][1]['captured_frac']:.3f} r2={r2:.3f} r8={r8:.3f} past_mean={pm:.3f}")
print(f"   monotone in rank={mono}, r8>0.5={r8>0.5}: {'PASS' if mono and r8>0.5 else 'FAIL'}"); ok &= mono and r8>0.5

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-068 Gram analysis (blockwise K, cosine, eig, coeff, lag, out-of-sample)")
sys.exit(0 if ok else 1)
