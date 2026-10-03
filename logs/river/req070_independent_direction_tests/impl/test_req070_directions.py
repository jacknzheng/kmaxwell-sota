"""CPU validation for REQ-070 direction extraction + perturbation."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req070_directions import extract_candidates, perturbation, unit_norm, _axpy_mean
rng = np.random.default_rng(0)
names = ["w"]; B = 10; T = 64; d = 50
# synthetic per-branch disp + grad sequences
disp = {b: {t: {"w": rng.standard_normal(d)} for t in range(T)} for b in range(B)}
grad = {b: {t: {"w": rng.standard_normal(d)} for t in range(T)} for b in range(B)}
disp_cb = [(lambda b: (lambda t: disp[b][t]))(b) for b in range(B)]
grad_cb = [(lambda b: (lambda t: grad[b][t]))(b) for b in range(B)]
ok = True

o = 48; br = 3
c = extract_candidates(disp_cb, grad_cb, names, br, o, temporal_k=8)

# T1 baseline_update == delta_{o-1} of branch br
e1 = np.abs(c["baseline_update"]["w"] - disp[br][o-1]["w"]).max()
print(f"T1 baseline_update==delta_(o-1): {e1:.1e} {'PASS' if e1<1e-12 else 'FAIL'}"); ok &= e1<1e-12
# T2 neg_gradient == -g_o
e2 = np.abs(c["neg_gradient"]["w"] + grad[br][o]["w"]).max()
print(f"T2 neg_gradient==-g_o: {e2:.1e} {'PASS' if e2<1e-12 else 'FAIL'}"); ok &= e2<1e-12
# T3 temporal_avg == mean delta_{o-8..o-1}
exp = np.mean([disp[br][j]["w"] for j in range(o-8, o)], axis=0)
e3 = np.abs(c["temporal_avg"]["w"] - exp).max()
print(f"T3 temporal_avg==mean(delta_o-8..o-1): {e3:.1e} {'PASS' if e3<1e-12 else 'FAIL'}"); ok &= e3<1e-12
# T4 ensemble_mean_LOO excludes branch br, averages others' delta_(o-1)
exp_loo = np.mean([disp[b][o-1]["w"] for b in range(B) if b != br], axis=0)
e4 = np.abs(c["ensemble_mean_LOO"]["w"] - exp_loo).max()
own_excluded = not np.allclose(c["ensemble_mean_LOO"]["w"], np.mean([disp[b][o-1]["w"] for b in range(B)], axis=0))
print(f"T4 ensemble LOO excludes own branch: match={e4:.1e} own_excluded={own_excluded} "
      f"{'PASS' if e4<1e-12 and own_excluded else 'FAIL'}"); ok &= e4<1e-12 and own_excluded
# T5 perturbation lengths: ||step|| == alpha * baseline_norm; alpha=0 -> unchanged; zero-dir flagged
theta = rng.standard_normal(d); bn = unit_norm(c["baseline_update"], names)
dvec = c["neg_gradient"]["w"]
for alpha in (0, 0.25, 0.5, 1, 2):
    p, zf = perturbation(theta, dvec, alpha, bn)
    steplen = np.linalg.norm(p - theta)
    want = alpha * bn
    good = abs(steplen - want) < 1e-9
    if alpha == 0: good = good and np.allclose(p, theta)
    if not good: print(f"   T5 alpha={alpha}: steplen {steplen:.4f} != {want:.4f} FAIL"); ok = False
pz, zf = perturbation(theta, np.zeros(d), 1.0, bn)
print(f"T5 perturbation lengths exact + alpha0 unchanged + zero-dir flagged({zf}): {'PASS' if zf and ok else ('PASS' if ok else 'FAIL')}")
ok &= zf

# T6 causality guard: offset 0 rejected
try:
    extract_candidates(disp_cb, grad_cb, names, br, 0); print("T6 offset0 guard: FAIL (no error)"); ok=False
except AssertionError:
    print("T6 offset0 causality guard: PASS")

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-070 direction extraction + perturbation")
sys.exit(0 if ok else 1)
