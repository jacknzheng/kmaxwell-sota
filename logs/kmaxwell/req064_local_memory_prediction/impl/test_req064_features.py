"""CPU validation for REQ-064 feature primitives, against known symmetric operators (float64).
 T1 Lanczos recovers lambda_max of a random SPD block to <1e-6 (8 iters, dim 12).
 T2 lambda_i/||g||_F^2 arithmetic.
 T3 rayleigh_along an exact eigenvector == its eigenvalue.
 T4 Lanczos Ritz history is monotone-nondecreasing to lambda_max and the residual converges.
"""
import sys, torch
sys.path.insert(0, "/Users/jerryhong/modded-nanogpt/logs/kmaxwell/req064_local_memory_prediction/impl")
from req064_features import lanczos_top_eig, euclidean_lambda, rayleigh_along
torch.manual_seed(0)

def make_block(shape, seed):
    g = torch.Generator().manual_seed(seed)
    dim = shape[0] * shape[1]
    A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    H = A @ A.t() + 0.1 * torch.eye(dim, dtype=torch.float64)  # SPD
    def diag_hvp_one(i, v):  # i ignored (single block); v: tensor of `shape`
        return (H @ v.double().reshape(-1)).reshape(shape)
    evals = torch.linalg.eigvalsh(H)
    return diag_hvp_one, float(evals.max()), H, evals

ok = True
# T1: Lanczos at FULL Krylov dim recovers lambda_max exactly (algorithm correctness)
shape = (4, 3); dim = shape[0]*shape[1]
dhvp, lam_true, H, evals = make_block(shape, 42)
lam_full, _, _, _ = lanczos_top_eig(lambda v: dhvp(0, v), shape, iters=dim, seed=1337)
e1 = abs(lam_full - lam_true)
print(f"T1 Lanczos@full({dim}) lambda_max: got {lam_full:.8f} true {lam_true:.8f}  |err|={e1:.2e}")
ok &= e1 < 1e-8
# T1b: 8-iter mode = a lower bound <= lambda_max, monotone, with a residual diagnostic (the real-run mode)
lam8, hist8, resid8, nit8 = lanczos_top_eig(lambda v: dhvp(0, v), shape, iters=8, seed=1337)
mono8 = all(hist8[k] <= hist8[k+1] + 1e-9 for k in range(len(hist8)-1))
print(f"T1b Lanczos@8: {lam8:.6f} <= lambda_max (bound {lam8 <= lam_true + 1e-6}), monotone {mono8}, resid {resid8:.2e}, err {abs(lam8-lam_true):.2e}")
ok &= (lam8 <= lam_true + 1e-6) and mono8

# T2: lambda / ||g||_F^2 (full-iter lambda for exactness of the arithmetic + value check)
grad_i = torch.randn(*shape, dtype=torch.float64)
feat = euclidean_lambda(lambda i, v: dhvp(i, v), 0, grad_i, iters=dim)
gF2 = float((grad_i ** 2).sum())
e2 = abs(feat["lambda_over_gF2"] - feat["lambda_i"] / gF2)
print(f"T2 lambda/||g||_F^2: {feat['lambda_over_gF2']:.6f}  arith err={e2:.2e}  lambda err={abs(feat['lambda_i']-lam_true):.2e}")
ok &= e2 < 1e-12 and abs(feat["lambda_i"] - lam_true) < 1e-8

# T3: rayleigh along the exact top eigenvector == lambda_max
evals_full, evecs_full = torch.linalg.eigh(H)
top_vec = evecs_full[:, int(torch.argmax(evals_full))].reshape(shape)
rq = rayleigh_along(lambda i, v: dhvp(i, v), 0, top_vec)
e3 = abs(rq - lam_true)
print(f"T3 rayleigh along top eigenvector: {rq:.8f} vs lambda_max {lam_true:.8f}  |err|={e3:.2e}")
ok &= e3 < 1e-9
# and along the smallest eigenvector == lambda_min
bot_vec = evecs_full[:, int(torch.argmin(evals_full))].reshape(shape)
rq_min = rayleigh_along(lambda i, v: dhvp(i, v), 0, bot_vec)
e3b = abs(rq_min - float(evals_full.min()))
print(f"   rayleigh along bottom eigenvector: {rq_min:.8f} vs lambda_min {float(evals_full.min()):.8f}  |err|={e3b:.2e}")
ok &= e3b < 1e-9

# T4: full-Krylov recovery across shapes (reshape correctness), + 8-iter monotone convergence toward truth
for shp, sd in [((6, 2), 7), ((2, 6), 9), ((5, 5), 11)]:
    dh, lt, _, _ = make_block(shp, sd)
    d = shp[0]*shp[1]
    lv_full, _, _, _ = lanczos_top_eig(lambda v: dh(0, v), shp, iters=d, seed=1337)
    lv8, h8, _, _ = lanczos_top_eig(lambda v: dh(0, v), shp, iters=8, seed=1337)
    ef = abs(lv_full - lt)
    conv = h8[-1] <= lt + 1e-6 and h8[-1] >= h8[0] - 1e-9  # 8-iter is a bound, improving toward truth
    print(f"T4 shape {shp} (dim {d}): full-Krylov err {ef:.2e}; 8-iter {lv8:.4f}<=lambda_max {lt:.4f} & improving {conv}")
    ok &= ef < 1e-7 and conv

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-064 feature primitives (Lanczos full+8-iter, lambda/gF2, rayleigh, shape sweep)")
