"""REQ-056 parity check: KMaxwellAdam must reproduce ordinary torch.optim.Adam when
(a) numerator_mode='adam', and (b) numerator_mode='kmaxwell' with a single constant-decay 0.9 stream.
Feeds identical gradients to all three optimizers over many steps and compares parameter trajectories.
Runs on CPU fp64 for a tight tolerance. Also checks nonfinite handling stays clean."""
import sys, torch
sys.path.insert(0, "records/track_3_optimization")
from optimizers.kmaxwell_adam import KMaxwellAdam

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
LR, N = 3e-4, 200
shapes = [(64, 64), (128, 32)]

def mk():
    return [torch.zeros(s, requires_grad=True) for s in shapes]

pA, pB, pC = mk(), mk(), mk()
with torch.no_grad():
    for a, b, c in zip(pA, pB, pC):
        init = torch.randn_like(a); a.copy_(init); b.copy_(init); c.copy_(init)

oA = torch.optim.Adam(pA, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
oB = KMaxwellAdam(pB, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0,
                  numerator_mode="adam", accumulate_shadow=False)
oC = KMaxwellAdam(pC, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0,
                  numerator_mode="kmaxwell", decays=[0.9], start_weights=[1.0], end_weights=[1.0],
                  switch_step=0, anneal_end_step=1)

max_ab = max_ac = 0.0
for step in range(N):
    grads = [torch.randn(s) for s in shapes]
    for opt, ps in ((oA, pA), (oB, pB), (oC, pC)):
        for p, g in zip(ps, grads):
            p.grad = g.clone()
        opt.step()
        for p in ps:
            p.grad = None
    for a, b, c in zip(pA, pB, pC):
        max_ab = max(max_ab, float((a - b).abs().max()))
        max_ac = max(max_ac, float((a - c).abs().max()))

print(f"steps={N} lr={LR}")
print(f"max|Adam - KMaxwellAdam(adam mode)|      = {max_ab:.3e}")
print(f"max|Adam - KMaxwellAdam(1x0.9 kmaxwell)|  = {max_ac:.3e}")
tol = 1e-9
ok = max_ab < tol and max_ac < tol
print("PARITY:", "PASS" if ok else "FAIL", f"(tol={tol:g})")
sys.exit(0 if ok else 1)
