"""CPU test: CaptureAdamW trajectory == torch.optim.AdamW (decoupled); u_t == -delta/lr (wd=0)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req072_capture_adamw import CaptureAdamW
torch.manual_seed(0); ok=True
for b1 in (0.9, 0.0):
    p1=torch.nn.Parameter(torch.randn(8,4)); p2=torch.nn.Parameter(p1.detach().clone())
    o1=CaptureAdamW([p1],lr=0.01,weight_decay=0.0,betas=(b1,0.95),eps=1e-10); o1._req072_ids={id(p1)}
    o2=torch.optim.AdamW([p2],lr=0.01,weight_decay=0.0,betas=(b1,0.95),eps=1e-10,fused=False)
    maxd=utd=0.0
    for _ in range(20):
        gg=torch.randn(8,4); p1.grad=gg.clone(); p2.grad=gg.clone(); before=p1.detach().clone()
        o1.step(); o2.step()
        maxd=max(maxd,(p1.detach()-p2.detach()).abs().max().item())
        utd=max(utd,(o1._req072_u[id(p1)]-(-(p1.detach()-before)/0.01)).abs().max().item())
        p1.grad=p2.grad=None
    print(f"b1={b1}: traj {maxd:.1e} u_t {utd:.1e} {'PASS' if maxd<1e-5 and utd<1e-4 else 'FAIL'}")
    ok &= maxd<1e-5 and utd<1e-4
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
