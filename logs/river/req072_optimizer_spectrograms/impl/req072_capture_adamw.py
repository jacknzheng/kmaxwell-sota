"""REQ-072 CaptureAdamW: a transparent (non-fused) decoupled AdamW that exposes the conditioned direction
u_t = m_hat / (sqrt(v_hat) + eps) for selected parameters BEFORE the learning-rate multiply and decoupled
weight decay. Mathematically identical trajectory to torch.optim.AdamW (decoupled, same betas/eps) so the
study's AdamW arms are a faithful AdamW; it only additionally stashes u_t for the tagged matrices.

beta1=0 gives the no-first-moment arm (keeps the sqrt(v) adaptive scale, per spec). Standalone + CPU-tested.
"""
from __future__ import annotations
import torch


class CaptureAdamW(torch.optim.Optimizer):
    def __init__(self, params, *, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-10):
        super().__init__(list(params), dict(lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps))
        self._req072_ids = set()        # id(param) -> stash u_t
        self._req072_u = {}             # id(param) -> cpu fp32 u_t of the last step

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]; eps = g["eps"]; lr = g["lr"]; wd = g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, dtype=torch.float32)
                    st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1; t = st["step"]
                grad = p.grad.float()
                st["m"].mul_(b1).add_(grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                mhat = st["m"] / (1 - b1 ** t) if b1 > 0 else st["m"]   # b1=0 -> m==grad, no bias corr needed
                vhat = st["v"] / (1 - b2 ** t)
                u = mhat / (vhat.sqrt() + eps)                           # conditioned direction (pre-LR)
                if id(p) in self._req072_ids:
                    self._req072_u[id(p)] = u.detach().to("cpu", torch.float32).clone()
                if wd != 0:
                    p.mul_(1 - lr * wd)                                  # decoupled weight decay
                p.add_(u.to(p.dtype), alpha=-lr)
