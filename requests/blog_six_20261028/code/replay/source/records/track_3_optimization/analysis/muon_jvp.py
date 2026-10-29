import torch
from torch import Tensor


def frozen_ns_jvp(momentum: Tensor):
    """Linear tangent propagation through the actual twelve BF16 NS primals.

    Record the optimizer's rounded primal intermediates exactly, then propagate
    tangents in FP32 (FP64 for FP64 input). Cast/rounding use the usual identity
    AD convention, but do NOT round the tangent at every primitive. Native BF16
    forward AD was measurably non-additive and is unsuitable as an Arnoldi map.
    This is the NS derivative, not ideal polar or a different NS polynomial.
    """
    tall = momentum.shape[0] > momentum.shape[1]
    g = momentum.detach().bfloat16()
    if tall:
        g = g.mT
    norm = g.norm(dim=(-2, -1), keepdim=True)
    denominator = norm + 1e-7
    if not bool(norm > 0):
        raise ValueError("NS norm derivative is undefined at zero momentum")
    x = g / denominator
    primals = []
    for _ in range(12):
        a = x @ x.mT
        b = -1.5 * a + 0.5 * a @ a
        primals.append((x, a, b))
        x = 2 * x + b @ x

    def jvp(tangent):
        dtype = torch.float64 if tangent.dtype == torch.float64 else torch.float32
        e = tangent.to(dtype)
        if tall:
            e = e.mT
        gg, dd = g.to(dtype), denominator.to(dtype)
        # Norm's identity-rounding AD derivative uses the unrounded real norm.
        # The denominator's PRIMAL remains the exact recorded BF16 value.
        dnorm = (gg * e).sum() / gg.norm()
        dx = e / dd - gg * dnorm / dd.square()
        for x, a, b in primals:
            xx, aa, bb = x.to(dtype), a.to(dtype), b.to(dtype)
            da = dx @ xx.mT + xx @ dx.mT
            db = -1.5 * da + 0.5 * (da @ aa + aa @ da)
            dx = 2 * dx + db @ xx + bb @ dx
        return dx.mT if tall else dx

    return jvp
