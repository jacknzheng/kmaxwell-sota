"""Literal ONE-kernel whole-network HVP proof of concept for a small two-layer MLP.

This is NOT the production GPT HVP. Closed-form forward/tangent/reverse-tangent
passes include both parameter blocks and their cross curvature. A single CTA is
appropriate only for this small experiment, not a 12-layer GPT batch.
"""
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton, tl = None, None


def mlp_loss(x, w1, w2):
    return .5 * ((x @ w1).relu().square() @ w2).square().sum()


def mlp_hvp_reference(x, w1, w2, v1, v2):
    a, da = x @ w1, x @ v1
    r = a.relu()
    u, du = r.square(), 2 * r * da
    y, dy = u @ w2, du @ w2 + u @ v2
    p, dp = y @ w2.T, dy @ w2.T + y @ v2.T
    return x.T @ (2 * r * dp + 2 * (a > 0) * da * p), du.T @ y + u.T @ dy


_kernel = None


def mlp_hvp_megakernel(x, w1, w2, v1, v2):
    global _kernel
    if not x.is_cuda:
        raise ValueError('literal megakernel requires CUDA')
    b, d = x.shape
    h, o = w2.shape
    if w1.shape != (d, h) or v1.shape != w1.shape or v2.shape != w2.shape:
        raise ValueError('incompatible network shapes')
    if any(n < 16 or n > 64 or n & (n - 1) for n in (b, d, h, o)):
        raise ValueError('prototype supports power-of-two dimensions 16..64 only')
    if any(t.dtype != torch.float32 or not t.is_contiguous() for t in (x, w1, w2, v1, v2)):
        raise ValueError('prototype requires contiguous FP32 tensors')
    if triton is None:
        raise ImportError('Triton is required for the literal megakernel')
    if _kernel is None:
        @triton.jit
        def run(X, W1, W2, V1, V2, H1, H2,
                B: tl.constexpr, D: tl.constexpr, H: tl.constexpr, O: tl.constexpr):
            ib, id = tl.arange(0, B), tl.arange(0, D)
            ih, io = tl.arange(0, H), tl.arange(0, O)
            x = tl.load(X + ib[:, None] * D + id[None, :])
            w1 = tl.load(W1 + id[:, None] * H + ih[None, :])
            w2 = tl.load(W2 + ih[:, None] * O + io[None, :])
            v1 = tl.load(V1 + id[:, None] * H + ih[None, :])
            v2 = tl.load(V2 + ih[:, None] * O + io[None, :])
            a = tl.dot(x, w1, input_precision='ieee')
            da = tl.dot(x, v1, input_precision='ieee')
            r = tl.maximum(a, 0)
            u, du = r * r, 2 * r * da
            y = tl.dot(u, w2, input_precision='ieee')
            dy = tl.dot(du, w2, input_precision='ieee') + tl.dot(u, v2, input_precision='ieee')
            p = tl.dot(y, tl.trans(w2), input_precision='ieee')
            dp = tl.dot(dy, tl.trans(w2), input_precision='ieee') + tl.dot(y, tl.trans(v2), input_precision='ieee')
            h1 = tl.dot(tl.trans(x), 2 * r * dp + 2 * (a > 0) * da * p, input_precision='ieee')
            h2 = tl.dot(tl.trans(du), y, input_precision='ieee') + tl.dot(tl.trans(u), dy, input_precision='ieee')
            tl.store(H1 + id[:, None] * H + ih[None, :], h1)
            tl.store(H2 + ih[:, None] * O + io[None, :], h2)
        _kernel = run
    h1, h2 = torch.empty_like(w1), torch.empty_like(w2)
    _kernel[(1,)](x, w1, w2, v1, v2, h1, h2, b, d, h, o, num_warps=4)
    return h1, h2
