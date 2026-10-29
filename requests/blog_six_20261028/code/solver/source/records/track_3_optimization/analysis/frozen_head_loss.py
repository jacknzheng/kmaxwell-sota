import torch
import torch.nn.functional as F

from ..harness.model_gpt import GPT


def cached_derivative(z, weight, targets, lse, direction=None):
    """Reuse the forward projection; leave GEMMs to the vendor library.

    Ordinary tensor epilogues remain visible to Inductor, which can fuse them.
    The directional projection is done only once. No approximation to the loss
    or Hessian is introduced.
    """
    inv = (z.square() + 225).rsqrt()
    probability = (15 * z * inv - lse[:, None]).exp()
    first = 3375 * inv.pow(3)
    # Scatter a single target per row rather than materializing one_hot.
    error = probability.scatter_add(
        1, targets[:, None], -torch.ones_like(targets[:, None], dtype=z.dtype)
    )
    if direction is None:
        derivative = error * first
    else:
        dz = F.linear(direction, weight)
        du = first * dz
        mean = (probability * du).sum(-1, keepdim=True)
        derivative = probability * (du - mean) * first
        derivative = derivative + error * (-10125 * z * inv.pow(5)) * dz
    return derivative @ weight


class _CachedHeadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, targets, lse, upstream):
        ctx.save_for_backward(z, weight, targets, lse, upstream)
        return cached_derivative(z, weight, targets, lse) * upstream

    @staticmethod
    def backward(ctx, direction):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Frozen head supports second derivatives, not third derivatives"
            )
        z, weight, targets, lse, upstream = ctx.saved_tensors
        dx = cached_derivative(z, weight, targets, lse, direction) * upstream
        d_upstream = None
        if ctx.needs_input_grad[5]:
            gradient = cached_derivative(z, weight, targets, lse)
            d_upstream = (direction * gradient).sum().reshape_as(upstream)
        # z and lse are frozen intermediates. The complete analytic Hessian
        # returns through x, not through a second path via cached logits.
        return dx, None, None, None, None, d_upstream


class _CachedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, targets):
        z = F.linear(x, weight, bias)
        u = 15 * z * (z.square() + 225).rsqrt()
        lse = u.logsumexp(-1)
        ctx.save_for_backward(x, z, weight, targets, lse)
        return (lse - u.gather(1, targets[:, None]).squeeze(1)).sum()

    @staticmethod
    def backward(ctx, upstream):
        x, z, weight, targets, lse = ctx.saved_tensors
        dx = _CachedHeadGradient.apply(x, z, weight, targets, lse, upstream)
        return dx, None, None, None


def frozen_head_loss(x, weight, bias, targets, *, chunk_size=32768, backend="cached"):
    """Token-sum softcapped cross-entropy with analytic double backward.

    Head parameters are frozen. Cache the vocabulary projection, but retain
    every token and vocabulary entry; chunking only bounds workspace.
    """
    if weight.requires_grad or bias.requires_grad:
        raise ValueError(
            "Frozen-head loss cannot differentiate trainable head parameters"
        )
    if chunk_size is None:
        chunk_size = 32768
    if chunk_size < 1:
        raise ValueError("positive token chunk required")
    if backend != "cached":
        raise ValueError("only the production cached head is supported")
    if x.shape[:-1] != targets.shape or x.shape[-1] != weight.shape[1]:
        raise ValueError("hidden states, targets, and weight shapes disagree")
    x = x.reshape(-1, x.shape[-1]).contiguous()
    targets = targets.reshape(-1).contiguous()
    return sum(
        (
            _CachedHeadLoss.apply(
                x[start : start + chunk_size],
                weight,
                bias,
                targets[start : start + chunk_size],
            )
            for start in range(0, len(x), chunk_size)
        ),
        x.new_zeros(()),
    )


class FrozenHeadFusionGPT(GPT):
    """Same keys and frozen-Adam loss; complete head uses analytic double AD."""

    def __init__(self, *args, head_backend="cached", head_chunk_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_backend = head_backend
        self.head_chunk_size = head_chunk_size

    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        x = self.norm2(x)
        return frozen_head_loss(
            x,
            self.proj.weight,
            self.proj.bias,
            targets,
            backend=self.head_backend,
            chunk_size=self.head_chunk_size,
        )
