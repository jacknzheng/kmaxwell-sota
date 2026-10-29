"""Opt-in exact-loss fusion experiments, never the production observer default.

The vocabulary softcap has exceptionally large intermediates. Supply its
analytic first/second derivatives so each derivative can be one fused
elementwise program. Forward values retain the original implementation.
No approximation to the Hessian, vocabulary, or token population is made.
"""
import torch
import torch.nn.functional as F

from ..harness.model_gpt import GPT


def softcap_reference(z):
    return 15 * z * (z.square() + 225).rsqrt()


def softcap_gradient(z, g):
    r = (z.square() + 225).rsqrt()
    return g * (3375 * r * r * r)


def softcap_double_backward(z, g, v):
    r = (z.square() + 225).rsqrt()
    first = 3375 * r * r * r
    second = -10125 * z * r * r * r * r * r
    return v * g * second, v * first


_compiled = {}


def derivative_program(name, compiled):
    fn = {'first': softcap_gradient, 'second': softcap_double_backward}[name]
    if not compiled:
        return fn
    if name not in _compiled:
        _compiled[name] = torch.compile(fn, fullgraph=True, dynamic=False,
            options={'triton.cudagraphs': False})
    return _compiled[name]


class _SoftcapGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, g, compiled):
        ctx.save_for_backward(z, g)
        ctx.compiled = compiled
        return derivative_program('first', compiled)(z, g)

    @staticmethod
    def backward(ctx, v):
        z, g = ctx.saved_tensors
        dz, dg = derivative_program('second', ctx.compiled)(z, g, v)
        return dz, dg, None


class _Softcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, compiled):
        ctx.save_for_backward(z)
        ctx.compiled = compiled
        return softcap_reference(z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        return _SoftcapGradient.apply(z, g, ctx.compiled), None


def fused_softcap(z, *, compiled=False):
    return _Softcap.apply(z, compiled)


class SoftcapFusionGPT(GPT):
    """Same checkpoint keys/forward loss; only softcap AD implementation changes."""
    def __init__(self, *args, compiled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_softcap = compiled

    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = fused_softcap(logits, compiled=self.compiled_softcap)
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction='sum')


class ProfileGPT(GPT):
    """The unchanged loss with range annotations, used only during profiling."""
    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        with torch.profiler.record_function('module::vocabulary_softcap'):
            logits = softcap_reference(logits)
        with torch.profiler.record_function('module::cross_entropy'):
            return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction='sum')
