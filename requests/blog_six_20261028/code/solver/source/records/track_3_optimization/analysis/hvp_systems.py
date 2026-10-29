import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from .coupled_muon import FrozenMuonHessian


class StagedHessian(FrozenMuonHessian):
    """Stage full microbatch HVPs, including all cross-parameter terms.

    Frozen model parameters/buffers are trace constants. Tokens and direction
    remain runtime inputs. Remainders have separate shape-specialized programs.
    The model must be deterministic: production GPT has no dropout.
    """

    def __init__(
        self, *args, mode="trace", compile_options=None, gemm_precision="fp32", **kwargs
    ):
        super().__init__(*args, **kwargs)
        if mode not in ("trace", "compile"):
            raise ValueError("invalid staging mode")
        if self.cache_batches:
            raise ValueError("staging does not retain first-gradient graphs")
        self.mode, self.programs, self.graph_metadata = mode, {}, {}
        self.compile_options = {"triton.cudagraphs": False, **(compile_options or {})}
        if gemm_precision not in ("fp32", "bf16_fp32"):
            raise ValueError("invalid GEMM precision")
        self.gemm_precision = gemm_precision
        if gemm_precision == "bf16_fp32":
            # No intermediate accumulator truncation is permitted in cuBLAS.
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def _micro(self, x, y, *direction):
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):

            def loss(current):
                params = dict(self.params)
                params.update(zip(self.names, current))
                return functional_call(
                    self.model,
                    (
                        {"inner." + n: p for n, p in params.items()},
                        {"inner." + n: b for n, b in self.buffers.items()},
                    ),
                    (x, y),
                )

            if self.backend == "forward":
                _, h = torch.func.jvp(
                    torch.func.grad(loss), (self.p,), (tuple(direction),)
                )
            else:
                g = torch.autograd.grad(loss(self.p), self.p, create_graph=True)
                active = [(a, v) for a, v in zip(g, direction) if a.requires_grad]
                h = (
                    torch.autograd.grad(
                        tuple(a for a, _ in active),
                        self.p,
                        grad_outputs=tuple(v for _, v in active),
                        allow_unused=True,
                    )
                    if active
                    else (None,) * len(self.p)
                )
            return tuple(
                torch.zeros_like(p) if value is None else value.detach()
                for p, value in zip(self.p, h)
            )

    def _program(self, inputs):
        key = tuple((tuple(t.shape), t.dtype, t.device.type) for t in inputs)
        if key in self.programs:
            return self.programs[key]
        from torch.fx.experimental.proxy_tensor import make_fx

        # Fake tracing avoids executing a full HVP just to discover its DAG.
        traced = make_fx(self._micro, tracing_mode="fake", _allow_non_fake_inputs=True)(
            *inputs
        )
        traced.graph.eliminate_dead_code()
        traced.recompile()
        replacements = {}
        if self.gemm_precision == "bf16_fp32":
            from .bf16_gemm import rewrite_bf16_gemms

            replacements = rewrite_bf16_gemms(traced)
        counts = {}
        for node in traced.graph.nodes:
            if node.op == "call_function":
                counts[str(node.target)] = counts.get(str(node.target), 0) + 1
        self.graph_metadata[str(key)] = dict(
            kind="ATen HVP DAG",
            operators=counts,
            nodes=len(list(traced.graph.nodes)),
            gemm_precision=self.gemm_precision,
            bf16_replacements=replacements,
            accumulation_dtype="float32",
        )
        # All differentiation has already been expanded into ATen operations.
        # Disable AOT backward generation: this observer returns detached HVPs.
        if self.mode == "compile":
            # Compile the callable directly. GraphModule.__call__ has an
            # exception wrapper whose traceback generator can mask OOMs.
            optimized = torch.compile(
                traced.forward,
                fullgraph=True,
                dynamic=False,
                options=self.compile_options,
            )

            def program(*values):
                with torch.no_grad():
                    return optimized(*values)
        else:

            def program(*values):
                with torch.no_grad():
                    return traced(*values)

        self.programs[key] = program
        return program

    def __call__(self, direction):
        if len(direction) != len(self.p):
            raise ValueError("one direction per selected parameter required")
        total = [torch.zeros_like(p) for p in self.p]
        for start in range(0, len(self.batch[0]), self.microbatch):
            inputs = tuple(
                b[start : start + self.microbatch] for b in self.batch
            ) + tuple(direction)
            outputs = self._program(inputs)(*inputs)
            for acc, value in zip(total, outputs):
                # Normalization is outside the compiled program: token-count
                # probes can reuse it without recompiling or changing the mean.
                acc.add_(value.detach() / self.tokens)
        return tuple(total)

    def close(self):
        super().close()
        self.programs.clear()


def relative_error(actual, expected):
    numerator = sum(
        float((a.double() - e.double()).square().sum())
        for a, e in zip(actual, expected)
    )
    denominator = sum(float(e.double().square().sum()) for e in expected)
    return (numerator / max(denominator, 1e-30)) ** 0.5
