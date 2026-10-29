import torch


def bf16_mm(a, b):
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


def bf16_bmm(a, b):
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    if a.is_cuda:
        return torch.bmm(a, b, out_dtype=torch.float32)
    return torch.bmm(a.float(), b.float())


def bf16_addmm(bias, a, b, *, beta=1, alpha=1):
    # Keep the bias/epilogue FP32; in particular beta=0 must not read NaN bias.
    product = bf16_mm(a, b) * alpha
    return product if beta == 0 else product + bias.float() * beta


def rewrite_bf16_gemms(traced):
    """Replace only FP32 dense GEMMs in an already differentiated FX program."""
    replacements = {
        torch.ops.aten.mm.default: bf16_mm,
        torch.ops.aten.bmm.default: bf16_bmm,
        torch.ops.aten.addmm.default: bf16_addmm,
    }
    counts = {}
    for node in traced.graph.nodes:
        if node.op != "call_function" or node.target not in replacements:
            continue
        # addmm's first operand is the bias; all relevant inputs must be FP32.
        operands = (
            node.args[:3]
            if node.target == torch.ops.aten.addmm.default
            else node.args[:2]
        )
        if not all(
            getattr(value.meta.get("val"), "dtype", None) == torch.float32
            for value in operands
            if isinstance(value, torch.fx.Node)
        ):
            continue
        original = str(node.target)
        node.target = replacements[node.target]
        counts[original] = counts.get(original, 0) + 1
    traced.graph.lint()
    traced.recompile()
    return counts
