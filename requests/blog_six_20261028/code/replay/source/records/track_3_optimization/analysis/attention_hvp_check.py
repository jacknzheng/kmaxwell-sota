"""Check actual second derivatives before using an attention-JVP package.

An accurate forward/JVP/ordinary backward does not establish an accurate HVP.
Both reverse-over-reverse and reverse-over-forward compositions are tested.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .hvp_systems import relative_error


def math_attention(q,k,v):
    # Explicit, non-inplace oracle: math SDPA's forward-AD softmax currently
    # causes an autograd version error in reverse-over-forward on CPU.
    scores=(q @ k.transpose(-2,-1))*.12
    causal=torch.ones(scores.shape[-2:],device=scores.device,dtype=torch.bool).tril()
    scores=scores.masked_fill(~causal,float('-inf'))
    # Avoid PyTorch's softmax forward-AD path too; it modifies an exp result
    # in-place. The detached stabilizer leaves softmax's exact derivatives.
    numerator=(scores-scores.detach().amax(dim=-1,keepdim=True)).exp()
    probabilities=numerator/numerator.sum(dim=-1,keepdim=True)
    return probabilities @ v


def sdpa_attention(q,k,v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=.12)


def reverse_hvp(attention,primals,directions):
    values=tuple(t.detach().clone().requires_grad_(True) for t in primals)
    loss=attention(*values).square().sum()/values[0].numel()
    gradient=torch.autograd.grad(loss,values,create_graph=True)
    if not all(g.requires_grad for g in gradient):
        raise RuntimeError('attention backward does not retain second-derivative graph')
    return torch.autograd.grad(gradient,values,grad_outputs=directions)


def reverse_forward_hvp(attention,primals,directions):
    values=tuple(t.detach().clone().requires_grad_(True) for t in primals)
    with torch.autograd.forward_ad.dual_level():
        duals=tuple(torch.autograd.forward_ad.make_dual(p,v) for p,v in zip(values,directions))
        dual_loss=attention(*duals).square().sum()/values[0].numel()
        _, tangent=torch.autograd.forward_ad.unpack_dual(dual_loss)
        if tangent is None or not tangent.requires_grad:
            raise RuntimeError('loss tangent is missing or detached')
        return torch.autograd.grad(tangent,values)


def check_candidate(attention,dual_attention,primals,directions):
    expected=reverse_hvp(math_attention,primals,directions)
    reference_dual=reverse_forward_hvp(math_attention,primals,directions)
    result={'reference_composition_error':relative_error(reference_dual,expected)}
    for name,implementation in [('reverse_over_reverse',lambda:reverse_hvp(attention,primals,directions)),
        ('reverse_over_forward',lambda:reverse_forward_hvp(dual_attention,primals,directions))]:
        try:
            actual=implementation()
            result[name]={'relative_hvp_error':relative_error(actual,expected),
                'finite':all(bool(torch.isfinite(t).all()) for t in actual)}
        except Exception as error:
            result[name]={'error':repr(error)}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'stage':'starting','library':str(args.library),'production_operator_replaced':False}
    try:
        sys.path.insert(0,str(args.library))
        from jvp_flash_attention.jvp_attention import JVPAttn
        torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest')
        report.update(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cases=[])
        def attention(*tensors):return JVPAttn.fwd(*tensors,causal=True,sm_scale=.12,USE_TMA=False,warp_specialize=False)
        def dual_attention(*tensors):return JVPAttn.fwd_dual(*tensors,causal=True,sm_scale=.12,USE_TMA=False,warp_specialize=False)
        for sequence in [128,1024]:
            torch.manual_seed(19)
            primals=tuple(torch.randn(1,6,sequence,128,device='cuda')*.1 for _ in range(3))
            directions=tuple(torch.randn_like(t)*.1 for t in primals)
            case=check_candidate(attention,dual_attention,primals,directions)
            case.update(shape=list(primals[0].shape),dtype=str(primals[0].dtype))
            report['cases'].append(case)
            (args.output/'attention_hvp.json').write_text(json.dumps(report,indent=2))
        report['stage']='complete'
    except Exception as error:
        report.update(stage='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        (args.output/'attention_hvp.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
