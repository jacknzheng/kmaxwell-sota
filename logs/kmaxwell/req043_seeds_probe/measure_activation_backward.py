"""REQ-038 (+REQ-041): per-Muon-matrix activation/backward statistics via forward/backward hooks.
Single process. One forward+backward at a fork-1500 checkpoint. Records per matrix:
 |a| (input activation) rms+frob, |d| (output-gradient) rms+frob, effective ranks of a and d (participation ratio),
 attention q.k-logit rms + per-head attention entropy, weight ||W||_F (REQ-041), token count.
Usage: python measure_activation_backward.py --model <model_step001500.pt> --out <json>"""
import argparse, json, math, sys
sys.path.insert(0, "records/track_3_optimization")
import torch
from harness.model_gpt import GPT
from harness.data_fineweb import iterate_batches_single_process

def frob(x): return float(x.float().norm().item())
def rms(x):  return float(x.float().pow(2).mean().sqrt().item())
def eff_rank(mat):  # participation ratio of the eigenvalue (sigma^2) spectrum of a 2D [tokens, feat] matrix
    m = mat.reshape(-1, mat.shape[-1]).float()
    if m.shape[0] < 2 or m.shape[1] < 2: return float("nan")
    # use covariance eigenvalues = sigma^2; PR = (sum ev)^2 / sum(ev^2)
    try: s = torch.linalg.svdvals(m)
    except Exception: return float("nan")
    ev = (s*s); num = float(ev.sum().item())**2; den = float((ev*ev).sum().item())
    return num/den if den>0 else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--tokens", type=int, default=128*1024); ap.add_argument("--mbs", type=int, default=8)
    a=ap.parse_args()
    dev="cuda:0"; torch.cuda.set_device(0)
    model=GPT(vocab_size=50304, num_layers=12, model_dim=768).to(dev)
    sd=torch.load(a.model, map_location=dev, weights_only=True)
    model.load_state_dict(sd); model.train()
    # collect Muon Linear modules -> name
    mods={}
    for bi,blk in enumerate(model.blocks):
        for sub,names in (("attn",["q","k","v","proj"]),("mlp",["fc","proj"])):
            m=getattr(blk,sub)
            for nm in names:
                mods[f"blocks.{bi}.{sub}.{nm}.weight"]=getattr(m,nm)
    acts={}; grads={}; qk={}  # name->tensor
    hooks=[]
    for name,mod in mods.items():
        def fwd(m,inp,out,nm=name): acts[nm]=inp[0].detach()
        hooks.append(mod.register_forward_hook(fwd))
        def bwd(m,gin,gout,nm=name):
            if gout and gout[0] is not None: grads[nm]=gout[0].detach()
        hooks.append(mod.register_full_backward_hook(bwd))
    # capture q,k OUTPUTS per block for attention logits
    for bi,blk in enumerate(model.blocks):
        def qh(m,inp,out,bi=bi): qk.setdefault(bi,{})["q"]=out.detach()
        def kh(m,inp,out,bi=bi): qk.setdefault(bi,{})["k"]=out.detach()
        hooks.append(blk.attn.q.register_forward_hook(qh))
        hooks.append(blk.attn.k.register_forward_hook(kh))
    # one forward+backward on a fixed token set
    it=iterate_batches_single_process(a.data, a.tokens, a.mbs, shard_rank=0, shard_world=1)
    inputs,targets=next(it)
    inputs=inputs.to(dev); targets=targets.to(dev)
    loss=model(inputs,targets); loss.backward()
    ntok=int(inputs.numel())
    # per-matrix stats
    out={"model":a.model,"tokens":ntok,"loss":float(loss.item()),"matrices":{}}
    for name in mods:
        w=mods[name].weight
        rec={"weight_frob":frob(w)}
        if name in acts: rec["a_rms"]=rms(acts[name]); rec["a_frob"]=frob(acts[name]); rec["a_eff_rank"]=eff_rank(acts[name])
        if name in grads: rec["d_rms"]=rms(grads[name]); rec["d_frob"]=frob(grads[name]); rec["d_eff_rank"]=eff_rank(grads[name])
        # REQ-043 priority 3: alignment ratio ||Sum_t d_t a_t^T||_F / (||d||_F ||a||_F).
        # For a Linear y=a@W^T (no bias) the weight-gradient IS Sum_t d_t a_t^T, so ||.||_F = ||W.grad||_F exactly.
        if w.grad is not None and name in acts and name in grads:
            gW=frob(w.grad); denom=rec["d_frob"]*rec["a_frob"]
            rec["grad_frob"]=gW
            rec["align_ratio"]=(gW/denom) if denom>0 else float("nan")
        out["matrices"][name]=rec
    # attention logits + entropy per block
    H=model.blocks[0].attn.num_heads if hasattr(model.blocks[0].attn,"num_heads") else 6
    attn={}
    for bi,qkd in qk.items():
        q=qkd["q"]; k=qkd["k"]; B,T,hd=q.shape; hdh=hd//H
        qh=q.view(B,T,H,hdh).transpose(1,2); kh=k.view(B,T,H,hdh).transpose(1,2)  # B,H,T,hdh
        logits=torch.matmul(qh,kh.transpose(-1,-2))/math.sqrt(hdh)  # B,H,T,T
        mask=torch.triu(torch.ones(T,T,device=logits.device),1).bool()
        logits=logits.masked_fill(mask, float("-inf"))
        p=torch.softmax(logits.float(),dim=-1)
        ent=(-(p*torch.log(p.clamp_min(1e-12))).sum(-1)).mean().item()  # mean attn entropy
        lg=logits[~logits.isinf()]
        attn[f"block{bi}"]={"qk_logit_rms":float(lg.float().pow(2).mean().sqrt().item()),"attn_entropy":float(ent)}
    out["attention"]=attn
    for h in hooks: h.remove()
    json.dump(out, open(a.out,"w"), indent=1)
    print(f"wrote {a.out}: {len(out['matrices'])} matrices, {len(attn)} attn blocks, tokens={ntok}, loss={loss.item():.4f}")

if __name__=="__main__": main()
