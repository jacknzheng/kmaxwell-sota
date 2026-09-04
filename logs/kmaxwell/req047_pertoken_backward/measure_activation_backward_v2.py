"""REQ-038/041/043 (+REQ-047): per-Muon-matrix activation/backward statistics via forward/backward hooks.
Single process, one forward+backward at a fork-1500 checkpoint. Per matrix records:
 a/d rms+frob+eff_rank, attention q.k-logit stats, ||W||_F, alignment ratio (REQ-043), and REQ-047:
 per-token norm distributions (mean/sd/participation) of ||a_t|| and ||d_t||, da_cos_mean (adjacent-token
 backward cosine ALONG THE SEQUENCE AXIS), and grad_rank1_frac (sigma_1^2/sum sigma_i^2 of Sum_t d_t a_t^T = W.grad).
Usage: python measure_activation_backward_v2.py --model <model_step001500.pt> --out <json>"""
import argparse, json, math, sys
sys.path.insert(0, "records/track_3_optimization")
import torch
from harness.model_gpt import GPT
from harness.data_fineweb import iterate_batches_single_process

def frob(x): return float(x.float().norm().item())
def rms(x):  return float(x.float().pow(2).mean().sqrt().item())
def eff_rank(mat):
    m = mat.reshape(-1, mat.shape[-1]).float()
    if m.shape[0] < 2 or m.shape[1] < 2: return float("nan")
    try: s = torch.linalg.svdvals(m)
    except Exception: return float("nan")
    ev = (s*s); num = float(ev.sum().item())**2; den = float((ev*ev).sum().item())
    return num/den if den>0 else float("nan")

def as_BTF(t, B, T):
    """Reshape a hooked tensor to [B, T, feat] so per-token ordering stats respect sequence boundaries."""
    t = t.float()
    if t.dim() == 3: return t
    if t.dim() == 2 and t.shape[0] == B*T: return t.reshape(B, T, t.shape[-1])
    return None

def token_norm_stats(t, B, T):
    x = as_BTF(t, B, T)
    if x is None: return None
    n = x.reshape(-1, x.shape[-1]).norm(dim=-1)          # [B*T] per-token norms
    n2 = n*n; s4 = float((n2*n2).sum().item())
    pr = float((n2.sum()**2).item())/s4 if s4 > 0 else float("nan")  # participation ratio of ||token||^2
    return {"mean": float(n.mean().item()), "sd": float(n.std().item()), "participation": pr}

def da_cos_mean(t, B, T):
    """Mean cosine between d_t and d_{t+1} ALONG the sequence axis, within each batch row (no cross-boundary pairs)."""
    x = as_BTF(t, B, T)
    if x is None or x.shape[1] < 2: return float("nan")
    a = x[:, :-1]; b = x[:, 1:]                          # [B, T-1, F]
    dot = (a*b).sum(-1); den = (a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)
    return float((dot/den).mean().item())

def rank1_frac(W):
    try: s = torch.linalg.svdvals(W.float())
    except Exception: return float("nan")
    ss = s*s; tot = float(ss.sum().item())
    return float(ss[0].item())/tot if tot > 0 else float("nan")

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
    mods={}
    for bi,blk in enumerate(model.blocks):
        for sub,names in (("attn",["q","k","v","proj"]),("mlp",["fc","proj"])):
            m=getattr(blk,sub)
            for nm in names: mods[f"blocks.{bi}.{sub}.{nm}.weight"]=getattr(m,nm)
    acts={}; grads={}; qk={}; hooks=[]
    for name,mod in mods.items():
        def fwd(m,inp,out,nm=name): acts[nm]=inp[0].detach()
        hooks.append(mod.register_forward_hook(fwd))
        def bwd(m,gin,gout,nm=name):
            if gout and gout[0] is not None: grads[nm]=gout[0].detach()
        hooks.append(mod.register_full_backward_hook(bwd))
    for bi,blk in enumerate(model.blocks):
        def qh(m,inp,out,bi=bi): qk.setdefault(bi,{})["q"]=out.detach()
        def kh(m,inp,out,bi=bi): qk.setdefault(bi,{})["k"]=out.detach()
        hooks.append(blk.attn.q.register_forward_hook(qh))
        hooks.append(blk.attn.k.register_forward_hook(kh))
    it=iterate_batches_single_process(a.data, a.tokens, a.mbs, shard_rank=0, shard_world=1)
    inputs,targets=next(it); inputs=inputs.to(dev); targets=targets.to(dev)
    B,T=inputs.shape[0], inputs.shape[1] if inputs.dim()>1 else (a.mbs, inputs.numel()//a.mbs)
    if inputs.dim()==1: B,T=a.mbs, inputs.numel()//a.mbs
    loss=model(inputs,targets); loss.backward()
    ntok=int(inputs.numel())
    out={"model":a.model,"tokens":ntok,"batch":B,"seq":T,"loss":float(loss.item()),"matrices":{}}
    for name in mods:
        w=mods[name].weight
        rec={"weight_frob":frob(w)}
        if name in acts:
            rec["a_rms"]=rms(acts[name]); rec["a_frob"]=frob(acts[name]); rec["a_eff_rank"]=eff_rank(acts[name])
            rec["a_token_norms"]=token_norm_stats(acts[name],B,T)              # REQ-047 (2)
        if name in grads:
            rec["d_rms"]=rms(grads[name]); rec["d_frob"]=frob(grads[name]); rec["d_eff_rank"]=eff_rank(grads[name])
            rec["d_token_norms"]=token_norm_stats(grads[name],B,T)             # REQ-047 (1)
            rec["da_cos_mean"]=da_cos_mean(grads[name],B,T)                    # REQ-047 (3) adjacent-token backward cosine
        if w.grad is not None and name in acts and name in grads:
            gW=frob(w.grad); denom=rec["d_frob"]*rec["a_frob"]
            rec["grad_frob"]=gW; rec["align_ratio"]=(gW/denom) if denom>0 else float("nan")
            rec["grad_rank1_frac"]=rank1_frac(w.grad)                          # REQ-047 (4)
        out["matrices"][name]=rec
    H=model.blocks[0].attn.num_heads if hasattr(model.blocks[0].attn,"num_heads") else 6
    attn={}
    for bi,qkd in qk.items():
        q=qkd["q"]; k=qkd["k"]; Bq,Tq,hd=q.shape; hdh=hd//H
        qh=q.view(Bq,Tq,H,hdh).transpose(1,2); kh=k.view(Bq,Tq,H,hdh).transpose(1,2)
        logits=torch.matmul(qh,kh.transpose(-1,-2))/math.sqrt(hdh)
        mask=torch.triu(torch.ones(Tq,Tq,device=logits.device),1).bool()
        logits=logits.masked_fill(mask, float("-inf"))
        p=torch.softmax(logits.float(),dim=-1)
        ent=(-(p*torch.log(p.clamp_min(1e-12))).sum(-1)).mean().item()
        lg=logits[~logits.isinf()]
        attn[f"block{bi}"]={"qk_logit_rms":float(lg.float().pow(2).mean().sqrt().item()),"attn_entropy":float(ent)}
    out["attention"]=attn
    for h in hooks: h.remove()
    json.dump(out, open(a.out,"w"), indent=1)
    print(f"wrote {a.out}: {len(out['matrices'])} matrices, B={B} T={T}, tokens={ntok}, loss={loss.item():.4f}")

if __name__=="__main__": main()
