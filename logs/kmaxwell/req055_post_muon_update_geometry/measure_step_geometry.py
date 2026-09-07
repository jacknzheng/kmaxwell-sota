"""REQ-055: geometry of the actual post-Muon update. Reads consecutive model dumps model_step{S}.pt (x_S,
pre-update) and model_step{S+1}.pt (x_{S+1}); delta_full = x_{S+1}-x_S is the realized displacement.

Per the request the analysis direction is the MUON GRADIENT-DRIVEN displacement, separated from decoupled
weight decay: Muon applies  p <- p*(1-lr*wd) - lr*update, so for each Muon param
  delta_full = delta_wd + delta_muon,  delta_wd = -(lr*wd)*x_S = wd_coef * x_S  (wd_coef passed in),
  delta_muon = delta_full - delta_wd  (the polar/shape/LR update itself).
Non-Muon displacement (embed/proj AdamW params) = x_{S+1}-x_S on those params.

On 3 fixed held-out probe minibatches (paired across arms), with model.eval() and exact param reset:
 - downhill alignment  -g.delta_muon/(||g|| ||delta_muon||)  (per matrix + JOINT); reports g.delta,||g||,||delta||,uphill frac
 - directional curvature v^T H v (v=delta_muon/||delta_muon||) TWO independent ways:
     (a) loss-scan symmetric 2nd difference:  (dL(+t)+dL(-t)) / (t^2 ||delta||^2)   (forward only)
     (b) gradient finite-difference HVP:  (g(x_S+e*v)-g(x_S-e*v)).delta / (2e ||delta||^2)  (single backward; flash has no double-backward)
   (b) also per-matrix (diagonal). Both retain sign; the JOINT (b) keeps cross-matrix terms (full-delta perturbation).
 - loss scan L(x_S + a*delta_muon)-L(x_S) for a in {-0.5,0,0.25,0.5,1,1.5,2} + quadratic prediction a*(g.d)+0.5a^2(d^T H d)
 - FULL realized displacement dL at a=1 (Muon delta_full + non-Muon delta), so weight decay + non-Muon updates are visible.
Usage: measure_step_geometry.py --dump_dir <dd> --step S --wd_coef <-(lr*wd)> --out <json>"""
import argparse, math, json, sys, os
sys.path.insert(0,"records/track_3_optimization")
import torch
from harness.model_gpt import GPT
from harness.data_fineweb import iterate_batches_single_process

def muon_names(model):
    ns=[]
    for bi,blk in enumerate(model.blocks):
        for sub,names in (("attn",["q","k","v","proj"]),("mlp",["fc","proj"])):
            for nm in names: ns.append(f"blocks.{bi}.{sub}.{nm}.weight")
    return ns

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump_dir",required=True); ap.add_argument("--step",type=int,required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--wd_coef",type=float,required=True,help="-(lr_eff*wd) at step S; delta_wd = wd_coef * x_S")
    ap.add_argument("--data",default="data/fineweb10B/fineweb_val_*.bin"); ap.add_argument("--tokens",type=int,default=32768)
    ap.add_argument("--mbs",type=int,default=8); ap.add_argument("--n_probe",type=int,default=3); ap.add_argument("--eps",type=float,default=1e-2)
    a=ap.parse_args(); dev="cuda:0"; torch.cuda.set_device(0); S=a.step
    xS=torch.load(f"{a.dump_dir}/model_step{S:06d}.pt",map_location=dev,weights_only=True)
    xS1=torch.load(f"{a.dump_dir}/model_step{S+1:06d}.pt",map_location=dev,weights_only=True)
    model=GPT(vocab_size=50304,num_layers=12,model_dim=768).to(dev); model.load_state_dict(xS); model.eval()
    name2p={n:p for n,p in model.named_parameters()}
    mnames=muon_names(model)
    nonmuon=[n for n in name2p if n not in set(mnames)]
    xSf={n:xS[n].to(dev).float().clone() for n in name2p}
    # displacement decomposition (Muon params)
    delta_full={n:(xS1[n].to(dev).float()-xSf[n]) for n in mnames}
    delta_wd  ={n:(a.wd_coef*xSf[n]) for n in mnames}
    delta     ={n:(delta_full[n]-delta_wd[n]) for n in mnames}          # delta_muon: the request's delta
    delta_non ={n:(xS1[n].to(dev).float()-xSf[n]) for n in nonmuon}     # non-Muon realized displacement
    def jn(d,keys): return math.sqrt(sum(float((d[k]*d[k]).sum()) for k in keys))
    dnorm=jn(delta,mnames); dfn=jn(delta_full,mnames); dwn=jn(delta_wd,mnames); dnn=jn(delta_non,nonmuon)
    recon=math.sqrt(sum(float(((delta[n]+delta_wd[n]-delta_full[n])**2).sum()) for n in mnames))  # == 0 by construction
    mparams=[name2p[n] for n in mnames]
    # fixed probe minibatches
    it=iterate_batches_single_process(a.data,a.tokens*a.n_probe,a.mbs,shard_rank=0,shard_world=1)
    batches=[(i.to(dev),t.to(dev)) for i,t in (next(it) for _ in range(a.n_probe))]
    def grad_muon():   # gradient of summed probe loss wrt Muon params, at current param values
        model.zero_grad(set_to_none=True); g={n:torch.zeros_like(name2p[n]) for n in mnames}
        for inp,tgt in batches:
            for i,gl in enumerate(torch.autograd.grad(model(inp,tgt),mparams)): g[mnames[i]]=g[mnames[i]]+gl
        return g
    def set_params(coef):  # params <- x_S + coef*delta_muon  (Muon only; non-Muon stay at x_S)
        with torch.no_grad():
            for n in mnames: name2p[n].copy_(xSf[n]+coef*delta[n])
    def reset():
        with torch.no_grad():
            for n in name2p: name2p[n].copy_(xSf[n])
    # gradient g at x_S
    g=grad_muon()
    gvec=[g[n].float() for n in mnames]; gnorm=math.sqrt(sum(float((x*x).sum()) for x in gvec))
    gd=sum(float((g[n].float()*delta[n]).sum()) for n in mnames)
    align_joint=-gd/(gnorm*dnorm) if gnorm>0 and dnorm>0 else float("nan")
    permat={}
    for n in mnames:
        gn=float(g[n].float().norm()); dn=float(delta[n].norm())
        permat[n]={"align":(-float((g[n].float()*delta[n]).sum())/(gn*dn)) if gn>0 and dn>0 else float("nan"),
                   "g_norm":gn,"delta_muon_norm":dn}
    # (b) gradient finite-difference HVP: central diff of grad along v=delta_muon
    e=a.eps
    set_params(+e); gp=grad_muon(); set_params(-e); gm=grad_muon(); reset()
    dHd_fd=sum(float(((gp[n]-gm[n])*delta[n]).sum()) for n in mnames)/(2*e)   # = delta^T H delta (cross terms kept)
    vHv_fd=dHd_fd/(dnorm*dnorm) if dnorm>0 else float("nan")
    for n in mnames:
        dn=float(delta[n].norm()); permat[n]["vHv_fd"]=float(((gp[n]-gm[n])*delta[n]).sum())/(2*e*dn*dn) if dn>0 else float("nan")
    # loss scan along delta_muon (forward only; exact reset each eval)
    def loss_muon(alpha):
        set_params(alpha); tot=0.0
        with torch.no_grad():
            for inp,tgt in batches: tot+=float(model(inp,tgt).item())
        reset(); return tot
    L0=loss_muon(0.0)
    alphas=(-0.5,0.0,0.25,0.5,1.0,1.5,2.0)
    scan={f"{al}":loss_muon(al)-L0 for al in alphas}
    # (a) loss-scan 2nd-difference curvature
    t=0.5; dHd_scan=(scan[f"{-t}"]+scan[f"{t}"])/(t*t)
    vHv_scan=dHd_scan/(dnorm*dnorm) if dnorm>0 else float("nan")
    # quadratic prediction using the FD Hessian term
    quad={f"{al}": al*gd + 0.5*al*al*dHd_fd for al in alphas}
    # FULL realized displacement at a=1 (Muon delta_full + non-Muon), relative to x_S
    with torch.no_grad():
        for n in mnames: name2p[n].copy_(xSf[n]+delta_full[n])
        for n in nonmuon: name2p[n].copy_(xSf[n]+delta_non[n])
    tot=0.0
    with torch.no_grad():
        for inp,tgt in batches: tot+=float(model(inp,tgt).item())
    reset(); dL_full=tot-L0
    out={"step":S,"L0_raw":L0,"wd_coef":a.wd_coef,"eps":e,
         "delta_muon_norm":dnorm,"delta_full_norm":dfn,"delta_wd_norm":dwn,"delta_nonmuon_norm":dnn,"recon_residual":recon,
         "g_norm":gnorm,"g_dot_delta":gd,"align_joint":align_joint,
         "vHv_fd":vHv_fd,"vHv_scan":vHv_scan,"dHd_fd":dHd_fd,"dHd_scan":dHd_scan,
         "uphill_matrix_frac":sum(1 for n in mnames if permat[n]["align"]<0)/len(mnames),
         "loss_scan_muon":scan,"quad_pred":quad,"dL_full_realized_a1":dL_full,"per_matrix":permat}
    json.dump(out,open(a.out,"w"),indent=1)
    print(f"step {S}: align={align_joint:+.4f} vHv_fd={vHv_fd:+.2e} vHv_scan={vHv_scan:+.2e} uphill={out['uphill_matrix_frac']:.2f} "
          f"|d_muon|={dnorm:.3f} |d_wd|={dwn:.3f} |d_non|={dnn:.3f} scan@1={scan['1.0']:+.3f} full@1={dL_full:+.3f}")
if __name__=="__main__": main()
