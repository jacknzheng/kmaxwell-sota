"""REQ-060 real-model diagnostic: at a dumped trajectory state, separate loss-cubic feedback from Muon's
polar-map nonlinearity along the (spectral-polar) sharp direction. Single-process (no dist).
For the 72 Muon matrices on 3 fixed diagnostic minibatches (mean-token gradients):
  delta_i = base_scale * spectral_polar(g_i)   (unit-spectral direction, the sharp direction)
  e_loss  = [g(c+delta)+g(c-delta)]/2 - g(c)   (loss-cubic feedback); scale test delta*{1,0.5,0.25} -> ~s^2
  Hd      = [g(c+delta)-g(c-delta)]/2          (odd part ~ H[delta])
  Phi(q)  = zeropower_via_newtonschulz5(q)*sqrt(max(1,rows/cols))   (the implemented polar/shape map)
  E_total = [Phi(g(c+delta))+Phi(g(c-delta))]/2 - Phi(g(c))
  E_map   = [Phi(g(c)+Hd)+Phi(g(c)-Hd)]/2 - Phi(g(c))          (polar-map nonlinearity; exists on a quadratic loss)
  E_loss_residual = E_total - E_map                            (extra curvature from loss-cubic feedback)
Reports, per matrix + joint: ||e_loss|| scale ratios, e_loss.delta projection, ||E_total||/||E_map||/||E_residual||,
and residual_frac=||E_residual||/||E_total||. This is a frozen-buffer diagnostic at the gradient-polar center
(c0=1 simplification, documented), not a full evolving-buffer replay. Usage:
  measure_req060_diag.py --state_dir DIR --step S --out OUT.json [--base_scale 0.02]
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--data", default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--tokens", type=int, default=32768); ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--n_probe", type=int, default=3); ap.add_argument("--base_scale", type=float, default=0.02)
    a = ap.parse_args(); dev = "cuda:0"; torch.cuda.set_device(0)
    import re
    from harness.model_gpt import GPT
    from harness.data_fineweb import iterate_batches_single_process
    from optimizers.muon import zeropower_via_newtonschulz5
    from check_secant_direction_with_hvp import load_model_at_checkpoint
    names = None
    mp = os.path.join(a.state_dir, f"train_state_model_step{a.step:06d}.pt")
    tmp = GPT(vocab_size=50304, num_layers=12, model_dim=768)
    names = [n for n, p in tmp.named_parameters() if re.match(r"^blocks\..*\.weight$", n) and p.ndim == 2]
    model, params = load_model_at_checkpoint(mp, [], names)
    name2p = dict(zip(names, params))
    batches = []
    it = iterate_batches_single_process(a.data, a.tokens * a.n_probe, a.mbs, shard_rank=0, shard_world=1)
    for _ in range(a.n_probe):
        i, t = next(it); batches.append((i.to(dev), t.to(dev)))

    def grad_at(shift):  # shift: dict name->tensor added to params; returns {name: grad}, params restored
        if shift:
            with torch.no_grad():
                for n, s in shift.items(): name2p[n].add_(s)
        g = {n: torch.zeros_like(name2p[n]) for n in names}
        model.zero_grad(set_to_none=True)
        for inp, tgt in batches:
            gl = torch.autograd.grad(model(inp, tgt), params)
            for i, n in enumerate(names): g[n] = g[n] + gl[i].detach()
        seen = sum(t.numel() for _, t in batches)
        for n in names: g[n] = g[n] / seen
        if shift:
            with torch.no_grad():
                for n, s in shift.items(): name2p[n].add_(-s)
        return g

    def spol(gm):  # unit-spectral polar direction
        u, _, vh = torch.linalg.svd(gm.double(), full_matrices=False); return (u @ vh).to(gm.dtype)

    def shape(p): return max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5

    g0 = grad_at(None)
    delta = {n: a.base_scale * spol(g0[n]) for n in names}
    # scale test + e_loss at base delta
    scales = [1.0, 0.5, 0.25]
    eloss_norm = {s: 0.0 for s in scales}
    for s in scales:
        sh = {n: s * delta[n] for n in names}
        gp = grad_at(sh); gm = grad_at({n: -v for n, v in sh.items()})
        for n in names:
            el = 0.5 * (gp[n] + gm[n]) - g0[n]
            eloss_norm[s] += float((el * el).sum())
        eloss_norm[s] = eloss_norm[s] ** 0.5
    # separation at base delta
    gp = grad_at(delta); gm = grad_at({n: -v for n, v in delta.items()})
    Et = En = Em = eldot = 0.0
    per = {}
    for n in names:
        p = name2p[n]; sc = shape(p)
        Phi = lambda q: zeropower_via_newtonschulz5(q) * sc
        Hd = 0.5 * (gp[n] - gm[n])            # odd part ~ H[delta]
        el = 0.5 * (gp[n] + gm[n]) - g0[n]    # even part = e_loss
        E_total = 0.5 * (Phi(gp[n]) + Phi(gm[n])) - Phi(g0[n])
        E_map = 0.5 * (Phi(g0[n] + Hd) + Phi(g0[n] - Hd)) - Phi(g0[n])
        E_res = E_total - E_map
        Et += float((E_total*E_total).sum()); Em += float((E_map*E_map).sum()); En += float((E_res*E_res).sum())
        eldot += float((el * spol(g0[n])).sum())
        per[n] = dict(e_loss_norm=float(el.norm()), E_total=float(E_total.norm()),
                      E_map=float(E_map.norm()), E_residual=float(E_res.norm()))
    Et, Em, En = Et**0.5, Em**0.5, En**0.5
    out = dict(step=a.step, base_scale=a.base_scale, tokens=a.tokens * a.n_probe,
               eloss_scale_norms={str(s): eloss_norm[s] for s in scales},
               eloss_scale_ratios={"0.5/1": eloss_norm[0.5]/eloss_norm[1.0] if eloss_norm[1.0] else None,
                                   "0.25/1": eloss_norm[0.25]/eloss_norm[1.0] if eloss_norm[1.0] else None},
               eloss_proj_on_sharp=eldot,
               E_total_joint=Et, E_map_joint=Em, E_residual_joint=En,
               residual_frac=En/Et if Et else None, map_frac=Em/Et if Et else None,
               per_matrix=per)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"step {a.step}: eloss s0.5/1={out['eloss_scale_ratios']['0.5/1']:.3f} (quad~0.25) "
          f"E_total={Et:.3e} E_map={Em:.3e} E_res={En:.3e} residual_frac={out['residual_frac']:.2%}")


if __name__ == "__main__":
    main()
