"""REQ-070 loss-probe driver (runs ON the box). For each probe offset (32/48/63) and each REQ-069 branch:
load the exact probe weights theta_b(offset), reconstruct causal candidate directions from the branch
captures, and measure FRESH probe loss at theta + (alpha*||baseline_update||) * d/||d|| for alpha in
{0,0.25,0.5,1,2}, restoring exact weights after each test. Fresh probe data = a fixed held-out val-token
set (not training data). baseline_update/temporal_avg/ensemble_mean_LOO are in optimizer-update units;
neg_gradient is raw downhill (direction, normalized). Leave-one-out ensemble never uses branch b's own noise.

Deliver loss-vs-distance curves per (offset, branch, candidate) + branch-level paired differences.
No training; forward-pass loss only. Writes JSON (no raw tensors).

Usage (torchrun --nproc_per_node=8 OR single-proc): req070_loss_probe.py --branch_root <dur> --probe_root <dur>
   --fork 1500 --offsets 32 48 63 --val_data 'data/fineweb10B/fineweb_val_*.bin' --val_tokens 2097152 --out <d>.json
"""
import argparse, json, os, sys
import numpy as np, torch
R68 = "/root/kmaxwell-sota/logs/river/req068_full_gradient_history/impl"
for p in (R68, "/root/kmaxwell-sota/records/track_3_optimization/offline_analysis",
          os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, p)
from req068_capture import GradientHistoryReader
from req070_directions import extract_candidates, unit_norm
NBR = 10
CANDS = ["baseline_update", "neg_gradient", "temporal_avg", "ensemble_mean_LOO"]
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def load_val_tokens(pattern, tokens):
    from harness.data_fineweb import distributed_data_generator
    import torch.distributed as dist
    bt = tokens
    gen = distributed_data_generator(pattern, bt)
    inp, tgt = next(gen)
    return inp, tgt


@torch.no_grad()
def eval_loss(model, inp, tgt, mbs=64):
    tot = 0.0; n = 0
    for i in range(0, inp.size(0), mbs):
        tot += float(model(inp[i:i+mbs], tgt[i:i+mbs]))
        n += int((tgt[i:i+mbs] >= 0).sum()) if (tgt < 0).any() else tgt[i:i+mbs].numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch_root", required=True); ap.add_argument("--probe_root", required=True)
    ap.add_argument("--fork", type=int, default=1500); ap.add_argument("--offsets", type=int, nargs="+", default=[32, 48, 63])
    ap.add_argument("--val_data", default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--val_tokens", type=int, default=2097152); ap.add_argument("--tokens", type=int, default=524288)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from check_secant_direction_with_hvp import load_model_at_checkpoint, start_distributed_if_launched
    if "RANK" in os.environ: start_distributed_if_launched()
    assert torch.cuda.is_available()

    # branch capture readers (grad mean-per-token; disp raw)
    gread = [GradientHistoryReader(os.path.join(a.branch_root, f"br{b}", "grads")) for b in range(NBR)]
    dread = [GradientHistoryReader(os.path.join(a.branch_root, f"br{b}", "disp")) for b in range(NBR)]
    names = [n for n in gread[0].params()]
    scale = float(a.tokens)
    def disp_cb(b): return lambda o: {n: dread[b].grad(a.fork + o, n).float().cpu().numpy().astype(np.float64).ravel() for n in names}
    def grad_cb(b): return lambda o: {n: (gread[b].grad(a.fork + o, n).float().cpu().numpy().astype(np.float64).ravel() / scale) for n in names}
    disp_cbs = [disp_cb(b) for b in range(NBR)]; grad_cbs = [grad_cb(b) for b in range(NBR)]

    inp, tgt = load_val_tokens(a.val_data, a.val_tokens)
    results = {"fork": a.fork, "offsets": a.offsets, "alphas": ALPHAS, "candidates": CANDS, "by_offset": {}}
    shape = {n: None for n in names}

    for off in a.offsets:
        per_branch = []
        for b in range(NBR):
            mp = os.path.join(a.probe_root, f"br{b}", "probes", f"train_state_model_step{a.fork+off:06d}.pt")
            model, _ = load_model_at_checkpoint(mp, [], [])
            named = {n.removeprefix("_orig_mod."): p for n, p in model.named_parameters()}
            for n in names:
                if shape[n] is None: shape[n] = named[n].shape
            theta0 = {n: named[n].detach().clone() for n in names}
            cands = extract_candidates(disp_cbs, grad_cbs, names, b, off)
            bnorm = unit_norm(cands["baseline_update"], names)
            rec = {"branch": b, "baseline_norm": bnorm, "curves": {}}
            for cname in CANDS:
                d = cands[cname]; dnorm = unit_norm(d, names)
                losses = {}
                for alpha in ALPHAS:
                    if alpha != 0.0 and dnorm == 0:
                        losses[str(alpha)] = None; continue
                    with torch.no_grad():
                        for n in names:
                            if alpha == 0.0:
                                named[n].copy_(theta0[n])
                            else:
                                step = torch.tensor((alpha * bnorm / dnorm) * d[n], dtype=named[n].dtype,
                                                    device=named[n].device).reshape(named[n].shape)
                                named[n].copy_(theta0[n] + step)
                    losses[str(alpha)] = eval_loss(model, inp, tgt)
                    with torch.no_grad():
                        for n in names: named[n].copy_(theta0[n])  # restore exactly
                rec["curves"][cname] = losses
            per_branch.append(rec)
            del model, theta0; torch.cuda.empty_cache()
        results["by_offset"][str(off)] = per_branch
        # paired summary: mean loss delta vs baseline (alpha=0) at alpha=1, per candidate
        b0 = np.mean([per_branch[b]["curves"]["baseline_update"]["0.0"] for b in range(NBR)])
        print(f"offset {off}: zero-loss={b0:.4f}", flush=True)
        for cname in CANDS:
            d1 = np.mean([(per_branch[b]["curves"][cname]["1.0"] - per_branch[b]["curves"][cname]["0.0"])
                          for b in range(NBR) if per_branch[b]["curves"][cname]["1.0"] is not None])
            print(f"  {cname}: mean loss delta @alpha1 = {d1:+.4f}", flush=True)
    json.dump(results, open(a.out, "w"))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
