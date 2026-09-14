"""REQ-057 GPU pilot: isolated + joint shape-weighted spectral sharpness, G/Z, cross-layer coupling, and
HVP validation, on a saved checkpoint, in MEAN-per-valid-token loss scale.

Data-parallel over a fixed token subset (shard_rank/shard_world), all_reduce every HVP so all ranks hold the
identical H action (REQ-019 pattern). One parallelism mode for both joint and isolated FW. Loss is the
harness summed cross-entropy; we divide the summed gradient AND summed HVP by the total valid token count
(published as token_scale) to get the mean-per-token Hessian action -- never BATCH_TOKENS.

Per requested subset this computes, for the selected matrices:
  - block gradients (mean-token), shape-weighted radii r_i = sqrt(max(1,rows/cols)) * lr_mult
  - joint S_joint (sphere, best boundary witness) at each K in iters_list, G_joint, Z_joint
  - isolated S_i at each K, G_i, Z_i
  - lambda_i (8-iter Lanczos, diagonal) and lambda_i/||g_i||_F^2 for comparison (retained, not a reference)
  - (subset 0 only) 18x18 interaction matrix Q_ij at embedded gradient-polar directions; HVP validation
    (central grad-difference vs autograd) at eps 0.005/0.01/0.02/0.04 with plateau + 5% check.
Writes one JSON per (step, subset). Reliability gates are evaluated offline by analyze_pilot.py.

Run (whole box):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/offline_analysis/measure_layerwise_pilot.py \
    --state_dir req057_state_s0 --step 2000 --pilot --iters_list 20 50 --restarts 5 \
    --subset_tokens 8192 --n_subsets 3 --nested_tokens 32768 --out_dir logs/kmaxwell/req057_layerwise_spectral_sharpness/raw
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # harness importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling probe module
import measure_layerwise_spectral_sharpness as M


def pilot_matrix_names(all_names):
    """18 pilot matrices = all 6 types in blocks 0, 6, 11 (types: attn.q/k/v/proj, mlp.fc/proj)."""
    want_blocks = {0, 6, 11}
    sel = [n for n in all_names if int(n.split(".")[1]) in want_blocks]
    return sel


def radii_for(params, lr_mults=None):
    r = []
    for i, p in enumerate(params):
        rows, cols = p.shape[-2], p.shape[-1]
        r.append(M.shape_radius(rows, cols, (lr_mults[i] if lr_mults else 1.0)))
    return r


def _iter(data, tokens, mbs, rank, world):
    from harness.data_fineweb import iterate_batches_single_process
    return iterate_batches_single_process(data, tokens, mbs, shard_rank=rank, shard_world=world)


def block_grads_mean(model, params, data, tokens, mbs, rank, world):
    """Mean-per-valid-token gradient blocks over the fixed subset (all_reduced)."""
    g = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    seen = 0
    for inp, tgt in _iter(data, tokens, mbs, rank, world):
        loss = model(inp, tgt)
        for acc, gp in zip(g, torch.autograd.grad(loss, params)):
            acc += gp.detach().float()
        seen += tgt.numel()
    cnt = torch.tensor([seen], device=params[0].device)
    if world > 1:
        for acc in g:
            dist.all_reduce(acc)
        dist.all_reduce(cnt)
    seen = int(cnt[0]); assert seen > 0
    for acc in g:
        acc /= seen
    return g, seen


def joint_hvp_mean(model, params, vecs, data, tokens, mbs, rank, world):
    """Joint HVP (Hv)_m = sum_n H_mn v_n, mean-per-valid-token, all_reduced. SDPBackend.MATH for 2nd backward."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    seen = 0
    for inp, tgt in _iter(data, tokens, mbs, rank, world):
        with sdpa_kernel([SDPBackend.MATH]):
            loss = model(inp, tgt)
        g1 = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g1[n] * vecs[n]).sum() for n in range(len(params)))
        hv = torch.autograd.grad(dot, params)
        for acc, h in zip(out, hv):
            acc += h.detach().float()
        seen += tgt.numel()
        del loss, g1, dot, hv
    cnt = torch.tensor([seen], device=params[0].device)
    if world > 1:
        for acc in out:
            dist.all_reduce(acc)
        dist.all_reduce(cnt)
    seen = int(cnt[0]); assert seen > 0
    for acc in out:
        acc /= seen
    return out


def run_subset(model, params, names, radii, args, rank, world, data, tokens, do_extras):
    """All sharpness measurements for one probe subset. Returns a JSON-able dict."""
    t0 = time.time()
    grads, seen = block_grads_mean(model, params, data, tokens, args.mbs, rank, world)
    Gi, Gj = M.G_denominators(grads, radii)
    eta = args.eta

    def joint_hvp(v):
        return joint_hvp_mean(model, params, v, data, tokens, args.mbs, rank, world)

    # JOINT sharpness (all matrices in this run), sphere best-boundary witness
    joint = M.fw_best_sphere(joint_hvp, grads, radii, args.iters_list, args.restarts)
    S_joint = joint["best_sphere"]
    Z_joint = M.Z_stat(eta, S_joint, Gj) if S_joint is not None else None

    # ISOLATED sharpness per matrix (diagonal HVP via joint HVP with only slot i nonzero)
    def diag_hvp_one(i, vi):
        ej = [torch.zeros_like(params[k]) for k in range(len(params))]
        ej[i] = vi
        return joint_hvp(ej)[i]

    isolated = {}
    for i, n in enumerate(names):
        res = M.isolated_sharpness(lambda j, v, _i=i: diag_hvp_one(_i, v), grads[i], radii[i], 0,
                                   args.iters_list, args.restarts)
        Si = res["best_sphere"]
        isolated[n] = dict(S_i_per_k={str(k): res["best_sphere_per_k"][k] for k in args.iters_list},
                           S_i=Si, G_i=Gi[i], Z_i=(M.Z_stat(eta, Si, Gi[i]) if Si is not None else None),
                           radius=radii[i], grad_frob=float(grads[i].norm()),
                           grad_nuclear=M.nuclear_norm(grads[i]),
                           negative_curvature=res["negative_curvature"])
        if rank == 0:
            print(f"  isolated {n}: S_i={Si} G_i={Gi[i]:.4e} ({time.time()-t0:.0f}s)", flush=True)

    out = dict(step=args.step, tokens=seen, token_scale=1.0 / seen, eta=eta,
               radii={n: radii[i] for i, n in enumerate(names)},
               S_joint=S_joint, S_joint_per_k={str(k): joint["best_sphere_per_k"][k] for k in args.iters_list},
               G_joint=Gj, Z_joint=Z_joint, joint_negative_curvature=joint["negative_curvature"],
               isolated=isolated, seconds=round(time.time() - t0, 1))

    if do_extras and rank == 0:
        print("  extras: Q_ij + HVP validation on rank 0 (single-process over full subset)", flush=True)
    if do_extras:
        # Interaction matrix Q_ij at embedded gradient-polar directions (all ranks cooperate on each HVP)
        polar_dirs = [M.polar_lmo(grads[i], radii[i]) for i in range(len(params))]
        Q = M.interaction_matrix(joint_hvp, polar_dirs)
        cross = M.cross_layer_decomposition(joint_hvp, polar_dirs,
                                            lambda i, vi: diag_hvp_one(i, vi))
        out["Q_ij"] = Q
        out["Q_names"] = names
        out["cross_layer_at_gradpolar"] = cross
        # HVP validation: central grad-difference vs autograd on the joint gradient-polar direction
        def grad_fn():
            g = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
            seen2 = 0
            for inp, tgt in _iter(data, tokens, args.mbs, rank, world):
                loss = model(inp, tgt)
                for acc, gp in zip(g, torch.autograd.grad(loss, params)):
                    acc += gp.detach().float()
                seen2 += tgt.numel()
            if world > 1:
                for acc in g:
                    dist.all_reduce(acc)
            c = torch.tensor([seen2], device=params[0].device)
            if world > 1:
                dist.all_reduce(c)
            for acc in g:
                acc /= int(c[0])
            return g
        ag = joint_hvp(polar_dirs)
        ag_dot = float(sum((polar_dirs[i] * ag[i]).sum() for i in range(len(params))))
        val = {}
        for eps in args.eps_list:
            cd = M.central_diff_hvp(grad_fn, params, polar_dirs, eps)
            cd_dot = float(sum((polar_dirs[i] * cd[i]).sum() for i in range(len(params))))
            rel = abs(cd_dot - ag_dot) / (abs(ag_dot) + 1e-30)
            val[str(eps)] = dict(central_diff_dHd=cd_dot, rel_err_vs_autograd=rel)
        out["hvp_validation"] = dict(autograd_dHd=ag_dot, by_eps=val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--data", default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.02, help="Muon base LR (for Z); logged, from config")
    ap.add_argument("--iters_list", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--subset_tokens", type=int, default=8192)
    ap.add_argument("--n_subsets", type=int, default=3)
    ap.add_argument("--nested_tokens", type=int, default=32768)
    ap.add_argument("--eps_list", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04])
    ap.add_argument("--pilot", action="store_true", help="restrict to the 18 pilot matrices")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available()

    from check_secant_direction_with_hvp import load_model_at_checkpoint, start_distributed_if_launched
    if "RANK" in os.environ:
        rank, world = start_distributed_if_launched()
    else:
        rank, world = 0, 1
    all_names = M.muon_matrix_names()
    assert len(all_names) == 72, len(all_names)
    names = pilot_matrix_names(all_names) if args.pilot else all_names

    model_path = os.path.join(args.state_dir, f"train_state_model_step{args.step:06d}.pt")
    model, params = load_model_at_checkpoint(model_path, [], names)
    radii = radii_for(params)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"step {args.step}: {len(names)} matrices, radii sample {radii[:3]}, {world} ranks", flush=True)

    # TRUE disjoint probe subsets = distinct training files (reproducible by filename, committed below).
    # subset s -> fineweb_train_{s+1:06d}.bin; nested sensitivity = subset 0's file at nested_tokens
    # (a strict superset of subset 0's first subset_tokens window, so it is genuinely nested).
    subset_globs = [f"data/fineweb10B/fineweb_train_{s+1:06d}.bin" for s in range(args.n_subsets)]
    results = {"provenance": dict(subset_files=subset_globs, subset_tokens=args.subset_tokens,
                                  nested_file=subset_globs[0], nested_tokens=args.nested_tokens,
                                  seq_len=1024, mbs=args.mbs)}
    for s in range(args.n_subsets):
        if rank == 0:
            print(f"subset {s}: {args.subset_tokens} tokens from {subset_globs[s]}", flush=True)
        res = run_subset(model, params, names, radii, args, rank, world, subset_globs[s],
                         args.subset_tokens, do_extras=(s == 0))
        res["subset"] = s
        res["data"] = subset_globs[s]
        results[f"subset{s}"] = res
        if world > 1 and dist.is_initialized():
            dist.barrier()
    if args.nested_tokens:
        if rank == 0:
            print(f"nested sensitivity: {args.nested_tokens} tokens from {subset_globs[0]}", flush=True)
        res = run_subset(model, params, names, radii, args, rank, world, subset_globs[0],
                         args.nested_tokens, do_extras=False)
        res["subset"] = "nested"
        res["data"] = subset_globs[0]
        results["nested"] = res

    if rank == 0:
        tag = f"pilot_step{args.step}"
        with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
            json.dump(results, f, indent=1)
        print(f"REQ057_PILOT_DONE wrote {os.path.join(args.out_dir, tag+'.json')}", flush=True)
    if world > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
