"""REQ-064 pretreatment feature probe. At a saved base state (post-switch, before the first affected
update), measures for each Muon matrix, at the SAME weights + diagnostic token ranges:
  - M2 sharpness: isolated shape-weighted spectral S_i, G_i=r_i*||g_i||_nuclear, S_i/G_i (REQ-057 FW).
  - Euclidean comparator: lambda_i via Lanczos (8 iters + convergence), lambda_i/||g_i||_F^2.
  - M1 type/depth prior is derived offline from S_i (no extra measurement).
18 sentinels get the full FW witness + budget data (K20/K50, 5 starts); all 72 get the cheap features.
Reuses the REQ-057 driver primitives (grads, joint/diagonal HVP, FW) + req064_features Lanczos. Mean-per-
valid-token loss scale. Writes one JSON per (seed, subset). M3 actual-direction is a separate pass.

Usage (from repo root, torchrun --nproc_per_node=8):
  measure_req064_features.py --state_dir req064_state_s0 --step 2000 [--pilot] \
      --iters_list 20 50 --restarts 5 --lanczos_iters 8 --out_dir logs/kmaxwell/req064_.../raw/feat_s0
"""
import argparse, json, os, sys
import torch
import torch.distributed as dist

# REQ-057 primitives module + driver helpers, and the harness offline_analysis utility
# (check_secant_direction_with_hvp: load_model_at_checkpoint, start_distributed_if_launched).
for _p in ("/root/kmaxwell-sota/logs/kmaxwell/req057_layerwise_spectral_sharpness/impl",
           "logs/kmaxwell/req057_layerwise_spectral_sharpness/impl",
           "/root/kmaxwell-sota/records/track_3_optimization/offline_analysis",
           "records/track_3_optimization/offline_analysis",
           os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, _p)
import measure_layerwise_spectral_sharpness as M
import measure_layerwise_pilot as P
import req064_features as F


def run_subset_features(model, params, names, radii, args, rank, world, data, tokens, do_full_fw):
    grads = P.block_grads_mean(model, params, data, tokens, args.mbs, rank, world)
    Gi, Gjoint = M.G_denominators(grads, radii)

    def joint_hvp(v):
        return P.joint_hvp_mean(model, params, v, data, tokens, args.mbs, rank, world)

    def diag_hvp_one(i, vi):
        ej = [torch.zeros_like(p) for p in params]
        ej[i] = vi
        return joint_hvp(ej)[i]

    per = {}
    for i, name in enumerate(names):
        rec = {"G_i": Gi[i], "gF2": float((grads[i].double() ** 2).sum())}
        # M2 spectral S_i via REQ-057 isolated FW witness (shared diag HVP)
        s_res = M.isolated_sharpness(lambda j, v, _i=i: diag_hvp_one(_i, v), grads[i], radii[i], 0,
                                     iters_list=args.iters_list, restarts=args.restarts)
        S_i = s_res["best_sphere"]
        rec["S_i"] = S_i
        rec["S_i_over_G_i"] = (S_i / Gi[i]) if (S_i is not None and Gi[i]) else None
        rec["S_i_per_k"] = {str(k): s_res["best_sphere_per_k"][k] for k in args.iters_list}
        rec["negative_curvature"] = s_res.get("negative_curvature")
        rec["Z_i"] = (M.Z_stat(args.eta, S_i, Gi[i]) if S_i is not None else None)
        # Euclidean comparator via Lanczos on H_ii (same diag_hvp_one)
        eu = F.euclidean_lambda(diag_hvp_one, i, grads[i], iters=args.lanczos_iters, seed=1337)
        rec["lambda_i"] = eu["lambda_i"]
        rec["lambda_over_gF2"] = eu["lambda_over_gF2"]
        rec["lambda_ritz_residual"] = eu["ritz_residual"]
        rec["lambda_ritz_history"] = eu["ritz_history"]
        rec["lambda_converged"] = eu["converged"]
        per[name] = rec
        if rank == 0 and (i % 12 == 0):
            print(f"  [{i}/{len(names)}] {name} S_i={S_i} lambda_i={eu['lambda_i']:.4g} resid={eu['ritz_residual']:.2e}", flush=True)
    return {"G_joint": Gjoint, "per_matrix": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.025)
    ap.add_argument("--iters_list", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--lanczos_iters", type=int, default=8)
    ap.add_argument("--subset_tokens", type=int, default=8192)
    ap.add_argument("--n_subsets", type=int, default=3)
    ap.add_argument("--nested_tokens", type=int, default=32768)
    ap.add_argument("--pilot", action="store_true", help="restrict to the 18 sentinel matrices")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available()

    from check_secant_direction_with_hvp import load_model_at_checkpoint, start_distributed_if_launched
    rank, world = (P.start_distributed_if_launched() if "RANK" in os.environ else (0, 1))
    all_names = M.muon_matrix_names()
    assert len(all_names) == 72, len(all_names)
    names = P.pilot_matrix_names(all_names) if args.pilot else all_names

    model_path = os.path.join(args.state_dir, f"train_state_model_step{args.step:06d}.pt")
    model, params = load_model_at_checkpoint(model_path, [], names)
    radii = P.radii_for(params)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"REQ-064 features: step {args.step}, {len(names)} matrices, {world} ranks, lanczos {args.lanczos_iters}", flush=True)

    subset_globs = [f"data/fineweb10B/fineweb_train_{s+1:06d}.bin" for s in range(args.n_subsets)]
    results = {"provenance": dict(state_dir=args.state_dir, step=args.step, eta=args.eta,
                                  subset_files=subset_globs, subset_tokens=args.subset_tokens,
                                  nested_file=subset_globs[0], nested_tokens=args.nested_tokens,
                                  seq_len=1024, mbs=args.mbs, lanczos_iters=args.lanczos_iters,
                                  fw_iters_list=args.iters_list, fw_restarts=args.restarts)}
    for s in range(args.n_subsets):
        if rank == 0:
            print(f"subset {s}: {args.subset_tokens} tok from {subset_globs[s]}", flush=True)
        res = run_subset_features(model, params, names, radii, args, rank, world,
                                  subset_globs[s], args.subset_tokens, do_full_fw=True)
        res["subset"] = s; res["data"] = subset_globs[s]
        results[f"subset{s}"] = res
        if world > 1 and dist.is_initialized():
            dist.barrier()
    if args.nested_tokens:
        if rank == 0:
            print(f"nested: {args.nested_tokens} tok from {subset_globs[0]}", flush=True)
        res = run_subset_features(model, params, names, radii, args, rank, world,
                                  subset_globs[0], args.nested_tokens, do_full_fw=False)
        res["subset"] = "nested"; res["data"] = subset_globs[0]
        results["nested"] = res

    if rank == 0:
        out = os.path.join(args.out_dir, f"features_{os.path.basename(args.state_dir)}_step{args.step:06d}.json")
        json.dump(results, open(out, "w"), indent=1)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
