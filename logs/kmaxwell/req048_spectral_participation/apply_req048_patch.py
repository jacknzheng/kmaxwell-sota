"""REQ-048: add spectral participation ratio (m=16 Hutchinson) + curvature_along_weight/random to the curvature probe.
Admissible under rule 13: fresh random probes, no Lanczos/tridiagonal state. PR = trace(H)^2/(n*trace(H^2)) is
scale-invariant (batched_block_hvp's BATCH_TOKENS scale cancels in the ratio)."""
import sys
P="records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py"
s=open(P).read()

# 1. insert the PR / diagnostic HVP block right after the polar HVP line.
anchor="        hv_p = batched_block_hvp(model, params, polars, args.data, args.tokens, args.mbs)\n"
block='''        # === REQ-048: spectral participation ratio via m=16 Hutchinson probes (fresh Rademacher;
        # admissible under rule 13 -- no alphas/offdiags/eigh). PR = trace(H)^2 / (n * trace(H^2)),
        # scale-invariant. Muon matrices only (skip embed/proj). +m+2 HVPs ~= +2x probe time. ===
        M_PROBES = 16
        muon = [i for i in range(len(params)) if not my_names[i].startswith(("embed", "proj"))]
        t1 = [0.0]*len(params); t2 = [0.0]*len(params); perprobe = [[] for _ in params]
        for pj in range(M_PROBES):
            gen = torch.Generator(device=params[muon[0]].device).manual_seed(10_000*step + 100*pj + rank)
            rvecs = [((torch.randint(0, 2, params[i].shape, generator=gen, device=params[i].device,
                                     dtype=torch.float32)*2 - 1) if i in muon else None)
                     for i in range(len(params))]
            hvh = batched_block_hvp(model, params, rvecs, args.data, args.tokens, args.mbs)
            for i in muon:
                vHv = float((rvecs[i]*hvh[i]).sum()); Hv2 = float((hvh[i]*hvh[i]).sum())
                t1[i]+=vHv; t2[i]+=Hv2; perprobe[i].append(vHv)
            del rvecs, hvh
        t1 = [x/M_PROBES for x in t1]; t2 = [x/M_PROBES for x in t2]
        wdirs = [(params[i]/params[i].norm() if (i in muon and float(params[i].norm())>0) else None)
                 for i in range(len(params))]
        hvw = batched_block_hvp(model, params, wdirs, args.data, args.tokens, args.mbs)
        caw = [(float((hvw[i]*wdirs[i]).sum()) if wdirs[i] is not None else None) for i in range(len(params))]
        rgen = torch.Generator(device=params[muon[0]].device).manual_seed(777 + rank)
        rdirs = [(torch.randn(params[i].shape, generator=rgen, device=params[i].device) if i in muon else None)
                 for i in range(len(params))]
        rdirs = [(d/d.norm() if d is not None else None) for d in rdirs]
        hvr = batched_block_hvp(model, params, rdirs, args.data, args.tokens, args.mbs)
        car = [(float((hvr[i]*rdirs[i]).sum()) if rdirs[i] is not None else None) for i in range(len(params))]
'''
assert anchor in s and s.count(anchor)==1, "polar-hvp anchor missing/non-unique"
s=s.replace(anchor, anchor+block, 1)

# 2. add the new fields to step_out[name] (before the closing paren of the dict).
rec_anchor="                shape=list(params[i].shape))\n"
rec_new='''                shape=list(params[i].shape),
                trace_est=t1[i], trace_sq_est=t2[i], n_params=int(params[i].numel()),
                participation_ratio=(t1[i]**2/(params[i].numel()*t2[i]) if (t2[i] and t2[i]>0) else None),
                pr_per_probe_vHv=perprobe[i],
                curvature_along_weight=caw[i], curvature_along_random=car[i])
'''
assert rec_anchor in s and s.count(rec_anchor)==1, "record anchor missing/non-unique"
s=s.replace(rec_anchor, rec_new, 1)
open(P,"w").write(s)
print("REQ-048 patch applied to measure_per_matrix_curvature.py")
