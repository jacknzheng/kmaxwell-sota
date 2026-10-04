"""REQ-075 Stage 1: frozen-state gradient-noise measurement at the REQ-073 B-mom @2500 checkpoint.

At the FROZEN common state x_2500, compute the selected-matrix (block-5 Q, K, MLP-proj) mean gradients on
`--probes` (default 64, up to 128) INDEPENDENTLY sampled B-sized probe batches drawn from a probe region
DISJOINT from the baseline/continuation data (and from val). Each B-probe reproduces the training gradient
exactly: per-rank 64-seq microbatch, token-SUM cross-entropy backward, all_reduce(SUM) across 8 ranks; then
divided by B for mean-per-token analysis units (REQ-073/074 convention). No optimizer, no param change.

Rank 0 then forms NESTED disjoint groups (1×B, 4B = 4 consecutive probes, 16B = 16 consecutive) -- paired
observations, not independent replicates -- and per matrix reports: within-state variance (trace of the
per-probe covariance), absolute mean-gradient (signal) norm with noise-bias correction, noise-to-signal
ratio at each batch size with bootstrap uncertainty, the measured noise-vs-batch scaling (departure from the
1/sqrt(B) ideal), and directional agreement (pairwise cosine) between the independent 16B group means.

PREDECLARED low-noise criterion (fixed BEFORE labelling any batch size): a batch size is "low-noise" iff its
estimated NSR (RMS gradient noise / bias-corrected signal norm) < 0.10, with the bootstrap 16-84% interval
reported. 4 nested 16B groups are a pilot variance estimate, not precise; limitation reported if 16B fails.

Raw per-probe gradients are saved off-Git on the durable FS (manifest); only the JSON summary is committed.

Usage (torchrun, 8 ranks, cwd = harness root so data globs resolve):
  torchrun --standalone --nproc_per_node=8 req075_stage1_noise.py \
    --model_ckpt <...>/train_state_model_step002500.pt \
    --probe_glob 'data/fineweb10B/fineweb_train_00010*.bin' --probes 64 \
    --out_json <dur>/stage1/noise.json --raw_dir <dur>/stage1/raw
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.distributed as dist

HARNESS = "/root/kmaxwell-sota/records/track_3_optimization"
sys.path.insert(0, HARNESS)
from harness.model_gpt import GPT
from harness.data_fineweb import iterate_batches_single_process

B_TOKENS = 524288
NAMES = ["blocks.5.attn.q.weight", "blocks.5.attn.k.weight", "blocks.5.mlp.proj.weight"]
LOW_NOISE_NSR = 0.10   # PREDECLARED criterion, fixed before any labelling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--probe_glob", required=True)
    ap.add_argument("--probes", type=int, default=64)
    ap.add_argument("--out_json", required=True); ap.add_argument("--raw_dir", required=True)
    a = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert world == 8, f"expected 8 ranks, got {world}"
    assert B_TOKENS % (world * 64 * 1024) == 0, "B must be one 64-seq microbatch per rank"

    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
    sd = torch.load(a.model_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()                                   # no dropout in this model; eval for determinism
    params = {n: p for n, p in model.named_parameters()}
    for n in NAMES:
        assert n in params, f"missing {n}"

    # probe iterator: each rank strides the global microbatch stream; one microbatch/rank == 1/8 of a B-probe
    it = iterate_batches_single_process(a.probe_glob, total_tokens=a.probes * B_TOKENS,
                                        microbatch_sequences=64, shard_rank=rank, shard_world=world)
    series = {n: [] for n in NAMES}                # rank 0: list of (N,) fp32 per-probe mean grads
    t0 = time.time()
    SUBSEQ = 8   # sub-microbatch (seqs) for forward/backward; grad SUM-accumulates -> identical to 64-seq
    for p in range(a.probes):
        model.zero_grad(set_to_none=True)
        inp, tgt = next(it)                         # (64, 1024) this rank's 1/8 of the B-probe
        nseq = inp.shape[0]
        assert nseq % SUBSEQ == 0
        for s in range(0, nseq, SUBSEQ):            # token-SUM loss => summing sub-chunk backwards == full
            model(inp[s:s+SUBSEQ], tgt[s:s+SUBSEQ]).backward()
        for n in NAMES:
            g = params[n].grad
            dist.all_reduce(g, op=dist.ReduceOp.SUM)     # global token-SUM grad (replicated params)
            if rank == 0:
                series[n].append((g.detach().float().cpu().numpy().ravel() / B_TOKENS).astype(np.float32))
        if rank == 0 and (p + 1) % 16 == 0:
            print(f"  probe {p+1}/{a.probes} ({time.time()-t0:.0f}s)", flush=True)
    dist.barrier()
    if rank != 0:
        dist.destroy_process_group(); return

    os.makedirs(a.raw_dir, exist_ok=True); os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    out = {"model_ckpt": a.model_ckpt, "probe_glob": a.probe_glob, "probes": a.probes,
           "B_tokens": B_TOKENS, "low_noise_nsr_criterion": LOW_NOISE_NSR, "matrices": {}}
    rng = np.random.default_rng(0)
    for n in NAMES:
        G = np.stack(series[n])                     # (P, N) fp32
        np.save(os.path.join(a.raw_dir, f"{n.replace('.', '_')}_probes.npy"), G)
        P, N = G.shape
        G64 = G.astype(np.float64)

        def group_stats(k):
            """Nested disjoint groups of k consecutive probes; returns group means + measured noise norm."""
            ng = P // k
            if ng < 1:
                return np.zeros((0, N)), np.zeros(N), float("nan"), 0
            gm = np.stack([G64[i*k:(i+1)*k].mean(0) for i in range(ng)])  # (ng, N) kB-batch means
            grand = gm.mean(0)
            # per-group deviation norm (noise of a kB estimate), RMS across groups
            noise = float(np.sqrt(((gm - grand) ** 2).sum(1).mean())) if ng > 1 else float("nan")
            return gm, grand, noise, ng

        gmB, grand, noiseB, ngB = group_stats(1)
        # per-probe covariance trace (within-state variance) and bias-corrected signal
        resid = G64 - grand
        cov_trace = float((resid ** 2).sum() / (P - 1))          # trace of single-probe covariance
        signal_norm2_biased = float((grand ** 2).sum())
        signal_norm2_corr = max(0.0, signal_norm2_biased - cov_trace / P)   # E||mean||^2 bias removal
        signal_norm = float(np.sqrt(signal_norm2_corr))
        signal_norm_biased = float(np.sqrt(signal_norm2_biased))

        _, _, noise4, ng4 = group_stats(4)
        gm16, grand16, noise16, ng16 = group_stats(16)
        # NSR at each batch size = (noise of that-size estimate) / corrected signal norm
        def nsr(noise): return float(noise / signal_norm) if signal_norm > 0 else float("inf")
        # bootstrap NSR_16B uncertainty over the 4 groups (pilot; limited)
        boot = []
        if ng16 > 1:
            for _ in range(2000):
                idx = rng.integers(0, ng16, ng16)
                gg = gm16[idx]; nz = np.sqrt(((gg - gg.mean(0)) ** 2).sum(1).mean())
                boot.append(nz / signal_norm if signal_norm > 0 else np.inf)
            b16, b50, b84 = np.percentile(boot, [16, 50, 84])
        else:
            b16 = b50 = b84 = float("nan")
        # directional agreement: pairwise cosine between independent 16B group means
        cos = []
        for i in range(ng16):
            for j in range(i + 1, ng16):
                a_, b_ = gm16[i], gm16[j]
                d = np.linalg.norm(a_) * np.linalg.norm(b_)
                if d > 0: cos.append(float(a_ @ b_ / d))
        # measured noise-vs-batch scaling vs 1/sqrt(B) ideal (normalized to B=1)
        scaling = {"noise_B": noiseB, "noise_4B": noise4, "noise_16B": noise16,
                   "ideal_4B": noiseB / 2.0, "ideal_16B": noiseB / 4.0}
        out["matrices"][n] = {
            "N": N, "within_state_variance_trace": cov_trace,
            "signal_norm_biased": signal_norm_biased, "signal_norm_bias_corrected": signal_norm,
            "nsr_B": nsr(noiseB), "nsr_4B": nsr(noise4), "nsr_16B": nsr(noise16),
            "nsr_16B_boot_16_50_84": [float(b16), float(b50), float(b84)],
            "dir_agreement_16B_cos_mean": float(np.mean(cos)) if cos else float("nan"),
            "dir_agreement_16B_cos_std": float(np.std(cos)) if cos else float("nan"),
            "noise_scaling": scaling,
            "low_noise_16B": bool(nsr(noise16) < LOW_NOISE_NSR),
            "n_groups_16B": ng16}
        print(f"  {n}: signal={signal_norm:.3e} NSR_B={nsr(noiseB):.3f} NSR_16B={nsr(noise16):.3f} "
              f"(boot {b16:.3f}-{b84:.3f}) dirCos16B={np.mean(cos) if cos else float('nan'):.3f} "
              f"low_noise_16B={nsr(noise16) < LOW_NOISE_NSR}", flush=True)
    json.dump(out, open(a.out_json, "w"), indent=2)
    print("WROTE", a.out_json, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
