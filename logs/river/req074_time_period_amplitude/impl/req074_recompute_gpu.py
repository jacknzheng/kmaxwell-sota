"""REQ-074 GPU recompute: identical math to req074_spectrogram.time_resolved_power (Hann, per-window linear
detrend optional, rfft, |F|^2 summed over coords, /(fs*win_energy)), but (a) loads each capture chunk ONCE
filling all requested matrices together (3x less durable-FS I/O than per-matrix passes) and (b) runs the STFT
on GPU via torch.fft in fp64. Parity-checked against the numpy reference (--selfcheck). Saves the same .npz
schema req074_recompute writes, so req074_plot consumes it unchanged.

Usage: req074_recompute_gpu.py --src_dir <dir> --names <m1 m2 m3> --tokens <B> --out <x>.npz [--device cuda]
       req074_recompute_gpu.py --selfcheck
"""
import argparse, os, sys, time, warnings
import numpy as np, torch
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req068_capture import GradientHistoryReader
from req074_spectrogram import time_resolved_power as _ref

WINDOWS = [128, 32, 256]


def stft_power_gpu(series, window_W=128, hop=1, fs=1.0, detrend=False, device="cuda",
                   coord_block=2048, win_block=512):
    """series: (N,T) np fp32. Returns dict matching req074_spectrogram.time_resolved_power."""
    N, T = series.shape
    h = torch.hann_window(window_W, periodic=False, dtype=torch.float64, device=device)
    win_energy = float((h ** 2).sum().item())
    starts = list(range(0, T - window_W + 1, hop))
    step_end = [s + window_W - 1 for s in starts]
    freqs = np.fft.rfftfreq(window_W, d=1.0 / fs)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    nF = len(freqs); nW = len(starts)
    P = torch.zeros((nW, nF), dtype=torch.float64, device=device)
    x = torch.arange(window_W, dtype=torch.float64, device=device)
    xm = x.mean(); sxx = ((x - xm) ** 2).sum()
    for b0 in range(0, N, coord_block):
        blk = torch.from_numpy(series[b0:b0 + coord_block]).to(device=device, dtype=torch.float64)  # (Nb,T)
        # all windows via unfold: (Nb, nW, W) with step=hop
        w = blk.unfold(dimension=1, size=window_W, step=hop)  # view
        # process window columns in sub-batches to bound memory
        for w0 in range(0, nW, win_block):
            seg = w[:, w0:w0 + win_block, :].contiguous().to(torch.float64)  # (Nb, wb, W)
            if detrend:
                mean = seg.mean(dim=2, keepdim=True)
                slope = ((seg - mean) * (x - xm)).sum(dim=2, keepdim=True) / sxx
                seg = seg - (slope * (x - xm) + mean)
            seg = seg * h
            F = torch.fft.rfft(seg, dim=2)                    # (Nb, wb, nF)
            P[w0:w0 + seg.shape[1]] += (F.abs() ** 2).sum(dim=0)
        del blk, w
    P = (P / (fs * win_energy)).cpu().numpy()
    return {"freqs": freqs.tolist(), "periods": periods.tolist(), "step_end": step_end,
            "P": P, "window": window_W, "hop": hop, "detrend": detrend, "N": N, "T": T,
            "n_columns": nW, "first_valid_step_end": step_end[0] if step_end else None}


def selfcheck(device):
    rng = np.random.default_rng(1); N, T = 300, 600
    t = np.arange(T)
    sig = (((-1.0) ** t)[None, :] * rng.standard_normal((N, 1))
           + 0.3 * np.sin(2 * np.pi * t / 10)[None, :] + 0.01 * t[None, :] * rng.standard_normal((N, 1)))
    sig = sig.astype(np.float32)
    ok = True
    for W in (128, 32):
        for det in (False, True):
            a = _ref(sig, window_W=W, hop=1, detrend=det)
            b = stft_power_gpu(sig, window_W=W, hop=1, detrend=det, device=device)
            rel = np.abs(a["P"] - b["P"]).max() / (np.abs(a["P"]).max() + 1e-30)
            good = rel < 1e-9
            print(f"  selfcheck W={W} detrend={det}: max_rel_err={rel:.2e} {'PASS' if good else 'FAIL'}")
            ok &= good
    print("SELFCHECK", "PASS" if ok else "FAIL"); return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--src_dir"); ap.add_argument("--names", nargs="+")
    ap.add_argument("--tokens", type=float, default=1.0); ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if a.selfcheck:
        sys.exit(0 if selfcheck(dev) else 1)
    r = GradientHistoryReader(a.src_dir)
    v = r.verify()
    steps = r.steps(); T = len(steps)
    N = {nm: int(r.grad(steps[0], nm).numel()) for nm in a.names}
    ser = {nm: np.empty((N[nm], T), dtype=np.float32) for nm in a.names}
    t0 = time.time()
    for j, s in enumerate(steps):              # chunk cache => each chunk read once for ALL names
        fs = r.full_step(s)
        for nm in a.names:
            ser[nm][:, j] = fs[nm].float().cpu().numpy().ravel() / a.tokens
    print(f"  loaded {len(a.names)} series (N={N}) T={T} in {time.time()-t0:.0f}s (verify_ok={v['ok']})", flush=True)
    save = {"src_dir": a.src_dir, "verify_ok": int(v["ok"]), "n_steps": T,
            "step0": steps[0], "stepN": steps[-1], "tokens": a.tokens, "device": dev}
    usable = [W for W in WINDOWS if W <= T]     # short series (e.g. 46-update 16B) skip oversized windows
    for nm in a.names:
        key = nm.replace(".", "_"); tc = time.time()
        first = True
        for W in usable:
            res = stft_power_gpu(ser[nm], window_W=W, hop=1, device=dev)
            save[f"{key}__w{W}__P"] = res["P"].astype(np.float32)
            save[f"{key}__w{W}__step_end"] = np.array(res["step_end"])
            if first:
                save[f"{key}__freqs"] = np.array(res["freqs"]); save[f"{key}__periods"] = np.array(res["periods"])
                first = False
        if 128 <= T:
            resd = stft_power_gpu(ser[nm], window_W=128, hop=1, detrend=True, device=dev)
            save[f"{key}__w128_detrend__P"] = resd["P"].astype(np.float32)
        print(f"  {nm}: windows={usable}+detrend(if T>=128) ({time.time()-tc:.0f}s)", flush=True)
        del ser[nm]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **save)
    print(f"WROTE {a.out} (verify_ok={v['ok']}, {len(a.names)} matrices)", flush=True)


if __name__ == "__main__":
    main()
