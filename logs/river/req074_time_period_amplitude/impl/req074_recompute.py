"""REQ-074 recompute driver (runs ON the box): recompute TIME-RESOLVED spectrograms from the saved
REQ-072/073 raw stage histories (no training rerun). Per series (stream/stage × matrix), load the
(N_coords, T) temporal series ONCE, then compute P(step_end, period) for window 128 (primary), 32 & 256
(sensitivity), and 128-detrended — all hop=1 — and save to one compressed .npz. Memory-bounded: one series
in memory at a time; raw histories retained on the durable FS (never dropped).

Also records manifest verification (coverage + chunk count) per source dir.

Usage: req074_recompute.py --src_dir <history_dir> --names <matrices> --tokens <B> --out <dir>/<tag>.npz
       (src_dir holds the capture chunks+manifest for one stream/stage; tokens scales s1/grad to mean-per-token)
"""
import argparse, json, os, sys, warnings, time
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "req068_full_gradient_history", "impl"))
from req068_capture import GradientHistoryReader
from req074_spectrogram import time_resolved_power

WINDOWS = [128, 32, 256]  # primary first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--tokens", type=float, default=1.0, help="divide series by this (B for raw-grad -> mean/token)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = GradientHistoryReader(a.src_dir)
    v = r.verify()  # coverage + chunk hashes
    steps = r.steps()
    save = {"src_dir": a.src_dir, "verify_ok": int(v["ok"]), "n_steps": len(steps),
            "step0": steps[0], "stepN": steps[-1], "tokens": a.tokens}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for name in a.names:
        T = len(steps); N = int(r.grad(steps[0], name).numel())
        ser = np.empty((N, T), dtype=np.float32)   # fp32 series (halves memory; STFT accumulates fp64)
        t0 = time.time()
        for j, s in enumerate(steps):
            ser[:, j] = r.grad(s, name).float().cpu().numpy().ravel() / a.tokens
        key = name.replace(".", "_")
        for W in WINDOWS:
            res = time_resolved_power(ser, window_W=W, hop=1)
            save[f"{key}__w{W}__P"] = res["P"].astype(np.float32)
            save[f"{key}__w{W}__step_end"] = np.array(res["step_end"])
            if W == WINDOWS[0]:
                save[f"{key}__freqs"] = np.array(res["freqs"]); save[f"{key}__periods"] = np.array(res["periods"])
        resd = time_resolved_power(ser, window_W=128, hop=1, detrend=True)
        save[f"{key}__w128_detrend__P"] = resd["P"].astype(np.float32)
        print(f"  {name}: N={N} T={T} windows={WINDOWS}+detrend ({time.time()-t0:.0f}s)", flush=True)
        del ser
    np.savez_compressed(a.out, **save)
    print(f"WROTE {a.out} (verify_ok={v['ok']}, {len(a.names)} matrices)", flush=True)


if __name__ == "__main__":
    main()
