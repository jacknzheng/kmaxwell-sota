"""REQ-072 spectrogram analysis driver (runs ON the box). For one arm, for each selected matrix and each
stream (conditioned u_t primary; raw g_t secondary), assemble the (N_coords, T) temporal series from the
capture reader, compute the temporal STFT aggregate power (req072_spectrogram), and emit a compact JSON:
time-averaged spectrum P(f), period-two/low/mid band fractions, total + mean-per-coord power. u_t is the
key stream (the conditioned direction); g_t shows the raw-gradient spectrum for contrast.

Mean-per-token for g_t (divide by tokens); u_t/displacement are in optimizer-update units (no scaling).
Memory-bounded: one matrix-stream (N,T) at a time (~15 GB for 768x768 x 3250 fp64).

Usage: req072_analyze.py --arm_dir <dur>/<arm> --names blocks.0.attn.proj.weight ... --tokens 524288
       --window 64 --hop 32 --out <d>/<arm>.json
"""
import argparse, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "req068_full_gradient_history", "impl"))
from req068_capture import GradientHistoryReader
from req072_spectrogram import stft_power, band_power


def assemble_series(reader, name, scale):
    steps = reader.steps()
    T = len(steps)
    first = reader.grad(steps[0], name)
    N = int(first.numel())
    series = np.empty((N, T), dtype=np.float64)
    for j, s in enumerate(steps):
        g = reader.grad(s, name)
        series[:, j] = (g.float().cpu().numpy().astype(np.float64).ravel() / scale)
    return series, steps


def analyze(arm_dir, names, tokens, window, hop):
    out = {"arm_dir": arm_dir, "window": window, "hop": hop, "streams": {}}
    for stream, sub, scale in (("u_t", "uT", 1.0), ("g_t", "grad", float(tokens))):
        d = os.path.join(arm_dir, sub)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        r = GradientHistoryReader(d)
        out["streams"][stream] = {}
        for name in names:
            series, steps = assemble_series(r, name, scale)
            sp = stft_power(series, window_W=window, hop=hop, fs=1.0)
            bp = band_power(sp["freqs"], sp["spectrum_f"])
            out["streams"][stream][name] = {
                "freqs": sp["freqs"], "spectrum_f": sp["spectrum_f"].tolist(),
                "band": bp, "total_power": sp["total_power"],
                "mean_per_coord_power": sp["mean_per_coord_power"], "N": sp["N"], "T": sp["T"]}
            print(f"  {stream} {name}: period2={bp['period2_frac']:.3f} low={bp['low_frac']:.3f} "
                  f"p2/low={bp['period2_over_low']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=524288); ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--hop", type=int, default=32); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyze(a.arm_dir, a.names, a.tokens, a.window, a.hop)
    json.dump(res, open(a.out, "w"))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
