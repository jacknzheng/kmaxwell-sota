"""REQ-073 stage-spectra analysis (runs ON the box). For one arm, for each of the 4 pipeline stages
(s1_grad, s2_premom, s3_postpolar, s4_disp) and each selected matrix (blocks.5 Q/K/mlp.proj), assemble the
(N_coords, T) temporal series from the capture reader and compute the STFT aggregate power + band fractions
(reuse req072_spectrogram). This shows WHERE in the Muon pipeline the period-two→low-f transition happens,
per matrix, per arm (B/4B × momentum). Memory-bounded (one matrix-stage at a time). Writes compact JSON.

s1_grad is mean-per-token (÷ batch_tokens); s2/s3/s4 are in optimizer-update units (no scale). For the 4B arms
s1 raw is the pre-divide 4B sum, so ÷ batch_tokens still gives mean-per-token.

Usage: req073_analyze.py --arm_dir <dur>/<arm> --names <3 names> --batch_tokens <B or 4B> --window 64 --hop 32 --out <d>.json
"""
import argparse, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "req072_optimizer_spectrograms", "impl"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "req068_full_gradient_history", "impl"))
from req068_capture import GradientHistoryReader
from req072_spectrogram import stft_power, band_power

STAGES = [("s1_grad", "s1_grad"), ("s2_premom", "s2_premom"),
          ("s3_postpolar", "s3_postpolar"), ("s4_disp", "s4_disp")]


def assemble(reader, name, scale):
    steps = reader.steps(); T = len(steps)
    N = int(reader.grad(steps[0], name).numel())
    ser = np.empty((N, T), dtype=np.float64)
    for j, s in enumerate(steps):
        ser[:, j] = reader.grad(s, name).float().cpu().numpy().astype(np.float64).ravel() / scale
    return ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--batch_tokens", type=int, default=524288)
    ap.add_argument("--window", type=int, default=64); ap.add_argument("--hop", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"arm_dir": a.arm_dir, "batch_tokens": a.batch_tokens, "window": a.window, "stages": {}}
    for stage, sub in STAGES:
        d = os.path.join(a.arm_dir, sub)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        r = GradientHistoryReader(d)
        scale = float(a.batch_tokens) if stage == "s1_grad" else 1.0
        res["stages"][stage] = {}
        for name in a.names:
            ser = assemble(r, name, scale)
            sp = stft_power(ser, window_W=a.window, hop=a.hop)
            bp = band_power(sp["freqs"], sp["spectrum_f"])
            res["stages"][stage][name] = {"band": bp, "spectrum_f": sp["spectrum_f"].tolist(),
                                          "freqs": sp["freqs"], "total_power": sp["total_power"],
                                          "mean_per_coord_power": sp["mean_per_coord_power"]}
            print(f"  {stage:13s} {name[-18:]}: period2={bp['period2_frac']:.3f} low={bp['low_frac']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
