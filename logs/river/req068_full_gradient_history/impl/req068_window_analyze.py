"""REQ-068 amended analysis driver (runs ON the box). For each window (500/1500/2500) and each stream
(raw gradient mean-per-token, actual displacement delta_t), builds the 64x64 temporal Gram via
req068_gram and emits a compact JSON (K, C, eigenvalues, signed coefficient time-courses for top modes,
centered-variant eigenvalues, lag-1/2/4/8/16/32 agreement, per-parameter-family K energy share, and the
out-of-sample first-32->next-32 captured/residual energy at ranks 1/2/4/8 vs past-mean/exp-avg baselines).

Gradient stream dirs come from the FULL run (window steps are a subset) OR a window fork capture; the
displacement stream comes from the window fork capture. Vectors are whole-model, assembled per step from
the chunk reader (one chunk holds all 64 window steps when chunk_steps=64).

Usage: req068_window_analyze.py --grad_dir <d> --disp_dir <d> --window 500 --tokens 524288 --out <d>/win500.json
       (grad_dir may be the full-run grads; the window steps are selected by --window)
"""
import argparse, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req068_capture import GradientHistoryReader
from req068_gram import (temporal_gram, cosine_from_gram, gram_eig, signed_coefficients,
                         center_gram, lag_agreement, out_of_sample)

FAMILIES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj", "embed", "head", "other"]
def family_of(n):
    if n.startswith("embed"): return "embed"
    if n.startswith("proj"): return "head"
    for t in FAMILIES[:6]:
        if t in n: return t
    return "other"


def load_window_vectors(reader, window, scale):
    """List (len 64) of {name: 1-D float64 array} for steps window..window+63; grads /scale (mean-per-token)."""
    steps = list(range(window, window + 64))
    names = reader.params()
    out = []
    for s in steps:
        fs = reader.full_step(s)
        out.append({n: (fs[n].to("cpu").numpy().astype(np.float64).ravel() / scale)
                    for n in names if fs[n] is not None})
    return out, [n for n in names if out[0].get(n) is not None]


def analyze_stream(reader, window, scale, label):
    vecs, names = load_window_vectors(reader, window, scale)
    K, famK = temporal_gram(vecs, names, family_of)
    C, zmask = cosine_from_gram(K)
    lam, V = gram_eig(K)
    coeff = signed_coefficients(lam, V)
    Kc = center_gram(K); lam_c, _ = gram_eig(Kc)
    total = float(np.trace(K))
    fam_share = {f: float(np.trace(m)) / total for f, m in famK.items()} if total > 0 else {}
    return {
        "label": label, "window": window, "n": len(vecs), "scale": scale,
        "K_diag": np.diag(K).tolist(),
        "eigenvalues": lam.tolist(), "eigenvalue_energy_frac": (lam / lam.sum()).tolist() if lam.sum() > 0 else [],
        "centered_eigenvalues": lam_c.tolist(),
        "coeff_top8": coeff[:, :8].tolist(),              # signed time-course of the 8 leading modes
        "lag_agreement": lag_agreement(C),
        "family_energy_share": fam_share,
        "near_zero_steps": int(zmask.sum()),
        "out_of_sample": out_of_sample(K, ranks=(1, 2, 4, 8), fit=32),
        "C": C.tolist(),                                   # 64x64 cosine (for heatmap)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad_dir"); ap.add_argument("--disp_dir")
    ap.add_argument("--window", type=int, required=True); ap.add_argument("--tokens", type=int, default=524288)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"window": a.window, "tokens_per_update": a.tokens}
    if a.grad_dir:
        r = GradientHistoryReader(a.grad_dir)
        res["gradient"] = analyze_stream(r, a.window, a.tokens, "raw_gradient_mean_per_token")
        print(f"grad window {a.window}: lag1={res['gradient']['lag_agreement'].get(1):+.3f} "
              f"OOS r8={res['gradient']['out_of_sample']['by_rank'][8]['captured_frac']:.3f}", flush=True)
    if a.disp_dir:
        r = GradientHistoryReader(a.disp_dir)
        res["displacement"] = analyze_stream(r, a.window, 1.0, "actual_displacement_delta")
        print(f"disp window {a.window}: lag1={res['displacement']['lag_agreement'].get(1):+.3f} "
              f"OOS r8={res['displacement']['out_of_sample']['by_rank'][8]['captured_frac']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
