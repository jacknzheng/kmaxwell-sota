"""REQ-075 Stage 2 analysis (runs ON the box): per-arm mechanism summary from the 4-stage captures.

For each arm and each matrix (block-5 Q/K/MLP), reads the 4 stage histories (s1 raw grad, s2 post-momentum,
s3 post-polar, s4 displacement) over the 256-update pilot and computes, per update t:
  - stage norms ||v_t|| (grad in mean-per-token units for s1; others in optimizer-update units),
  - lag-1 and lag-2 cosine of the per-update vectors cos(v_t, v_{t+/-}) -- the DIRECT period-two signature
    (lag-1 < 0 == alternating/period-two motion; REQ-068 convention),
  - the switching transient (first 8 updates) reported separately from the settled tail.
Absolute + normalized period-two spectral power comes from the companion req074_recompute_gpu spectra (run
separately); here we report the time-domain lag agreement, which is the robust period-two discriminator.

Saves one JSON per run with per-matrix, per-stage series + early/late summaries. Lightweight (CPU).

Usage: req075_stage2_analyze.py --arm_dir <dur>/stage2/<arm> --tokens <batch> --out <dir>/<arm>.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "/root/impl")
sys.path.insert(0, "logs/river/req068_full_gradient_history/impl")
from req068_capture import GradientHistoryReader

NAMES = ["blocks.5.attn.q.weight", "blocks.5.attn.k.weight", "blocks.5.mlp.proj.weight"]
STAGES = [("s1_grad", "grad"), ("s2_premom", "post_momentum"), ("s3_postpolar", "post_polar"),
          ("s4_disp", "displacement")]
TRANSIENT = 8   # first updates after the mom/nomom switch, reported separately


def lag_cos(V, lag):
    """mean cosine between v_t and v_{t+lag} over t (V: (T,N) fp32)."""
    a = V[:-lag]; b = V[lag:]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    m = den > 0
    return float((num[m] / den[m]).mean()) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", required=True); ap.add_argument("--tokens", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"arm_dir": a.arm_dir, "tokens": a.tokens, "transient_updates": TRANSIENT, "matrices": {}}
    for name in NAMES:
        out["matrices"][name] = {}
        for sub, label in STAGES:
            d = os.path.join(a.arm_dir, sub)
            if not os.path.exists(os.path.join(d, "manifest.json")):
                continue
            r = GradientHistoryReader(d)
            steps = r.steps()
            scale = a.tokens if sub == "s1_grad" else 1.0
            V = np.stack([r.grad(s, name).float().cpu().numpy().ravel().astype(np.float32) / scale
                          for s in steps])     # (T, N)
            norms = np.linalg.norm(V, axis=1)
            tr, tail = slice(0, TRANSIENT), slice(TRANSIENT, None)
            out["matrices"][name][label] = {
                "n_updates": len(steps), "step0": steps[0], "stepN": steps[-1],
                "norm_mean": float(norms.mean()), "norm_tail_mean": float(norms[tail].mean()),
                "norm_transient_mean": float(norms[tr].mean()),
                "lag1_cos_all": lag_cos(V, 1), "lag2_cos_all": lag_cos(V, 2),
                "lag1_cos_tail": lag_cos(V[tail], 1), "lag2_cos_tail": lag_cos(V[tail], 2),
                "lag1_cos_transient": lag_cos(V[tr], 1) if TRANSIENT > 2 else float("nan"),
                "norm_series": norms.astype(np.float64).round(8).tolist()}
            print(f"  {name} {label}: norm_tail={norms[tail].mean():.3e} lag1_tail={lag_cos(V[tail],1):+.3f} "
                  f"lag2_tail={lag_cos(V[tail],2):+.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
