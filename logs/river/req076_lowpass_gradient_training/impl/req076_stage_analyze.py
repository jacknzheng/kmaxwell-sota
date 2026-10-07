"""REQ-076/077 stage analysis (runs ON the box): per-arm lag-1 cosine + norms across stages, and the key
'intervention chain' check -- does filtering reduce period-two in the ACTUAL displacement (s4), not just in
the filtered gradient (s5)? Reads s1_grad/s5_filtered/s2_premom/s3_postpolar/s4_disp captures.

Usage: req076_stage_analyze.py --arm_dir <dur>/<arm> --out <json>   [--tail N]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "/root/impl")
from req068_capture import GradientHistoryReader

NAMES = ["blocks.5.attn.q.weight", "blocks.5.attn.k.weight", "blocks.5.mlp.proj.weight"]
STAGES = ["s1_grad", "s5_filtered", "s2_premom", "s3_postpolar", "s4_disp"]


def lag1(V):
    a, b = V[:-1], V[1:]
    num = (a * b).sum(1); den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1); m = den > 0
    return float((num[m] / den[m]).mean()) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tail", type=int, default=120)
    a = ap.parse_args()
    out = {"arm_dir": a.arm_dir, "matrices": {}}
    for nm in NAMES:
        out["matrices"][nm] = {}
        for st in STAGES:
            d = os.path.join(a.arm_dir, st)
            if not os.path.exists(os.path.join(d, "manifest.json")):
                continue
            r = GradientHistoryReader(d); steps = r.steps()[-a.tail:]
            V = np.stack([r.grad(s, nm).float().cpu().numpy().ravel() for s in steps])
            out["matrices"][nm][st] = {"lag1_cos": lag1(V), "norm_mean": float(np.linalg.norm(V, axis=1).mean()),
                                       "n": len(steps)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    q = out["matrices"]["blocks.5.attn.q.weight"]
    print(f"  {os.path.basename(a.arm_dir):14s} "
          + " ".join(f"{st.split('_')[0]}={q[st]['lag1_cos']:+.3f}" for st in STAGES if st in q)
          + f"  dispnorm={q.get('s4_disp',{}).get('norm_mean',float('nan')):.3e}", flush=True)


if __name__ == "__main__":
    main()
