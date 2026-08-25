#!/usr/bin/env python3
"""Statsig scoring for the WR fleets + Fleet C, from pulled ledger verdicts.

Usage:
  score_wr.py wr   <ledger_dir> <out_dir>     # Fleet A (CWDf_s*) + Fleet B (BM339_s*)
  score_wr.py abl  <ledger_dir> <out_dir> <prefix>   # Fleet C n=8 (e.g. ANL64_s)

Protocol (Track 3 acceptance, NOT first-cross):
  at each grid step S: mean over seeds; margin = (3.28 - mean) * sqrt(n)
  reported step = smallest S with margin >= 0.004 (n must be 8)
Pairwise (README formula): lhs = (bm_mean - km_mean) / sqrt(1/8 + 1/8) = diff/0.5
"""
import sys, os, json, math

TARGET = 3.28
MARGIN_REQ = 0.004

def load_curves(ledger, prefix, key):
    curves = {}
    for s in range(8):
        p = f"{ledger}/{prefix}{s}/verdict.json"
        if not os.path.exists(p):
            continue
        v = json.load(open(p))
        if v.get("status") not in ("ok",):
            print(f"WARN {prefix}{s} status={v.get('status')} problems={v.get('problems')}")
        c = v.get(key) or {}
        curves[s] = {int(k): float(x) for k, x in c.items()}
    return curves

def table(curves, grid):
    rows = []
    n = len(curves)
    for S in grid:
        vals = [curves[s][S] for s in sorted(curves) if S in curves[s]]
        if len(vals) != n or n == 0:
            rows.append((S, None, None, len(vals)))
            continue
        mean = sum(vals) / n
        margin = (TARGET - mean) * math.sqrt(n)
        rows.append((S, mean, margin, n))
    return rows

def first_pass(rows):
    for S, mean, margin, n in rows:
        if mean is not None and n == 8 and margin >= MARGIN_REQ:
            return S
    return None

def emit(rows, curves, path, label):
    seeds = sorted(curves)
    with open(path, "w") as f:
        f.write("step\t" + "\t".join(f"seed{s}" for s in seeds) + "\tmean\tmargin\tpass\n")
        for S, mean, margin, n in rows:
            per = "\t".join(f"{curves[s][S]:.6f}" if S in curves[s] else "-" for s in seeds)
            if mean is None:
                f.write(f"{S}\t{per}\t-\t-\tincomplete({n})\n")
            else:
                f.write(f"{S}\t{per}\t{mean:.6f}\t{margin:.5f}\t{'PASS' if margin >= MARGIN_REQ else 'fail'}\n")
    fp = first_pass(rows)
    print(f"{label}: n={len(curves)} first_passing_S={fp}")
    return fp

def main():
    mode, ledger, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    if mode == "wr":
        grid = list(range(2580, 2725, 5))
        km = load_curves(ledger, "CWDf_s", "ema_curve")
        bm = load_curves(ledger, "BM339_s", "ema_curve")
        km_rows = table(km, grid); bm_rows = table(bm, grid)
        kfp = emit(km_rows, km, f"{out}/cwd_frozen_n8_summary.tsv", "KM (CWD frozen)")
        bfp = emit(bm_rows, bm, f"{out}/bimaxwell339_n8_summary.tsv", "#339 bi-Maxwell")
        # combined pairwise table
        with open(f"{out}/pairwise.tsv", "w") as f:
            f.write("step\tkm_mean\tkm_margin\tbm_mean\tbm_margin\tpairwise_lhs\n")
            for (S, kmn, kmg, kn), (_, bmn, bmg, bn) in zip(km_rows, bm_rows):
                if kmn is None or bmn is None:
                    f.write(f"{S}\t-\t-\t-\t-\t-\n"); continue
                lhs = (bmn - kmn) / 0.5
                f.write(f"{S}\t{kmn:.6f}\t{kmg:.5f}\t{bmn:.6f}\t{bmg:.5f}\t{lhs:.5f}\n")
        print(f"KM first-pass={kfp} (record 2690) | #339 first-pass={bfp} (their H100 2645)")
        if kfp and bfp and kfp != bfp:
            print(f"step-extrapolation needed: delta_steps={bfp - kfp}")
    elif mode == "abl":
        prefix = sys.argv[4]
        grid = [3150, 3160, 3170, 3180, 3190, 3200, 3210, 3225, 3250]
        # kmanneal verdicts: curve = {step: {val_loss: ...}}
        curves = {}
        for s in range(8):
            p = f"{ledger}/{prefix}{s}/verdict.json"
            if not os.path.exists(p): continue
            v = json.load(open(p))
            curves[s] = {int(k): float(x["val_loss"]) for k, x in (v.get("curve") or {}).items()}
        rows = table(curves, grid)
        fp = emit(rows, curves, f"{out}/{prefix.rstrip('_s')}_n8_summary.tsv", prefix)
        print(f"vs #340: their 3210 mean=3.27817 margin=0.00518; 3200 mean=3.27881; first-pass S<3210 beats them")
        for S, mean, margin, n in rows:
            if S in (3200, 3210) and mean is not None:
                ref = 3.27881 if S == 3200 else 3.27817
                print(f"  pairwise@{S}: lhs=({ref:.6f}-{mean:.6f})/0.5={(ref-mean)/0.5:.5f} (sig if >=0.004)")
    else:
        sys.exit(f"unknown mode {mode}")

if __name__ == "__main__":
    main()
