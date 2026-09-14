"""REQ-059 allocation generator: assign a memory multiplier a in {0.5,1,2} to each of the 72 matrices under
the balanced constraint (exactly 4 of each type's 12 block-matrices at each a -> 24 matrices per a).
REQ-058 result: sharper matrices prefer SHORTER memory, so the guided rule gives the 4 sharpest per type
a=0.5, the 4 least-sharp a=2, the middle 4 a=1.

Arms:
  guided     : rank each type's matrices by measured spectral S_i (this seed's fork features)
  euclidean  : rank by Euclidean Lanczos lambda_i (same balanced constraint)
  typedepth  : rank by the DEV type/depth prior (mean S_i per (type,block) over dev seeds 0,1,2) -- no this-seed measurement
  shuffled   : guided assignment permuted within type (preserves each type's 4/4/4 histogram); one perm/seed
  reversed   : guided with a=0.5 and a=2 swapped (a=1 fixed)
  global a05/a1/a2 : all 72 at one a ; nomom : no-momentum Muon (beat all globals in REQ-058)
Usage: make_req059_alloc.py --features feats.json --dev_features dev.json --seed S --out alloc.json
feats.json: {matrix: {type, block, S_i, lambda_i}} for THIS seed's fork.
dev.json:   {matrix: {type, block, S_i_dev_mean}} averaged over dev seeds (for the typedepth prior).
"""
import argparse, json, random
from collections import defaultdict

TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]


def balanced_assign(matrices_by_type, key):
    """For each type, sort its matrices by key (desc); top 4 -> a=0.5, bottom 4 -> a=2, middle 4 -> a=1."""
    a = {}
    for typ, mats in matrices_by_type.items():
        s = sorted(mats, key=lambda m: -key(m))  # sharpest first
        n = len(s); q = n // 3
        for m in s[:q]: a[m["matrix"]] = 0.5          # sharpest -> shorter memory
        for m in s[q:2*q]: a[m["matrix"]] = 1.0
        for m in s[2*q:]: a[m["matrix"]] = 2.0        # least sharp -> longer memory
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--dev_features", default=None)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feats = json.load(open(args.features))
    by_type = defaultdict(list)
    for name, d in feats.items():
        by_type[d["type"]].append({"matrix": name, **d})

    guided = balanced_assign(by_type, key=lambda m: m["S_i"] if m["S_i"] is not None else -1e9)
    euclid = balanced_assign(by_type, key=lambda m: m.get("lambda_i") or -1e9)
    # type/depth prior: rank by dev-mean S_i per (type,block); fall back to this seed if no dev file
    if args.dev_features:
        dev = json.load(open(args.dev_features))
        td = balanced_assign(by_type, key=lambda m: (dev.get(m["matrix"], {}).get("S_i_dev_mean") or -1e9))
    else:
        td = guided
    # shuffled: permute the guided a-values within each type (preserves 4/4/4), fixed by seed
    rng = random.Random(args.seed)
    shuffled = {}
    for typ, mats in by_type.items():
        vals = [guided[m["matrix"]] for m in mats]; rng.shuffle(vals)
        for m, v in zip(mats, vals): shuffled[m["matrix"]] = v
    reversed_a = {k: (0.5 if v == 2.0 else 2.0 if v == 0.5 else 1.0) for k, v in guided.items()}

    out = {"seed": args.seed,
           "guided": guided, "euclidean": euclid, "typedepth": td,
           "shuffled": shuffled, "reversed": reversed_a,
           "global_a05": {m: 0.5 for m in feats}, "global_a1": {m: 1.0 for m in feats},
           "global_a2": {m: 2.0 for m in feats}}
    # histogram check
    for arm in ("guided", "euclidean", "typedepth", "shuffled", "reversed"):
        hist = defaultdict(int)
        for v in out[arm].values(): hist[v] += 1
        assert hist[0.5] == 24 and hist[1.0] == 24 and hist[2.0] == 24, (arm, dict(hist))
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"seed{args.seed}: wrote 8 allocations (+nomom is a separate optimizer). balanced 24/24/24 verified.")
    # show guided assignment summary by type/a
    for typ in TYPES:
        mats = [m for m in by_type[typ]]
        a05 = [m["matrix"].split(".")[1] for m in mats if guided[m["matrix"]] == 0.5]
        print(f"  guided {typ:10s}: a=0.5 blocks {sorted(int(b) for b in a05)}")


if __name__ == "__main__":
    main()
