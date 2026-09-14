"""REQ-057 pilot gate analysis (pure Python, reads the pilot JSON; no torch needed).

Gates (both must pass to expand to the 72-matrix stage):
  (a) >= 90% of positively-resolved sentinel S_i change by <= 5% from K=20 to K=50 (after sphere normalization)
  (b) median pairwise Spearman rank correlation of S_i/G_i across the disjoint probe subsets >= 0.8
Also reports: joint vs diagonal curvature (c_cross/c at the gradient-polar direction), HVP central-difference
plateau + 5% agreement, and whether spectral normalization changes the within-type depth ordering vs the
Euclidean Lanczos comparator. Usage: python analyze_pilot.py <pilot_step2000.json>
"""
import json, sys, statistics


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def main():
    d = json.load(open(sys.argv[1]))
    subsets = [k for k in d if k.startswith("subset") and k != "provenance"]
    subsets.sort()
    names = list(d[subsets[0]]["isolated"].keys())

    print("=== Gate (a): sentinel S_i stability K=20 -> K=50 (subset 0) ===")
    iso0 = d["subset0"]["isolated"]
    stable, resolved = 0, 0
    rows = []
    for n in names:
        pk = iso0[n]["S_i_per_k"]
        s20, s50 = pk.get("20"), pk.get("50")
        if s20 is None or s50 is None or s20 <= 0:
            rows.append((n, s20, s50, "unresolved"))
            continue
        resolved += 1
        rel = abs(s50 - s20) / s20
        ok = rel <= 0.05
        stable += ok
        rows.append((n, s20, s50, f"{rel:.1%}{'' if ok else '  >5%'}"))
    for n, s20, s50, tag in rows:
        print(f"  {n:28s} S20={s20 if s20 is None else round(s20,4)}  S50={s50 if s50 is None else round(s50,4)}  {tag}")
    frac = stable / resolved if resolved else 0.0
    gate_a = frac >= 0.90
    print(f"  -> {stable}/{resolved} positively-resolved stable <=5%  ({frac:.0%})  gate_a={'PASS' if gate_a else 'FAIL'} (need >=90%)")

    print("\n=== Gate (b): median pairwise Spearman of S_i/G_i across subsets ===")
    ratio_vecs = {}
    for sk in subsets:
        iso = d[sk]["isolated"]
        vec = []
        for n in names:
            Si, Gi = iso[n]["S_i"], iso[n]["G_i"]
            vec.append((Si / Gi) if (Si is not None and Gi and Gi > 0) else float("nan"))
        ratio_vecs[sk] = vec
    # drop matrices unresolved in ANY subset for a clean rank comparison
    keep = [i for i in range(len(names)) if all(v[i] == v[i] for v in ratio_vecs.values())]
    corrs = []
    import itertools
    for a, b in itertools.combinations(subsets, 2):
        va = [ratio_vecs[a][i] for i in keep]
        vb = [ratio_vecs[b][i] for i in keep]
        c = spearman(va, vb)
        corrs.append(c)
        print(f"  {a} vs {b}: Spearman(S_i/G_i) = {c:.3f}  (n={len(keep)})")
    med = statistics.median(corrs) if corrs else float("nan")
    gate_b = med >= 0.8
    print(f"  -> median = {med:.3f}  gate_b={'PASS' if gate_b else 'FAIL'} (need >=0.8)")

    print("\n=== Joint vs diagonal curvature (subset 0, gradient-polar direction) ===")
    cross = d["subset0"].get("cross_layer_at_gradpolar")
    if cross:
        c, cd, cc = cross["c"], cross["c_diag"], cross["c_cross"]
        print(f"  c={c:.4e}  c_diag={cd:.4e}  c_cross={cc:.4e}  cross/|c|={cc/abs(c):.1%} (signed cancellation possible)")

    print("\n=== HVP validation (subset 0): central grad-difference vs autograd ===")
    hv = d["subset0"].get("hvp_validation")
    if hv:
        print(f"  autograd dHd = {hv['autograd_dHd']:.4e}")
        rels = []
        for eps, r in sorted(hv["by_eps"].items(), key=lambda kv: float(kv[0])):
            print(f"    eps={eps}: central={r['central_diff_dHd']:.4e}  rel_err={r['rel_err_vs_autograd']:.2%}")
            rels.append(r["rel_err_vs_autograd"])
        plateau = sum(1 for r in rels if r <= 0.05) >= 2
        print(f"  -> {'PLATEAU+5% OK' if plateau else 'NO PLATEAU / >5% (mark unresolved)'}")

    print("\n=== Spectral vs Euclidean within-type depth ordering (subset 0) ===")
    # group sentinel S_i by matrix TYPE across blocks 0/6/11; report S_i ordering
    from collections import defaultdict
    by_type = defaultdict(list)
    for n in names:
        typ = ".".join(n.split(".")[2:])  # e.g. attn.q.weight
        blk = int(n.split(".")[1])
        by_type[typ].append((blk, iso0[n]["S_i"]))
    for typ, lst in sorted(by_type.items()):
        lst.sort()
        print(f"  {typ:16s} S_i by block {[ (b, None if s is None else round(s,4)) for b,s in lst ]}")

    print(f"\nSUMMARY: gate_a={'PASS' if gate_a else 'FAIL'}  gate_b={'PASS' if gate_b else 'FAIL'}  "
          f"-> {'EXPAND to 72-matrix stage' if (gate_a and gate_b) else 'DO NOT EXPAND; report reliability limits'}")


if __name__ == "__main__":
    main()
