"""REQ-064 analysis: the PRE-REGISTERED H1 test. Fit each feature model on the 18 dev sentinels
(seeds 0,1 pooled) to the primary per-sentinel response, then predict on prospective test seeds 7,8
INDIVIDUALLY with a per-seed 10% RMSE gate + sign-recovery rate.

Per-sentinel primary response (step-256, vs the all-reference a_star global):
    r_i = loss(sel_i_long @2256) - loss(sel_i_short @2256)   (>0 => shorter memory helps matrix i)
Feature models: M1 = type/depth prior (dev-mean S_i per (type,block)); M2 = S_i/G_i sharpness (primary);
Euclidean = lambda_i/||g_i||_F^2. Each is a 1-D least-squares fit r ~ a*feature + b on dev, applied to test.

H1 (pre-registered): M2 has LOWER per-seed RMSE than BOTH M1 and Euclidean on BOTH test seeds, AND recovers
the sign of r_i on >= 13/18 sentinels per test seed. Correlation sign alone is NOT sufficient.

Usage:  analyze_req064.py --feat_dir DIR --loss_dir DIR   (or --selftest)
Inputs: feat_dir/features_req064_state_s{seed}_step002000.json ; loss_dir/{arm}_s{seed}.tsv ({step val_loss}).
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict

DEV_SEEDS = [0, 1]
TEST_SEEDS = [7, 8]
TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]
SENT_BLOCKS = [0, 6, 11]
SENTINELS = [f"blocks.{b}.{t}.weight" for b in SENT_BLOCKS for t in TYPES]
ENDPOINT = 2256


def _lstsq_1d(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    a = sxy/sxx if sxx > 0 else 0.0; b = my - a*mx
    return a, b


def _rmse(pred, actual):
    n = len(pred); return math.sqrt(sum((pred[i]-actual[i])**2 for i in range(n))/n)


def fit_and_test(dev_feat, dev_resp, test_by_seed):
    """dev_feat/dev_resp: pooled dev lists. test_by_seed: seed -> (feat_list, resp_list). Returns per-seed
    rmse, relative rmse (vs response std), and sign accuracy."""
    a, b = _lstsq_1d(dev_feat, dev_resp)
    out = {}
    for seed, (xf, yr) in test_by_seed.items():
        pred = [a*x + b for x in xf]
        rmse = _rmse(pred, yr)
        rng = (max(yr)-min(yr)) or 1e-12
        sign_ok = sum(1 for i in range(len(yr)) if (pred[i] > 0) == (yr[i] > 0))
        out[seed] = dict(rmse=rmse, rel_rmse=rmse/rng, sign_ok=sign_ok, n=len(yr), a=a, b=b)
    return out


def load_feature(feat_dir, seed, key):
    p = os.path.join(feat_dir, f"features_req064_state_s{seed}_step{2000:06d}.json")
    d = json.load(open(p))
    # average the requested feature across measurement subsets
    subs = [k for k in d if k.startswith("subset")]
    vals = {}
    for name in SENTINELS:
        vv = [d[s]["per_matrix"][name][key] for s in subs
              if name in d[s]["per_matrix"] and d[s]["per_matrix"][name].get(key) is not None]
        vals[name] = (sum(vv)/len(vv)) if vv else None
    return vals


def load_response(loss_dir, seed):
    def ep(arm):
        p = os.path.join(loss_dir, f"{arm}_s{seed}.tsv")
        if not os.path.exists(p): return None
        d = {int(x.split()[0]): float(x.split()[1]) for x in open(p) if x.strip()}
        return d.get(ENDPOINT)
    resp = {}
    for name in SENTINELS:
        b = name.split(".")[1]; t = ".".join(name.split(".")[2:4]); tag = f"b{b}_{t.replace('.', '')}"
        ls, ll = ep(f"sel_{tag}_short"), ep(f"sel_{tag}_long")
        resp[name] = (ll - ls) if (ls is not None and ll is not None) else None
    return resp


def run(feat_dir, loss_dir):
    typedepth = defaultdict(list)  # (type,block)->S_i over dev seeds -> M1 prior
    for seed in DEV_SEEDS:
        si = load_feature(feat_dir, seed, "S_i")
        for name, v in si.items():
            if v is not None:
                b = name.split(".")[1]; t = ".".join(name.split(".")[2:4]); typedepth[(t, b)].append(v)
    prior = {k: sum(v)/len(v) for k, v in typedepth.items()}

    def feat_vec(seed, model):
        if model == "M1":
            return {n: prior.get((".".join(n.split(".")[2:4]), n.split(".")[1])) for n in SENTINELS}
        if model == "M2":
            return load_feature(feat_dir, seed, "S_i_over_G_i")
        if model == "EUC":
            return load_feature(feat_dir, seed, "lambda_over_gF2")
        raise ValueError(model)

    resp = {s: load_response(loss_dir, s) for s in DEV_SEEDS + TEST_SEEDS}
    report = {}
    for model in ("M1", "M2", "EUC"):
        dev_f, dev_r = [], []
        for s in DEV_SEEDS:
            fv = feat_vec(s, model)
            for n in SENTINELS:
                if fv.get(n) is not None and resp[s].get(n) is not None:
                    dev_f.append(fv[n]); dev_r.append(resp[s][n])
        test_by = {}
        for s in TEST_SEEDS:
            fv = feat_vec(s, model)
            xf, yr = [], []
            for n in SENTINELS:
                if fv.get(n) is not None and resp[s].get(n) is not None:
                    xf.append(fv[n]); yr.append(resp[s][n])
            test_by[s] = (xf, yr)
        report[model] = fit_and_test(dev_f, dev_r, test_by)
    return report


def verdict(report):
    ok = True; lines = []
    for s in TEST_SEEDS:
        m2 = report["M2"][s]; m1 = report["M1"][s]; eu = report["EUC"][s]
        beats = m2["rmse"] < m1["rmse"] and m2["rmse"] < eu["rmse"]
        sign_ok = m2["sign_ok"] >= 13
        lines.append(f"  seed{s}: M2 rmse={m2['rmse']:.4f} (M1 {m1['rmse']:.4f}, EUC {eu['rmse']:.4f}) "
                     f"beats={beats}; M2 sign {m2['sign_ok']}/{m2['n']} (>=13: {sign_ok})")
        ok &= beats and sign_ok
    return ok, lines


# ---------- synthetic self-test of the H1 logic ----------
def selftest():
    import random; random.seed(0)
    # ground truth: response driven by M2 feature + small noise; M1/EUC are noisier proxies
    feat = {}
    resp = {}
    for s in DEV_SEEDS + TEST_SEEDS:
        f = {}; r = {}
        for n in SENTINELS:
            true = random.uniform(-1, 1)
            f[n] = {"S_i_over_G_i": true, "lambda_over_gF2": true*0.3 + random.gauss(0, 0.6),
                    "S_i": true + random.gauss(0, 0.05)}
            r[n] = 2.0*true + random.gauss(0, 0.05)
        feat[s] = f; resp[s] = r
    # inline fit/test using the same helpers
    def fv(s, model):
        if model == "M2": return {n: feat[s][n]["S_i_over_G_i"] for n in SENTINELS}
        if model == "EUC": return {n: feat[s][n]["lambda_over_gF2"] for n in SENTINELS}
        return {n: feat[s][n]["S_i"] for n in SENTINELS}  # M1 proxy
    rep = {}
    for model in ("M1", "M2", "EUC"):
        dev_f = [fv(s, model)[n] for s in DEV_SEEDS for n in SENTINELS]
        dev_r = [resp[s][n] for s in DEV_SEEDS for n in SENTINELS]
        test_by = {s: ([fv(s, model)[n] for n in SENTINELS], [resp[s][n] for n in SENTINELS]) for s in TEST_SEEDS}
        rep[model] = fit_and_test(dev_f, dev_r, test_by)
    ok, lines = verdict(rep)
    print("SELFTEST (M2 is the true driver -> H1 should PASS):")
    for l in lines: print(l)
    print(f"  => {'PASS' if ok else 'FAIL'} (expected PASS)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir"); ap.add_argument("--loss_dir"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    report = run(a.feat_dir, a.loss_dir)
    print(json.dumps(report, indent=1))
    ok, lines = verdict(report)
    print("\nH1 (pre-registered): M2 beats M1+Euclidean on per-seed RMSE AND sign>=13/18, both test seeds")
    for l in lines: print(l)
    print(f"=> H1 {'PASS -> REQ-065 may proceed' if ok else 'FAIL -> negative/again-inconclusive; no policy trial'}")


if __name__ == "__main__":
    main()
