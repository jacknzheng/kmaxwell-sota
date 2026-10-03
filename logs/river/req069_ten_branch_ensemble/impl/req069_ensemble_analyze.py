"""REQ-069 ensemble analysis driver (runs ON the box; memory-bounded streaming). For each stream
(gradient mean-per-token, displacement delta), over 10 branch capture dirs forked from theta_1500:
 - build the ensemble-MEAN whole-model sequence (stream per step: load 10 branches, average) -> 64 mean dicts
 - ensemble-mean temporal Gram (req068_gram) + lag cosine profile
 - per-branch lag cosine (one branch loaded at a time) -> mean individual-branch lag -> alternation reduction
 - two-group (0-4 vs 5-9) agreement cosine per step; convergence (1/2/5/10) + dispersion + mean/rms at mid step
 - per-family energy share of the ensemble mean
Never holds more than the 64 ensemble-mean vectors (fp32) + 10 branch vectors at one step.
Writes a compact JSON (no raw tensors).

Usage: req069_ensemble_analyze.py --branch_root <dur> --stream grad --window 1500 --tokens 524288 --out <d>.json
       branch dirs: <branch_root>/br{0..9}/{grads,disp}
"""
import argparse, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "req068_full_gradient_history", "impl"))
from req068_capture import GradientHistoryReader
from req068_gram import temporal_gram, cosine_from_gram, gram_eig, lag_agreement
from req069_ensemble import lag_cosine_sequence, _flat

NBR = 10
FAMILIES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj", "embed", "head", "other"]
def family_of(n):
    if n.startswith("embed"): return "embed"
    if n.startswith("proj"): return "head"
    for t in FAMILIES[:6]:
        if t in n: return t
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch_root", required=True); ap.add_argument("--stream", choices=["grad", "disp"], required=True)
    ap.add_argument("--window", type=int, default=1500); ap.add_argument("--tokens", type=int, default=524288)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sub = "grads" if a.stream == "grad" else "disp"
    scale = float(a.tokens) if a.stream == "grad" else 1.0
    readers = [GradientHistoryReader(os.path.join(a.branch_root, f"br{b}", sub)) for b in range(NBR)]
    names = [n for n in readers[0].params()]
    steps = list(range(a.window, a.window + 64))
    T = len(steps)

    def branch_step(b, s):  # dict {name: fp64 array}, mean-per-token for grads
        fs = readers[b].full_step(s)
        return {n: (fs[n].float().cpu().numpy().astype(np.float64).ravel() / scale) for n in names if fs[n] is not None}

    # pass 1: ensemble-mean sequence (stream per step) + per-step group agreement + mid-step convergence
    mean_seq = []            # list of {name: fp64}
    group_agree = []
    conv = None
    gA, gB = list(range(5)), list(range(5, 10))
    for ti, s in enumerate(steps):
        bvecs = [branch_step(b, s) for b in range(NBR)]          # 10 dicts at this step
        m = {n: np.mean([bvecs[b][n] for b in range(NBR)], axis=0) for n in names}
        mean_seq.append(m)
        fa = _flat({n: np.mean([bvecs[b][n] for b in gA], axis=0) for n in names}, names)
        fb = _flat({n: np.mean([bvecs[b][n] for b in gB], axis=0) for n in names}, names)
        na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
        group_agree.append(float(fa @ fb / (na * nb)) if na > 0 and nb > 0 else float("nan"))
        if ti == T // 2:
            flats = [_flat(bvecs[b], names) for b in range(NBR)]
            allm = np.mean(flats, axis=0); dn = np.linalg.norm(allm)
            rms = float(np.sqrt(np.mean([f @ f for f in flats])))
            conv = {str(n): float(np.linalg.norm(np.mean(flats[:n], axis=0))) for n in (1, 2, 5, 10)}
            conv["dispersion"] = float(np.mean([np.linalg.norm(f - allm) for f in flats]) / dn) if dn > 0 else None
            conv["mean_to_rms"] = float(dn / rms) if rms > 0 else None
        del bvecs

    # ensemble-mean Gram + lag + per-family
    K, famK = temporal_gram(mean_seq, names, family_of)
    C, _ = cosine_from_gram(K); lam, _ = gram_eig(K)
    mean_flats = [_flat(m, names) for m in mean_seq]
    mean_lag = lag_cosine_sequence(mean_flats)
    total = float(np.trace(K)); fam_share = {f: float(np.trace(mm)) / total for f, mm in famK.items()} if total > 0 else {}

    # pass 2: per-branch lag cosine (one branch at a time)
    indiv_lag = {L: [] for L in (1, 2, 4, 8)}
    for b in range(NBR):
        bf = [_flat(branch_step(b, s), names) for s in steps]
        for L in indiv_lag:
            v = lag_cosine_sequence(bf, (L,)).get(L)
            if v is not None: indiv_lag[L].append(v)
    indiv_lag_mean = {L: (float(np.mean(v)) if v else None) for L, v in indiv_lag.items()}

    res = {
        "stream": a.stream, "window": a.window, "n_branches": NBR, "n_steps": T,
        "ensemble_mean_lag_cosine": mean_lag,
        "mean_individual_branch_lag_cosine": indiv_lag_mean,
        "alternation_reduction_lag1": {"individual": indiv_lag_mean.get(1), "ensemble": mean_lag.get(1)},
        "group_agreement_per_step_mean": float(np.nanmean(group_agree)),
        "group_agreement_first_last": [group_agree[0], group_agree[-1]],
        "convergence_mid_step": conv,
        "ensemble_mean_eig_energy_frac": (lam / lam.sum()).tolist()[:8] if lam.sum() > 0 else [],
        "ensemble_mean_family_share": fam_share,
        "C_ensemble_mean": C.tolist(),
    }
    json.dump(res, open(a.out, "w"))
    print(f"{a.stream} win{a.window}: indiv lag1={indiv_lag_mean.get(1):+.3f} ensemble lag1={mean_lag.get(1):+.3f} "
          f"group_agree={res['group_agreement_per_step_mean']:+.3f} disp={conv['dispersion']:.3f}", flush=True)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
