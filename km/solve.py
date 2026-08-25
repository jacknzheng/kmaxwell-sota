#!/usr/bin/env python3
"""kmaxwell run solver + verdict parser (pure python, no torch).

Reuses the repo's OWN kmaxwell_kernel.py so mix weights and filter stats are
DERIVED from the spec, never hand-typed. A spec is (k, tmin, tmax, age, shape);
weights are solved deterministically. This is the anti-hallucination anchor:
the only free inputs are the 5 knobs + seed, and the box computes the rest.
"""
import sys, os, json, math, re, argparse

REPO = os.environ.get("KM_REPO", "/root/modded-nanogpt")
KERNEL_DIR = REPO + "/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210"
sys.path.insert(0, KERNEL_DIR)
from kmaxwell_kernel import build_kmaxwell_kernel, nesterov_filter_stats  # noqa: E402

SHAPES = {
    "linear": lambda i: float(i),
    "quad":   lambda i: float(i) ** 2,
    "cub":    lambda i: float(i) ** 3,
    "geo":    lambda i: 2.0 ** (i - 1),
    "sqrt":   lambda i: math.sqrt(i),
    "flat":   lambda i: 1.0,
    # reversed ramps: more mass on FAST buffers, leftover still on slowest
    # (bimodal fast+slow, echoes the original bi-Maxwell). k-dependent, bound at call.
    "rlinear": lambda i: None,  # placeholder; resolved in solve_weights with k
    "rquad":   lambda i: None,
}
_REVERSED = {"rlinear": 1, "rquad": 2}


def solve_weights(k, tmin, tmax, age, shape):
    """Ramp family: w_i = c*s_i on ticks 1..k-1, leftover on the slow tick,
    c chosen so mean age Sum(w*tau) == age. Raises if any weight <= 0."""
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; known: {sorted(SHAPES)}")
    lo, hi = math.log10(tmin), math.log10(tmax)
    t = [10.0 ** (lo + i * (hi - lo) / (k - 1)) for i in range(k)]
    if shape in _REVERSED:  # decreasing ramp: score (k-1-i)^p over ticks 1..k-1
        p = _REVERSED[shape]
        s = [float(k - 1 - i) ** p + 1e-6 for i in range(k - 1)]
    else:
        s = [SHAPES[shape](i + 1) for i in range(k - 1)]
    denom = t[-1] * sum(s) - sum(si * ti for si, ti in zip(s, t))
    c = (t[-1] - age) / denom
    w = [c * si for si in s] + [1.0 - c * sum(s)]
    if any(x <= 1e-9 for x in w):
        raise ValueError(f"infeasible: k={k} [{tmin},{tmax}] age={age} shape={shape} -> {w}")
    return w


def stats_for(k, tmin, tmax, weights):
    tau, betas, wn, mean_age = build_kmaxwell_kernel(k, tmin, tmax, 1.0, weights=weights)
    lag_m, lag_x, noise = nesterov_filter_stats(betas, wn)
    return {"tau": tau, "betas": betas, "weights": wn, "mean_age": mean_age,
            "lag_m": lag_m, "noise_gain": noise}


def resolve_spec(spec):
    """Fill weights + stats from a spec dict. Accepts explicit 'weights' (comma
    string or list), the native gaussian mode (shape='gaussian' + sigma; age
    emerges), or derives from (k,tmin,tmax,age,shape) via the ramp family."""
    k = int(spec["k"])
    tmin = float(spec["tmin"]); tmax = float(spec["tmax"])
    if spec.get("weights"):
        w = spec["weights"]
        w = [float(x) for x in w.split(",")] if isinstance(w, str) else list(w)
        shape = spec.get("shape", "explicit"); age = None
    elif spec.get("shape") == "gaussian":
        # native kernel scoring: bell curve on log(tau), centered at geo-mid.
        # age is NOT pinned here — it emerges from sigma + window.
        shape = "gaussian"; age = None
        _, _, w, _ = build_kmaxwell_kernel(k, tmin, tmax, float(spec["sigma"]), weights=None)
    else:
        shape = spec["shape"]; age = float(spec["age"])
        w = solve_weights(k, tmin, tmax, age, shape)
    st = stats_for(k, tmin, tmax, w)
    out = dict(spec)
    out.update({"k": k, "tmin": tmin, "tmax": tmax, "shape": shape,
                "weights": ",".join(f"{x:.6f}" for x in st["weights"]),
                "weights_list": st["weights"], "mean_age": round(st["mean_age"], 4),
                "noise_gain": round(st["noise_gain"], 6), "lag_m": round(st["lag_m"], 4)})
    if age is not None:
        out["age"] = age
    return out


STEP_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([\d.]+) train_time:([\d.]+)s.*step_avg:([\d.]+)ms")


def parse_log(path):
    """Extract the val curve + key milestones from a captured trainer stdout."""
    curve = {}
    total = None
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                step, tot, loss, t, savg = m.groups()
                total = int(tot)
                curve[int(step)] = {"val_loss": float(loss), "train_time": float(t),
                                    "step_avg_ms": float(savg)}
    v = {"total_steps": total, "curve": curve,
         "val_3150": curve.get(3150, {}).get("val_loss"),
         "val_3250": curve.get(3250, {}).get("val_loss"),
         "train_time_3150": curve.get(3150, {}).get("train_time"),
         "train_time_3250": curve.get(3250, {}).get("train_time"),
         "n_val_points": len(curve)}
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve"); r.add_argument("spec_json")
    p = sub.add_parser("parse"); p.add_argument("logfile")
    a = ap.parse_args()
    if a.cmd == "resolve":
        print(json.dumps(resolve_spec(json.loads(a.spec_json))))
    elif a.cmd == "parse":
        print(json.dumps(parse_log(a.logfile)))
