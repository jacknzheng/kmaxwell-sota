"""REQ-060 aggregation: is the memory->curvature change loss-cubic feedback, Muon's polar-map nonlinearity, or both?
Reads raw/diag/{arm}_s{seed}_step{2032,2064}.json. Reports, by arm (a05/a1/a2) and offset (32/64):
 - e_loss scale ratio (0.5/1); ~0.25 => cubic (loss third-derivative); >0.25 => higher-order contamination (flag)
 - map_frac = ||E_map||/||E_total||, residual_frac = ||E_residual||/||E_total|| (polar-map vs loss-cubic share)
 - whether the memory intervention (a) changes e_loss / residual_frac (does shorter/longer memory move the cubic feedback?)
"""
import json, os, sys, glob, statistics
from collections import defaultdict
RAW = sys.argv[1] if len(sys.argv) > 1 else "raw/diag"
ARMS = ["a05", "a1", "a2"]; SEEDS = [0, 1, 2]; OFFS = [2032, 2064]
def load(arm, s, step):
    p = os.path.join(RAW, f"{arm}_s{s}_step{step}.json"); return json.load(open(p)) if os.path.exists(p) else None
print("=== per (arm, offset): e_loss scale ratio, map_frac, residual_frac (mean over seeds) ===")
print(f"{'arm':>4} {'off':>5} {'eloss_0.5/1':>12} {'map_frac':>9} {'resid_frac':>10} {'eloss_norm':>11}")
agg = {}
for arm in ARMS:
    for step in OFFS:
        rows = [load(arm, s, step) for s in SEEDS]; rows = [r for r in rows if r]
        if not rows: continue
        r05 = statistics.mean(r["eloss_scale_ratios"]["0.5/1"] for r in rows)
        mf = statistics.mean(r["map_frac"] for r in rows if r.get("map_frac"))
        rf = statistics.mean(r["residual_frac"] for r in rows if r.get("residual_frac"))
        en = statistics.mean(r["eloss_scale_norms"]["1.0"] for r in rows)
        agg[(arm, step)] = (r05, mf, rf, en)
        print(f"{arm:>4} {step:>5} {r05:>12.3f} {mf:>9.3f} {rf:>10.3f} {en:>11.3e}")
print("\n=== does memory (a) change the loss-cubic feedback? (offset 64, mean over seeds) ===")
for step in OFFS:
    if all((arm, step) in agg for arm in ARMS):
        print(f"  offset {step}: e_loss_norm a05={agg[('a05',step)][3]:.3e} a1={agg[('a1',step)][3]:.3e} a2={agg[('a2',step)][3]:.3e}; "
              f"resid_frac a05={agg[('a05',step)][2]:.2f} a1={agg[('a1',step)][2]:.2f} a2={agg[('a2',step)][2]:.2f}")
print("\nInterpretation: scale ratio ~0.25 => loss-cubic (third-derivative) origin; both map_frac and residual_frac")
print("substantial => BOTH nonlinearities contribute. Monotone e_loss_norm in a (a05<a1<a2 or reverse) => memory")
print("modulates the cubic feedback. Large scale-ratio deviation from 0.25 => higher-order contamination -> INCONCLUSIVE on pure-cubic.")
