"""REQ-055 — geometry of the realized MUON update: K-Maxwell (kmax) vs age-matched single EMA (agema), seed 0.
Consumes raw_json/geom_{kmax,agema}_s0_step{2050,2250,2500,2749}.json (measure_step_geometry.py).
Analysis direction = delta_muon (realized displacement minus decoupled weight decay). Reports, per step and
as the across-step kmax-agema contrast:
  align_joint  -g.delta_muon/(||g|| ||delta_muon||)          (higher = more downhill)
  vHv          delta_muon^T H delta_muon / ||delta_muon||^2  by gradient-FD HVP and by loss-scan 2nd diff (cross-check)
  uphill_frac  frac of 72 Muon matrices whose own step is uphill
  decomposition |delta_full| = |delta_wd + delta_muon|, |delta_wd|, |delta_nonmuon|, reconstruction residual
  loss scan vs quadratic prediction; full-realized-displacement dL at a=1."""
import json, os
RAW=os.path.join(os.path.dirname(__file__),"raw_json"); STEPS=[2050,2250,2500,2749]
def load(arm,S):
    p=os.path.join(RAW,f"geom_{arm}_s0_step{S}.json"); return json.load(open(p)) if os.path.exists(p) else None
def sc(d,a): return d["loss_scan_muon"].get(f"{a}",d["loss_scan_muon"].get(f"{float(a)}"))
rows=[(S,load("kmax",S),load("agema",S)) for S in STEPS]
rows=[(S,k,a) for S,k,a in rows if k and a]
print(f"{'step':>5} | {'align_joint (k/a/d)':>24} | {'vHv_fd (k/a/d)':>26} | {'uphill (k/a/d)':>16} | {'|d_muon| |d_wd| |d_non| (kmax)':>32}")
agg={m:[] for m in ("align","vHv","uphill")}
for S,k,a in rows:
    dal=k["align_joint"]-a["align_joint"]; dvhv=k["vHv_fd"]-a["vHv_fd"]; duf=k["uphill_matrix_frac"]-a["uphill_matrix_frac"]
    agg["align"].append(dal); agg["vHv"].append(dvhv); agg["uphill"].append(duf)
    print(f"{S:>5} | {k['align_joint']:+.4f} {a['align_joint']:+.4f} {dal:+.4f} | "
          f"{k['vHv_fd']:+.2e} {a['vHv_fd']:+.2e} {dvhv:+.1e} | "
          f"{k['uphill_matrix_frac']:.2f} {a['uphill_matrix_frac']:.2f} {duf:+.2f} | "
          f"{k['delta_muon_norm']:.3f} {k['delta_wd_norm']:.3f} {k['delta_nonmuon_norm']:.3f}")
mean=lambda x:sum(x)/len(x) if x else float("nan")
print("\n=== across-step mean of (kmax - agema) ===")
print(f"  align_joint : {mean(agg['align']):+.4f}   (>0 => K-Maxwell steps MORE downhill)")
print(f"  vHv_fd      : {mean(agg['vHv']):+.3e}   (<0 => K-Maxwell hits LESS curvature)")
print(f"  uphill_frac : {mean(agg['uphill']):+.4f}   (<0 => fewer uphill matrices under K-Maxwell)")
print("\n=== curvature cross-check: gradient-FD HVP vs loss-scan 2nd diff (should agree) ===")
for S,k,a in rows:
    for arm,d in (("kmax",k),("agema",a)):
        print(f"  step {S} {arm:5s}: vHv_fd={d['vHv_fd']:+.3e}  vHv_scan={d['vHv_scan']:+.3e}  "
              f"reldiff={abs(d['vHv_fd']-d['vHv_scan'])/(abs(d['vHv_fd'])+1e-9):.2%}")
print("\n=== displacement decomposition (reconstruction residual must be ~0) ===")
for S,k,a in rows:
    for arm,d in (("kmax",k),("agema",a)):
        print(f"  step {S} {arm:5s}: |d_full|={d['delta_full_norm']:.4f} |d_wd|={d['delta_wd_norm']:.4f} "
              f"|d_muon|={d['delta_muon_norm']:.4f} |d_nonmuon|={d['delta_nonmuon_norm']:.4f} recon_resid={d['recon_residual']:.2e}  "
              f"wd_frac={d['delta_wd_norm']/d['delta_full_norm']:.1%}")
print("\n=== loss scan along delta_muon: observed vs quadratic prediction a*(g.d)+0.5a^2(d'Hd) ===")
al=[-0.5,0.25,0.5,1.0,1.5,2.0]
for S,k,a in rows:
    for arm,d in (("kmax",k),("agema",a)):
        obs="  ".join(f"a{x:+.2f}:{sc(d,x):+.1f}" for x in al)
        pr ="  ".join(f"{d['quad_pred'][f'{x}']:+.1f}" for x in al)
        print(f"  step {S} {arm:5s} obs : {obs}")
        print(f"  {'':>12}   pred: {pr}")
print("\n=== full realized displacement (Muon delta_full + non-Muon) dL at a=1 ===")
for S,k,a in rows:
    print(f"  step {S}: kmax full@1={k['dL_full_realized_a1']:+.3f} (muon-only@1={sc(k,1.0):+.3f})   "
          f"agema full@1={a['dL_full_realized_a1']:+.3f} (muon-only@1={sc(a,1.0):+.3f})")
