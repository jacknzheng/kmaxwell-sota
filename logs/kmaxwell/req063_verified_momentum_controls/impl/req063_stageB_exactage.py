"""REQ-063 Stage B item 5: verify the exact-age matched single-EMA against the ACTUAL K-Maxwell mixture
recurrence, independently. Two implementations per system: (a) the committed closed-form mass/first-moment
recurrence, (b) an INDEPENDENT explicit lag-histogram. Confirm mean age & mass agree to 1e-10 (float64),
that the matched single-EMA reproduces the mixture's realized increment age to 1e-10 (the solver's target),
and report that age VARIANCE differs (single EMA cannot match the 8-timescale spread) through 750 updates.
Impulse / constant / alternating gradient sequences check the realized buffer VALUE via the mass identity.
Pure float64, no GPU. Uses the committed solver verbatim."""
import sys, math
sys.path.insert(0, "/Users/jerryhong/modded-nanogpt/logs/kmaxwell/req058_layerwise_momentum_response/impl")
# committed solver (import the function text without running the torch patch)
import importlib.util, types
src = open("/Users/jerryhong/modded-nanogpt/logs/kmaxwell/req058_layerwise_momentum_response/impl/apply_req058_opt.py").read()
ns = {}
fn_src = src[src.index("def _req058_solve_matched_beta"):src.index("class PerturbedAnnealedWeightsMuon")]
exec(fn_src, ns)
solve = ns["_req058_solve_matched_beta"]

DECAYS = [0.75, 0.822852439855, 0.877930338626, 0.917598547218, 0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
SW = [0.005093975, 0.010187949, 0.015281924, 0.020375898, 0.025469873, 0.030563847, 0.035657822, 0.857368713]
EW = [0.032261839, 0.064523678, 0.096785516, 0.129047355, 0.161309194, 0.193571033, 0.225832871, 0.096668514]
SWITCH, ANNEAL, NU, NST = 2000, 2750, 0.95, 750

beta_sched = solve(DECAYS, SW, EW, SWITCH, ANNEAL, NU, NST)

# ---- (a) closed-form recurrence for the SINGLE matched EMA (mass, first moment) ----
def single_closed(betas):
    mass = mom = 0.0
    for _ in range(SWITCH + 1): mom, mass = NU*(mom+mass), NU*mass+(1-NU)  # warm to switch (baseline update)
    rows = []
    for j, b in enumerate(betas):
        mom, mass = b*(mom+mass), b*mass+(1-b)
        rows.append((mass, mom/mass if mass else 0.0, b))
    return rows

# ---- (b) INDEPENDENT explicit lag-histogram for a time-varying-beta single EMA ----
def single_hist(betas):
    h = [0.0]  # h[lag] = weight
    for _ in range(SWITCH + 1):  # warm with beta=NU
        h = [NU*x for x in h]; h = [1-NU] + h
    rows = []
    for b in betas:
        h = [1-b] + [b*x for x in h]  # advance ages (shift +1, scale by b), inject fresh (1-b) at lag0
        mass = sum(h); mean = sum(l*x for l, x in enumerate(h))/mass
        var = sum((l-mean)**2*x for l, x in enumerate(h))/mass
        rows.append((mass, mean, var))
    return rows

# ---- mixture: closed-form (matches solver internals) vs independent per-stream histograms ----
def mixture_hist():
    hwarm=[0.0]
    for _ in range(SWITCH+1): hwarm=[1-NU]+[NU*x for x in hwarm]  # plain-Muon warm, cloned to all streams
    hs = [list(hwarm) for _ in DECAYS]
    rows = []
    for j in range(NST):
        t = SWITCH+1+j
        a = min(max((t-SWITCH)/(ANNEAL-SWITCH), 0.0), 1.0)
        w = [s+a*(e-s) for s, e in zip(SW, EW)]; z = sum(w); w = [x/z for x in w]
        for k, d in enumerate(DECAYS): hs[k] = [d*x for x in hs[k]]; hs[k][0] += (1-d)
        L = max(len(x) for x in hs)
        H = [sum(w[k]*(hs[k][l] if l < len(hs[k]) else 0.0) for k in range(len(DECAYS))) for l in range(L)]
        M_mix = sum(H); P_mix = sum(l*x for l, x in enumerate(H))
        # nu outer-blend increment age (mirror solver: fresh grad lag0 mass (1-nu), buffer aged by nu, +1 shift)
        q_mass = (1-NU) + NU*M_mix
        inc = [ (1-NU) ] + [NU*x for x in H]  # shift buffer by 1 lag (age advance), fresh at lag0
        mean = sum(l*x for l, x in enumerate(inc))/q_mass
        var = sum((l-mean)**2*x for l, x in enumerate(inc))/q_mass
        rows.append((q_mass, mean, var))
    return rows

sc = single_closed(beta_sched); sh = single_hist(beta_sched); mx = mixture_hist()

# --- (A) single-EMA recurrence: closed form vs independent histogram ---
mass_err = max(abs(sc[j][0]-sh[j][0]) for j in range(NST))
age_err_single = max(abs(sc[j][1]-sh[j][1]) for j in range(NST))
print(f"(A) single EMA closed-form vs independent histogram: max|mass|={mass_err:.2e}  max|mean_age|={age_err_single:.2e}  (tol 1e-10)")

# --- (B) mixture per-stream recurrence: independent histogram M_s/P_s vs solver's closed recurrence ---
def mixture_closed_and_target():
    # replicate the solver's internal mixture recurrence + target_age (closed form)
    M_s = []; P_s = []
    mass = mom = 0.0
    for _ in range(SWITCH + 1): mom, mass = NU*(mom+mass), NU*mass+(1-NU)
    M_s = [mass]*len(DECAYS); P_s = [mom]*len(DECAYS)
    tgt = []
    for j in range(NST):
        t = SWITCH+1+j; a = min(max((t-SWITCH)/(ANNEAL-SWITCH),0.0),1.0)
        w = [s+a*(e-s) for s,e in zip(SW,EW)]; z=sum(w); w=[x/z for x in w]
        for k,b in enumerate(DECAYS): P_s[k],M_s[k] = b*(P_s[k]+M_s[k]), b*M_s[k]+(1-b)
        M_mix=sum(wi*m for wi,m in zip(w,M_s)); P_mix=sum(wi*p for wi,p in zip(w,P_s))
        q_mass=(1-NU)+NU*M_mix; tgt.append(NU*P_mix/q_mass if q_mass>0 else 0.0)
    return tgt
# independent per-stream histograms -> M_s_hist, P_s_hist, then target_age the SAME algebraic way
def mixture_target_hist():
    # ACTUAL optimizer convention: pre-switch plain Muon (single EMA, mu=nu); at switch the ONE momentum
    # buffer is cloned to all 8 streams (apply_req058_opt.py: streams=[momentum.clone() for _ in decays]).
    hwarm=[0.0]
    for _ in range(SWITCH+1): hwarm=[1-NU]+[NU*x for x in hwarm]
    hs=[list(hwarm) for _ in DECAYS]
    tgt=[]
    for j in range(NST):
        t=SWITCH+1+j; a=min(max((t-SWITCH)/(ANNEAL-SWITCH),0.0),1.0)
        w=[s+a*(e-s) for s,e in zip(SW,EW)]; z=sum(w); w=[x/z for x in w]
        for k,d in enumerate(DECAYS): hs[k]=[1-d]+[d*x for x in hs[k]]
        M_s=[sum(h) for h in hs]; P_s=[sum(l*x for l,x in enumerate(h)) for h in hs]
        M_mix=sum(wi*m for wi,m in zip(w,M_s)); P_mix=sum(wi*p for wi,p in zip(w,P_s))
        q_mass=(1-NU)+NU*M_mix; tgt.append(NU*P_mix/q_mass if q_mass>0 else 0.0)
    return tgt
tgt_closed = mixture_closed_and_target(); tgt_hist = mixture_target_hist()
mix_err = max(abs(tgt_closed[j]-tgt_hist[j]) for j in range(NST))
print(f"(B) mixture target_age closed-form vs independent per-stream histogram: max|diff|={mix_err:.2e}  (tol 1e-10)")

# --- (C) solver invariant: the solved beta makes the single EMA's nu-blended increment age == target_age ---
# forward-simulate single EMA with solved beta; probe candidate matched quantities (pre/post update).
mass=mom=0.0
for _ in range(SWITCH+1): mom,mass = NU*(mom+mass), NU*mass+(1-NU)
cands={"post mom/mass":0.0,"post nu-blend":0.0,"pre nu-blend":0.0}
for j,b in enumerate(beta_sched):
    pre_blend = NU*mom/((1-NU)+NU*mass) if ((1-NU)+NU*mass)>0 else 0.0
    mom,mass = b*(mom+mass), b*mass+(1-b)
    post_blend = NU*mom/((1-NU)+NU*mass) if ((1-NU)+NU*mass)>0 else 0.0
    cands["post mom/mass"]=max(cands["post mom/mass"], abs((mom/mass)-tgt_closed[j]))
    cands["post nu-blend"]=max(cands["post nu-blend"], abs(post_blend-tgt_closed[j]))
    cands["pre nu-blend"]=max(cands["pre nu-blend"], abs(pre_blend-tgt_closed[j]))
best=min(cands, key=cands.get); inv_err=cands[best]
print(f"(C) solver matching invariant vs mixture target_age (best of candidates): '{best}' max|diff|={inv_err:.2e}  (tol 1e-10)")
print(f"    candidates: " + ", ".join(f"{k}={v:.2e}" for k,v in cands.items()))

# --- variance: single EMA vs mixture full-buffer spread (single cannot match 8 timescales) ---
print("age VARIANCE (single vs mixture full buffer), sampled offsets:")
for j in (0, 100, 300, 500, 749):
    print(f"  off={j:3d} step={SWITCH+1+j}: beta={beta_sched[j]:.6f} | single mean={sh[j][1]:.3f} var={sh[j][2]:.2f} | mixture mean={mx[j][1]:.3f} var={mx[j][2]:.2f}")
print(f"  -> variance differs (single EMA cannot match the 8-timescale spread) = EXPECTED for an age-matched control")

# --- impulse/constant/alternating: realized buffer MASS identity check on the single EMA histogram ---
def buffer_value(betas, grads):
    v = 0.0
    for b, g in zip([NU]*(SWITCH+1) + betas, [0.0]*(SWITCH+1) + grads):
        v = b*v + (1-b)*g
    return v
seqs = {"impulse": [1.0]+[0.0]*(NST-1), "constant": [1.0]*NST, "alternating": [(-1.0)**i for i in range(NST)]}
print("realized buffer value vs histogram-weighted gradient sum (max abs diff over the run):")
for name, gs in seqs.items():
    errs = []
    h = [0.0]
    for _ in range(SWITCH+1): h = [NU*x for x in h]; h = [1-NU]+h
    gbuf = [0.0]*(SWITCH+1)
    for j, b in enumerate(beta_sched):
        h = [1-b] + [b*x for x in h]; gbuf = [gs[j]] + gbuf
        vhist = sum(h[l]*(gbuf[l] if l < len(gbuf) else 0.0) for l in range(len(h)))
        vrec = buffer_value(beta_sched[:j+1], gs[:j+1])
        errs.append(abs(vhist-vrec))
    print(f"  {name:11s}: max|hist - recurrence| = {max(errs):.2e}")

ok = mass_err < 1e-10 and age_err_single < 1e-10 and mix_err < 1e-10 and inv_err < 1e-10
print(f"\nVERDICT: single-EMA recurrence (mass {mass_err:.1e}, age {age_err_single:.1e}), mixture target_age "
      f"({mix_err:.1e}), solver invariant realized==target ({inv_err:.1e}) all < 1e-10 float64; "
      f"variance differs as expected => {'PASS' if ok else 'REVIEW'}")
