"""CPU validation for the REQ-076/077 causal filters (Stage-1 synthetic checks)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req07x_filters import make_filter, PAIR, FIR9, EMA_ALPHA, pair_response, fir9_response, ema_response
ok = True
rng = np.random.default_rng(0)

# T1: constant-input gain == 1 for all filters (after startup fill)
for kind in ("pair", "fir9", "ema_third"):
    f = make_filter(kind); c = np.ones(5) * 3.14
    for _ in range(40): out = f.step(c, True)
    good = np.allclose(out, c, atol=1e-9)
    print(f"T1 {kind:9s} constant gain=1: {'PASS' if good else 'FAIL'}"); ok &= good

# T2: pure period-2 alternation g=s+(-1)^t a -> pair=s, ema=s+0.5(-1)^t a, fir9~=s
s = np.array([1.0]); a = np.array([0.4])
res = {}
for kind in ("pair", "fir9", "ema_third"):
    f = make_filter(kind); last = None
    for t in range(60): last = f.step(s + ((-1.0) ** t) * a, True); tt = t
    res[kind] = last
pair_ok = np.allclose(res["pair"], s, atol=1e-9)
ema_ok = np.allclose(abs(res["ema_third"] - s), 0.5 * a, atol=1e-6)
fir_ok = abs(res["fir9"] - s) < 0.02
print(f"T2 pair cancels period-2 -> s: {'PASS' if pair_ok else 'FAIL'} ({res['pair']})"); ok &= pair_ok
print(f"T2 ema retains half amp: {'PASS' if ema_ok else 'FAIL'} (|dev|={abs(res['ema_third']-s)} exp {0.5*a})"); ok &= ema_ok
print(f"T2 fir9 ~cancels period-2: {'PASS' if fir_ok else 'FAIL'} (dev={abs(res['fir9']-s)})"); ok &= fir_ok

# T3: sinusoidal amplitude matches published response (pair & fir9) via DFT projection (phase-robust)
for period, kind, respfn in [(4,"pair",pair_response),(8,"pair",pair_response),(4,"fir9",fir9_response),(8,"fir9",fir9_response)]:
    fr = 1.0/period; f = make_filter(kind); outs=[]
    for t in range(400): o=f.step(np.array([np.cos(2*np.pi*fr*t)]), True); outs.append(float(o))
    tail=np.array(outs[-200:]); tt=np.arange(len(tail))
    # amplitude of a single real tone via quadrature projection
    c=2*np.mean(tail*np.cos(2*np.pi*fr*tt)); s=2*np.mean(tail*np.sin(2*np.pi*fr*tt))
    meas_amp=float(np.hypot(c,s))
    exp_amp = float(respfn(np.array([fr]))[0])
    good = abs(meas_amp-exp_amp) < 0.01
    print(f"T3 {kind} period {period}: meas amp {meas_amp:.4f} vs published {exp_amp:.4f} {'PASS' if good else 'FAIL'}"); ok &= good

# T4: irregular noisy-vector correctness (fir9 == manual convolution of last 9)
f = make_filter("fir9"); hist=[]; t4ok=True
for t in range(20):
    g = rng.standard_normal(7); hist.append(g.copy()); out=f.step(g, True)
    if t>=8:
        man = sum(FIR9[i]*hist[t-i] for i in range(9))
        if not np.allclose(out, man, atol=1e-10): t4ok=False; print("  T4 mismatch at t",t)
print(f"T4 fir9 == manual convolution: {'PASS' if t4ok else 'FAIL'}"); ok &= t4ok

# T5: no aliasing -- filter does not mutate the input tensor, history independent
f = make_filter("pair"); g0=np.ones(3); f.step(g0, True); g1=np.array([5.,5.,5.]); f.step(g1, True)
g1[0]=999.0  # mutate caller's tensor after the fact
out=f.step(np.ones(3)*2, True)  # should use stored copy (5,5,5) not the mutated 999
alias_ok = abs(out[0]-(0.5*2+0.5*5)) < 1e-9
print(f"T5 no input aliasing: {'PASS' if alias_ok else 'FAIL'} (out0={out[0]}, want {0.5*2+0.5*5})"); ok &= alias_ok

# T6: matched steady-state properties pair vs ema (sum, age, variance)
kk=np.arange(256); hema=EMA_ALPHA*(1-EMA_ALPHA)**kk
m_ok = (abs(PAIR.sum()-1)<1e-9 and abs((np.arange(2)*PAIR).sum()-0.5)<1e-9 and abs((PAIR**2).sum()-0.5)<1e-9
        and abs(hema.sum()-1)<1e-6 and abs((kk*hema).sum()-0.5)<1e-6 and abs((hema**2).sum()-0.5)<1e-6)
print(f"T6 matched sum/age/var pair==ema: {'PASS' if m_ok else 'FAIL'}"); ok &= m_ok

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-076/077 filter core")
sys.exit(0 if ok else 1)
