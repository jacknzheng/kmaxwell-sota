"""CPU validation for REQ-074 time-resolved spectrogram."""
import sys, os, warnings, numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req074_spectrogram import time_resolved_power
rng = np.random.default_rng(0); ok = True
T, N, W = 3250, 500, 128

# T1: column count + step_end axis (3123 columns ending 127..3249 for T=3250,W=128,hop=1)
t = np.arange(T)
sig = (((-1.0) ** t)[None, :] * rng.standard_normal((N, 1)))  # period-2 everywhere
r = time_resolved_power(sig, window_W=W, hop=1)
c_ok = r["n_columns"] == 3123 and r["step_end"][0] == 127 and r["step_end"][-1] == 3249
print(f"T1 columns={r['n_columns']} step_end[0]={r['step_end'][0]} [-1]={r['step_end'][-1]} {'PASS' if c_ok else 'FAIL'}")
ok &= c_ok

# T2: period-2 -> power peaks at period 2 (f=0.5, last freq bin) at ALL times
P = np.array(r["P"]); periods = np.array(r["periods"])
peak_period = periods[P.mean(0).argmax()]
print(f"T2 dominant period={peak_period:.2f} (expect 2.0) {'PASS' if abs(peak_period-2.0)<0.01 else 'FAIL'}")
ok &= abs(peak_period - 2.0) < 0.01

# T3: TIME-LOCALIZATION — a burst of period-2 only in the first half shows high power early, low late
burst = np.zeros((N, T)); burst[:, :T//2] = ((-1.0) ** t[:T//2])[None, :] * rng.standard_normal((N, 1))
rb = time_resolved_power(burst, window_W=W, hop=1)
Pb = np.array(rb["P"]); se = np.array(rb["step_end"]); f = np.array(rb["freqs"])
p2col = Pb[:, f >= 0.49].sum(1)  # period-two power per column
early = p2col[se < T//2 - W].mean(); late = p2col[se > T//2 + W].mean()
print(f"T3 time-localization: early p2 power={early:.1f} >> late={late:.3f} {'PASS' if early > 10*max(late,1e-9) else 'FAIL'}")
ok &= early > 10 * max(late, 1e-9)

# T4: coord-blocking exact
rb2 = time_resolved_power(sig, window_W=W, hop=1, coord_block=137)
e4 = np.abs(np.array(rb2["P"]) - np.array(r["P"])).max()
print(f"T4 coord-blocking exact: {e4:.2e} {'PASS' if e4<1e-7 else 'FAIL'}"); ok &= e4 < 1e-7

# T5: detrend removes a linear ramp's low-f leakage but keeps period-2
ramp = (t.astype(float)[None, :] * rng.standard_normal((N, 1)) * 0.01) + ((-1.0) ** t)[None, :]
rraw = time_resolved_power(ramp, W, 1, detrend=False); rdet = time_resolved_power(ramp, W, 1, detrend=True)
fr = np.array(rraw["freqs"]); lowraw = np.array(rraw["P"])[:, fr <= 0.05].sum(); lowdet = np.array(rdet["P"])[:, fr <= 0.05].sum()
print(f"T5 detrend reduces low-f: raw={lowraw:.1f} det={lowdet:.1f} {'PASS' if lowdet < lowraw else 'FAIL'}"); ok &= lowdet < lowraw

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-074 time-resolved spectrogram core")
sys.exit(0 if ok else 1)
