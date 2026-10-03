"""CPU validation for the REQ-072 STFT spectrogram core."""
import sys, os, warnings, numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req072_spectrogram import stft_power, band_power
rng = np.random.default_rng(0)
T, N = 512, 2000
t = np.arange(T)
ok = True

# T1: pure period-two alternation per coord -> power concentrated at f=0.5
alt = ((-1.0) ** t)[None, :] * rng.standard_normal((N, 1))
r = stft_power(alt, window_W=64, hop=32)
bp = band_power(r["freqs"], r["spectrum_f"])
print(f"T1 period-2 signal: period2_frac={bp['period2_frac']:.3f} low_frac={bp['low_frac']:.3f} "
      f"{'PASS' if bp['period2_frac']>0.8 else 'FAIL'}")
ok &= bp["period2_frac"] > 0.8

# T2: slow sinusoid (period 128) per coord -> power at low f
slow = np.cos(2*np.pi*t/128)[None, :] * rng.standard_normal((N, 1))
r2 = stft_power(slow, window_W=64, hop=32)
bp2 = band_power(r2["freqs"], r2["spectrum_f"])
print(f"T2 slow signal: low_frac={bp2['low_frac']:.3f} period2_frac={bp2['period2_frac']:.3f} "
      f"{'PASS' if bp2['low_frac']>bp2['period2_frac'] and bp2['period2_frac']<0.1 else 'FAIL'}")
ok &= bp2["low_frac"] > bp2["period2_frac"] and bp2["period2_frac"] < 0.1

# T3: coordinate blocking is exact (block size < N gives same result as one block)
rb = stft_power(alt, window_W=64, hop=32, coord_block=137)
e3 = np.abs(rb["spectrum_f"] - r["spectrum_f"]).max()
print(f"T3 coord-blocking exact: max|d|={e3:.2e} {'PASS' if e3<1e-9 else 'FAIL'}"); ok &= e3 < 1e-9

# T4: sum-of-squares across coords (not cancellation): two coords with opposite sign same freq still add power
opp = np.vstack([((-1.0)**t), -((-1.0)**t)])  # opposite-sign alternation
ro = stft_power(opp, window_W=64, hop=32)
# signed average would cancel to ~0; sum of |FFT|^2 must be > 0 and in period-2 band
bpo = band_power(ro["freqs"], ro["spectrum_f"])
print(f"T4 no cancellation (opposite-sign coords): total_power={ro['total_power']:.2f}>0, period2_frac={bpo['period2_frac']:.3f} "
      f"{'PASS' if ro['total_power']>0 and bpo['period2_frac']>0.8 else 'FAIL'}")
ok &= ro["total_power"] > 0 and bpo["period2_frac"] > 0.8

# T5: mean-per-coord power scales as total/N
print(f"T5 mean_per_coord == total/N: {abs(r['mean_per_coord_power'] - r['total_power']/r['N']):.2e} "
      f"{'PASS' if abs(r['mean_per_coord_power']-r['total_power']/r['N'])<1e-9 else 'FAIL'}")
ok &= abs(r["mean_per_coord_power"] - r["total_power"]/r["N"]) < 1e-9

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-072 STFT spectrogram core")
sys.exit(0 if ok else 1)
