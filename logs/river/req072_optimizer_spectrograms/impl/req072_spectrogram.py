"""REQ-072 temporal STFT spectrogram core (CPU-testable). A parameter matrix supplies N signed scalar time
series (one per coordinate) over T optimizer steps. We compute the temporal STFT per coordinate and SUM the
squared magnitudes across coordinates (NOT norms/abs/signed-average, which would hide sign-flipping
oscillation or cancel coordinates):

  P_m(tau, f) = sum_i | sum_j h_j u_{tau+j, i} e^{-2 pi i f j} |^2

Hann window h, hop, normalized by window energy and sampling frequency (fs = 1 per optimizer step, so f is
in cycles/step; Nyquist = 0.5 = the period-two band). Computed in coordinate blocks (never loads all coords
into one FFT). Returns the per-window spectrogram P(tau, f), the time-averaged spectrum P(f), and total +
mean-per-coordinate power. band_power splits low-f vs the period-two (f near 0.5) band.
"""
from __future__ import annotations
import numpy as np


def stft_power(series, window_W=64, hop=32, fs=1.0, coord_block=8192):
    """series: (N_coords, T) float array. Returns dict with freqs, spectrogram P(tau,f) summed over coords,
    time-averaged spectrum P(f), tau starts, N, total_power, mean_per_coord_power."""
    N, T = series.shape
    h = np.hanning(window_W)
    win_energy = float((h ** 2).sum())
    starts = list(range(0, T - window_W + 1, hop))
    freqs = np.fft.rfftfreq(window_W, d=1.0 / fs)  # 0 .. 0.5 (Nyquist = period-two)
    P = np.zeros((len(starts), len(freqs)), dtype=np.float64)  # summed over coords
    for b0 in range(0, N, coord_block):
        blk = series[b0:b0 + coord_block]  # (nb, T)
        for ti, s in enumerate(starts):
            seg = blk[:, s:s + window_W] * h[None, :]              # (nb, W)
            F = np.fft.rfft(seg, axis=1)                            # (nb, F)
            P[ti] += (np.abs(F) ** 2).sum(axis=0)
    # PSD normalization: / (fs * window_energy)
    P /= (fs * win_energy)
    Pf = P.mean(axis=0)                                            # time-averaged spectrum
    total = float(Pf.sum())
    return {"freqs": freqs.tolist(), "starts": starts, "spectrogram": P, "spectrum_f": Pf,
            "N": N, "T": T, "total_power": total, "mean_per_coord_power": total / N if N else 0.0}


def band_power(freqs, spectrum_f, period2_half_width=0.05):
    """Fraction of time-averaged power in the period-two band (f near 0.5) vs the low band (f near 0),
    vs mid. Returns the three fractions + the period-two/low ratio."""
    f = np.asarray(freqs); Pf = np.asarray(spectrum_f)
    tot = Pf.sum() or 1.0
    p2 = Pf[f >= 0.5 - period2_half_width].sum()
    low = Pf[f <= period2_half_width].sum()
    mid = tot - p2 - low
    return {"period2_frac": float(p2 / tot), "low_frac": float(low / tot), "mid_frac": float(mid / tot),
            "period2_over_low": float(p2 / low) if low > 0 else float("inf")}
