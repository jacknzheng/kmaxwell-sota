"""REQ-074 time-resolved spectrogram core: retains the TIME dimension that REQ-072/073 analyzers discarded.
For a per-coordinate temporal series (N_coords, T), compute the STFT aggregate power P(step_end, f) =
sum_i |sum_j h_j v_{tau+j,i} e^-2pi i f j|^2 at EVERY window (hop=1), so we can deliver training-time × period ×
amplitude spectrograms. Period = 1/f (f in cycles/update; f=0.5 => period 2).

Primary: window=128, Hann, hop=1, fs=1/update, no padding, no detrending. For a zero-based T-update capture,
columns end at each update W-1 .. T-1 (3123 columns for T=3250, W=128); the first W-1 updates have insufficient
history (marked, never faked). Also supports window 32/256 (sensitivity) and a per-window linear-detrended
variant (raw always retained). Computed in coordinate blocks (never one giant FFT).
"""
from __future__ import annotations
import numpy as np


def time_resolved_power(series, window_W=128, hop=1, fs=1.0, coord_block=8192, detrend=False):
    """series: (N, T) float. Returns freqs, periods, step_end (per column), and P[col, f] summed over coords."""
    N, T = series.shape
    h = np.hanning(window_W)
    win_energy = float((h ** 2).sum())
    starts = list(range(0, T - window_W + 1, hop))
    step_end = [s + window_W - 1 for s in starts]           # x-axis: last update in each window
    freqs = np.fft.rfftfreq(window_W, d=1.0 / fs)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)   # y-axis: oscillation period (updates/cycle)
    P = np.zeros((len(starts), len(freqs)), dtype=np.float64)
    for b0 in range(0, N, coord_block):
        blk = series[b0:b0 + coord_block]
        for ci, s in enumerate(starts):
            seg = blk[:, s:s + window_W].astype(np.float64)
            if detrend:  # per-window linear detrend along time
                x = np.arange(window_W)
                xm = x.mean(); sxx = ((x - xm) ** 2).sum()
                slope = ((seg - seg.mean(1, keepdims=True)) * (x - xm)).sum(1, keepdims=True) / sxx
                seg = seg - (slope * (x - xm) + seg.mean(1, keepdims=True))
            seg = seg * h[None, :]
            F = np.fft.rfft(seg, axis=1)
            P[ci] += (np.abs(F) ** 2).sum(axis=0)
    P /= (fs * win_energy)
    return {"freqs": freqs.tolist(), "periods": periods.tolist(), "step_end": step_end,
            "P": P, "window": window_W, "hop": hop, "detrend": detrend, "N": N, "T": T,
            "n_columns": len(starts), "first_valid_step_end": step_end[0] if step_end else None}
