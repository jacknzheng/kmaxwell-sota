"""Shared causal low-pass gradient filters for REQ-076 (pair, fir9) and REQ-077 (pair, ema_third).

All operate on a per-coordinate signed raw-gradient history g_raw[t], g_raw[t-1], ... (current + past only,
no look-ahead). Constant-input gain is 1 for all. Registered, fixed designs — coefficients are NOT selected
from training outcomes.

  pair     : g_f[t] = 0.5 g[t] + 0.5 g[t-1]              amp |cos(pi f)|; cancels period-2 exactly; age 0.5
  fir9     : windowed-sinc low-pass, fc=0.20 cyc/update, Hann window, unity DC gain, 9 taps (linear phase,
             4-update delay); strong attenuation of periods 2-4; some negative taps.
  ema_third: v[t] = (2/3) g[t] + (1/3) v[t-1]            h_k=(2/3)(1/3)^k; matches pair's sum=1, age=1/2,
             sum h^2=1/2 (so equal constant gain, mean age, and independent-noise output variance), but
             retains half the period-2 amplitude (quarter power).

`FIR9` (the 9 taps) and `fir9_response`/`pair_response`/`ema_response` are the published responses.
`make_filter(kind)` returns a stateful per-call filter object usable both in CPU tests and the training hook.
"""
from __future__ import annotations
import numpy as np

# Registered FIR9 (fc=0.20, Hann, 9 taps) -- see design in req requests.md
_fc = 0.20
_k = np.arange(9)
_a = 2 * _fc * np.sinc(2 * _fc * (_k - 4)) * (0.5 - 0.5 * np.cos(2 * np.pi * _k / 8))
FIR9 = (_a / _a.sum()).astype(np.float64)        # sum == 1
PAIR = np.array([0.5, 0.5], dtype=np.float64)
EMA_ALPHA = 2.0 / 3.0                             # v = alpha*g + (1-alpha)*v_prev; (1-alpha)=1/3
STARTUP = 8                                       # first 8 updates unfiltered (record history)


def _amp(h, f):
    kk = np.arange(len(h))
    return np.abs(np.sum(np.asarray(h)[None, :] * np.exp(-2j * np.pi * np.asarray(f)[:, None] * kk[None, :]), axis=1))


def pair_response(f): return _amp(PAIR, f)
def fir9_response(f): return _amp(FIR9, f)
def ema_response(f):
    kk = np.arange(256); h = EMA_ALPHA * (1 - EMA_ALPHA) ** kk
    return _amp(h, f)


def responses_table():
    periods = np.array([2, 3, 4, 8, 16, 32]); f = 1.0 / periods
    return {"periods": periods.tolist(), "freqs": f.tolist(),
            "pair_amp": pair_response(f).tolist(), "fir9_amp": fir9_response(f).tolist(),
            "ema_third_amp": ema_response(f).tolist(),
            "FIR9_coeffs": FIR9.tolist(),
            "matched_pair": {"sum": float(PAIR.sum()), "age": float((np.arange(2) * PAIR).sum()),
                             "sum_sq": float((PAIR ** 2).sum())},
            "matched_ema": {"sum": 1.0, "age": 0.5, "sum_sq": 0.5}}


class Filter:
    """Stateful causal filter over tensors (numpy or torch). Call .step(g, filtering_active) each update;
    returns the filtered tensor (or g unchanged during startup). Maintains its own history; never aliases g."""
    def __init__(self, kind: str):
        assert kind in ("none", "pair", "fir9", "ema_third")
        self.kind = kind
        self._hist = []          # pair/fir9: list of recent raw copies, newest last
        self._ema = None         # ema_third shadow state

    def _clone(self, g):
        return g.detach().clone() if hasattr(g, "detach") else np.array(g, copy=True)

    def step(self, g, filtering_active: bool):
        if self.kind == "none":
            return g
        if self.kind == "ema_third":
            if self._ema is None:
                self._ema = self._clone(g)                     # init shadow with first real gradient
            else:
                self._ema = EMA_ALPHA * g + (1 - EMA_ALPHA) * self._ema
            return self._ema if filtering_active else g
        # pair / fir9: push raw copy, keep needed depth
        taps = PAIR if self.kind == "pair" else FIR9
        depth = len(taps)
        self._hist.append(self._clone(g))
        if len(self._hist) > depth:
            self._hist.pop(0)
        if not filtering_active:
            return g
        # convolve newest..oldest with taps[0..]
        hs = self._hist[::-1]                                   # index 0 = current g[t]
        out = taps[0] * hs[0]
        for i in range(1, min(depth, len(hs))):
            out = out + taps[i] * hs[i]
        return out


def make_filter(kind): return Filter(kind)
