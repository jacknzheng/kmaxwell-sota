"""Portable actual-kernel frozen root-locus prediction; NumPy/SciPy only.

kappa_boundary is copied from the existing plot_dense_interim.py solver.
Input must be the actual256 FP32 taps promoted to CPU NumPy float64.
This is a frozen-coefficient boundary, not a changing-system stability claim.
"""
import numpy as np
from scipy.optimize import brentq


def kappa_boundary(c, a=1., grid=16385):
    """First positive unit-circle crossing, including theta=pi, avoiding poles."""
    c = np.asarray(c, dtype=np.float64)
    theta = np.linspace(1e-9, np.pi, grid)
    def value(t):
        h = np.polynomial.polynomial.polyval(np.exp(-1j*t), c)
        return (a-np.exp(1j*t))*np.conj(h)
    y = value(theta).imag
    crossings = np.flatnonzero(y[:-1]*y[1:] < 0)
    roots = [brentq(lambda t: value(t).imag, theta[j], theta[j+1], xtol=1e-14)
             for j in crossings] + [np.pi]
    accepted = []
    for t in roots:
        h = np.polynomial.polynomial.polyval(np.exp(-1j*t), c)
        if abs(h) < 1e-10:
            continue
        k = (a-np.exp(1j*t))/h
        if k.real > 0 and abs(k.imag) <= 1e-7*max(1., abs(k.real)):
            accepted.append((float(k.real), float(t)))
    if not accepted:
        raise ValueError('No resolved positive boundary')
    return min(accepted)


def predict_kernel(weights, learning_rate=.025, weight_decay=.05,
                   train_tokens=524288, grid=16385):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (256,) or not np.isfinite(weights).all():
        raise ValueError('Supply all256 actual finite kernel taps, newest first')
    if learning_rate <= 0 or train_tokens <= 0 or weight_decay < 0:
        raise ValueError('Invalid learning-rate, token or weight-decay convention')
    a = 1-learning_rate*weight_decay
    kappa, theta = kappa_boundary(weights, a, grid)
    return dict(kappa_star=kappa, crossing_angle=theta,
                prediction=kappa/(learning_rate*train_tokens)+weight_decay/train_tokens,
                learning_rate=learning_rate, weight_decay=weight_decay,
                train_tokens=train_tokens, kernel_mass=float(weights.sum()),
                convention='actual256taps; frozen-kernel first positive unit-circle contact; total zeta includes weight_decay/train_tokens')
