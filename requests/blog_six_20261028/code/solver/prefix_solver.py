"""Experimental per-column checks; use the unchanged production Arnoldi arithmetic."""
import numpy as np
from records.track_3_optimization.analysis import coupled_muon


def qualify_prefix(apply, space, dimensions, *, initial, progress=None, checkpoint=None,
                   eigenvalue_rtol=.001, window=9, **unused):
    if initial is None:
        raise ValueError('This benchmark requires the same previous-checkpoint start as its reference')
    state = coupled_muon.ShardedArnoldi(apply, space, 921, initial)
    history = []
    accepted = False
    for n in range(1, max(dimensions) + 1):
        state.extend(n)
        values = np.linalg.eigvals(state.h[:n, :n])
        positive = [z.real for z in values if z.real > 0 and abs(z.imag) <= 1e-7 * max(1., abs(z.real))]
        mu = float(max(positive)) if positive else None
        row = dict(dimension=n, seed=921, mu=mu, residual=None, relative_change=None,
                   spectrum=[[float(z.real), float(z.imag)] for z in values])
        history.append(row)
        recent = [p['mu'] for p in history[-window:]]
        spread = ((max(recent) - min(recent)) / abs(mu)
                  if len(recent) == window and all(v is not None and np.isfinite(v) and v > 0 for v in recent)
                  else None)
        row['relative_change'] = spread
        accepted = bool(spread is not None and spread <= eigenvalue_rtol)
        if state.breakdown:
            accepted = mu is not None and np.isfinite(mu)
        if accepted or state.breakdown or n == max(dimensions):
            # Keep the production vector reconstruction and direct residual diagnostic.
            final, vector = state.ritz()
            final.update(seed=921, relative_change=spread)
            history[-1] = final
            if checkpoint is not None and vector is not None:
                checkpoint(final, vector)
            if progress:
                progress(final)
            break
        if progress and n % 4 == 0:
            progress(row)
    return dict(qualified=bool(accepted), searches=[dict(qualified=bool(accepted), history=history)],
        seed_relative_spread=None, residual_gating=False,
        stopping_rule=f'positive_real_ritz_spread_over_{window}_consecutive_dimensions',
        independent_start_checked=False, fresh_start_reason=None, warm_start_missed_larger_mode=False,
        mu=history[-1]['mu'], experimental_stopping=True), vector
