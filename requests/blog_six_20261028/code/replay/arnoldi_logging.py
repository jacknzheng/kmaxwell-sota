"""Small, diagnostic-only Arnoldi records; no extra operator applications."""
import numpy as np
from common import write


def instrumented_arnoldi(base, output, tokens, apply):
    serial = 0

    class LoggedArnoldi(base):
        def __init__(self, operator, space, seed, initial=None):
            nonlocal serial
            super().__init__(operator, space, seed, initial)
            serial += 1
            self.diagnostic_path = output / f'arnoldi_N{tokens}_search{serial:02d}_seed{seed}.json'
            self.diagnostic = dict(version=1, tokens=tokens, seed=seed,
                warm_start=initial is not None, prefixes=[], direct_checks=[],
                note='Prefix Ritz values are diagnostics, not a certificate of the largest mode. '
                     'Production arithmetic, direct residual checks, and stopping are unchanged.')

        def save_diagnostic(self):
            if self.space.rank == 0:
                self.diagnostic.update(hessenberg=self.h.tolist(),
                    operator_calls=apply.calls, operator_seconds=apply.seconds)
                write(self.diagnostic_path, self.diagnostic)

        def extend(self, dimension):
            while self.h.shape[1] < dimension and not self.breakdown:
                super().extend(self.h.shape[1] + 1)
                if self.space.rank != 0:
                    continue
                n = self.h.shape[1]
                values = np.linalg.eigvals(self.h[:n, :n])
                positive = [z.real for z in values
                            if z.real > 0 and abs(z.imag) <= 1e-7 * max(1., abs(z.real))]
                mu = float(max(positive)) if positive else None
                previous = self.diagnostic['prefixes'][-1]['mu'] if self.diagnostic['prefixes'] else None
                self.diagnostic['prefixes'].append(dict(dimension=n, mu=mu,
                    relative_change=abs(mu - previous) / abs(mu) if mu and previous else None,
                    spectrum=[[float(z.real), float(z.imag)] for z in values]))
                self.save_diagnostic()

        def ritz(self):
            row, vector = super().ritz()
            if self.space.rank == 0:
                self.diagnostic['direct_checks'].append(dict(row))
                self.save_diagnostic()
            return row, vector

    return LoggedArnoldi
