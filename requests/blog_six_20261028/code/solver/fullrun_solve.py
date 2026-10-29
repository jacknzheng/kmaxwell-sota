"""Select the Arnoldi start; all polar/Hessian mathematics stays in production code."""
import os
import runpy
import sys
import time
from pathlib import Path
from common import write

# This runs before importing the model or allocating CUDA memory. All eight
# existing workers wait while one separate, sequential benchmark uses the node.
# Optional benchmark hook deliberately excluded from this authorized burnin adapter.

from records.track_3_optimization.analysis import coupled_muon
from arnoldi_logging import instrumented_arnoldi
from hvp_graph_cost import install as install_graph_cost_logging
install_graph_cost_logging()

original = coupled_muon.qualify_eigenpair


def qualify(apply, space, dimensions, **kwargs):
    # The queue uses fixed 16M probes and reserves audits for extreme jumps.
    # Each audit is a fresh Gaussian start at 16M; do not redo a second warm solve.
    independent = os.environ.get('DENSE_INDEPENDENT') == '1'
    if independent:
        kwargs['initial'] = space.random(77193)
    if kwargs.get('initial') is None:
        kwargs['initial'] = space.random(921)
    kwargs.update(fresh_start=False, previous_mu=None)
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    tokens = apply.hessian.tokens
    def with_heartbeat(v):
        result = apply(v)
        if space.rank == 0 and apply.calls % 4 == 0:
            write(output / 'heartbeat.json', dict(time=time.time(), operator_calls=apply.calls,
                  operator_seconds=apply.seconds, tokens=tokens))
        return result
    base = coupled_muon.ShardedArnoldi
    coupled_muon.ShardedArnoldi = instrumented_arnoldi(base, output, tokens, apply)
    try:
        if os.environ.get('DENSE_PREFIX_WINDOW'):
            from prefix_solver import qualify_prefix
            result, vector = qualify_prefix(with_heartbeat, space, dimensions,
                window=int(os.environ['DENSE_PREFIX_WINDOW']), **kwargs)
        else:
            result, vector = original(with_heartbeat, space, dimensions, **kwargs)
    finally:
        coupled_muon.ShardedArnoldi = base
    result['queue_initialization'] = 'independent_gaussian_77193' if independent else 'warm_previous'
    if os.environ.get('DENSE_SPEEDUP_BENCHMARK') == '1':
        result['queue_initialization'] = ('previous_checkpoint' if tokens == 1048576 or os.environ.get('DENSE_PREFIX_WINDOW')
                                         else 'current_checkpoint_1m')
    return result, vector


coupled_muon.qualify_eigenpair = qualify
runpy.run_module('records.track_3_optimization.analysis.coupled_muon_study', run_name='__main__')
