"""REQ-073 Muon stage-split: expose the two Muon intermediates WITHOUT perturbing training. We compute
stage2 from copies (out-of-place) and then call the REAL muon_update for the actual result, so training stays
bit-for-bit identical to stock Muon (no reimplementation of the bf16 orthogonalization).

  stage2 = Muon input after momentum/Nesterov = grad.lerp(momentum_after, mu), momentum_after = momentum + (1-mu)(grad-momentum)
  stage3 = post-polar conditioned direction = the actual muon_update(...) output

split_capture(real_muon_update, grad, momentum, mu, nesterov=True) -> (final_update, stage2, stage3).
`final_update`, and the in-place mutation of grad+momentum, come entirely from real_muon_update (bit-identical).
CPU-tested: final/grad/momentum bit-identical to calling muon_update directly; stage2 matches the internal
pre-orthogonalization input.
"""
from __future__ import annotations


def split_capture(real_muon_update, grad, momentum, mu=0.95, nesterov=True):
    # stage2 from copies (do NOT mutate grad/momentum): replicate muon_update's internal pre-zeropower update
    mom_after = momentum.clone()
    mom_after.lerp_(grad, 1 - mu)                               # = momentum after muon_update's momentum.lerp_
    stage2 = grad.lerp(mom_after, mu) if nesterov else mom_after  # out-of-place (grad untouched)
    stage2 = stage2.detach().clone()
    # actual update via the REAL muon_update (mutates grad+momentum exactly as stock -> bit-identical training)
    final = real_muon_update(grad, momentum, mu=mu)
    stage3 = final.detach().clone()
    return final, stage2, stage3
