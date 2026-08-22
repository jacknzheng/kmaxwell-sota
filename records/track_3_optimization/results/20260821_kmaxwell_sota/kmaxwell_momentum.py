"""K-Maxwell first-moment mix. Mutates the stacked EMA buffers, not grad.

Split out so identity checks can import the mix without launching the SOTA
trainer (which initializes NCCL at import).
"""


def kmaxwell_first_moment(grad, m, betas, weights, mu):
    """In-place K-EMA mix. m has shape (K, *grad.shape). Does not mutate grad."""
    lerp_coeff = (1 - betas).reshape(-1, *([1] * grad.ndim))
    m.lerp_(grad.unsqueeze(0), lerp_coeff)
    m_eff = (weights.reshape(-1, *([1] * grad.ndim)) * m).sum(dim=0)
    return grad.lerp(m_eff, mu)
