import torch

from records.track_3_optimization.optimizers.ringed.momentum.constants import HISTORY_LENGTH
from records.track_3_optimization.optimizers.ringed.momentum import GradientRing, push_and_apply
from records.track_3_optimization.optimizers.ringed.state import FiniteHistoryOptimizer


def row_weights(state, prefix, kernel, device):
    weights = kernel.weights.to(device=device)
    if (weights < 0).any():
        raise ValueError("SOAP statistic kernels must be nonnegative")
    if kernel.normalize_available:
        filled = state[prefix + "_filled"]
        mass = weights[:filled].sum()
        if mass <= 0:
            raise ValueError("SOAP available mass must be positive")
        weights = weights / mass
    ages = (state[prefix + "_newest"] - torch.arange(HISTORY_LENGTH, device=device)) % HISTORY_LENGTH
    return weights[ages]


class SOAP(FiniteHistoryOptimizer):
    def __init__(self, params, gradient_momentum_scheduler,
                 preconditioner_basis_momentum_scheduler,
                 preconditioner_spectrum_momentum_scheduler,
                 lr=0.02, weight_decay=0.0, eps=1e-8, precondition_frequency=1):
        params = list(params)
        if not params or any(p.ndim != 2 for p in params):
            raise ValueError("SOAP currently supports nonempty matrix parameters")
        if eps <= 0 or precondition_frequency < 1:
            raise ValueError("positive epsilon and basis frequency are required")
        self.gradient_scheduler = gradient_momentum_scheduler
        self.basis_scheduler = preconditioner_basis_momentum_scheduler
        self.spectrum_scheduler = preconditioner_spectrum_momentum_scheduler
        self.eps = eps
        self.precondition_frequency = precondition_frequency
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, iteration=0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            iteration = group["iteration"]
            gradient_kernel = self.gradient_scheduler(iteration)
            basis_kernel = self.basis_scheduler(iteration)
            spectrum_kernel = self.spectrum_scheduler(iteration)
            for p in group["params"]:
                state = self.state[p]
                raw = p.grad if p.grad is not None else torch.zeros_like(p)
                if raw.is_sparse:
                    raise ValueError("SOAP does not support sparse gradients")
                momentum = push_and_apply(GradientRing(state), raw, gradient_kernel)
                # Raw gradient history is shared for gradient and basis statistics.
                physical = state["gradient_history"].view(HISTORY_LENGTH, *p.shape)
                basis_weights = row_weights(state, "gradient", basis_kernel, p.device)
                if "q_row" not in state or iteration % self.precondition_frequency == 0:
                    row_gram = torch.einsum("k,kij,klj->il", basis_weights, physical, physical)
                    col_gram = torch.einsum("k,kij,kil->jl", basis_weights, physical, physical)
                    state["q_row"] = torch.linalg.eigh(row_gram)[1]
                    state["q_col"] = torch.linalg.eigh(col_gram)[1]
                push_and_apply(GradientRing(state, "spectrum"), momentum, spectrum_kernel)
                q_row, q_col = state["q_row"], state["q_col"]
                past = state["spectrum_history"].view(HISTORY_LENGTH, *p.shape)
                projected_history = q_row.T @ past @ q_col
                weights = row_weights(state, "spectrum", spectrum_kernel, p.device)
                variance = torch.einsum("k,kij->ij", weights, projected_history.square())
                state["variance"] = variance
                projected = q_row.T @ momentum @ q_col
                update = q_row @ (projected / variance.clamp_min(self.eps ** 2).sqrt()) @ q_col.T
                if p.grad is not None:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            group["iteration"] += 1
        return loss
