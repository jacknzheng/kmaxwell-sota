import torch
import torch.distributed as dist

from records.track_3_optimization.optimizers.ringed.momentum import GradientRing, MomentumScheduler, push_and_apply
from records.track_3_optimization.optimizers.ringed.state import FiniteHistoryOptimizer


def zeropower_via_newtonschulz5(G):
    """The approved record's 12-step BF16 polar approximation."""
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def order_params_like_record(params):
    return sorted(params, key=lambda x: x.size(), reverse=True)


def owned_param_indices(num_params, rank, world_size):
    return range(rank, num_params, world_size)


class Muon(FiniteHistoryOptimizer):
    def __init__(self, params, momentum_scheduler: MomentumScheduler,
                 lr=0.02, weight_decay=0.0):
        params = order_params_like_record(list(params))
        if not params or any(p.ndim != 2 for p in params):
            raise ValueError("Muon requires nonempty matrix parameters")
        self.momentum_scheduler = momentum_scheduler
        self.polar = torch.compile(zeropower_via_newtonschulz5) if torch.cuda.is_available() else zeropower_via_newtonschulz5
        # Observer receives the actual pre-polar input, not the executed
        # parameter displacement (polar map, scaling, decay and retraction follow).
        # Read-only use is a callback contract; live references are not isolated.
        self.momentum_observer = None
        self.capture_polar_observer = None
        self.capture_pre_polar = self.capture_post_polar = None
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, iteration=0))

    def sorted_params(self):
        return self.param_groups[0]["params"]

    def direction(self, p, state, kernel, iteration):
        return push_and_apply(GradientRing(state), p.grad, kernel)

    def apply_update(self, p, update, group):
        p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update, alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            iteration = group["iteration"]
            kernel = self.momentum_scheduler(iteration)
            params = group["params"]
            # Same ownership/sort/all-gather grouping as the record.
            padded = params + [torch.empty_like(params[-1])] * (world - len(params) % world)
            for base in range(0, len(params), world):
                if base + rank < len(params):
                    p = params[base + rank]
                    if p.grad is None:
                        # This ring advances in optimizer iterations, not observed-gradient count.
                        push_and_apply(GradientRing(self.state[p]), torch.zeros_like(p), kernel)
                    else:
                        if p.grad.is_sparse:
                            raise ValueError("Muon does not support sparse gradients")
                        direction = self.direction(p, self.state[p], kernel, iteration)
                        if self.capture_pre_polar is not None:
                            self.capture_pre_polar[id(p)] = direction.detach().clone()
                        if self.momentum_observer is not None:
                            self.momentum_observer(p, p.grad, direction, kernel, iteration, self.state[p])
                        update = self.polar(direction)
                        if self.capture_polar_observer is not None:
                            self.capture_polar_observer(p, p.grad, direction, update,
                                                        kernel, iteration, self.state[p])
                        update *= max(1, p.size(-2) / p.size(-1)) ** 0.5
                        if self.capture_post_polar is not None:
                            self.capture_post_polar[id(p)] = update.detach().clone()
                        self.apply_update(p, update, group)
                if distributed:
                    dist.all_gather(padded[base:base + world], padded[base + rank])
            group["iteration"] += 1
        return loss


class MuonH(Muon):
    """Frobenius-sphere updates; optional reviewed radial/tangent fraction.

    The fraction is a radial input transformation, NOT a gamma momentum kernel.
    This preserves the transform's definition, not full trainer/rounding parity.
    """
    def __init__(self, params, momentum_scheduler, lr=0.02,
                 radial_to_tangent_ratio=None, radial_start_step=780):
        super().__init__(params, momentum_scheduler, lr=lr, weight_decay=0.0)
        if radial_to_tangent_ratio is not None and radial_to_tangent_ratio < 0:
            raise ValueError("radial fraction must be nonnegative")
        self.radial_to_tangent_ratio = radial_to_tangent_ratio
        self.radial_start_step = radial_start_step

    def direction(self, p, state, kernel, iteration):
        momentum = super().direction(p, state, kernel, iteration)
        ratio = self.radial_to_tangent_ratio
        if ratio is None or iteration < self.radial_start_step:
            return momentum
        parameter = p.float()
        norm_sq = parameter.square().sum()
        if norm_sq <= 0:
            raise ValueError("radial projection requires a nonzero matrix")
        coefficient = (momentum * parameter).sum() / norm_sq
        tangent = momentum - coefficient * parameter
        target = ratio * torch.sign(coefficient) * tangent.norm() / norm_sq.sqrt()
        return tangent + target * parameter

    def apply_update(self, p, update, group):
        p_norm = p.norm()
        candidate = p - group["lr"] * update * p_norm / update.norm().clamp_min(1e-10)
        p.copy_(candidate / candidate.norm().clamp_min(1e-10) * p_norm)
