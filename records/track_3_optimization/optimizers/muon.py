from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist
from torch import Tensor

from secant_gmres_solver.serialization import PR340_BIMAXWELL


@torch.compiler.disable
@torch.no_grad()
def _accumulate_activation_covariance(module: torch.nn.Module,
                                      inputs: tuple[Tensor, ...], output: Tensor) -> None:
    """Accumulate X'X only on optimizer-selected refresh steps."""
    ref = getattr(module, "_newton_covariance_ref", None)
    if ref is None or not ref["collect"]:
        return
    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
    blocks = ref["blocks"]
    width = x.shape[-1] // blocks
    xb = x.reshape(x.shape[0], blocks, width).transpose(0, 1)
    ref["accum"].add_(torch.bmm(xb.transpose(1, 2), xb))
    ref["count"].add_(x.shape[0])


def attach_newton_muon_activation_stats(model: torch.nn.Module) -> list[Any]:
    """Attach shared activation-covariance records to GPT hidden-matrix weights."""
    handles: list[Any] = []

    def attach(linear: torch.nn.Module, blocks: int = 1) -> dict[str, Any]:
        width = linear.weight.shape[1] // blocks
        ref: dict[str, Any] = {
            "blocks": blocks, "collect": False,
            "accum": torch.zeros(blocks, width, width, device=linear.weight.device),
            "count": torch.zeros((), device=linear.weight.device),
        }
        linear.weight._newton_covariance_ref = ref
        linear._newton_covariance_ref = ref
        handles.append(linear.register_forward_hook(_accumulate_activation_covariance))
        return ref

    for block in model.blocks:
        qkv = attach(block.attn.q)
        block.attn.k.weight._newton_covariance_ref = qkv
        block.attn.v.weight._newton_covariance_ref = qkv
        attach(block.attn.proj)
        attach(block.mlp.fc)
        attach(block.mlp.proj, blocks=4)
    return handles


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """The record's Newton-Schulz polar approximation, byte-identical to the anchor
    (train_gpt_simple.py) -- any change alters the compiled kernels and hence the
    bf16 rounding of every Muon-family trajectory."""
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    """The record's single-EMA Nesterov + polar kernel, byte-identical to the anchor
    (note the load-bearing in-place grad.lerp_ mutation)."""
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def compile_bimaxwell_kernel(fast_decay: float, slow_decay: float,
                             fast_weight: float) -> Callable[..., Tensor]:
    """Compiles PR #340's bi-Maxwell kernel with the rates closure-captured as
    constants, exactly as the artifact bakes them in via module globals -- passing
    them as tensor arguments would change the compiled kernels and the rounding."""
    @torch.compile
    def advance_bimaxwell_and_polar(grad: Tensor, m_fast: Tensor, m_slow: Tensor,
                                    mu: float = 0.95) -> Tensor:
        # bi-Maxwell stress memory: two fixed-rate EMA units, convex combination; the
        # Nesterov mix and the orthogonalization downstream are the baseline's, unchanged.
        m_fast.lerp_(grad, 1 - fast_decay)
        m_slow.lerp_(grad, 1 - slow_decay)
        m_eff = fast_weight * m_fast + (1 - fast_weight) * m_slow
        update = grad.lerp_(m_eff, mu)
        update = zeropower_via_newtonschulz5(update)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update
    return advance_bimaxwell_and_polar


@torch.compile
def annealed_decay_update(grad: Tensor, momentum: Tensor, beta: Tensor,
                          mu: float = 0.95) -> Tensor:
    """One dynamically-decayed EMA with a fixed Nesterov blend coefficient."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, mu)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def compile_fixed_decay_kernel(decay: float) -> Callable[..., Tensor]:
    @torch.compile
    def fixed_decay_update(grad: Tensor, momentum: Tensor,
                           mu: float = 0.95) -> Tensor:
        momentum.lerp_(grad, 1 - decay)
        update = grad.lerp_(momentum, mu)
        update = zeropower_via_newtonschulz5(update)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update
    return fixed_decay_update


def order_params_like_record(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    """The record's stable size-descending sort, which fixes rank-ownership order
    (and therefore dump sorted_index) for the whole Muon family."""
    return sorted(params, key=lambda x: x.size(), reverse=True)


def owned_param_indices(num_params: int, rank: int, world_size: int) -> range:
    """Indices of the sorted parameter list this rank owns -- the single definition
    of Muon ownership, shared with the recorder hooks."""
    return range(rank, num_params, world_size)


class Muon(torch.optim.Optimizer):
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95) -> None:
        """The record's Muon: single-EMA Nesterov momentum, blockwise polar map.

        The modded-nanogpt communication pattern lives entirely in step():
        parameters sorted size-descending, rank r owns indices r, r+world, ...,
        each owner computes its polar update, and the fleet all_gathers refreshed
        parameters. The params_pad expression is kept bug-compatible (it appends
        world_size pads when len(params) %% world_size == 0, which the 72-param
        record hits) -- do not "fix" it.
        """
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = order_params_like_record(params)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        assert len(self.param_groups) == 1
        self._muon_steps_seen = 0

    def sorted_params(self) -> list[torch.nn.Parameter]:
        """The size-descending parameter list that defines rank ownership."""
        return self.param_groups[0]["params"]

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        """Produces the polarized, shape-scaled update for one owned parameter;
        subclasses replace this to change the pre-polar direction only."""
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)
        return muon_update(p.grad, state["momentum"], mu=group["mu"])

    @torch.no_grad()
    def step(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    update = self.compute_polar_input(p, self.state[p], group)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self._muon_steps_seen += 1


class SgdBlocks(torch.optim.Optimizer):
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.1,
                 weight_decay: float = 0.0, momentum: float = 0.0) -> None:
        """Plain SGD over the blocks group, with the Muon family's sorted-parameter
        seam so the recording hooks work unchanged.

        Gradients are already all-reduced before apply_updates, and the update is
        elementwise, so every rank applies the identical update with no gathering.
        momentum > 0 gives standard heavy-ball accumulation (buf = mu*buf + g);
        the default 0 is the momentum-free control.
        """
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = order_params_like_record(params)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))
        assert len(self.param_groups) == 1

    def sorted_params(self) -> list[torch.nn.Parameter]:
        """The size-descending parameter list that defines rank ownership."""
        return self.param_groups[0]["params"]

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = p.grad
                if group["momentum"] > 0:
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    state["momentum"].mul_(group["momentum"]).add_(p.grad)
                    update = state["momentum"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


class BimaxwellMuon(Muon):
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95,
                 fast_decay: float = PR340_BIMAXWELL["fast_decay"],
                 slow_decay: float = PR340_BIMAXWELL["slow_decay"],
                 fast_weight: float = PR340_BIMAXWELL["fast_weight"],
                 switch_step: int = PR340_BIMAXWELL["switch_step"]) -> None:
        """PR #340's bi-Maxwell Muon: baseline until switch_step, then the two-rate
        convex combination, with both buffers lazy-initialized from the single-EMA
        momentum so the switch step itself is bit-identical to the baseline's."""
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu)
        self.switch_step = switch_step
        self._advance_bimaxwell_and_polar = compile_bimaxwell_kernel(fast_decay, slow_decay, fast_weight)

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        """Routes one parameter through baseline, switch-step, or bi-Maxwell logic.

        Before switch_step: the single-EMA baseline kernel. On the switch step:
        advance the single-EMA momentum once more and lazy-initialize both
        m_fast and m_slow from it, so that step's update is bit-identical to the
        baseline's by construction. After: the compiled bi-Maxwell kernel.
        """
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)
        if self._muon_steps_seen >= self.switch_step:
            if "m_fast" not in state:
                # switch step: advance the single-EMA momentum once more and
                # lazy-init both units from it -> this step's update is
                # bit-identical to the baseline's, by construction
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                state["m_fast"] = state["momentum"].clone()
                state["m_slow"] = state["momentum"].clone()
                return update
            return self._advance_bimaxwell_and_polar(p.grad, state["m_fast"], state["m_slow"], mu=group["mu"])
        return muon_update(p.grad, state["momentum"], mu=group["mu"])


class RightPreconditionedMuonMixin:
    """Right-precondition raw gradients, then apply the selected momentum kernel."""
    def __init__(self, *args: Any, newton_alpha: float = 0.0,
                 precond_refresh_interval: int = 10, precond_beta: float = 0.95,
                 precond_damping: float = 0.2, precond_eps: float = 1e-8,
                 **kwargs: Any) -> None:
        assert 0 <= newton_alpha <= 1 and precond_refresh_interval >= 1
        super().__init__(*args, **kwargs)
        self.newton_alpha = float(newton_alpha)
        self.precond_refresh_interval = int(precond_refresh_interval)
        self.precond_beta = float(precond_beta)
        self.precond_damping = float(precond_damping)
        self.precond_eps = float(precond_eps)
        self._newton_collecting = False

    def _covariance_refs(self) -> list[dict[str, Any]]:
        refs, seen = [], set()
        for p in self.sorted_params():
            ref = getattr(p, "_newton_covariance_ref", None)
            if ref is not None and id(ref) not in seen:
                refs.append(ref)
                seen.add(id(ref))
        return refs

    @torch.no_grad()
    def prepare_forward(self, step: int) -> None:
        if self.newton_alpha == 0:
            return
        self._newton_collecting = (step + 1) % self.precond_refresh_interval == 0
        for ref in self._covariance_refs():
            ref["collect"] = self._newton_collecting
            if self._newton_collecting:
                ref["accum"].zero_()
                ref["count"].zero_()

    @torch.no_grad()
    def _refresh_preconditioners(self) -> None:
        if not self._newton_collecting:
            return
        for ref in self._covariance_refs():
            dist.all_reduce(ref["accum"])
            dist.all_reduce(ref["count"])
            batch_cov = ref["accum"] / ref["count"].clamp_min(1)
            if "covariance" not in ref:
                ref["covariance"] = torch.eye(
                    batch_cov.shape[-1], device=batch_cov.device,
                    dtype=batch_cov.dtype).expand_as(batch_cov).clone().mul_(0.001)
            ref["covariance"].lerp_(batch_cov, 1 - self.precond_beta)
            cov = (ref["covariance"] + ref["covariance"].mT) * 0.5
            values, vectors = torch.linalg.eigh(cov)
            damping = self.precond_damping * values.mean(dim=-1, keepdim=True)
            powers = (values.clamp_min(0) + damping + self.precond_eps).pow(-self.newton_alpha)
            ref["inverse_power"] = (vectors * powers.unsqueeze(-2)) @ vectors.mT
            ref["collect"] = False
        self._newton_collecting = False

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        if self.newton_alpha == 0:
            return super().compute_polar_input(p, state, group)
        ref = getattr(p, "_newton_covariance_ref", None)
        if ref is None or "inverse_power" not in ref:
            return super().compute_polar_input(p, state, group)
        raw_grad = p.grad
        inv = ref["inverse_power"].to(raw_grad.dtype)
        if ref["blocks"] == 1:
            p.grad = raw_grad @ inv[0]
        else:
            out, total = raw_grad.shape
            blocks, width = ref["blocks"], total // ref["blocks"]
            parts = raw_grad.reshape(out, blocks, width).transpose(0, 1)
            p.grad = torch.bmm(parts, inv).transpose(0, 1).reshape(out, total)
        try:
            return super().compute_polar_input(p, state, group)
        finally:
            p.grad = raw_grad

    @torch.no_grad()
    def step(self) -> None:
        self._refresh_preconditioners()
        super().step()


class NewtonBimaxwellMuon(RightPreconditionedMuonMixin, BimaxwellMuon):
    """Bi-Maxwell Muon with activation right-preconditioning."""
    pass


class NewtonShortEmaMuon(RightPreconditionedMuonMixin, Muon):
    """Short-memory control, initialized from a serialized bi-Maxwell fast stream."""
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95,
                 decay: float = 0.85, **kwargs: Any) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu, **kwargs)
        self.decay = float(decay)
        self._short_update = compile_fixed_decay_kernel(self.decay)

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        if "short_momentum" not in state:
            if "m_fast" not in state:
                raise RuntimeError("short EMA requires a serialized bi-Maxwell m_fast stream")
            state["short_momentum"] = state["m_fast"].clone()
        return self._short_update(p.grad, state["short_momentum"], mu=group["mu"])


class AnnealedDecayMuon(Muon):
    """Single-EMA shape control with beta linearly scheduled after step 1000.

    The applied optimizer is exactly baseline Muon through ``switch_step``.
    Thereafter only the EMA pole moves; the raw-gradient/momentum blend remains
    fixed at ``mu``. This isolates exponential-kernel scheduling from changing
    Nesterov amplitude and from multi-pole expressivity.
    """
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95,
                 beta_start: float = 0.95, beta_end: float = 0.95,
                 switch_step: int = 1000,
                 anneal_end_step: int = 3249) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu)
        assert 0 <= beta_start < 1 and 0 <= beta_end < 1
        assert anneal_end_step > switch_step
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.switch_step = int(switch_step)
        self.anneal_end_step = int(anneal_end_step)

    def interpolation_fraction(self) -> float:
        return min(max((self._muon_steps_seen - self.switch_step) /
                       (self.anneal_end_step - self.switch_step), 0.0), 1.0)

    def current_beta(self) -> float:
        alpha = self.interpolation_fraction()
        return self.beta_start + alpha * (self.beta_end - self.beta_start)

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)
        if self._muon_steps_seen <= self.switch_step:
            return muon_update(p.grad, state["momentum"], mu=group["mu"])
        beta = p.grad.new_tensor(self.current_beta())
        return annealed_decay_update(p.grad, state["momentum"], beta,
                                     mu=group["mu"])


def compile_weighted_decays_kernel(decays: tuple[float, ...],
                                   weights: tuple[float, ...]) -> Callable[..., Tensor]:
    """Compiles the N-decay mixture kernel with rates and weights closure-captured
    as constants (same compile discipline as the bi-Maxwell kernel: passing them
    as tensor arguments would change the compiled kernels and the rounding)."""
    @torch.compile
    def advance_weighted_decays_and_polar(grad: Tensor, streams: list[Tensor],
                                          mu: float = 0.95) -> Tensor:
        for m, decay in zip(streams, decays):
            m.lerp_(grad, 1 - decay)
        m_eff = weights[0] * streams[0]
        for m, w in zip(streams[1:], weights[1:]):
            m_eff = m_eff + w * m
        update = grad.lerp_(m_eff, mu)
        update = zeropower_via_newtonschulz5(update)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update
    return advance_weighted_decays_and_polar


def compile_annealed_decays_kernel(
        decays: tuple[float, ...], start_weights: tuple[float, ...],
        end_weights: tuple[float, ...],
        bias_correct_streams: bool = False) -> Callable[..., Tensor]:
    """Compile a continuously interpolated pair of kernels.

    ``alpha`` is a scalar tensor so Dynamo sees one dynamic input instead of a
    different Python float (and therefore a new specialization) at every step.
    Since both endpoint kernels use the same EMA basis, interpolating their
    coefficients exactly interpolates the realized lag kernels.
    """
    @torch.compile
    def advance_annealed_decays_and_polar(grad: Tensor, streams: list[Tensor],
                                          alpha: Tensor,
                                          finite_masses: Tensor,
                                          mu: float = 0.95) -> Tensor:
        for m, decay in zip(streams, decays):
            m.lerp_(grad, 1 - decay)
        weight = start_weights[0] + alpha * (end_weights[0] - start_weights[0])
        first = streams[0] / finite_masses[0] if bias_correct_streams else streams[0]
        m_eff = weight * first
        for i, m in enumerate(streams[1:], start=1):
            weight = start_weights[i] + alpha * (end_weights[i] - start_weights[i])
            if bias_correct_streams:
                m = m / finite_masses[i]
            m_eff = m_eff + weight * m
        update = grad.lerp_(m_eff, mu)
        update = zeropower_via_newtonschulz5(update)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update
    return advance_annealed_decays_and_polar


def least_squares_momentum_weights(decays: list[float], gamma: float,
                                   window: int = 512,
                                   ridge: float = 1e-10) -> list[float]:
    """Coefficients over exponential buffers whose combined kernel best matches
    the power law w(dt) proportional to dt^(-gamma) on lags 1..window (gamma >= 0
    is the recency-decay exponent; CONVENTION CHANGED 2026-08-25 from the older
    shifted form k^(gamma_old - 1), gamma_old = 1 - gamma), squared error at each
    lag weighted by the target itself.

    A buffer with decay beta weights a gradient k steps old by (1-beta)*beta^k;
    the returned coefficients c_i make sum_i c_i (1-beta_i) beta_i^k track the
    target (audit 2026-08-25: 3-9 percent error in the 100-buffer basis at the default
    ridge, which bounds coefficients to fp32-safe magnitudes; ridge -> 0
    approaches exactness but with +-1e8 coefficients unusable in training).
    Coefficients are generally NOT convex -- roughly half
    the mass is negative. That is benign: every buffer averages the same
    gradient stream, so the update's noise passthrough depends only on the
    realized kernel, not on coefficient magnitudes. gamma = 0 targets the flat
    (uniform) kernel; gamma = 1 targets 1/dt.
    """
    betas = torch.tensor(decays, dtype=torch.float64)
    k = torch.arange(1, window + 1, dtype=torch.float64)
    design = (1 - betas)[None, :] * betas[None, :] ** k[:, None]
    target = k ** (-gamma)
    target = target / target.sum()
    rootw = target.sqrt()
    a = design * rootw[:, None]
    b = target * rootw
    # ridge bounds the coefficients (order <= ~10): the exact minimum-error
    # solution uses +-1e8 cancellations, which are exact in real arithmetic but
    # drown the realized kernel in fp32 rounding when used for training
    m = a.T @ a + ridge * torch.eye(len(decays), dtype=torch.float64)
    return [float(v) for v in torch.linalg.solve(m, a.T @ b)]


def finite_history_power_law_weights(decays: list[float], gamma: float,
                                     horizon: int,
                                     ridge: float = 1e-10,
                                     bias_corrected: bool = True) -> list[float]:
    """Fit a unit-gain power law over the gradient history available at T.

    The optimizer advances each EMA before reading it, so its tap at age ``k``
    is ``(1-beta) * beta**k`` for k=0..T. The target is therefore
    ``(k+1)**(-gamma)`` (the +1 makes the current-gradient tap finite).
    When ``bias_corrected`` (the experiment default), each column is divided by
    its finite-history mass; the equality constraint on coefficients then makes
    deployed DC gain exactly one at that horizon. Without normalization, gamma
    spuriously changes the balance between the raw-gradient and momentum
    branches. This is the endpoint solver used by AnnealedWeightsMuon
    experiments.
    """
    assert gamma >= 0
    assert horizon >= 1
    betas = torch.tensor(decays, dtype=torch.float64)
    k = torch.arange(horizon + 1, dtype=torch.float64)
    design = (1 - betas)[None, :] * betas[None, :] ** k[:, None]
    if bias_corrected:
        design = design / (1 - betas ** (horizon + 1))[None, :]
    target = (k + 1) ** (-gamma)
    target = target / target.sum()
    rootw = target.sqrt()
    a = design * rootw[:, None]
    b = target * rootw
    normal = a.T @ a + ridge * torch.eye(len(decays), dtype=torch.float64)
    rhs = a.T @ b
    # Each normalized EMA has unit DC gain, hence sum(coefficients) == 1.
    q = torch.ones_like(betas)
    kkt = torch.zeros((len(decays) + 1, len(decays) + 1), dtype=torch.float64)
    kkt[:-1, :-1] = normal
    kkt[:-1, -1] = q
    kkt[-1, :-1] = q
    constrained_rhs = torch.cat((rhs, torch.ones(1, dtype=torch.float64)))
    return [float(v) for v in torch.linalg.solve(kkt, constrained_rhs)[:-1]]


class WeightedDecaysMuon(Muon):
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95,
                 decays: list[float] = (0.85, 0.921, 0.96, 0.98),
                 weights: list[float] = (0.25, 0.25, 0.25, 0.25),
                 switch_step: int = 1000,
                 mu_after_switch: float | None = None,
                 log_buffer_gradient_dots_every: int = 0) -> None:
        """Muon whose momentum is a convex combination of N fixed-decay EMA streams.

        mu_after_switch, if set, replaces mu in the post-switch update blend
        (1-mu)*grad + mu*mixture; mu_after_switch = 1.0 removes the Nesterov
        gradient term after the switch while keeping the record's pre-switch
        single-EMA Nesterov phase intact. The pre-switch update always uses mu.

        Protocol matches the record's bi-Maxwell run shape: the single-EMA baseline
        update until switch_step, the N-stream mixture after. Unlike BimaxwellMuon,
        every stream advances from step 0 (they are warm long before any sensible
        switch), so the buffer-gradient dot products are measurable from the start;
        the switch step therefore is not bit-identical to the baseline's step.

        log_buffer_gradient_dots_every > 0: every that-many steps, before the
        streams absorb the new gradient, log sum-over-params <m_i, grad> per
        stream (all-reduced over ranks, printed on rank 0). The incoming
        minibatch's noise is independent of the streams' contents, so these dots
        estimate sum_k w_i(k) C(k) noise-free in expectation; regressing their
        logs against log(mean age) measures the autocovariance tail exponent.
        """
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu)
        assert len(decays) == len(weights)
        # weights need not be convex: L2-solved kernel coefficients are ~half
        # negative; the realized kernel w(k), not the coefficients, carries the
        # normalization and the noise passthrough
        assert all(abs(w) < 1e3 for w in weights)
        self.decays = tuple(float(d) for d in decays)
        self.mixture_weights = tuple(float(w) for w in weights)
        self.switch_step = switch_step
        self.mu_after_switch = None if mu_after_switch is None else float(mu_after_switch)
        self.log_dots_every = log_buffer_gradient_dots_every
        self._advance_and_polar = compile_weighted_decays_kernel(
            self.decays, self.mixture_weights)
        self._dot_accumulator: list[Tensor] | None = None

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        if "streams" not in state:
            state["streams"] = [torch.zeros_like(p) for _ in self.decays]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)
        if self.log_dots_every and self._muon_steps_seen % self.log_dots_every == 0:
            if self._dot_accumulator is None:
                self._dot_accumulator = [
                    torch.zeros((), dtype=torch.float64, device=p.device)
                    for _ in self.decays]
            for acc, m in zip(self._dot_accumulator, state["streams"]):
                acc += (m.double() * p.grad.double()).sum()
        if self._muon_steps_seen >= self.switch_step:
            # streams advance inside the compiled kernel; retire the baseline buffer
            mu = group["mu"] if self.mu_after_switch is None else self.mu_after_switch
            return self._advance_and_polar(p.grad, state["streams"], mu=mu)
        for m, decay in zip(state["streams"], self.decays):
            m.lerp_(p.grad, 1 - decay)
        return muon_update(p.grad, state["momentum"], mu=group["mu"])

    @torch.no_grad()
    def step(self) -> None:
        super().step()
        if self._dot_accumulator is not None:
            stacked = torch.stack(self._dot_accumulator)
            if dist.is_initialized():
                dist.all_reduce(stacked)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                ages = [d / (1 - d) for d in self.decays]
                pairs = " ".join(f"age{a:.1f}:{v:.6e}"
                                 for a, v in zip(ages, stacked.tolist()))
                print(f"bufferdot step:{self._muon_steps_seen - 1} {pairs}",
                      flush=True)
            self._dot_accumulator = None


class NewtonWeightedDecaysMuon(RightPreconditionedMuonMixin, WeightedDecaysMuon):
    """Fixed weighted-decay kernel with activation right-preconditioning."""
    pass


class ScheduledWeightsMuon(WeightedDecaysMuon):
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95,
                 decays: list[float] = (0.85, 0.921, 0.96, 0.98),
                 weight_segments: list[list] = ((1000, (0.25, 0.25, 0.25, 0.25)),),
                 mu_after_switch: float | None = None,
                 log_buffer_gradient_dots_every: int = 0) -> None:
        """WeightedDecaysMuon whose mixture weights change at scheduled steps.

        weight_segments: [[start_step, [w1..wN]], ...] in ascending start order.
        Baseline single-EMA update before the first start_step; from each
        start_step on, that segment's convex weights apply. One compiled kernel
        per segment (weights are closure-baked constants, same compile
        discipline as the record's kernels); a handful of segments means a
        handful of compiles.
        """
        segs = [(int(s), tuple(float(w) for w in ws)) for s, ws in weight_segments]
        assert segs == sorted(segs), "segments must be in ascending step order"
        for _, ws in segs:
            assert len(ws) == len(decays) and abs(sum(ws) - 1.0) < 1e-6
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu,
                         decays=decays, weights=segs[0][1],
                         switch_step=segs[0][0], mu_after_switch=mu_after_switch,
                         log_buffer_gradient_dots_every=log_buffer_gradient_dots_every)
        self._segments = segs
        self._kernels = [compile_weighted_decays_kernel(self.decays, ws)
                         for _, ws in segs]

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        active = -1
        for i, (start, _) in enumerate(self._segments):
            if self._muon_steps_seen >= start:
                active = i
        self._advance_and_polar = self._kernels[active] if active >= 0 else self._kernels[0]
        return super().compute_polar_input(p, state, group)


class AnnealedWeightsMuon(WeightedDecaysMuon):
    """Linearly interpolate between two lag kernels after a baseline phase.

    Muon uses only its ordinary single-decay buffer for updates through
    ``switch_step``. With ``warm_streams_before_switch=True``, the EMA bank is
    nevertheless advanced invisibly from step zero, so the first mixed update
    represents the actual gradient history rather than a cloned 0.95-EMA
    surrogate. Otherwise every stream is lazily initialized from that buffer on
    the switch step, exactly matching bi-Maxwell's no-jump protocol.
    The start kernel applies on the next step and the end kernel applies at
    ``anneal_end_step``. Endpoint weights may be signed, which is required for
    accurate finite-basis power-law approximations.
    """
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95,
                 decays: list[float] = (0.85, 0.921, 0.96, 0.98),
                 start_weights: list[float] = (0.25, 0.25, 0.25, 0.25),
                 end_weights: list[float] = (0.25, 0.25, 0.25, 0.25),
                 switch_step: int = 1000, anneal_end_step: int = 3249,
                 mu_after_switch: float | None = None,
                 warm_streams_before_switch: bool = False,
                 bias_correct_streams: bool = False) -> None:
        assert len(decays) == len(start_weights) == len(end_weights)
        assert anneal_end_step > switch_step
        assert all(abs(w) < 1e3 for w in (*start_weights, *end_weights))
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu,
                         decays=decays, weights=start_weights,
                         switch_step=switch_step, mu_after_switch=mu_after_switch)
        self.start_weights = tuple(float(w) for w in start_weights)
        self.end_weights = tuple(float(w) for w in end_weights)
        self.anneal_end_step = int(anneal_end_step)
        self.warm_streams_before_switch = bool(warm_streams_before_switch)
        self.bias_correct_streams = bool(bias_correct_streams)
        self._advance_annealed_and_polar = compile_annealed_decays_kernel(
            self.decays, self.start_weights, self.end_weights,
            self.bias_correct_streams)

    def interpolation_fraction(self) -> float:
        return min(max((self._muon_steps_seen - self.switch_step) /
                       (self.anneal_end_step - self.switch_step), 0.0), 1.0)

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        if self._muon_steps_seen <= self.switch_step:
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
            if self.warm_streams_before_switch:
                if "streams" not in state:
                    state["streams"] = [torch.zeros_like(p) for _ in self.decays]
                for stream, decay in zip(state["streams"], self.decays):
                    stream.lerp_(p.grad, 1 - decay)
            update = muon_update(p.grad, state["momentum"], mu=group["mu"])
            if self.warm_streams_before_switch:
                return update
            elif self._muon_steps_seen == self.switch_step:
                state["streams"] = [state["momentum"].clone() for _ in self.decays]
            return update
        if "streams" not in state:
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
            update = muon_update(p.grad, state["momentum"], mu=group["mu"])
            state["streams"] = [state["momentum"].clone() for _ in self.decays]
            return update
        mu = group["mu"] if self.mu_after_switch is None else self.mu_after_switch
        alpha = p.grad.new_tensor(self.interpolation_fraction())
        age = self._muon_steps_seen + 1
        finite_masses = p.grad.new_tensor(
            [1 - decay ** age for decay in self.decays])
        return self._advance_annealed_and_polar(
            p.grad, state["streams"], alpha, finite_masses, mu=mu)
