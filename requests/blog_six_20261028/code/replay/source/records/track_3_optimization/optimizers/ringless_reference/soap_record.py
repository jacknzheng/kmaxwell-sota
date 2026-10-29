from dataclasses import dataclass
import torch
import torch.distributed as dist
from records.track_3_optimization.optimizers.ringless_reference.soap_record_math import (
    gram_frobenius_norm_estimate, zeropower_via_newtonschulz5,
    tensor_cosine, norm_preserving_blend, target_radius_after_update,
    rescale_to_radius, soap_eigenbasis, soap_basis_qr)


@dataclass(frozen=True)
class RecordSoapConfig:
    basis_beta: float
    spectrum_beta: float
    basis_frequency: int
    denominator_power: float
    trust_floor: float
    trust_cap: float
    trust_floor_end: int
    trust_fade_end: int
    trust_min_agreement: float
    trust_min_gradient_alignment: float
    trust_power: float
    outward_scale: float
    inward_scale: float
    target_update_weight_ratio: float
    row_floor: bool
    row_floor_power: float
    cautious_decay: float
    maxwell_start: int
    maxwell_fast_beta: float
    maxwell_slow_beta: float
    maxwell_fast_weight: float

def recursive_momentum(state, gradient, mu, step, config):
    """Lazy switch copies CURRENT scheduled momentum; no extra lerp at switch."""
    state["momentum"].lerp_(gradient, 1 - mu)
    result = gradient.lerp(state["momentum"], mu)
    if step >= config.maxwell_start:
        fast = state.get("bm_mf")
        if fast is None:
            state["bm_mf"] = state["momentum"].detach().clone()
            state["bm_ms"] = state["momentum"].detach().clone()
        else:
            fast.lerp_(gradient, 1 - config.maxwell_fast_beta)
            state["bm_ms"].lerp_(gradient, 1 - config.maxwell_slow_beta)
        effective = torch.lerp(state["bm_ms"], state["bm_mf"], config.maxwell_fast_weight)
        result = gradient.lerp(effective, mu)
    return result


def initialize_soap_state(parameter, state):
    state.update(momentum=torch.zeros_like(parameter),
                 exp_avg_sq=torch.zeros_like(parameter, dtype=torch.float32),
                 row_gg=torch.zeros(parameter.size(0), parameter.size(0), device=parameter.device),
                 col_gg=torch.zeros(parameter.size(1), parameter.size(1), device=parameter.device),
                 q_row=None, q_col=None, soap_step=0)


def precondition_momentum(update, state, config, eps=1e-8):
    if state["q_row"] is None:
        return update
    raw = update.float()
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ raw @ q_col
    state["exp_avg_sq"].mul_(config.spectrum_beta).add_(projected.square(), alpha=1-config.spectrum_beta)
    denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(config.denominator_power)
    transformed = q_row @ (projected / denom) @ q_col.T
    transformed.mul_(gram_frobenius_norm_estimate(raw, eps=eps) /
                     gram_frobenius_norm_estimate(transformed, eps=eps))
    return transformed.to(update.dtype)


def update_preconditioner(gradient, state, config):
    raw = gradient.float()
    state["row_gg"].lerp_(raw @ raw.T, 1-config.basis_beta)
    state["col_gg"].lerp_(raw.T @ raw, 1-config.basis_beta)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % config.basis_frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"], state["col_gg"], state["q_row"], state["q_col"], state["exp_avg_sq"])
    state["soap_step"] += 1


def bounded_trust(raw, soap, gradient, step, config, eps=1e-8):
    raw_grad = tensor_cosine(raw, gradient, eps)
    soap_grad = tensor_cosine(soap, gradient, eps)
    soap_raw = tensor_cosine(soap, raw, eps)
    agreement = ((soap_raw-config.trust_min_agreement)/(1-config.trust_min_agreement)).clamp(0, 1)
    denominator = (raw_grad-config.trust_min_gradient_alignment).clamp_min(eps)
    gradient_gate = ((soap_grad-config.trust_min_gradient_alignment)/denominator).clamp(0, 1)
    gate = (agreement*gradient_gate).clamp(0, 1)
    if config.trust_power != 1.:
        gate = gate.pow(config.trust_power)
    if config.trust_fade_end <= config.trust_floor_end:
        floor = 0. if step >= config.trust_fade_end else config.trust_floor
    elif step < config.trust_floor_end:
        floor = config.trust_floor
    elif step >= config.trust_fade_end:
        floor = 0.
    else:
        floor = config.trust_floor*(config.trust_fade_end-step)/(config.trust_fade_end-config.trust_floor_end)
    cap = config.trust_cap if step < config.trust_fade_end else 1.
    return gate.clamp(min=floor, max=cap)


def radius_pinned_update(parameter, update, lr, config):
    raw, weight = update.float(), parameter.float()
    coefficient = (raw*weight).sum()/(weight*weight).sum().clamp_min(1e-12)
    radial = coefficient*weight
    scale = torch.where(coefficient < 0, raw.new_tensor(config.outward_scale),
                        raw.new_tensor(config.inward_scale))
    update = (raw-radial+scale*radial).to(update.dtype)
    if config.row_floor and parameter.ndim == 2:
        radius = weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        length = update.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
        factor = torch.clamp(config.target_update_weight_ratio*radius/length, min=1.).pow(config.row_floor_power)
        update = (update.float()*factor).to(update.dtype)
    else:
        radius, length = weight.norm().clamp_min(1e-8), update.float().norm().clamp_min(1e-8)
        factor = torch.where(length/radius < config.target_update_weight_ratio,
                             config.target_update_weight_ratio*radius/length, torch.ones_like(radius))
        update = update*factor.to(update.dtype)
    target = target_radius_after_update(parameter, update, lr)
    if config.cautious_decay > 0 and parameter.ndim == 2:
        mask = (update.float()*parameter.float() > 0).to(parameter.dtype)
    parameter.add_(update, alpha=-lr)
    rescale_to_radius(parameter, target)
    if config.cautious_decay > 0 and parameter.ndim == 2:
        parameter.mul_(1.-(lr*config.cautious_decay)*mask)


class RecordSoapMuon(torch.optim.Optimizer):
    """Distributed hidden matrices with recursive QR SOAP, not the ring prototype."""
    def __init__(self, named_parameters, *, lr, mu, config):
        named_parameters = list(named_parameters)
        if not named_parameters or any(p.ndim != 2 for _, p in named_parameters):
            raise ValueError("Record SOAP requires nonempty named matrices")
        self.attention = {p for name, p in named_parameters if ".attn." in name}
        self.attention_projection = {p for name, p in named_parameters if name.endswith(".attn.proj.weight")}
        self.configuration = config
        self.step_count = 0
        parameters = sorted([p for _, p in named_parameters], key=lambda p: p.size(), reverse=True)
        super().__init__(parameters, dict(lr=lr, mu=mu, weight_decay=0.))

    def state_dict(self):
        result = super().state_dict()
        result["record_step_count"] = self.step_count
        result["record_configuration"] = self.configuration.__dict__
        return result

    def load_state_dict(self, state_dict):
        if state_dict["record_configuration"] != self.configuration.__dict__:
            raise ValueError("Record SOAP configuration changed on resume")
        super().load_state_dict(state_dict)
        # Optimizer's generic loader casts floating state to parameter dtype.
        # SOAP bases/Grams/variance must retain the ORIGINAL FP32 values; casting
        # the already-loaded BF16 tensors back would silently lose precision.
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
                saved = state_dict["state"].get(saved_id, {})
                for key in ("exp_avg_sq", "row_gg", "col_gg", "q_row", "q_col"):
                    if isinstance(saved.get(key), torch.Tensor):
                        self.state[parameter][key] = saved[key].to(
                            device=parameter.device, dtype=torch.float32).clone()
        self.step_count = state_dict["record_step_count"]

    @torch.no_grad()
    def step(self):
        world, rank = dist.get_world_size(), dist.get_rank()
        config = self.configuration
        for group in self.param_groups:
            parameters = group["params"]
            padded = parameters+[torch.empty_like(parameters[-1])]*(world-len(parameters)%world)
            for base in range(0, len(parameters), world):
                if base+rank < len(parameters):
                    parameter = parameters[base+rank]
                    gradient, state = parameter.grad, self.state[parameter]
                    if gradient is None:
                        raise ValueError("Record SOAP expects a gradient for every selected matrix")
                    if not state:
                        initialize_soap_state(parameter, state)
                    momentum = recursive_momentum(state, gradient, group["mu"], self.step_count, config)
                    transformed = precondition_momentum(momentum, state, config)
                    if parameter in self.attention:
                        gate = (bounded_trust(momentum, transformed, gradient, self.step_count, config)
                                if parameter in self.attention_projection else
                                torch.ones((), dtype=torch.float32, device=parameter.device))
                        momentum = norm_preserving_blend(momentum, transformed, gate)
                    else:
                        momentum = transformed
                    update = zeropower_via_newtonschulz5(momentum)
                    update *= max(1, update.size(-2)/update.size(-1))**.5
                    radius_pinned_update(parameter, update, group["lr"], config)
                    update_preconditioner(gradient, state, config)
                dist.all_gather(padded[base:base+world], padded[base+rank])
        self.step_count += 1
