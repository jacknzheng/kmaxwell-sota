import json
import hashlib
import re
from dataclasses import dataclass
import torch
from torch import nn

from records.track_3_optimization.optimizers import Muon, MuonH, SOAP
from records.track_3_optimization.optimizers.ringed.momentum.config import momentum_scheduler
from records.track_3_optimization.optimizers.ringless_reference.record_auxiliary import BiasCorrectionFreeAdam, EMANesterov
from records.track_3_optimization.optimizers.ringless_reference.soap_record import RecordSoapMuon, RecordSoapConfig



def build_record_adamw(params: list[nn.Parameter], *, lr: float, weight_decay: float,
                       betas: tuple[float, float] = (0.8, 0.95), eps: float = 1e-10,
                       fused: bool = True) -> torch.optim.AdamW:
    """The record's auxiliary AdamW: its non-default betas/eps baked in as defaults,
    so a YAML entry only states lr and weight_decay.

    One instance is built per regex group instead of the anchor's one-instance,
    three-group AdamW; that is bitwise-safe because AdamW is elementwise per
    parameter and every parameter keeps identical hyperparameters and step counts.
    fused falls back to False off-CUDA so CPU tests can construct groups; every
    real run is on CUDA and keeps the record's fused=True.
    """
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                             betas=tuple(betas), eps=eps,
                             fused=fused and torch.cuda.is_available())


def scheduler_config_sha256(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_optimizer(name, params, *, momentum=None, gradient_momentum=None,
                    basis_momentum=None, spectrum_momentum=None, **hyperparameters):
    if name in ("muon", "muonh"):
        if momentum is None or any(x is not None for x in (gradient_momentum, basis_momentum, spectrum_momentum)):
            raise ValueError("Muon requires exactly one momentum scheduler")
        cls = Muon if name == "muon" else MuonH
        optimizer = cls(params, momentum_scheduler(momentum), **hyperparameters)
        identity = {"momentum": momentum}
    elif name == "soap":
        if momentum is not None or any(x is None for x in (gradient_momentum, basis_momentum, spectrum_momentum)):
            raise ValueError("SOAP requires three explicit momentum schedulers")
        optimizer = SOAP(params, momentum_scheduler(gradient_momentum),
                         momentum_scheduler(basis_momentum), momentum_scheduler(spectrum_momentum), **hyperparameters)
        identity = {"gradient": gradient_momentum, "basis": basis_momentum, "spectrum": spectrum_momentum}
    else:
        raise ValueError(f"unsupported selected optimizer {name!r}")
    # Bind scheduler identity in every checkpoint. A loader checks this value;
    # changing the constructor's scheduler after resume must not happen silently.
    for group in optimizer.param_groups:
        group["scheduler_config_sha256"] = scheduler_config_sha256(identity)
    return optimizer


@dataclass(frozen=True)
class GroupSpec:
    pattern: str
    optimizer: str


class GroupedOptimizers:
    """Selected harness groups; fullmatch routing refuses overlap or omissions.

    Only AdamW auxiliary parameters and the explicit scheduler factory are
    supported. This is not an alias for the historical optimizer registry.
    """
    def __init__(self, model, specifications):
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        used = set()
        self.groups = []
        for row in specifications:
            spec = GroupSpec(row["pattern"], row["optimizer"])
            selected = [(name, p) for name, p in named if re.fullmatch(spec.pattern, name)]
            if not selected or any(name in used for name, _ in selected):
                raise ValueError("optimizer patterns must select disjoint nonempty groups")
            used.update(name for name, _ in selected)
            params = [p for _, p in selected]
            kwargs = dict(row.get("hyperparams", {}))
            if spec.optimizer == "adamw":
                built = build_record_adamw(params, **kwargs)
            elif spec.optimizer == "bias_correction_free_adam":
                built = BiasCorrectionFreeAdam(params, **kwargs)
            elif spec.optimizer == "record_soap":
                configuration = RecordSoapConfig(**kwargs.pop("record_configuration", {}))
                built = RecordSoapMuon(selected, config=configuration, **kwargs)
            else:
                built = build_optimizer(spec.optimizer, params, **kwargs)
            self.groups.append((spec, built))
            for group in built.param_groups:
                group.setdefault("initial_lr", group["lr"])
        if used != {name for name, _ in named}:
            raise ValueError("optimizer specifications omit trainable parameters")
        self.outer = None

    def install_outer_nesterov(self, model, configuration):
        if self.outer is not None:
            raise ValueError("outer optimizer is already installed")
        self.outer = EMANesterov(list(model.parameters()),
                                 [built for _, built in self.groups], **configuration)

    def before_forward(self):
        if self.outer is not None:
            self.outer.nesterov_step()

    def step(self):
        if self.outer is not None:
            self.outer.step()
            return
        for _, built in self.groups:
            built.step()

    def zero_grad(self, set_to_none=True):
        for _, built in self.groups:
            built.zero_grad(set_to_none=set_to_none)

    def scale_learning_rates(self, factor):
        for _, built in self.groups:
            for group in built.param_groups:
                group["lr"] = group["initial_lr"] * factor


def check_optimizer_groups_resolve(config):
    """Preflight the selected factory on meta matrices, before CUDA setup."""
    for row in config['optimizer_groups']:
        re.compile(row['pattern'])
        name = row['optimizer']
        kwargs = dict(row.get('hyperparams', {}))
        parameter = torch.nn.Parameter(torch.empty(2, 2, device='meta'))
        if name == 'adamw':
            build_record_adamw([parameter], **kwargs)
        elif name == 'bias_correction_free_adam':
            BiasCorrectionFreeAdam([parameter], **kwargs)
        elif name == 'record_soap':
            configuration = RecordSoapConfig(**kwargs.pop('record_configuration', {}))
            RecordSoapMuon([('blocks.0.mlp.fc.weight', parameter)], config=configuration, **kwargs)
        else:
            build_optimizer(name, [parameter], **kwargs)
