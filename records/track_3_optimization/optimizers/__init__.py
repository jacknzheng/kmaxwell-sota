from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .muon import (AnnealedDecayMuon, AnnealedWeightsMuon, BimaxwellMuon, Muon,
                   NewtonBimaxwellMuon, NewtonShortEmaMuon, NewtonWeightedDecaysMuon,
                   ScheduledWeightsMuon, SgdBlocks, WeightedDecaysMuon)
from .secant_gmres_muon import SecantGmresMuon


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


# Registry name == the YAML name (optimizer_groups[i].optimizer); one entry per
# optimizer, no aliases -- a registry name is part of every reproduction's config,
# so renaming one silently breaks a YAML (lock-tested).
_REGISTRY: dict[str, Any] = {
    "adamw": build_record_adamw,
    "muon": Muon,
    "bimaxwell_muon": BimaxwellMuon,
    "newton_bimaxwell_muon": NewtonBimaxwellMuon,
    "newton_short_ema_muon": NewtonShortEmaMuon,
    "newton_weighted_decays_muon": NewtonWeightedDecaysMuon,
    "secant_gmres_muon": SecantGmresMuon,
    "sgd_blocks": SgdBlocks,
    "weighted_decays_muon": WeightedDecaysMuon,
    "scheduled_weights_muon": ScheduledWeightsMuon,
    "annealed_weights_muon": AnnealedWeightsMuon,
    "annealed_decay_muon": AnnealedDecayMuon,
}


def build_registered_optimizer(name: str, params: list[nn.Parameter], **hyperparams: Any) -> torch.optim.Optimizer:
    """Builds a registered optimizer over params with YAML hyperparams.

    Raises:
        ValueError: On an unregistered name.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown optimizer {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](params, **hyperparams)


def check_optimizer_groups_resolve(config: dict[str, Any]) -> None:
    """Fails at launch, not at step 0: every group's optimizer name must be
    registered, its regex must compile, and its hyperparams must bind against the
    constructor signature.

    Raises:
        ValueError: On a missing optimizer_groups section or unregistered name.
        re.error: On a regex that does not compile.
        TypeError: On hyperparams that do not fit the constructor.
    """
    groups = config.get("optimizer_groups")
    if not groups:
        raise ValueError("config has no optimizer_groups")
    for entry in groups:
        name = entry["optimizer"]
        if name not in _REGISTRY:
            raise ValueError(f"unknown optimizer {name!r}. Available: {sorted(_REGISTRY)}")
        re.compile(entry["pattern"])
        signature = inspect.signature(_REGISTRY[name])
        signature.bind([None], **(entry.get("hyperparams") or {}))


@dataclass(frozen=True)
class GroupSpec:
    """One optimizer_groups YAML entry: which parameters (by name regex), which
    optimizer (registry name), with which hyperparams."""
    pattern: str
    optimizer: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


def parse_group_specs(config_groups: list[dict[str, Any]]) -> list[GroupSpec]:
    """Parses the raw optimizer_groups config list into GroupSpecs."""
    return [GroupSpec(pattern=entry["pattern"], optimizer=entry["optimizer"],
                      hyperparams=dict(entry.get("hyperparams") or {}))
            for entry in config_groups]


class GroupedOptimizers:
    def __init__(self, model: nn.Module, specs: list[GroupSpec]) -> None:
        """Regex-driven hybrid optimizer: each parameter lands in the first group
        whose pattern matches its name; one real torch optimizer per group.

        A pattern that matches nothing raises (a dead YAML entry is a config bug);
        a parameter no pattern matches raises (silent untrained parameters are
        worse). After assembly, coverage is asserted against the model's full
        parameter set -- the record artifacts' own coverage assert, kept.

        Raises:
            ValueError: On an unmatched parameter or a dead pattern.
        """
        compiled = [(spec, re.compile(spec.pattern)) for spec in specs]
        assigned: list[list[nn.Parameter]] = [[] for _ in specs]
        for name, p in model.named_parameters():
            for group_index, (spec, regex) in enumerate(compiled):
                if regex.search(name):
                    assigned[group_index].append(p)
                    break
            else:
                raise ValueError(f"parameter {name!r} matched no optimizer_groups pattern")
        self.groups: list[tuple[GroupSpec, torch.optim.Optimizer]] = []
        for spec, params in zip(specs, assigned):
            if not params:
                raise ValueError(f"optimizer_groups pattern {spec.pattern!r} matched no parameters")
            built = build_registered_optimizer(spec.optimizer, params, **spec.hyperparams)
            for param_group in built.param_groups:
                param_group["initial_lr"] = param_group["lr"]
            self.groups.append((spec, built))
        covered = set(p for _, built in self.groups
                      for param_group in built.param_groups for p in param_group["params"])
        assert covered == set(model.parameters()), "optimizer groups do not cover the model"

    def apply_updates(self) -> None:
        """Steps every group's optimizer, in config order."""
        for _, built in self.groups:
            built.step()

    def prepare_forward(self, step: int) -> None:
        """Let optimizers enable measurements needed by the coming forward pass."""
        for _, built in self.groups:
            prepare = getattr(built, "prepare_forward", None)
            if prepare is not None:
                prepare(step)

    def scale_learning_rates(self, eta: float) -> None:
        """The single schedule lever: every group's lr becomes its construction-time
        base lr times eta -- exactly the record's initial_lr * eta convention."""
        for _, built in self.groups:
            for param_group in built.param_groups:
                param_group["lr"] = param_group["initial_lr"] * eta


def find_muon_family_group(grouped_optimizers: GroupedOptimizers) -> tuple[GroupSpec, Muon]:
    """The single Muon-family group inside a GroupedOptimizers, with its spec.

    The recorder hooks use this seam to reach the sorted parameter list (so their
    sorted_index and rank ownership match the optimizer's exactly) and the spec
    (for dump provenance).

    Raises:
        ValueError: When there is not exactly one Muon-family group.
    """
    matches = [(spec, built) for spec, built in grouped_optimizers.groups
               if isinstance(built, (Muon, SgdBlocks))]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Muon-family optimizer group, found {len(matches)}")
    return matches[0]
