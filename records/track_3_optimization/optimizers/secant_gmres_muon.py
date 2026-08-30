from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from secant_gmres_solver.solve import solve_secant_least_squares
from secant_gmres_solver.streams import (GRAM_CHUNK, accumulate_secant_gram,
                                         advance_or_first_touch, build_decay_ladder,
                                         build_one_minus_betas, secant_direction)

from .muon import Muon, owned_param_indices, zeropower_via_newtonschulz5


@torch.compile
def polar_with_shape_scale(d: Tensor) -> Tensor:
    """Polar map + the record's aspect-ratio scale, for an externally formed direction."""
    update = zeropower_via_newtonschulz5(d)
    update *= max(1, d.size(-2) / d.size(-1))**0.5
    return update


@torch.compile
def nesterov_mix_then_polar(grad: Tensor, momentum: Tensor, mu: float) -> Tensor:
    """The baseline path with momentum already advanced by pass 1: same arithmetic
    as the record's muon_update, without re-advancing or mutating the gradient."""
    return polar_with_shape_scale(torch.lerp(grad, momentum, mu))


class SecantGmresMuon(Muon):
    def __init__(self, params: list[torch.nn.Parameter], lr: float = 0.02,
                 weight_decay: float = 0, mu: float = 0.95, num_buffers: int = 100,
                 min_decay: float = 0.8, max_decay: float = 0.995,
                 pin_exact_rates: bool = True,
                 truncated_rank_accumulated_parameter_displacements: int = 32,
                 truncated_rank_gmres: int = 8, solve_start: int = 1000) -> None:
        """Muon whose pre-polar input is the secant restricted-Newton direction.

        Two-pass step. Pass 1 walks this rank's owned parameters at (x_t, g_t) --
        before any parameter changes -- advancing the paired EMA streams and the
        baseline momentum, and accumulating this rank's share of the secant Gram.
        One tiny fp64 all_reduce closes the Gram; every rank then solves the
        identical CPU system, so the mixing weights w agree bitwise across ranks.
        Pass 2 is the record's comm loop with pre-polar input -S w, or the
        baseline Nesterov path before solve_start and whenever the solve is
        degenerate or gives an ascent displacement (delta.g >= 0) -- fallbacks are
        counted and reported in last_solve_statistics.

        Args:
            params: The Muon-managed parameters (2D block weights).
            lr: Learning rate (scaled externally by the schedule).
            weight_decay: Decoupled weight decay.
            mu: Nesterov momentum rate of the fallback/baseline path.
            num_buffers: k, the number of paired EMA streams.
            min_decay: Smallest ladder rate.
            max_decay: Largest ladder rate.
            pin_exact_rates: Keep 0.85 and 0.98 exact in the ladder.
            truncated_rank_accumulated_parameter_displacements: r1 of the solve.
            truncated_rank_gmres: r2 of the solve.
            solve_start: First Muon step driven by the solve.
        """
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu)
        self.num_buffers = num_buffers
        self.truncated_rank_accumulated_parameter_displacements = \
            truncated_rank_accumulated_parameter_displacements
        self.truncated_rank_gmres = truncated_rank_gmres
        self.solve_start = solve_start
        self.ladder = build_decay_ladder(num_buffers, min_decay, max_decay,
                                         (0.85, 0.98) if pin_exact_rates else None)
        self._one_minus_betas: Tensor | None = None   # (k, 1) fp32 on device, set lazily
        self._current_w: Tensor | None = None         # this step's mixing weights
        self.fallback_count = 0
        self.last_solve_statistics: dict[str, Any] | None = None

    @torch.no_grad()
    def _advance_streams_and_solve(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        group = self.param_groups[0]
        params = group["params"]
        device = params[0].device
        if self._one_minus_betas is None:
            self._one_minus_betas = build_one_minus_betas(self.ladder, device)
        solving = self._muon_steps_seen >= self.solve_start
        G = torch.zeros(2 * self.num_buffers + 1, 2 * self.num_buffers + 1,
                        dtype=torch.float64, device=device) if solving else None
        for idx in owned_param_indices(len(params), rank, world_size):
            p = params[idx]
            state = self.state[p]
            g_flat = p.grad.flatten()
            x_flat = p.detach().flatten()
            advance_or_first_touch(state, g_flat, x_flat, self._one_minus_betas, self.num_buffers)
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
            state["momentum"].lerp_(p.grad, 1 - group["mu"])
            if solving:
                accumulate_secant_gram(G, state["grad_avgs"], state["param_avgs"],
                                       g_flat, x_flat, GRAM_CHUNK)
        self._current_w = None
        if solving:
            dist.all_reduce(G)
            w, stats = solve_secant_least_squares(
                G, self.num_buffers,
                self.truncated_rank_accumulated_parameter_displacements,
                self.truncated_rank_gmres)
            if w is not None and stats["delta_dot_g"] < 0:
                self._current_w = w
            else:
                self.fallback_count += 1
                if stats["fallback_reason"] is None:
                    stats["fallback_reason"] = "ascent_displacement"
            stats["muon_step"] = self._muon_steps_seen
            stats["fallback_count"] = self.fallback_count
            self.last_solve_statistics = stats

    def compute_polar_input(self, p: torch.nn.Parameter, state: dict[str, Any],
                            group: dict[str, Any]) -> Tensor:
        if self._current_w is not None:
            d = secant_direction(state["param_avgs"], p.detach().flatten(),
                                 self._current_w, GRAM_CHUNK).view_as(p)
            return polar_with_shape_scale(d)
        return nesterov_mix_then_polar(p.grad, state["momentum"], group["mu"])

    @torch.no_grad()
    def step(self) -> None:
        self._advance_streams_and_solve()
        super().step()
