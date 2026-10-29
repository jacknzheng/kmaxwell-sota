import time

import numpy as np
import torch
import torch.distributed as dist
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from .muon_jvp import frozen_ns_jvp
from .response_decay import decay_shift


class EagerForward(torch.nn.Module):
    """Bypass only torch.compile; keep the model's original forward arithmetic."""

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, *args):
        return self.inner.forward(*args)


class FrozenMuonHessian:
    """Exact reverse-over-reverse HVP over ALL selected parameters together.

    Recomputing the graph per microbatch bounds memory. Retaining selected
    first-gradient graphs is optional; cache lifetime is exactly this probe.
    Loss must return a SUM, and normalization is the GLOBAL token count when
    ranks process disjoint subsets of that loss.
    """

    def __init__(
        self,
        model,
        params,
        buffers,
        names,
        batch,
        microbatch,
        global_tokens=None,
        cache_batches=0,
        backend="reverse",
    ):
        if len(set(names)) != len(names) or not names:
            raise ValueError("nonempty unique Muon names required")
        self.names = tuple(names)
        self.params = {
            n: (
                p.detach().float()
                if p.dtype in (torch.bfloat16, torch.float16)
                else p.detach()
            ).requires_grad_(n in names)
            for n, p in params.items()
        }
        self.p = tuple(self.params[n] for n in names)
        self.buffers = {n: b.detach().clone() for n, b in buffers.items()}
        self.model = EagerForward(model)
        self.batch, self.microbatch = batch, microbatch
        self.tokens = global_tokens or batch[1].numel()
        if microbatch < 1 or self.tokens < batch[1].numel():
            raise ValueError("invalid microbatch/global normalization")
        self.cache_batches = cache_batches
        if backend not in ("reverse", "forward"):
            raise ValueError("invalid exact HVP backend")
        self.backend = backend
        self.cache = {}
        self.cpu_rng = torch.get_rng_state().clone()
        device = self.p[0].device
        self.cuda_rng = (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        )

    def _gradient(self, start):
        x, y = (b[start : start + self.microbatch] for b in self.batch)
        device = self.p[0].device
        rng_devices = [device.index] if device.type == "cuda" else []
        # Same RNG state per call, without advancing caller RNG.
        with torch.random.fork_rng(devices=rng_devices), sdpa_kernel(SDPBackend.MATH):
            torch.set_rng_state(self.cpu_rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state(self.cuda_rng, device)
            loss = functional_call(
                self.model,
                (
                    {"inner." + n: p for n, p in self.params.items()},
                    {"inner." + n: b for n, b in self.buffers.items()},
                ),
                (x, y),
            )
            return torch.autograd.grad(loss, self.p, create_graph=True)

    def __call__(self, direction):
        if len(direction) != len(self.p):
            raise ValueError("one direction per selected matrix required")
        total = [torch.zeros_like(p) for p in self.p]
        for ordinal, start in enumerate(range(0, len(self.batch[0]), self.microbatch)):
            if self.backend == "forward":
                x, y = (b[start : start + self.microbatch] for b in self.batch)

                def loss(current):
                    params = dict(self.params)
                    params.update(zip(self.names, current))
                    return functional_call(
                        self.model,
                        (
                            {"inner." + n: p for n, p in params.items()},
                            {"inner." + n: b for n, b in self.buffers.items()},
                        ),
                        (x, y),
                    )

                device = self.p[0].device
                with (
                    torch.random.fork_rng(
                        devices=[device.index] if device.type == "cuda" else []
                    ),
                    sdpa_kernel(SDPBackend.MATH),
                ):
                    torch.set_rng_state(self.cpu_rng)
                    if self.cuda_rng is not None:
                        torch.cuda.set_rng_state(self.cuda_rng, device)
                    _, h = torch.func.jvp(
                        torch.func.grad(loss),
                        (self.p,),
                        (tuple(v.to(p.dtype) for v, p in zip(direction, self.p)),),
                    )
                for acc, value in zip(total, h):
                    acc.add_(value.detach() / self.tokens)
                del h
                continue
            keep = ordinal < self.cache_batches
            gradient = self.cache.get(start)
            if gradient is None:
                gradient = self._gradient(start)
                if keep:
                    self.cache[start] = gradient
            active = [
                (g, v.to(p.dtype))
                for g, v, p in zip(gradient, direction, self.p)
                if g.requires_grad
            ]
            if active:
                h = torch.autograd.grad(
                    tuple(g for g, _ in active),
                    self.p,
                    grad_outputs=tuple(v for _, v in active),
                    retain_graph=keep,
                    allow_unused=True,
                )
                for acc, value in zip(total, h):
                    if value is not None:
                        acc.add_(value.detach() / self.tokens)
            del gradient
        return tuple(total)

    def close(self):
        self.cache.clear()


class ShardedVector:
    """Canonical rank-owned matrix packing; padding never enters scientific dots."""

    def __init__(self, layouts, rank=0, device="cpu", dtype=torch.float32):
        self.layouts, self.rank = layouts, rank
        self.device, self.dtype = torch.device(device), dtype
        self.sizes = [
            sum(int(np.prod(shape)) for _, shape in group) for group in layouts
        ]
        self.width = max(self.sizes)
        self.world = len(layouts)
        self.names = [n for group in layouts for n, _ in group]
        if len(set(self.names)) != len(self.names):
            raise ValueError("matrix ownership must be unique")

    def pack(self, tensors, all_ranks=False):
        groups = range(self.world) if all_ranks else [self.rank]
        out = torch.zeros(
            (len(groups), self.width), device=self.device, dtype=self.dtype
        )
        for row, rank in enumerate(groups):
            pos = 0
            for name, shape in self.layouts[rank]:
                count = int(np.prod(shape))
                out[row, pos : pos + count].copy_(tensors[name].reshape(-1))
                pos += count
        return out.reshape(-1)

    def unpack(self, flat, all_ranks=False):
        groups = range(self.world) if all_ranks else [self.rank]
        flat = flat.reshape(len(groups), self.width)
        result = {}
        for row, rank in enumerate(groups):
            pos = 0
            for name, shape in self.layouts[rank]:
                count = int(np.prod(shape))
                result[name] = flat[row, pos : pos + count].view(shape)
                pos += count
        return result

    def dot(self, a, b):
        value = torch.sum(
            a[: self.sizes[self.rank]].double() * b[: self.sizes[self.rank]].double()
        )
        if self.world > 1:
            dist.all_reduce(value)
        return value.item()

    def norm(self, v):
        return self.dot(v, v) ** 0.5

    def random(self, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed + 104729 * self.rank)
        v = torch.randn(self.width, device=self.device, dtype=self.dtype, generator=gen)
        v[self.sizes[self.rank] :] = 0
        return v / self.norm(v)


class CoupledResponse:
    def __init__(
        self,
        space,
        hessian,
        momenta,
        *,
        weight_decay=0.0,
        normalization_tokens=1,
        include_weight_decay=True,
    ):
        self.space, self.hessian = space, hessian
        self.jvps = {n: frozen_ns_jvp(m) for n, m in momenta.items()}
        self.scales = {
            n: max(1.0, m.shape[0] / m.shape[1]) ** 0.5 for n, m in momenta.items()
        }
        coefficients = (
            weight_decay
            if isinstance(weight_decay, dict)
            else {n: weight_decay for n in momenta}
        )
        if set(coefficients) != set(momenta):
            raise ValueError("one decay coefficient per owned Muon matrix required")
        self.decay_shifts = {
            n: decay_shift(coefficients[n], normalization_tokens, include_weight_decay)
            for n in momenta
        }
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, local):
        space = self.space
        start = time.perf_counter()
        full = torch.empty(
            space.world * space.width, device=space.device, dtype=space.dtype
        )
        if space.world > 1:
            dist.all_gather_into_tensor(full, local.contiguous())
        else:
            full.copy_(local)
        direction = space.unpack(full, all_ranks=True)
        values = self.hessian(tuple(direction[n] for n in self.hessian.names))
        packed = space.pack(dict(zip(self.hessian.names, values)), all_ranks=True)
        reduced = torch.empty_like(local)
        if space.world > 1 and dist.get_backend() == "nccl":
            dist.reduce_scatter_tensor(reduced, packed)
        elif space.world > 1:
            dist.all_reduce(packed)
            reduced.copy_(packed.view(space.world, space.width)[space.rank])
        else:
            reduced.copy_(packed)
        h = space.unpack(reduced)
        result = space.pack(
            {
                n: self.scales[n] * jvp(h[n]) + self.decay_shifts[n] * direction[n]
                for n, jvp in self.jvps.items()
            }
        )
        if space.device.type == "cuda":
            torch.cuda.synchronize()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return result


class ShardedArnoldi:
    """Incremental, two-pass MGS. FP32 GPU basis, FP64 reductions/small eigensystem.

    Basis is sharded across ranks, never assembled into a complex full-model
    matrix. Explicit residual evaluation is independent of the Hessenberg bound.
    """

    def __init__(self, apply, space, seed, initial=None):
        self.apply, self.space = apply, space
        first = space.random(seed) if initial is None else initial.clone()
        self.q = [first / space.norm(first)]
        self.h = np.zeros((1, 0), dtype=np.float64)
        self.breakdown = False

    def extend(self, dimension):
        while self.h.shape[1] < dimension and not self.breakdown:
            j = self.h.shape[1]
            v = self.apply(self.q[j])
            coeff = np.zeros(j + 1)
            for _ in range(2):
                for i, q in enumerate(self.q):
                    value = self.space.dot(q, v)
                    coeff[i] += value
                    v.add_(q, alpha=-value)
            tail = self.space.norm(v)
            h = np.zeros((j + 2, j + 1))
            h[: j + 1, :j] = self.h
            h[: j + 1, j], h[j + 1, j] = coeff, tail
            self.h = h
            if tail < 1e-12:
                self.breakdown = True
            else:
                self.q.append(v / tail)

    def ritz(self):
        n = self.h.shape[1]
        values, coefficients = np.linalg.eig(self.h[:n, :n])
        positive = [
            i
            for i, z in enumerate(values)
            if z.real > 0 and abs(z.imag) <= 1e-7 * max(1.0, abs(z.real))
        ]
        if not positive:
            return dict(
                mu=None,
                residual=None,
                spectrum=[[float(z.real), float(z.imag)] for z in values],
            ), None
        index = max(positive, key=lambda i: values[i].real)
        vector = torch.zeros_like(self.q[0])
        for q, coefficient in zip(self.q, coefficients[:, index].real):
            vector.add_(q, alpha=float(coefficient))
        vector /= self.space.norm(vector)
        mu = float(values[index].real)
        tv = self.apply(vector)
        residual = self.space.norm(tv - mu * vector) / max(
            self.space.norm(tv) + abs(mu), 1e-30
        )
        return dict(
            mu=mu,
            residual=residual,
            dimension=n,
            spectrum=[[float(z.real), float(z.imag)] for z in values],
        ), vector


def qualify_eigenpair(
    apply,
    space,
    dimensions=(8, 16, 32, 64, 128, 256),
    eigenvalue_rtol=0.001,
    progress=None,
    initial=None,
    checkpoint=None,
    fresh_start=True,
    previous_mu=None,
    jump_rtol=0.2,
):
    """Stop on eigenvalue agreement across dimensions and independent starts.

    Direct residuals remain diagnostics, never a numerical-precision veto.
    Finite positive modes and independent discovery still matter: a stale warm
    eigenvector must not hide a newly larger mode.
    """
    searches, vectors = [], []
    seeds = [921, 1523] if fresh_start or initial is None else [921]
    warm_missed_larger = False
    fresh_start_reason = "scheduled_or_no_warm_start" if len(seeds) == 2 else None
    for seed in seeds:
        state = ShardedArnoldi(apply, space, seed, initial if seed == 921 else None)
        history, previous, accepted = [], None, False
        for dimension in dimensions:
            state.extend(dimension)
            row, vector = state.ritz()
            change = (
                abs(row["mu"] - previous["mu"]) / abs(row["mu"])
                if row["mu"] and previous and previous["mu"]
                else None
            )
            row.update(seed=seed, relative_change=change)
            history.append(row)
            if checkpoint is not None and vector is not None:
                checkpoint(row, vector)
            if progress:
                progress(row)
            if (
                row["mu"]
                and np.isfinite(row["mu"])
                and previous
                and change is not None
                and change <= eigenvalue_rtol
            ):
                accepted = True
                break
            if state.breakdown:
                accepted = row["mu"] is not None and np.isfinite(row["mu"])
                break
            previous = row
        searches.append(dict(qualified=accepted, history=history))
        vectors.append(vector)
        del state
        if space.device.type == "cuda":
            torch.cuda.empty_cache()
        if len(searches) == 1 and len(seeds) == 1:
            mu = history[-1]["mu"]
            jumped = (
                mu
                and previous_mu
                and abs(mu - previous_mu) / abs(previous_mu) > jump_rtol
            )
            if not accepted or jumped:
                seeds.append(1523)
                fresh_start_reason = (
                    "warm_search_not_converged" if not accepted else "mu_jump"
                )
        if len(searches) == 2 and initial is not None:
            a, b = (s["history"][-1]["mu"] for s in searches)
            if a and b and b > a * (1 + eigenvalue_rtol) and searches[1]["qualified"]:
                # A converged old mode need not still lead. Confirm the newly
                # larger mode with another genuinely fresh search, not with a
                # clone of its just-discovered eigenvector.
                warm_missed_larger = True
                seeds.append(2027)
    values = [s["history"][-1]["mu"] for s in searches]
    compared = searches[-2:] if warm_missed_larger else searches
    compared_values = [s["history"][-1]["mu"] for s in compared]
    spread = (
        (max(compared_values) - min(compared_values)) / max(compared_values)
        if all(compared_values)
        else None
    )
    independent = len(searches) > 1
    if not independent:
        spread = None  # No fabricated independent-start agreement.
    good = all(s["qualified"] for s in compared) and (
        not independent or (spread is not None and spread <= eigenvalue_rtol)
    )
    winner = max(range(len(values)), key=lambda i: values[i] or float("-inf"))
    return dict(
        qualified=good,
        searches=searches,
        seed_relative_spread=spread,
        residual_gating=False,
        stopping_rule=(
            "eigenvalue_dimension_and_start_agreement"
            if independent
            else "eigenvalue_dimension_agreement_warm_only"
        ),
        independent_start_checked=independent,
        fresh_start_reason=fresh_start_reason,
        warm_start_missed_larger_mode=warm_missed_larger,
        mu=values[winner],
    ), vectors[winner]
