"""Matrix-free frozen local Muon response, not a momentum stability operator."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn

from records.track_3_optimization.optimizers.ringed.muon import zeropower_via_newtonschulz5
from .response_decay import decay_shift


def frozen_ns_jvp(momentum: Tensor):
    """Linear tangent propagation through the actual twelve BF16 NS primals.

    Record the optimizer's rounded primal intermediates exactly, then propagate
    tangents in FP32 (FP64 for FP64 input). Cast/rounding use the usual identity
    AD convention, but do NOT round the tangent at every primitive. Native BF16
    forward AD was measurably non-additive and is unsuitable as an Arnoldi map.
    This is the NS derivative, not ideal polar or a different NS polynomial.
    """
    tall = momentum.shape[0] > momentum.shape[1]
    g = momentum.detach().bfloat16()
    if tall:
        g = g.mT
    norm = g.norm(dim=(-2, -1), keepdim=True)
    denominator = norm + 1e-7
    if not bool(norm > 0):
        raise ValueError("NS norm derivative is undefined at zero momentum")
    x = g / denominator
    primals = []
    for _ in range(12):
        a = x @ x.mT
        b = -1.5 * a + .5 * a @ a
        primals.append((x, a, b))
        x = 2 * x + b @ x

    def jvp(tangent):
        dtype = torch.float64 if tangent.dtype == torch.float64 else torch.float32
        e = tangent.to(dtype)
        if tall:
            e = e.mT
        gg, nn, dd = g.to(dtype), norm.to(dtype), denominator.to(dtype)
        # Norm's identity-rounding AD derivative uses the unrounded real norm.
        # The denominator's PRIMAL remains the exact recorded BF16 value.
        dnorm = (gg * e).sum() / gg.norm()
        dx = e / dd - gg * dnorm / dd.square()
        for x, a, b in primals:
            xx, aa, bb = x.to(dtype), a.to(dtype), b.to(dtype)
            da = dx @ xx.mT + xx @ dx.mT
            db = -1.5 * da + .5 * (da @ aa + aa @ da)
            dx = 2 * dx + db @ xx + bb @ dx
        return dx.mT if tall else dx
    return jvp


def ideal_polar_jvp(momentum: Tensor, tangent: Tensor) -> Tensor:
    """Exact full-rank rectangular polar derivative; refuses rank singularities.

    Thin SVD formula avoids differentiating singular vectors at repeated values.
    This is ideal msign, NOT the BF16 Newton--Schulz optimizer map.
    """
    if momentum.shape[0] > momentum.shape[1]:
        return ideal_polar_jvp(momentum.mT, tangent.mT).mT
    u, sigma, vh = torch.linalg.svd(momentum.double(), full_matrices=False)
    floor = torch.finfo(sigma.dtype).eps * max(momentum.shape) * sigma.max()
    if not bool((sigma > floor).all()):
        raise ValueError("ideal polar derivative requires full rank")
    e = tangent.double()
    uv = u.mT @ e @ vh.mT
    omega = (uv - uv.mT) / (sigma[:, None] + sigma[None, :])
    transverse = u.mT @ e - uv @ vh
    return u @ (omega @ vh + transverse / sigma[:, None])


def local_response(hvp: Callable[[Tensor], Tensor], momentum: Tensor,
                   mode: str = "newton_schulz", *, weight_decay=0.,
                   normalization_tokens=1, include_weight_decay=True) -> Callable[[Tensor], Tensor]:
    """T[v] = shape_multiplier * Dmsign(M)[H[v]] + decay_shift * v.

    Decay is included by default; disable explicitly for the original operator.
    For a mean-loss HVP of sum-gradient training, normalization_tokens=N_train.
    No learning rate, lag-0 tap, or derivative of the history/kernel.
    NS uses the identity rounding AD convention through actual BF16 primals;
    a floating-point program is not literally differentiable under quantization.
    """
    if momentum.ndim != 2 or not torch.isfinite(momentum).all():
        raise ValueError("momentum must be a finite matrix")
    if mode not in ("newton_schulz", "ideal_polar"):
        raise ValueError("unknown msign derivative mode")
    m = momentum.detach().clone()
    shift = decay_shift(weight_decay, normalization_tokens, include_weight_decay)
    scale = max(1, m.shape[0] / m.shape[1]) ** .5
    if mode == "newton_schulz" and not bool(m.norm() > 0):
        raise ValueError("NS norm derivative is undefined at zero momentum")
    ns_jvp = frozen_ns_jvp(m) if mode == "newton_schulz" else None

    def apply(v):
        if v.shape != m.shape or not torch.isfinite(v).all():
            raise ValueError("invalid response direction")
        h = hvp(v).detach()
        if h.shape != m.shape or not torch.isfinite(h).all():
            raise ValueError("invalid Hessian-vector product")
        with torch.enable_grad():
            if mode == "ideal_polar":
                j = ideal_polar_jvp(m, h)
            else:
                # Forward JVP, not VJP: preserve the linear tangent in FP32.
                j = ns_jvp(h)
        result = scale * j.detach().double() + shift * v.detach().double()
        if not torch.isfinite(result).all():
            raise ValueError("non-finite local response")
        return result
    return apply


class ArnoldiBasis:
    """Extend a single current-operator Krylov basis; never reuse old matvecs."""
    def __init__(self, apply, shape, device, capacity, seed, initial_vector=None):
        self.apply, self.shape = apply, shape
        n = math.prod(shape)
        generator = torch.Generator(device=device).manual_seed(seed)
        if initial_vector is None:
            q = torch.randn(n, generator=generator, device=device, dtype=torch.float64)
        else:
            if tuple(initial_vector.shape) != tuple(shape):
                raise ValueError("invalid warm-start shape")
            if initial_vector.is_complex():
                if float(initial_vector.imag.norm()) > 1e-7 * float(initial_vector.real.norm()):
                    raise ValueError("warm start must be a real eigenvector")
                initial_vector = initial_vector.real
            q = initial_vector.detach().to(device=device, dtype=torch.float64).flatten().clone()
            if not torch.isfinite(q).all() or float(q.norm()) == 0:
                raise ValueError("invalid warm-start vector")
        self.q = q / q.norm()
        self.basis = []
        self.h = torch.zeros(min(n, capacity) + 1, min(n, capacity),
                             device=device, dtype=torch.float64)
        self.breakdown = False
        self.calls = 0

    def extend(self, dimension):
        k = min(math.prod(self.shape), dimension)
        if k > self.h.shape[1]:
            raise ValueError("Arnoldi basis capacity exceeded")
        while len(self.basis) < k and not self.breakdown:
            j = len(self.basis)
            self.basis.append(self.q)
            w = self.apply(self.q.view(self.shape)).reshape(-1).double()
            self.calls += 1
            if w.numel() != self.q.numel() or not torch.isfinite(w).all():
                raise ValueError("invalid Arnoldi matvec")
            original_norm = float(w.norm())
            for _ in range(2):
                for i, qi in enumerate(self.basis):
                    alpha = torch.dot(qi, w)
                    self.h[i, j] += alpha
                    w = w - alpha * qi
            beta = w.norm()
            self.h[j + 1, j] = beta
            self.breakdown = float(beta) <= 1e-12 * max(1e-300, original_norm)
            if not self.breakdown:
                self.q = w / beta


def arnoldi_response(apply: Callable[[Tensor], Tensor], shape: tuple[int, int], *,
                     device="cpu", krylov_dim=32, seed=0, residual_tol=.02,
                     real_tol=1e-7, substantial_fraction=.1, initial_vector=None,
                     _basis=None, _verify_spectrum=True) -> tuple[dict, Tensor | None]:
    """Two-pass MGS Arnoldi; direct residuals for selected real/complex Ritz pairs.

    Bounded Krylov search is not a certificate of the largest eigenvalue or of
    absence of a positive eigenvalue. Inconclusive results stay explicitly so.
    All projected eigenvalues are retained; selected eigenvectors are rechecked.
    residual_tol is a legacy diagnostic reference, not an acceptance requirement.
    """
    n = math.prod(shape)
    if len(shape) != 2 or min(shape) < 1 or type(krylov_dim) is not int or krylov_dim < 1:
        raise ValueError("invalid Arnoldi shape/dimension")
    if not (0 < residual_tol < 1 and 0 <= real_tol < 1 and 0 <= substantial_fraction <= 1):
        raise ValueError("invalid Arnoldi tolerances")
    state = _basis or ArnoldiBasis(apply, shape, device, krylov_dim, seed, initial_vector)
    before_calls = state.calls
    state.extend(krylov_dim)
    basis, h, breakdown = state.basis, state.h, state.breakdown
    calls = state.calls - before_calls
    count = len(basis)
    values, small_vectors = torch.linalg.eig(h[:count, :count])
    positive = [i for i, v in enumerate(values) if float(v.real) > 0 and
                abs(float(v.imag)) <= real_tol * max(1e-300, abs(float(v.real)))]
    leading = max(positive, key=lambda i: float(values[i].real)) if positive else None
    radius = float(values.abs().max())
    selected = set(sorted(range(count), key=lambda i: float(values[i].real), reverse=True)[:3])
    selected.update(i for i, v in enumerate(values) if float(v.real) < -substantial_fraction * radius)
    selected.update(i for i, v in enumerate(values) if abs(float(v.imag)) > substantial_fraction * radius)
    if leading is not None:
        selected.add(leading)
    if not _verify_spectrum:
        # Intermediate checks still reapply the CURRENT operator to the leading
        # vector. Full negative/complex residual diagnostics are deferred once.
        selected = {leading} if leading is not None else set()
    b = torch.stack(basis, dim=1).to(torch.complex128)
    diagnostics = []
    leading_vector = None
    leading_residual = None
    for i in sorted(selected, key=lambda i: float(values[i].real), reverse=True):
        eigenvalue = values[i]
        e = b @ small_vectors[:, i]
        e /= e.norm()
        ae = apply(e.real.view(shape)).flatten().to(torch.complex128); calls += 1
        if bool(e.imag.abs().max() > 1e-12):
            ae += 1j * apply(e.imag.view(shape)).flatten(); calls += 1
        absolute = float((ae - eigenvalue * e).norm())
        relative = absolute / max(float(ae.norm()) + float(eigenvalue.abs()), 1e-300)
        diagnostics.append(dict(index=i, real=float(eigenvalue.real), imag=float(eigenvalue.imag),
                                residual_absolute=absolute, residual_relative=relative,
                                residual_pass=relative <= residual_tol))
        if i == leading:
            leading_vector = e.view(shape).detach().cpu()
            leading_residual = relative
    accepted = leading is not None and math.isfinite(float(values[leading].real))
    summary = dict(
        schema="local_muon_response_v1", status=("positive_ritz_estimate" if accepted else
                 "no_positive_real_ritz_found"),
        mu_estimate=float(values[leading].real) if accepted else None,
        leading_positive_candidate=float(values[leading].real) if leading is not None else None,
        leading_positive_residual=leading_residual, residual_tolerance=residual_tol,
        residual_gating=False,
        real_tolerance=real_tol, substantial_fraction=substantial_fraction,
        largest_positive_certified=False, absence_of_positive_certified=False,
        krylov_dimension=count, requested_krylov_dimension=krylov_dim,
        invariant_subspace_breakdown=breakdown, matvec_calls=calls, seed=seed,
        basis_matvec_calls=state.calls, warm_start=initial_vector is not None,
        spectrum_residuals_verified=_verify_spectrum,
        ritz_values=[dict(real=float(v.real), imag=float(v.imag)) for v in values],
        most_negative_real_ritz=min(0., float(values.real.min())),
        max_abs_imag_ritz=float(values.imag.abs().max()),
        eigenpairs=diagnostics,
        orthogonality_error=float((b.mH @ b - torch.eye(count, device=device)).norm()))
    return summary, leading_vector


def converged_response(apply, shape, *, device="cpu", krylov_dims=(16, 32, 64),
                       starts=2, seed=0, eigenvalue_rtol=.01, residual_tol=.02,
                       real_tol=1e-7, substantial_fraction=.1, initial_vector=None):
    """Require agreement across dimensions and an independent random search.

    This is empirical convergence, not proof that an unobserved larger positive
    eigenvalue does not exist. Residuals are diagnostics, not stopping criteria.
    Every attempt is retained, including failures and competing Ritz values.
    """
    if starts < 2 or not 0 < eigenvalue_rtol < 1:
        raise ValueError("convergence requires multiple starts and a valid tolerance")
    if len(krylov_dims) < 2 or any(type(k) is not int or k < 1 for k in krylov_dims):
        raise ValueError("convergence requires at least two positive Krylov dimensions")
    if any(a >= b for a, b in zip(krylov_dims, krylov_dims[1:])):
        raise ValueError("Krylov dimensions must increase")
    history, previous, chosen_vector = [], None, None
    total_calls = 0
    states = [ArnoldiBasis(apply, shape, device, max(krylov_dims),
                          seed + start * 104729, initial_vector if start == 0 else None)
              for start in range(starts)]
    for dim in krylov_dims:
        current = []
        vectors = []
        for start in range(starts):
            summary, vector = arnoldi_response(apply, shape, device=device,
                krylov_dim=dim, seed=seed + start * 104729,
                residual_tol=residual_tol, real_tol=real_tol,
                substantial_fraction=substantial_fraction,
                initial_vector=initial_vector if start == 0 else None,
                _basis=states[start], _verify_spectrum=False)
            total_calls += summary["matvec_calls"]
            current.append(summary)
            vectors.append(vector)
            history.append(summary)
            if start == 0:
                chosen_vector = vector
        values = [s["mu_estimate"] for s in current]
        previous_values = [s["mu_estimate"] for s in previous] if previous else []
        all_values = values + previous_values
        valid = previous is not None and all(v is not None for v in all_values)
        spread = ((max(all_values) - min(all_values)) / max(abs(v) for v in all_values)
                  if valid else None)
        if initial_vector is not None and all(v is not None for v in values):
            discovered = max(range(starts), key=lambda i: values[i])
            if values[discovered] > values[0] * (1 + eigenvalue_rtol):
                # An old exact eigenvector can trap Arnoldi in its invariant
                # subspace. Recover using the larger mode found by the fresh
                # search; two-dimension agreement is still required afterwards.
                states[0] = ArnoldiBasis(apply, shape, device, max(krylov_dims), seed,
                                         vectors[discovered])
                current[0]["warm_mode_replaced_by_fresh_discovery"] = True
        if valid and spread <= eigenvalue_rtol:
            result, chosen_vector = arnoldi_response(apply, shape, device=device,
                krylov_dim=dim, seed=seed, residual_tol=residual_tol, real_tol=real_tol,
                substantial_fraction=substantial_fraction, initial_vector=initial_vector,
                _basis=states[0])
            total_calls += result["matvec_calls"]
            history.append(dict(result, final_spectrum_check=True))
            if result["mu_estimate"] is None:
                previous = current
                continue
            result.update(schema="local_muon_response_convergence_v1",
                status="positive_ritz_empirically_converged", empirical_convergence=True,
                eigenvalue_relative_spread=spread, eigenvalue_relative_tolerance=eigenvalue_rtol,
                convergence_starts=starts, convergence_attempts=history, matvec_calls=total_calls)
            result["basis_reused_across_dimensions"] = True
            result["fresh_random_searches"] = starts - int(initial_vector is not None)
            result["maximum_search_dimension"] = max(s["krylov_dimension"] for s in current)
            return result, chosen_vector
        previous = current
    result, chosen_vector = arnoldi_response(apply, shape, device=device,
        krylov_dim=krylov_dims[-1], seed=seed, residual_tol=residual_tol,
        real_tol=real_tol, substantial_fraction=substantial_fraction,
        initial_vector=initial_vector, _basis=states[0])
    total_calls += result["matvec_calls"]
    history.append(dict(result, final_spectrum_check=True))
    result.update(schema="local_muon_response_convergence_v1", mu_estimate=None,
        status="positive_ritz_convergence_not_established", empirical_convergence=False,
        eigenvalue_relative_spread=spread, eigenvalue_relative_tolerance=eigenvalue_rtol,
        convergence_starts=starts, convergence_attempts=history, matvec_calls=total_calls)
    return result, chosen_vector


class EagerForward(nn.Module):
    """Bypass only torch.compile; keep the model's original forward arithmetic."""
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, *args):
        return self.inner.forward(*args)


def frozen_hessian_block(model: nn.Module, parameters: dict[str, Tensor],
                         buffers: dict[str, Tensor], name: str,
                         batch: tuple[Tensor, Tensor], microbatch_sequences: int,
                         normalization: str = "sum"):
    """Supplied token-sum or token-mean loss Hessian; other weights fixed.

    autograd.grad never writes .grad. Recompute each microbatch per matvec to
    bound graph memory. Math SDPA supports double backward; no Flash fallback
    silently substitutes a different batch or a Gauss--Newton approximation.
    """
    if name not in parameters or parameters[name].ndim != 2 or microbatch_sequences < 1:
        raise ValueError("invalid frozen Hessian block")
    inputs, targets = batch
    if normalization not in ("sum", "mean") or targets.numel() == 0:
        raise ValueError("invalid Hessian loss normalization/batch")
    if len(inputs) != len(targets) or len(inputs) % microbatch_sequences:
        raise ValueError("Hessian batch must have complete microbatches")
    def accurate(v):
        return v.float() if v.dtype in (torch.bfloat16, torch.float16) else v
    base = {"inner." + k: accurate(v.detach()) for k, v in parameters.items()}
    base.update({"inner." + k: accurate(v.detach().clone()) for k, v in buffers.items()})
    wrapper = EagerForward(model)
    devices = [parameters[name].device.index] if parameters[name].is_cuda else []
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = [torch.cuda.get_rng_state(d).clone() for d in devices]

    def hvp(v):
        # Reproduce a fixed stochastic loss for every matvec and restore the
        # training generators. No random Krylov direction uses the global RNG.
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(initial_cpu_rng)
            for device, rng in zip(devices, initial_cuda_rng):
                torch.cuda.set_rng_state(rng, device)
            return calculate(v)

    def calculate(v):
        total = torch.zeros_like(accurate(parameters[name]))
        for offset in range(0, len(inputs), microbatch_sequences):
            with torch.enable_grad():
                p = accurate(parameters[name].detach()).requires_grad_(True)
                current = dict(base); current["inner." + name] = p
                context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                with context:
                    loss = torch.func.functional_call(wrapper, current,
                        (inputs[offset:offset + microbatch_sequences],
                         targets[offset:offset + microbatch_sequences]), strict=True)
                    if loss.ndim or not torch.isfinite(loss):
                        raise ValueError("Hessian loss must be a finite scalar")
                    g, = torch.autograd.grad(loss, p, create_graph=True)
                    if g.requires_grad:
                        result, = torch.autograd.grad((g * v.to(p.dtype)).sum(), p,
                                                      allow_unused=True)
                        if result is not None:
                            total += result.detach()
        return total / targets.numel() if normalization == "mean" else total
    return hvp
