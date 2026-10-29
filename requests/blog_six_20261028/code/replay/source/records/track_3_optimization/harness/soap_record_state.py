from contextlib import contextmanager
import torch
import torch.distributed as dist


class SoapRecordState:
    def __init__(self, model, hidden_optimizer):
        self.model = model
        self.hidden = hidden_optimizer
        self.accum = self.count = self.cov = self.cov_fast = self.cov_slow = self.inverse = None
        self.tail = None
        self.handle = None
        self.parameter_names = [name for name, _ in model.named_parameters()]

    @torch.compiler.disable
    @torch.no_grad()
    def collect(self, module, inputs, output):
        step = self.hidden.step_count
        if not self.model.training or not 1400 <= step < 2640:
            return
        x = inputs[0].detach().flatten(0, -2)[::16].float()
        self.accum.addmm_(x.mT, x)
        self.count.add_(x.size(0))

    @torch.no_grad()
    def enable(self, step):
        if self.handle is not None or step < 1400:
            return
        weight = self.model.proj.weight
        if self.accum is None:
            self.accum = torch.zeros(weight.size(1), weight.size(1), dtype=torch.float32, device=weight.device)
            self.count = torch.zeros((), dtype=torch.float32, device=weight.device)
        self.handle = self.model.proj.register_forward_hook(self.collect)
        torch.compiler.reset()

    @staticmethod
    def schedule(step):
        ramp = min(1.0, max(0.0, (step - 1500) / (1750 - 1500)))
        return .25 + ramp * (.75 - .25), .15 + ramp * (.05 - .15)

    @torch.no_grad()
    def apply(self, step):
        if not 1400 <= step < 2640:
            return
        if step == 1400:
            assert self.count.item() > 0, "output KFAC hook did not capture activations"
        dist.all_reduce(self.accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)
        batch_cov = self.accum / self.count.clamp_min(1.0)
        if self.cov_fast is None:
            self.cov_fast = batch_cov.clone()
            self.cov_slow = batch_cov.clone()
        else:
            self.cov_fast.lerp_(batch_cov, 1.0 - .8)
            self.cov_slow.lerp_(batch_cov, 1.0 - .98)
        self.cov = torch.lerp(self.cov_slow, self.cov_fast, 8.0 / 9.0)
        self.accum.zero_()
        self.count.zero_()
        if step < 1500:
            return
        strength, damping_fraction = self.schedule(step)
        if self.inverse is None or (step - 1500) % 8 == 0:
            eigenvalues, eigenvectors = torch.linalg.eigh(self.cov)
            damping = damping_fraction * eigenvalues.mean().clamp_min(1e-8)
            scales = (eigenvalues.clamp_min(0.0) + damping).pow(-.5)
            self.inverse = (eigenvectors * scales.unsqueeze(0)) @ eigenvectors.mT
        grad = self.model.proj.weight.grad
        raw = grad.float()
        natural = raw @ self.inverse
        raw_norm = raw.norm()
        natural.mul_(raw_norm / natural.norm().clamp_min(1e-12))
        mixed = raw.lerp(natural, strength)
        mixed.mul_(raw_norm / mixed.norm().clamp_min(1e-12))
        grad.copy_(mixed.to(grad.dtype))

    @torch.no_grad()
    def update_tail(self, boundary):
        if not 2040 <= boundary < 2840:
            return
        if self.tail is None:
            self.tail = [None if p is self.model.embed.weight else p.detach().float().clone()
                         for p in self.model.parameters()]
        else:
            for p, value in zip(self.model.parameters(), self.tail, strict=True):
                if value is not None:
                    value.add_(p.detach().float() - value, alpha=1.0 / 150.)

    @staticmethod
    def blend(name, parameter):
        if name == "proj.weight":
            return .60
        if name.startswith("blocks.") and parameter.ndim >= 2:
            return .90 if int(name.split(".", 2)[1]) == 0 else .65
        return 1.0

    @contextmanager
    def tail_weights(self):
        if self.tail is None:
            yield
            return
        with torch.no_grad():
            stash = [p.detach().clone() for p in self.model.parameters()]
            try:
                for (name, p), value in zip(self.model.named_parameters(), self.tail, strict=True):
                    if value is not None:
                        blend = self.blend(name, p)
                        p.copy_(((1.0 - blend) * p.float() + blend * value).to(p.dtype))
                yield
            finally:
                for p, original in zip(self.model.parameters(), stash, strict=True):
                    p.copy_(original)

    def state_dict(self):
        return dict(parameter_names=self.parameter_names,
                    tensors={name: getattr(self, name) for name in (
                        "accum", "count", "cov", "cov_fast", "cov_slow", "inverse", "tail")})

    def load_state_dict(self, saved):
        if saved["parameter_names"] != self.parameter_names:
            raise ValueError("SOAP readout parameter order changed on resume")
        expected = {"accum", "count", "cov", "cov_fast", "cov_slow", "inverse", "tail"}
        if set(saved["tensors"]) != expected:
            raise ValueError("SOAP readout checkpoint fields changed")
        tail = saved["tensors"]["tail"]
        if tail is not None:
            if len(tail) != len(self.parameter_names):
                raise ValueError("SOAP tail checkpoint length changed")
            for p, value in zip(self.model.parameters(), tail, strict=True):
                if (p is self.model.embed.weight) != (value is None):
                    raise ValueError("SOAP tail embedding exclusion changed")
                if value is not None and value.shape != p.shape:
                    raise ValueError("SOAP tail checkpoint shape changed")
        device = self.model.proj.weight.device
        for name, value in saved["tensors"].items():
            if name == "tail" and value is not None:
                value = [None if v is None else v.to(device=device, dtype=torch.float32).clone() for v in value]
            elif value is not None:
                value = value.to(device=device, dtype=torch.float32).clone()
            setattr(self, name, value)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
