"""REQ-056 — K-Maxwell memory inside STANDARD Adam.

One torch.optim.Optimizer that always maintains, per parameter:
  m1  : ordinary Adam first moment (beta1)            v : Adam second moment (beta2)
  streams[K] : K fixed-decay first-moment EMAs (the K-Maxwell mixture poles)
  m_age, M_age, P_age : a single EMA with time-varying decay beta(t), its running normalization
                        mass M_age = 1 - prod_t beta(t) (variable-decay, NOT 1-beta^t), and the
                        running age-moment P_age for the REALIZED (finite-history) mean age.

`numerator_mode` chooses which first-moment estimate feeds the update; the second moment, denominator,
eps, LR schedule and weight decay are identical across modes (this is standard Adam, not AdamW-with-tricks
and NOT AdamH: no polar transform, no Nesterov blend, no hyperball/scale-invariant projection):
  'adam'        : m1 / (1 - beta1^t)
  'kmaxwell'    : sum_i w_i(t) * streams[i] / (1 - decay_i^t)      (per constant-decay stream: 1-beta^t)
  'age_matched' : m_age / M_age                                    (variable-decay mass)
update = numerator / (sqrt(v/(1-beta2^t)) + eps);  p -= lr*update  (+ decoupled wd if nonzero).

Because every buffer accumulates every step regardless of mode, a base run in 'adam' mode warms the
K/age buffers WITHOUT changing the base trajectory (the update only reads the adam numerator), so a fork
into 'kmaxwell'/'age_matched' inherits real shadow history. Weights w_i(t) anneal start->end linearly over
[switch_step, anneal_end_step]; before switch_step the weights are held at start (alpha=0)."""
import torch
from torch.optim import Optimizer


class KMaxwellAdam(Optimizer):
    def __init__(self, params, *, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 numerator_mode="adam", decays=None, start_weights=None, end_weights=None,
                 switch_step=1000, anneal_end_step=3250, accumulate_shadow=True):
        assert numerator_mode in ("adam", "kmaxwell", "age_matched")
        decays = list(decays) if decays else []
        start_weights = list(start_weights) if start_weights else []
        end_weights = list(end_weights) if end_weights else []
        needs_streams = numerator_mode in ("kmaxwell", "age_matched") or accumulate_shadow
        if needs_streams:
            assert len(decays) == len(start_weights) == len(end_weights) and len(decays) > 0, \
                "kmaxwell/age_matched (or shadow) require equal-length nonempty decays/start_weights/end_weights"
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.numerator_mode = numerator_mode
        self.decays = decays
        self.start_weights = start_weights
        self.end_weights = end_weights
        self.switch_step = int(switch_step)
        self.anneal_end_step = int(anneal_end_step)
        self.accumulate_shadow = bool(needs_streams)
        self._K = len(decays)
        self.last_telemetry = {}  # filled at telemetry() calls

    # ---- schedule helpers ----
    def _alpha(self, t):
        if self.anneal_end_step <= self.switch_step:
            return 1.0 if t > self.switch_step else 0.0
        return min(max((t - self.switch_step) / (self.anneal_end_step - self.switch_step), 0.0), 1.0)

    def _weights(self, t):
        a = self._alpha(t)
        return [s + a * (e - s) for s, e in zip(self.start_weights, self.end_weights)]

    def scheduled_age(self, t):
        w = self._weights(t)
        num = sum(wi * bi / (1.0 - bi) for wi, bi in zip(w, self.decays))
        den = sum(w)
        return num / den if den > 0 else 0.0

    def _age_beta(self, t):
        A = self.scheduled_age(t)
        return A / (1.0 + A)

    @staticmethod
    def _const_beta_realized_age(beta, t):
        """Exact finite-history mean age of a constant-beta EMA after t updates."""
        if t <= 0:
            return 0.0
        bt = beta ** t
        denom = (1.0 - beta) * (1.0 - bt)
        if denom <= 0:
            return beta / (1.0 - beta)
        return beta * (1.0 - t * (beta ** (t - 1)) + (t - 1) * bt) / denom

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, b1, b2, eps, wd = group["lr"], group["beta1"], group["beta2"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    if self.accumulate_shadow:
                        st["streams"] = torch.zeros((self._K, *p.shape), device=p.device, dtype=p.dtype)
                        st["m_age"] = torch.zeros_like(p)
                        st["M_age"] = torch.zeros((), device=p.device, dtype=torch.float64)
                        st["P_age"] = torch.zeros((), device=p.device, dtype=torch.float64)
                st["step"] += 1
                t = st["step"]
                # second moment (standard Adam, always)
                v = st["v"]
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                vhat = v / (1 - b2 ** t)
                # adam first moment (always maintained)
                m1 = st["m1"]
                m1.mul_(b1).add_(g, alpha=1 - b1)
                # shadow K streams + age EMA (always maintained if enabled)
                if self.accumulate_shadow:
                    streams = st["streams"]
                    dec = g.new_tensor(self.decays).view(-1, *([1] * g.ndim))
                    streams.lerp_(g.unsqueeze(0), 1 - dec)
                    beta_age = self._age_beta(t)
                    st["m_age"].mul_(beta_age).add_(g, alpha=1 - beta_age)
                    # variable-decay mass and realized age-moment (finite history)
                    st["P_age"].mul_(beta_age).add_(beta_age * float(st["M_age"]))  # old ages +1, decayed
                    st["M_age"].mul_(beta_age).add_(1 - beta_age)
                # select numerator
                if self.numerator_mode == "adam":
                    num = m1 / (1 - b1 ** t)
                elif self.numerator_mode == "kmaxwell":
                    w = g.new_tensor(self._weights(t)).view(-1, *([1] * g.ndim))
                    dec = g.new_tensor(self.decays).view(-1, *([1] * g.ndim))
                    bc = 1 - dec ** t
                    num = (w * st["streams"] / bc).sum(dim=0)
                else:  # age_matched
                    M = float(st["M_age"])
                    num = st["m_age"] / M if M > 0 else torch.zeros_like(p)
                update = num / (vhat.sqrt() + eps)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return None

    @torch.no_grad()
    def telemetry(self):
        """Aggregate diagnostics across all params at the current step (call after step()).
        Records scheduled + realized first-moment ages, numerator/denominator/update norms, nonfinite count."""
        b1 = self.param_groups[0]["beta1"]
        eps = self.param_groups[0]["eps"]
        b2 = self.param_groups[0]["beta2"]
        num_sq = den_sq = upd_sq = 0.0
        nonfinite = 0
        t = 0
        age_real_num = age_real_den = 0.0  # for age EMA realized age (mass-weighted over params)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                if len(st) == 0:
                    continue
                t = st["step"]
                vhat = st["v"] / (1 - b2 ** t)
                den = vhat.sqrt() + eps
                if self.numerator_mode == "adam":
                    num = st["m1"] / (1 - b1 ** t)
                elif self.numerator_mode == "kmaxwell":
                    w = p.new_tensor(self._weights(t)).view(-1, *([1] * p.ndim))
                    dec = p.new_tensor(self.decays).view(-1, *([1] * p.ndim))
                    num = (w * st["streams"] / (1 - dec ** t)).sum(dim=0)
                else:
                    M = float(st["M_age"])
                    num = st["m_age"] / M if M > 0 else torch.zeros_like(p)
                    age_real_num += float(st["P_age"]); age_real_den += float(st["M_age"])
                upd = num / den
                num_sq += float((num * num).sum())
                den_sq += float((den * den).sum())
                upd_sq += float((upd * upd).sum())
                nonfinite += int((~torch.isfinite(upd)).sum())
        out = {"step": t, "mode": self.numerator_mode,
               "numerator_norm": num_sq ** 0.5, "denominator_norm": den_sq ** 0.5,
               "update_norm": upd_sq ** 0.5, "nonfinite": nonfinite,
               "scheduled_age": self.scheduled_age(t)}
        if self.numerator_mode == "age_matched":
            out["realized_age_ema"] = age_real_num / age_real_den if age_real_den > 0 else 0.0
        if self._K:
            w = self._weights(t); den_w = sum(w)
            out["realized_age_mixture"] = (sum(wi * self._const_beta_realized_age(bi, t)
                                               for wi, bi in zip(w, self.decays)) / den_w) if den_w > 0 else 0.0
        self.last_telemetry = out
        return out
