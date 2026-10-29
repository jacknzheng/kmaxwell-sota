import torch


class BiasCorrectionFreeAdam(torch.optim.Optimizer):
    """Bias-correction-free Adam for the aux (<2D) param groups.

    This is exactly the SOTA's CenterShrinkAdam with center_shrink (rho) = 1.0:
    at rho=1.0 the center-shrink collapses to the identity (mixed == adam_dir and
    the rms-rescale ratio is 1), leaving plain Adam with raw (uncorrected) EMAs.
    NOTE: this intentionally does NOT apply the (1-beta^t) bias correction, so it
    is NOT the same as torch.optim.AdamW. It matches the source under
    AUX_CENTER_SHRINK=1.0 bit-for-bit.
    """
    def __init__(self, params, lr=0.01, betas=(0.8, 0.99), eps=1e-10):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
                saved = state_dict["state"].get(saved_id, {})
                if "exp_avg_sq" in saved:
                    self.state[parameter]["exp_avg_sq"] = saved["exp_avg_sq"].to(
                        device=parameter.device, dtype=torch.float32).clone()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.float().square(), 1 - beta2)
                adam_dir = exp_avg.float() / (exp_avg_sq.sqrt() + eps)
                p.add_(adam_dir.to(p.dtype), alpha=-group["lr"])


class EMANesterov(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer, lookahead_stepsize=0, use_scheduled_lookahead_stepsize=True, lookahead_ema=0.9, prefill_steps=0, rest_steps=0):
        if lookahead_stepsize < 0.0:
            raise ValueError("Invalid momentum value: {}".format(lookahead_stepsize))

        super(EMANesterov, self).__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.use_scheduled_lookahead_stepsize = use_scheduled_lookahead_stepsize
        self.lookahead_stepsize = lookahead_stepsize
        self.lookahead_ema = lookahead_ema
        self.prefill_steps = prefill_steps
        self.rest_steps = rest_steps
        self.it = 0
        self.lookahead_status = False
        self.current_lookahead_stepsize = 0
        self.initialize_buffers()

    def state_dict(self):
        result = super().state_dict()
        inner = self.inner_optimizer if isinstance(self.inner_optimizer, list) else [self.inner_optimizer]
        result["inner_optimizers"] = [optimizer.state_dict() for optimizer in inner]
        result["outer_counters"] = {"it": self.it, "lookahead_status": self.lookahead_status,
                                    "current_lookahead_stepsize": self.current_lookahead_stepsize}
        result["outer_configuration"] = self.configuration()
        return result

    def configuration(self):
        return {name: getattr(self, name) for name in (
            "use_scheduled_lookahead_stepsize", "lookahead_stepsize", "lookahead_ema",
            "prefill_steps", "rest_steps")}

    def load_state_dict(self, state_dict):
        if state_dict["outer_configuration"] != self.configuration():
            raise ValueError("Outer EMA-Nesterov configuration changed on resume")
        inner = self.inner_optimizer if isinstance(self.inner_optimizer, list) else [self.inner_optimizer]
        if len(inner) != len(state_dict["inner_optimizers"]):
            raise ValueError("Inner optimizer count changed on resume")
        super().load_state_dict(state_dict)
        for optimizer, saved in zip(inner, state_dict["inner_optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        for name, value in state_dict["outer_counters"].items():
            setattr(self, name, value)

    def __setstate__(self, state):
        super(EMANesterov, self).__setstate__(state)

    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'prev_params' not in param_state:
                    param_state['prev_params'] = (p.clone(), self.it)

                if 'lookahead_buffer' not in param_state:
                    param_state['lookahead_buffer'] = (torch.zeros_like(p), -1)

    def get_lr_lambda(self):
        if isinstance(self.inner_optimizer, list):
            lr_lambda = self.inner_optimizer[0].param_groups[0]["lr"] / self.inner_optimizer[0].param_groups[0]["initial_lr"]
        else:
            lr_lambda = self.inner_optimizer.param_groups[0]["lr"] / self.inner_optimizer.param_groups[0]["initial_lr"]
        return lr_lambda

    @torch.no_grad()
    def lookahead_step(self):
        """Performs nesterov's lookahead."""
        if self.use_scheduled_lookahead_stepsize:
            lookahead_stepsize = self.lookahead_stepsize * self.get_lr_lambda()
        else:
            lookahead_stepsize = self.lookahead_stepsize
        self.current_lookahead_stepsize = lookahead_stepsize

        for group in self.param_groups:
            for p in group['params']:

                param_state = self.state[p]

                lookahead = param_state['lookahead_buffer'][0]

                p.add_(lookahead, alpha=lookahead_stepsize)



    @torch.no_grad()
    def accum_lookahead(self):
        """Update nesterov's lookahead direction."""
        lookahead_ema = self.lookahead_ema
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                look = p.add(param_state['prev_params'][0], alpha=-1)

                # update lookahead buffer
                buf = param_state['lookahead_buffer'][0]

                param_state['lookahead_buffer'] = (buf.lerp_(look, 1 - lookahead_ema), self.it) # m^{t+1} = beta * m^t + (1-beta) * look

                # update prev_params buffer
                param_state['prev_params'] = (param_state['prev_params'][0].copy_(p), self.it)


    @torch.no_grad()
    def nesterov_step(self):
        if self.it + 1 > self.prefill_steps and self.it < self.rest_steps and not self.lookahead_status:
            self.lookahead_step()
        else:
            self.current_lookahead_stepsize = 0
        self.lookahead_status = True

    @torch.no_grad()
    def step(self):
        if not self.lookahead_status:
            raise ValueError("optimizer.nesterov_step() should be invoked before model forward pass.")
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()

        self.accum_lookahead()

        self.lookahead_status = False
        self.it += 1
