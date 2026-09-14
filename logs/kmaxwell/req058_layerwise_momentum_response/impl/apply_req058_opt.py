"""REQ-058: install the memory-intervention optimizers + target-tagging hook. Idempotent. Run from repo root.
 - PerturbedAnnealedWeightsMuon: AnnealedWeightsMuon whose SELECTED params use decays beta_k**(1/a)
   (mixture weights unchanged); non-selected params keep the base decays. a=1 -> identical to parent.
 - ExactAgeMatchedMuon: single-EMA Muon whose beta(t) is solved so its realized age matches the K-Maxwell
   mixture's realized q_age at every step (the missing REQ-054 finite-history control).
 - tag_req058_perturb_target(target_name|"all"): setup hook (after assemble_grouped_optimizer) that resolves
   the target parameter by name and records its id on the blocks-group optimizer.
"""
import os, sys
MU = "records/track_3_optimization/optimizers/muon.py"
INIT = "records/track_3_optimization/optimizers/__init__.py"
HK = "records/track_3_optimization/harness/hooks.py"

CLS = '''

# ===== REQ-058 memory-intervention optimizers =====
def _req058_solve_matched_beta(decays, start_weights, end_weights, switch_step, anneal_end_step,
                               nu, n_steps, warm_extra=1):
    """Solve single-EMA beta(t) matching the mixture's realized q_age each step (see logs/kmaxwell/
    req058_layerwise_momentum_response). Returns list indexed by offset (t-switch-1)."""
    nstream = len(decays)
    mass = mom = 0.0
    for _ in range(switch_step + warm_extra):  # baseline update at switch, then clone
        mom, mass = nu * (mom + mass), nu * mass + (1 - nu)
    M_s = [mass] * nstream; P_s = [mom] * nstream
    out = []
    for j in range(n_steps):
        t = switch_step + 1 + j
        a = min(max((t - switch_step) / (anneal_end_step - switch_step), 0.0), 1.0)
        w = [s + a * (e - s) for s, e in zip(start_weights, end_weights)]; z = sum(w); w = [x / z for x in w]
        for k, b in enumerate(decays):
            P_s[k], M_s[k] = b * (P_s[k] + M_s[k]), b * M_s[k] + (1 - b)
        M_mix = sum(wi * m for wi, m in zip(w, M_s)); P_mix = sum(wi * p for wi, p in zip(w, P_s))
        q_mass = (1 - nu) + nu * M_mix; target_age = nu * P_mix / q_mass if q_mass > 0 else 0.0
        denom = nu * (mom + mass + target_age * (1 - mass))
        beta = target_age / denom if denom != 0 else 0.0
        assert 0.0 <= beta < 1.0, f"infeasible matched beta {beta} at t={t}"
        out.append(beta)
        mom, mass = beta * (mom + mass), beta * mass + (1 - beta)
    return out


class PerturbedAnnealedWeightsMuon(AnnealedWeightsMuon):
    """K-Maxwell mixture with a per-matrix decay-time intervention beta_k(a)=beta_k**(1/a) on tagged params.
    Mixture weights are unchanged (a memory intervention, not a reweighting). perturb_all applies to every
    matrix; otherwise tag_req058_perturb_target sets self._perturb_param_ids."""
    def __init__(self, params, perturb_a=1.0, perturb_all=False, **kw):
        super().__init__(params, **kw)
        self.perturb_a = float(perturb_a)
        self.perturb_all = bool(perturb_all)
        self._perturb_param_ids = set()
        self._perturbed_decays = tuple(b ** (1.0 / self.perturb_a) for b in self.decays)
        self._advance_perturbed = compile_annealed_decays_kernel(
            self._perturbed_decays, self.start_weights, self.end_weights, self.bias_correct_streams)

    def _is_perturbed(self, p):
        return self.perturb_a != 1.0 and (self.perturb_all or id(p) in self._perturb_param_ids)

    def compute_polar_input(self, p, state, group):
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
        if self._is_perturbed(p):
            decays, kernel = self._perturbed_decays, self._advance_perturbed
        else:
            decays, kernel = self.decays, self._advance_annealed_and_polar
        finite_masses = p.grad.new_tensor([1 - d ** age for d in decays])
        return kernel(p.grad, state["streams"], alpha, finite_masses, mu=mu)


class ExactAgeMatchedMuon(AnnealedDecayMuon):
    """Single-EMA Muon whose beta(t) is solved to match the K-Maxwell mixture's realized age each step."""
    def __init__(self, params, decays, start_weights, end_weights, switch_step=2000,
                 anneal_end_step=2750, nu=0.95, n_steps=4000, **kw):
        super().__init__(params, switch_step=switch_step, anneal_end_step=anneal_end_step, **kw)
        self._schedule = _req058_solve_matched_beta(list(decays), list(start_weights), list(end_weights),
                                                    int(switch_step), int(anneal_end_step), float(nu), int(n_steps))

    def current_beta(self):
        if self._muon_steps_seen <= self.switch_step:
            return self.beta_start
        off = self._muon_steps_seen - self.switch_step - 1
        off = min(max(off, 0), len(self._schedule) - 1)
        return self._schedule[off]
# ===== end REQ-058 =====
'''

HOOK = '''

def tag_req058_perturb_target(*, target_name: str):
    """Setup hook (place after assemble_grouped_optimizer): tag the perturbation target param by name on the
    PerturbedAnnealedWeightsMuon group. target_name=\"all\" sets perturb_all on that group."""
    def hook(config, state):
        named = dict(state["model"].named_parameters())
        for _, built in state["optimizer"].groups:
            if built.__class__.__name__ == "PerturbedAnnealedWeightsMuon":
                if target_name == "all":
                    built.perturb_all = True
                else:
                    assert target_name in named, f"perturb target {target_name} not in model"
                    built._perturb_param_ids = {id(named[target_name])}
                state["print_log"](f"req058 perturb target={target_name} a={built.perturb_a}", console=True)
        return state
    return hook
'''

t = open(MU).read()
if "class PerturbedAnnealedWeightsMuon" not in t:
    open(MU, "w").write(t + CLS); print("patched muon.py")
else:
    print("muon.py already patched")

t = open(INIT).read()
ch = False
if "PerturbedAnnealedWeightsMuon" not in t:
    i = t.index("_REGISTRY")
    t = t[:i] + "from .muon import PerturbedAnnealedWeightsMuon, ExactAgeMatchedMuon\n\n" + t[i:]
    key = '"adamw": build_record_adamw,'
    t = t.replace(key, key + '\n    "perturbed_annealed_weights_muon": PerturbedAnnealedWeightsMuon,'
                  '\n    "exact_age_matched_muon": ExactAgeMatchedMuon,', 1)
    ch = True
if ch:
    open(INIT, "w").write(t); print("patched __init__.py")
else:
    print("__init__.py already patched")

t = open(HK).read()
if "def tag_req058_perturb_target" not in t:
    # append the hook function, and register it in the hook registry
    t = t + HOOK
    # define the hook function AND (after it) register into the existing _HOOKS dict, so the assignment
    # runs after both the dict and the function exist (avoids a load-time NameError).
    t = t + HOOK + '\n_HOOKS["tag_req058_perturb_target"] = tag_req058_perturb_target\n'
    open(HK, "w").write(t); print("patched hooks.py + registered hook")
else:
    print("hooks.py already patched")

sys.path.insert(0, "records/track_3_optimization")
from optimizers import _REGISTRY
assert "perturbed_annealed_weights_muon" in _REGISTRY and "exact_age_matched_muon" in _REGISTRY
print("OK registered:", [k for k in _REGISTRY if "req058" in k or "perturbed" in k or "exact_age" in k])
