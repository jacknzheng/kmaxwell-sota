"""REQ-059: add AllocatedAnnealedWeightsMuon (per-matrix memory multiplier a from an allocation map) +
tag_req059_allocation hook. Idempotent. Extends the REQ-058 patch (must be applied after apply_req058_opt.py).
"""
import sys
MU = "records/track_3_optimization/optimizers/muon.py"
INIT = "records/track_3_optimization/optimizers/__init__.py"
HK = "records/track_3_optimization/harness/hooks.py"

CLS = '''

class AllocatedAnnealedWeightsMuon(AnnealedWeightsMuon):
    """K-Maxwell mixture with a PER-MATRIX memory intervention: each param gets a multiplier a in {0.5,1,2}
    applied to its stream decays as beta_k**(1/a) (weights unchanged). a=1 uses the base kernel. The a-map
    (param id -> a) is set by tag_req059_allocation after construction; a_values precompiles the kernels."""
    def __init__(self, params, a_values=(0.5, 2.0), **kw):
        super().__init__(params, **kw)
        self._per_matrix_a = {}   # id(param) -> a (default 1.0)
        self._a_kernels = {}      # a -> (perturbed_decays, compiled kernel)
        for a in a_values:
            if float(a) == 1.0:
                continue
            pd = tuple(b ** (1.0 / float(a)) for b in self.decays)
            self._a_kernels[float(a)] = (pd, compile_annealed_decays_kernel(
                pd, self.start_weights, self.end_weights, self.bias_correct_streams))

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
        a = self._per_matrix_a.get(id(p), 1.0)
        if a == 1.0 or a not in self._a_kernels:
            decays, kernel = self.decays, self._advance_annealed_and_polar
        else:
            decays, kernel = self._a_kernels[a]
        finite_masses = p.grad.new_tensor([1 - d ** age for d in decays])
        return kernel(p.grad, state["streams"], alpha, finite_masses, mu=mu)
'''

HOOK = '''

def tag_req059_allocation(*, alloc_file: str, arm: str):
    """Setup hook (after assemble_grouped_optimizer): set the per-matrix a-map on the
    AllocatedAnnealedWeightsMuon group from alloc_file[arm] = {matrix_name: a}."""
    import json as _json
    def hook(config, state):
        alloc = _json.load(open(alloc_file))[arm]
        named = dict(state["model"].named_parameters())
        for _, built in state["optimizer"].groups:
            if built.__class__.__name__ == "AllocatedAnnealedWeightsMuon":
                amap = {}
                for name, a in alloc.items():
                    if name in named:
                        amap[id(named[name])] = float(a)
                built._per_matrix_a = amap
                n05 = sum(1 for v in amap.values() if v == 0.5); n2 = sum(1 for v in amap.values() if v == 2.0)
                state["print_log"](f"req059 arm={arm}: {len(amap)} matrices (a05={n05} a2={n2})", console=True)
        return state
    return hook
'''

t = open(MU).read()
if "class AllocatedAnnealedWeightsMuon" not in t:
    open(MU, "w").write(t + CLS); print("patched muon.py (AllocatedAnnealedWeightsMuon)")
else:
    print("muon.py already has AllocatedAnnealedWeightsMuon")

t = open(INIT).read()
if "AllocatedAnnealedWeightsMuon" not in t:
    i = t.index("_REGISTRY")
    t = t[:i] + "from .muon import AllocatedAnnealedWeightsMuon\n\n" + t[i:]
    key = '"perturbed_annealed_weights_muon": PerturbedAnnealedWeightsMuon,'
    t = t.replace(key, key + '\n    "allocated_annealed_weights_muon": AllocatedAnnealedWeightsMuon,', 1)
    open(INIT, "w").write(t); print("patched __init__.py")
else:
    print("__init__.py already has it")

t = open(HK).read()
if "def tag_req059_allocation" not in t:
    t = t + HOOK + '\n_HOOKS["tag_req059_allocation"] = tag_req059_allocation\n'
    open(HK, "w").write(t); print("patched hooks.py + registered")
else:
    print("hooks.py already patched")

sys.path.insert(0, "records/track_3_optimization")
from optimizers import _REGISTRY
from harness.hooks import _HOOKS
assert "allocated_annealed_weights_muon" in _REGISTRY and "tag_req059_allocation" in _HOOKS
print("OK: allocated_annealed_weights_muon + tag_req059_allocation registered")
