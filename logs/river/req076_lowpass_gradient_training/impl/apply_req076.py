"""REQ-076/077 integration patch (idempotent; run from harness root AFTER apply_req073_capture.py).

Adds req076_filter: a pre_optimizer hook that, for EVERY Muon-managed hidden matrix, maintains a per-param
causal raw-gradient history and overwrites p.grad with the filtered gradient BEFORE Muon's transform. The raw
synchronized gradient (post all_reduce SUM, pre-clip/Muon) is copied into history first; during the first
`startup` updates after the loop's start_step the gradient passes through UNFILTERED while history is recorded;
from the 9th update the filter is active. Filters (torch, replicated on every rank so owner-rank Muon sees the
same filtered grad): 'none', 'pair' (½g+½g₋₁), 'fir9' (9-tap windowed-sinc fc=0.20), 'ema_third' (⅔g+⅓v₋₁).

Placement: list req076_filter in pre_optimizer AFTER req073_pre (so REQ-073 stage-1 still captures the RAW
grad) and BEFORE scale_grad_for_4B / schedule / Muon. It also stashes the filtered grad for the diagnostic
matrices into state['_r76_filtered'] so the capture can write a 'filtered' stage. Weight decay is applied by
the optimizer separately and is never filtered. Exactly one forward/backward + update per batch.

FIR9 taps and EMA alpha are the registered values (see req07x_filters.py / filter_responses.json).
"""
import os
HOOKS = "records/track_3_optimization/harness/hooks.py"

CODE = '''

# ===== REQ-076/077 raw-gradient causal low-pass filter =====
import torch as _r76_torch
_R76_FIR9 = [-0.0, -0.009206, 0.047148, 0.260461, 0.403195, 0.260461, 0.047148, -0.009206, -0.0]
_R76_EMA_ALPHA = 2.0 / 3.0
_R76_STARTUP = 8


def req076_filter(*, kind, startup=_R76_STARTUP, selected=()):
    """pre_optimizer (ALL ranks): overwrite every Muon hidden-matrix p.grad with its causal-filtered value.
    kind in {none,pair,fir9,ema_third}. Maintains per-param history in state['_r76']. Records genuine raw
    history during the first `startup` updates (pass-through), then filters."""
    assert kind in ("none", "pair", "fir9", "ema_third")
    taps = {"pair": [0.5, 0.5], "fir9": _R76_FIR9}.get(kind)

    def hook(config, state):
        if kind == "none":
            return state
        st = state.setdefault("_r76", {"hist": {}, "ema": {}})
        _, muon = find_muon_family_group(state["optimizer"])
        muon_params = set(id(p) for p in muon.param_groups[0]["params"])
        active = (state["step"] - config.get("start_step", 0)) >= startup
        sel = set(selected)
        # resolve selected diagnostic params by name for the 'filtered' capture
        name_by_id = {id(p): n for n, p in state["model"].named_parameters()}
        filtered_cap = {}
        with _r76_torch.no_grad():
            for p in muon.param_groups[0]["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                g = p.grad
                if kind == "ema_third":
                    prev = st["ema"].get(pid)
                    cur = g.detach().clone() if prev is None else (_R76_EMA_ALPHA * g + (1 - _R76_EMA_ALPHA) * prev)
                    st["ema"][pid] = cur
                    if active:
                        p.grad.copy_(cur)
                else:
                    h = st["hist"].setdefault(pid, [])
                    h.append(g.detach().clone())
                    if len(h) > len(taps):
                        h.pop(0)
                    if active:
                        hs = h[::-1]
                        out = taps[0] * hs[0]
                        for i in range(1, min(len(taps), len(hs))):
                            out = out + taps[i] * hs[i]
                        p.grad.copy_(out)
                nm = name_by_id.get(pid)
                if active and nm in sel:
                    filtered_cap[nm] = p.grad.detach().clone()
        state["_r76_filtered"] = filtered_cap
        return state
    return hook

_HOOKS["req076_filter"] = req076_filter
# ===== end REQ-076/077 =====
'''

h = open(HOOKS).read()
if "req076_filter" in h:
    print("hooks.py already has REQ-076 filter hook")
else:
    open(HOOKS, "w").write(h + CODE)
    print("patched hooks.py with REQ-076 filter hook")
