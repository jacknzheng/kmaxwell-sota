"""REQ-072 integration patch (idempotent; run from repo root). Captures 3 streams (raw g_t, conditioned u_t,
displacement delta_t) for the selected matrices only, across the 6 optimizer arms.

- Registers CaptureAdamW as optimizer 'capture_adamw'.
- Wraps Muon.compute_polar_input so Muon stashes u_t (its scaled orthogonalized direction) for tagged ids.
- SgdBlocks u_t is read post-step (momentum>0 -> state['momentum'] buffer == the update used; momentum==0 ->
  u_t == g_t, the raw gradient).
- Hooks: tag_req072_capture (setup) / req072_pre (pre_optimizer) / req072_post (post_optimizer) /
  finalize_req072 (teardown). Writers reuse GradientHistoryWriter.
Depends on REQ-068's req068_capture + the clean-name helper (apply gradient capture patch first for that helper).
"""
import os, shutil
HOOKS = "records/track_3_optimization/harness/hooks.py"
OPT_INIT = "records/track_3_optimization/optimizers/__init__.py"
OPTDIR = "records/track_3_optimization/optimizers"

# 1. place CaptureAdamW module in the optimizers package + register it
src = os.path.join(os.path.dirname(__file__), "req072_capture_adamw.py")
shutil.copy(src, os.path.join(OPTDIR, "req072_capture_adamw.py"))
t = open(OPT_INIT).read()
if "capture_adamw" not in t:
    anchor = '_REGISTRY'
    i = t.index(anchor)
    t = t[:i] + "from .req072_capture_adamw import CaptureAdamW\n\n" + t[i:]
    # add registry entry after the dict opens
    t = t.replace('"adamw": build_record_adamw,', '"adamw": build_record_adamw,\n    "capture_adamw": CaptureAdamW,')
    open(OPT_INIT, "w").write(t); print("registered capture_adamw")
else:
    print("capture_adamw already registered")

CODE = '''

# ===== REQ-072 three-stream capture (selected matrices) =====
import sys as _r72_sys
for _p in ("logs/river/req072_optimizer_spectrograms/impl", "logs/river/req068_full_gradient_history/impl"):
    _r72_sys.path.insert(0, _p)


def _r72_wrap_muon():
    """Wrap Muon.compute_polar_input once so it stashes u_t (scaled orthogonalized dir) for tagged ids."""
    from optimizers.muon import Muon
    if getattr(Muon, "_r72_wrapped", False):
        return
    orig = Muon.compute_polar_input
    def wrapped(self, p, state, group):
        u = orig(self, p, state, group)
        ids = getattr(self, "_req072_ids", None)
        if ids and id(p) in ids:
            if not hasattr(self, "_req072_u"): self._req072_u = {}
            self._req072_u[id(p)] = u.detach().to("cpu", _r72_torch.float32).clone()
        return u
    Muon.compute_polar_input = wrapped
    Muon._r72_wrapped = True

import torch as _r72_torch


def tag_req072_capture(*, names, grad_dir, uT_dir, disp_dir, chunk_steps=500):
    """Setup hook (after assemble_grouped_optimizer): tag selected param ids on their optimizers + open writers."""
    from req068_capture import GradientHistoryWriter, build_param_schema
    def hook(config, state):
        _r72_wrap_muon()
        model = state["model"]
        named = {n[len("_orig_mod."):] if n.startswith("_orig_mod.") else n: p
                 for n, p in model.named_parameters()}
        sel = [(n, named[n]) for n in names if n in named]
        assert len(sel) == len(names), f"missing selected matrices: {set(names) - set(n for n,_ in sel)}"
        ids = {id(p) for _, p in sel}
        for _, built in state["optimizer"].groups:
            own = {id(p) for p in getattr(built, "param_groups", [{}])[0].get("params", [])}
            tag = ids & own
            if tag:
                built._req072_ids = tag
                if not hasattr(built, "_req072_u"): built._req072_u = {}
        schema = build_param_schema(sel)
        prov = {"code_sha": "365c392d695f95dc9a4fb89095e85a6a7b5d551e", "run_id": config.get("run_id"),
                "selected": list(names), "batch_tokens": config.get("batch_tokens")}
        state["_r72"] = dict(
            names=list(names), sel=sel,
            wg=GradientHistoryWriter(grad_dir, schema, chunk_steps=chunk_steps,
                                     tokens_per_update=config.get("batch_tokens"), provenance={**prov, "stream": "raw_grad"}, grad_reduction="sum"),
            wu=GradientHistoryWriter(uT_dir, schema, chunk_steps=chunk_steps, provenance={**prov, "stream": "conditioned_u_t"}, grad_reduction="conditioned_pre_lr"),
            wd=GradientHistoryWriter(disp_dir, schema, chunk_steps=chunk_steps, provenance={**prov, "stream": "displacement"}, grad_reduction="displacement"),
        )
        state["print_log"](f"req072 capture tagged {len(sel)} matrices across arms -> {grad_dir}", console=True)
        return state
    return hook


def req072_pre():
    """pre_optimizer: stash selected g_t (raw) + theta_before (master only; grads replicated, params replicated)."""
    def hook(config, state):
        if not state.get("master", False): return state
        r = state.get("_r72");  r is None and None
        if r is None: return state
        r["g"] = {n: (p.grad.detach().to("cpu", _r72_torch.float32).clone() if p.grad is not None else None)
                  for n, p in r["sel"]}
        r["theta"] = {n: p.detach().to("cpu", _r72_torch.float32).clone() for n, p in r["sel"]}
        return state
    return hook


def req072_post():
    """post_optimizer: write g_t, u_t (per optimizer type), delta_t for selected matrices."""
    def hook(config, state):
        if not state.get("master", False): return state
        r = state.get("_r72")
        if r is None: return state
        step = state["step"]
        # find owning optimizer per selected param
        id2opt = {}
        for _, built in state["optimizer"].groups:
            for p in getattr(built, "param_groups", [{}])[0].get("params", []):
                id2opt[id(p)] = built
        g_out, u_out, d_out = {}, {}, {}
        for n, p in r["sel"]:
            g_out[n] = r["g"].get(n)
            # u_t
            built = id2opt.get(id(p))
            u = getattr(built, "_req072_u", {}).get(id(p)) if built is not None else None
            if u is None and built is not None and built.__class__.__name__ == "SgdBlocks":
                grp = built.param_groups[0]
                if grp.get("momentum", 0) > 0 and id(p) in [id(x) for x in grp["params"]]:
                    buf = built.state[p].get("momentum")
                    u = buf.detach().to("cpu", _r72_torch.float32).clone() if buf is not None else None
                else:
                    u = r["g"].get(n)  # sgd-nomom: u_t == raw grad
            u_out[n] = u
            # delta
            d_out[n] = (p.detach().to("cpu", _r72_torch.float32) - r["theta"][n]) if n in r.get("theta", {}) else None
        r["wg"].record(step, g_out); r["wu"].record(step, u_out); r["wd"].record(step, d_out)
        return state
    return hook


def finalize_req072():
    def hook(config, state):
        r = state.get("_r72")
        if r is not None:
            for w in ("wg", "wu", "wd"):
                m = r[w].close()
            state["print_log"](f"req072 capture closed ({r['wg'].schema['total_numel']} params/stream)")
        return state
    return hook


_HOOKS["tag_req072_capture"] = tag_req072_capture
_HOOKS["req072_pre"] = req072_pre
_HOOKS["req072_post"] = req072_post
_HOOKS["finalize_req072"] = finalize_req072
# ===== end REQ-072 =====
'''

h = open(HOOKS).read()
if "tag_req072_capture" in h:
    print("hooks.py already has REQ-072 hooks")
else:
    open(HOOKS, "w").write(h + CODE)
    print("patched hooks.py with REQ-072 capture hooks")
