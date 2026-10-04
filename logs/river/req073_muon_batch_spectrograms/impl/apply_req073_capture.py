"""REQ-073 integration patch (idempotent; run from repo root). Captures the 4-stage Muon pipeline for the
selected hidden matrices (blocks.5 Q/K/mlp.proj), handling Muon's per-rank sharding via all_reduce.

- Wraps Muon.compute_polar_input: for tagged selected params, uses split_capture (bit-identical to stock
  muon_update) and stashes stage2 (post-momentum pre-polar) + stage3 (post-polar u_t) on GPU per rank. Each
  selected matrix's stages are nonzero only on its owning rank.
- req073_post (ALL ranks): all_reduce(SUM) the stage2/3 tensors so master gathers each matrix's value from its
  owner, then master writes stages 1 (raw g_t, from pre), 2, 3, 4 (displacement) uniformly. Stages 1 & 4 are
  replicated (captured on master); 2 & 3 come via the reduce.
- scale_grad_4B(factor): pre_optimizer, divides .grad by `factor` (=0.25 for 4B arms) AFTER stage-1 capture,
  preserving the B token-sum optimizer convention. Stage-1 records the raw (pre-divide) sum + normalization.
Reuses req068_capture.GradientHistoryWriter. Depends on req073_stages.split_capture.
"""
import os
HOOKS = "records/track_3_optimization/harness/hooks.py"

CODE = '''

# ===== REQ-073 four-stage Muon pipeline capture =====
import sys as _r73_sys, torch as _r73_torch, torch.distributed as _r73_dist
for _p in ("logs/river/req073_muon_batch_spectrograms/impl", "logs/river/req068_full_gradient_history/impl"):
    _r73_sys.path.insert(0, _p)


def _r73_wrap_muon():
    from optimizers.muon import Muon, muon_update
    if getattr(Muon, "_r73_wrapped", False):
        return
    from req073_stages import split_capture
    orig = Muon.compute_polar_input
    def wrapped(self, p, state, group):
        ids = getattr(self, "_req073_ids", None)
        if not ids or id(p) not in ids:
            return orig(self, p, state, group)
        if "momentum" not in state:
            state["momentum"] = _r73_torch.zeros_like(p)
        final, s2, s3 = split_capture(muon_update, p.grad, state["momentum"], mu=group["mu"])
        if not hasattr(self, "_req073_s2"): self._req073_s2 = {}; self._req073_s3 = {}
        self._req073_s2[id(p)] = s2; self._req073_s3[id(p)] = s3   # stay on GPU for all_reduce
        return final
    Muon.compute_polar_input = wrapped
    Muon._r73_wrapped = True


def tag_req073_capture(*, names, out_base, chunk_steps=500):
    """Setup hook: wrap Muon, resolve selected matrices, tag the Muon group, open master writers (stages 1-4)."""
    from req068_capture import GradientHistoryWriter, build_param_schema
    def hook(config, state):
        _r73_wrap_muon()
        named = {n[len("_orig_mod."):] if n.startswith("_orig_mod.") else n: p
                 for n, p in state["model"].named_parameters()}
        sel = [(n, named[n]) for n in names if n in named]
        assert len(sel) == len(names), f"missing: {set(names)-set(n for n,_ in sel)}"
        ids = {id(p) for _, p in sel}
        for _, built in state["optimizer"].groups:
            if built.__class__.__name__.endswith("Muon"):
                own = {id(p) for p in built.param_groups[0]["params"]}
                if ids & own:
                    built._req073_ids = ids & own
        schema = build_param_schema(sel)
        prov = {"code_sha": "365c392d695f95dc9a4fb89095e85a6a7b5d551e", "run_id": config.get("run_id"),
                "selected": list(names), "batch_tokens": config.get("batch_tokens")}
        state["_r73"] = dict(names=list(names), sel=sel, shapes={n: tuple(p.shape) for n, p in sel},
            w1=GradientHistoryWriter(f"{out_base}/s1_grad", schema, chunk_steps=chunk_steps, tokens_per_update=config.get("batch_tokens"), provenance={**prov, "stage": "1_raw_grad"}, grad_reduction="sum"),
            w2=GradientHistoryWriter(f"{out_base}/s2_premom", schema, chunk_steps=chunk_steps, provenance={**prov, "stage": "2_post_momentum_pre_polar"}, grad_reduction="muon_input"),
            w3=GradientHistoryWriter(f"{out_base}/s3_postpolar", schema, chunk_steps=chunk_steps, provenance={**prov, "stage": "3_post_polar_u_t"}, grad_reduction="conditioned"),
            w4=GradientHistoryWriter(f"{out_base}/s4_disp", schema, chunk_steps=chunk_steps, provenance={**prov, "stage": "4_displacement"}, grad_reduction="displacement"))
        state["print_log"](f"req073 capture tagged {len(sel)} matrices -> {out_base}", console=True)
        return state
    return hook


def req073_pre():
    """pre_optimizer (master): stash stage-1 raw g_t + theta_before for selected matrices."""
    def hook(config, state):
        r = state.get("_r73")
        if r is None or not state.get("master", False): return state
        r["g1"] = {n: (p.grad.detach().to("cpu", _r73_torch.float32).clone() if p.grad is not None else None) for n, p in r["sel"]}
        r["theta"] = {n: p.detach().to("cpu", _r73_torch.float32).clone() for n, p in r["sel"]}
        return state
    return hook


def scale_grad_for_4B(*, factor):
    """pre_optimizer (ALL ranks): divide every param's .grad by factor (4B -> /4), after stage-1 capture."""
    def hook(config, state):
        if factor == 1.0: return state
        with _r73_torch.no_grad():
            for p in state["model"].parameters():
                if p.grad is not None: p.grad.mul_(factor)
        return state
    return hook


def req073_post():
    """post_optimizer (ALL ranks): all_reduce stage2/3 (owner-sharded) to master; master writes stages 1-4."""
    def hook(config, state):
        r = state.get("_r73")
        if r is None: return state
        # gather stage2/3 across ranks: each selected matrix nonzero only on its owner
        muon = None
        for _, built in state["optimizer"].groups:
            if getattr(built, "_req073_ids", None): muon = built
        s2g, s3g = {}, {}
        for n, p in r["sel"]:
            shp = r["shapes"][n]; dev = p.device
            t2 = (muon._req073_s2.get(id(p)) if muon and hasattr(muon, "_req073_s2") else None)
            t3 = (muon._req073_s3.get(id(p)) if muon and hasattr(muon, "_req073_s3") else None)
            b2 = t2 if t2 is not None else _r73_torch.zeros(shp, device=dev, dtype=_r73_torch.float32)
            b3 = t3 if t3 is not None else _r73_torch.zeros(shp, device=dev, dtype=_r73_torch.float32)
            b2 = b2.to(_r73_torch.float32); b3 = b3.to(_r73_torch.float32)
            if _r73_dist.is_initialized() and _r73_dist.get_world_size() > 1:
                _r73_dist.all_reduce(b2); _r73_dist.all_reduce(b3)
            s2g[n] = b2; s3g[n] = b3
        if muon is not None and hasattr(muon, "_req073_s2"):
            muon._req073_s2.clear(); muon._req073_s3.clear()
        if not state.get("master", False):
            return state
        step = state["step"]
        r["w1"].record(step, r.get("g1", {}))
        r["w2"].record(step, {n: s2g[n].cpu() for n in s2g})
        r["w3"].record(step, {n: s3g[n].cpu() for n in s3g})
        r["w4"].record(step, {n: (p.detach().to("cpu", _r73_torch.float32) - r["theta"][n]) for n, p in r["sel"] if n in r.get("theta", {})})
        return state
    return hook


def finalize_req073():
    def hook(config, state):
        r = state.get("_r73")
        if r is not None and state.get("master", False):
            for w in ("w1", "w2", "w3", "w4"): r[w].close()
            state["print_log"]("req073 capture closed (4 stages)")
        return state
    return hook


_HOOKS["tag_req073_capture"] = tag_req073_capture
_HOOKS["req073_pre"] = req073_pre
_HOOKS["scale_grad_for_4B"] = scale_grad_for_4B
_HOOKS["req073_post"] = req073_post
_HOOKS["finalize_req073"] = finalize_req073
# ===== end REQ-073 =====
'''

h = open(HOOKS).read()
if "tag_req073_capture" in h:
    print("hooks.py already has REQ-073 hooks")
else:
    open(HOOKS, "w").write(h + CODE)
    print("patched hooks.py with REQ-073 capture hooks")
