"""REQ-068 amendment: capture actual parameter displacements delta_t = theta_(t+1) - theta_t alongside
gradients, for the three 64-step windows. Idempotent patch to hooks.py (run from repo root).

Two hooks (master only; params are replicated after Muon's all_gather):
 - snapshot_weights_pre_update (pre_optimizer): store theta_t = {name: p.float().cpu()} BEFORE apply_updates.
 - capture_displacement (post_optimizer): after apply_updates + zero_grad, delta = theta_(t+1) - theta_t in
   fp32 (sufficient precision to represent the bf16 diff), written via the generic GradientHistoryWriter to
   a separate out_dir. Records the calculation + dtype. delta carries optimizer conditioning + LR + decay +
   rounding — kept distinct from the raw gradient.
Reuses req068_capture.GradientHistoryWriter/Reader (already CPU-tested). Raw tensors stay off Git.
"""
import os
HOOKS = "records/track_3_optimization/harness/hooks.py"

CODE = '''

# ===== REQ-068 displacement capture =====
def snapshot_weights_pre_update():
    """pre_optimizer: snapshot theta_t (master) before apply_updates mutates the params."""
    def hook(config, state):
        if not state.get("master", False):
            return state
        named = _req068_clean_named_params(state["model"])
        state["_req068_theta_before"] = {n: p.detach().float().cpu().clone() for n, p in named}
        return state
    return hook


def capture_displacement(*, out_dir: str, chunk_steps: int = 64):
    """post_optimizer: delta_t = theta_(t+1) - theta_t (fp32), written via GradientHistoryWriter."""
    from req068_capture import GradientHistoryWriter, build_param_schema

    def hook(config, state):
        if not state.get("master", False):
            return state
        before = state.get("_req068_theta_before")
        if before is None:
            return state
        named = _req068_clean_named_params(state["model"])
        delta = {n: (p.detach().float().cpu() - before[n]) for n, p in named if n in before}
        writer = state.get("_req068_disp_writer")
        if writer is None:
            schema = build_param_schema([(n, before[n]) for n, _ in named if n in before])
            schema["sharding"] = "replicated_post_allgather"
            schema["stream"] = "displacement_delta_theta_fp32"
            prov = {"code_sha": "365c392d695f95dc9a4fb89095e85a6a7b5d551e",
                    "run_id": config.get("run_id"), "seed": config.get("seed"),
                    "note": "delta_t = theta_(t+1)-theta_t computed fp32 from pre/post weights; "
                            "includes optimizer conditioning + LR + weight decay + rounding."}
            writer = GradientHistoryWriter(out_dir, schema, chunk_steps=chunk_steps,
                                           tokens_per_update=config.get("batch_tokens"),
                                           provenance=prov, grad_reduction="displacement")
            state["_req068_disp_writer"] = writer
            state["print_log"](f"req068 displacement: writer at {out_dir}", console=True)
        writer.record(state["step"], delta)
        state["_req068_theta_before"] = None
        return state
    return hook


def finalize_displacement_capture():
    def hook(config, state):
        w = state.get("_req068_disp_writer")
        if w is not None:
            m = w.close()
            state["print_log"](f"req068 displacement closed: {m['n_steps']} steps, {m['total_bytes']} bytes")
        return state
    return hook


_HOOKS["snapshot_weights_pre_update"] = snapshot_weights_pre_update
_HOOKS["capture_displacement"] = capture_displacement
_HOOKS["finalize_displacement_capture"] = finalize_displacement_capture
# ===== end REQ-068 displacement =====
'''

t = open(HOOKS).read()
if "capture_displacement" in t:
    print("hooks.py already has REQ-068 displacement hooks")
elif "_req068_clean_named_params" not in t:
    raise SystemExit("FATAL: apply the gradient capture patch (apply_req068_capture.py) first")
else:
    open(HOOKS, "w").write(t + CODE)
    print("patched hooks.py with displacement capture hooks")
