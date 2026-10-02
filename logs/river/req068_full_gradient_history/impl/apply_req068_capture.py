"""REQ-068: patch hooks.py with gradient-history capture hooks (idempotent; run from repo root).
 - capture_full_gradients  : pre_optimizer hook. On master (rank 0; grads are replicated via all_reduce SUM)
   clones every param's .grad at the capture point (after accumulation+sync, before apply_updates mutates
   .grad) into a GradientHistoryWriter. Builds the param schema (names w/ _orig_mod stripped, shapes,
   dtypes, flatten offsets, tying by data_ptr, sharding=replicated) on the first call.
 - finalize_gradient_capture : teardown hook. Closes the writer -> writes manifest.json.
The writer/reader live in req068_capture.py (CPU-tested). Raw tensors stay off Git; only manifest+schema do.
"""
import os
HOOKS = "records/track_3_optimization/harness/hooks.py"

CODE = '''

# ===== REQ-068 full gradient history capture =====
import sys as _req068_sys
_req068_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                        "logs", "river", "req068_full_gradient_history", "impl"))
_req068_sys.path.insert(0, "logs/river/req068_full_gradient_history/impl")


def _req068_clean_named_params(model):
    """named_parameters with the torch.compile _orig_mod. prefix stripped."""
    out = []
    for n, p in model.named_parameters():
        out.append((n[len("_orig_mod."):] if n.startswith("_orig_mod.") else n, p))
    return out


def capture_full_gradients(*, out_dir: str, chunk_steps: int = 50,
                           tokens_per_update: int = None):
    """pre_optimizer hook: capture every parameter's full gradient at every update (master only).
    Grads are replicated across ranks (all_reduce SUM), so rank-0 capture is complete and lossless."""
    from req068_capture import GradientHistoryWriter, build_param_schema

    def hook(config, state):
        if not state.get("master", False):
            return state
        named = _req068_clean_named_params(state["model"])
        writer = state.get("_req068_writer")
        if writer is None:
            schema = build_param_schema(named)
            # tying: group params sharing storage; sharding: replicated (grads all_reduced)
            ptr = {}
            for n, p in named:
                ptr.setdefault(p.data_ptr(), []).append(n)
            schema["tying"] = {k: v for k, v in ((names[0], names) for names in ptr.values()) if len(v) > 1}
            schema["sharding"] = "replicated_all_reduce_sum"
            prov = {"code_sha": "365c392d695f95dc9a4fb89095e85a6a7b5d551e",
                    "run_id": config.get("run_id"), "seed": config.get("seed"),
                    "train_steps": config.get("train_steps"),
                    "batch_tokens": config.get("batch_tokens"),
                    "microbatch_sequences": config.get("microbatch_sequences"),
                    "world_size": state.get("world_size")}
            tpu = tokens_per_update if tokens_per_update is not None else config.get("batch_tokens")
            writer = GradientHistoryWriter(out_dir, schema, chunk_steps=chunk_steps,
                                           tokens_per_update=tpu, provenance=prov, grad_reduction="sum")
            state["_req068_writer"] = writer
            state["print_log"](f"req068 capture: writer at {out_dir}, {len(named)} params, "
                               f"chunk_steps={chunk_steps}, tokens/update={tpu}", console=True)
        grads = {n: p.grad for n, p in named}  # None entries recorded as explicit None flags
        writer.record(state["step"], grads)
        return state
    return hook


def finalize_gradient_capture():
    """teardown hook: flush + write the manifest."""
    def hook(config, state):
        w = state.get("_req068_writer")
        if w is not None:
            m = w.close()
            state["print_log"](f"req068 capture closed: {m['n_steps']} steps, "
                               f"{m['total_bytes']} bytes, {len(m['chunks'])} chunks", console=True)
        return state
    return hook


_HOOKS["capture_full_gradients"] = capture_full_gradients
_HOOKS["finalize_gradient_capture"] = finalize_gradient_capture
# ===== end REQ-068 =====
'''

t = open(HOOKS).read()
if "capture_full_gradients" in t:
    print("hooks.py already has REQ-068 capture hooks")
else:
    open(HOOKS, "w").write(t + CODE)
    print("patched hooks.py with capture_full_gradients + finalize_gradient_capture")
