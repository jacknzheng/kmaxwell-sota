"""REQ-063 restore fix: after load_training_state's built.load_state_dict(sd) overwrites each optimizer's
param-group hyperparameters from the checkpoint, REAPPLY the declared treatment hyperparameters (mu, etc.)
while keeping the restored state buffers and counters. Without this, a nomom arm (mu=0) loading a mixture
checkpoint (mu=0.95) is silently run at mu=0.95 (see impl/req063_stageB_restore.py).

Idempotent; patches harness/hooks.py:load_training_state in place. Run from the repo root.
"""
HOOKS = "records/track_3_optimization/harness/hooks.py"

OLD = '''        for (_, built), sd in zip(state["optimizer"].groups,
                                  shard["optimizer_state_dicts"]):
            built.load_state_dict(sd)'''

NEW = '''        for (_, built), sd in zip(state["optimizer"].groups,
                                  shard["optimizer_state_dicts"]):
            # REQ-063: snapshot the DECLARED param-group hyperparameters, because torch's
            # Optimizer.load_state_dict overwrites them from the checkpoint. Reapply after load so a
            # treatment arm (e.g. nomom mu=0) is not silently reset to the base's mu=0.95, while keeping
            # the restored state buffers/counters.
            declared = [{k: v for k, v in g.items() if k != "params"} for g in built.param_groups]
            built.load_state_dict(sd)
            for g, d in zip(built.param_groups, declared):
                g.update(d)
            state["print_log"](
                f"req063 restore: reapplied declared hyperparams to {built.__class__.__name__}: "
                f"mu={built.param_groups[0].get('mu')}", console=True)'''

t = open(HOOKS).read()
if "REQ-063: snapshot the DECLARED" in t:
    print("hooks.py already has REQ-063 restore fix")
elif OLD in t:
    open(HOOKS, "w").write(t.replace(OLD, NEW))
    print("patched hooks.py:load_training_state with REQ-063 restore fix")
else:
    raise SystemExit("FATAL: load_training_state restore block not found; harness may have drifted from 365c392d")
