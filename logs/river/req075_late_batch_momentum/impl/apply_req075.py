"""REQ-075 integration patch (idempotent; run from harness root AFTER apply_req073_capture.py).

Adds hooks REQ-075 needs on top of the reused REQ-073 four-stage capture:

- req075_set_muon_group(mu, base_lr, clear_buffer): a SETUP hook placed right after load_training_state.
  load_training_state restores the fork's optimizer param_groups (mu=0.95, initial_lr=0.025) and momentum
  buffers, which would override the nomom / half-LR arm configs -- so this re-asserts the Muon group's
  `mu`, `initial_lr` and `lr` to the arm's intended values, and (nomom arms) zeroes the inherited Muon
  momentum buffer. Mom arms call it with the same values (harmless) + clear_buffer=False to preserve the
  valid inherited buffer. scale_learning_rates(eta) later multiplies initial_lr, so setting initial_lr here
  fixes the constant checkpoint LR (and the half-LR control).

- cool_down_by_tokens(cooldown_frac, total_steps, tokens_per_step_base, fork_step): a TOKEN-INDEXED schedule
  for Stage 3, so B and 16B arms get identical eta at equal token positions (the step-indexed
  cool_down_learning_rate would misread a 16B arm that advances one state["step"] per 16B update). eta uses
  progress = (fork_step*base + tokens_since_fork)/(total_steps*base), same linear cool-down as the record.
  Not used by Stage 2 (constant LR via cool_down_learning_rate schedule_step_max=fork_step).
"""
import os
HOOKS = "records/track_3_optimization/harness/hooks.py"

CODE = '''

# ===== REQ-075 late-batch/momentum hooks =====
def req075_set_muon_group(*, mu, base_lr, clear_buffer):
    """Setup hook (place AFTER load_training_state): re-assert the Muon group's mu + base lr (load restores
    the fork's), and optionally zero the inherited momentum buffer for no-momentum arms."""
    def hook(config, state):
        _, muon = find_muon_family_group(state["optimizer"])
        for pg in muon.param_groups:
            pg["mu"] = mu
            pg["initial_lr"] = base_lr
            pg["lr"] = base_lr
        if clear_buffer:
            import torch as _t
            with _t.no_grad():
                for p in muon.param_groups[0]["params"]:
                    st = muon.state.get(p, {})
                    if "momentum" in st:
                        st["momentum"].zero_()
        state["print_log"](f"req075 set muon mu={mu} base_lr={base_lr} clear_buffer={clear_buffer}",
                           console=True)
        return state
    return hook


def cool_down_by_tokens(*, cooldown_frac, total_steps, tokens_per_step_base, fork_step):
    """Token-indexed cool-down: eta from token position so B and 16B arms match at equal tokens.
    progress = (fork_step*base + tokens_consumed_since_fork)/(total_steps*base). Linear cool-down (record)."""
    denom = float(total_steps * tokens_per_step_base)
    base_tokens = float(fork_step * tokens_per_step_base)
    def hook(config, state):
        batch = float(config["batch_tokens"])
        consumed = (state["step"] - fork_step) * batch       # tokens since fork (this arm's batch size)
        progress = (base_tokens + consumed) / denom
        progress = min(max(progress, 0.0), 0.999999)
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            u = (progress - (1 - cooldown_frac)) / cooldown_frac
            eta = 1 - u
        state["optimizer"].scale_learning_rates(eta)
        return state
    return hook

_HOOKS["req075_set_muon_group"] = req075_set_muon_group
_HOOKS["cool_down_by_tokens"] = cool_down_by_tokens
# ===== end REQ-075 =====
'''

h = open(HOOKS).read()
if "req075_set_muon_group" in h:
    print("hooks.py already has REQ-075 hooks")
else:
    open(HOOKS, "w").write(h + CODE)
    print("patched hooks.py with REQ-075 hooks")
