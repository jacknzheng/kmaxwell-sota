# Regression for the REQ-025 alpha>0 crash: the hook factory
# attach_newton_muon_activation_stats shares its public name with the
# optimizer-level helper it must call; an unaliased import is shadowed at call
# time and raises TypeError on every nonzero-alpha arm. The alpha=0 gate never
# executes this path, so only a test that ENABLES the hook catches it.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import hooks
from harness.model_gpt import GPT


def _config(alpha: float) -> dict:
    return {"optimizer_groups": [
        {"pattern": r"^blocks\..*\.weight$", "optimizer": "newton_bimaxwell_muon",
         "hyperparams": {"newton_alpha": alpha}}]}


def test_nonzero_alpha_installs_activation_hooks_on_the_model():
    model = GPT(vocab_size=256, num_layers=1, model_dim=64)
    state = hooks._HOOKS["attach_newton_muon_activation_stats"]()(_config(0.5), {"model": model})
    handles = state["newton_activation_hook_handles"]
    assert handles, "nonzero alpha must install activation-covariance hooks"
    for handle in handles:
        handle.remove()


def test_zero_alpha_installs_nothing():
    model = GPT(vocab_size=256, num_layers=1, model_dim=64)
    state = hooks._HOOKS["attach_newton_muon_activation_stats"]()(_config(0.0), {"model": model})
    assert "newton_activation_hook_handles" not in state


if __name__ == "__main__":
    test_nonzero_alpha_installs_activation_hooks_on_the_model()
    test_zero_alpha_installs_nothing()
    print("2 tests passed")
