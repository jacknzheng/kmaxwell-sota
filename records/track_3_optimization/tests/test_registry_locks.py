# Surface locks: registry names are public config API (every YAML references
# them), so a rename must fail a test, not silently break reproductions. Also
# locks the factory discipline: every hook factory takes keyword-only
# hyperparameters and returns a hook whose parameters are exactly (config, state).

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import hooks
import optimizers

LOCKED_HOOK_NAMES = {
    "open_rank_zero_log",
    "load_validation_tokens",
    "build_compiled_gpt",
    "seed_then_initialize_parameters",
    "attach_newton_muon_activation_stats",
    "assemble_grouped_optimizer",
    "open_training_batches",
    "broadcast_initial_parameters",
    "validate_at_step_boundaries",
    "cool_down_learning_rate",
    "print_training_progress",
    "record_paired_averages",
    "probe_secant_solve_at_cadence",
    "dump_secant_state_at_steps",
    "checkpoint_model_at_cadence",
    "dump_training_state_at_steps",
    "load_training_state",
    "record_reference_gradients_in_windows",
    "log_solve_statistics_at_cadence",
    "mark_log_finished",
}

LOCKED_OPTIMIZER_NAMES = {"adamw", "muon", "bimaxwell_muon", "newton_bimaxwell_muon", "newton_short_ema_muon", "newton_weighted_decays_muon", "secant_gmres_muon", "sgd_blocks", "weighted_decays_muon", "scheduled_weights_muon", "annealed_weights_muon", "annealed_decay_muon"}


def test_hook_registry_is_locked():
    assert set(hooks._HOOKS) == LOCKED_HOOK_NAMES


def test_optimizer_registry_is_locked():
    assert set(optimizers._REGISTRY) == LOCKED_OPTIMIZER_NAMES


def test_every_hook_is_a_factory_over_keyword_only_hyperparams():
    for name, factory in hooks._HOOKS.items():
        for parameter in inspect.signature(factory).parameters.values():
            assert parameter.kind == inspect.Parameter.KEYWORD_ONLY, \
                f"{name}: factory hyperparams must be keyword-only, got {parameter}"


def test_unknown_hook_name_fails_at_bind_time():
    config = {"setup": [{"name": "definitely_not_a_hook"}], "pre_optimizer": [],
              "post_optimizer": [], "teardown": []}
    try:
        hooks.bind_sites(config)
    except ValueError as error:
        assert "definitely_not_a_hook" in str(error)
    else:
        raise AssertionError("bind_sites accepted an unknown hook name")


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
