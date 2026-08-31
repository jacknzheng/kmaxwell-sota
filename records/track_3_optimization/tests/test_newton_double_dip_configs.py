import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_analysis.make_newton_double_dip_configs import ALPHAS, FORKS, KERNELS, config_for


def test_grid_names_map_to_declared_alpha_and_kernel():
    seen = set()
    for fork, stop in FORKS:
        for kernel in KERNELS:
            for alpha in ALPHAS:
                config = config_for(fork, stop, alpha, kernel, "state")
                block = config["optimizer_groups"][2]
                assert block["hyperparams"]["newton_alpha"] == alpha
                expected = ("newton_bimaxwell_muon" if kernel == "record"
                            else "newton_short_ema_muon")
                assert block["optimizer"] == expected
                assert config["run_id"] not in seen
                seen.add(config["run_id"])
    assert len(seen) == 12
