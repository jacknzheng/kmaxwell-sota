import torch


class FiniteHistoryOptimizer(torch.optim.Optimizer):
    def load_state_dict(self, state_dict):
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("checkpoint parameter group count changed")
        for saved, current in zip(saved_groups, self.param_groups):
            if saved.get("scheduler_config_sha256") != current.get("scheduler_config_sha256"):
                raise ValueError("checkpoint scheduler configuration differs from the constructor")
        saved_ids = [i for group in state_dict["param_groups"] for i in group["params"]]
        live_params = [p for group in self.param_groups for p in group["params"]]
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict casts floating tensors to parameter dtype.
        # Restore histories/bases directly from the original checkpoint, never
        # through that lossy BF16 round trip. Scheduler configuration is supplied
        # by the caller and must be separately bound to the reproduction config.
        for saved_id, p in zip(saved_ids, live_params):
            for key, value in state_dict["state"].get(saved_id, {}).items():
                if torch.is_tensor(value) and (key.endswith("_history") or key in ("q_row", "q_col", "variance")):
                    self.state[p][key] = value.to(device=p.device, dtype=torch.float32).clone()
