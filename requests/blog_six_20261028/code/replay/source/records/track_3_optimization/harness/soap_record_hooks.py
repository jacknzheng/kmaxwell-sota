import time
import torch
import torch.distributed as dist
from records.track_3_optimization.harness.soap_record_state import SoapRecordState
from records.track_3_optimization.harness.soap_record_protocol import soap341_validation_due


def install_soap_record_state():
    def hook(config, state):
        hidden = [opt for _, opt in state["optimizer"].groups if hasattr(opt, "step_count")]
        if len(hidden) != 1:
            raise ValueError("SOAP record lifecycle requires one hidden optimizer")
        state["soap_record"] = SoapRecordState(state["model"], hidden[0])
        return state
    return hook


def enable_output_kfac():
    def hook(config, state):
        state["soap_record"].enable(state["step"])
        return state
    return hook


def apply_output_kfac():
    def hook(config, state):
        state["soap_record"].apply(state["step"])
        return state
    return hook


def update_soap_tail():
    def hook(config, state):
        state["soap_record"].update_tail(state["step"] + 1)
        return state
    return hook


def close_soap_record_state():
    def hook(config, state):
        state["soap_record"].close()
        return state
    return hook


def validate_soap_record():
    def hook(config, state):
        boundary = state["step"] + 1 if "step" in state else config.get("start_step", 0)
        if "clock_start" not in state:
            state["training_time"] = 0.
            dist.barrier()
            state["clock_start"] = time.perf_counter()
        if not soap341_validation_due(boundary, stop_step=config.get("stop_step", 2600)):
            return state
        dist.barrier()
        state["training_time"] += time.perf_counter() - state["clock_start"]
        model = state["model"]
        model.eval()
        inputs, targets = state["validation_batch"]
        size = config["microbatch_sequences"]
        def evaluate():
            loss = 0
            assert len(inputs) % size == 0
            for i in range(len(inputs) // size):
                loss += model(inputs[i*size:(i+1)*size], targets[i*size:(i+1)*size])
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= config["val_tokens"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SOAP validation loss")
            return loss
        try:
            with torch.no_grad():
                raw = evaluate()
                ema = raw
                if state["soap_record"].tail is not None:
                    with state["soap_record"].tail_weights():
                        ema = evaluate()
            state["print_log"](f"step:{boundary}/{config['train_steps']} val_loss:{raw:.5f}"
                + f" val_ema_loss:{ema:.5f} train_time:{state['training_time']:.3f}s"
                + f" step_avg:{1000*state['training_time']/max(boundary, 1):.2f}ms", console=True)
            if state.get("wandb") is not None:
                state["wandb"].log(dict(step=boundary, val_loss=raw.item(),
                                         val_ema_loss=ema.item(), objective=ema.item()))
        finally:
            model.train()
        dist.barrier()
        state["clock_start"] = time.perf_counter()
        return state
    return hook
