import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist


def protocol_hash(config):
    execution_keys = {"run_id", "output_dir", "checkpoint_dir", "checkpoint_every",
                      "resume_checkpoint", "resume_anchor", "start_step", "stop_step"}
    protocol = {k: v for k, v in config.items() if k not in execution_keys}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def save_checkpoint(config, state, boundary):
    folder = Path(config["checkpoint_dir"]) / f"boundary_{boundary}"
    folder.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    payload = {"protocol_sha256": protocol_hash(config), "world_size": dist.get_world_size(),
               "boundary": boundary, "model": state["model"].state_dict(),
               "optimizers": [opt.state_dict() for _, opt in state["optimizer"].groups],
               "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
    outer = getattr(state["optimizer"], "outer", None)
    if outer is not None:
        payload["outer_optimizer"] = outer.state_dict()
    if "soap_record" in state:
        payload["soap_record"] = state["soap_record"].state_dict()
    temporary = folder / f"rank{rank}.pt.tmp"
    destination = folder / f"rank{rank}.pt"
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    dist.barrier()
    if rank == 0:
        receipt = {"boundary": boundary, "world_size": dist.get_world_size(),
                   "protocol_sha256": protocol_hash(config)}
        (folder / "COMPLETE.json").write_text(json.dumps(receipt, sort_keys=True))
    dist.barrier()


def load_checkpoint(config, state):
    folder = Path(config["resume_checkpoint"])
    receipt = json.loads((folder / "COMPLETE.json").read_text())
    if receipt["world_size"] != dist.get_world_size() or receipt["protocol_sha256"] != config.get("resume_protocol_sha256", protocol_hash(config)):
        raise ValueError("resume protocol or world size changed")
    payload = torch.load(folder / f"rank{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    if payload["protocol_sha256"] != receipt["protocol_sha256"] or payload["boundary"] != receipt["boundary"]:
        raise ValueError("checkpoint shard does not match completion receipt")
    outer = getattr(state["optimizer"], "outer", None)
    if (outer is not None) != ("outer_optimizer" in payload):
        raise ValueError("checkpoint outer-optimizer lifecycle changed")
    if ("soap_record" in state) != ("soap_record" in payload):
        raise ValueError("checkpoint SOAP lifecycle changed")
    state["model"].load_state_dict(payload["model"])
    for (_, optimizer), saved in zip(state["optimizer"].groups, payload["optimizers"], strict=True):
        optimizer.load_state_dict(saved)
    if outer is not None:
        outer.load_state_dict(payload["outer_optimizer"])
    if "soap_record" in state:
        state["soap_record"].load_state_dict(payload["soap_record"])
    torch.set_rng_state(payload["rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    return payload["boundary"]
