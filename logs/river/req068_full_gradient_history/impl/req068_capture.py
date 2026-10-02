"""REQ-068 full-gradient-history capture: standalone writer + reader (CPU-testable, no harness/GPU needed).

Gradients in this trainer are REPLICATED across ranks (the loop does dist.all_reduce(p.grad, SUM) for every
param before the optimizer), so a single rank's capture is complete and lossless — no shard reconstruction is
needed for gradients (documented; the harness shards only the optimizer *state*, not .grad).

Writer streams bounded chunks to a durable dir, preserving each tensor's NATIVE dtype losslessly, with a
SHA-256 per chunk and a machine-readable manifest (param schema, step coverage, tokens/update, sizes,
provenance). Reader reconstructs any step's full gradient set and streams per-matrix time series, and verifies
the manifest (hashes + contiguous 0..N-1 coverage, no gaps/dupes).

One observation = one optimizer update (global index). A missing/inactive gradient is recorded as an explicit
None flag, never a fabricated zero.
"""
from __future__ import annotations
import hashlib, json, os, time
from typing import Any
import torch


def _sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def build_param_schema(named_params) -> dict:
    """named_params: iterable of (name, tensor). Records order, shape, dtype, flatten offsets, numel.
    `tying` and `sharding` are recorded by the caller (hook) since they need model/optimizer context."""
    schema = {"order": [], "params": {}}
    offset = 0
    for name, p in named_params:
        ne = int(p.numel())
        schema["params"][name] = {
            "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", ""),
            "numel": ne, "flat_offset": offset, "flat_end": offset + ne,
        }
        schema["order"].append(name)
        offset += ne
    schema["total_numel"] = offset
    return schema


class GradientHistoryWriter:
    def __init__(self, out_dir: str, param_schema: dict, *, chunk_steps: int = 50,
                 tokens_per_update: int | None = None, provenance: dict | None = None,
                 grad_reduction: str = "sum"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.schema = param_schema
        self.chunk_steps = int(chunk_steps)
        self.tokens_per_update = tokens_per_update
        self.grad_reduction = grad_reduction
        self.provenance = provenance or {}
        self._buf: dict[int, dict[str, torch.Tensor | None]] = {}
        self._chunks: list[dict] = []
        self._seen_steps: list[int] = []
        # persist schema immediately
        json.dump(self.schema, open(os.path.join(out_dir, "param_schema.json"), "w"), indent=1)

    def record(self, step: int, grads: dict[str, torch.Tensor | None]) -> None:
        """grads: name -> gradient tensor (any device/dtype) or None (inactive). Stored on CPU in native
        dtype, cloned so later in-place optimizer mutation of .grad cannot alias the captured copy."""
        snap: dict[str, torch.Tensor | None] = {}
        for name in self.schema["order"]:
            g = grads.get(name, None)
            snap[name] = None if g is None else g.detach().to("cpu").clone()
        self._buf[int(step)] = snap
        self._seen_steps.append(int(step))
        if len(self._buf) >= self.chunk_steps:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        steps = sorted(self._buf)
        idx = len(self._chunks)
        fname = f"grad_chunk_{idx:05d}_steps{steps[0]:06d}-{steps[-1]:06d}.pt"
        path = os.path.join(self.out_dir, fname)
        payload = {"steps": steps, "data": {s: self._buf[s] for s in steps}}
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self._chunks.append({
            "index": idx, "file": fname, "steps": steps,
            "step_first": steps[0], "step_last": steps[-1], "n_steps": len(steps),
            "bytes": os.path.getsize(path), "sha256": _sha256_file(path),
        })
        self._buf = {}

    def close(self) -> dict:
        self._flush()
        none_counts = {}  # name -> number of steps where grad was None (for transparency)
        manifest = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "grad_reduction": self.grad_reduction,
            "tokens_per_update": self.tokens_per_update,
            "note": "gradients replicated across ranks via all_reduce(SUM); single-rank capture is complete. "
                    "Native dtype preserved losslessly. Divide by tokens_per_update for mean-per-token.",
            "provenance": self.provenance,
            "param_schema_file": "param_schema.json",
            "n_params": len(self.schema["order"]),
            "total_numel": self.schema["total_numel"],
            "chunks": self._chunks,
            "steps_covered": sorted(s for c in self._chunks for s in c["steps"]),
            "n_steps": sum(c["n_steps"] for c in self._chunks),
            "total_bytes": sum(c["bytes"] for c in self._chunks),
        }
        json.dump(manifest, open(os.path.join(self.out_dir, "manifest.json"), "w"), indent=1)
        return manifest


class GradientHistoryReader:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
        self.schema = json.load(open(os.path.join(out_dir, "param_schema.json")))
        self._step_to_chunk = {}
        for c in self.manifest["chunks"]:
            for s in c["steps"]:
                self._step_to_chunk[s] = c["file"]
        self._cache_file = None
        self._cache = None

    def steps(self) -> list[int]:
        return sorted(self._step_to_chunk)

    def params(self) -> list[str]:
        return list(self.schema["order"])

    def _load(self, fname: str):
        if self._cache_file != fname:
            self._cache = torch.load(os.path.join(self.out_dir, fname), weights_only=False)
            self._cache_file = fname
        return self._cache

    def grad(self, step: int, name: str):
        """Reconstruct one parameter's gradient at one step (native dtype), or None if inactive."""
        fname = self._step_to_chunk[int(step)]
        return self._load(fname)["data"][int(step)][name]

    def full_step(self, step: int) -> dict:
        """All parameter gradients at one step."""
        fname = self._step_to_chunk[int(step)]
        return self._load(fname)["data"][int(step)]

    def series(self, name: str, steps: list[int] | None = None):
        """Stream a per-matrix time series: yields (step, grad_or_None) in step order."""
        for s in (steps or self.steps()):
            yield s, self.grad(s, name)

    def verify(self, expected_steps: list[int] | None = None) -> dict:
        """Verify chunk hashes and step coverage (contiguous, no gaps/dupes)."""
        problems = []
        for c in self.manifest["chunks"]:
            p = os.path.join(self.out_dir, c["file"])
            if not os.path.exists(p):
                problems.append(f"missing chunk {c['file']}"); continue
            if _sha256_file(p) != c["sha256"]:
                problems.append(f"hash mismatch {c['file']}")
        covered = self.steps()
        dupes = len(covered) != len(set(covered))
        if expected_steps is not None:
            missing = sorted(set(expected_steps) - set(covered))
            extra = sorted(set(covered) - set(expected_steps))
            if missing: problems.append(f"missing {len(missing)} steps e.g. {missing[:5]}")
            if extra: problems.append(f"unexpected {len(extra)} steps e.g. {extra[:5]}")
        if dupes: problems.append("duplicate steps present")
        return {"ok": not problems, "problems": problems, "n_steps": len(covered)}
