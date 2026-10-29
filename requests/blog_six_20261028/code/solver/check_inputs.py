"""CPU-only input check for one solve: validation shard and the eight capture shards.

Writes OUT_DIR/input_manifest.json in the production layout (pins + tokens + update);
the solver reads it only when resuming. Needs numpy; torch only for --inspect.
"""
import argparse, hashlib, json, sys
from pathlib import Path

VALIDATION_SHA256 = "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640"
VALIDATION_BYTES = 200001024
SOURCE_HEAD = "a3e212ec312e0921a9875807ad5a820c1912564b72b6f8bb93fada3ca452a56f"  # sha256 of the production source.tgz
TOKENS = 16777216


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--update", type=int, required=True)
    ap.add_argument("--skip-hash", action="store_true", help="trust index.jsonl hashes; check sizes only")
    ap.add_argument("--inspect", action="store_true", help="also open rank0 with torch and print the saved settings")
    a = ap.parse_args()
    problems = []

    # Validation shard: fixed bytes, fixed sha256, fineweb header (magic, version, token count).
    import numpy as np
    size = a.validation.stat().st_size
    if size != VALIDATION_BYTES:
        problems.append(f"validation bytes {size} != {VALIDATION_BYTES}")
    header = np.fromfile(a.validation, dtype=np.int32, count=256)
    if len(header) != 256 or header[0] != 20240520 or header[1] != 1 or header[2] < TOKENS + 1:
        problems.append("validation header is not a fineweb shard with >= 16M+1 tokens")
    if not a.skip_hash:
        digest = sha256(a.validation)
        if digest != VALIDATION_SHA256:
            problems.append(f"validation sha256 {digest} != {VALIDATION_SHA256}")

    # Capture shards: rank0..7/boundary_N.pt, pinned by index.jsonl when present.
    index = a.capture / "index.jsonl"
    pinned = {}
    if index.exists():
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("kind") == "boundary" and d.get("update") == a.update:
                pinned[d["file"]] = d
    pins = {}
    for rank in range(8):
        rel = f"rank{rank}/boundary_{a.update:06d}.pt"
        path = a.capture / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
            continue
        entry = pinned.get(rel)
        if entry is not None and path.stat().st_size != entry["bytes"]:
            problems.append(f"{rel}: bytes {path.stat().st_size} != index {entry['bytes']}")
        if a.skip_hash and entry is not None:
            pins[rel] = entry["sha256"]
        else:
            pins[rel] = sha256(path)
            if entry is not None and pins[rel] != entry["sha256"]:
                problems.append(f"{rel}: sha256 differs from index.jsonl")
    if index.exists() and len(pinned) != 8:
        problems.append(f"index.jsonl lists {len(pinned)} boundary shards for update {a.update}, expected 8")

    if a.inspect and not problems:
        import torch
        x = torch.load(a.capture / f"rank0/boundary_{a.update:06d}.pt", map_location="cpu", weights_only=True, mmap=True)
        muon = [s for s in x["settings"] if s["optimizer"] in ("muon", "muonh")]
        tokens = x.get("training_batch_tokens") if x.get("format") == 3 else x["batch_receipts"]["used"]["batch_tokens"]
        print(json.dumps(dict(update=x["update"], format=x.get("format"), training_batch_tokens=tokens,
                              muon_groups=[s["groups"] for s in muon],
                              owned_muon_matrices_rank0=len(x["post_insertion_prepolar_momentum"]),
                              model_tensors=len(x["model_before"])), indent=2))
        if x["update"] != a.update:
            problems.append(f"rank0 shard says update {x['update']}")
        if tokens != 524288:
            problems.append(f"training_batch_tokens {tokens} != 524288")

    if problems:
        print("INPUT CHECK FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)
    pins["validation"] = VALIDATION_SHA256
    manifest = dict(source_head=SOURCE_HEAD, pins=pins, tokens=TOKENS, update=a.update)
    a.output.mkdir(parents=True, exist_ok=True)
    existing = a.output / "input_manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        print("INPUT CHECK FAILED: output dir already has a different input_manifest.json", file=sys.stderr)
        sys.exit(1)
    existing.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"inputs ok: update {a.update}, 8 shards, validation sha256 verified={'no (skipped)' if a.skip_hash else 'yes'}")


if __name__ == "__main__":
    main()
