"""CPU smoke tests for the REQ-068 capture writer/reader (no harness/GPU).
 T1 round-trip exact equality across steps, native dtype (bf16 + fp32) preserved.
 T2 no aliasing: mutating the source .grad after record() does not change the stored copy.
 T3 None (inactive) grads stored as explicit None, not fabricated zeros.
 T4 chunking: partial last chunk handled; manifest hashes verify; coverage contiguous (no gaps/dupes).
 T5 per-matrix series + full_step reconstruction.
 T6 verify() catches a corrupted chunk and a missing step.
"""
import os, sys, tempfile, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req068_capture import GradientHistoryWriter, GradientHistoryReader, build_param_schema

torch.manual_seed(0)
# synthetic model params: mixed dtypes + a scalar, mirroring the real model (bf16 matrices, fp32 vectors)
params = {
    "embed.weight": torch.randn(50, 8, dtype=torch.bfloat16),
    "blocks.0.attn.q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    "blocks.0.norm1.gains": torch.randn(8, dtype=torch.float32),
    "proj.weight": torch.randn(50, 8, dtype=torch.bfloat16),
}
schema = build_param_schema(params.items())
STEPS = 7; CHUNK = 3  # -> chunks [0,1,2],[3,4,5],[6] (partial last)
truth = {}  # step -> name -> tensor or None

d = tempfile.mkdtemp(prefix="req068_test_")
w = GradientHistoryWriter(d, schema, chunk_steps=CHUNK, tokens_per_update=524288,
                          provenance={"test": True}, grad_reduction="sum")
for s in range(STEPS):
    grads = {}
    for name, p in params.items():
        if name == "blocks.0.norm1.gains" and s == 4:
            grads[name] = None  # inactive at step 4
        else:
            grads[name] = torch.randn_like(p)
    truth[s] = {k: (None if v is None else v.clone()) for k, v in grads.items()}
    w.record(s, grads)
    # T2: mutate the source tensors in place AFTER record -> stored copy must be unaffected
    for v in grads.values():
        if v is not None:
            v.add_(100.0)
manifest = w.close()

r = GradientHistoryReader(d)
ok = True

# T1 + T2 + T3: exact round-trip, native dtype, no aliasing, None preserved
for s in range(STEPS):
    for name in params:
        got = r.grad(s, name); exp = truth[s][name]
        if exp is None:
            if got is not None: print(f"T3 FAIL s{s} {name}: expected None"); ok = False
            continue
        if got.dtype != exp.dtype:
            print(f"T1 FAIL dtype s{s} {name}: {got.dtype} != {exp.dtype}"); ok = False
        if not torch.equal(got, exp):
            print(f"T1/T2 FAIL value s{s} {name} (max|d|={(got.float()-exp.float()).abs().max()})"); ok = False
print(f"T1 round-trip exact + native dtype, T2 no-aliasing, T3 None flags: {'PASS' if ok else 'FAIL'}")

# T4: chunk structure + manifest verify
exp_chunks = 3
c_ok = len(manifest["chunks"]) == exp_chunks and manifest["chunks"][-1]["n_steps"] == 1
v = r.verify(expected_steps=list(range(STEPS)))
print(f"T4 chunking ({len(manifest['chunks'])} chunks, last n={manifest['chunks'][-1]['n_steps']}) + "
      f"verify {v['ok']} {v['problems']}: {'PASS' if c_ok and v['ok'] else 'FAIL'}")
ok &= c_ok and v["ok"]

# T5: series + full_step
series = list(r.series("blocks.0.attn.q.weight"))
s5 = len(series) == STEPS and all(torch.equal(series[s][1], truth[s]["blocks.0.attn.q.weight"]) for s in range(STEPS))
fs = r.full_step(2); f5 = set(fs) == set(params) and torch.equal(fs["embed.weight"], truth[2]["embed.weight"])
print(f"T5 per-matrix series + full_step: {'PASS' if s5 and f5 else 'FAIL'}")
ok &= s5 and f5

# T6: verify catches corruption + missing step
with open(os.path.join(d, manifest["chunks"][0]["file"]), "ab") as f:
    f.write(b"corrupt")
v2 = r.verify(expected_steps=list(range(STEPS + 1)))  # expect an extra step that isn't there
caught = (not v2["ok"]) and any("hash" in p for p in v2["problems"]) and any("missing" in p for p in v2["problems"])
print(f"T6 verify catches corruption+missing: {'PASS' if caught else 'FAIL'} {v2['problems']}")
ok &= caught

print(f"\n{'PASS' if ok else 'FAIL'}: REQ-068 capture writer/reader ({manifest['total_bytes']} bytes, "
      f"{manifest['n_steps']} steps, {manifest['n_params']} params)")
sys.exit(0 if ok else 1)
