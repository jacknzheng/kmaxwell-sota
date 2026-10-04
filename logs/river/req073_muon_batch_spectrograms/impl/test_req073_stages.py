"""CPU test: split_capture is bit-identical to stock muon_update (update/grad/momentum) + exposes stage2/3."""
import sys, os, torch
sys.path.insert(0, "/root/kmaxwell-sota/records/track_3_optimization")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                "records", "track_3_optimization"))  # fallback
try:
    from optimizers.muon import muon_update
except Exception:
    sys.path.insert(0, "/Users/jerryhong/.claude/jobs/e9994aef/tmp/harness365/records/track_3_optimization")
    from optimizers.muon import muon_update
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req073_stages import split_capture
ok = True
for (rows, cols) in [(8, 8), (16, 8), (8, 16)]:
    for mu in (0.95, 0.0):
        torch.manual_seed(0)
        g1 = torch.randn(rows, cols); m1 = torch.randn(rows, cols) * 0.1
        g2 = g1.clone(); m2 = m1.clone()
        u1 = muon_update(g1, m1, mu=mu)
        u2, s2, s3 = split_capture(muon_update, g2, m2, mu=mu)
        good = (u1 - u2).abs().max().item() == 0 and (m1 - m2).abs().max().item() == 0 \
            and (g1 - g2).abs().max().item() == 0 and (u2 - s3).abs().max().item() == 0 \
            and bool(torch.isfinite(s2).all()) and float(s2.abs().max()) > 0
        print(f"({rows}x{cols}) mu={mu}: {'PASS' if good else 'FAIL'}"); ok &= good
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
