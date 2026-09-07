# REQ-052 — matched uniform-vs-mixed LR controls — **band-67 writer/internal contrast is mixed-LR-only (n=4)**

**SHA `25d3208`, 2 nodes (wdkrxgq + q4zk6y3, 8×H100, ≤2-node ceiling), venv019 torch 2.10.0+cu128.**
Four independent seeds (bases in `provenance.tsv`), each forked from its own step-2000 base into **5
control arms** (fork@2000→2750, combined curvature+activation probe at 2050 & 2750, same operator as
REQ-051):

- **uniform-Muon** — all 72 Muon multipliers = 0.65 / 1.00 / 1.70, non-Muon AdamW at reference (`u065/u100/u170`);
- **full-global** — all 72 Muon = 0.65 / 1.70 **and** all non-Muon AdamW group LRs ×0.65 / ×1.70 (`fg065/fg170`), with `u100` the shared 1.00 control.

The test: does the band-67 **writer/internal** LR-response contrast — strongly positive under REQ-023's
*mixed* per-matrix LR (+0.92 / +1.17) but negative under REQ-035's *global* LR (−0.19…−0.20) — reproduce
under a directly-controlled uniform/global LR ladder? (Writers = `attn.proj`, `mlp.proj`; internal = q, k,
v, `mlp.fc`.) `k = −d log top_eigenvalue / d log(uniform mult)` over each 3-point ladder, per seed at 2750.

## Result (`readout.tsv`, n=4 @2750)

| ladder | writers − internal | v − (q,k) |
|:-------|-------------------:|----------:|
| uniform-Muon | **+0.028** (sd 0.05) | **+0.298** (sd 0.07) |
| full-global | **+0.108** (sd 0.08) | **+0.415** (sd 0.16) |
| *REQ-023 mixed LR (ref)* | +0.92 / +1.17 | negative |
| *REQ-035 global LR (ref)* | −0.19 … −0.20 | +0.36 … +0.46 |

**The writer/internal contrast collapses to ≈0 under both uniform-Muon (+0.03) and full-global (+0.11)
LR — it is nowhere near the +0.9…+1.2 seen under REQ-023's mixed per-matrix LR.** So the band-67
writer/internal effect is **specific to mixed per-matrix LR** and does not transfer to a uniform or
full-global LR treatment. Its previously-untested four-seed criterion, evaluated here under directly-
controlled global LR, is not met: the effect is a property of the *mixed-LR design*, not of writer vs
internal matrices per se.

**The v − (q,k) contrast is positive under both uniform (+0.30) and full-global (+0.42) LR**, matching
REQ-035's global-LR sign (+0.36…+0.46) and opposite REQ-023's mixed-LR (negative). So this contrast is
also LR-design-dependent, and REQ-052 reproduces the global-LR sign at n=4 with directly-controlled LRs.

Full-global (Muon + AdamW scaled together) shifts both contrasts slightly more positive than uniform-Muon
(writers−internal +0.11 vs +0.03; v−qk +0.42 vs +0.30), i.e. adding the non-Muon LR scaling nudges them up,
but neither approaches the mixed-LR magnitude.

## Caveats

- n=4 independent bases; k from a 3-point ladder per matrix (uniform mult shared across all matrices in an
  arm — a genuine global treatment, unlike REQ-051's per-matrix Latin square). Judged per seed.
- Changing LR also changes realized decoupled weight decay; this is *not* framed as isolating the
  gradient-driven vs weight-decay channels (per the request). Combined curv+act probe at 32k tokens.
- REQ-023's five-checkpoint "seeds" were dependent; this is the first genuine 4-seed uniform-LR test.

## Files

- `readout.tsv` — writer/internal + v-(q,k) contrasts, uniform & full-global, per seed + pooled.
- `analyze_req052.py` — reproducible from the raw JSONs. `make_req052_arms.py` / `req052_run.sh` — generator + driver.
- `provenance.tsv` — 4 base hashes + val@2000.
- `raw_json/req052_s{0..3}_{u065,u100,u170,fg065,fg170}_step{2050,2750}_{curv,act}.json` — 80 combined-probe files.

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
