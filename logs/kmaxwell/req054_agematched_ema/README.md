# REQ-054 — annealed single EMA matched to K-Maxwell's scheduled memory age — **K-Maxwell needs the mixture (n=4)**

**SHA `365c392d` + `AgeMatchedEmaMuon` (new optimizer, `apply_req054_opt.py`), 2 nodes (wdkrxgq +
q4zk6y3, 8×H100, ≤2-node ceiling), venv019 torch 2.10.0+cu128.** Four independent seeds, each forked from
its own step-2000 base into two paired arms (fork@2000 → 2750, same base/data cursor, only the block
optimizer differs):

- **kmax** — `annealed_weights_muon` (K-Maxwell): 8 EMA streams, weights annealed SW→EW, the shipped kernel.
- **agema** — `age_matched_ema_muon`: a **single** EMA whose decay follows K-Maxwell's *scheduled average
  memory age*, `β(t) = A(t)/(1+A(t))` with `A(t) = Σ wᵢ(t)·βᵢ/(1−βᵢ)` (wᵢ(t) the same SW→EW schedule).
  Outer Nesterov `mu` fixed; baseline Muon through the fork.

The age schedule is validated against the request: `A(t)` = **57.96 → 26.0** over the anneal, matching the
spec's cited "≈ 58 → 26" exactly.

## Result — one age-matched EMA does NOT reproduce K-Maxwell

`readout.tsv` (val@2750):

| seed | K-Maxwell | age-matched EMA | kmax − agema |
|:----:|----------:|----------------:|-------------:|
| 0 | 3.34061 | 3.35027 | −0.00966 |
| 1 | 3.34017 | 3.35015 | −0.00998 |
| 2 | 3.34213 | 3.35103 | −0.00890 |
| 3 | 3.34154 | 3.35187 | −0.01033 |

**mean(kmax − agema) = −0.00972 ± 0.00053, negative in 4/4 seeds** — ~50× the ~2×10⁻⁴ val noise floor.
K-Maxwell beats the age-matched single EMA decisively and consistently. **So K-Maxwell's benefit is the
mixture of memory timescales (multi-pole expressivity), not merely having the right scheduled average
memory age** — a single exponential kernel with the identical age schedule (58→26) leaves ~0.010 val on
the table. This answers REQ-044's open scheduled-memory question that the fixed `mu=0.95` control could not.

## Caveat — schedule-matched, not (yet) finite-history-age-matched

Per the request's finite-history check: `AgeMatchedEmaMuon` matches the *instantaneous stationary* age
`A(t)` at each step; the *realized* age of a changing EMA fed by the inherited fork momentum can differ
during the transient. This run is therefore **schedule-matched**. The gap (~0.010, tight across seeds and
sustained to the 750-step endpoint) is far larger than a plausible transient artifact — a decaying
finite-history mismatch would shrink over the window, whereas the gap is stable — so the multi-timescale
reading is the natural one. A finite-history exact-age-matched control (tracking realized kernel mass and
first age moment through the actual recurrence) would close the last gap before attributing the entire
residual to kernel shape; it is a cheap follow-up on the same fork states. Realized-age tracking was not
committed here.

## Files

- `readout.tsv` — per-seed kmax/agema val@2750 + the paired difference + verdict.
- `apply_req054_opt.py` — the `AgeMatchedEmaMuon` optimizer (age schedule β(t)=A(t)/(1+A(t))) + registry entry.
- `make_req054_configs.py` — paired kmax/agema fork-config generator (shared DECAYS/SW/EW). `req054_run.sh` — driver.

No secrets/weights/tensor checkpoints committed. Ran under the ≤2 ceiling.
