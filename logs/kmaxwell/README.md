K-Maxwell seed-0 hunts (not n=8 records — those stay in `logs/wr/`).

Layout is **k / τ window / mean age**, not wave or stage.

```
kmaxwell/
  k{K}/t{min}-{max}/a{age}/     one folder per kernel
  k4/interiors_a35/             INT_bunch / INT_mid / INT_spread (custom ticks)
  k{K}/t{min}-{max}/mix_*/      same k/window/age, different mix shape
  stacks/                       SOAP / Aurora / Adam on a frozen kernel
  anneal_sweeps/                seed-0 anneal endpoints (C1_*)
  _hunts/waveN/                 original PLAN.txt + leftover .sh
```

Mean-age folders round to the nearest integer (`a35` = target 35, or computed Σ wᵢ τᵢ). Filenames are unchanged.

Current hunt family (wave 4–5):

| path | what |
|---|---|
| `k6/t3-56/a32` … `a39` | K6 age neighbors on `[3,56]` |
| `k6/t2-56/a35`, `a38` | same family, τ_min=2 |
| `k6/t3-56/mix_ws40`, `mix_ws50` | leftover-on-slow mixes at age 35 |
| `k4/t3-56/a35` | DEC frozen mix (wave 2) |
| `stacks/k6_t3-56_a35/` | SOAP / Aurora / Adam on K6_a35 |

`anneal_sweeps/` and `logs/wr/` were not moved.
