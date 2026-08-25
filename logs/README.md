Training logs. n=8 record fleets live in `wr/`. Seed-0 hunts live in `kmaxwell/`, grouped by kernel (`k`, τ window, mean age) — not by wave/stage.

```
logs/
  wr/                       n=8 records (see wr/README.md)
    muon_anneal_3160/       Muon winner
    cwd_k6_a35_n8/
    cwd_bimaxwell339_n8/
  kmaxwell/                 seed-0 hunts (see kmaxwell/README.md)
    k{K}/t{min}-{max}/a{age}/
    anneal_sweeps/          seed-0 anneal endpoints (C1_*), not the n=8 winner
    stacks/                 SOAP / Aurora / Adam on a frozen kernel
    _hunts/                 original wave PLAN.txt / launch scripts
```

Muon winner: `wr/muon_anneal_3160/` (k=8, tau `[3,64]`, age 58→26, `train_gpt_kmaxwell_anneal.py`).
