World-record and n=8 fleets. Each folder is one config, `seed{0-7}.*`.

| folder | stack | first statsig `<3.28` | notes |
|---|---|---|---|
| `muon_anneal_3160/` | Muon #36 + k=8 `[3,64]` age 58→26 | **3160** val_loss n=8 | Muon-SOTA vs #36 (3250) and bi-Maxwell (3210). Pairwise at 3250. |
| `cwd_k6_a35_n8/` | SOAP #46 + frozen K6_a35 | **2680** val_ema n=8 | −10 vs published #46 (2690). Not pairwise. Loses to 339. |
| `cwd_bimaxwell339_n8/` | SOAP #46 + bi-Maxwell k=2 | **2640** val_ema n=8 | Stronger CWD stack. Pairwise vs #46 and vs K6_a35. |

`pairwise_cwd_k6_vs_339.tsv` — same-step CWD k=6 vs 339.

Plots (Muon only): `figure.png`, `zoomed_figure.png` — anneal vs [#36](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md) vs [bi-Maxwell 3210](https://github.com/jacknzheng/modded-nanogpt/tree/master/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210). Rebuild: `python3 logs/wr/plot_muon_compare.py`.
