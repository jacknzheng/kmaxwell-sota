# State — CAMPAIGN 2 COMPLETE (2026-08-25). All boxes STOPPED.

Both fleets of boxes are gone (old: 3mpryz3 qed6y1w qvg5eeq qj0gkgw wdp8knw 3mpk7o3;
new: wlmz82q 3y51enw wxgj90q q974r03 qrgxkr3 qz7dvew). Everything is harvested
locally — nothing lives on boxes.

## Final results (Track 3 statsig rule, n=8, margin ≥ 0.004)

| fleet | config | first-pass | reference | verdict |
|---|---|---|---|---|
| A | frozen KM k6 [3,56] a35 on #46 CWD stack | **2680** | #46 = 2690 | beats merged record by 10 |
| B | #339 bi-Maxwell byte-identical | **2640** | claim 2645 | reproduces; pairwise-beats KM (lhs ≈ −0.005) |
| C | anneal 58→26 k8 [3,64] on ablation stack | **3160** | #340 = 3210 | beats #340 by 50 (lhs 0.0073 @3210) |

Details: WRITEUP.md Part 2. Grids: ~/modded-nanogpt/logs/kmaxwell/
{cwd_frozen_n8,bimaxwell339_n8,ablation_anneal_n8}/summary.tsv + pairwise_km_vs_339.tsv.
Ledger: ~/jackzhengretardruns/ledger/ (84 runs). Scoring tool: km/score_wr.py.
Pins: km/pins.json, pins_anneal.json, pins_wr.json (kmwr/kmanneal gate on them).

## If relaunching ever

bootstrap_box.sh is the full recipe (incl. python3-dev fix). Serialize ssh
(cert race). Ship ~/modded-nanogpt tarball + /root/km. See WRITEUP Part 2 infra notes.
