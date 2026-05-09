# Contra-Muon to Soft-Muon results

Start with contra-muon for diversity/exploration, interpolate through standard muon to soft-muon for exploitation.
See https://web.archive.org/web/20260505070142/https://nilin.github.io/contra-muon-and-soft-muon/

![Muon, Contra-Muon, and Soft-Muon singular-value maps on a log x-axis](soft_muon_contra125_maps_logx.png)

The run terminates at 3040 steps. The soft-Muon blend is capped at `0.8`.

Across 26 non-cherry-picked seeds, the step 3040 mean validation loss is
3.27854077. Under the Track 3 README's one-sided z-test with `sigma=0.0016`,
this gives `z=4.6504` and `p=1.66e-6`, satisfying the p<0.001 criterion.

| Seed | 3030 val | 3040 val |
| -: | -: | -: |
| 0 | 3.27779 | 3.27729 |
| 1 | 3.27854 | 3.27801 |
| 2 | 3.28014 | 3.27962 |
| 3 | 3.28093 | 3.28043 |
| 4 | 3.27875 | 3.27822 |
| 5 | 3.27986 | 3.27934 |
| 6 | 3.27762 | 3.27710 |
| 7 | 3.27756 | 3.27705 |
| 8 | 3.28073 | 3.28027 |
| 9 | 3.27869 | 3.27818 |
| 10 | 3.27877 | 3.27827 |
| 11 | 3.27892 | 3.27839 |
| 12 | 3.27685 | 3.27633 |
| 13 | 3.27991 | 3.27942 |
| 14 | 3.27961 | 3.27908 |
| 15 | 3.27882 | 3.27828 |
| 16 | 3.27906 | 3.27855 |
| 17 | 3.27800 | 3.27750 |
| 18 | 3.28011 | 3.27960 |
| 19 | 3.27924 | 3.27871 |
| 20 | 3.27692 | 3.27641 |
| 21 | 3.27888 | 3.27836 |
| 22 | 3.27897 | 3.27843 |
| 23 | 3.27942 | 3.27892 |
| 24 | 3.28132 | 3.28081 |
| 25 | 3.28002 | 3.27949 |
| **Mean** | **3.27905500** | **3.27854077** |

# Credits

This submission incorporates features from the following previous submissions:

- @kumarkrishna PR274 / Skylight-001: NorMuon-lite row/column variance normalization, u/w floor postprocessing, and lr=0.0375 style Muon setup.
- @nilin (me) PR275 / Contra-Muon: introduces Contra-Muon update term.
- @samacqua PR278 / MLP SOAP preconditioning: SOAP preconditioning machinery / MLP SOAP idea. Our script uses that SOAP machinery and extends the selected SOAP set to MLP+V.
- @yash-oai PR287 / power law LR schedule PowerCool: learning rate `c * (t_end - step)^1.2` during cooldown.

The SOAP-like preconditioning from samacqua / PR278 is also applied to the attention value-projection (V) matrices in this submission.
