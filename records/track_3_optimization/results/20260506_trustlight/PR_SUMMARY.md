# PR summary

## Title

Track 3: Trustlight reaches 3.28 in 3130 steps

## Summary

This PR adds the `20260506_trustlight` Track 3 optimization record. Trustlight
starts from the SOAP-MLP Contra/NorMuon method in PR #278 by @samacqua and adds
a bounded trust-gated SOAP path for attention output projection matrices
(`attn.proj.weight`). The model architecture, dataset, batch size, and training
loss computation are unchanged from the Track 3 baseline.

The key optimizer change is deliberately narrow: keep SOAP preconditioning on
`mlp.fc.weight` and `mlp.proj.weight`, then add SOAP only to attention output
projections with a denominator floor and agreement-based trust gate. This tries
to capture useful attention-output curvature while avoiding the instability that
can happen when SOAP is applied too broadly to attention matrices.

## Result

At the fixed checkpoint `step=3130`, six non-cherry-picked seeds reach:

```text
seed 0: 3.27881
seed 1: 3.27748
seed 2: 3.27911
seed 3: 3.27966
seed 4: 3.27786
seed 5: 3.27647
mean:   3.27823167
```

Track 3 significance:

```text
(3.28 - 3.27823167) * sqrt(6) = 0.00433151 >= 0.004
z = 3.3319
p = 4.31e-4
```

Step 3125 is the last failing checkpoint in these logs, with mean
`3.27859667`. The same six runs also reach mean `3.27764000` at step 3140,
`3.27717333` at step 3150, and `3.27654500` at step 3175. All runs used
`EARLY_STOP=0` and `TARGET_VAL_LOSS=0`; the checkpoint is selected uniformly
across all seeds.

## Credits

Starting point and main inspiration: PR #278,
"New record: Track_3_optimization: Add SOAP preconditioning to MLPs (3150, -75
steps)", by @samacqua. Trustlight keeps the PR #278 MLP SOAP path and explores a
bounded way to recover some benefit from SOAP on attention output projections.
