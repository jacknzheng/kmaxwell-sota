# September 15 audit: complete the controls before another layer-wise policy

Reviewed `jerry-agent` at `dabff19` (latest remote at submission), including all returned
REQ-057–060 implementations and scalar artifacts. No REQ-063 or later result directory exists
under `logs/kmaxwell`; REQ-061/062 concern other projects. No GPU work was run for this audit.

The project still needs to show that a sharpness measurement helps choose **different momentum
memories for different matrices**, and that the resulting optimizer beats a strong global setting.
REQ-059's negative outcome survives arithmetic checks. REQ-063–065 repair prerequisites and test
an allocation around the best global mixture without forcing some matrices into longer memory.

## What the returned evidence supports

- **REQ-057:** 18/18 pilot estimates change less than 5% at K20→K50; median cross-subset rank
  correlation of S_i/G_i is 0.983. The full map has 72 matrices at nine seed/checkpoint states.
  Cross terms contribute 0.96813 on average (sample SD 0.00106) to curvature at the joint
  **gradient-polar diagnostic direction**. This is not the realized momentum update. The
  diagonal curvature at that direction is not the sum of isolated maxima S_i. Separately,
  sum(S_i)/S_joint is about 0.0325–0.0354 for the reported optimization witnesses. Neither
  ratio is a general decomposition of model sharpness. Full-map finite differences fail the
  5% check at several states; K20 with two restarts/one subset was not fully revalidated from
  the five-restart/three-subset pilot. Raw pilot S_i falls to a median 0.536 of its value on
  the nested larger token set, while the S_i/G_i ratio changes to a median 0.975. Stable
  rankings do not certify stable absolute values or an optimizer stability boundary.
- **REQ-058:** the delivered raw-S_i correlation is reproducible: +0.604 on development rows,
  +0.591 on pooled seeds 1/2; separately +0.474 and +0.736 on those two seeds. It does not
  implement the requested type/depth, Euclidean, S_i/G_i and actual-update model comparisons,
  per-seed 10% RMSE gate, or committed predictions before test outcomes. All 72 selective
  comparisons favor short over long memory; that alone does not identify a useful mixed
  allocation. The source analysis silently substitutes the last available evaluation when
  fork+64 is absent: all 41 seed-0/fork-1500 files end at 1560, not 1564. Those are 60-update
  observations as labeled, unless full logs establish a different convention.
- **REQ-059:** all 36 files contain step 2750. Mean guided minus global a=0.5 is +0.009135;
  versus Euclidean -0.000120; versus type/depth -0.000175; versus shuffled -0.0005725.
  The policy used raw S_i rankings, not a fitted S_i/G_i response model. It forced a
  24/24/24 short/ordinary/long allocation. Its failure does not reject an unconstrained
  policy, but there is no demonstrated practical gain from spectral-specific assignment.
- **REQ-060:** finite-scale gradient/polar nonlinearities are present in a simplified
  gradient-centered diagnostic. There is no actual-buffer mechanism identification or
  causal removal replay. The observed scaling differs from the pure-cubic expectation.
  Keep the memory-to-sharpness causal explanation **unresolved**.

## Reproducibility issues to resolve

1. In `365c392d`'s `harness/hooks.py:594`, `load_training_state` calls each optimizer's
   `load_state_dict`. Muon inherits PyTorch's parameter-group restore. REQ-058/059 initialize
   the `nomom` arm with `mu=0`, then restore the base's `mu=0.95`; no committed hook reapplies
   zero. Loading that state into the actual historical Muon class reproduces **0→0.95** on
   CPU. Runtime patches could resolve this, but are not delivered. Do not interpret the
   existing `nomom` labels as verified zero momentum.
2. REQ-059's analysis uses normal-tail probabilities with n=4, does not enforce monotonic
   Holm adjusted p-values, and its decision ignores its Holm values. Correct paired t
   probabilities (df=3) plus Holm leave the negative primary outcome unchanged: adjusted
   p≈0.1883 against Euclidean, ≈0.01055 against shuffle. Report ordinary paired 95% CIs as
   ordinary CIs, not multiplicity-adjusted intervals.
3. REQ-058/059 config generators use the same validation-file glob with different token
   counts; the pinned loader takes the first batch from that stream. There is no explicit
   disjoint final-set offset in these files. Recover runtime data-range manifests before
   describing selection and final validation as disjoint. An offset must be implemented
   and checked in the actual loader, not just a config comment.
4. Delivered configs are generators, and REQ-059's actual seed feature/allocation JSONs,
   full stdout and state manifests are missing. Recover them before attributing small
   differences to named per-matrix assignments or joining regenerated bases by seed alone.

## CPU reproduction

`verify.py` checks endpoint coverage, recomputes the REQ-057/059 arithmetic and correct paired
statistics, and reproduces parameter-group overwrite using the Muon class extracted from the
pinned source (no training or GPU required):

```bash
python verify.py --harness-repo /path/to/checkout-with-365c392d
```

Run from this directory, or pass `--repo` for the artifact checkout. PyTorch is required only
for the reload reproduction. `verification.json` contains the actual local output; known
missing endpoints are reported as findings, not silently scored as completed runs.

REQ-063 first recovers provenance and repairs controls; REQ-064 makes a newly registered
prediction test on unused seeds, with actual-update geometry and normalized sharpness;
REQ-065 runs only after that gate passes. Older outcomes stay in place with these scope
corrections. The cubic mechanism request remains secondary and is not being expanded.
