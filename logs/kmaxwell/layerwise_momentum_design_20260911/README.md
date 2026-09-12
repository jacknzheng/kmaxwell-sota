# Layer-wise momentum: experiment design review, September 11, 2026

**Recommendation: validate layer-wise, shape-weighted spectral sharpness; test whether it predicts
the response to changing memory; then test that allocation against global and shuffled controls.**
Do not launch another broad LR sweep or infer a momentum rule from the old curvature bowl.

This is a design review and CPU verification, **not a new GPU result**. The audit used branch
`jerry-agent` at `98c92ea718ba15a7d11c1f6079c9cdca2e583c21` (September 7), still the remote head
when inspected on September 11. New requests are [REQ-057–060](../../../requests.md#req-057-validate-layer-wise-spectral-sharpness-and-cross-layer-coupling).
The supplied local `momentum_sharpness_investigation/INVESTIGATION.md` and its September 7 review
were read; the latter's arithmetic was re-run against the newly fetched branch files.

## What sharpness needs to mean here

Sharpness describes how quickly a small parameter change bends the loss upward. But "small"
needs a ruler. Euclidean sharpness uses the total squared change in individual entries. Muon
works with matrices and transforms their singular values, so that ruler may rank dangerous
directions incorrectly. Changing the ruler does not change the Hessian; it changes which
perturbations have the same allowed size.

[Islamov, Crawshaw, Cohen and Gower, v3](https://arxiv.org/html/2603.05002v3) define sharpness by
maximizing Hessian curvature under the optimizer's norm. For normalized spectral descent,
Section 4 also divides by the dual gradient norm. The joint block-spectral choice uses the sum
of nuclear gradient norms. The paper's exact loss identity concerns directional smoothness;
replacing it with local maximum curvature is a diagnostic approximation. Its conclusion leaves
stochastic and momentum extensions open.

Our extension to the actual shape-scaled Muon subspace is specified in REQ-057. It distinguishes:

- **Isolated matrix sharpness:** move one matrix and freeze the others.
- **Joint sharpness:** move all Muon matrices together, retaining interactions.
- **Actual-direction curvature:** measure curvature along the direction chosen by the live
  memory and polar transformation.
- **Future stability/performance:** what happens over subsequent updates, measured separately.

These are different questions. Summing isolated maxima does not recover joint sharpness.
Similarly, a layer's contribution to joint curvature can include its interactions with all other
layers. It cannot be relabeled an isolated block maximum.

The supplied investigation correctly limits the ordinary heavy-ball formula to that optimizer.
Under Muon, the old `lambda/||g||_F²` ratio can remain a comparison feature but is not a validated
momentum-setting rule. A sharpness-to-memory mapping must be learned and tested; even the sign
of the useful adjustment should not be assumed. A reliable predictive proxy could suffice for
control; computing an exact global maximum at every training step is not a prerequisite.

## Existing experiments and the remaining gap

| Existing evidence | What is reusable | What it does not settle |
|---|---|---|
| [REQ-019 FW calibration](../req019_fw_calibration/README.md) | Joint spectral-ball HVP/FW code with cross-matrix coupling, two checkpoints, five starts, 50 iterations. Reported cost about 23 minutes per checkpoint on 8 H100s. | One global unweighted measurement; no isolated spectral layer map or memory-response prediction. Saved gradient norms are Frobenius. A restart average is not an estimate of the maximum in the same sense as the best feasible witness. |
| [REQ-023](../req023_per_matrix_lr/README.md), [REQ-045](../req045_crossed_global_permatrix_lr/README.md) | Changing a matrix's LR changes its Euclidean curvature; useful assignment/state machinery. | LR response is not momentum response. A nonsignificant aggregate neighbor coefficient does not establish independent layers. |
| [REQ-036](../req036_equalized_curvature_lr/README.md) | Equalized-curvature LR allocation was tested. The reported polar-target arm reduced spread from 0.246 to 0.128 but worsened validation loss by about 0.024. | Equalizing a measured quantity is not automatically a useful training objective. This was an LR intervention, with one seed per arm. |
| [REQ-044](../req043_paired_kernel_batch_ablation/README.md) | Global memory-kernel benefits depend on batch regime; K-Maxwell retains an advantage at large batches. | No layer-specific assignment or direct curvature-mediated cause. Fixed steps across batches also change token budgets. |
| [REQ-046](../req046_permatrix_clip_instrument/README.md) | Pre-polar gradient scaling had almost no equilibrium effect after normalization. | Do not repeat this ineffective intervention as a way to control curvature. |
| [REQ-048](../req048_spectral_participation/README.md) | Euclidean spectral-concentration measurements and repeated-seed profiles. | Participation ratio is not the paper's matrix-spectral sharpness. Trace cancellation matters for indefinite Hessians. |
| [REQ-050](../req050_curvature_at_init/README.md) | Curvature is measurable early after the zero output projection starts learning. | Zero curvature at initialization masks architectural effects; it does not rule them out. |
| [REQ-051](../req051_lr_curvature_decomp/README.md) and [REQ-052](../req052_uniform_lr_controls/README.md) | Multi-seed LR interventions and raw records. | All 48 curvature/activation pairs use 32768 versus 8192 tokens. All four same-label base hashes and base losses differ across 051/052. They do not meet the requested same-probe and exact-state comparisons. |
| [REQ-053](../req053_mlpproj_mechanism/README.md) | Some MLP-output LR sensitivity persists across tested expansions and GELU. | All expansion ratios exceed one; this does not eliminate the fan-in/fan-out explanation or inform a memory rule directly. |
| [REQ-054](../req054_agematched_ema/README.md) | K-Maxwell beats the implemented scheduled single EMA by mean 0.0097175 validation loss over four seeds. | The delivered experiment is 1× only with two arms, not the full requested batch/no-momentum grid. Realized age was not matched. Raw curves/full provenance are missing from this directory. |
| [REQ-055](../req055_post_muon_update_geometry/README.md) | A first actual-update probe and joint finite-difference measurements. | One seed, four post-fork states, no shared-state counterfactual, no epsilon sweep; per-matrix values contain cross terms. The "geometrically indistinguishable" verdict is stronger than the data. |
| [REQ-056](../req056_adam_kmaxwell/README.md) | Raw validation curves permit an independent arithmetic check. | K-Maxwell versus ordinary Adam remains inconclusive: mean +0.00645, paired t interval about [-0.02225,+0.03514], n=3. It does not justify expanding the Adam campaign before the layer-wise question. |

The review searched the complete committed `logs/kmaxwell` file inventory, relevant README/source
files and `FINDINGS.md` for layer-wise/per-matrix momentum, generalized sharpness, coupling and
actual-update probes. **No committed experiment was found that combines isolated, normalized
spectral sharpness with selective memory interventions and held-out layer-allocation validation.**
This is an archive finding; it is not proof that no uncommitted remote job exists. The runner must
check live jobs before pickup and preserve the two-node fleet limit.

The 95% paired t interval for REQ-054's table-level K-Maxwell-minus-scheduled-EMA difference is
[-0.0106879,-0.0087471]. It verifies endpoint-table arithmetic, not missing training provenance or
the proposed explanation. The late REQ-055 curvature estimates differ by 45.1% and 46.4%; without
a numerical sensitivity check, that disagreement cannot be assigned specifically to higher-order
loss terms. Averages with sign changes are not equivalence tests.

## Source-code checks that affect the requests

The training source was read from the available local Git object at
`365c392d695f95dc9a4fb89095e85a6a7b5d551e`, the SHA cited by REQ-054/055/056. Its harness files
are absent from the current artifact-branch tip. A runner must recover this version or verify
parity with a pinned successor; the artifact branch alone is not a runnable training checkout.
For example, in a training checkout of the same repository:

```sh
git fetch origin 365c392d695f95dc9a4fb89095e85a6a7b5d551e
git worktree add --detach ../kmaxwell-layerwise-training 365c392d695f95dc9a4fb89095e85a6a7b5d551e
```

Then apply the committed relevant experiment patches, record their hashes, and add the new
instrumentation with parity checks before training. The commands above identify the source;
they do not substitute for dependency setup or the new implementation.

- `optimizers/muon.py` uses `sqrt(max(1, rows/cols))`, a bfloat16 Newton-Schulz implementation
  with **12 iterations despite its `...5` function name**, and an in-place `grad.lerp_` mutation.
  Record raw gradients before mutation and preserve the implemented transformation.
- `AnnealedWeightsMuon` performs a baseline update at the switch and then clones the buffer.
  The first mixed update is later; log counter semantics instead of inferring them from filenames.
- `harness/model_gpt.py` returns a summed cross-entropy loss. The old curvature probe rescales
  by `524288/tokens_seen`. New mean-token diagnostics must convert both gradient and Hessian,
  using actual valid tokens, not nominal batch labels.
- REQ-019's `frank_wolfe` returns an interior iterate without the radial normalization discussed
  in the paper. Recompute feasible sphere objectives, retain raw traces, and report the best
  restart as a lower-bound witness. Good convergence is not a proof of global optimality.
- REQ-055 perturbs every matrix at once; its per-matrix dot products therefore contain joint
  Hessian contributions. The replacement must explicitly separate H_ii from H_ij terms.

## Decision sequence

1. **REQ-057:** determine whether layer-wise spectral measurements are accurate, repeatable and
   distinguishable from their Euclidean comparators. Include cross-layer interactions.
2. **REQ-058:** shorten/lengthen one matrix's memory from shared states at fixed LR; predict its
   future loss response on held-out seeds. Include the missing exact-age control.
3. **REQ-059:** test the frozen allocation on four fresh seeds against global, Euclidean,
   type/depth, shuffled and reversed controls. This is the direct project success test.
4. **REQ-060:** separately test whether loss-cubic feedback explains a resolved effect, retaining
   Muon's normalization contribution. It need not block an empirically useful policy.

Full designs, budgets, stopping rules and required artifacts are in `requests.md`. Gates prevent
spending the full budget if the measurement or predictive premise fails. Do not change a
criterion after seeing the test seeds. A null result can mean the statistic is too noisy, too
incomplete, or not useful for choosing memory; it does not establish that every conceivable
layer-wise optimizer is impossible.

## Verification performed here

Run `python logs/kmaxwell/layerwise_momentum_design_20260911/verify.py` with PyTorch installed.
[verification.json](verification.json) records the results. No tensor checkpoints are loaded.

- All eight existing REQ-019 CPU tests pass, invoked directly because pytest was unavailable.
- Two tiny Hessians have identical Euclidean eigenvalues [0.1,0.1,0.1,1.1] but spectral sharpness
  1.2 versus 2.2, with matching attained and analytical bounds.
- A two-layer example has isolated maxima summing to 2 but joint sharpness 3.8.
- Loss scaling leaves exact-polar directions and S/G unchanged while changing `lambda/||g||²`.
- A memory-selected polar direction points uphill in a toy example despite positive nuclear
  gradient norm, illustrating why the no-momentum denominator is insufficient by itself.
- An explicit scalar replay of the published clone protocol matches realized mean age including
  the outer blend to about 1.4e-14, with feasible decays and mass discrepancy below 2e-15.
  The stationary-age approximation differs by up to about 2.60 steps in this replay. This is a
  reproducible control construction, not evidence that the age mismatch caused the observed loss gap.
- Committed REQ-051/052 provenance and REQ-054/056 endpoint arithmetic were rechecked.

These checks verify the design's foundations. GPU precision, real-model measurement accuracy,
predictive power and training gains remain to be tested by the newly submitted requests.
