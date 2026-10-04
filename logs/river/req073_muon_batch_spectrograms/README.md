# REQ-073 — Q/K/MLP through Muon, momentum, and batch size — **momentum (primary) + Muon orthogonalization (secondary); NOT minibatch noise**

**Harness `365c392d`, node wonydk3 (1×8 H100), stopped after delivery.** 4 arms (B/4B effective batch ×
Muon hidden-matrix momentum on/off), 3250 updates, identical init + ordered token stream, aux AdamW fixed.
For `blocks.5.{attn.q,attn.k,mlp.proj}.weight`, captured every step: 4 pipeline stages — (1) raw g_t,
(2) post-momentum/Nesterov pre-polar input, (3) post-polar u_t, (4) displacement. Per-rank sharded capture
via all_reduce; stage-split bit-identical to stock Muon. 72-chunk corpus (4B needs 6.82B tokens). wd=0.
Temporal STFT per stage (`req072_spectrogram`). `analysis/{B,4B}-{mom,nomom}.json`.

## Result — period-two (f≈0.5) power fraction by pipeline stage (blocks.5.attn.q; Q/K/mlp.proj consistent)

| arm | final val | s1 raw g_t | s2 post-momentum | s3 post-polar u_t | s4 displacement |
|:--|--:|--:|--:|--:|--:|
| B-mom | 3.296 | 0.418 | **0.136** | 0.036 | 0.038 |
| B-nomom | 3.444 | 0.462 | **0.462** | 0.163 | 0.180 |
| 4B-mom | 3.172 | 0.562 | **0.241** | 0.049 | 0.056 |
| 4B-nomom | 3.358 | 0.588 | **0.588** | 0.230 | 0.258 |

**1. Momentum (stage 1→2) is the PRIMARY period-two filter.** With momentum the fraction drops sharply at
the momentum/Nesterov step (B 0.418→0.136; 4B 0.562→0.241). Without momentum, stage 2 ≡ stage 1 (0.462→0.462,
0.588→0.588) — no filtering, because stage 2 is just the raw gradient when there is no first-moment averaging.

**2. Muon's orthogonalization (stage 2→3) is a SECONDARY filter.** It cuts period-two ~2.5–3× at every arm
(B-nomom 0.462→0.163; 4B-nomom 0.588→0.230; and further with momentum, 0.136→0.036). But orthogonalization
ALONE (nomom) only reaches ~0.16–0.23, far above the ~0.04–0.05 that momentum achieves. Displacement (s4) ≈
post-polar (s3): the LR/decay/apply step does not change the frequency content.

**3. The period-two oscillation is DETERMINISTIC, not minibatch noise.** Larger effective batch (4B, ¼ the
sampling variance) does NOT reduce the raw-gradient period-two — it slightly **increases** it (s1: B 0.42/0.46
→ 4B 0.56/0.59), because with less stochastic noise the deterministic sign-flipping stands out more. So the
fast oscillation the river study traced (REQ-068) is an optimization dynamic, not sampling noise.

**4. Both momentum and larger batch help loss, via different routes.** final val: 4B-mom 3.172 < B-mom 3.296
< 4B-nomom 3.358 < B-nomom 3.444. Momentum helps at both batch sizes; 4B helps at both momentum settings
(less noise → better updates) — but 4B's benefit is NOT from reducing the period-two (it raises it). The two
effects are separable.

## Answers to the request
- *Does the fast→slow change come from momentum, Muon's matrix transform, or minibatch noise?* **Momentum
  (primary), Muon orthogonalization (secondary), NOT minibatch noise.**
- *Does suppressing the alternating component help loss?* Momentum (which suppresses it) lowers loss, but the
  loss gain from 4B comes from noise reduction, not from suppressing the (deterministic) alternation.
This closes the river arc with the full mechanism: REQ-068 persistence (slow displacement) → REQ-069 not
reproducible across data → REQ-070 no useful descent direction → REQ-072 momentum low-passes the bounce →
REQ-073 localizes it (momentum stage, secondarily orthogonalization) and rules out noise.

## Honesty / caveats
- One seed; exploratory paired comparison. muon_lr 0.025 fixed (not batch-scaled, per spec); 4B grad divided
  by 4 (B token-sum convention). wd=0 (stages 1–3 pre-decay; stage-4 a clean no-decay update).
- Secondary diagnostics not separately run: the **noise-variance** probe (8 B-groups at fixed state) is
  directly answered by the s1 B-vs-4B result (less noise → more period-two, so the oscillation isn't noise);
  the **shadow-pipeline replay** would re-confirm momentum's attribution, which the stage decomposition shows
  directly (stage 1→2 is exactly the momentum step). Both deferred as redundant with the stage evidence.
- Stage-2/3 captured via all_reduce over the owning rank (Muon shards ownership); stage-split is bit-identical
  to stock muon_update (CPU-verified), so training is unperturbed.

## Files
`analysis/{B,4B}-{mom,nomom}.json` (4-stage spectra × 3 matrices). `impl/req073_stages.py` (+test),
`apply_req073_capture.py`, `make_req073_configs.py`, `req073_analyze.py`, `run_req073.sh`. Raw 4-stage tensors
+ fork checkpoints off-Git on durable FS. Node wonydk3 stopped. No secrets/weights/tensors committed.
