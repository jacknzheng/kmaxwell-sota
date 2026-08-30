# 8xA100 runbook

Ordered commands for the weekend box. Run everything from the repo root.
Do not skip a gate; each one exists because the next stage burns more GPU time.

## 1. Environment

```bash
pip install torch==2.11 huggingface_hub pyyaml pytest
```

## 2. Data

```bash
python data/cached_fineweb10B.py 20    # ~1.9B train tokens + val shards, enough for 3250 steps
ls data/fineweb10B | head
```

## 3. CPU tests on the box (no GPU minutes yet)

```bash
python -m pytest records/track_3_optimization/tests -q
```

All green before proceeding.

## 4. k=4 GPU smoke of the full record->dump->analyze path (~5 min)

```bash
torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py \
    records/track_3_optimization/configs/smoke_bimaxwell_record_k4.yaml
python records/track_3_optimization/offline_analysis/check_secant_direction_with_hvp.py \
    --dump_dir secant_dumps_smoke --step 40 --tokens 262144 --bootstrap_microbatches 16
python records/track_3_optimization/offline_analysis/replay_probe_history.py \
    secant_dumps_smoke/probe_history.pt --rank_grid 2,4 --gmres_grid 2,4
```

Checks: probe lines in the log, shards at steps 10 and 40, both analysis tools
finish and write JSON.

## 5. Harness-vs-artifact Bi-Maxwell equivalence (gate for everything downstream)

Launch both with the full `train_steps: 3250` (the LR schedule depends on it)
and kill each after the step-500 validation line appears:

```bash
timeout 900 torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/train_gpt_bimaxwell_baseline.py --seed 0 | tee /tmp/artifact.txt
timeout 900 torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/run.py records/track_3_optimization/configs/bimaxwell.yaml seed=0 | tee /tmp/harness.txt
grep -o 'step:[0-9]*/3250 val_loss:[0-9.]*' /tmp/artifact.txt > /tmp/a
grep -o 'step:[0-9]*/3250 val_loss:[0-9.]*' /tmp/harness.txt > /tmp/b
diff /tmp/a /tmp/b && echo EQUIVALENT
```

Expect identical val_loss at boundaries {0, 125, 250, 375, 500} to all printed
decimals (bitwise, if the box is run-to-run deterministic). **Do not proceed on
a mismatch** -- find the divergence first (suspects: kernel compile boundaries,
RNG order, AdamW constants, params_pad structure).

## 6. Disk gate before k=100

```bash
df -h .
```

Needed: 3 pinned dumps x 68 GB + ~14 model checkpoints x 0.6 GB ~= **213 GB** free
(85M Muon params x 2 stacks x k=100 x 4 B = 68 GB per pinned copy; resident
buffer state is 8.5 GB/GPU on 8 ranks). If the disk is much larger, this is the
moment to decide whether MORE pinned steps are acceptable -- edit
`dump_secant_state_at_steps.steps` in `configs/bimaxwell_record_secant.yaml`
at ~68 GB per added step.

## 7. The instrumented Bi-Maxwell run (the first science run, ~half a day)

```bash
torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py \
    records/track_3_optimization/configs/bimaxwell_record_secant.yaml seed=0
```

Produces: the PR #340 trajectory + `secant_dumps/` with pinned shards at steps
{10, 250, 3000}, model checkpoints every 250, and `probe_history.pt` (a full
Gram every 10 steps).

## 8. The three checkpoint analyses (single GPU, minutes each)

```bash
for s in 10 250 3000; do
  torchrun --standalone --nproc_per_node=8 \
      records/track_3_optimization/offline_analysis/check_secant_direction_with_hvp.py \
      --dump_dir secant_dumps --step $s --tokens 4194304 --mbs 8 --basis_rank 12 \
      --holdout_data 'data/fineweb10B/fineweb_train_000020.bin' \
      --bootstrap_resamples 8
done
python records/track_3_optimization/offline_analysis/replay_probe_history.py \
    secant_dumps/probe_history.pt --rank_grid 8,16,32,64 --gmres_grid 4,8,16,32 --left_out
```

(The holdout glob must be shards the run never trained on; with
`cached_fineweb10B.py 20` the late-numbered train shards qualify -- verify
against the tokens actually consumed, 3250 steps x 524288 ~= 1.7B.)

## 9. Decision gate (read before spending the scored run)

Proceed to the scored secant run only if, at steps 250 and 3000:

- `eps_sym` is small and stable after the Bi-Maxwell switch (the quadratic model
  is not self-contradictory);
- the restricted-Hessian spectrum is positive and resolved at the chosen r1
  (`restricted_negative_fraction` ~ 0);
- leave-one-decay-out prediction error and the held-out residual are materially
  below 1 (the operator generalizes across timescales and data);
- `cos(p_ref, p_secant_mb)` beats `cos(p_ref, d_bimaxwell)` (the estimand test);
- bootstrap cosine spread is small pre- AND post-polar, and `coefficient_norm`
  is not exploding as r2 grows (noise cancellation signature).

Fix `truncated_rank_accumulated_parameter_displacements` and
`truncated_rank_gmres` in `configs/secant_gmres.yaml` from the replay grid.

## 10. The scored secant run

```bash
torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py \
    records/track_3_optimization/configs/secant_gmres.yaml seed=0
```

More seeds as fleet time allows (`seed=1`, `seed=2`, ...); the crossing zone is
densely validated from step 2500.

## 11. Baseline repros + archive

```bash
torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/run.py records/track_3_optimization/configs/muon_baseline.yaml
torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/train_gpt_bimaxwell_baseline.py --seed 0
tar czf logs_$(date +%Y%m%d).tgz logs/ secant_dumps/*.json secant_dumps/probe_history.pt
```
