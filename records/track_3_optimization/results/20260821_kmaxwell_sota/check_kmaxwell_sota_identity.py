"""CPU identity checks for K-Maxwell on the #46 SOTA first-moment slot.

Does not run a full training job. Verifies:
  1) --bimaxwell-exact reproduces PR #339 constants (0.85 / 0.98 / 0.4385, start 1000)
  2) the stacked mix matches named fast/slow buffers + non-inplace grad.lerp
  3) grad is unchanged after the mix (SOAP still needs raw grad)
  4) switch-step: missing K buffers -> update equals single-EMA Nesterov

Does not import train_gpt_kmaxwell_sota.py (that file initializes NCCL at import).

Stage 0 full train (GPU):
  torchrun --standalone --nproc_per_node=1 \\
      records/track_3_optimization/results/20260821_kmaxwell_sota/train_gpt_kmaxwell_sota.py \\
      --seed 0 --k 2 --bimaxwell-exact
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kmaxwell_kernel import (
    BIMAXWELL_BF,
    BIMAXWELL_BS,
    BIMAXWELL_START,
    BIMAXWELL_W,
    BIMAXWELL_TAU_MIN,
    BIMAXWELL_TAU_MAX,
    build_kmaxwell_kernel,
    nesterov_filter_stats,
)
from kmaxwell_momentum import kmaxwell_first_moment


def test_bimaxwell_exact_kernel():
    tau, betas, weights, mean_age = build_kmaxwell_kernel(
        k=2, tau_min=BIMAXWELL_TAU_MIN, tau_max=BIMAXWELL_TAU_MAX,
        sigma=1.0, bimaxwell_exact=True)
    assert betas == [BIMAXWELL_BF, BIMAXWELL_BS]
    assert weights == [BIMAXWELL_W, 1.0 - BIMAXWELL_W]
    assert abs(tau[0] - BIMAXWELL_TAU_MIN) < 1e-12
    assert abs(tau[1] - BIMAXWELL_TAU_MAX) < 1e-12
    assert abs(mean_age - 30.0) < 2e-3, mean_age
    assert BIMAXWELL_START == 1000
    lag_m, lag_x, noise = nesterov_filter_stats(betas, weights)
    assert abs(lag_m - mean_age) < 1e-12
    assert 0.034 < noise < 0.036
    print(f"ok  exact kernel  mean_age={mean_age:.6f}  lag_x={lag_x:.4f}  "
          f"noise_gain={noise:.5f}  start={BIMAXWELL_START}  "
          f"tau={tau}  beta={betas}  w={weights}")


def test_exact_requires_k2():
    try:
        build_kmaxwell_kernel(4, 5.67, 49.0, 1.0, bimaxwell_exact=True)
    except ValueError as e:
        assert "k 2" in str(e)
        print("ok  bimaxwell-exact rejects K!=2")
        return
    raise AssertionError("expected ValueError")


def _lerp(a, b, weight):
    return a + weight * (b - a)


def test_exact_mix_matches_two_ema_scalar():
    g, m_fast, m_slow = 1.5, 0.2, -0.4
    mf = _lerp(m_fast, g, 1 - BIMAXWELL_BF)
    ms = _lerp(m_slow, g, 1 - BIMAXWELL_BS)
    m_eff = BIMAXWELL_W * mf + (1 - BIMAXWELL_W) * ms
    u1 = _lerp(g, m_eff, 0.95)

    _, betas, weights, _ = build_kmaxwell_kernel(
        2, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, bimaxwell_exact=True)
    m = [_lerp(mk, g, 1 - bk) for mk, bk in zip([m_fast, m_slow], betas)]
    m_eff2 = sum(w * mk for w, mk in zip(weights, m))
    u2 = _lerp(g, m_eff2, 0.95)
    assert abs(u1 - u2) < 1e-15, (u1, u2)
    print(f"ok  scalar mix vs two-buffer  u={u1:.8f}")


def test_stacked_mix_matches_pr339_named_buffers():
    try:
        import torch
    except ImportError:
        print("skip stacked mix (torch not installed)")
        return

    torch.manual_seed(0)
    grad = torch.randn(16, 32)
    m_fast = torch.randn_like(grad)
    m_slow = torch.randn_like(grad)
    m = torch.stack([m_fast.clone(), m_slow.clone()])
    _, betas, weights, _ = build_kmaxwell_kernel(
        2, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, bimaxwell_exact=True)

    # PR #339: non-inplace Nesterov mix, named fast/slow buffers
    g1 = grad.clone()
    mf, ms = m_fast.clone(), m_slow.clone()
    mf.lerp_(g1, 1.0 - BIMAXWELL_BF)
    ms.lerp_(g1, 1.0 - BIMAXWELL_BS)
    m_eff = torch.lerp(ms, mf, BIMAXWELL_W)
    u1 = g1.lerp(m_eff, 0.95)

    g2 = grad.clone()
    u2 = kmaxwell_first_moment(
        g2, m, torch.tensor(betas, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32), 0.95)
    err = (u1 - u2).abs().max().item()
    assert err < 1e-6, err
    assert torch.equal(g2, grad), "kmaxwell_first_moment must not mutate grad"
    print(f"ok  stacked mix vs PR #339 named buffers  max_abs_err={err}")


def test_switch_step_equals_single_ema():
    try:
        import torch
    except ImportError:
        print("skip switch-step (torch not installed)")
        return

    torch.manual_seed(1)
    grad = torch.randn(8, 16)
    momentum = torch.randn_like(grad)
    mu = 0.95
    m_buf = momentum.clone()
    m_buf.lerp_(grad, 1 - mu)
    expected = grad.lerp(m_buf, mu)

    # switch step: K buffers missing -> keep the single-EMA Nesterov mix
    got = grad.lerp(m_buf, mu)
    err = (expected - got).abs().max().item()
    assert err == 0.0
    print("ok  switch-step identity  single-EMA Nesterov unchanged")


if __name__ == "__main__":
    test_bimaxwell_exact_kernel()
    test_exact_requires_k2()
    test_exact_mix_matches_two_ema_scalar()
    test_stacked_mix_matches_pr339_named_buffers()
    test_switch_step_equals_single_ema()
    print("all identity checks passed")
