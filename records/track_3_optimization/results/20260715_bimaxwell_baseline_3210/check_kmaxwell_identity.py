"""CPU identity checks for K-Maxwell vs the bi-Maxwell submission kernel.

Does not run a full training job. Verifies:
  1) --bimaxwell-exact reproduces betas/weights/mean-age of the 3210 artifact
  2) Gaussian weights are deterministic, sum to 1, and freeze given the knobs
  3) the K=2 mix matches named fast/slow buffers (scalar always; torch if present)

Stage 0 full train (GPU, 3250 steps):
  torchrun --standalone --nproc_per_node=1 \\
      records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py \\
      --seed 0 --k 2 --bimaxwell-exact
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kmaxwell_kernel import (
    BIMAXWELL_BF,
    BIMAXWELL_BS,
    BIMAXWELL_W,
    BIMAXWELL_TAU_MIN,
    BIMAXWELL_TAU_MAX,
    build_kmaxwell_kernel,
    nesterov_filter_stats,
)


def test_bimaxwell_exact_kernel():
    tau, betas, weights, mean_age = build_kmaxwell_kernel(
        k=2, tau_min=BIMAXWELL_TAU_MIN, tau_max=BIMAXWELL_TAU_MAX,
        sigma=1.0, bimaxwell_exact=True)
    assert betas == [BIMAXWELL_BF, BIMAXWELL_BS]
    assert weights == [BIMAXWELL_W, 1.0 - BIMAXWELL_W]
    assert abs(tau[0] - BIMAXWELL_TAU_MIN) < 1e-12
    assert abs(tau[1] - BIMAXWELL_TAU_MAX) < 1e-12
    assert abs(mean_age - 30.0) < 2e-3, mean_age
    lag_m, lag_x, noise = nesterov_filter_stats(betas, weights)
    assert abs(lag_m - mean_age) < 1e-12
    assert 0.034 < noise < 0.036
    print(f"ok  exact kernel  mean_age={mean_age:.6f}  lag_x={lag_x:.4f}  "
          f"noise_gain={noise:.5f}  tau={tau}  beta={betas}  w={weights}")


def test_gaussian_weights_deterministic():
    a = build_kmaxwell_kernel(4, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, False)
    b = build_kmaxwell_kernel(4, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, False)
    assert a == b
    tau, betas, weights, mean_age = a
    assert abs(sum(weights) - 1.0) < 1e-12
    assert min(weights) > 0
    assert min(betas) > 0 and max(betas) < 1
    w_fmt = [f"{x:.4f}" for x in weights]
    print(f"ok  gaussian K=4  mean_age={mean_age:.4f}  w={w_fmt}")


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
    m = [m_fast, m_slow]
    m = [_lerp(mk, g, 1 - bk) for mk, bk in zip(m, betas)]
    m_eff2 = sum(w * mk for w, mk in zip(weights, m))
    u2 = _lerp(g, m_eff2, 0.95)
    assert abs(u1 - u2) < 1e-15, (u1, u2)
    print(f"ok  scalar mix vs two-buffer  u={u1:.8f}")


def test_stacked_mix_matches_two_buffers():
    try:
        import torch
    except ImportError:
        print("skip stacked mix (torch not installed)")
        return

    sys.path.insert(0, str(HERE))
    # Import mix helper without running training (train_gpt_kmaxwell is __main__-guarded)
    from train_gpt_kmaxwell import kmaxwell_momentum

    torch.manual_seed(0)
    grad = torch.randn(16, 32)
    m_fast = torch.randn_like(grad)
    m_slow = torch.randn_like(grad)
    m = torch.stack([m_fast.clone(), m_slow.clone()])
    _, betas, weights, _ = build_kmaxwell_kernel(
        2, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, bimaxwell_exact=True)

    g1 = grad.clone()
    mf, ms = m_fast.clone(), m_slow.clone()
    mf.lerp_(g1, 1 - BIMAXWELL_BF)
    ms.lerp_(g1, 1 - BIMAXWELL_BS)
    m_eff = BIMAXWELL_W * mf + (1 - BIMAXWELL_W) * ms
    u1 = g1.lerp_(m_eff, 0.95)

    g2 = grad.clone()
    u2 = kmaxwell_momentum(
        g2, m, torch.tensor(betas, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32))
    err = (u1 - u2).abs().max().item()
    assert err < 1e-6, err
    print(f"ok  stacked mix vs two-buffer  max_abs_err={err}")


def print_stage1_derived_mean_ages():
    print("stage-1 derived mean ages (tau_min=5.666..., tau_max=49, sigma=1):")
    for k in (2, 3, 4, 6, 8, 12, 16):
        tau, _, weights, mean_age = build_kmaxwell_kernel(
            k, BIMAXWELL_TAU_MIN, BIMAXWELL_TAU_MAX, 1.0, False)
        print(f"  K={k:2d}  mean_age={mean_age:7.3f}  w_sum={sum(weights):.6f}  "
              f"tau[0]={tau[0]:.3f}  tau[-1]={tau[-1]:.3f}")


if __name__ == "__main__":
    test_bimaxwell_exact_kernel()
    test_gaussian_weights_deterministic()
    test_exact_requires_k2()
    test_exact_mix_matches_two_ema_scalar()
    test_stacked_mix_matches_two_buffers()
    print_stage1_derived_mean_ages()
    print("all identity checks passed")
