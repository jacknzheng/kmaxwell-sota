"""CPU validation of the REQ-058 exact-age control + stream-decay perturbation.
Mirrors the design-review reference (verify.py cloned_muon_age_toy): the solved single EMA must reproduce
the mixture's realized age to ~1e-10 with feasible beta in [0,1), and the stationary schedule beta=A/(1+A)
must NOT match (that is REQ-054's insufficient control). Run: python test_age_match.py
"""
import req058_age_match as R


def test_perturb_decays():
    d = R.DECAYS
    assert R.perturb_decays(d, 1.0) == d
    a2 = R.perturb_decays(d, 2.0)   # longer memory: beta -> beta^(1/2), closer to 1
    a05 = R.perturb_decays(d, 0.5)  # shorter memory: beta -> beta^2, closer to 0
    for b, b2, b05 in zip(d, a2, a05):
        assert b2 > b > b05, (b, b2, b05)
        assert abs(b2 - b ** 0.5) < 1e-12 and abs(b05 - b ** 2) < 1e-12
    print("PASS test_perturb_decays")


def test_exact_age_match():
    # a=1 mixture (unperturbed): solve matched single EMA, must reproduce realized age to ~1e-10
    res = R.solve_matched_ema(R.DECAYS, R.normalized_weights, steps=749)
    assert res["max_age_error"] < 1e-10, res["max_age_error"]
    assert res["max_mass_difference"] < 1e-11, res["max_mass_difference"]
    lo, hi = res["beta_range"]
    assert 0.0 <= lo and hi < 1.0, res["beta_range"]
    print(f"PASS test_exact_age_match (max_age_err={res['max_age_error']:.2e}, "
          f"max_mass_diff={res['max_mass_difference']:.2e}, beta in [{lo:.4f},{hi:.4f}])")


def test_stationary_schedule_insufficient():
    # REQ-054's schedule-only control beta=A/(1+A) does NOT reproduce the finite-history realized age.
    M_s, P_s, mass, mom = R.clone_init(R.DECAYS)
    smass = smom = mass  # single stationary EMA seeded from the same clone (scalar)
    smom = mom
    max_stat_err = 0.0
    for t in range(R.SWITCH + 1, R.SWITCH + 1 + 749):
        w = R.normalized_weights(t)
        for k, b in enumerate(R.DECAYS):
            P_s[k], M_s[k] = b * (P_s[k] + M_s[k]), b * M_s[k] + (1 - b)
        _, target_age = R.mixture_age(M_s, P_s, w)
        A = sum(wi * bi / (1 - bi) for wi, bi in zip(w, R.DECAYS))  # stationary age of the mixture
        sbeta = A / (1 + A)
        smom, smass = sbeta * (smom + smass), sbeta * smass + (1 - sbeta)
        sage = R.NU * smom / ((1 - R.NU) + R.NU * smass)
        max_stat_err = max(max_stat_err, abs(sage - target_age))
    assert max_stat_err > 1.0, f"stationary schedule unexpectedly matched (err {max_stat_err})"
    print(f"PASS test_stationary_schedule_insufficient (stationary control off by up to {max_stat_err:.2f} steps)")


def test_perturbed_matches_too():
    # the exact-age solver must also work for a perturbed (a=2, longer-memory) mixture
    for a in (0.5, 2.0):
        pert = R.perturb_decays(R.DECAYS, a)
        res = R.solve_matched_ema(pert, R.normalized_weights, steps=749)
        assert res["max_age_error"] < 1e-9, (a, res["max_age_error"])
        assert 0.0 <= res["beta_range"][0] and res["beta_range"][1] < 1.0
    print("PASS test_perturbed_matches_too")


if __name__ == "__main__":
    for t in [test_perturb_decays, test_exact_age_match, test_stationary_schedule_insufficient, test_perturbed_matches_too]:
        t()
    print("\nALL AGE-MATCH CPU TESTS PASSED")
