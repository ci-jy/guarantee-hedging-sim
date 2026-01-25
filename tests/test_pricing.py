"""Statistical checks of the Monte Carlo guarantee price."""
import numpy as np
import pytest

from ghedge import GBM, GMMBContract, Heston
from ghedge import blackscholes as bs
from ghedge import heston
from ghedge.montecarlo import METHODS, control_variate, convergence_slope, convergence_study, price_gmmb

GUARANTEES = [80.0, 100.0, 120.0]
MATURITIES = [1.0, 5.0, 10.0, 20.0]


@pytest.mark.parametrize("method", ["plain", "antithetic_control"])
@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("guarantee", GUARANTEES)
def test_mc_price_within_three_standard_errors(guarantee, maturity, method):
    c = GMMBContract(guarantee=guarantee, maturity=maturity)
    exact = float(bs.gmmb_value(c, 0.2))
    seed = int(guarantee * 100 + maturity)
    res = price_gmmb(c, GBM(0.2), 200_000, method, rng=seed)
    assert abs(res.estimate - exact) < 3 * res.std_error, (res.estimate, exact, res.std_error)
    lo, hi = res.ci(0.997)
    assert lo < exact < hi


def test_confidence_interval_coverage():
    """About 95% of independent 95% intervals should contain the closed form."""
    c = GMMBContract(maturity=5.0)
    exact = float(bs.gmmb_value(c, 0.2))
    seeds = np.random.SeedSequence(11).spawn(400)
    hits = 0
    for s in seeds:
        lo, hi = price_gmmb(c, GBM(0.2), 2_000, "plain", np.random.default_rng(s)).ci(0.95)
        hits += lo < exact < hi
    coverage = hits / len(seeds)
    # Binomial(400, 0.95) has sd ~ 0.011; allow about 3 sd.
    assert 0.915 < coverage < 0.985


def test_control_variate_reduces_variance(contract):
    """Variance reduction factors per simulated path, measured on 400k paths."""
    v = {m: price_gmmb(contract, GBM(0.2), 400_000, m, rng=7).variance_per_path for m in METHODS}
    cv_factor = v["plain"] / v["control"]
    anti_factor = v["plain"] / v["antithetic"]
    both_factor = v["plain"] / v["antithetic_control"]
    # Measured factors for the 10-year ATM guarantee are about 1.8, 3.7 and 10.
    assert cv_factor > 1.5
    assert anti_factor > 2.5
    assert both_factor > 6.0
    assert both_factor > max(cv_factor, anti_factor)


def test_heston_control_variate_reduces_variance(heston_params):
    c = GMMBContract(maturity=5.0)
    plain = price_gmmb(c, heston_params, 40_000, "plain", rng=8, steps_per_year=25)
    ctrl = price_gmmb(c, heston_params, 40_000, "control", rng=8, steps_per_year=25)
    assert plain.variance_per_path / ctrl.variance_per_path > 2.0
    assert ctrl.beta.shape == (2,)


def test_control_variate_is_unbiased_with_exact_control():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000)
    y = 3.0 + 2.0 * x
    adj, beta = control_variate(y, x, [0.0])
    assert beta[0] == pytest.approx(2.0)
    np.testing.assert_allclose(adj, 3.0)


def test_convergence_rate_is_inverse_square_root():
    c = GMMBContract(maturity=10.0)
    df = convergence_study(c, GBM(0.2), [500, 2_000, 8_000, 32_000], n_reps=60, seed=3)
    slope = convergence_slope(df)
    assert -0.6 < slope < -0.4, df
    # Error at 64x the paths should be roughly 8x smaller.
    ratio = df["rmse"].iloc[0] / df["rmse"].iloc[-1]
    assert 5.0 < ratio < 12.0


def test_heston_mc_matches_semi_analytic(heston_params):
    c = GMMBContract(maturity=1.0)
    exact = float(heston.gmmb_value_and_greeks(c, heston_params)[0][0])
    res = price_gmmb(c, heston_params, 200_000, "antithetic_control", rng=9, steps_per_year=250)
    assert abs(res.estimate - exact) < 3 * res.std_error, (res.estimate, exact, res.std_error)


def test_heston_with_tiny_vol_of_vol_matches_black_scholes():
    h = Heston(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=-0.5)
    for g in GUARANTEES:
        for t in [0.5, 5.0, 20.0]:
            c = GMMBContract(guarantee=g, maturity=t)
            assert float(heston.gmmb_value_and_greeks(c, h)[0][0]) == pytest.approx(
                float(bs.gmmb_value(c, 0.2)), abs=2e-3)


def test_heston_greeks_match_finite_differences(heston_params):
    c = GMMBContract(maturity=4.0)
    t, s, v, h = 1.0, 90.0, 0.05, 1e-3
    val = lambda x, var: float(heston.gmmb_value_and_greeks(c, heston_params, x, t, var)[0][0])
    _, delta, dv = heston.gmmb_value_and_greeks(c, heston_params, s, t, v)
    assert float(delta[0]) == pytest.approx((val(s + h, v) - val(s - h, v)) / (2 * h), abs=1e-5)
    assert float(dv[0]) == pytest.approx((val(s, v + 1e-5) - val(s, v - 1e-5)) / 2e-5, rel=1e-4)


def test_rejects_unknown_method(contract, gbm):
    with pytest.raises(ValueError):
        price_gmmb(contract, gbm, 10, "quasi")
