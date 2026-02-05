import numpy as np
import pytest

from ghedge.rates import HullWhite, ZeroCurve, fit_to_yield_volatility, yield_volatility
from ghedge.rates.curve import _cashflows
from ghedge.rates.hullwhite import b_factor, mc_bond_option, mc_zcb, ou_step_cov
from ghedge.data import load_treasury


@pytest.fixture(scope="module")
def hw(treasury_curve):
    return HullWhite(treasury_curve, a=0.05, sigma=0.01)


def test_affine_bond_price_matches_curve_at_time_zero(hw, treasury_curve):
    t = np.array([0.5, 1.0, 5.0, 10.0, 30.0])
    np.testing.assert_allclose(hw.zcb(0.0, t, hw.r0), treasury_curve.discount(t), rtol=1e-12)
    assert float(hw.zcb(3.0, 3.0, 0.05)) == pytest.approx(1.0)


def test_theta_is_the_drift_of_alpha(hw):
    """r = x + alpha with dx = -a x dt + sigma dW, so theta = alpha' + a alpha."""
    t = np.array([0.7, 2.0, 6.0, 12.0, 25.0])
    h = 1e-4
    d_alpha = (hw.alpha(t + h) - hw.alpha(t - h)) / (2 * h)
    np.testing.assert_allclose(hw.theta(t), d_alpha + hw.a * hw.alpha(t), atol=2e-6)


def test_mc_zero_coupon_bonds_reproduce_the_curve(hw, treasury_curve):
    maturities = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    est, se = mc_zcb(hw, maturities, 100_000, rng=1)
    z = (est - treasury_curve.discount(maturities)) / se
    assert np.all(np.abs(z) < 3), z


def test_multi_step_simulation_matches_moments(hw, treasury_curve):
    paths = hw.simulate(10.0, 40, 40_000, rng=2)
    for j in [8, 20, 40]:
        t = paths.times[j]
        r = paths.short_rate[:, j]
        var_r = hw.sigma**2 * b_factor(2 * hw.a, t)
        assert abs(r.mean() - float(hw.alpha(t))) < 3 * np.sqrt(var_r / r.size)
        assert r.var() == pytest.approx(var_r, rel=0.04)
        d = paths.discount[:, j]
        assert abs(d.mean() - float(treasury_curve.discount(t))) < 3 * d.std() / np.sqrt(d.size)
        # Var(ln D) = Var(int x ds)
        assert np.log(d).var() == pytest.approx(float(hw.integral_variance(t)), rel=0.04)


def test_step_covariance_against_fine_euler():
    a, s, dt = 0.3, 0.02, 1.0
    cov = ou_step_cov(a, s, dt)
    rng = np.random.default_rng(3)
    n, m = 200_000, 400
    h = dt / m
    x = np.zeros(n)
    integ = np.zeros(n)
    for _ in range(m):
        integ += x * h
        x += -a * x * h + s * np.sqrt(h) * rng.standard_normal(n)
    emp = np.cov(np.vstack([x, integ]))
    np.testing.assert_allclose(emp, cov, rtol=0.03)


def test_zero_volatility_is_deterministic(treasury_curve):
    hw0 = HullWhite(treasury_curve, a=0.1, sigma=0.0)
    paths = hw0.simulate(10.0, 10, 100, rng=0)
    np.testing.assert_allclose(paths.discount, np.broadcast_to(treasury_curve.discount(paths.times),
                                                               paths.discount.shape), rtol=1e-12)
    np.testing.assert_allclose(paths.short_rate[:, 3], treasury_curve.forward(paths.times[3]), atol=1e-12)


def test_zcb_option_put_call_parity(hw, treasury_curve):
    T, S = 3.0, 10.0
    for k in [0.6, 0.75, 0.9]:
        c, p = hw.zcb_option(T, S, k, "call"), hw.zcb_option(T, S, k, "put")
        assert c - p == pytest.approx(treasury_curve.discount(S) - k * treasury_curve.discount(T), abs=1e-14)


@pytest.mark.parametrize("control", [False, True])
def test_mc_zcb_options_within_three_standard_errors(hw, treasury_curve, control):
    T, S = 5.0, 10.0
    atm = float(treasury_curve.discount(S) / treasury_curve.discount(T))
    for i, k in enumerate([0.95 * atm, atm, 1.05 * atm]):
        for kind in ["call", "put"]:
            exact = float(hw.zcb_option(T, S, k, kind))
            res = mc_bond_option(hw, T, [S], [1.0], k, kind, 200_000, rng=10 + i, control=control)
            assert abs(res.estimate - exact) < 3 * res.std_error, (k, kind, res.estimate, exact)


@pytest.mark.parametrize("control", [False, True])
def test_mc_coupon_bond_options_match_jamshidian(hw, control):
    """Option at 5 years on a 10-year 5% semi-annual bond."""
    T = 5.0
    times, amounts = _cashflows(10.0, 0.05)
    times = times + T
    for i, k in enumerate([0.9, 1.0, 1.1]):
        for kind in ["call", "put"]:
            exact = hw.coupon_bond_option(T, times, amounts, k, kind)
            res = mc_bond_option(hw, T, times, amounts, k, kind, 200_000, rng=20 + i, control=control)
            assert abs(res.estimate - exact) < 3 * res.std_error, (k, kind, res.estimate, exact)


def test_control_variates_reduce_bond_option_error(hw):
    times, amounts = _cashflows(10.0, 0.05)
    plain = mc_bond_option(hw, 5.0, times + 5.0, amounts, 1.0, "put", 100_000, rng=5)
    cv = mc_bond_option(hw, 5.0, times + 5.0, amounts, 1.0, "put", 100_000, rng=5, control=True)
    assert cv.std_error < 0.5 * plain.std_error


def test_jamshidian_single_cash_flow_equals_zcb_option(hw):
    assert hw.coupon_bond_option(2.0, [7.0], [1.0], 0.8, "put") == pytest.approx(
        float(hw.zcb_option(2.0, 7.0, 0.8, "put")), rel=1e-10)


def test_bond_option_value_grows_with_rate_volatility(treasury_curve):
    vals = [float(HullWhite(treasury_curve, 0.05, s).zcb_option(5.0, 10.0, 0.75, "call"))
            for s in [0.005, 0.01, 0.02]]
    assert vals[0] < vals[1] < vals[2]


def test_fit_to_yield_volatility_recovers_parameters():
    tau = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    vols = 0.012 * b_factor(0.08, tau) / tau
    a, s = fit_to_yield_volatility(tau, vols)
    assert a == pytest.approx(0.08, rel=1e-4) and s == pytest.approx(0.012, rel=1e-4)


def test_yield_volatility_from_snapshot():
    vols = yield_volatility(load_treasury(), start="2023-10-01")
    assert vols.index.min() >= 2.0
    assert ((vols > 0.002) & (vols < 0.03)).all()


def test_invalid_parameters(treasury_curve):
    with pytest.raises(ValueError):
        HullWhite(treasury_curve, a=0.0)
    with pytest.raises(ValueError):
        HullWhite(treasury_curve, a=0.1).zcb_option(1.0, 2.0, 0.9, "straddle")
    with pytest.raises(ValueError):
        HullWhite(treasury_curve, a=0.1).coupon_bond_option(5.0, [3.0, 6.0], [0.1, 1.0], 1.0)
