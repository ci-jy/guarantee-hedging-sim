from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from ghedge import GMMBContract
from ghedge import blackscholes as bs
from ghedge.montecarlo import price_gmmb
from ghedge.rates import HullWhite, HybridModel, ZeroCurve
from ghedge.rates import hybrid
from ghedge.rates.hullwhite import b_factor


@pytest.fixture(scope="module")
def hw(treasury_curve):
    return HullWhite(treasury_curve, a=0.05, sigma=0.01)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("maturity", [10.0, 20.0])
@pytest.mark.parametrize("method", ["plain", "antithetic_control"])
def test_mc_matches_stochastic_rate_closed_form(hw, rho, maturity, method):
    model = HybridModel(hw, sigma=0.2, rho=rho)
    c = GMMBContract(maturity=maturity)
    exact = float(hybrid.gmmb_value(c, model))
    seed = int(100 * maturity + 10 * rho + 3)
    res = price_gmmb(c, model, 200_000, method, rng=seed)
    assert abs(res.estimate - exact) < 3 * res.std_error, (res.estimate, exact, res.std_error)


def test_mc_on_multi_step_grid_matches_closed_form(hw):
    model = HybridModel(hw, sigma=0.2, rho=0.3)
    c = GMMBContract(maturity=10.0)
    paths = model.simulate(100.0, 10.0, 40, 100_000, rng=4)
    y = paths.discount[:, -1] * c.payoff(paths.terminal)
    assert abs(y.mean() - float(hybrid.gmmb_value(c, model))) < 3 * y.std() / np.sqrt(y.size)


def test_discounted_index_and_discount_factor_are_martingales(hw, treasury_curve):
    model = HybridModel(hw, sigma=0.25, rho=-0.4)
    paths = model.simulate(100.0, 15.0, 30, 50_000, rng=6)
    disc_index = paths.discount * paths.index
    for j in [10, 30]:
        x = disc_index[:, j]
        assert abs(x.mean() - 100.0) < 3 * x.std() / np.sqrt(x.size)
        d = paths.discount[:, j]
        assert abs(d.mean() - float(treasury_curve.discount(paths.times[j]))) < 3 * d.std() / np.sqrt(d.size)
    # The bond price path ends at par and starts at the curve's discount factor.
    np.testing.assert_allclose(paths.bond[:, -1], 1.0, atol=1e-12)
    np.testing.assert_allclose(paths.bond[:, 0], treasury_curve.discount(15.0), rtol=1e-12)


def test_equity_rate_correlation(hw):
    model = HybridModel(hw, sigma=0.2, rho=0.6)
    paths = model.simulate(100.0, 1.0, 50, 20_000, rng=7)
    dlog_s = np.diff(np.log(paths.index), axis=1).ravel()
    dr = np.diff(paths.short_rate, axis=1).ravel()
    assert np.corrcoef(dlog_s, dr)[0, 1] == pytest.approx(0.6, abs=0.02)


def test_forward_variance_matches_numerical_integral(hw):
    model = HybridModel(hw, sigma=0.18, rho=-0.3)
    tau = 12.0
    integrand = lambda u: (0.18**2 + 2 * -0.3 * 0.18 * hw.sigma * b_factor(hw.a, tau - u)
                           + hw.sigma**2 * b_factor(hw.a, tau - u) ** 2)
    assert float(model.forward_variance(tau)) == pytest.approx(quad(integrand, 0, tau)[0], rel=1e-10)


def test_zero_rate_volatility_reduces_to_black_scholes():
    curve = ZeroCurve.flat(0.03)
    model = HybridModel(HullWhite(curve, a=0.1, sigma=0.0), sigma=0.2, rho=0.5)
    c = GMMBContract(maturity=10.0, rate=0.03)
    assert float(hybrid.gmmb_value(c, model)) == pytest.approx(float(bs.gmmb_value(c, 0.2)), rel=1e-10)
    assert float(hybrid.deterministic_rate_value(c, replace(model, hw=replace(model.hw, sigma=0.02)))) == \
        pytest.approx(float(bs.gmmb_value(c, 0.2)), rel=1e-10)


def test_value_rises_with_correlation_and_rate_volatility(hw):
    c = GMMBContract(maturity=15.0)
    grid = hybrid.sensitivity_grid(c, HybridModel(hw, 0.2), [0.0, 0.005, 0.01, 0.015], [-0.5, 0.0, 0.5])
    assert np.all(np.diff(grid.to_numpy()[1:], axis=1) > 0)      # increasing in rho
    assert np.all(np.diff(grid[0.0].to_numpy()) > 0)               # increasing in rate vol at rho = 0
    assert np.allclose(grid.loc[0.0], grid.loc[0.0].iloc[0])        # rho irrelevant without rate vol


def test_deltas_match_finite_differences_and_replicate(hw):
    model = HybridModel(hw, sigma=0.2, rho=-0.2)
    c = GMMBContract(maturity=10.0)
    s, p, t = np.array([80.0, 100.0, 130.0]), np.array([0.7, 0.65, 0.6]), 2.5
    d_s, d_p = hybrid.gmmb_deltas(c, model, s, t, p)
    v = lambda ss, pp: hybrid.gmmb_value(c, model, ss, t, pp)
    np.testing.assert_allclose(d_s, (v(s + 1e-4, p) - v(s - 1e-4, p)) / 2e-4, rtol=1e-6)
    np.testing.assert_allclose(d_p, (v(s, p + 1e-7) - v(s, p - 1e-7)) / 2e-7, rtol=1e-5)
    np.testing.assert_allclose(s * d_s + p * d_p, v(s, p), rtol=1e-12)


def test_real_world_drift_is_removed_for_pricing(hw):
    c = GMMBContract(maturity=10.0)
    rn = price_gmmb(c, HybridModel(hw, 0.2), 20_000, rng=1).estimate
    rw = price_gmmb(c, HybridModel(hw, 0.2, equity_premium=0.05), 20_000, rng=1).estimate
    assert rn == rw


def test_invalid_correlation(hw):
    with pytest.raises(ValueError):
        HybridModel(hw, rho=1.5)
