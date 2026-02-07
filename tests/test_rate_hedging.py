import numpy as np
import pytest

from ghedge import GMMBContract
from ghedge.hedging import BSHedger, backtest
from ghedge.models import Paths
from ghedge.rates import HullWhite, HybridHedger, HybridModel, ZeroCurve


@pytest.fixture(scope="module")
def setup(treasury_curve):
    c = GMMBContract(maturity=10.0)
    model = HybridModel(HullWhite(treasury_curve, a=0.05, sigma=0.01), sigma=0.2, rho=0.0,
                        equity_premium=0.04)
    paths = model.simulate(100.0, 10.0, 520, 1_500, rng=12)
    r_eff = float(-np.log(treasury_curve.discount(10.0)) / 10.0)
    bs_hedger = BSHedger(c.with_(rate=r_eff), float(model.effective_vol(10.0)), label="BS")
    return c, model, paths, bs_hedger


def test_bond_hedge_removes_most_of_the_rate_driven_error(setup):
    c, model, paths, bs_hedger = setup
    both = backtest(paths, c, HybridHedger(c, model, use_bond=True), 1)
    equity_only = backtest(paths, c, HybridHedger(c, model, use_bond=False), 1)
    bs_only = backtest(paths, c, bs_hedger, 1)
    assert both.pnl.std() < 0.3 * equity_only.pnl.std()
    assert both.pnl.std() < 0.3 * bs_only.pnl.std()
    # All three charge the same stochastic-rate value at issue.
    assert both.price[0] == pytest.approx(bs_only.price[0], rel=1e-10)


def test_bond_hedge_is_unbiased_and_error_falls_with_frequency(setup):
    c, model, paths, _ = setup
    h = HybridHedger(c, model)
    weekly = backtest(paths, c, h, 1)
    monthly = backtest(paths, c, h, 4)
    quarterly = backtest(paths, c, h, 13)
    assert abs(weekly.pnl.mean()) < 3 * weekly.pnl.std() / np.sqrt(weekly.pnl.size)
    assert weekly.pnl.std() < monthly.pnl.std() < quarterly.pnl.std()


def test_costs_include_bond_trades(setup):
    c, model, paths, _ = setup
    h = HybridHedger(c, model)
    free = backtest(paths, c, h, 4)
    costly = backtest(paths, c, h, 4, cost_rate=0.001, bond_cost_rate=0.0005)
    equity_cost_only = backtest(paths, c, h, 4, cost_rate=0.001, bond_cost_rate=0.0)
    np.testing.assert_allclose(free.pnl - costly.pnl, costly.costs, rtol=1e-9, atol=1e-9)
    assert np.all(costly.costs > equity_cost_only.costs)


def test_deterministic_rates_reproduce_the_flat_rate_backtest():
    """With zero rate volatility on a flat curve the hybrid paths, hedger and
    pathwise discounting reduce to the existing flat-rate Black-Scholes case."""
    c = GMMBContract(maturity=2.0, rate=0.03)
    model = HybridModel(HullWhite(ZeroCurve.flat(0.03), a=0.1, sigma=0.0), sigma=0.2,
                        equity_premium=0.04)
    paths = model.simulate(100.0, 2.0, 104, 300, rng=1)
    flat = Paths(paths.times, paths.index, None, paths.brownian)
    ref = backtest(flat, c, BSHedger(c, 0.2), 1, cost_rate=0.001)
    for h in [HybridHedger(c, model, use_bond=False), BSHedger(c, 0.2)]:
        res = backtest(paths, c, h, 1, cost_rate=0.001)
        np.testing.assert_allclose(res.pnl, ref.pnl, rtol=1e-8, atol=1e-10)


def test_single_period_bond_hedge_accounting():
    c = GMMBContract(maturity=1.0)

    class Fixed:
        label = "fixed"
        def value(self, t, s, v=None, rate=None):
            return np.full_like(np.asarray(s, dtype=float), 4.0)
        def delta(self, t, s, v=None, rate=None):
            return np.full_like(np.asarray(s, dtype=float), -0.3)
        def bond_units(self, t, s, v=None, rate=None):
            return np.full_like(np.asarray(s, dtype=float), 50.0)

    times = np.array([0.0, 0.5, 1.0])
    s = np.array([[100.0, 90.0, 80.0]])
    disc = np.array([[1.0, 0.98, 0.95]])
    bond = np.array([[0.95, 0.97, 1.0]])
    paths = Paths(times, s, None, np.zeros(1), short_rate=np.full((1, 3), 0.04), discount=disc, bond=bond)
    res = backtest(paths, c, Fixed(), rebalance_every=10, cost_rate=0.01, bond_cost_rate=0.002)
    cash0 = 4.0 + 0.3 * 100 - 0.01 * 0.3 * 100 - 50 * 0.95 - 0.002 * 50 * 0.95
    port = cash0 / 0.95 - 0.3 * 80 - 0.01 * 0.3 * 80 + 50 * 1.0 - 0.002 * 50
    assert res.pnl[0] == pytest.approx(0.95 * (port - float(c.payoff(80.0))), rel=1e-12)


def test_hedger_needs_rates(setup):
    c, model, paths, _ = setup
    with pytest.raises(ValueError):
        HybridHedger(c, model).delta(0.0, 100.0)
    flat = Paths(paths.times, paths.index, None, paths.brownian)
    with pytest.raises(ValueError):
        backtest(flat, c, HybridHedger(c, model))
