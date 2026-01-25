import numpy as np
import pytest

from ghedge import GBM, GMMBContract, Heston
from ghedge import blackscholes as bs
from ghedge.hedging import BSHedger, HestonHedger, backtest, cvar, rebalancing_table, risk_summary
from ghedge.models import Paths


@pytest.fixture(scope="module")
def short_contract():
    return GMMBContract(maturity=2.0)


@pytest.fixture(scope="module")
def gbm_paths(short_contract):
    return GBM(0.2, mu=0.07).simulate(100.0, short_contract.rate, 2.0, 504, 4_000, rng=10)


def test_hedge_error_falls_with_rebalancing_frequency(short_contract, gbm_paths):
    table = rebalancing_table(gbm_paths, short_contract, [BSHedger(short_contract, 0.2)])
    std = table.set_index("frequency")["std_pnl"]
    assert std["daily"] < std["weekly"] < std["monthly"]
    # Discrete-hedging error scales like sqrt(dt): 21 days vs 1 day -> about sqrt(21) = 4.6.
    assert 3.0 < std["monthly"] / std["daily"] < 6.5
    # Daily hedging leaves a small error relative to the guarantee price.
    assert std["daily"] < 0.1 * table["price"].iloc[0]


def test_hedge_is_unbiased_without_costs(short_contract, gbm_paths):
    res = backtest(gbm_paths, short_contract, BSHedger(short_contract, 0.2), 1, 0.0)
    se = res.pnl.std(ddof=1) / np.sqrt(res.pnl.size)
    assert abs(res.pnl.mean()) < 3 * se


def test_transaction_costs_lower_pnl_and_grow_with_frequency(short_contract, gbm_paths):
    h = BSHedger(short_contract, 0.2)
    free = backtest(gbm_paths, short_contract, h, 1, 0.0)
    daily = backtest(gbm_paths, short_contract, h, 1, 0.002)
    monthly = backtest(gbm_paths, short_contract, h, 21, 0.002)
    assert np.all(daily.pnl <= free.pnl + 1e-12)
    assert daily.costs.mean() > monthly.costs.mean() > 0
    np.testing.assert_allclose(free.pnl - daily.pnl, daily.costs, rtol=1e-9, atol=1e-9)


def test_unhedged_position_pnl_is_price_minus_claim():
    c = GMMBContract(maturity=1.0)

    class NoHedge:
        label = "none"
        def value(self, t, s, v=None):
            return np.full_like(np.asarray(s, dtype=float), 5.0)
        def delta(self, t, s, v=None):
            return np.zeros_like(np.asarray(s, dtype=float))

    s = np.array([[100.0, 90.0, 80.0], [100.0, 110.0, 120.0]])
    paths = Paths(np.array([0.0, 0.5, 1.0]), s, None, np.zeros(2))
    res = backtest(paths, c, NoHedge())
    expected = 5.0 - np.exp(-c.rate) * c.payoff(s[:, -1])
    np.testing.assert_allclose(res.pnl, expected)


def test_single_period_hedge_accounting():
    """One rebalance at t=0 then hold: check against a hand calculation."""
    c = GMMBContract(maturity=1.0, rate=0.05)
    h = BSHedger(c, 0.2)
    s = np.array([[100.0, 95.0, 85.0]])
    paths = Paths(np.array([0.0, 0.5, 1.0]), s, None, np.zeros(1))
    res = backtest(paths, c, h, rebalance_every=10, cost_rate=0.01)
    v0 = float(bs.gmmb_value(c, 0.2))
    d0 = float(bs.gmmb_delta(c, 0.2))
    cash_T = (v0 - d0 * 100.0 - 0.01 * abs(d0) * 100.0) * np.exp(0.05)
    port = cash_T + d0 * 85.0 - 0.01 * abs(d0) * 85.0
    expected = np.exp(-0.05) * (port - float(c.payoff(85.0)))
    assert res.pnl[0] == pytest.approx(expected, rel=1e-12)


def test_risk_summary_values():
    pnl = -np.arange(100, dtype=float)  # losses 0..99
    out = risk_summary(pnl, 0.95)
    assert out["var_95"] == pytest.approx(np.quantile(np.arange(100.0), 0.95))
    assert out["cvar_95"] == pytest.approx(np.mean([95, 96, 97, 98, 99]))
    assert cvar(pnl) >= out["var_95"]


def test_model_risk_bs_hedge_worse_under_heston():
    """Hedging a Heston market with Black-Scholes deltas leaves a much larger
    error than hedging a GBM market, and the minimum-variance Heston delta
    beats the plain Black-Scholes hedge."""
    c = GMMBContract(maturity=2.0)
    h = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, mu=0.07)
    hp = h.simulate(100.0, c.rate, 2.0, 504, 1_000, rng=11)
    gp = GBM(0.2, mu=0.07).simulate(100.0, c.rate, 2.0, 504, 1_000, rng=11)
    bs_hedger = BSHedger(c, 0.2)
    heston_price = float(HestonHedger(c, h).value(0.0, 100.0, h.v0)[0])
    under_heston = backtest(hp, c, bs_hedger, 5, price=heston_price)
    under_gbm = backtest(gp, c, bs_hedger, 5)
    mv = backtest(hp, c, HestonHedger(c, h, minimum_variance=True), 5)
    assert under_heston.pnl.std() > 2.0 * under_gbm.pnl.std()
    assert mv.pnl.std() < under_heston.pnl.std()


def test_heston_hedger_with_tiny_vol_of_vol_matches_bs():
    c = GMMBContract(maturity=1.0)
    h = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=1e-4, rho=-0.7)
    s = np.array([80.0, 100.0, 125.0])
    np.testing.assert_allclose(HestonHedger(c, h).delta(0.3, s, np.full(3, 0.04)),
                               BSHedger(c, 0.2).delta(0.3, s), atol=1e-4)
    with pytest.raises(ValueError):
        HestonHedger(c, h).delta(0.3, s, None)
