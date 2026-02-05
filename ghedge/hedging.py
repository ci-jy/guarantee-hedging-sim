"""Discrete delta-hedging backtester with proportional transaction costs.

The insurer sells the guarantee, receives its model value as the guarantee price at
time 0 and holds ``delta`` units of the index (``delta = dV/dS``, negative for
a put, i.e. a short index position) with the remainder in a cash account
earning the flat rate. The position is rebalanced every ``k`` grid steps;
each trade costs ``cost_rate * |traded units| * S``. At maturity the hedge is
unwound (also at a cost) and the guarantee claim is paid.

The hedge P&L per initial policy, discounted to time 0, is
``PV(final portfolio) - PV(guarantee claim)``. A perfect continuous hedge
with no costs gives zero on every path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import blackscholes as bs
from . import heston
from .contract import GMMBContract
from .models import Heston, Paths

FREQUENCIES = {"daily": 1, "weekly": 5, "monthly": 21}
TRADING_DAYS = 252


class BSHedger:
    """Values and hedges with Black-Scholes at a fixed volatility and the
    contract's flat rate (a simulated short rate, if given, is ignored)."""

    def __init__(self, contract: GMMBContract, sigma: float, label: str | None = None):
        self.contract, self.sigma = contract, sigma
        self.label = label or f"BS delta (sigma={sigma:.1%})"

    def value(self, t, index, variance=None, rate=None):
        return bs.gmmb_value(self.contract, self.sigma, index, t)

    def delta(self, t, index, variance=None, rate=None):
        return bs.gmmb_delta(self.contract, self.sigma, index, t)


class HestonHedger:
    """Values and hedges with the Heston model, observing the current variance.

    With ``minimum_variance=True`` the index position also offsets the part of
    the variance risk that is correlated with the index:
    ``delta_MV = dV/dS + rho * xi / S * dV/dv``.
    """

    def __init__(self, contract: GMMBContract, params: Heston, minimum_variance=False,
                 label: str | None = None):
        self.contract, self.params, self.minimum_variance = contract, params, minimum_variance
        self.label = label or ("Heston min-variance delta" if minimum_variance else "Heston delta")

    def _greeks(self, t, index, variance):
        if variance is None:
            raise ValueError("Heston hedger needs the variance path")
        return heston.gmmb_value_and_greeks(self.contract, self.params, index, t, variance)

    def value(self, t, index, variance=None):
        return self._greeks(t, index, variance)[0]

    def delta(self, t, index, variance=None):
        _, d, dv = self._greeks(t, index, variance)
        if self.minimum_variance:
            d = d + self.params.rho * self.params.xi / np.asarray(index) * dv
        return d


@dataclass
class HedgeResult:
    label: str
    rebalance_every: int
    cost_rate: float
    price: np.ndarray
    pnl: np.ndarray
    costs: np.ndarray

    def summary(self, alpha: float = 0.95) -> dict:
        return {"strategy": self.label, "rebalance_every": self.rebalance_every,
                "cost_rate": self.cost_rate, "price": float(np.mean(self.price)),
                **risk_summary(self.pnl, alpha), "mean_costs": float(self.costs.mean())}


def risk_summary(pnl, alpha: float = 0.95) -> dict:
    """Mean, standard deviation, VaR and CVaR (expected shortfall) of the loss
    ``-pnl`` at confidence ``alpha``."""
    loss = -np.asarray(pnl, dtype=float)
    var = float(np.quantile(loss, alpha))
    tail = loss[loss >= var]
    return {"mean_pnl": float(np.mean(pnl)), "std_pnl": float(np.std(pnl, ddof=1)),
            f"var_{int(alpha * 100)}": var, f"cvar_{int(alpha * 100)}": float(tail.mean())}


def cvar(pnl, alpha: float = 0.95) -> float:
    return risk_summary(pnl, alpha)[f"cvar_{int(alpha * 100)}"]


def backtest(paths: Paths, contract: GMMBContract, hedger, rebalance_every: int = 1,
             cost_rate: float = 0.0, price=None, bond_cost_rate: float | None = None) -> HedgeResult:
    """Run the discrete hedge on every simulated (or historical) path.

    ``paths.times`` must start at 0 and end at the contract maturity.
    ``price`` (the amount charged for the guarantee; scalar or one value per
    path) defaults to the hedger's own
    model value at time 0.

    If the paths carry a stochastic discount factor (``paths.discount``), cash
    accrues at the simulated short rate, P&L is discounted pathwise and the
    hedger is called with the current short rate (``rate=`` keyword). A hedger
    with a ``bond_units`` method also trades the zero-coupon bond whose prices
    are in ``paths.bond``, paying ``bond_cost_rate`` (default: ``cost_rate``)
    per unit of value traded.
    """
    times = paths.times
    s = paths.index
    v = paths.variance
    n_steps = len(times) - 1
    if not np.isclose(times[-1], contract.maturity):
        raise ValueError("path grid must end at the contract maturity")
    r = contract.rate
    var_at = (lambda j: None) if v is None else (lambda j: v[:, j])
    stochastic = paths.discount is not None
    if stochastic:
        disc = paths.discount
        state = lambda j: {"rate": paths.short_rate[:, j]}
        growth = lambda j0, j1: disc[:, j0] / disc[:, j1]
        pv = lambda j: disc[:, j]
    else:
        state = lambda j: {}
        growth = lambda j0, j1: np.exp(r * (times[j1] - times[j0]))
        pv = lambda j: np.exp(-r * times[j])
    use_bond = hasattr(hedger, "bond_units")
    if use_bond and paths.bond is None:
        raise ValueError("hedger trades a bond but the paths carry no bond prices")
    bond_cost_rate = cost_rate if bond_cost_rate is None else bond_cost_rate

    if price is None:
        price = hedger.value(0.0, s[:, 0], var_at(0), **state(0))
    price = np.broadcast_to(np.asarray(price, dtype=float), s[:, 0].shape)
    dates = list(range(0, n_steps, rebalance_every))

    delta = np.broadcast_to(hedger.delta(times[0], s[:, 0], var_at(0), **state(0)),
                            s[:, 0].shape).astype(float)
    trade_cost = cost_rate * np.abs(delta) * s[:, 0]
    cash = price - delta * s[:, 0] - trade_cost
    if use_bond:
        units = np.broadcast_to(hedger.bond_units(times[0], s[:, 0], var_at(0), **state(0)),
                                s[:, 0].shape).astype(float)
        bond_cost = bond_cost_rate * np.abs(units) * paths.bond[:, 0]
        cash = cash - units * paths.bond[:, 0] - bond_cost
        trade_cost = trade_cost + bond_cost
    costs = trade_cost.copy()
    j_prev = 0
    for j in dates[1:]:
        t = times[j]
        cash = cash * growth(j_prev, j)
        new_delta = hedger.delta(t, s[:, j], var_at(j), **state(j))
        traded = new_delta - delta
        trade_cost = cost_rate * np.abs(traded) * s[:, j]
        cash = cash - traded * s[:, j] - trade_cost
        if use_bond:
            new_units = hedger.bond_units(t, s[:, j], var_at(j), **state(j))
            bond_cost = bond_cost_rate * np.abs(new_units - units) * paths.bond[:, j]
            cash = cash - (new_units - units) * paths.bond[:, j] - bond_cost
            trade_cost = trade_cost + bond_cost
            units = new_units
        costs += trade_cost * pv(j)
        delta, j_prev = new_delta, j

    cash = cash * growth(j_prev, n_steps)
    unwind_cost = cost_rate * np.abs(delta) * s[:, -1]
    portfolio = cash + delta * s[:, -1] - unwind_cost
    if use_bond:
        bond_cost = bond_cost_rate * np.abs(units) * paths.bond[:, -1]
        portfolio = portfolio + units * paths.bond[:, -1] - bond_cost
        unwind_cost = unwind_cost + bond_cost
    costs += unwind_cost * pv(n_steps)
    pnl = pv(n_steps) * (portfolio - contract.payoff(s[:, -1]))
    return HedgeResult(getattr(hedger, "label", type(hedger).__name__), rebalance_every,
                       cost_rate, price, pnl, costs)


def rebalancing_table(paths: Paths, contract: GMMBContract, hedgers, frequencies=None,
                      cost_rates=(0.0,), price=None) -> pd.DataFrame:
    """Hedge-error statistics for every hedger, frequency and cost rate on the
    same set of paths."""
    frequencies = frequencies or FREQUENCIES
    rows = []
    for hedger in hedgers:
        for name, k in frequencies.items():
            for c in cost_rates:
                res = backtest(paths, contract, hedger, k, c, price)
                rows.append({"frequency": name, **res.summary()})
    return pd.DataFrame(rows)
