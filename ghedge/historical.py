"""Hedging the GMMB on historical index paths.

Overlapping contracts are issued at regular intervals along the historical
series. Each contract's index path is rescaled to start at the premium and
hedged with Black-Scholes deltas at a volatility fixed at issue (the trailing
realised volatility, or a constant). Since all windows have the same length
they are stacked and hedged in one vectorised pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import GMMBContract
from .hedging import FREQUENCIES, TRADING_DAYS, BSHedger, backtest, risk_summary
from .models import Paths


def realised_vol(levels: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Annualised trailing volatility of daily log returns."""
    log_ret = np.log(levels).diff()
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def historical_windows(levels: pd.Series, term_days: int, start_every: int = 21,
                       vol_window: int = TRADING_DAYS, premium: float = 100.0):
    """Stack rescaled index windows of ``term_days`` trading days.

    Returns ``(paths, starts, issue_vol, realised)`` where ``issue_vol`` is the
    trailing volatility known at issue and ``realised`` the volatility realised
    over the contract's life.
    """
    values = levels.to_numpy(dtype=float)
    trailing = realised_vol(levels, vol_window).to_numpy()
    first = vol_window
    starts = np.arange(first, len(values) - term_days, start_every)
    if starts.size == 0:
        raise ValueError("series too short for the requested term and volatility window")
    block = np.stack([values[i:i + term_days + 1] for i in starts])
    block = premium * block / block[:, :1]
    log_ret = np.diff(np.log(block), axis=1)
    realised = log_ret.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    maturity = term_days / TRADING_DAYS
    times = np.linspace(0.0, maturity, term_days + 1)
    paths = Paths(times, block, None, np.zeros(len(starts)))
    return paths, levels.index[starts], trailing[starts], realised


def historical_backtest(levels: pd.Series, contract: GMMBContract, term_days: int | None = None,
                        start_every: int = 21, vol_window: int = TRADING_DAYS, sigma=None,
                        frequencies=None, cost_rate: float = 0.0):
    """Hedge overlapping GMMB contracts along a historical index series.

    ``sigma=None`` uses the trailing realised volatility at issue for both
    pricing and hedging; a number fixes the volatility for every contract.
    Returns ``(per_window, summary)`` DataFrames. P&L is per initial policy
    of size ``contract.premium`` and discounted to the issue date.
    """
    term_days = term_days or int(round(contract.maturity * TRADING_DAYS))
    paths, starts, issue_vol, realised = historical_windows(levels, term_days, start_every,
                                                            vol_window, contract.premium)
    contract = contract.with_(maturity=paths.times[-1])
    sig = issue_vol if sigma is None else np.full(len(starts), float(sigma))
    hedger = BSHedger(contract, sig, label="BS delta (trailing vol)" if sigma is None
                      else f"BS delta (sigma={float(sigma):.1%})")
    frequencies = frequencies or FREQUENCIES
    per_window = pd.DataFrame({"issue_date": starts, "issue_vol": issue_vol,
                               "realised_vol": realised,
                               "index_return": paths.index[:, -1] / paths.index[:, 0] - 1.0})
    rows = []
    for name, k in frequencies.items():
        res = backtest(paths, contract, hedger, k, cost_rate)
        per_window[f"pnl_{name}"] = res.pnl
        rows.append({"frequency": name, "rebalance_every": k, "n_contracts": len(starts),
                     "mean_price": float(np.mean(res.price)), **risk_summary(res.pnl)})
    return per_window, pd.DataFrame(rows)
