"""Closed-form Black-Scholes values and Greeks for a put on an asset with a
continuous yield. For the GMMB the yield is the management fee: the fund
``F_t = S_t exp(-fee t)`` behaves like a stock paying dividends at rate ``fee``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_TINY = 1e-12


def _d1_d2(spot, strike, tau, rate, yield_, sigma):
    spot = np.asarray(spot, dtype=float)
    tau_safe = np.maximum(np.asarray(tau, dtype=float), _TINY)
    vol = sigma * np.sqrt(tau_safe)
    d1 = (np.log(spot / strike) + (rate - yield_ + 0.5 * sigma**2) * tau_safe) / vol
    return d1, d1 - vol


def put_price(spot, strike, tau, rate, yield_, sigma):
    """European put on an asset with continuous yield ``yield_``."""
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, d2 = _d1_d2(spot, strike, tau, rate, yield_, sigma)
    value = strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * np.exp(-yield_ * tau) * norm.cdf(-d1)
    return np.where(tau > _TINY, value, np.maximum(strike - spot, 0.0))


def call_price(spot, strike, tau, rate, yield_, sigma):
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    parity = spot * np.exp(-yield_ * tau) - strike * np.exp(-rate * tau)
    return put_price(spot, strike, tau, rate, yield_, sigma) + parity


def put_delta(spot, strike, tau, rate, yield_, sigma):
    """dPut/dSpot."""
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, _ = _d1_d2(spot, strike, tau, rate, yield_, sigma)
    delta = -np.exp(-yield_ * tau) * norm.cdf(-d1)
    return np.where(tau > _TINY, delta, -(spot < strike).astype(float))


def put_gamma(spot, strike, tau, rate, yield_, sigma):
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, _ = _d1_d2(spot, strike, tau, rate, yield_, sigma)
    tau_safe = np.maximum(tau, _TINY)
    gamma = np.exp(-yield_ * tau) * norm.pdf(d1) / (spot * sigma * np.sqrt(tau_safe))
    return np.where(tau > _TINY, gamma, 0.0)


def put_vega(spot, strike, tau, rate, yield_, sigma):
    """dPut/dsigma (same as the call vega)."""
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, _ = _d1_d2(spot, strike, tau, rate, yield_, sigma)
    return spot * np.exp(-yield_ * tau) * norm.pdf(d1) * np.sqrt(np.maximum(tau, 0.0))


def implied_vol_put(price, spot, strike, tau, rate, yield_, lo=1e-4, hi=3.0):
    """Black-Scholes implied volatility of a put price (scalar)."""
    f = lambda s: float(put_price(spot, strike, tau, rate, yield_, s)) - price
    return brentq(f, lo, hi, xtol=1e-12)


# --- GMMB wrappers -----------------------------------------------------------

def gmmb_value(contract, sigma, index=None, t=0.0):
    """Closed-form cohort value of the maturity guarantee per initial policy.

    ``index`` is the index level at time ``t`` (defaults to the premium at t=0).
    """
    index = contract.premium if index is None else index
    fund_t = contract.fund(index, t)
    tau = contract.maturity - np.asarray(t, dtype=float)
    put = put_price(fund_t, contract.guarantee, tau, contract.rate, contract.fee, sigma)
    return contract.maturity_survival * put


def gmmb_delta(contract, sigma, index=None, t=0.0):
    """Sensitivity of the cohort guarantee value to the index level (hedge ratio
    in index units)."""
    index = contract.premium if index is None else index
    t = np.asarray(t, dtype=float)
    fund_t = contract.fund(index, t)
    tau = contract.maturity - t
    d_put_d_fund = put_delta(fund_t, contract.guarantee, tau, contract.rate, contract.fee, sigma)
    return contract.maturity_survival * d_put_d_fund * np.exp(-contract.fee * t)


def gmmb_gamma(contract, sigma, index=None, t=0.0):
    index = contract.premium if index is None else index
    t = np.asarray(t, dtype=float)
    fund_t = contract.fund(index, t)
    tau = contract.maturity - t
    g = put_gamma(fund_t, contract.guarantee, tau, contract.rate, contract.fee, sigma)
    return contract.maturity_survival * g * np.exp(-2.0 * contract.fee * t)


def gmmb_vega(contract, sigma, index=None, t=0.0):
    index = contract.premium if index is None else index
    fund_t = contract.fund(index, t)
    tau = contract.maturity - np.asarray(t, dtype=float)
    return contract.maturity_survival * put_vega(fund_t, contract.guarantee, tau, contract.rate, contract.fee, sigma)
