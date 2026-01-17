"""Semi-analytic Heston put prices and sensitivities.

Uses the Gil-Pelaez inversion ``P_j = 1/2 + 1/pi * int_0^inf Re[...] du`` with
the "little Heston trap" form of the characteristic function (Albrecher et al.,
2007), which avoids branch-cut discontinuities for long maturities.

The integrals are evaluated on a shared quadrature grid so that thousands of
(spot, variance) states at the same time-to-maturity are priced in one
vectorised pass; the hedging backtester relies on this.
"""
from __future__ import annotations

import numpy as np

from .models import Heston

_LEG_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _legendre_01(n):
    if n not in _LEG_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _LEG_CACHE[n] = (0.5 * (x + 1.0), 0.5 * w)
    return _LEG_CACHE[n]


def _cf_coeffs(u, tau, carry, p: Heston):
    """C(u) and D(u) with ``phi(u) = exp(C + D v + i u ln S)`` (u may be complex)."""
    iu = 1j * u
    a = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(a * a + p.xi**2 * (iu + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * tau)
    c = carry * iu * tau + p.kappa * p.theta / p.xi**2 * (
        (a - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
    )
    dd = (a - d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    return c, dd


def put_and_greeks(spot, strike, tau, rate, yield_, variance, params: Heston, n_nodes=192):
    """Heston put price, delta (dV/dS) and variance sensitivity (dV/dv).

    ``spot`` and ``variance`` are broadcast together; ``tau`` is a scalar.
    """
    spot = np.atleast_1d(np.asarray(spot, dtype=float))
    variance = np.atleast_1d(np.asarray(variance, dtype=float))
    spot, variance = np.broadcast_arrays(spot, variance)
    if tau <= 1e-10:
        price = np.maximum(strike - spot, 0.0)
        delta = -(spot < strike).astype(float)
        return price, delta, np.zeros_like(spot)

    carry = rate - yield_
    # Map Legendre nodes on (0, 1) to (0, inf); the scale puts the bulk of
    # the integrand's decay (roughly exp(-u^2 v tau / 2)) mid-interval.
    x, w = _legendre_01(n_nodes)
    v_ref = max(params.theta, 1e-3)
    scale = 1.0 / np.sqrt(v_ref * tau)
    u = scale * x / (1.0 - x)
    du = w * scale / (1.0 - x) ** 2

    c2, d2 = _cf_coeffs(u, tau, carry, params)
    c1, d1 = _cf_coeffs(u - 1j, tau, carry, params)
    c1 = c1 - carry * tau

    logm = np.log(spot / strike)[:, None]
    v = np.maximum(variance, 0.0)[:, None]
    iu = 1j * u
    phase = iu * logm
    f1 = np.exp(c1 + d1 * v + phase) / iu
    f2 = np.exp(c2 + d2 * v + phase) / iu

    p1 = 0.5 + (f1.real @ du) / np.pi
    p2 = 0.5 + (f2.real @ du) / np.pi
    dp1 = ((d1 * f1).real @ du) / np.pi
    dp2 = ((d2 * f2).real @ du) / np.pi

    df_s = spot * np.exp(-yield_ * tau)
    df_k = strike * np.exp(-rate * tau)
    price = df_k * (1.0 - p2) - df_s * (1.0 - p1)
    # Guard against tiny negative values from quadrature error.
    price = np.maximum(price, np.maximum(df_k - df_s, 0.0))
    delta = np.exp(-yield_ * tau) * (p1 - 1.0)
    dv = df_s * dp1 - df_k * dp2
    return price, delta, dv


def put_price(spot, strike, tau, rate, yield_, variance, params: Heston, n_nodes=192):
    return put_and_greeks(spot, strike, tau, rate, yield_, variance, params, n_nodes)[0]


def gmmb_value_and_greeks(contract, params: Heston, index=None, t=0.0, variance=None):
    """Cohort GMMB value, index delta and variance sensitivity under Heston."""
    index = contract.premium if index is None else index
    variance = params.v0 if variance is None else variance
    fund_t = contract.fund(index, t)
    tau = contract.maturity - float(t)
    price, delta_f, dv = put_and_greeks(fund_t, contract.guarantee, tau, contract.rate,
                                        contract.fee, variance, params)
    surv = contract.maturity_survival
    return surv * price, surv * delta_f * np.exp(-contract.fee * t), surv * dv
