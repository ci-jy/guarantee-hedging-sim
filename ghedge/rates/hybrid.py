"""Hybrid model: GBM equity index with correlated Hull-White short rates.

Under the risk-neutral measure

    dS / S = r(t) dt + sigma_S dW_S,     dr = (theta(t) - a r) dt + sigma_r dW_r,
    d<W_S, W_r> = rho dt.

With ``r = x + alpha(t)`` (see :mod:`ghedge.rates.hullwhite`) the triple
``(x, int x ds, W_S)`` is Gaussian over any step, so the index, the short
rate and the discount factor are simulated exactly. The discounted index is
``D(0, t) S_t = S_0 exp(sigma_S W_S - sigma_S^2 t / 2)``, an exact martingale.

**Closed form.** Under the ``T``-forward measure the forward index
``S_t / P(t, T)`` is lognormal with instantaneous volatility vector
``sigma_S dW_S + sigma_r B(t, T) dW_r``, so the maturity guarantee is a
Black-Scholes put with discount factor ``P(0, T)`` and total variance

    v(tau) = sigma_S^2 tau + 2 rho sigma_S sigma_r int_0^tau B(u) du
             + sigma_r^2 int_0^tau B(u)^2 du.

Equivalently: Black-Scholes at the zero rate ``-ln P(0, T) / T`` with the
volatility raised from ``sigma_S`` to ``sqrt(v / T)``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .. import blackscholes as bs
from ..contract import GMMBContract
from ..models import Paths, _normals
from .hullwhite import HullWhite, b_factor, cov_root, ou_step_cov


@dataclass(frozen=True)
class HybridModel:
    """GBM index (volatility ``sigma``) with Hull-White rates ``hw`` and
    instantaneous correlation ``rho`` between the index and the short rate.

    ``equity_premium`` is the real-world excess drift of the index over the
    short rate (0 means risk-neutral). Rates are always simulated under the
    risk-neutral measure (no market price of interest-rate risk).
    """

    hw: HullWhite
    sigma: float = 0.2
    rho: float = 0.0
    equity_premium: float = 0.0

    def __post_init__(self):
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

    def risk_neutral(self) -> "HybridModel":
        return replace(self, equity_premium=0.0) if self.equity_premium else self

    def forward_variance(self, tau):
        """Total variance ``v(tau)`` of the log forward index over ``tau`` years."""
        tau = np.asarray(tau, dtype=float)
        a, sr, ss = self.hw.a, self.hw.sigma, self.sigma
        int_b = (tau - b_factor(a, tau)) / a
        int_b2 = (tau - 2 * b_factor(a, tau) + b_factor(2 * a, tau)) / a**2
        return ss**2 * tau + 2 * self.rho * ss * sr * int_b + sr**2 * int_b2

    def effective_vol(self, tau):
        """Black-Scholes volatility that reproduces the stochastic-rate value."""
        tau = np.asarray(tau, dtype=float)
        return np.sqrt(self.forward_variance(tau) / np.maximum(tau, 1e-12))

    def step_cov(self, dt: float) -> np.ndarray:
        """Covariance of the innovations of ``(x, int x ds, W_S)`` over ``dt``."""
        a, sr = self.hw.a, self.hw.sigma
        cov = np.zeros((3, 3))
        cov[:2, :2] = ou_step_cov(a, sr, dt)
        cov[0, 2] = cov[2, 0] = self.rho * sr * b_factor(a, dt)
        cov[1, 2] = cov[2, 1] = self.rho * sr * (dt - b_factor(a, dt)) / a
        cov[2, 2] = dt
        return cov

    def simulate(self, s0, maturity, n_steps, n_paths, rng=None, antithetic=False,
                 bond_maturity=None) -> Paths:
        """Exact simulation on a uniform grid.

        The returned :class:`Paths` carries the short rate, the discount factor
        ``D(0, t)`` and the price ``P(t, bond_maturity)`` of a zero-coupon bond
        (default maturity: the end of the grid), the second hedge instrument.
        """
        rng = np.random.default_rng(rng)
        hw = self.hw
        dt = maturity / n_steps
        times = np.linspace(0.0, maturity, n_steps + 1)
        z = _normals(rng, n_paths, (n_steps, 3), antithetic) @ cov_root(self.step_cov(dt)).T
        decay = np.exp(-hw.a * dt)
        bdt = float(b_factor(hw.a, dt))
        x = np.zeros((n_paths, n_steps + 1))
        y = np.zeros_like(x)
        w = np.zeros_like(x)
        for k in range(n_steps):
            x[:, k + 1] = x[:, k] * decay + z[:, k, 0]
            y[:, k + 1] = y[:, k] + x[:, k] * bdt + z[:, k, 1]
            w[:, k + 1] = w[:, k] + z[:, k, 2]
        rates = hw._paths_from_factor(times, x, y)
        index = s0 * np.exp((self.equity_premium - 0.5 * self.sigma**2) * times
                            + self.sigma * w) / rates.discount
        t_bond = maturity if bond_maturity is None else bond_maturity
        bond = hw.zcb(times, t_bond, rates.short_rate)
        return Paths(times, index, None, w[:, -1], short_rate=rates.short_rate,
                     discount=rates.discount, bond=bond)

    def simulate_terminal(self, s0, maturity, n_paths, rng=None, antithetic=False):
        """One exact step to ``maturity``: returns ``(S_T, D(0, T), W_S(T))``."""
        p = self.simulate(s0, maturity, 1, n_paths, rng, antithetic)
        return p.terminal, p.discount[:, -1], p.brownian


# --- closed-form guarantee value ---------------------------------------------

def _put_inputs(contract: GMMBContract, model: HybridModel, index, t, bond):
    t = np.asarray(t, dtype=float)
    tau = contract.maturity - t
    tau_safe = np.maximum(tau, 1e-12)
    r_eff = -np.log(bond) / tau_safe
    return contract.fund(index, t), tau, r_eff, model.effective_vol(tau)


def gmmb_value(contract: GMMBContract, model: HybridModel, index=None, t=0.0, bond=None):
    """Closed-form cohort guarantee value with stochastic rates.

    ``bond`` is the price ``P(t, T)`` of the zero-coupon bond maturing with
    the contract (defaults to the initial curve's ``P(0, T)``, valid at t=0).
    """
    index = contract.premium if index is None else index
    bond = model.hw.curve.discount(contract.maturity) if bond is None else bond
    fund_t, tau, r_eff, vol = _put_inputs(contract, model, index, t, bond)
    put = bs.put_price(fund_t, contract.guarantee, tau, r_eff, contract.fee, vol)
    return contract.maturity_survival * put


def gmmb_deltas(contract: GMMBContract, model: HybridModel, index, t, bond):
    """Hedge ratios ``(dV/dS, dV/dP)``: units of the index and of the
    zero-coupon bond maturing at ``T``. ``V`` is homogeneous of degree one in
    ``(S, P)``, so ``V = S dV/dS + P dV/dP`` and the two positions replicate
    the guarantee without a cash account."""
    t = np.asarray(t, dtype=float)
    fund_t, tau, r_eff, vol = _put_inputs(contract, model, index, t, bond)
    p = contract.maturity_survival
    d_s = p * bs.put_delta(fund_t, contract.guarantee, tau, r_eff, contract.fee, vol) \
        * np.exp(-contract.fee * t)
    value = p * bs.put_price(fund_t, contract.guarantee, tau, r_eff, contract.fee, vol)
    d_p = (value - np.asarray(index) * d_s) / bond
    return d_s, d_p


def deterministic_rate_value(contract: GMMBContract, model: HybridModel):
    """Guarantee value if rates followed the initial forward curve
    (``sigma_r = 0``): Black-Scholes at the zero rate to maturity."""
    return gmmb_value(contract, replace(model, hw=replace(model.hw, sigma=0.0)))


def sensitivity_grid(contract: GMMBContract, model: HybridModel, rate_vols, rhos) -> pd.DataFrame:
    """Closed-form guarantee value for each (rate volatility, correlation)."""
    rows = {}
    for sr in rate_vols:
        m = replace(model, hw=replace(model.hw, sigma=float(sr)))
        rows[sr] = [float(gmmb_value(contract, replace(m, rho=float(rho)))) for rho in rhos]
    out = pd.DataFrame(rows, index=list(rhos)).T
    out.index.name, out.columns.name = "rate_vol", "rho"
    return out


class HybridHedger:
    """Values and hedges with the hybrid closed form, observing the short rate.

    With ``use_bond=True`` it holds ``dV/dS`` index units and ``dV/dP``
    zero-coupon bonds maturing with the contract, which neutralises both
    equity and interest-rate risk. With ``use_bond=False`` it holds only the
    index delta and keeps the rest in the money-market account, leaving the
    rate exposure open.
    """

    def __init__(self, contract: GMMBContract, model: HybridModel, use_bond: bool = True,
                 label: str | None = None):
        self.contract, self.model, self.use_bond = contract, model, use_bond
        self.label = label or ("hybrid delta, index + bond" if use_bond else "hybrid delta, index only")

    def _bond(self, t, rate):
        if rate is None:
            raise ValueError("hybrid hedger needs the short-rate path")
        return self.model.hw.zcb(t, self.contract.maturity, rate)

    def value(self, t, index, variance=None, rate=None):
        return gmmb_value(self.contract, self.model, index, t, self._bond(t, rate))

    def delta(self, t, index, variance=None, rate=None):
        return gmmb_deltas(self.contract, self.model, index, t, self._bond(t, rate))[0]

    def bond_units(self, t, index, variance=None, rate=None):
        d_s, d_p = gmmb_deltas(self.contract, self.model, index, t, self._bond(t, rate))
        return d_p if self.use_bond else np.zeros_like(d_p)
