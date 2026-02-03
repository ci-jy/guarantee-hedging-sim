"""One-factor Hull-White short-rate model fitted to an initial zero curve.

Risk-neutral dynamics ``dr = (theta(t) - a r) dt + sigma dW``. Writing
``r(t) = x(t) + alpha(t)`` with the Ornstein-Uhlenbeck factor
``dx = -a x dt + sigma dW``, ``x(0) = 0`` and

    alpha(t) = f(0, t) + sigma^2 / (2 a^2) (1 - exp(-a t))^2

makes the model reproduce the input discount curve exactly for any
``(a, sigma)``; equivalently ``theta(t) = f_t(0, t) + a f(0, t) + sigma^2 /
(2a) (1 - exp(-2 a t))``.

Simulation is exact: ``x`` and its time integral are jointly Gaussian over
any step, so the short rate and the stochastic discount factor
``D(0, t) = exp(-int_0^t r ds)`` are sampled without discretisation bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from ..models import _normals
from .curve import ZeroCurve


def b_factor(a: float, tau):
    """``B(tau) = (1 - exp(-a tau)) / a``: bond-price sensitivity to ``r``."""
    tau = np.asarray(tau, dtype=float)
    return -np.expm1(-a * tau) / a


def cov_root(cov) -> np.ndarray:
    """Matrix square root ``L`` with ``L L^T = cov`` that also works for
    singular covariances (zero volatility, perfect correlation)."""
    w, v = np.linalg.eigh(np.asarray(cov, dtype=float))
    return v * np.sqrt(np.clip(w, 0.0, None))


def ou_step_cov(a: float, sigma: float, dt: float) -> np.ndarray:
    """Covariance of the Gaussian innovations of ``(x, int x ds)`` over a step
    of length ``dt`` (given the value of ``x`` at the start of the step)."""
    e1, e2 = np.exp(-a * dt), np.exp(-2 * a * dt)
    var_x = sigma**2 * (1 - e2) / (2 * a)
    cov = sigma**2 * (1 - e1) ** 2 / (2 * a**2)
    var_i = sigma**2 / a**2 * (dt - 2 * (1 - e1) / a + (1 - e2) / (2 * a))
    return np.array([[var_x, cov], [cov, var_i]])


@dataclass
class RatePaths:
    """Simulated short rate and discount factor on a time grid.

    ``short_rate`` and ``discount`` have shape (n_paths, n_steps + 1);
    ``discount[:, j] = exp(-int_0^{t_j} r ds)``.
    """

    times: np.ndarray
    short_rate: np.ndarray
    discount: np.ndarray


@dataclass(frozen=True)
class HullWhite:
    """Hull-White model with mean reversion ``a`` and volatility ``sigma``,
    fitted to ``curve``."""

    curve: ZeroCurve
    a: float = 0.05
    sigma: float = 0.01

    def __post_init__(self):
        if self.a <= 0 or self.sigma < 0:
            raise ValueError("need a > 0 and sigma >= 0")

    # --- curve fit -----------------------------------------------------------
    def alpha(self, t):
        t = np.asarray(t, dtype=float)
        return self.curve.forward(t) + self.sigma**2 / 2 * b_factor(self.a, t) ** 2

    def theta(self, t):
        """Drift function ``theta(t)`` that fits the initial curve."""
        t = np.asarray(t, dtype=float)
        f = self.curve.forward(t)
        return (self.curve.forward_slope(t) + self.a * f
                + self.sigma**2 / (2 * self.a) * (1 - np.exp(-2 * self.a * t)))

    @property
    def r0(self) -> float:
        return float(self.curve.forward(0.0))

    def integral_variance(self, t):
        """``Var(int_0^t x ds)``."""
        t = np.asarray(t, dtype=float)
        a, s = self.a, self.sigma
        return s**2 / a**2 * (t - 2 * b_factor(a, t) + b_factor(2 * a, t))

    # --- closed forms --------------------------------------------------------
    def zcb(self, t, maturity, r):
        """``P(t, T)`` given the short rate ``r`` at ``t`` (affine formula)."""
        t = np.asarray(t, dtype=float)
        b = b_factor(self.a, np.asarray(maturity, dtype=float) - t)
        p0 = self.curve.discount
        log_a = (np.log(p0(maturity) / p0(t)) + b * self.curve.forward(t)
                 - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * b**2)
        return np.exp(log_a - b * np.asarray(r, dtype=float))

    def zcb_vol(self, expiry, maturity):
        """Total standard deviation of ``ln P(T, S)`` seen from 0, ``sigma_P``."""
        return self.sigma * b_factor(self.a, np.asarray(maturity, dtype=float) - expiry) \
            * np.sqrt(b_factor(2 * self.a, expiry))

    def zcb_option(self, expiry, maturity, strike, kind: str = "call"):
        """Time-0 price of a European option expiring at ``expiry`` on the
        zero-coupon bond maturing at ``maturity`` (Hull-White closed form)."""
        p_t = self.curve.discount(expiry)
        p_s = self.curve.discount(maturity)
        sp = self.zcb_vol(expiry, maturity)
        strike = np.asarray(strike, dtype=float)
        h = np.log(p_s / (strike * p_t)) / sp + sp / 2
        if kind == "call":
            return p_s * norm.cdf(h) - strike * p_t * norm.cdf(h - sp)
        if kind == "put":
            return strike * p_t * norm.cdf(sp - h) - p_s * norm.cdf(-h)
        raise ValueError("kind must be 'call' or 'put'")

    def coupon_bond_option(self, expiry, times, amounts, strike, kind: str = "call") -> float:
        """European option on a coupon bond (cash flows ``amounts`` at
        ``times > expiry``) by Jamshidian's decomposition: find the short rate
        ``r*`` at which the bond is worth the strike, then sum options on each
        cash flow struck at its zero-coupon price under ``r*``."""
        times = np.asarray(times, dtype=float)
        amounts = np.asarray(amounts, dtype=float)
        if np.any(times <= expiry):
            raise ValueError("cash flows must fall after the option expiry")
        value = lambda r: float(amounts @ self.zcb(expiry, times, r)) - strike
        lo, hi = -1.0, 1.0
        while value(lo) < 0:
            lo *= 2
        while value(hi) > 0:
            hi *= 2
        r_star = brentq(value, lo, hi, xtol=1e-14)
        strikes = self.zcb(expiry, times, r_star)
        return float(np.sum(amounts * self.zcb_option(expiry, times, strikes, kind)))

    # --- simulation ----------------------------------------------------------
    def simulate(self, horizon: float, n_steps: int, n_paths: int, rng=None,
                 antithetic: bool = False) -> RatePaths:
        """Exact joint simulation of ``r(t)`` and ``D(0, t)`` on a uniform grid."""
        rng = np.random.default_rng(rng)
        times = np.linspace(0.0, horizon, n_steps + 1)
        dt = horizon / n_steps
        chol = cov_root(ou_step_cov(self.a, self.sigma, dt))
        z = _normals(rng, n_paths, (n_steps, 2), antithetic) @ chol.T
        decay = np.exp(-self.a * dt)
        bdt = float(b_factor(self.a, dt))
        x = np.zeros((n_paths, n_steps + 1))
        y = np.zeros((n_paths, n_steps + 1))
        for k in range(n_steps):
            x[:, k + 1] = x[:, k] * decay + z[:, k, 0]
            y[:, k + 1] = y[:, k] + x[:, k] * bdt + z[:, k, 1]
        return self._paths_from_factor(times, x, y)

    def _paths_from_factor(self, times, x, y) -> RatePaths:
        r = x + self.alpha(times)
        disc = self.curve.discount(times) * np.exp(-0.5 * self.integral_variance(times) - y)
        return RatePaths(times, r, disc)


def mc_zcb(model: HullWhite, maturities, n_paths: int = 100_000, rng=None):
    """Monte Carlo zero-coupon bond prices ``E[D(0, T)]`` with standard errors.

    The discount factor is sampled exactly at each maturity (one Gaussian
    step per maturity), so any difference from the curve is sampling error.
    Returns ``(estimates, std_errors)``.
    """
    rng = np.random.default_rng(rng)
    est, se = [], []
    for t in np.atleast_1d(np.asarray(maturities, dtype=float)):
        d = model.simulate(float(t), 1, n_paths, rng).discount[:, -1]
        est.append(d.mean())
        se.append(d.std(ddof=1) / np.sqrt(n_paths))
    return np.array(est), np.array(se)


def mc_bond_option(model: HullWhite, expiry: float, times, amounts, strike: float,
                   kind: str = "call", n_paths: int = 200_000, rng=None, control: bool = False):
    """Monte Carlo price of a European option on a coupon bond (or a
    zero-coupon bond: one cash flow of 1).

    ``r(expiry)`` and ``D(0, expiry)`` are sampled exactly; the bond price at
    expiry uses the affine formula. With ``control=True`` the discount factor
    and the discounted bond price, whose expectations ``P(0, T)`` and the
    bond's present value are known from the curve, are linear control
    variates. Returns a :class:`ghedge.montecarlo.MCResult`.
    """
    from ..montecarlo import _summarise, control_variate

    times = np.asarray(times, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    paths = model.simulate(float(expiry), 1, n_paths, rng)
    r_t, disc = paths.short_rate[:, -1], paths.discount[:, -1]
    bond = model.zcb(expiry, times[None, :], r_t[:, None]) @ amounts
    if kind == "call":
        payoff = np.maximum(bond - strike, 0.0)
    elif kind == "put":
        payoff = np.maximum(strike - bond, 0.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    y = disc * payoff
    beta = None
    if control:
        p0 = model.curve.discount
        means = [float(p0(expiry)), float(amounts @ p0(times))]
        y, beta = control_variate(y, np.column_stack([disc, disc * bond]), means)
    return _summarise(y, n_paths, "control" if control else "plain", beta)


def fit_to_yield_volatility(maturities, vols, a_bounds=(1e-3, 2.0)):
    """Least-squares ``(a, sigma)`` matching the model's zero-yield volatility
    term structure ``sigma B(tau) / tau`` to observed yield volatilities.

    This is a historical (real-world) volatility fit, used because swaption
    volatilities are not freely available."""
    tau = np.asarray(maturities, dtype=float)
    vols = np.asarray(vols, dtype=float)
    resid = lambda p: p[1] * b_factor(p[0], tau) / tau - vols
    sol = least_squares(resid, [0.1, float(np.mean(vols))],
                        bounds=([a_bounds[0], 1e-6], [a_bounds[1], 1.0]))
    return float(sol.x[0]), float(sol.x[1])


def yield_volatility(history, min_maturity: float = 2.0, start=None, freq: str = "W-FRI"):
    """Annualised standard deviation of weekly yield changes per tenor, from
    a CMT history frame (see :func:`ghedge.data.load_treasury`)."""
    from ..data import series_maturity

    h = history.dropna()
    if start is not None:
        h = h.loc[start:]
    changes = h.resample(freq).last().diff().dropna()
    vols = changes.std() * np.sqrt(52)
    vols.index = [series_maturity(c) for c in vols.index]
    vols.index.name = "maturity"
    return vols[vols.index >= min_maturity]
