"""Equity index models: geometric Brownian motion and Heston stochastic volatility.

Both models simulate the *index* ``S`` (the tradable hedge instrument). The
policyholder fund is derived from it with the contract's fee drag
(``GMMBContract.fund``). Simulations return a :class:`Paths` object on a
uniform time grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Paths:
    """Simulated paths on a uniform grid.

    Attributes
    ----------
    times : (n_steps + 1,) array
    index : (n_paths, n_steps + 1) array of index levels
    variance : (n_paths, n_steps + 1) array of instantaneous variance, or None
    brownian : (n_paths,) terminal value of the Brownian motion driving the
        index (used for pathwise Greeks and control variates)
    short_rate, discount, bond : (n_paths, n_steps + 1) arrays, or None
        Stochastic-rate models only: the short rate, the discount factor
        ``D(0, t)`` and the price of a zero-coupon bond usable as a hedge.
        When ``discount`` is None, the backtester uses the contract's flat rate.
    """

    times: np.ndarray
    index: np.ndarray
    variance: np.ndarray | None
    brownian: np.ndarray
    short_rate: np.ndarray | None = None
    discount: np.ndarray | None = None
    bond: np.ndarray | None = None

    @property
    def n_paths(self) -> int:
        return self.index.shape[0]

    @property
    def terminal(self) -> np.ndarray:
        return self.index[:, -1]


def _normals(rng, n_paths, shape_tail, antithetic):
    """Standard normals with optional antithetic pairing.

    With ``antithetic=True`` rows ``i`` and ``i + n_paths // 2`` are mirror
    images, so pair averages can be formed with :func:`pair_means`.
    """
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic sampling")
        z = rng.standard_normal((n_paths // 2,) + shape_tail)
        return np.concatenate([z, -z], axis=0)
    return rng.standard_normal((n_paths,) + shape_tail)


def pair_means(x):
    """Average antithetic pairs produced by :func:`_normals`."""
    half = x.shape[0] // 2
    return 0.5 * (x[:half] + x[half:])


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian motion ``dS = mu S dt + sigma S dW``.

    ``mu=None`` means the risk-neutral drift (the flat rate passed to
    :meth:`simulate`).
    """

    sigma: float = 0.2
    mu: float | None = None

    def simulate(self, s0, rate, maturity, n_steps, n_paths, rng=None, antithetic=False):
        rng = np.random.default_rng(rng)
        drift = rate if self.mu is None else self.mu
        dt = maturity / n_steps
        z = _normals(rng, n_paths, (n_steps,), antithetic)
        increments = (drift - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
        times = np.linspace(0.0, maturity, n_steps + 1)
        brownian = np.sqrt(dt) * z.sum(axis=1)
        return Paths(times, s0 * np.exp(log_paths), None, brownian)

    def simulate_terminal(self, s0, rate, maturity, n_paths, rng=None, antithetic=False):
        """Exact one-step simulation of ``S_T`` (GBM needs no time grid)."""
        rng = np.random.default_rng(rng)
        drift = rate if self.mu is None else self.mu
        z = _normals(rng, n_paths, (), antithetic)
        w = np.sqrt(maturity) * z
        s_t = s0 * np.exp((drift - 0.5 * self.sigma**2) * maturity + self.sigma * w)
        return s_t, w


@dataclass(frozen=True)
class Heston:
    """Heston stochastic volatility model.

    ``dS = mu S dt + sqrt(v) S dW1``,
    ``dv = kappa (theta - v) dt + xi sqrt(v) dW2``, ``d<W1, W2> = rho dt``.

    Discretised with the full-truncation Euler scheme (Lord, Koekkoek and van
    Dijk, 2010): the variance may go negative on the grid but only its
    positive part ``v+`` enters drift and diffusion terms. The log-price is
    advanced with ``v+`` so the index stays positive.
    """

    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    mu: float | None = None

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``; values below 1 mean v can touch zero."""
        return 2.0 * self.kappa * self.theta / self.xi**2

    def mean_variance(self, maturity: float) -> float:
        """Expected average variance over ``[0, maturity]``."""
        if maturity <= 0:
            return self.v0
        k = self.kappa * maturity
        return self.theta + (self.v0 - self.theta) * (1.0 - np.exp(-k)) / k

    def simulate(self, s0, rate, maturity, n_steps, n_paths, rng=None, antithetic=False,
                 store_paths=True):
        rng = np.random.default_rng(rng)
        drift = rate if self.mu is None else self.mu
        dt = maturity / n_steps
        sqdt = np.sqrt(dt)
        rho_c = np.sqrt(1.0 - self.rho**2)
        log_s = np.full(n_paths, np.log(s0))
        v = np.full(n_paths, self.v0, dtype=float)
        w1 = np.zeros(n_paths)
        if store_paths:
            idx = np.empty((n_paths, n_steps + 1))
            var = np.empty((n_paths, n_steps + 1))
            idx[:, 0] = s0
            var[:, 0] = self.v0
        for k in range(n_steps):
            z = _normals(rng, n_paths, (2,), antithetic)
            z1 = z[:, 0]
            z2 = self.rho * z1 + rho_c * z[:, 1]
            v_pos = np.maximum(v, 0.0)
            sq_v = np.sqrt(v_pos)
            log_s += (drift - 0.5 * v_pos) * dt + sq_v * sqdt * z1
            v = v + self.kappa * (self.theta - v_pos) * dt + self.xi * sq_v * sqdt * z2
            w1 += sqdt * z1
            if store_paths:
                idx[:, k + 1] = np.exp(log_s)
                var[:, k + 1] = np.maximum(v, 0.0)
        times = np.linspace(0.0, maturity, n_steps + 1)
        if not store_paths:
            idx = np.stack([np.full(n_paths, float(s0)), np.exp(log_s)], axis=1)
            var = np.stack([np.full(n_paths, self.v0), np.maximum(v, 0.0)], axis=1)
            times = np.array([0.0, maturity])
        return Paths(times, idx, var, w1)
