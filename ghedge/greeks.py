"""Monte Carlo Greeks of the GMMB: pathwise and finite-difference estimators.

Finite differences use common random numbers (the same seed for the bumped
and unbumped valuations), so the per-path differences have low variance and
a standard error can be attached to each estimate.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contract import GMMBContract
from .models import GBM, Heston
from .montecarlo import MCResult, _summarise, simulate_terminal


def _risk_neutral(model):
    return replace(model, mu=None) if model.mu is not None else model


def _payoff_samples(contract, model, n_paths, seed, steps_per_year):
    s_t, w_t = simulate_terminal(contract, _risk_neutral(model), n_paths,
                                 np.random.default_rng(seed), False, steps_per_year)
    disc = np.exp(-contract.rate * contract.maturity)
    return disc * contract.payoff(s_t), s_t, w_t


def pathwise_delta(contract: GMMBContract, model, n_paths=100_000, seed=0,
                   steps_per_year=50) -> MCResult:
    """dV/dS0 by differentiating the payoff along each path.

    Both models are homogeneous in the initial index, so ``dS_T/dS_0 = S_T/S_0``
    and the derivative of ``max(G - F_T, 0)`` is ``-1{F_T < G} dF_T/dS_0``.
    """
    _, s_t, _ = _payoff_samples(contract, model, n_paths, seed, steps_per_year)
    disc = np.exp(-contract.rate * contract.maturity)
    f_t = contract.fund(s_t, contract.maturity)
    samples = -disc * contract.maturity_survival * (f_t < contract.guarantee) * f_t / contract.premium
    return _summarise(samples, n_paths, "pathwise")


def pathwise_vega(contract: GMMBContract, model: GBM, n_paths=100_000, seed=0) -> MCResult:
    """dV/dsigma under GBM: ``dS_T/dsigma = S_T (W_T - sigma T)``."""
    if not isinstance(model, GBM):
        raise TypeError("pathwise vega is implemented for GBM only")
    _, s_t, w_t = _payoff_samples(contract, model, n_paths, seed, 50)
    disc = np.exp(-contract.rate * contract.maturity)
    f_t = contract.fund(s_t, contract.maturity)
    df_dsigma = f_t * (w_t - model.sigma * contract.maturity)
    samples = -disc * contract.maturity_survival * (f_t < contract.guarantee) * df_dsigma
    return _summarise(samples, n_paths, "pathwise")


def fd_delta(contract: GMMBContract, model, n_paths=100_000, seed=0, bump=0.01,
             steps_per_year=50) -> MCResult:
    """Central finite-difference delta with common random numbers."""
    h = bump * contract.premium
    up, _, _ = _payoff_samples(contract.with_(premium=contract.premium + h), model, n_paths, seed, steps_per_year)
    dn, _, _ = _payoff_samples(contract.with_(premium=contract.premium - h), model, n_paths, seed, steps_per_year)
    return _summarise((up - dn) / (2.0 * h), n_paths, "finite_difference")


def fd_gamma(contract: GMMBContract, model, n_paths=100_000, seed=0, bump=0.02,
             steps_per_year=50) -> MCResult:
    """Central second difference. The payoff's kink makes pathwise gamma
    unavailable, so this is the Monte Carlo gamma estimator offered."""
    h = bump * contract.premium
    up, _, _ = _payoff_samples(contract.with_(premium=contract.premium + h), model, n_paths, seed, steps_per_year)
    mid, _, _ = _payoff_samples(contract, model, n_paths, seed, steps_per_year)
    dn, _, _ = _payoff_samples(contract.with_(premium=contract.premium - h), model, n_paths, seed, steps_per_year)
    return _summarise((up - 2.0 * mid + dn) / h**2, n_paths, "finite_difference")


def fd_vega(contract: GMMBContract, model, n_paths=100_000, seed=0, bump=0.005,
            steps_per_year=50) -> MCResult:
    """Central difference in volatility: ``sigma`` for GBM, ``v0`` for Heston
    (reported per unit of ``v0``)."""
    if isinstance(model, GBM):
        up_m, dn_m, width = (replace(model, sigma=model.sigma + bump),
                             replace(model, sigma=model.sigma - bump), 2.0 * bump)
    elif isinstance(model, Heston):
        b = bump * model.v0 / 0.04
        up_m, dn_m, width = replace(model, v0=model.v0 + b), replace(model, v0=model.v0 - b), 2.0 * b
    else:
        raise TypeError(f"unsupported model {type(model).__name__}")
    up, _, _ = _payoff_samples(contract, up_m, n_paths, seed, steps_per_year)
    dn, _, _ = _payoff_samples(contract, dn_m, n_paths, seed, steps_per_year)
    return _summarise((up - dn) / width, n_paths, "finite_difference")
