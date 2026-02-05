"""Monte Carlo pricing of the GMMB with antithetic and control variates."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import blackscholes as bs
from .contract import GMMBContract
from .models import GBM, Heston, pair_means
from .rates.hybrid import HybridModel

METHODS = ("plain", "antithetic", "control", "antithetic_control")


@dataclass
class MCResult:
    estimate: float
    std_error: float
    n_paths: int
    method: str
    beta: np.ndarray | None = None

    @property
    def variance_per_path(self) -> float:
        """``n_paths * SE^2``: estimator variance normalised to one simulated
        path, so methods that use paths differently can be compared."""
        return self.n_paths * self.std_error**2

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = norm.ppf(0.5 + level / 2.0)
        return self.estimate - z * self.std_error, self.estimate + z * self.std_error


def control_variate(samples, controls, control_means):
    """Adjust ``samples`` with linear control variates.

    ``controls`` has shape (n, k) and known expectations ``control_means``.
    The coefficients are the least-squares (variance-minimising) betas
    estimated from the same sample. Returns ``(adjusted_samples, beta)``.
    """
    y = np.asarray(samples, dtype=float)
    x = np.asarray(controls, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    beta, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    adjusted = y - (x - np.asarray(control_means, dtype=float)) @ beta
    return adjusted, beta


def _summarise(samples, n_paths, method, beta=None) -> MCResult:
    samples = np.asarray(samples, dtype=float)
    se = samples.std(ddof=1) / np.sqrt(samples.size)
    return MCResult(float(samples.mean()), float(se), n_paths, method, beta)


def simulate_terminal(contract: GMMBContract, model, n_paths, rng=None, antithetic=False,
                      steps_per_year=50):
    """Terminal index levels and the terminal Brownian driver of the index."""
    if isinstance(model, GBM):
        return model.simulate_terminal(contract.premium, contract.rate, contract.maturity,
                                       n_paths, rng, antithetic)
    if isinstance(model, Heston):
        n_steps = max(1, int(round(steps_per_year * contract.maturity)))
        paths = model.simulate(contract.premium, contract.rate, contract.maturity, n_steps,
                               n_paths, rng, antithetic, store_paths=False)
        return paths.terminal, paths.brownian
    raise TypeError(f"unsupported model {type(model).__name__}")


def shadow_gbm_sigma(contract: GMMBContract, model) -> float:
    """Volatility of the GBM used as a control variate for a stochastic-vol model."""
    if isinstance(model, Heston):
        return float(np.sqrt(model.mean_variance(contract.maturity)))
    return model.sigma


def risk_neutral(model):
    """The pricing-measure version of a model with a real-world drift."""
    if isinstance(model, HybridModel):
        return model.risk_neutral()
    return replace(model, mu=None) if model.mu is not None else model


def price_gmmb(contract: GMMBContract, model, n_paths=100_000, method="plain", rng=None,
               steps_per_year=50) -> MCResult:
    """Risk-neutral Monte Carlo value of the cohort maturity guarantee.

    Methods
    -------
    plain
        Average of discounted payoffs.
    antithetic
        Mirror-image normal draws; the estimator averages each pair.
    control
        Linear control variates. Always uses the discounted terminal index
        (known mean ``S_0``). Under Heston it also uses the guarantee payoff on
        a "shadow" GBM path driven by the same Brownian motion, whose value is
        known in closed form.
    antithetic_control
        Both techniques: control variates applied to antithetic pair averages.

    For a :class:`~ghedge.rates.hybrid.HybridModel` each path is discounted
    with its own stochastic discount factor ``D(0, T)`` instead of the flat
    rate, and ``D(0, T)`` (mean ``P(0, T)``) is an extra control.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    model = risk_neutral(model)
    antithetic = method.startswith("antithetic")
    if isinstance(model, HybridModel):
        s_t, disc, w_t = model.simulate_terminal(contract.premium, contract.maturity, n_paths,
                                                 rng, antithetic)
    else:
        s_t, w_t = simulate_terminal(contract, model, n_paths, rng, antithetic, steps_per_year)
        disc = np.exp(-contract.rate * contract.maturity)
    y = disc * contract.payoff(s_t)

    controls, means = [disc * s_t], [contract.premium]
    if isinstance(model, HybridModel):
        controls.append(disc)
        means.append(float(model.hw.curve.discount(contract.maturity)))
    if isinstance(model, Heston):
        sig = shadow_gbm_sigma(contract, model)
        shadow = contract.premium * np.exp((contract.rate - 0.5 * sig**2) * contract.maturity + sig * w_t)
        controls.append(disc * contract.payoff(shadow))
        means.append(float(bs.gmmb_value(contract, sig)))
    x = np.column_stack(controls)

    if antithetic:
        y, x = pair_means(y), pair_means(x)
    beta = None
    if method.endswith("control"):
        y, beta = control_variate(y, x, means)
    return _summarise(y, n_paths, method, beta)


def convergence_study(contract: GMMBContract, model: GBM, sizes, n_reps=50, method="plain",
                      seed=0) -> pd.DataFrame:
    """Empirical RMSE against the closed form for increasing path counts.

    Each size is priced ``n_reps`` times with independent seeds; the RMSE of
    the estimates around the Black-Scholes value measures the true error.
    """
    exact = float(bs.gmmb_value(contract, model.sigma))
    seeds = np.random.SeedSequence(seed)
    rows = []
    for n in sizes:
        children = seeds.spawn(n_reps)
        est = np.array([price_gmmb(contract, model, n, method, np.random.default_rng(c)).estimate
                        for c in children])
        rows.append({"n_paths": n, "rmse": float(np.sqrt(np.mean((est - exact) ** 2))),
                     "mean_estimate": float(est.mean()), "exact": exact})
    return pd.DataFrame(rows)


def convergence_slope(df: pd.DataFrame) -> float:
    """Least-squares slope of log(RMSE) on log(N); about -0.5 for Monte Carlo."""
    return float(np.polyfit(np.log(df["n_paths"]), np.log(df["rmse"]), 1)[0])
