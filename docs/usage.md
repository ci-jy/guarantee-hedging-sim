# Using ghedge from Python

Python 3.10+. On Debian/Ubuntu the system packages work instead of pip:

```bash
sudo apt-get install python3-numpy python3-scipy python3-pandas python3-matplotlib python3-pytest jupyter-nbconvert python3-ipykernel python3-nbformat
```

The report notebook is generated from `notebooks/build_report.py`; regenerate it with `python3 notebooks/build_report.py`.

## Flat rates: pricing and a rebalancing study

```python
from ghedge import GBM, Heston, GMMBContract
from ghedge import blackscholes as bs
from ghedge.montecarlo import price_gmmb
from ghedge.hedging import BSHedger, rebalancing_table

c = GMMBContract(guarantee=100, maturity=10, fee=0.02, rate=0.03)
print(bs.gmmb_value(c, 0.2))                                    # closed form
res = price_gmmb(c, GBM(0.2), 200_000, "antithetic_control", rng=1)
print(res.estimate, res.std_error, res.ci())

c5 = c.with_(maturity=5)
paths = GBM(0.2, mu=0.07).simulate(100, c5.rate, 5, 5 * 252, 2_000, rng=2)
print(rebalancing_table(paths, c5, [BSHedger(c5, 0.2)], cost_rates=(0, 0.001)))
```

## Stochastic rates

```python
from ghedge.data import load_treasury, par_curve
from ghedge.rates import HullWhite, HybridModel, HybridHedger, bootstrap_series, hybrid
from ghedge.hedging import backtest

curve = bootstrap_series(par_curve(load_treasury()), "monotone_convex")
hw = HullWhite(curve, a=0.05, sigma=0.01)
print(hw.coupon_bond_option(5.0, [6.0, 7.0, 8.0], [0.05, 0.05, 1.05], 1.0, "put"))  # Jamshidian

model = HybridModel(hw, sigma=0.2, rho=0.0)
print(hybrid.gmmb_value(c, model), hybrid.deterministic_rate_value(c, model))
print(price_gmmb(c, model, 200_000, "antithetic_control", rng=1).estimate)

paths = HybridModel(hw, 0.2, equity_premium=0.04).simulate(100, 10, 520, 2_000, rng=3)
print(backtest(paths, c, HybridHedger(c, model, use_bond=True)).summary())
```

## Hedgers

- `BSHedger(contract, sigma)`: Black-Scholes delta at a fixed volatility. `sigma` may be one value per path, as in the historical backtest.
- `HestonHedger(contract, params)`: true-model delta, observing the current variance `v_t`.
- `HestonHedger(..., minimum_variance=True)`: `dV/dS + rho xi / S * dV/dv`, the index position that minimises the instantaneous hedge variance.
- `HybridHedger(contract, model, use_bond=...)`: hybrid-model delta, optionally with the zero-coupon bond maturing with the guarantee.

## Where things live

| path | contents |
|---|---|
| `ghedge/contract.py` | `GMMBContract` (premium, guarantee, maturity, fee, rate) and `Decrements` (constant lapse force, Gompertz-Makeham mortality) |
| `ghedge/models.py` | `GBM` and `Heston` path simulators (full-truncation Euler for Heston), antithetic sampling |
| `ghedge/blackscholes.py` | closed-form put with continuous yield, delta/gamma/vega, implied vol, GMMB wrappers |
| `ghedge/heston.py` | semi-analytic Heston put price, delta and variance sensitivity, vectorised over states |
| `ghedge/montecarlo.py` | Monte Carlo pricer, control-variate regression, confidence intervals, convergence study |
| `ghedge/greeks.py` | pathwise delta/vega and finite-difference delta/gamma/vega with common random numbers |
| `ghedge/hedging.py` | hedgers, discrete hedging backtester, VaR/CVaR |
| `ghedge/data.py` | FRED download and CSV loading with a committed fallback snapshot, par-curve selection |
| `ghedge/historical.py` | rolling historical hedge backtest |
| `ghedge/rates/curve.py` | `ZeroCurve` (monotone-convex and cubic interpolation), par-bond bootstrap, repricing errors |
| `ghedge/rates/hullwhite.py` | `HullWhite`: fitted `theta(t)`, affine bond prices, zero-coupon and Jamshidian coupon-bond options, exact simulation, Monte Carlo pricers, historical volatility fit |
| `ghedge/rates/hybrid.py` | `HybridModel` (GBM index + correlated Hull-White), forward-measure closed form, hedge ratios, sensitivity grid, `HybridHedger` |
| `scripts/fetch_index.py`, `scripts/fetch_treasury.py` | refresh the FRED snapshots |
| `notebooks/build_report.py` | source of the report notebook |
