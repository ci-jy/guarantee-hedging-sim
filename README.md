# guarantee-hedging-sim

Monte Carlo pricing and dynamic hedging of variable annuity maturity
guarantees, with constant and Hull-White stochastic interest rates.

A segregated fund or variable annuity with a guaranteed minimum maturity
benefit (GMMB) promises the policyholder at least `G` at maturity, whatever
the fund is worth. The insurer is therefore short a put on the policyholder's
fund and usually hedges it dynamically. This project:

* prices the guarantee by Monte Carlo (plain, antithetic, control variates)
  and checks the price against the closed-form Black-Scholes value;
* computes Greeks by pathwise and finite-difference methods and cross-checks them;
* backtests discrete delta hedging with proportional transaction costs, with
  daily, weekly and monthly rebalancing, and reports the hedge P&L
  distribution, its standard deviation, VaR and CVaR;
* measures model risk: Black-Scholes hedging while the market follows Heston
  stochastic volatility, compared with hedging under the true model;
* hedges overlapping one-year contracts on historical S&P 500 paths;
* bootstraps a zero curve from U.S. Treasury par yields (monotone-convex or
  cubic interpolation) and fits a one-factor Hull-White short-rate model to it
  exactly, validated against closed-form bond and bond-option prices;
* prices the guarantee in a hybrid model (GBM index, correlated Hull-White
  rates) by Monte Carlo and in closed form, measures its sensitivity to rate
  volatility and equity-rate correlation and the error from assuming
  deterministic rates;
* hedges the guarantee with the index plus a zero-coupon bond and compares
  that with index-only delta hedging.

The full write-up with charts and tables is the notebook
[`notebooks/report.ipynb`](notebooks/report.ipynb), exported as
[`docs/report.html`](docs/report.html).

## Quick start

```bash
# Python 3.10+; on Debian/Ubuntu the system packages work:
sudo apt-get install python3-numpy python3-scipy python3-pandas python3-matplotlib \
    python3-pytest jupyter-nbconvert python3-ipykernel python3-nbformat
# or, in a virtual environment:
python3 -m pip install -r requirements.txt     # or: pip install -e ".[test,notebook]"

python3 -m pytest -q                                   # test suite (~10 s)
jupyter nbconvert --to html --execute notebooks/report.ipynb --output-dir /tmp   # report (~30 s)
python3 scripts/fetch_index.py                         # refresh the S&P 500 snapshot from FRED
python3 scripts/fetch_treasury.py                      # refresh the Treasury yield snapshot from FRED
python3 notebooks/build_report.py                      # regenerate the notebook from its source
```

Example:

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

Stochastic rates:

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

## Layout

| path | contents |
|---|---|
| `ghedge/contract.py` | `GMMBContract` (premium, guarantee, maturity, fee, rate) and `Decrements` (constant lapse force, Gompertz-Makeham mortality) |
| `ghedge/models.py` | `GBM` and `Heston` path simulators (full-truncation Euler for Heston), antithetic sampling |
| `ghedge/blackscholes.py` | closed-form put with continuous yield, delta/gamma/vega, implied vol, GMMB wrappers |
| `ghedge/heston.py` | semi-analytic Heston put price, delta and variance sensitivity, vectorised over states |
| `ghedge/montecarlo.py` | Monte Carlo pricer, control-variate regression, confidence intervals, convergence study |
| `ghedge/greeks.py` | pathwise delta/vega and finite-difference delta/gamma/vega with common random numbers |
| `ghedge/hedging.py` | hedgers (Black-Scholes, Heston, Heston minimum-variance), discrete hedging backtester, VaR/CVaR |
| `ghedge/data.py`, `scripts/fetch_index.py` | FRED download and CSV loading with a committed fallback snapshot |
| `ghedge/historical.py` | rolling historical hedge backtest |
| `ghedge/rates/curve.py` | `ZeroCurve` (monotone-convex and cubic interpolation), par-bond bootstrap, repricing errors |
| `ghedge/rates/hullwhite.py` | `HullWhite`: fitted `theta(t)`, affine bond prices, zero-coupon and Jamshidian coupon-bond options, exact simulation, Monte Carlo bond and option pricers, historical volatility fit |
| `ghedge/rates/hybrid.py` | `HybridModel` (GBM index + correlated Hull-White), forward-measure closed form, hedge ratios, sensitivity grid, `HybridHedger` |
| `ghedge/data.py`, `scripts/fetch_treasury.py` | Treasury constant-maturity yields from FRED, par-curve selection |
| `data/sp500_fred.csv` | S&P 500 daily closes from FRED (series `SP500`, 2016-10-03 to 2026-10-02) |
| `data/treasury_cmt_fred.csv` | Treasury CMT yields from FRED (`DGS1MO` ... `DGS30`, 2016-10-03 to 2026-10-01) |
| `notebooks/build_report.py` | source of the report notebook |
| `tests/` | pytest suite, including statistical checks |

## Model assumptions

* **Fund.** The fund tracks an equity index `S` less a continuous management
  fee `m`: `F_t = S_t exp(-m t)`, with `S_0 = F_0 = premium`. The index is the
  hedge instrument and pays no dividends.
* **Guarantee.** Survivors at maturity receive `max(F_T, G)`. The insurer's
  claim is `(G - F_T)^+` per surviving policy. Only the maturity benefit is
  modelled: no death benefit, ratchets or resets.
* **Decrements.** Lapses at a constant force (3% a year by default) and
  Gompertz-Makeham mortality `mu(x) = A + B c^x` (A = 0.00022, B = 2.7e-6,
  c = 1.124, issue age 55). Decrements are deterministic and independent of
  the market, so the cohort liability is `_T p_x` times one put. Policyholders
  who leave early take their fund value and the guarantee lapses with them.
* **Rates.** Sections 1-8 of the report use one flat continuously compounded
  rate (`GMMBContract.rate`). The stochastic-rate extension uses a one-factor
  Hull-White model `dr = (theta(t) - a r) dt + sigma_r dW_r` fitted to the
  bootstrapped Treasury curve, with `a` and `sigma_r` fitted to historical
  yield volatilities (no swaption data). Rates are simulated under the
  risk-neutral measure only; the real-world equity drift is the short rate
  plus a constant premium.
* **Hybrid model.** `dS/S = r dt + sigma_S dW_S`, `d<W_S, W_r> = rho dt`,
  constant `sigma_S` and `rho`. The fund is still `S_t exp(-m t)`.
* **Black-Scholes.** The fund is a stock with dividend yield `m`, so the
  guarantee value is the textbook put with yield, scaled by `_T p_x`.
* **Heston.** `dv = kappa (theta - v) dt + xi sqrt(v) dW2`, `corr(dW1, dW2) = rho`.
  Default parameters `v0 = theta = 0.04`, `kappa = 1.5`, `xi = 0.5`,
  `rho = -0.7` (the Feller condition fails, so the variance does reach zero).
  Real-world and risk-neutral variance dynamics are the same (no variance risk
  premium); the real-world equity drift is set separately (`mu`).

## Methods

**Monte Carlo pricing.** GBM terminal values are simulated exactly in one
step. Heston paths use the full-truncation Euler scheme (Lord, Koekkoek and
van Dijk, 2010) on the log-price, which keeps the discounted index an exact
martingale on the grid. The discretisation bias of the Heston price is
first order in the step size; it is visible against the semi-analytic
price at 12 steps a year and negligible at 250.

* *Antithetic variates* pair each normal vector with its negative; the
  standard error is computed from the pair averages.
* *Control variates* use the discounted terminal index (known mean `S_0`). For
  Heston a second control is the guarantee payoff on a "shadow" GBM path
  driven by the same Brownian motion, with volatility equal to the square root
  of the expected average variance; its expectation is the closed-form price.
  Coefficients are the least-squares betas estimated on the same sample (the
  resulting bias is O(1/N)).
* Results carry a standard error, normal-approximation confidence intervals
  and `variance_per_path = N * SE^2`, which compares methods at equal cost.

**Semi-analytic Heston.** Gil-Pelaez inversion with the "little Heston trap"
characteristic function (Albrecher et al., 2007). The integral is mapped from
`(0, inf)` to `(0, 1)` and evaluated with 192-point Gauss-Legendre
quadrature on a grid that depends only on time to maturity. Thousands of
(index, variance) states at the same date are therefore priced in one matrix
product, which makes Heston hedging backtests feasible. Delta is
`exp(-q tau) (P1 - 1)`, and `dV/dv` comes from differentiating the
characteristic function (`D(u) phi(u)`).

**Greeks.** Both models are homogeneous in `S_0`, so the pathwise delta is
`-1{F_T < G} F_T / S_0` (discounted, times `_T p_x`). The GBM pathwise vega uses
`dF_T/dsigma = F_T (W_T - sigma T)`. Finite differences are central bumps that
re-simulate with the same seed (common random numbers). The payoff has a kink,
so there is no pathwise gamma; gamma is by finite difference only.

**Hedging backtester.** The insurer receives a price for the guarantee at
issue (by default the hedger's own model value), holds `dV/dS` units of the
index and keeps the rest in cash at the flat rate. It rebalances every `k` grid
steps, paying `cost_rate * |trade| * S` per trade, including the final unwind.
The hedge P&L is the discounted value of the final portfolio minus the
discounted claim, per 100 of premium. Reported statistics: mean, standard
deviation, 95% VaR and CVaR (mean of the worst 5% of losses).

Hedgers:

* `BSHedger(contract, sigma)`: Black-Scholes delta at a fixed volatility.
  `sigma` may be one value per path, as in the historical backtest.
* `HestonHedger(contract, params)`: true-model delta, observing the current
  variance `v_t`.
* `HestonHedger(..., minimum_variance=True)`: `dV/dS + rho xi / S * dV/dv`,
  the index position that minimises the instantaneous hedge variance.

**Curve bootstrap.** CMT yields are par yields on a semi-annual
bond-equivalent basis. Maturities up to six months are zero-coupon
instruments; longer ones are semi-annual par bonds (no accrued interest or
day counts). The curve interpolates `-ln P(0, t)` with either the
monotone-convex scheme of Hagan and West (2006), whose forward curve is
continuous and local, or a natural cubic spline; the forward is held flat
beyond 30 years. Because both schemes couple neighbouring segments, all knot
zero rates are solved at once with SciPy's Powell hybrid root finder until
every instrument prices to par (typically to 1e-16).

**Hull-White.** With `r = x + alpha(t)`, `dx = -a x dt + sigma_r dW`,
`alpha(t) = f(0,t) + sigma_r^2 B(t)^2 / 2`, the model reproduces `P(0, t)` for
any parameters. `x` and its integral are jointly Gaussian over a step, so the
short rate and `D(0, t) = exp(-int r)` are simulated exactly (no
discretisation error). Closed forms: affine bond prices `P(t, T) = A e^{-B r}`,
the Hull-White zero-coupon bond option, and coupon-bond options by
Jamshidian's decomposition. The Monte Carlo bond-option pricer can use `D(0,T)`
and the discounted bond price as control variates. `(a, sigma_r)` are fitted
by least squares to the model's yield-volatility curve `sigma_r B(tau)/tau`
against the annualised standard deviation of weekly 2-30 year yield changes
over the last three years.

**Hybrid closed form.** Under the `T`-forward measure, `S_t / P(t, T)` is
lognormal with total variance
`v(T) = sigma_S^2 T + 2 rho sigma_S sigma_r int B + sigma_r^2 int B^2`, so the
guarantee is the Black-Scholes put at the zero rate `-ln P(0,T)/T` with
volatility `sqrt(v(T)/T)`. Monte Carlo simulates `(x, int x, W_S)` exactly,
discounts each path with its own `D(0, T)`, and uses `D(0,T) S_T` and `D(0,T)`
as controls (`price_gmmb` accepts a `HybridModel`).

**Index + bond hedge.** The value `V(t, S, P)` is homogeneous of degree one in
the index and the price `P(t, T)` of the zero-coupon bond maturing with the
guarantee, so holding `dV/dS` index units and `dV/dP` bonds replicates it.
The backtester accrues cash at the simulated short rate, discounts each path
with `D(0, t)`, and trades the bond when the hedger has a `bond_units` method
(bond trades cost `bond_cost_rate`, by default the same as index trades).
With flat-rate paths it behaves exactly as before.

**Historical backtest.** One-year contracts are issued every 21 trading days
along the S&P 500 series. Each window is rescaled to start at 100, priced and
hedged with Black-Scholes at the trailing one-year realised volatility at
issue, with a 2% rate and 10 bp costs.

## Results

All numbers come from fixed seeds and are reproduced by the notebook
(`docs/report.html`).

* **Price accuracy.** On a grid of guarantee levels 80/100/120 and maturities
  1/5/10/20 years (200,000 paths each), every plain and antithetic+control
  estimate is within 3 standard errors of the closed form (largest |z| = 1.72).
  Asserted in `tests/test_pricing.py`.
* **Variance reduction** (10-year ATM guarantee, GBM, variance per path
  relative to plain Monte Carlo): control variate 1.8x, antithetic 3.7x, both
  10.5x. Under Heston (5-year), the two controls give 3.9x, and with
  antithetic sampling 7.4x. The tests assert lower bounds of 1.5x, 2.5x, 6x
  and 2x.
* **Convergence.** RMSE against the closed form, over 60 replications at each
  N from 500 to 64,000, falls with a log-log slope close to -0.5. The test
  requires a slope in (-0.6, -0.4).
* **Greeks.** Pathwise and finite-difference deltas and vegas agree within
  their standard errors and with the closed form. Under Heston the pathwise,
  finite-difference and semi-analytic deltas also agree.
* **Rebalancing frequency** (5-year guarantee, GBM, 2,000 paths, no costs).
  Hedge-error standard deviation is 1.40 monthly, 0.71 weekly and 0.32 daily
  (per 100 premium, against a price of 11.48). The monthly/daily ratio is 4.4,
  close to sqrt(21) = 4.6. Costs of 10-20 bp per trade reduce mean P&L by
  0.86-1.73 for daily rebalancing and 0.23-0.45 for monthly.
* **Model risk** (same contract, Heston market, weekly rebalancing). The
  hedge-error standard deviation is 3.28 for Black-Scholes at the implied
  volatility, 3.86 for the Heston delta and 2.91 for the Heston
  minimum-variance delta, against 0.71 for Black-Scholes hedging in a
  Black-Scholes market. Delta hedging alone cannot remove variance risk, even
  with the true model. The minimum-variance delta lowers the spread but gives
  up more equity risk premium (mean P&L -1.16 under a 7% drift). The
  Black-Scholes hedge P&L has a correlation of -0.74 with realised volatility.
* **Historical** (96 overlapping one-year contracts, 2017-2025 issues). The
  standard deviation of hedge error is 3.3-4.2 per 100 premium, and the
  weekly P&L has a correlation of -0.92 with the gap between realised and
  issue-date volatility. Rebalancing more often did not reduce the error on
  this sample.

* **Stochastic rates** (curve of 2026-10-01, fitted `a = 0.0127`,
  `sigma_r = 0.94%`, `sigma_S = 20%`). Both interpolations reprice all 11 par
  instruments to about 1e-16. Monte Carlo zero-coupon bond prices at 1-30
  years are within 3 SE of the curve (largest |z| = 2.24), and 12 zero-coupon
  and coupon bond-option prices are within 3 SE of the closed forms (largest
  |z| 1.84 plain, 1.45 with controls; the controls cut the SE 1.3-3.4x).
  Hybrid Monte Carlo agrees with the closed form for 10 and 20-year
  guarantees at `rho` = -0.3, 0, 0.3 (largest |z| = 1.55).
* **Value effect.** At `rho = 0` the 10-year guarantee is worth 6.35 with
  stochastic rates against 6.00 with deterministic rates (-5.5% error), and
  the 20-year guarantee 2.69 against 2.10 (-22%). With `rho = +0.3` the
  deterministic error reaches -15% at 10 years and -36% at 20 years; with
  `rho = -0.3` deterministic rates overstate the value by 1.8-6.5%. The flat 3%
  rate of the earlier sections, below the current curve, overstates the
  value by far more (11.23 at 10 years).
* **Hedging with stochastic rates** (10-year guarantee, 2,000 paths, weekly
  grid, no costs). Index-only hedges leave a hedge-error standard deviation of
  2.78 (Black-Scholes at deterministic rates) and 3.10 (hybrid delta),
  against 0.41 for the same Black-Scholes hedge when rates are deterministic.
  Adding the zero-coupon bond brings it back to 0.41 weekly, 0.79 monthly and
  1.45 quarterly, about 7.5x lower than the index-only hybrid hedge.

## Tests

`python3 -m pytest -q` runs 136 tests in about 10 seconds. The statistical
tests use fixed seeds and check:

* the Monte Carlo price is within 3 SE of the closed form on the 3 x 4 grid of
  guarantees and maturities, for plain and antithetic+control estimators;
* 95% confidence intervals cover the closed form in 91.5-98.5% of 400
  independent runs;
* the variance-reduction factors stay above their thresholds;
* the convergence slope is within (-0.6, -0.4);
* pathwise and finite-difference Greeks agree with each other (within 3
  combined SE) and with closed-form Greeks;
* the Heston Monte Carlo price matches the semi-analytic price within 3 SE;
  semi-analytic Heston reduces to Black-Scholes as `xi -> 0`; Heston delta and
  `dV/dv` match finite differences;
* hedge-error standard deviation falls from monthly to weekly to daily
  rebalancing under Black-Scholes, with a monthly/daily ratio near sqrt(21);
* the zero-cost hedge is unbiased; transaction costs equal the P&L
  difference exactly; single-period accounting matches a hand calculation;
* Black-Scholes hedging in a Heston market has more than twice the error of
  the same hedge in a GBM market, and the minimum-variance delta improves on it;
* historical CSV parsing (FRED `.` markers), window alignment and the
  backtest on the snapshot;
* the bootstrapped curve reprices the Treasury par bonds to within 1e-6 on
  four dates (normal, near-zero and inverted curves), for both
  interpolations; the forward curve integrates to `-ln P`; the
  monotone-convex segment integrals vanish; flat curves stay flat;
* Hull-White: `theta` equals `alpha' + a alpha`; Monte Carlo zero-coupon bond
  prices are within 3 SE of the curve at 1-30 years; simulated moments and
  the exact step covariance match a fine Euler scheme; zero-coupon and
  Jamshidian coupon-bond option prices are within 3 SE of Monte Carlo, with
  and without control variates; put-call parity; the volatility fit
  recovers known parameters;
* hybrid: Monte Carlo within 3 SE of the closed form for three correlations
  and two maturities (one-step and multi-step grids); the discounted index
  and the discount factor are martingales; the simulated correlation is
  `rho`; the forward variance matches numerical integration; `sigma_r = 0`
  reduces to Black-Scholes; hedge ratios match finite differences and
  replicate the value;
* stochastic-rate hedging: the index + bond hedge has under 0.3x the error
  of the index-only hedges, is unbiased and improves with frequency; bond
  trading costs are accounted exactly; with zero rate volatility on a flat
  curve the new backtest path reproduces the flat-rate backtest.

## Limitations

* Decrements are deterministic and market-independent. There is no dynamic
  (moneyness-dependent) lapse, no mortality improvement and no death benefit.
* The hedge is funded by an upfront price equal to the model value. The fee
  income that funds guarantees in practice, its split from management fees,
  and fair-fee solving are not modelled.
* No vega hedging with options, so the Heston variance risk is left open by
  design. The rate hedge uses one zero-coupon bond matching the guarantee.
* The Heston hedger observes the true instantaneous variance and parameters.
  In practice both would be estimated or calibrated, which adds error.
* Rates: one factor, so all yields move together and the curve cannot twist;
  the observed hump in yield volatility (peaking near five years) is not
  reproduced. `(a, sigma_r)` come from historical yield changes, not from
  swaption prices, and there is no market price of rate risk. Rates can go
  negative. The hybrid model has constant equity volatility; combining
  Heston with Hull-White is not implemented. The CMT bootstrap ignores day
  counts, accrued interest and the bill/bond quoting differences.
* No basis risk between fund and index.
* The historical backtest uses a price index without dividends and a flat
  2% rate, over about ten years of FRED data (FRED only publishes the most
  recent ten years of `SP500`). The windows overlap heavily, so they are not
  independent samples.
* This is not a full actuarial liability model or a regulatory capital
  calculation.

## Data

`data/sp500_fred.csv` holds S&P 500 daily closing levels from the Federal
Reserve Bank of St. Louis (FRED, series `SP500`), downloaded on 2026-10-03.
`scripts/fetch_index.py` refreshes it, and the committed file is used when
the download fails.

`data/treasury_cmt_fred.csv` holds daily U.S. Treasury constant-maturity
yields (percent) from FRED, series `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`,
`DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10`, `DGS20` and `DGS30`, from 2016-10-03
to 2026-10-01, downloaded on 2026-10-03. `scripts/fetch_treasury.py`
refreshes it, with the same snapshot fallback. Tests and the report read the
cached file, so they run offline.

## References

* M. Hardy, *Investment Guarantees: Modeling and Risk Management for
  Equity-Linked Life Insurance*, Wiley, 2003.
* S. Heston, "A closed-form solution for options with stochastic volatility",
  *Review of Financial Studies*, 1993.
* H. Albrecher, P. Mayer, W. Schoutens, J. Tistaert, "The little Heston trap",
  *Wilmott Magazine*, 2007.
* R. Lord, R. Koekkoek, D. van Dijk, "A comparison of biased simulation
  schemes for stochastic volatility models", *Quantitative Finance*, 2010.
* P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
* J. Hull, A. White, "Pricing interest-rate-derivative securities",
  *Review of Financial Studies*, 1990.
* F. Jamshidian, "An exact bond option formula", *Journal of Finance*, 1989.
* P. Hagan, G. West, "Interpolation methods for curve construction",
  *Applied Mathematical Finance*, 2006.
* D. Brigo, F. Mercurio, *Interest Rate Models: Theory and Practice*,
  Springer, 2nd ed., 2006.
* J. Hull, A. White, "Optimal delta hedging for options", *Journal of Banking
  & Finance*, 2017.

Project period: 2026-01-12 to 2026-02-13.
