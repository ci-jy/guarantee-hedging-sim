# guarantee-hedging-sim

Monte Carlo pricing and dynamic delta hedging of variable annuity maturity
guarantees.

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
* hedges overlapping one-year contracts on historical S&P 500 paths.

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
jupyter nbconvert --to html --execute notebooks/report.ipynb --output-dir /tmp   # report (~25 s)
python3 scripts/fetch_index.py                         # refresh the S&P 500 snapshot from FRED
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
| `data/sp500_fred.csv` | S&P 500 daily closes from FRED (series `SP500`, 2016-10-03 to 2026-10-02) |
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
* **Rates.** One flat continuously compounded rate. No interest-rate model.
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

## Tests

`python3 -m pytest -q` runs 72 tests in about 10 seconds. The statistical
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
  backtest on the snapshot.

## Limitations

* Decrements are deterministic and market-independent. There is no dynamic
  (moneyness-dependent) lapse, no mortality improvement and no death benefit.
* The hedge is funded by an upfront price equal to the model value. The fee
  income that funds guarantees in practice, its split from management fees,
  and fair-fee solving are not modelled.
* One hedge instrument (the index). No vega hedging with options, so the
  Heston variance risk is left open by design.
* The Heston hedger observes the true instantaneous variance and parameters.
  In practice both would be estimated or calibrated, which adds error.
* Flat interest rate, no interest-rate or basis risk between fund and index.
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
* J. Hull, A. White, "Optimal delta hedging for options", *Journal of Banking
  & Finance*, 2017.
