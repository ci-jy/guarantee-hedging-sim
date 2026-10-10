# Results

All numbers come from fixed seeds and are reproduced by the notebook (`docs/report.html`). Hedge P&L figures are per 100 of premium.

## Price accuracy

On a grid of guarantee levels 80/100/120 and maturities 1/5/10/20 years (200,000 paths each), every plain and antithetic+control estimate is within 3 standard errors of the closed form (largest |z| = 1.72). Asserted in `tests/test_pricing.py`.

## Variance reduction

10-year ATM guarantee, GBM, variance per path relative to plain Monte Carlo: control variate 1.8x, antithetic 3.7x, both 10.5x. Under Heston (5-year), the two controls give 3.9x, and with antithetic sampling 7.4x. The tests assert lower bounds of 1.5x, 2.5x, 6x and 2x.

## Convergence

RMSE against the closed form, over 60 replications at each N from 500 to 64,000, falls with a log-log slope close to -0.5. The test requires a slope in (-0.6, -0.4).

## Greeks

Pathwise and finite-difference deltas and vegas agree within their standard errors and with the closed form. Under Heston the pathwise, finite-difference and semi-analytic deltas also agree.

## Rebalancing frequency

5-year guarantee, GBM, 2,000 paths, no costs. Hedge-error standard deviation is 1.40 monthly, 0.71 weekly and 0.32 daily (against a price of 11.48). The monthly/daily ratio is 4.4, close to sqrt(21) = 4.6. Costs of 10-20 bp per trade reduce mean P&L by 0.86-1.73 for daily rebalancing and 0.23-0.45 for monthly.

## Model risk

Same contract, Heston market, weekly rebalancing. The hedge-error standard deviation is 3.28 for Black-Scholes at the implied volatility, 3.86 for the Heston delta and 2.91 for the Heston minimum-variance delta, against 0.71 for Black-Scholes hedging in a Black-Scholes market. Delta hedging alone cannot remove variance risk, even with the true model. The minimum-variance delta lowers the spread but gives up more equity risk premium (mean P&L -1.16 under a 7% drift). The Black-Scholes hedge P&L has a correlation of -0.74 with realised volatility.

## Historical

96 overlapping one-year contracts on the S&P 500 series. The standard deviation of hedge error is 3.3-4.2, and the weekly P&L has a correlation of -0.92 with the gap between realised and issue-date volatility. Rebalancing more often did not reduce the error on this sample.

## Stochastic-rate validation

Latest curve in the snapshot, fitted `a = 0.0127`, `sigma_r = 0.94%`, `sigma_S = 20%`. Both interpolations reprice all 11 par instruments to about 1e-16. Monte Carlo zero-coupon bond prices at 1-30 years are within 3 SE of the curve (largest |z| = 2.24), and 12 zero-coupon and coupon bond-option prices are within 3 SE of the closed forms (largest |z| 1.84 plain, 1.45 with controls; the controls cut the SE 1.3-3.4x). Hybrid Monte Carlo agrees with the closed form for 10 and 20-year guarantees at `rho` = -0.3, 0, 0.3 (largest |z| = 1.55).

## Value effect of stochastic rates

At `rho = 0` the 10-year guarantee is worth 6.35 with stochastic rates against 6.00 with deterministic rates (-5.5% error), and the 20-year guarantee 2.69 against 2.10 (-22%). With `rho = +0.3` the deterministic error reaches -15% at 10 years and -36% at 20 years; with `rho = -0.3` deterministic rates overstate the value by 1.8-6.5%. The flat 3% rate of the earlier sections, below the current curve, overstates the value by far more (11.23 at 10 years).

## Hedging with stochastic rates

10-year guarantee, 2,000 paths, weekly grid, no costs.

| hedge | rebalancing | hedge-error sd |
|---|---|---|
| Black-Scholes at deterministic rates, index only | weekly | 2.78 |
| hybrid delta, index only | weekly | 3.10 |
| hybrid delta, index + zero-coupon bond | weekly | 0.41 |
| hybrid delta, index + zero-coupon bond | monthly | 0.79 |
| hybrid delta, index + zero-coupon bond | quarterly | 1.45 |

For comparison, the same Black-Scholes hedge when rates are deterministic has a standard deviation of 0.41. The index + bond hedge is about 7.5x lower than the index-only hybrid hedge.
