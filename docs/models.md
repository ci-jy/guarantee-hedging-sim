# Model assumptions and methods

## Assumptions

Fund. The fund tracks an equity index `S` less a continuous management fee `m`: `F_t = S_t exp(-m t)`, with `S_0 = F_0 = premium`. The index is the hedge instrument and pays no dividends.

Guarantee. Survivors at maturity receive `max(F_T, G)`. The insurer's claim is `(G - F_T)^+` per surviving policy. Only the maturity benefit is modelled: no death benefit, ratchets or resets.

Decrements. Lapses at a constant force (3% a year by default) and Gompertz-Makeham mortality `mu(x) = A + B c^x` (A = 0.00022, B = 2.7e-6, c = 1.124, issue age 55). Decrements are deterministic and independent of the market, so the cohort liability is `_T p_x` times one put. Policyholders who leave early take their fund value and the guarantee lapses with them.

Rates. Sections 1-8 of the report use one flat continuously compounded rate (`GMMBContract.rate`). The stochastic-rate extension uses a one-factor Hull-White model `dr = (theta(t) - a r) dt + sigma_r dW_r` fitted to the bootstrapped Treasury curve, with `a` and `sigma_r` fitted to historical yield volatilities (no swaption data). Rates are simulated under the risk-neutral measure only; the real-world equity drift is the short rate plus a constant premium.

Hybrid model. `dS/S = r dt + sigma_S dW_S`, `d<W_S, W_r> = rho dt`, constant `sigma_S` and `rho`. The fund is still `S_t exp(-m t)`.

Black-Scholes. The fund is a stock with dividend yield `m`, so the guarantee value is the textbook put with yield, scaled by `_T p_x`.

Heston. `dv = kappa (theta - v) dt + xi sqrt(v) dW2`, `corr(dW1, dW2) = rho`. Default parameters `v0 = theta = 0.04`, `kappa = 1.5`, `xi = 0.5`, `rho = -0.7` (the Feller condition fails, so the variance does reach zero). Real-world and risk-neutral variance dynamics are the same (no variance risk premium); the real-world equity drift is set separately (`mu`).

## Monte Carlo pricing

GBM terminal values are simulated exactly in one step. Heston paths use the full-truncation Euler scheme (Lord, Koekkoek and van Dijk, 2010) on the log-price, which keeps the discounted index an exact martingale on the grid. The discretisation bias of the Heston price is first order in the step size; it is visible against the semi-analytic price at 12 steps a year and negligible at 250.

- Antithetic variates pair each normal vector with its negative; the standard error is computed from the pair averages.
- Control variates use the discounted terminal index (known mean `S_0`). For Heston a second control is the guarantee payoff on a "shadow" GBM path driven by the same Brownian motion, with volatility equal to the square root of the expected average variance; its expectation is the closed-form price. Coefficients are the least-squares betas estimated on the same sample (the resulting bias is O(1/N)).
- Results carry a standard error, normal-approximation confidence intervals and `variance_per_path = N * SE^2`, which compares methods at equal cost.

## Semi-analytic Heston

Gil-Pelaez inversion with the "little Heston trap" characteristic function (Albrecher et al., 2007). The integral is mapped from `(0, inf)` to `(0, 1)` and evaluated with 192-point Gauss-Legendre quadrature on a grid that depends only on time to maturity. Thousands of (index, variance) states at the same date are therefore priced in one matrix product, which makes Heston hedging backtests feasible. Delta is `exp(-q tau) (P1 - 1)`, and `dV/dv` comes from differentiating the characteristic function (`D(u) phi(u)`).

## Greeks

Both models are homogeneous in `S_0`, so the pathwise delta is `-1{F_T < G} F_T / S_0` (discounted, times `_T p_x`). The GBM pathwise vega uses `dF_T/dsigma = F_T (W_T - sigma T)`. Finite differences are central bumps that re-simulate with the same seed (common random numbers). The payoff has a kink, so there is no pathwise gamma; gamma is by finite difference only.

## Hedging backtester

The insurer receives a price for the guarantee at issue (by default the hedger's own model value), holds `dV/dS` units of the index and keeps the rest in cash at the flat rate. It rebalances every `k` grid steps, paying `cost_rate * |trade| * S` per trade, including the final unwind. The hedge P&L is the discounted value of the final portfolio minus the discounted claim, per 100 of premium. Reported statistics: mean, standard deviation, 95% VaR and CVaR (mean of the worst 5% of losses).

## Curve bootstrap

CMT yields are par yields on a semi-annual bond-equivalent basis. Maturities up to six months are zero-coupon instruments; longer ones are semi-annual par bonds (no accrued interest or day counts). The curve interpolates `-ln P(0, t)` with either the monotone-convex scheme of Hagan and West (2006), whose forward curve is continuous and local, or a natural cubic spline; the forward is held flat beyond 30 years. Because both schemes couple neighbouring segments, all knot zero rates are solved at once with SciPy's Powell hybrid root finder until every instrument prices to par (typically to 1e-16).

## Hull-White

With `r = x + alpha(t)`, `dx = -a x dt + sigma_r dW`, `alpha(t) = f(0,t) + sigma_r^2 B(t)^2 / 2`, the model reproduces `P(0, t)` for any parameters. `x` and its integral are jointly Gaussian over a step, so the short rate and `D(0, t) = exp(-int r)` are simulated exactly (no discretisation error). Closed forms: affine bond prices `P(t, T) = A e^{-B r}`, the Hull-White zero-coupon bond option, and coupon-bond options by Jamshidian's decomposition. The Monte Carlo bond-option pricer can use `D(0,T)` and the discounted bond price as control variates. `(a, sigma_r)` are fitted by least squares to the model's yield-volatility curve `sigma_r B(tau)/tau` against the annualised standard deviation of weekly 2-30 year yield changes over the last three years.

## Hybrid closed form

Under the `T`-forward measure, `S_t / P(t, T)` is lognormal with total variance `v(T) = sigma_S^2 T + 2 rho sigma_S sigma_r int B + sigma_r^2 int B^2`, so the guarantee is the Black-Scholes put at the zero rate `-ln P(0,T)/T` with volatility `sqrt(v(T)/T)`. Monte Carlo simulates `(x, int x, W_S)` exactly, discounts each path with its own `D(0, T)`, and uses `D(0,T) S_T` and `D(0,T)` as controls (`price_gmmb` accepts a `HybridModel`).

## Index + bond hedge

The value `V(t, S, P)` is homogeneous of degree one in the index and the price `P(t, T)` of the zero-coupon bond maturing with the guarantee, so holding `dV/dS` index units and `dV/dP` bonds replicates it. The backtester accrues cash at the simulated short rate, discounts each path with `D(0, t)`, and trades the bond when the hedger has a `bond_units` method (bond trades cost `bond_cost_rate`, by default the same as index trades). With flat-rate paths it behaves exactly as before.

## Historical backtest

One-year contracts are issued every 21 trading days along the S&P 500 series. Each window is rescaled to start at 100, priced and hedged with Black-Scholes at the trailing one-year realised volatility at issue, with a 2% rate and 10 bp costs.

## References

- M. Hardy, Investment Guarantees: Modeling and Risk Management for Equity-Linked Life Insurance, Wiley, 2003.
- S. Heston, "A closed-form solution for options with stochastic volatility", Review of Financial Studies, 1993.
- H. Albrecher, P. Mayer, W. Schoutens, J. Tistaert, "The little Heston trap", Wilmott Magazine, 2007.
- R. Lord, R. Koekkoek, D. van Dijk, "A comparison of biased simulation schemes for stochastic volatility models", Quantitative Finance, 2010.
- P. Glasserman, Monte Carlo Methods in Financial Engineering, Springer, 2003.
- J. Hull, A. White, "Pricing interest-rate-derivative securities", Review of Financial Studies, 1990.
- F. Jamshidian, "An exact bond option formula", Journal of Finance, 1989.
- P. Hagan, G. West, "Interpolation methods for curve construction", Applied Mathematical Finance, 2006.
- D. Brigo, F. Mercurio, Interest Rate Models: Theory and Practice, Springer, 2nd ed., 2006.
- J. Hull, A. White, "Optimal delta hedging for options", Journal of Banking & Finance, 2017.
