# What the tests check

`python3 -m pytest -q` runs 136 tests in about 10 seconds. The statistical tests use fixed seeds and check:

- the Monte Carlo price is within 3 SE of the closed form on the 3 x 4 grid of guarantees and maturities, for plain and antithetic+control estimators;
- 95% confidence intervals cover the closed form in 91.5-98.5% of 400 independent runs;
- the variance-reduction factors stay above their thresholds;
- the convergence slope is within (-0.6, -0.4);
- pathwise and finite-difference Greeks agree with each other (within 3 combined SE) and with closed-form Greeks;
- the Heston Monte Carlo price matches the semi-analytic price within 3 SE; semi-analytic Heston reduces to Black-Scholes as `xi -> 0`; Heston delta and `dV/dv` match finite differences;
- hedge-error standard deviation falls from monthly to weekly to daily rebalancing under Black-Scholes, with a monthly/daily ratio near sqrt(21);
- the zero-cost hedge is unbiased; transaction costs equal the P&L difference exactly; single-period accounting matches a hand calculation;
- Black-Scholes hedging in a Heston market has more than twice the error of the same hedge in a GBM market, and the minimum-variance delta improves on it;
- historical CSV parsing (FRED `.` markers), window alignment and the backtest on the snapshot;
- the bootstrapped curve reprices the Treasury par bonds to within 1e-6 on four dates (normal, near-zero and inverted curves), for both interpolations; the forward curve integrates to `-ln P`; the monotone-convex segment integrals vanish; flat curves stay flat;
- Hull-White: `theta` equals `alpha' + a alpha`; Monte Carlo zero-coupon bond prices are within 3 SE of the curve at 1-30 years; simulated moments and the exact step covariance match a fine Euler scheme; zero-coupon and Jamshidian coupon-bond option prices are within 3 SE of Monte Carlo, with and without control variates; put-call parity; the volatility fit recovers known parameters;
- hybrid: Monte Carlo within 3 SE of the closed form for three correlations and two maturities (one-step and multi-step grids); the discounted index and the discount factor are martingales; the simulated correlation is `rho`; the forward variance matches numerical integration; `sigma_r = 0` reduces to Black-Scholes; hedge ratios match finite differences and replicate the value;
- stochastic-rate hedging: the index + bond hedge has under 0.3x the error of the index-only hedges, is unbiased and improves with frequency; bond trading costs are accounted exactly; with zero rate volatility on a flat curve the new backtest path reproduces the flat-rate backtest.
