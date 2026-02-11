"""Generate notebooks/report.ipynb from the cell sources below.

Usage: python3 notebooks/build_report.py
Then execute and export with
    jupyter nbconvert --to html --execute notebooks/report.ipynb --output-dir docs
"""
from pathlib import Path

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Pricing and hedging a variable annuity maturity guarantee

This report prices a guaranteed minimum maturity benefit (GMMB) by Monte Carlo,
checks the price against the closed-form Black-Scholes value, and studies
how well the guarantee can be delta-hedged: as a function of rebalancing
frequency and transaction costs, when the hedging model is wrong (the market
follows Heston), and on historical S&P 500 paths. Section 9 replaces the flat
discount rate with a Hull-White short-rate model fitted to the Treasury curve
and measures what assuming deterministic rates misses.

Every number below is produced by the `ghedge` package from fixed random seeds,
so re-running the notebook reproduces it.
""")

code(r"""
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / "ghedge").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

from ghedge import GBM, GMMBContract, Heston, Decrements
from ghedge import blackscholes as bs, heston
from ghedge.montecarlo import METHODS, price_gmmb, convergence_study, convergence_slope
from ghedge.greeks import pathwise_delta, fd_delta, pathwise_vega, fd_vega, fd_gamma
from ghedge.hedging import BSHedger, HestonHedger, backtest, rebalancing_table, risk_summary
from ghedge.data import load_index
from ghedge.historical import historical_backtest

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.grid": True, "grid.alpha": 0.3})
%matplotlib inline
""")

md(r"""
## 1. Contract and assumptions

* Single premium of 100 invested in a fund that tracks an equity index less a
  continuous management fee $m$: $F_t = S_t e^{-mt}$.
* At maturity $T$ each surviving policy receives $\max(F_T, G)$; the insurer
  owes the shortfall $(G - F_T)^+$, a put on the fund.
* Policyholders leave through lapses (constant force) and death
  (Gompertz-Makeham mortality). Decrements are deterministic and independent
  of the market, so the cohort liability is ${}_Tp_x \cdot \text{Put}$.
* Flat risk-free rate $r$. Under Black-Scholes the fund is a stock with
  dividend yield $m$:

$$V_0 = {}_Tp_x\left(G e^{-rT} N(-d_2) - F_0 e^{-mT} N(-d_1)\right),\quad
d_{1,2} = \frac{\ln(F_0/G) + (r - m \pm \sigma^2/2)T}{\sigma\sqrt{T}}.$$
""")

code(r"""
contract = GMMBContract(premium=100.0, guarantee=100.0, maturity=10.0, fee=0.02, rate=0.03,
                        decrements=Decrements(age=55, lapse=0.03))
sigma = 0.2
display(pd.Series({"premium": contract.premium, "guarantee": contract.guarantee,
                   "maturity (years)": contract.maturity, "fee": contract.fee, "rate": contract.rate,
                   "issue age": contract.decrements.age, "lapse force": contract.decrements.lapse,
                   "P(in force at maturity)": contract.maturity_survival,
                   "closed-form value": float(bs.gmmb_value(contract, sigma))}, name="value").to_frame())

t = np.linspace(0, 30, 301)
fig, ax = plt.subplots()
ax.plot(t, contract.decrements.survival(t), label="lapse + mortality")
ax.plot(t, Decrements(age=55, lapse=0.0).survival(t), label="mortality only")
ax.set_xlabel("years since issue"); ax.set_ylabel("fraction in force"); ax.legend()
ax.set_title("Cohort decrements (issue age 55)")
plt.show()
""")

md(r"""
## 2. Monte Carlo price against the closed form

200,000 paths per cell, plain Monte Carlo and antithetic + control variates.
The z-score is (MC - closed form) / standard error; the test suite asserts
$|z| < 3$ on this grid.
""")

code(r"""
rows = []
for g in [80.0, 100.0, 120.0]:
    for T in [1.0, 5.0, 10.0, 20.0]:
        c = contract.with_(guarantee=g, maturity=T)
        exact = float(bs.gmmb_value(c, sigma))
        for method in ["plain", "antithetic_control"]:
            res = price_gmmb(c, GBM(sigma), 200_000, method, rng=int(g * 100 + T))
            lo, hi = res.ci()
            rows.append({"guarantee": g, "maturity": T, "method": method, "closed_form": exact,
                         "mc": res.estimate, "std_error": res.std_error,
                         "ci95_low": lo, "ci95_high": hi,
                         "z": (res.estimate - exact) / res.std_error})
grid = pd.DataFrame(rows)
display(grid)
print(f"max |z| over {len(grid)} estimates: {grid['z'].abs().max():.2f}")
""")

md(r"""
## 3. Variance reduction

* **Antithetic variates** pair each normal draw $Z$ with $-Z$.
* **Control variate** (GBM): the discounted terminal index $e^{-rT}S_T$, whose
  mean $S_0$ is known. The coefficient is the least-squares $\beta$.
* **Control variates** (Heston): the discounted index plus the guarantee
  payoff on a *shadow* GBM path driven by the same Brownian motion with
  volatility $\sqrt{\bar v}$ (the expected average variance). Its value is
  known in closed form.

The factor is the variance per simulated path of plain Monte Carlo divided by
that of each method, i.e. the saving in paths for the same accuracy.
""")

code(r"""
def vr_table(model, c, n, **kw):
    res = {m: price_gmmb(c, model, n, m, rng=7, **kw) for m in METHODS}
    base = res["plain"].variance_per_path
    return pd.DataFrame({m: {"estimate": r.estimate, "std_error": r.std_error,
                             "variance_per_path": r.variance_per_path,
                             "variance_reduction": base / r.variance_per_path}
                         for m, r in res.items()}).T

print("GBM, 10-year at-the-money guarantee, 400,000 paths")
display(vr_table(GBM(sigma), contract, 400_000))

heston_model = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
c5 = contract.with_(maturity=5.0)
print("Heston, 5-year guarantee, 100,000 paths, 50 steps a year; "
      f"semi-analytic value {float(heston.gmmb_value_and_greeks(c5, heston_model)[0][0]):.4f}")
display(vr_table(heston_model, c5, 100_000, steps_per_year=50))
""")

md(r"""
## 4. Convergence

RMSE of 60 independent estimates around the closed form for each path count.
Monte Carlo error should fall like $N^{-1/2}$: a slope of $-0.5$ on a log-log
plot.
""")

code(r"""
sizes = [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
conv = {m: convergence_study(contract, GBM(sigma), sizes, n_reps=60, method=m, seed=3)
        for m in ["plain", "antithetic_control"]}
fig, ax = plt.subplots()
for m, df in conv.items():
    ax.loglog(df["n_paths"], df["rmse"], "o-", label=f"{m} (slope {convergence_slope(df):.3f})")
ref = conv["plain"]["rmse"].iloc[0] * np.sqrt(sizes[0] / np.array(sizes))
ax.loglog(sizes, ref, "k--", alpha=0.5, label=r"$N^{-1/2}$ reference")
ax.set_xlabel("paths N"); ax.set_ylabel("RMSE vs closed form"); ax.legend()
ax.set_title("Monte Carlo convergence")
plt.show()
display(pd.concat({m: df.set_index("n_paths")["rmse"] for m, df in conv.items()}, axis=1))
""")

md(r"""
## 5. Greeks

Pathwise derivatives differentiate the payoff along each path:
$\partial (G - F_T)^+/\partial S_0 = -\mathbf 1\{F_T < G\}\, F_T / S_0$ and, under GBM,
$\partial F_T/\partial\sigma = F_T (W_T - \sigma T)$. Finite differences use
central bumps with common random numbers. Gamma has no pathwise estimator
(the payoff has a kink), so only the finite-difference value is shown.
""")

code(r"""
rows = []
for g, T in [(80.0, 5.0), (100.0, 10.0), (120.0, 2.0)]:
    c = contract.with_(guarantee=g, maturity=T)
    for name, pw, fd, exact in [
        ("delta", pathwise_delta(c, GBM(sigma), 200_000, seed=1), fd_delta(c, GBM(sigma), 200_000, seed=1),
         float(bs.gmmb_delta(c, sigma))),
        ("vega", pathwise_vega(c, GBM(sigma), 200_000, seed=2), fd_vega(c, GBM(sigma), 200_000, seed=2),
         float(bs.gmmb_vega(c, sigma))),
    ]:
        rows.append({"guarantee": g, "maturity": T, "greek": name, "closed_form": exact,
                     "pathwise": pw.estimate, "pathwise_se": pw.std_error,
                     "finite_diff": fd.estimate, "fd_se": fd.std_error})
    gm = fd_gamma(c, GBM(sigma), 400_000, seed=3)
    rows.append({"guarantee": g, "maturity": T, "greek": "gamma", "closed_form": float(bs.gmmb_gamma(c, sigma)),
                 "pathwise": np.nan, "pathwise_se": np.nan, "finite_diff": gm.estimate, "fd_se": gm.std_error})
display(pd.DataFrame(rows))

c2 = contract.with_(maturity=2.0)
pw = pathwise_delta(c2, heston_model, 50_000, seed=4, steps_per_year=100)
fd = fd_delta(c2, heston_model, 50_000, seed=4, steps_per_year=100)
print(f"Heston 2-year delta: semi-analytic {float(heston.gmmb_value_and_greeks(c2, heston_model)[1][0]):.5f}, "
      f"pathwise {pw.estimate:.5f} ± {pw.std_error:.5f}, finite difference {fd.estimate:.5f} ± {fd.std_error:.5f}")
""")

md(r"""
## 6. Delta hedging under Black-Scholes

The insurer charges the Black-Scholes value, holds $\Delta = \partial V/\partial S$
units of the index and the rest in cash, and rebalances daily (every trading
day), weekly (5 days) or monthly (21 days). Paths follow GBM with a 7%
real-world drift and $\sigma = 20\%$; the contract is a 5-year at-the-money
guarantee. Hedge P&L is discounted to issue and quoted per 100 of premium.
CVaR is the mean loss in the worst 5% of paths.
""")

code(r"""
hedge_contract = contract.with_(maturity=5.0)
n_hedge_paths = 2_000
gbm_paths = GBM(sigma, mu=0.07).simulate(100.0, hedge_contract.rate, 5.0, 5 * 252, n_hedge_paths, rng=21)
bs_table = rebalancing_table(gbm_paths, hedge_contract, [BSHedger(hedge_contract, sigma)],
                             cost_rates=(0.0, 0.001, 0.002))
display(bs_table.drop(columns=["strategy"]))

fig, ax = plt.subplots()
for name, k in [("monthly", 21), ("weekly", 5), ("daily", 1)]:
    pnl = backtest(gbm_paths, hedge_contract, BSHedger(hedge_contract, sigma), k).pnl
    ax.hist(pnl, bins=60, alpha=0.5, density=True, label=f"{name} (sd {pnl.std():.2f})")
ax.set_xlabel("discounted hedge P&L per 100 premium"); ax.legend()
ax.set_title("Hedge P&L under GBM, no transaction costs")
plt.show()

nc = bs_table[bs_table.cost_rate == 0].set_index("frequency")["std_pnl"]
print(f"std of hedge error: monthly {nc['monthly']:.3f}, weekly {nc['weekly']:.3f}, daily {nc['daily']:.3f}; "
      f"monthly/daily ratio {nc['monthly'] / nc['daily']:.2f} (sqrt(21) = {np.sqrt(21):.2f})")
""")

md(r"""
With no costs the standard deviation of hedge error falls roughly like the
square root of the rebalancing interval. Proportional costs pull the mean
P&L down, and more so for frequent rebalancing: the frequency choice trades
hedge error against cost.
""")

md(r"""
## 7. Model risk: hedging a Heston market with Black-Scholes

The market now follows Heston with $v_0 = \theta = 0.04$ ($\sqrt\theta = 20\%$),
$\kappa = 1.5$, $\xi = 0.5$, $\rho = -0.7$ and a 7% drift. Four hedgers:

| strategy | price charged | hedge ratio |
|---|---|---|
| BS at implied vol | Heston value | BS delta at the volatility that reproduces the Heston value |
| BS at $\sqrt\theta$ | BS value at 20% | BS delta at 20% (the "natural" but wrong model) |
| Heston delta | Heston value | $\partial V/\partial S$ with the true model, observing $v_t$ |
| Heston min-variance | Heston value | $\partial V/\partial S + \rho\xi S^{-1}\partial V/\partial v$ |

The Heston hedgers know the true parameters and observe the current variance;
even so, a hedge in the index alone cannot remove variance risk. All four run on
the same 2,000 Heston paths; the GBM result from section 6 is the baseline.
""")

code(r"""
heston_market = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, mu=0.07)
heston_paths = heston_market.simulate(100.0, hedge_contract.rate, 5.0, 5 * 252, n_hedge_paths, rng=22)
heston_value = float(heston.gmmb_value_and_greeks(hedge_contract, heston_market)[0][0])
implied = bs.implied_vol_put(heston_value / hedge_contract.maturity_survival, 100.0, hedge_contract.guarantee,
                             hedge_contract.maturity, hedge_contract.rate, hedge_contract.fee)
print(f"Heston value {heston_value:.4f}; BS value at 20% {float(bs.gmmb_value(hedge_contract, 0.2)):.4f}; "
      f"implied vol {implied:.2%}")

hedgers = [BSHedger(hedge_contract, implied, label="BS at implied vol"),
           BSHedger(hedge_contract, 0.2, label="BS at sqrt(theta)"),
           HestonHedger(hedge_contract, heston_market, label="Heston delta"),
           HestonHedger(hedge_contract, heston_market, minimum_variance=True, label="Heston min-variance")]
freqs = {"weekly": 5, "monthly": 21}
mr = rebalancing_table(heston_paths, hedge_contract, hedgers, frequencies=freqs)
base = rebalancing_table(gbm_paths, hedge_contract, [BSHedger(hedge_contract, sigma, label="GBM market, BS delta")],
                         frequencies=freqs)
model_risk = pd.concat([base, mr], ignore_index=True)
display(model_risk[["strategy", "frequency", "price", "mean_pnl", "std_pnl", "var_95", "cvar_95"]])
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
weekly = {h.label: backtest(heston_paths, hedge_contract, h, 5) for h in hedgers}
gbm_weekly = backtest(gbm_paths, hedge_contract, BSHedger(hedge_contract, sigma), 5)
ax = axes[0]
ax.hist(gbm_weekly.pnl, bins=60, alpha=0.5, density=True, label="GBM market, BS delta")
for lab in ["BS at implied vol", "Heston min-variance"]:
    ax.hist(weekly[lab].pnl, bins=60, alpha=0.5, density=True, label=f"Heston market, {lab}")
ax.set_xlabel("discounted hedge P&L"); ax.legend(fontsize=8); ax.set_title("Weekly hedging")

realised_var = (np.diff(np.log(heston_paths.index), axis=1) ** 2).sum(axis=1) / hedge_contract.maturity
ax = axes[1]
ax.scatter(np.sqrt(realised_var), weekly["BS at implied vol"].pnl, s=4, alpha=0.4)
ax.axvline(implied, color="k", ls="--", lw=1, label="hedging vol")
ax.set_xlabel("realised volatility over the contract"); ax.set_ylabel("hedge P&L")
ax.set_title("BS hedge error is driven by realised vs assumed vol"); ax.legend()
plt.tight_layout(); plt.show()

corr = np.corrcoef(np.sqrt(realised_var), weekly["BS at implied vol"].pnl)[0, 1]
print(f"correlation of BS hedge P&L with realised vol: {corr:.2f}")
""")

code(r"""
w = model_risk[model_risk.frequency == "weekly"].set_index("strategy")
gbm_sd = w.loc["GBM market, BS delta", "std_pnl"]
print("Weekly hedge-error standard deviation relative to the GBM baseline:")
for s in ["BS at implied vol", "BS at sqrt(theta)", "Heston delta", "Heston min-variance"]:
    print(f"  {s:22s} sd {w.loc[s, 'std_pnl']:.3f}  ({w.loc[s, 'std_pnl'] / gbm_sd:.1f}x)  "
          f"CVaR95 {w.loc[s, 'cvar_95']:.3f}  mean {w.loc[s, 'mean_pnl']:+.3f}")
""")

md(r"""
**Reading the model-risk result.**

* With the market in Heston, the Black-Scholes hedge error is several times
  larger than the same hedge in a Black-Scholes market. More frequent
  rebalancing helps very little, because the error comes from the gap between
  realised and assumed volatility (right panel), not from discreteness.
* At the money the Heston value is below the Black-Scholes value at
  $\sqrt\theta$ (the implied volatility printed above is under 20%), so the
  "BS at $\sqrt\theta$" hedger overcharges. That shows up as a positive mean
  P&L, not as a smaller spread.
* Hedging with the true model's delta alone does **not** remove the error:
  the variance risk stays unhedged. The minimum-variance delta, which uses the
  correlation between index and variance, gives the smallest spread. Its mean
  P&L is lower under the real-world measure because it holds a larger short
  index position and so gives up more of the equity risk premium.
* Removing the remaining error needs a second instrument that trades
  volatility (for example listed index options), which is outside the scope here.
""")

md(r"""
## 8. Historical backtest on the S&P 500

Daily S&P 500 closing levels from FRED (`scripts/fetch_index.py` refreshes
the committed snapshot). One-year GMMB contracts are issued every 21 trading
days; each is priced and hedged with Black-Scholes at the trailing one-year
realised volatility known at issue. The S&P 500 series is a price index, so
dividends are ignored, and the rate is a flat 2%. Transaction costs are 10 bp.
Windows overlap, so the contracts are far from independent.
""")

code(r"""
levels = load_index()
print(f"{len(levels)} daily observations, {levels.index[0].date()} to {levels.index[-1].date()}")
hist_contract = contract.with_(maturity=1.0, rate=0.02)
per_window, hist_summary = historical_backtest(levels, hist_contract, cost_rate=0.001)
display(hist_summary)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(levels.index, levels.values)
axes[0].set_title("S&P 500 level"); axes[0].set_yscale("log")
gap = per_window["realised_vol"] - per_window["issue_vol"]
axes[1].scatter(gap, per_window["pnl_weekly"], s=12)
axes[1].set_xlabel("realised minus issue volatility"); axes[1].set_ylabel("weekly hedge P&L")
axes[1].set_title("Historical hedge error vs volatility surprise")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots()
for name in ["daily", "weekly", "monthly"]:
    ax.plot(per_window["issue_date"], per_window[f"pnl_{name}"], ".-", lw=0.8, label=name)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("discounted hedge P&L per 100"); ax.set_title("Hedge P&L by issue date"); ax.legend()
plt.show()
print(f"correlation of weekly P&L with the volatility surprise: {np.corrcoef(gap, per_window['pnl_weekly'])[0, 1]:.2f}")
""")

md(r"""
On real data the hedge error is dominated by volatility surprises (contracts
issued just before the 2020 sell-off lose the most), so rebalancing more often
does not reduce it the way it does in the simulated Black-Scholes world. In
this sample daily rebalancing even has the larger spread; costs explain part
of the gap in the mean, and one possible reason for the rest is that daily
returns were more volatile than returns measured over longer intervals, which
a monthly hedger never sees. With 96 overlapping windows this is weak evidence
either way. That
matches the Heston experiment in section 7: once volatility is stochastic,
rebalancing frequency is a second-order lever compared with the volatility
assumption.
""")

md(r"""
## 9. Stochastic interest rates: Hull-White and a hybrid equity-rate model

Sections 2-8 discount at one flat rate. For a 10-20 year guarantee the level
and the randomness of interest rates matter as much as equity volatility: the
guarantee pays $G$ far in the future, so its value moves with long rates.
This section

1. bootstraps a zero curve from the U.S. Treasury constant-maturity par
   yields on FRED (`DGS1MO` ... `DGS30`, cached in `data/treasury_cmt_fred.csv`);
2. fits a one-factor Hull-White model $dr = (\theta(t) - a r)\,dt + \sigma_r\,dW_r$
   to that curve exactly and checks it against its closed forms;
3. prices the guarantee in a hybrid model (GBM index with correlated
   Hull-White rates) and compares Monte Carlo with the closed form;
4. measures how the value depends on rate volatility and correlation, and the
   error from assuming deterministic rates;
5. hedges the guarantee with the index plus a zero-coupon bond.

### 9.1 Yield curve bootstrap

CMT yields are par yields on a semi-annual bond-equivalent basis. Maturities
up to six months are treated as zero-coupon instruments, longer ones as
semi-annual par bonds. The knot zero rates are solved simultaneously so that
every instrument prices to par. Between knots, $-\ln P(0,t)$ is interpolated
either with the monotone-convex method of Hagan and West (2006) or with a
natural cubic spline.
""")

code(r"""
from ghedge.data import load_treasury, par_curve
from ghedge.rates import (HullWhite, HybridHedger, HybridModel, bootstrap_series, repricing_errors,
                          fit_to_yield_volatility, yield_volatility)
from ghedge.rates import hybrid
from ghedge.rates.curve import _cashflows
from ghedge.rates.hullwhite import b_factor, mc_bond_option, mc_zcb

treasury = load_treasury()
par = par_curve(treasury)
print(f"Treasury curve of {par.name.date()} ({len(treasury)} dates in the cached history)")
curves = {m: bootstrap_series(par, m) for m in ["monotone_convex", "cubic"]}
curve = curves["monotone_convex"]
fit = pd.DataFrame({"par_yield": par.values}, index=pd.Index(par.index.round(3), name="maturity"))
for m, cv in curves.items():
    fit[f"zero_{m}"] = cv.zeros
    fit[f"reprice_err_{m}"] = repricing_errors(cv, par.index, par.values)
display(fit.style.format({c: "{:.3e}" if c.startswith("reprice") else "{:.5f}" for c in fit.columns}))
print("max |PV - par| : " + ", ".join(f"{m} {np.abs(fit[f'reprice_err_{m}']).max():.1e}" for m in curves))

t = np.linspace(0.0, 30.0, 601)
fig, ax = plt.subplots()
ax.plot(par.index, par.values, "ko", label="CMT par yields")
for m, cv in curves.items():
    ax.plot(t[1:], cv.zero_rate(t[1:]), label=f"zero rate, {m}")
    ax.plot(t, cv.forward(t), "--", label=f"instantaneous forward, {m}")
ax.set_xlabel("maturity (years)"); ax.set_ylabel("rate"); ax.legend(fontsize=8)
ax.set_title(f"Treasury curve {par.name.date()}")
plt.show()
""")

md(r"""
Both interpolations reprice the inputs exactly; they differ between the
knots, mostly in the forward curve beyond 10 years where the instruments are
sparse. The cubic spline's forward curve has a continuous slope but is
global: moving one input yield shifts the forward curve everywhere and it can
oscillate between sparse knots. The monotone-convex forward is only
continuous, but it is local (each segment depends on its neighbours only) and
keeps the forward curve monotone where the discrete forwards are.
The rest of the section uses the monotone-convex curve.

### 9.2 Hull-White parameters and the fitted drift

The drift $\theta(t)$ is chosen so that the model reproduces the curve for any
$(a, \sigma_r)$. Swaption volatilities are not freely available, so $(a, \sigma_r)$
are fitted to the historical volatility of weekly changes in the 2-30 year
yields over the last three years: the model's zero-yield volatility at tenor
$\tau$ is $\sigma_r B(\tau)/\tau$ with $B(\tau) = (1 - e^{-a\tau})/a$. This is
a real-world estimate used as a proxy for the risk-neutral volatility.
""")

code(r"""
vols = yield_volatility(treasury, min_maturity=2.0, start=par.name - pd.DateOffset(years=3))
a_fit, sigma_r_fit = fit_to_yield_volatility(vols.index, vols.values)
print(f"fitted a = {a_fit:.4f}, sigma_r = {sigma_r_fit:.4%}")
hw_model = HullWhite(curve, a=a_fit, sigma=sigma_r_fit)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
tau = np.linspace(1, 30, 100)
axes[0].plot(vols.index, vols.values, "ko", label="observed (weekly changes, annualised)")
axes[0].plot(tau, sigma_r_fit * b_factor(a_fit, tau) / tau, label="Hull-White fit")
axes[0].set_xlabel("tenor (years)"); axes[0].set_ylabel("yield volatility"); axes[0].legend()
axes[0].set_title("Yield volatility term structure")
tt = np.linspace(0, 30, 601)
axes[1].plot(tt, hw_model.theta(tt), label=r"$\theta(t)$")
axes[1].plot(tt, curve.forward(tt), "--", label=r"$f(0,t)$")
axes[1].plot(tt, hw_model.alpha(tt), ":", label=r"$E[r(t)] = \alpha(t)$")
axes[1].set_xlabel("t (years)"); axes[1].legend(); axes[1].set_title("Fitted drift")
plt.tight_layout(); plt.show()
""")

md(r"""
The fitted mean reversion is weak, so long yields are nearly as volatile as
short ones (the model is close to Ho-Lee). A one-factor model cannot produce
the hump in observed volatilities, which peaks around five years because the
short end is anchored by the policy rate; only tenors of two years and more
are used in the fit.

### 9.3 Validation against closed forms

**Zero-coupon bonds.** With $r = x + \alpha(t)$ and $x$ an Ornstein-Uhlenbeck
process, $x$ and $\int x\,ds$ are jointly Gaussian, so the discount factor
$D(0,T) = e^{-\int_0^T r\,ds}$ is sampled exactly. Its mean must equal
$P(0,T)$ from the curve.
""")

code(r"""
mats = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
est, se = mc_zcb(hw_model, mats, 200_000, rng=31)
zcb_table = pd.DataFrame({"curve P(0,T)": curve.discount(mats), "MC E[D(0,T)]": est, "std_error": se},
                         index=pd.Index(mats, name="T"))
zcb_table["z"] = (zcb_table["MC E[D(0,T)]"] - zcb_table["curve P(0,T)"]) / zcb_table["std_error"]
display(zcb_table)
""")

md(r"""
**Bond options.** A European option on a zero-coupon bond has the Hull-White
closed form; an option on a coupon bond is priced by Jamshidian's
decomposition (a portfolio of zero-coupon bond options struck at the bond
prices that correspond to the critical short rate $r^*$). Monte Carlo samples
$r(T)$ and $D(0,T)$ exactly and prices the bond at expiry with the affine
formula. The control-variate estimator uses $D(0,T)$ and the discounted bond
price, both with known means.
""")

code(r"""
rows = []
T_opt = 5.0
atm = float(curve.discount(10.0) / curve.discount(T_opt))
cpn_t, cpn_a = _cashflows(10.0, 0.05)
cases = [("ZCB 5y into 10y", [10.0], [1.0], k) for k in (0.95 * atm, atm, 1.05 * atm)] + \
        [("5% coupon bond 5y into 15y", cpn_t + T_opt, cpn_a, k) for k in (0.9, 1.0, 1.1)]
for i, (name, times, amounts, k) in enumerate(cases):
    for kind in ["call", "put"]:
        exact = hw_model.coupon_bond_option(T_opt, times, amounts, k, kind)
        plain = mc_bond_option(hw_model, T_opt, times, amounts, k, kind, 200_000, rng=40 + i)
        cv = mc_bond_option(hw_model, T_opt, times, amounts, k, kind, 200_000, rng=40 + i, control=True)
        rows.append({"underlying": name, "strike": k, "kind": kind, "closed_form": exact,
                     "mc_plain": plain.estimate, "z_plain": (plain.estimate - exact) / plain.std_error,
                     "mc_control": cv.estimate, "z_control": (cv.estimate - exact) / cv.std_error,
                     "se_ratio": plain.std_error / cv.std_error})
options = pd.DataFrame(rows)
display(options)
print(f"max |z|: plain {options['z_plain'].abs().max():.2f}, control {options['z_control'].abs().max():.2f}")
""")

md(r"""
### 9.4 Hybrid model: the guarantee with stochastic rates

The index follows $dS/S = r\,dt + \sigma_S\,dW_S$ with $d\langle W_S, W_r\rangle = \rho\,dt$.
Under the $T$-forward measure the forward index $S_t/P(t,T)$ is lognormal with
volatility vector $\sigma_S\,dW_S + \sigma_r B(T-t)\,dW_r$, so the guarantee is
a Black-Scholes put with discount factor $P(0,T)$ and total variance

$$v(T) = \sigma_S^2 T + 2\rho\sigma_S\sigma_r\int_0^T B(u)\,du + \sigma_r^2\int_0^T B(u)^2\,du.$$

That is Black-Scholes at the zero rate $-\ln P(0,T)/T$ with the volatility
raised from $\sigma_S$ to $\sqrt{v(T)/T}$. Monte Carlo simulates
$(x, \int x\,ds, W_S)$ exactly and discounts each path with its own $D(0,T)$;
the controls are $D(0,T) S_T$ (mean $S_0$) and $D(0,T)$ (mean $P(0,T)$).

The weekly correlation between S&P 500 log returns and changes in the
10-year yield over the cached history is printed below; it is close to zero,
so the base case uses $\rho = 0$ and the sensitivity covers $\pm 0.5$.
""")

code(r"""
weekly = pd.concat([np.log(levels).resample("W-FRI").last().diff(),
                    treasury["DGS10"].resample("W-FRI").last().diff()], axis=1).dropna()
print(f"corr(weekly S&P 500 log return, weekly change in 10y yield) = {weekly.corr().iloc[0, 1]:.3f}")

rows = []
for T in [10.0, 20.0]:
    c = contract.with_(maturity=T)
    for rho in [-0.3, 0.0, 0.3]:
        m = HybridModel(hw_model, sigma=sigma, rho=rho)
        exact = float(hybrid.gmmb_value(c, m))
        for method in ["plain", "antithetic_control"]:
            res = price_gmmb(c, m, 200_000, method, rng=int(10 * T + 10 * rho + 50))
            rows.append({"maturity": T, "rho": rho, "method": method, "effective_vol": float(m.effective_vol(T)),
                         "closed_form": exact, "mc": res.estimate, "std_error": res.std_error,
                         "z": (res.estimate - exact) / res.std_error})
hybrid_table = pd.DataFrame(rows)
display(hybrid_table)
print(f"max |z| over {len(hybrid_table)} estimates: {hybrid_table['z'].abs().max():.2f}")
""")

md(r"""
### 9.5 Sensitivity to rate volatility and correlation; the deterministic-rate error

The deterministic-rate value keeps the same curve but sets $\sigma_r = 0$
(Black-Scholes at the zero rate to maturity). The flat-rate value is the
original pricer at the 3% rate used in sections 1-7.
""")

code(r"""
rate_vols = [0.0, 0.005, sigma_r_fit, 0.0125, 0.015]
rhos = [-0.5, -0.25, 0.0, 0.25, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
grids = {}
for ax, T in zip(axes, [10.0, 20.0]):
    c = contract.with_(maturity=T)
    base = HybridModel(hw_model, sigma=sigma, rho=0.0)
    grids[T] = hybrid.sensitivity_grid(c, base, rate_vols, rhos)
    for rho in rhos:
        ax.plot(np.array(rate_vols) * 100, grids[T][rho], "o-", label=rf"$\rho$ = {rho}")
    ax.axhline(float(hybrid.deterministic_rate_value(c, base)), color="k", ls="--", lw=1, label="deterministic rates")
    ax.set_xlabel(r"rate volatility $\sigma_r$ (%)"); ax.set_ylabel("guarantee value per 100")
    ax.set_title(f"{T:.0f}-year guarantee"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
for T, g in grids.items():
    print(f"{T:.0f}-year guarantee value by rate volatility (rows) and correlation (columns):")
    display(g.rename(index=lambda v: f"{v:.2%}"))
""")

code(r"""
rows = []
for T in [5.0, 10.0, 15.0, 20.0]:
    c = contract.with_(maturity=T)
    for rho in [-0.3, 0.0, 0.3]:
        m = HybridModel(hw_model, sigma=sigma, rho=rho)
        stoch = float(hybrid.gmmb_value(c, m))
        det = float(hybrid.deterministic_rate_value(c, m))
        rows.append({"maturity": T, "rho": rho, "stochastic_rates": stoch, "deterministic_rates": det,
                     "flat_3pct": float(bs.gmmb_value(c, sigma)),
                     "det_error": det - stoch, "det_error_pct": 100 * (det - stoch) / stoch})
det_table = pd.DataFrame(rows)
display(det_table)
""")

md(r"""
At $\rho = 0$, stochastic rates add variance to the forward index, so the
deterministic-rate value **understates** the guarantee, and the gap grows
with maturity (the extra variance $\sigma_r^2\int_0^T B(u)^2du$ grows like
$\sigma_r^2 T^3/3$ when $a$ is small). A
negative correlation offsets the bond-price variance and can make the
deterministic value an overstatement. The flat 3% rate understates rates
relative to the current curve, so it overstates the guarantee by much more
than either effect: the level of the curve matters first, its volatility next.

### 9.6 Hedging with the index and a zero-coupon bond

The value $V(t, S, P)$ depends on the index and on the price $P(t,T)$ of the
zero-coupon bond maturing with the guarantee, and it is homogeneous of degree
one in $(S, P)$: $V = S\,\partial_S V + P\,\partial_P V$. Holding $\partial_S V$
index units and $\partial_P V$ bonds replicates the guarantee, and this hedge
neutralises both equity and rate risk. The backtester now accrues cash at the
simulated short rate and discounts each path with its own $D(0,t)$. Three
hedgers, all charging the stochastic-rate value:

| strategy | instruments | hedge ratio |
|---|---|---|
| BS, deterministic rates | index + cash | Black-Scholes delta at the zero rate and the effective volatility |
| hybrid delta, index only | index + cash | $\partial_S V$ from the hybrid model, observing $r_t$ |
| hybrid delta, index + bond | index + bond + cash | $\partial_S V$ and $\partial_P V$ |

Paths: 10-year guarantee, 2,000 paths on a weekly grid, 4% equity risk
premium, fitted Hull-White parameters, $\rho = 0$. The baseline is the same
index-only Black-Scholes hedge when rates are deterministic ($\sigma_r = 0$,
same random numbers).
""")

code(r"""
T_h = 10.0
rate_contract = contract.with_(maturity=T_h)
market = HybridModel(hw_model, sigma=sigma, rho=0.0, equity_premium=0.04)
rate_paths = market.simulate(100.0, T_h, int(52 * T_h), n_hedge_paths, rng=61)
det_market = HybridModel(HullWhite(curve, a=a_fit, sigma=0.0), sigma=sigma, rho=0.0, equity_premium=0.04)
det_paths = det_market.simulate(100.0, T_h, int(52 * T_h), n_hedge_paths, rng=61)

r_zero = float(-np.log(curve.discount(T_h)) / T_h)
bs_contract = rate_contract.with_(rate=r_zero)
rate_hedgers = [BSHedger(bs_contract, float(market.effective_vol(T_h)), label="BS, deterministic rates"),
                HybridHedger(rate_contract, market, use_bond=False),
                HybridHedger(rate_contract, market, use_bond=True)]
rate_freqs = {"weekly": 1, "monthly": 4, "quarterly": 13}
rate_hedge = rebalancing_table(rate_paths, rate_contract, rate_hedgers, frequencies=rate_freqs,
                               cost_rates=(0.0, 0.001))
det_base = rebalancing_table(det_paths, rate_contract, [BSHedger(bs_contract, sigma, label="baseline: sigma_r = 0, BS")],
                             frequencies=rate_freqs, cost_rates=(0.0, 0.001))
rate_hedge_all = pd.concat([det_base, rate_hedge], ignore_index=True)
display(rate_hedge_all[["strategy", "frequency", "cost_rate", "price", "mean_pnl", "std_pnl", "cvar_95", "mean_costs"]])
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
weekly_res = {h.label: backtest(rate_paths, rate_contract, h, 1) for h in rate_hedgers}
for lab, res in weekly_res.items():
    axes[0].hist(res.pnl, bins=60, alpha=0.5, density=True, label=f"{lab} (sd {res.pnl.std():.2f})")
axes[0].set_xlabel("discounted hedge P&L per 100"); axes[0].legend(fontsize=8)
axes[0].set_title("Weekly hedging, stochastic rates")

r_change = rate_paths.short_rate[:, -1] - rate_paths.short_rate[:, 0]
axes[1].scatter(r_change, weekly_res["hybrid delta, index only"].pnl, s=4, alpha=0.4, label="index only")
axes[1].scatter(r_change, weekly_res["hybrid delta, index + bond"].pnl, s=4, alpha=0.4, label="index + bond")
axes[1].set_xlabel("change in short rate over the contract"); axes[1].set_ylabel("hedge P&L")
axes[1].legend(); axes[1].set_title("Index-only hedge error is driven by rates")
plt.tight_layout(); plt.show()

nc = rate_hedge_all[rate_hedge_all.cost_rate == 0].pivot(index="strategy", columns="frequency", values="std_pnl")
display(nc[list(rate_freqs)])
w = nc["weekly"]
print(f"weekly hedge-error sd: deterministic-rate baseline {w['baseline: sigma_r = 0, BS']:.3f}; with stochastic rates "
      f"BS {w['BS, deterministic rates']:.3f}, hybrid index-only {w['hybrid delta, index only']:.3f}, "
      f"index + bond {w['hybrid delta, index + bond']:.3f} "
      f"({w['hybrid delta, index only'] / w['hybrid delta, index + bond']:.1f}x lower than index only)")
""")

md(r"""
**Reading the hedging result.** With deterministic rates the weekly
Black-Scholes hedge leaves only discretisation error. Once rates are
stochastic, any index-only hedge leaves a large error that tracks the change
in rates over the contract (right panel): the guarantee is long duration,
and the index position does nothing about that. Using the correct hybrid
equity delta does not help (here it is slightly worse), because the missing piece is the bond position,
not the equity delta. Adding the zero-coupon bond brings the error back close
to the deterministic-rate baseline, and the remaining error again falls with
rebalancing frequency. Transaction costs on the bond are charged at the same
rate as on the index here, which is conservative for Treasuries.
""")

md(r"""
## 10. Summary

* Monte Carlo prices agree with the closed form within three standard errors
  on a grid of guarantee levels and maturities; error falls like $N^{-1/2}$.
* Antithetic and control variates together cut the variance per path by about
  an order of magnitude for the 10-year guarantee.
* Pathwise and finite-difference Greeks agree with each other and with the
  closed form.
* Under Black-Scholes the hedge-error spread falls roughly like the square
  root of the rebalancing interval; costs grow as rebalancing gets more frequent.
* Under Heston, delta hedging with either model leaves a much larger error;
  the minimum-variance delta reduces but does not remove it.
* The bootstrapped Treasury curve reprices every input par bond to machine
  precision with either interpolation; Hull-White fitted to it reproduces the
  curve by Monte Carlo, and Monte Carlo bond-option prices agree with the
  closed form and Jamshidian's decomposition within three standard errors.
* Stochastic rates change the value of long-dated guarantees materially: the
  deterministic-rate value is off by the amounts in section 9.5, and the
  sign depends on the equity-rate correlation.
* With stochastic rates an index-only hedge leaves the interest-rate risk
  open; adding the zero-coupon bond maturing with the guarantee removes most
  of the hedge error.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent / "report.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
