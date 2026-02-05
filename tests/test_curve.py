import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from ghedge.data import (TREASURY_CSV, load_treasury, par_curve, parse_treasury_csv,
                         series_maturity)
from ghedge.rates import ZeroCurve, bootstrap, bootstrap_series, repricing_errors
from ghedge.rates.curve import METHODS, _cashflows, _monotone_convex_g

# Normal upward curve, near-zero rates (March 2020) and an inverted curve (July 2023).
DATES = ["2026-10-01", "2020-03-20", "2023-07-03", "2019-08-28"]


def test_series_maturity():
    assert series_maturity("DGS1MO") == pytest.approx(1 / 12)
    assert series_maturity("DGS6MO") == 0.5
    assert series_maturity("DGS30") == 30.0


def test_parse_treasury_handles_missing_values():
    text = "observation_date,DGS1,DGS10\n2020-01-02,1.56,1.88\n2020-01-03,.,.\n2020-01-06,1.54,\n"
    df = parse_treasury_csv(__import__("io").StringIO(text))
    assert list(df.index) == list(pd.to_datetime(["2020-01-02", "2020-01-06"]))
    assert df.loc["2020-01-02", "DGS10"] == pytest.approx(0.0188)
    assert np.isnan(df.loc["2020-01-06", "DGS10"])


def test_treasury_snapshot_and_par_curve():
    assert TREASURY_CSV.exists()
    history = load_treasury()
    assert len(history) > 2000 and history.index.is_monotonic_increasing
    curve = par_curve(history)
    assert list(curve.index) == sorted(curve.index)
    assert curve.index[0] == pytest.approx(1 / 12) and curve.index[-1] == 30.0
    assert ((curve > -0.01) & (curve < 0.2)).all()
    # A date without a full set of tenors falls back to the previous complete one.
    assert par_curve(history, "2026-10-04").name <= pd.Timestamp("2026-10-04")


def test_cashflows():
    t, a = _cashflows(0.25, 0.04)
    assert t.tolist() == [0.25] and a[0] == pytest.approx(1.02**0.5)
    t, a = _cashflows(2.0, 0.05)
    np.testing.assert_allclose(t, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(a, [0.025, 0.025, 0.025, 1.025])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("date", DATES)
def test_bootstrap_reprices_treasury_par_bonds(method, date):
    par = par_curve(load_treasury(), date)
    curve = bootstrap_series(par, method)
    err = repricing_errors(curve, par.index, par.values)
    assert np.max(np.abs(err)) < 1e-6
    # Par yields recomputed from the curve match the inputs.
    fitted = [float(curve.par_yield(m)) for m in par.index]
    np.testing.assert_allclose(fitted, par.values, atol=1e-8)
    np.testing.assert_allclose(curve.zero_rate(curve.times), curve.zeros, atol=1e-13)


@pytest.mark.parametrize("method", METHODS)
def test_forward_curve_integrates_to_discount_factor(method, treasury_par):
    curve = bootstrap_series(treasury_par, method)
    for t in [0.3, 2.5, 8.0, 15.0, 25.0, 40.0]:
        pts = [k for k in curve.times if k < t]
        integral = quad(lambda s: float(curve.forward(s)), 0.0, t, points=pts or None, limit=400)[0]
        assert integral == pytest.approx(-np.log(float(curve.discount(t))), abs=1e-8)


def test_monotone_convex_adjustment_properties():
    rng = np.random.default_rng(0)
    g0, g1 = rng.standard_normal(2000), rng.standard_normal(2000)
    g, big = _monotone_convex_g(1.0, g0, g1)
    np.testing.assert_allclose(big, 0.0, atol=1e-14)  # each segment reprices its discrete forward
    np.testing.assert_allclose(_monotone_convex_g(0.0, g0, g1)[0], g0, atol=1e-12)
    np.testing.assert_allclose(g, g1, atol=1e-12)
    for x in [0.1, 0.37, 0.8]:
        h = 1e-6
        deriv = (_monotone_convex_g(x + h, g0, g1)[1] - _monotone_convex_g(x - h, g0, g1)[1]) / (2 * h)
        np.testing.assert_allclose(deriv, _monotone_convex_g(x, g0, g1)[0], atol=1e-7)


def test_monotone_convex_forward_is_continuous_and_bounded(treasury_par):
    curve = bootstrap_series(treasury_par, "monotone_convex")
    for k in curve.times[:-1]:
        assert float(curve.forward(k - 1e-9)) == pytest.approx(float(curve.forward(k + 1e-9)), abs=1e-6)
    # Positive discrete forwards give a positive forward curve.
    t = np.linspace(0, 40, 4001)
    assert np.all(curve.forward(t) > 0)


@pytest.mark.parametrize("method", METHODS)
def test_flat_curve(method):
    curve = ZeroCurve.flat(0.03, method)
    t = np.array([0.0, 0.5, 3.0, 20.0, 60.0])
    np.testing.assert_allclose(curve.forward(t), 0.03, atol=1e-12)
    np.testing.assert_allclose(curve.discount(t), np.exp(-0.03 * t), rtol=1e-12)
    np.testing.assert_allclose(curve.forward_slope(t), 0.0, atol=1e-8)


def test_bootstrap_methods_agree_at_knots_but_differ_between(treasury_par):
    mc = bootstrap_series(treasury_par, "monotone_convex")
    cs = bootstrap_series(treasury_par, "cubic")
    np.testing.assert_allclose(mc.discount(mc.times[:8]), cs.discount(cs.times[:8]), rtol=1e-3)
    assert np.max(np.abs(mc.forward(np.linspace(10, 30, 50)) - cs.forward(np.linspace(10, 30, 50)))) > 1e-4


def test_invalid_curve_inputs():
    with pytest.raises(ValueError):
        ZeroCurve(np.array([1.0, 0.5]), np.array([0.01, 0.02]))
    with pytest.raises(ValueError):
        ZeroCurve(np.array([1.0]), np.array([0.01]), method="linear")
    with pytest.raises(ValueError):
        ZeroCurve.flat(0.02).discount(-1.0)
    curve = bootstrap([1.0, 5.0], [0.03, 0.04])
    assert np.max(np.abs(repricing_errors(curve, [1.0, 5.0], [0.03, 0.04]))) < 1e-12
