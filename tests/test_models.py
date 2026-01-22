import numpy as np
import pytest

from ghedge import GBM, Decrements, Heston
from ghedge.models import pair_means


def test_decrements_survival():
    d = Decrements()
    t = np.linspace(0, 30, 31)
    s = d.survival(t)
    assert s[0] == 1.0
    assert np.all(np.diff(s) < 0)
    # Survival is exp(-integral of total force); check numerically.
    grid = np.linspace(0, 10, 20001)
    force = d.lapse + d.mortality_force(grid)
    assert float(d.survival(10.0)) == pytest.approx(np.exp(-np.trapezoid(force, grid)), rel=1e-8)
    assert float(Decrements.none().survival(25.0)) == 1.0


def test_gbm_terminal_moments():
    m = GBM(0.25)
    s_t, w = m.simulate_terminal(100.0, 0.03, 2.0, 400_000, rng=1)
    disc = np.exp(-0.03 * 2.0) * s_t
    se = disc.std() / np.sqrt(disc.size)
    assert abs(disc.mean() - 100.0) < 3 * se
    assert np.log(s_t).std() == pytest.approx(0.25 * np.sqrt(2.0), rel=0.01)
    assert w.var() == pytest.approx(2.0, rel=0.01)


def test_gbm_paths_match_terminal_distribution():
    m = GBM(0.2, mu=0.07)
    p = m.simulate(100.0, 0.03, 1.0, 50, 100_000, rng=2)
    assert p.index.shape == (100_000, 51)
    assert np.all(p.index[:, 0] == 100.0)
    log_ret = np.log(p.terminal / 100.0)
    assert log_ret.mean() == pytest.approx(0.07 - 0.02, abs=4 * 0.2 / np.sqrt(100_000))
    # The stored Brownian terminal value reproduces the terminal index.
    np.testing.assert_allclose(p.terminal, 100 * np.exp((0.07 - 0.02) + 0.2 * p.brownian), rtol=1e-10)


def test_antithetic_pairs_mirror():
    s_t, w = GBM(0.2).simulate_terminal(100.0, 0.03, 1.0, 1000, rng=3, antithetic=True)
    np.testing.assert_allclose(pair_means(w), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        GBM(0.2).simulate_terminal(100.0, 0.03, 1.0, 999, rng=3, antithetic=True)


def test_heston_martingale_and_variance_mean(heston_params):
    T, r = 2.0, 0.03
    p = heston_params.simulate(100.0, r, T, 400, 100_000, rng=4)
    disc = np.exp(-r * T) * p.terminal
    se = disc.std() / np.sqrt(disc.size)
    assert abs(disc.mean() - 100.0) < 3 * se
    assert np.all(p.variance >= 0.0)
    expected_v = heston_params.theta + (heston_params.v0 - heston_params.theta) * np.exp(-heston_params.kappa * T)
    assert p.variance[:, -1].mean() == pytest.approx(expected_v, rel=0.03)


def test_heston_variance_mean_reverts_from_low_start():
    h = Heston(v0=0.01, kappa=2.0, theta=0.06, xi=0.3, rho=-0.5)
    p = h.simulate(100.0, 0.03, 1.0, 250, 50_000, rng=5)
    expected = h.theta + (h.v0 - h.theta) * np.exp(-h.kappa)
    assert p.variance[:, -1].mean() == pytest.approx(expected, rel=0.03)


def test_heston_store_paths_false_matches_terminal(heston_params):
    a = heston_params.simulate(100.0, 0.03, 1.0, 100, 1000, rng=6)
    b = heston_params.simulate(100.0, 0.03, 1.0, 100, 1000, rng=6, store_paths=False)
    np.testing.assert_allclose(a.terminal, b.terminal)
    np.testing.assert_allclose(a.brownian, b.brownian)


def test_mean_variance_formula(heston_params):
    h = Heston(v0=0.09, kappa=1.0, theta=0.04, xi=0.4, rho=-0.5)
    grid = np.linspace(0, 5, 100001)
    ev = h.theta + (h.v0 - h.theta) * np.exp(-h.kappa * grid)
    assert h.mean_variance(5.0) == pytest.approx(np.trapezoid(ev, grid) / 5.0, rel=1e-8)
