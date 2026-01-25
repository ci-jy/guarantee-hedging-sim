import numpy as np
import pytest

from ghedge import GBM, GMMBContract
from ghedge import blackscholes as bs
from ghedge import heston
from ghedge.greeks import fd_delta, fd_gamma, fd_vega, pathwise_delta, pathwise_vega


def _agree(a, b, k=3.0):
    return abs(a.estimate - b.estimate) < k * np.hypot(a.std_error, b.std_error)


@pytest.mark.parametrize("guarantee,maturity", [(80.0, 5.0), (100.0, 10.0), (120.0, 2.0)])
def test_pathwise_and_fd_delta_agree_with_each_other_and_closed_form(guarantee, maturity):
    c = GMMBContract(guarantee=guarantee, maturity=maturity)
    pw = pathwise_delta(c, GBM(0.2), 200_000, seed=1)
    fd = fd_delta(c, GBM(0.2), 200_000, seed=1)
    exact = float(bs.gmmb_delta(c, 0.2))
    assert _agree(pw, fd)
    assert abs(pw.estimate - exact) < 3 * pw.std_error
    assert abs(fd.estimate - exact) < 3 * fd.std_error
    # Same random numbers: the two estimators differ by far less than their noise.
    assert abs(pw.estimate - fd.estimate) < 0.25 * pw.std_error


def test_pathwise_and_fd_vega_agree(contract):
    pw = pathwise_vega(contract, GBM(0.2), 200_000, seed=2)
    fd = fd_vega(contract, GBM(0.2), 200_000, seed=2)
    exact = float(bs.gmmb_vega(contract, 0.2))
    assert _agree(pw, fd)
    assert abs(pw.estimate - exact) < 3 * pw.std_error


def test_fd_gamma_matches_closed_form(contract):
    g = fd_gamma(contract, GBM(0.2), 400_000, seed=3)
    assert abs(g.estimate - float(bs.gmmb_gamma(contract, 0.2))) < 3 * g.std_error


def test_heston_pathwise_and_fd_delta_agree(heston_params):
    c = GMMBContract(maturity=2.0)
    pw = pathwise_delta(c, heston_params, 50_000, seed=4, steps_per_year=100)
    fd = fd_delta(c, heston_params, 50_000, seed=4, steps_per_year=100)
    exact = float(heston.gmmb_value_and_greeks(c, heston_params)[1][0])
    assert _agree(pw, fd)
    # Allow for the Euler discretisation bias as well as noise.
    assert abs(pw.estimate - exact) < 3 * pw.std_error + 0.003


def test_real_world_drift_is_ignored_for_greeks(contract):
    a = pathwise_delta(contract, GBM(0.2, mu=0.08), 10_000, seed=5)
    b = pathwise_delta(contract, GBM(0.2), 10_000, seed=5)
    assert a.estimate == b.estimate


def test_heston_fd_vega_positive(heston_params):
    c = GMMBContract(maturity=2.0)
    v = fd_vega(c, heston_params, 20_000, seed=6, steps_per_year=50)
    assert v.estimate > 0
