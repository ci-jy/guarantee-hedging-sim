import numpy as np
import pytest

from ghedge import Decrements, GMMBContract
from ghedge import blackscholes as bs


def test_put_call_parity():
    s, k, tau, r, q, sig = 95.0, 100.0, 3.0, 0.03, 0.02, 0.25
    lhs = bs.call_price(s, k, tau, r, q, sig) - bs.put_price(s, k, tau, r, q, sig)
    assert lhs == pytest.approx(s * np.exp(-q * tau) - k * np.exp(-r * tau), abs=1e-10)


def test_known_value():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5 -> put 0.81
    assert float(bs.put_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.2)) == pytest.approx(0.8086, abs=1e-4)


@pytest.mark.parametrize("s", [70.0, 100.0, 140.0])
def test_greeks_match_finite_differences(s):
    k, tau, r, q, sig, h = 100.0, 5.0, 0.03, 0.02, 0.2, 1e-3
    p = lambda x, v=sig: float(bs.put_price(x, k, tau, r, q, v))
    assert float(bs.put_delta(s, k, tau, r, q, sig)) == pytest.approx((p(s + h) - p(s - h)) / (2 * h), abs=1e-6)
    assert float(bs.put_gamma(s, k, tau, r, q, sig)) == pytest.approx(
        (p(s + 0.1) - 2 * p(s) + p(s - 0.1)) / 0.01, rel=1e-4)
    assert float(bs.put_vega(s, k, tau, r, q, sig)) == pytest.approx(
        (p(s, sig + 1e-5) - p(s, sig - 1e-5)) / 2e-5, rel=1e-6)


def test_expiry_limits():
    assert float(bs.put_price(90.0, 100.0, 0.0, 0.03, 0.02, 0.2)) == 10.0
    assert float(bs.put_delta(90.0, 100.0, 0.0, 0.03, 0.02, 0.2)) == -1.0
    assert float(bs.put_delta(110.0, 100.0, 0.0, 0.03, 0.02, 0.2)) == 0.0


def test_implied_vol_round_trip():
    price = float(bs.put_price(100.0, 110.0, 4.0, 0.03, 0.02, 0.27))
    assert bs.implied_vol_put(price, 100.0, 110.0, 4.0, 0.03, 0.02) == pytest.approx(0.27, abs=1e-8)


def test_gmmb_value_scales_with_survival():
    base = GMMBContract(decrements=Decrements.none())
    with_dec = GMMBContract()
    assert base.maturity_survival == 1.0
    assert 0.5 < with_dec.maturity_survival < 1.0
    ratio = float(bs.gmmb_value(with_dec, 0.2)) / float(bs.gmmb_value(base, 0.2))
    assert ratio == pytest.approx(with_dec.maturity_survival, rel=1e-12)


def test_gmmb_value_increases_with_fee():
    lo = float(bs.gmmb_value(GMMBContract(fee=0.0), 0.2))
    hi = float(bs.gmmb_value(GMMBContract(fee=0.03), 0.2))
    assert hi > lo


def test_gmmb_delta_matches_finite_difference_at_later_time(contract):
    t, s, h = 3.0, 120.0, 1e-3
    fd = (bs.gmmb_value(contract, 0.2, s + h, t) - bs.gmmb_value(contract, 0.2, s - h, t)) / (2 * h)
    assert float(bs.gmmb_delta(contract, 0.2, s, t)) == pytest.approx(float(fd), abs=1e-7)
