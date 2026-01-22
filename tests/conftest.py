import pytest

from ghedge import GBM, GMMBContract, Heston


@pytest.fixture
def contract():
    return GMMBContract(premium=100.0, guarantee=100.0, maturity=10.0, fee=0.02, rate=0.03)


@pytest.fixture
def gbm():
    return GBM(sigma=0.2)


@pytest.fixture
def heston_params():
    return Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
