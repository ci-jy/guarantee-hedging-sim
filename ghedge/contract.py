"""Guaranteed minimum maturity benefit (GMMB) contract and policyholder decrements."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Decrements:
    """Deterministic lapse and mortality decrements.

    Mortality follows a Gompertz-Makeham force of mortality
    ``mu(x) = A + B * c**x`` and lapses occur at a constant force ``lapse``.
    Both are assumed independent of the market, so the expected fraction
    of the initial cohort still in force at time ``t`` is a deterministic
    function of ``t``.
    """

    age: float = 55.0
    lapse: float = 0.03
    makeham_a: float = 0.00022
    makeham_b: float = 2.7e-6
    makeham_c: float = 1.124

    def mortality_force(self, t):
        t = np.asarray(t, dtype=float)
        return self.makeham_a + self.makeham_b * self.makeham_c ** (self.age + t)

    def survival(self, t):
        """Probability that a policy issued at time 0 is still in force at ``t``."""
        t = np.asarray(t, dtype=float)
        log_c = np.log(self.makeham_c)
        gompertz = self.makeham_b * self.makeham_c**self.age * (self.makeham_c**t - 1.0) / log_c
        return np.exp(-(self.lapse + self.makeham_a) * t - gompertz)

    @classmethod
    def none(cls) -> "Decrements":
        """No lapses and no deaths: every policy reaches maturity."""
        return cls(lapse=0.0, makeham_a=0.0, makeham_b=0.0)


@dataclass(frozen=True)
class GMMBContract:
    """A single-premium contract with a guaranteed minimum maturity benefit.

    The policyholder's fund tracks an equity index ``S`` less a continuous
    management fee ``fee``: ``F_t = S_t * exp(-fee * t)`` with ``F_0 = premium``
    when ``S_0 = premium``. At maturity ``T`` each surviving policy receives
    ``max(F_T, guarantee)``; the insurer's liability is the shortfall
    ``max(guarantee - F_T, 0)``, i.e. a put on the fund.
    """

    premium: float = 100.0
    guarantee: float = 100.0
    maturity: float = 10.0
    fee: float = 0.02
    rate: float = 0.03
    decrements: Decrements = field(default_factory=Decrements)

    def with_(self, **kwargs) -> "GMMBContract":
        return replace(self, **kwargs)

    @property
    def maturity_survival(self) -> float:
        """Fraction of the initial cohort that reaches maturity."""
        return float(self.decrements.survival(self.maturity))

    def fund(self, index, t):
        """Fund value per unit premium when the index is at ``index`` at time ``t``.

        The index is normalised so that ``index == premium`` at time 0.
        """
        return np.asarray(index) * np.exp(-self.fee * np.asarray(t))

    def payoff(self, terminal_index):
        """Cohort guarantee payoff at maturity per initial policy."""
        fund_t = self.fund(terminal_index, self.maturity)
        return self.maturity_survival * np.maximum(self.guarantee - fund_t, 0.0)
