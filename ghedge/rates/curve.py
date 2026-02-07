"""Zero-coupon curve bootstrapped from par yields.

The curve stores continuously compounded zero rates ``r_i`` at knot
maturities ``t_i`` and interpolates ``y(t) = r(t) t = -ln P(0, t)`` with one
of two schemes:

``monotone_convex``
    Hagan and West (2006). The instantaneous forward curve is built from the
    discrete forwards between knots; it is continuous and local, preserves
    monotonicity of the discrete forwards and reproduces every knot exactly.
``cubic``
    Natural cubic spline through ``(0, 0)`` and ``(t_i, y_i)``; the forward
    curve is its derivative, so it is smooth but can oscillate.

Beyond the last knot the instantaneous forward is held flat.

The bootstrap solves for all knot rates at once so that every input
instrument prices to par. Treasury constant-maturity yields are par yields
on a semi-annual bond-equivalent basis: maturities up to six months are
treated as zero-coupon instruments with semi-annually compounded yields,
longer ones as semi-annual par bonds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root

METHODS = ("monotone_convex", "cubic")


def _monotone_convex_g(x, g0, g1):
    """Hagan-West forward adjustment ``g(x)`` and its integral ``G(x)`` on a
    unit interval, given the end-point adjustments ``g0, g1``. ``G(1) = 0``."""
    x, g0, g1 = np.broadcast_arrays(np.asarray(x, float), np.asarray(g0, float), np.asarray(g1, float))
    g = np.zeros_like(x)
    big = np.zeros_like(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        zone1 = ((g0 < 0) & (g1 >= -0.5 * g0) & (g1 <= -2 * g0)) | \
                ((g0 > 0) & (g1 <= -0.5 * g0) & (g1 >= -2 * g0))
        zone2 = ((g0 < 0) & (g1 > -2 * g0)) | ((g0 > 0) & (g1 < -2 * g0))
        zone3 = ((g0 > 0) & (g1 < 0) & (g1 > -0.5 * g0)) | ((g0 < 0) & (g1 > 0) & (g1 < -0.5 * g0))
        zero = (g0 == 0) & (g1 == 0)
        zone4 = ~(zone1 | zone2 | zone3 | zero)

        m = zone1
        g[m] = g0[m] * (1 - 4 * x[m] + 3 * x[m] ** 2) + g1[m] * (-2 * x[m] + 3 * x[m] ** 2)
        big[m] = g0[m] * (x[m] - 2 * x[m] ** 2 + x[m] ** 3) + g1[m] * (x[m] ** 3 - x[m] ** 2)

        m = zone2
        eta = (g1 + 2 * g0) / (g1 - g0)
        after = np.clip(x - eta, 0.0, None)
        g[m] = (g0 + (g1 - g0) * (after / (1 - eta)) ** 2)[m]
        big[m] = (g0 * x + (g1 - g0) * after**3 / (3 * (1 - eta) ** 2))[m]

        m = zone3
        eta = 3 * g1 / (g1 - g0)
        before = np.clip(eta - x, 0.0, None)
        g[m] = (g1 + (g0 - g1) * (before / eta) ** 2)[m]
        big[m] = (g1 * x + (g0 - g1) * (eta**3 - before**3) / (3 * eta**2))[m]

        m = zone4
        eta = g1 / (g1 + g0)
        a = -g0 * g1 / (g1 + g0)
        before = np.clip(eta - x, 0.0, None)
        after = np.clip(x - eta, 0.0, None)
        tail = np.where(after > 0, (g1 - a) * after**2 / np.where(eta < 1, (1 - eta) ** 2, 1.0), 0.0)
        g[m] = (a + np.where(x <= eta, (g0 - a) * (before / np.where(eta > 0, eta, 1.0)) ** 2, tail))[m]
        head = np.where(eta > 0, (g0 - a) * (eta**3 - before**3) / (3 * np.where(eta > 0, eta, 1.0) ** 2), 0.0)
        tail_int = np.where(after > 0, (g1 - a) * after**3 / (3 * np.where(eta < 1, (1 - eta) ** 2, 1.0)), 0.0)
        big[m] = (a * x + head + tail_int)[m]
    return g, big


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve with knot maturities ``times``
    (positive, increasing) and zero rates ``zeros``."""

    times: np.ndarray
    zeros: np.ndarray
    method: str = "monotone_convex"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        t = np.asarray(self.times, dtype=float)
        z = np.asarray(self.zeros, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or np.any(np.diff(t) <= 0) or t[0] <= 0:
            raise ValueError("times must be positive and increasing, one zero rate per time")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "zeros", z)
        knots = np.concatenate([[0.0], t])
        y = np.concatenate([[0.0], z * t])
        if self.method == "cubic":
            spline = CubicSpline(knots, y, bc_type="natural")
            object.__setattr__(self, "_spline", spline)
            f_end = float(spline(t[-1], 1))
        else:
            fd = np.diff(y) / np.diff(knots)
            n = fd.size
            f = np.empty(n + 1)
            if n > 1:
                h = np.diff(knots)
                f[1:n] = (h[:-1] * fd[1:] + h[1:] * fd[:-1]) / (h[:-1] + h[1:])
                f[0] = fd[0] - 0.5 * (f[1] - fd[0])
                f[n] = fd[-1] - 0.5 * (f[n - 1] - fd[-1])
            else:
                f[:] = fd[0]
            object.__setattr__(self, "_fd", fd)
            object.__setattr__(self, "_fknot", f)
            f_end = f[n]
        object.__setattr__(self, "_knots", knots)
        object.__setattr__(self, "_y", y)
        object.__setattr__(self, "_f_end", f_end)

    @classmethod
    def flat(cls, rate: float, method: str = "monotone_convex", horizon: float = 50.0) -> "ZeroCurve":
        times = np.array([1.0, horizon])
        return cls(times, np.full(2, float(rate)), method)

    # y(t) = -ln P(0, t) and its derivative, the instantaneous forward
    def _y_and_f(self, t):
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("times must be non-negative")
        knots, y = self._knots, self._y
        t_end = knots[-1]
        inside = np.minimum(t, t_end)
        if self.method == "cubic":
            yy = self._spline(inside)
            ff = self._spline(inside, 1)
        else:
            i = np.clip(np.searchsorted(knots, inside, side="left"), 1, knots.size - 1)
            h = knots[i] - knots[i - 1]
            x = (inside - knots[i - 1]) / h
            fd = self._fd[i - 1]
            g, big = _monotone_convex_g(x, self._fknot[i - 1] - fd, self._fknot[i] - fd)
            yy = y[i - 1] + h * (fd * x + big)
            ff = fd + g
        beyond = t > t_end
        yy = np.where(beyond, y[-1] + self._f_end * (t - t_end), yy)
        ff = np.where(beyond, self._f_end, ff)
        return yy, ff

    def discount(self, t):
        """``P(0, t)``."""
        return np.exp(-self._y_and_f(t)[0])

    def zero_rate(self, t):
        """Continuously compounded zero rate; the short end uses the forward at 0."""
        t = np.asarray(t, dtype=float)
        yy, ff = self._y_and_f(t)
        return np.where(t > 0, yy / np.where(t > 0, t, 1.0), ff)

    def forward(self, t):
        """Instantaneous forward rate ``f(0, t)``."""
        return self._y_and_f(t)[1]

    def forward_slope(self, t, h: float = 1e-4):
        """``df(0, t)/dt`` by central differences (one-sided at 0)."""
        t = np.asarray(t, dtype=float)
        lo = np.maximum(t - h, 0.0)
        return (self.forward(t + h) - self.forward(lo)) / (t + h - lo)

    def par_yield(self, maturity, freq: int = 2):
        """Par yield of a bond with coupons ``freq`` times a year (the same
        convention as the bootstrap instruments)."""
        times, _ = _cashflows(float(maturity), 1.0, freq)
        annuity = np.sum(self.discount(times)) / freq
        if maturity <= 1.0 / freq:
            return freq * (self.discount(maturity) ** (-1.0 / (freq * maturity)) - 1.0)
        return (1.0 - self.discount(maturity)) / annuity


def _cashflows(maturity: float, par_yield: float, freq: int = 2):
    """Cash-flow times and amounts of a par instrument per unit face.

    Maturities up to one coupon period are zero-coupon: one payment of
    ``(1 + y/freq)^(freq T)`` at ``T``. Longer maturities pay ``y/freq`` on a
    coupon schedule counted back from maturity, plus the face at maturity.
    Each instrument is worth exactly 1 on a correct curve.
    """
    if maturity <= 1.0 / freq + 1e-12:
        return np.array([maturity]), np.array([(1.0 + par_yield / freq) ** (freq * maturity)])
    n = int(np.floor(maturity * freq + 1e-9))
    times = maturity - np.arange(n)[::-1] / freq
    times = times[times > 1e-9]
    amounts = np.full(times.size, par_yield / freq)
    amounts[-1] += 1.0
    return times, amounts


def repricing_errors(curve: ZeroCurve, maturities, par_yields, freq: int = 2) -> np.ndarray:
    """Present value minus par (1.0) for each input instrument."""
    out = []
    for m, y in zip(maturities, par_yields):
        times, amounts = _cashflows(float(m), float(y), freq)
        out.append(float(amounts @ curve.discount(times)) - 1.0)
    return np.array(out)


def bootstrap(maturities, par_yields, method: str = "monotone_convex", freq: int = 2,
              tol: float = 1e-12) -> ZeroCurve:
    """Fit knot zero rates at the instrument maturities so that every par
    instrument reprices to 1.

    Because both interpolation schemes couple neighbouring segments, the knot
    rates are solved simultaneously (Powell hybrid method) rather than one
    at a time.
    """
    m = np.asarray(maturities, dtype=float)
    y = np.asarray(par_yields, dtype=float)
    order = np.argsort(m)
    m, y = m[order], y[order]
    guess = freq * np.log1p(y / freq)

    def residual(z):
        return repricing_errors(ZeroCurve(m, z, method), m, y, freq)

    sol = root(residual, guess, method="hybr", options={"xtol": 1e-15})
    curve = ZeroCurve(m, sol.x, method)
    err = np.max(np.abs(residual(sol.x)))
    if err > tol:
        raise RuntimeError(f"bootstrap did not converge (max repricing error {err:.2e})")
    return curve


def bootstrap_series(par, method: str = "monotone_convex", freq: int = 2) -> ZeroCurve:
    """Bootstrap from a pandas Series of par yields indexed by maturity."""
    return bootstrap(par.index.to_numpy(dtype=float), par.to_numpy(dtype=float), method, freq)
