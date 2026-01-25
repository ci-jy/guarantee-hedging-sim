import numpy as np
import pandas as pd
import pytest

from ghedge import GMMBContract
from ghedge.data import DEFAULT_CSV, load_index, parse_index_text
from ghedge.historical import historical_backtest, historical_windows, realised_vol


def test_parse_handles_missing_values():
    text = "observation_date,SP500\n2020-01-02,3257.85\n2020-01-03,.\n2020-01-06,3246.28\n2020-01-07,\n"
    s = parse_index_text(text)
    assert list(s.values) == [3257.85, 3246.28]
    assert isinstance(s.index, pd.DatetimeIndex)


def test_snapshot_loads():
    assert DEFAULT_CSV.exists()
    s = load_index()
    assert len(s) > 2000
    assert s.index.is_monotonic_increasing
    assert (s > 0).all()


def test_windows_are_rescaled_and_aligned():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=800)
    levels = pd.Series(1000 * np.exp(np.cumsum(0.01 * rng.standard_normal(800))), index=idx)
    paths, starts, issue_vol, realised = historical_windows(levels, 252, 21, 252)
    assert paths.index.shape[1] == 253
    np.testing.assert_allclose(paths.index[:, 0], 100.0)
    first = levels.index.get_loc(starts[0])
    np.testing.assert_allclose(paths.index[0, -1], 100 * levels.iloc[first + 252] / levels.iloc[first])
    assert np.allclose(realised, 0.01 * np.sqrt(252), rtol=0.2)
    assert issue_vol[0] == pytest.approx(realised_vol(levels).iloc[first])


def test_historical_backtest_on_snapshot():
    per_window, summary = historical_backtest(load_index(), GMMBContract(maturity=1.0, rate=0.02),
                                              cost_rate=0.001)
    assert len(per_window) == summary["n_contracts"].iloc[0] > 50
    assert set(summary["frequency"]) == {"daily", "weekly", "monthly"}
    assert np.isfinite(per_window.filter(like="pnl_").to_numpy()).all()
    assert (summary["cvar_95"] >= summary["var_95"]).all()


def test_fixed_volatility_option():
    _, summary = historical_backtest(load_index(), GMMBContract(maturity=1.0), sigma=0.18,
                                     frequencies={"weekly": 5})
    assert summary["mean_price"].iloc[0] > 0
