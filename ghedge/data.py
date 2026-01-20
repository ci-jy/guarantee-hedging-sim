"""Loading and downloading public daily equity index levels."""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "sp500_fred.csv"


def parse_index_text(text: str) -> pd.Series:
    """Parse CSV text; see :func:`parse_index_csv`."""
    return parse_index_csv(io.StringIO(text))


def parse_index_csv(path_or_buffer) -> pd.Series:
    """Parse a two-column (date, level) CSV into a float series indexed by date.

    FRED marks holidays with ``.`` or leaves them blank; those rows are dropped.
    """
    df = pd.read_csv(path_or_buffer)
    date_col, value_col = df.columns[0], df.columns[1]
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.to_datetime(df[date_col]), name=value_col)
    series = series.dropna()
    series = series[series > 0].sort_index()
    return series[~series.index.duplicated(keep="last")]


def download_fred(series: str = "SP500", timeout: float = 120.0) -> str:
    """Download a FRED daily series as CSV text."""
    with urllib.request.urlopen(FRED_URL.format(series=series), timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def load_index(path=None, download: bool = False, series: str = "SP500") -> pd.Series:
    """Load index levels, optionally refreshing from FRED first.

    If the download fails the committed CSV snapshot is used instead.
    """
    path = Path(path) if path is not None else DEFAULT_CSV
    if download:
        try:
            text = download_fred(series)
            parsed = parse_index_text(text)
            if len(parsed) > 500:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return parsed
        except Exception as exc:  # network failure: fall back to the snapshot
            print(f"download failed ({exc}); using snapshot {path}")
    return parse_index_csv(path)
