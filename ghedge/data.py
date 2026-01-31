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


# --- Treasury constant-maturity par yields ------------------------------------

TREASURY_SERIES = ("DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7",
                   "DGS10", "DGS20", "DGS30")
TREASURY_CSV = DATA_DIR / "treasury_cmt_fred.csv"


def series_maturity(series: str) -> float:
    """Maturity in years encoded in a FRED CMT series id (``DGS3MO`` -> 0.25)."""
    code = series.removeprefix("DGS")
    if code.endswith("MO"):
        return int(code[:-2]) / 12.0
    return float(code)


def parse_treasury_csv(path_or_buffer) -> pd.DataFrame:
    """Parse a FRED multi-series CSV of CMT yields (percent) into a frame of
    decimal yields indexed by date, one column per series."""
    df = pd.read_csv(path_or_buffer)
    df = df.set_index(pd.to_datetime(df[df.columns[0]])).drop(columns=df.columns[0])
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df.dropna(how="all").sort_index()


def download_treasury(series=TREASURY_SERIES, start: str = "2016-10-01",
                      timeout: float = 120.0) -> str:
    """Download CMT yields from FRED as CSV text, keeping dates from ``start``."""
    text = download_fred(",".join(series), timeout)
    df = pd.read_csv(io.StringIO(text))
    df = df[pd.to_datetime(df[df.columns[0]]) >= pd.Timestamp(start)]
    return df.to_csv(index=False)


def load_treasury(path=None, download: bool = False) -> pd.DataFrame:
    """Load the CMT yield history, optionally refreshing it from FRED first.

    If the download fails the committed CSV snapshot is used instead.
    """
    path = Path(path) if path is not None else TREASURY_CSV
    if download:
        try:
            text = download_treasury()
            parsed = parse_treasury_csv(io.StringIO(text))
            if len(parsed) > 500:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return parsed
        except Exception as exc:  # network failure: fall back to the snapshot
            print(f"download failed ({exc}); using snapshot {path}")
    return parse_treasury_csv(path)


def par_curve(history: pd.DataFrame, date=None) -> pd.Series:
    """Par yields on ``date`` (default: the latest date with every tenor
    quoted), indexed by maturity in years."""
    complete = history.dropna()
    if complete.empty:
        raise ValueError("no date has a full set of tenors")
    if date is None:
        row = complete.iloc[-1]
    else:
        row = complete.loc[:pd.Timestamp(date)].iloc[-1]
    out = pd.Series(row.to_numpy(dtype=float), index=[series_maturity(c) for c in row.index],
                    name=row.name)
    out.index.name = "maturity"
    return out.sort_index()
