"""Download daily U.S. Treasury constant-maturity yields (DGS1MO .. DGS30)
from FRED into data/treasury_cmt_fred.csv.

Usage: python3 scripts/fetch_treasury.py

The committed CSV is used as a fallback snapshot when the download fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghedge.data import TREASURY_CSV, load_treasury, par_curve  # noqa: E402

if __name__ == "__main__":
    history = load_treasury(download=True)
    curve = par_curve(history)
    print(f"{len(history)} dates {history.index[0].date()} .. {history.index[-1].date()} -> {TREASURY_CSV}")
    print(f"latest full curve ({curve.name.date()}):")
    print(curve.to_string())
