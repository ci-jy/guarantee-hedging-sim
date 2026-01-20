"""Download daily S&P 500 levels from FRED into data/sp500_fred.csv.

Usage: python3 scripts/fetch_index.py [SERIES_ID]

The committed CSV is used as a fallback snapshot when the download fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghedge.data import DEFAULT_CSV, load_index  # noqa: E402

if __name__ == "__main__":
    series_id = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    target = DEFAULT_CSV if series_id == "SP500" else DEFAULT_CSV.with_name(f"{series_id.lower()}_fred.csv")
    levels = load_index(target, download=True, series=series_id)
    print(f"{series_id}: {len(levels)} observations {levels.index[0].date()} .. {levels.index[-1].date()} -> {target}")
