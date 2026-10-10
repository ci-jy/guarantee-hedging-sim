# Data

Both files come from FRED (Federal Reserve Bank of St. Louis). Tests and the report read the committed snapshots, so they run offline.

`data/sp500_fred.csv` holds S&P 500 daily closing levels (series `SP500`), about ten years, which is all FRED publishes for this series. It is a price index, so dividends are not included. Refresh it with:

```bash
python3 scripts/fetch_index.py
```

`data/treasury_cmt_fred.csv` holds daily U.S. Treasury constant-maturity yields in percent, series `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10`, `DGS20` and `DGS30`, over the same span. Refresh it with:

```bash
python3 scripts/fetch_treasury.py
```

Both scripts fall back to the committed file when the download fails. FRED marks missing values with `.`, which the loader handles.
