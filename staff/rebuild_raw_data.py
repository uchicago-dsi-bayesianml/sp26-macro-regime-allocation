#!/usr/bin/env python3
"""Course-staff script: rebuild macro_asset_raw.csv, which this copy leaves out.

The authors' raw file held Yahoo Finance adjusted ETF prices and the VIX and Moody's
BAA-AAA levels, whose terms do not let us republish them (see data/README.md). Cell 21
of final_project_bayesian.ipynb reads it, so the notebook needs it to run from the top.

This follows the logic of the authors' Data Collection.ipynb (cells 6 to 9) but needs no
API key: ETF prices come from yfinance and the seven FRED series from FRED's public
download, https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>. It writes the raw
file to the repository root, where the notebook expects it and where .gitignore keeps it
out of commits, and leaves the committed tables alone. It then rebuilds, in memory, the
two derived tables the notebook uses (macro_asset_dataset.csv as Data Collection writes
it, and macro_asset_dataset_expanding_z.csv as cell 22 writes it) and reports how close
they come to the committed ones. Yahoo revises adjusted prices as dividends accrue and
FRED revises monthly series, so expect close but not identical numbers.

The authors' file also had a row for May 2026, a partial month when they pulled the data.
Their model tables drop it, because the April value of the activity index was not yet
published. Today that value exists, so the row would survive and add a 218th month to
every fit. The rebuilt file therefore stops at April 2026 unless --through says otherwise.

Needs NumPy, pandas, yfinance and network access.
Usage:  python staff/rebuild_raw_data.py [--asof 2026-05-27] [--through 2026-04-30] [--out PATH] [--force]
"""
import argparse
import io
import pathlib
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

# As in Data Collection.ipynb, cells 6 and 7.
MACRO_PULL_START = "2003-01-01"
ETF_PULL_START = "2005-01-01"
FINAL_START = "2006-03-01"
MACRO_PUBLICATION_LAG = 1
FRED_SERIES = {"T10Y2Y": "T10Y2Y", "VIX": "VIXCLS", "BAA": "BAA", "AAA": "AAA",
               "UNRATE": "UNRATE", "INDPRO": "INDPRO", "CFNAI": "CFNAI"}
TICKERS = ["SPY", "EFA", "TLT", "GLD", "DBC", "VNQ"]
MACRO_FEATURES = ["T10Y2Y", "VIX", "CREDIT", "UNRATE", "UNRATE_CHG", "INDPRO_YOY", "CFNAI"]


def fetch_fred(asof):
    cols = {}
    for name, sid in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        with urllib.request.urlopen(url, timeout=60) as r:
            s = pd.read_csv(io.BytesIO(r.read()), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        s = s.loc[MACRO_PULL_START:asof].astype(float)
        cols[name] = s.resample("ME").last()
        print(f"  FRED {sid:7s} {s.notna().sum():5d} observations to {s.dropna().index.max():%Y-%m-%d}")
    return pd.DataFrame(cols)


def fetch_etfs(asof):
    import yfinance as yf
    end = (pd.Timestamp(asof) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # yfinance's end is exclusive
    with warnings.catch_warnings():           # yfinance is noisy under recent pandas
        warnings.simplefilter("ignore")
        data = yf.download(TICKERS, start=ETF_PULL_START, end=end, auto_adjust=True, progress=False)
    close = data["Close"][TICKERS]
    print(f"  Yahoo {', '.join(TICKERS)}: daily closes {close.index.min():%Y-%m-%d} to {close.index.max():%Y-%m-%d}")
    return close.resample("ME").last()


def build(macro, prices):
    """Data Collection.ipynb cell 9, build_dataset, unchanged in substance."""
    macro = macro.copy()
    macro["CREDIT"] = macro["BAA"] - macro["AAA"]
    macro["UNRATE_CHG"] = macro["UNRATE"].diff()
    macro["INDPRO_YOY"] = macro["INDPRO"].pct_change(12, fill_method=None) * 100
    macro = macro[MACRO_FEATURES].shift(MACRO_PUBLICATION_LAG)
    log_returns = np.log(prices / prices.shift(1))
    log_returns.columns = [f"{t}_ret" for t in log_returns.columns]
    raw = prices.join(macro, how="outer").join(log_returns, how="outer")
    raw = raw.loc[raw.index >= FINAL_START]
    m = macro.loc[macro.index >= FINAL_START]
    macro_z = (m - m.mean()) / m.std()
    macro_z.columns = [f"{c}_z" for c in macro_z.columns]
    model = log_returns.loc[log_returns.index >= FINAL_START].join(macro_z, how="inner").dropna()
    return raw, model


def expanding_table(raw):
    """final_project_bayesian.ipynb cells 21 and 22."""
    x = raw[MACRO_FEATURES]
    z = (x - x.expanding(min_periods=24).mean()) / x.expanding(min_periods=24).std().replace(0, np.nan)
    z.columns = [f"{c}_z_exp" for c in MACRO_FEATURES]
    return raw[[f"{t}_ret" for t in TICKERS]].join(z, how="inner").dropna()


def compare(name, rebuilt):
    committed = pd.read_csv(ROOT / name, index_col=0, parse_dates=True)
    common = committed.index.intersection(rebuilt.index)
    only_c = committed.index.difference(rebuilt.index)
    only_r = rebuilt.index.difference(committed.index)
    print(f"\n{name}: {len(committed)} committed rows, {len(rebuilt)} rebuilt, {len(common)} in common"
          + (f"; only committed: {[f'{d:%Y-%m}' for d in only_c]}" if len(only_c) else "")
          + (f"; only rebuilt: {[f'{d:%Y-%m}' for d in only_r]}" if len(only_r) else ""))
    a, b = committed.loc[common], rebuilt.loc[common, committed.columns]
    diff = (a - b).abs()
    print(pd.DataFrame({"max_abs_diff": diff.max(), "mean_abs_diff": diff.mean(),
                        "correlation": [a[c].corr(b[c]) for c in a.columns]}).to_string(float_format="%.5f"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default="2026-05-27",
                    help="drop observations after this date (default: the authors' last commit)")
    ap.add_argument("--through", default="2026-04-30",
                    help="last month-end to keep (default: the last month of the authors' model tables)")
    ap.add_argument("--out", default=str(ROOT / "macro_asset_raw.csv"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to overwrite it")
    try:
        print("Fetching:")
        macro, prices = fetch_fred(args.asof), fetch_etfs(args.asof)
    except Exception as exc:  # network or service failure: say so, leave the instructions in data/README.md
        sys.exit(f"Download failed ({exc!r}). See data/README.md for the sources and steps.")
    raw, model = build(macro.loc[: args.through], prices.loc[: args.through])
    raw.to_csv(out)
    print(f"Wrote {out}: {len(raw)} rows, {raw.index.min():%Y-%m} to {raw.index.max():%Y-%m}")
    compare("macro_asset_dataset.csv", model)
    compare("macro_asset_dataset_expanding_z.csv", expanding_table(raw))


if __name__ == "__main__":
    main()
