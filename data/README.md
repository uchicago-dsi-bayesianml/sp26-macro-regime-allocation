# Data

The tables the notebooks read sit in the repository root, where the authors put them, so that every path in their code still works. This note says where the numbers come from, under what terms, what is and is not in this copy, and how to rebuild the one file that is missing.

## What is here

All of these are monthly, with month-end dates. Asset returns are log returns, log(P<sub>t</sub> / P<sub>t−1</sub>) of the adjusted close; the macro columns are z-scores, not levels.

- `macro_asset_dataset.csv`: 240 months, March 2006 to April 2026. The six ETF returns (`SPY_ret` … `VNQ_ret`) and the seven macro features standardized with the full-sample mean and standard deviation (`T10Y2Y_z`, `VIX_z`, `CREDIT_z`, `UNRATE_z`, `UNRATE_CHG_z`, `INDPRO_YOY_z`, `CFNAI_z`). Written by `Data Collection.ipynb`; read by cell 4 of `final_project_bayesian.ipynb` for the exploratory part and by `scripts/eda_overview.py`.
- `macro_asset_dataset_expanding_z.csv`: 217 months, February 2008 to April 2026. The same returns and the macro features standardized with an expanding window of at least 24 months (`*_z_exp`), so that each month's z-score uses only earlier and current values. Written by cell 25 of the notebook; every model in the notebook and every staff script starts from it.
- The result tables, each also carrying the returns and the expanding z-scores: `baseline_regime_results.csv` (Gaussian mixture, cell 34), `bayesian_hmm_results.csv` (Gibbs HMM, cell 44), `markov_switching_results.csv` (Markov-switching regression, cell 52), `regime_backtest_results.csv` (template strategy, cell 63) and `refined_regime_backtest_results.csv` (refined strategy, cell 72).
- `outputs/tables/`: summary statistics, missing counts and the correlation matrix of `macro_asset_dataset.csv`, written by `scripts/eda_overview.py`.

So the repository keeps derived values only: monthly returns of the six funds and standardized versions of the seven macro series, including `CREDIT_z` and `CREDIT_z_exp`, which are computed from Moody's yields, and the two VIX columns. It keeps no price, yield or index level.

**Two months are missing.** November and December 2025 are absent from both tables. The Bureau of Labor Statistics published no unemployment rate for October 2025 (FRED's series has a blank there), and with the one-month publication lag that blank lands on November, and the one-month change on November and December. Both rows are dropped, so every model and backtest steps from October 2025 straight to January 2026; the January return is still January's alone.

## What is not here

`macro_asset_raw.csv`, which the authors committed and course staff left out of this copy. It held 243 months, March 2006 to May 2026, of the six ETFs' adjusted closing prices, the seven macro features as levels, and the log returns. The terms of Yahoo Finance's prices and of the VIX and Moody's levels do not let us republish them (see the terms below). Cell 21 of `final_project_bayesian.ipynb` reads this file, so the notebook stops there until the file is rebuilt; the cells before it run on the committed tables, and the staff scripts do not need it.

## Where the data comes from

**ETF prices.** Yahoo Finance, through the `yfinance` package: daily closes adjusted for splits and dividends (`auto_adjust=True`), the last close of each month. SPY (S&P 500), EFA (MSCI EAFE developed markets), TLT (20+ year Treasuries), GLD (gold), DBC (a broad commodity index) and VNQ (US REITs). The start, March 2006, is DBC's first month with a computable monthly return (`Data Collection.ipynb`, cell 0).

**Macro series.** FRED, the Federal Reserve Bank of St. Louis's database. Daily series are taken at the last observation of each month; every macro column is then shifted one month later, as a publication lag.

| Column | FRED series | Source | Transformation | FRED's terms tag |
|---|---|---|---|---|
| `T10Y2Y` | [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y), daily | Federal Reserve Bank of St. Louis | 10-year minus 2-year Treasury yield, in percent | Copyrighted: Citation Required |
| `VIX` | [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS), daily | Chicago Board Options Exchange | close of the CBOE Volatility Index | Copyrighted: Citation Required |
| `CREDIT` | [BAA](https://fred.stlouisfed.org/series/BAA) and [AAA](https://fred.stlouisfed.org/series/AAA), monthly | Moody's | Baa minus Aaa seasoned corporate bond yield | Copyrighted: Citation Required |
| `UNRATE`, `UNRATE_CHG` | [UNRATE](https://fred.stlouisfed.org/series/UNRATE), monthly | U.S. Bureau of Labor Statistics | the rate, and its one-month change | Public Domain: Citation Requested |
| `INDPRO_YOY` | [INDPRO](https://fred.stlouisfed.org/series/INDPRO), monthly | Board of Governors of the Federal Reserve System | twelve-month percent change of the index | Public Domain: Citation Requested |
| `CFNAI` | [CFNAI](https://fred.stlouisfed.org/series/CFNAI), monthly | Federal Reserve Bank of Chicago | the Chicago Fed National Activity Index | Copyrighted: Citation Required |

The project proposal named the ISM manufacturing index; `Data Collection.ipynb` (cell 7) explains that ISM data left FRED in 2016 and uses the Chicago Fed index instead.

## Terms

- **Yahoo Finance.** Yahoo's terms of service allow personal, non-commercial use and do not allow redistribution of its data. `yfinance` "is not affiliated, endorsed, or vetted by Yahoo, Inc."; its documentation refers users to Yahoo's terms of use and adds that "the Yahoo! finance API is intended for personal use only." Do not commit downloaded prices.
- **FRED.** FRED's [terms of use](https://fred.stlouisfed.org/legal/), as of October 2026, say "You are free to access and use FRED® at no cost for your own personal use". They forbid providing "any stored, cached, or archived portion of the FRED® Services or FRED® Content to any third party", and using FRED's services or content in the development or training of software or machine-learning systems. Series with a copyright notice belong to third parties: they must be cited, and uses beyond personal ones need the owner's permission, which FRED cannot grant. The notes on Moody's BAA and AAA series say that Moody's information may not be copied, reproduced, repackaged or further transmitted without Moody's prior written consent; the VIX notes say the series is reprinted on FRED with CBOE's permission. The public-domain series ask for a citation. This copy keeps no FRED series, only the monthly z-scores the authors derived from them.
- **Citations**, in FRED's suggested form, for the series used here:
  - Federal Reserve Bank of St. Louis, 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity [T10Y2Y], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/T10Y2Y.
  - Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/VIXCLS.
  - Moody's, Moody's Seasoned Baa Corporate Bond Yield [BAA], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/BAA.
  - Moody's, Moody's Seasoned Aaa Corporate Bond Yield [AAA], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/AAA.
  - U.S. Bureau of Labor Statistics, Unemployment Rate [UNRATE], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/UNRATE.
  - Board of Governors of the Federal Reserve System (US), Industrial Production: Total Index [INDPRO], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/INDPRO.
  - Federal Reserve Bank of Chicago, Chicago Fed National Activity Index [CFNAI], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CFNAI.
  - `staff/check_walk_forward.py --rf` also downloads Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill Secondary Market Rate, Discount Basis [TB3MS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/TB3MS (public domain). It is used in memory and not saved.

## How to rebuild the raw file

**Without a key.** `staff/rebuild_raw_data.py` follows `Data Collection.ipynb` (cells 6 to 9) step by step but downloads the FRED series from FRED's public CSV links (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`) and the prices with `yfinance`. It keeps observations up to May 27, 2026, the day the authors committed, and months up to April 2026, the last month of their model tables. It writes `macro_asset_raw.csv` to the repository root, which `.gitignore` keeps out of commits, and does not touch the committed tables. It then rebuilds both derived tables in memory and reports how close they come to the committed ones.

```sh
pip install numpy pandas yfinance
python staff/rebuild_raw_data.py          # about 10 seconds; --force to overwrite
```

In the staff run on October 29, 2026, the rebuilt returns matched the committed ones to five decimals in every month, and so did the expanding z-scores of five of the seven macro features. Two columns differ because FRED has revised the series since May: `INDPRO_YOY_z_exp` by up to 0.094 and `CFNAI_z_exp` by up to 0.139, with correlations of 0.99998 and 0.99994 with the committed values. Neither column feeds a model. The full-sample z-scores in `macro_asset_dataset.csv` differ by up to 0.03 in the other five columns as well, because the authors' file had a partial May 2026 row that entered the full-sample mean and standard deviation; the script stops at April. Run the notebook after the script and it reproduces the committed result tables, up to the run-to-run variation of the Markov-switching fit described in the staff README.

**With the authors' notebook.** `Data Collection.ipynb` needs the `fredapi` package and a free FRED API key in the environment variable `FRED_API_KEY` (this copy no longer carries a default key, so the notebook stops with a message if it is unset). It writes both `macro_asset_raw.csv` and `macro_asset_dataset.csv`, so running it replaces the committed full-sample table with a fresh pull, through whatever month it is run in.
