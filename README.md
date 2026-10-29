<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayesian Regime Inference for Macro-Driven Asset Allocation

> **Example project from Spring 2026.** This is a copy of [bosen999/Bayesian-Machine-Learning](https://github.com/bosen999/Bayesian-Machine-Learning) at commit dcc1b18, without its commit history, because the first commit carries a FRED API key and a raw data file whose terms do not let us republish it. The authors' own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project) and three short scripts in `staff/`. The prices come from Yahoo Finance and the macro series from FRED, several of them copyrighted; this copy keeps only the returns and z-scores derived from them, and [data/README.md](data/README.md) explains the sources, their terms and how to rebuild the raw file.

Markets seem to move between calm stretches and stressed ones, and a portfolio that suits one is wrong for the other. Can a model that infers the current regime allocate a portfolio of exchange-traded funds better than simple benchmarks? Team GANgsters took monthly returns of six funds (US and international stocks, long Treasuries, gold, commodities and real estate) and seven macroeconomic indicators from FRED, 2006 to April 2026. They wrote a Gibbs sampler with forward filtering and backward sampling for a two-state Bayesian hidden Markov model of the equal-weight portfolio's return, and fit a two-state Markov-switching regression with statsmodels, which they make their main model. The regression's filtered probability of the low-return regime drives two allocation rules, a blend of two hand-set portfolios and a "refined" rule that builds a mean-variance portfolio for each regime. Backtested from 2011, the refined rule beats an equal-weight portfolio (Sharpe ratio 0.72 against 0.60) and loses to a 60/40 mix of stocks and Treasuries (0.87), which the write-up reports as an honest result.

## Summary

**Data.** Monthly log returns of SPY, EFA, TLT, GLD, DBC and VNQ from Yahoo Finance's adjusted closes; and from FRED the 10-year minus 2-year Treasury spread, the VIX, the Baa minus Aaa corporate bond spread, unemployment and its monthly change, industrial production growth and the Chicago Fed National Activity Index, each lagged one month for publication. The macro features are z-scored with an expanding window, which needs 24 months of history, so the modelling table runs from February 2008 to April 2026, 217 months. It lacks November and December 2025, because no unemployment rate was published for October 2025.

**Models.** A Gaussian mixture on the six returns is the baseline. The Bayesian HMM and the Markov-switching regression are both fit to one series, the equal-weight average of the six log returns, with two states that differ in mean and variance: the HMM by a Gibbs sampler written from scratch, the regression by maximum likelihood. No model reads the macro variables; they are averaged within each fitted regime afterwards, to describe it.

**Allocation and backtest.** The template rule mixes a risk-on and a risk-off portfolio by the regression's filtered probability of the low-return regime. The refined rule weights past months by their regime probabilities, computes a long-only, capped mean-variance portfolio for each regime, and mixes the two the same way. Weights chosen at the end of a month earn the next month's returns. The benchmarks, equal weight and 60/40, are rebalanced monthly; nothing pays trading costs.

**Results as reported.** From February 2011 the refined rule earns 7.51 percent a year, against 5.93 for the template rule, 6.08 for equal weight and 8.78 for 60/40. The write-up concludes that regime models "identify economically meaningful market states", that "those states can improve dynamic allocation decisions", and that 60/40 "remains difficult to beat".

The staff section below shows that the code reproduces these numbers, that the Bayesian model never reaches the trades, that a real-time refit changes little, that the same optimizer without regimes does as well, and that the returns are compounded the wrong way.

## Team and contributions

Team "GANgsters", Spring 2026: Bosen Yang, Jingnan Chi, Kevin Long and Mike Peng.

- **Bosen Yang** ([@bosen999](https://github.com/bosen999)): both commits, on May 27, 2026. The first, made as "Bosen", added every file: both notebooks, the EDA script, the planning note, all tables and figures, the proposal as a Pages file, and the write-up. The second, seven minutes later in GitHub's web editor, added the team list to the write-up.
- **Jingnan Chi**: no commits. Jingnan Chi submitted the team's three Canvas milestones: the topic proposal on April 19, the data and methodology document on May 3, and the repository link on May 27.
- **Kevin Long** and **Mike Peng**: no commits.

The whole project arrived in one commit, so the history records who uploaded it, not who wrote what. The data notebook's saved outputs and the write-up's links come from two different user folders, so more than one computer was involved, and members without commits may have contributed outside the repository. The attributions follow the authors' own [commit history](https://github.com/bosen999/Bayesian-Machine-Learning/commits/main), since this copy has none.

## Getting started

The authors gave no requirements; `requirements.txt` pins the versions course staff used on Python 3.12.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
python staff/rebuild_raw_data.py      # writes macro_asset_raw.csv, which cell 21 needs
```

Cell numbers in this README count every cell of a notebook, markdown included, from 0 at the top.

**The pipeline.**

1. `Data Collection.ipynb` downloads the data with `fredapi` and `yfinance` and writes `macro_asset_raw.csv` and `macro_asset_dataset.csv`. It needs a FRED API key and replaces the committed table with a fresh pull; `staff/rebuild_raw_data.py` writes only the raw file and needs no key.
2. `final_project_bayesian.ipynb`, top to bottom, about 30 seconds: exploration of the full-sample table (cells 4 to 17), expanding z-scores from the raw file (21 to 25), the Gaussian mixture (28 to 34), the Gibbs HMM (37 to 44), the Markov-switching regression (47 to 52), the template backtest (55 to 63) and the refined backtest (66 to 72). Each stage writes its table to the repository root.
3. `scripts/eda_overview.py` redraws five exploratory figures and three tables in `outputs/` from `macro_asset_dataset.csv`.

Course staff ran the notebook top to bottom, with the authors' original raw file and with the rebuilt one, and it reproduces the committed result tables.

**Staff checks.** Both read only the committed CSV files and reuse the authors' functions from the unchanged notebook (`staff/common.py` checks its hash first). `check_walk_forward.py` refits the Markov-switching regression on the full sample, matching the committed probabilities to within 3 × 10⁻⁵, then every month on the data to date; adds the refined optimizer without regimes as a control; and rescores every rule with simple returns, a trading cost and the bill rate. `check_bayesian_hmm_realtime.py` reruns the Gibbs sampler, matching the committed probabilities exactly, refits it yearly on the data to date, and trades on its filtered probabilities.

```sh
python staff/check_walk_forward.py --rf        # under a minute; --rf downloads the bill rate
python staff/check_bayesian_hmm_realtime.py    # about two minutes
```

**What does not run as committed.** Cell 21 reads `macro_asset_raw.csv`, which this copy leaves out; the cells before it run on the committed tables, and `staff/rebuild_raw_data.py` restores the file. `Data Collection.ipynb` now stops with a message unless `FRED_API_KEY` is set. The write-up's thirteen file links point to `/Users/bosen/Desktop/…`, and it describes `01_Project_Roadmap_and_EDA.ipynb`, which is not in the repository; the analysis notebook is `final_project_bayesian.ipynb`, committed with no outputs, so its numbers appear only in its markdown. The three figures in the write-up (`outputs/figures/readme_*.png`) were made by code that is not in the repository. Cell 47 seeds NumPy's global generator, which statsmodels 0.15 no longer uses for its random starting values, so the Markov-switching fit can move in the fourth decimal between runs.

## Data

[data/README.md](data/README.md) has the details, the terms and the FRED citations.

- **Sources and terms:** Yahoo Finance through `yfinance`, for personal use only and not for redistribution; FRED series T10Y2Y, VIXCLS, BAA, AAA, UNRATE, INDPRO and CFNAI. FRED tags the VIX (CBOE), the Moody's yields, the Treasury spread and the activity index "Copyrighted: Citation Required", and Moody's notes forbid retransmitting its data without consent.
- **In this copy:** the six monthly log returns and the seven macro features as z-scores, plus the result tables. The raw file of prices and levels is left out; `staff/rebuild_raw_data.py` rebuilds it, with returns that match the committed ones to five decimals.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up. |
| `Data Collection.ipynb` | Downloads the FRED and Yahoo Finance data and builds the raw and full-sample tables. |
| `final_project_bayesian.ipynb` | The whole analysis, from exploration to the refined backtest. Committed without outputs. |
| `scripts/eda_overview.py` | Five exploratory figures and three tables from the full-sample table. |
| `docs/modeling_blueprint.md` | The team's plan: a Bayesian HMM on the six returns with two and three states in PyMC, and its baselines. |
| `Bayesian ML Group Project proposal.pages` | The topic proposal, in Apple Pages format. |
| `macro_asset_dataset.csv`, `macro_asset_dataset_expanding_z.csv` | The full-sample and expanding-window tables. See `data/README.md`. |
| `*_results.csv` | The three regime models' probabilities, and the two backtests' probabilities, weights and returns. |
| `outputs/` | Exploratory figures and tables, and the write-up's three figures. |
| `data/README.md`, `staff/`, `requirements.txt` | Course staff's data note, checks and raw-file rebuild, and pinned versions. |

## Method

**Preprocessing.** `Data Collection.ipynb` takes the last value of each month, builds the credit spread, the change in unemployment and the twelve-month growth of industrial production, shifts all seven macro columns one month later, and computes log returns. Its z-scores use the full-sample mean and standard deviation, which the notebook flags as look-ahead (cells 19 and 20), so cells 21 and 22 standardize each month with that month and the earlier ones only, at least 24 of them. Every later step uses this table.

**Baseline.** A Gaussian mixture with full covariance on the six returns, two and three components, 20 random starts each (cell 28). BIC prefers two. The rarer component covers 17 months, 7.8 percent of the sample, with an average monthly return across the six funds of −4.65 percent against +0.90 percent elsewhere.

**Bayesian HMM.** Let *y*ₜ be the equal-weight log return. With hidden states *z*ₜ ∈ {0, 1},

> *z*₁ ~ Categorical(π), *z*ₜ | *z*ₜ₋₁ ~ Categorical(A[*z*ₜ₋₁]), *y*ₜ | *z*ₜ = *k* ~ Normal(μₖ, σₖ²),

with σₖ² ~ Inverse-Gamma(3, 2*s*²), where *s*² is the sample variance of *y*, μₖ | σₖ² ~ Normal(0, σₖ²), π ~ Dirichlet(1, 1), and the rows of A ~ Dirichlet(12, 2) and Dirichlet(2, 12), a prior that favors staying. Each Gibbs sweep (cell 38) draws the state path by forward filtering and backward sampling, each state's mean and variance from its Normal-Inverse-Gamma posterior, and A and π from their Dirichlet posteriors, then reorders the states so that state 0 has the lower mean. One chain of 1,600 sweeps starts from the mixture's labels and keeps every fifth after 600, 200 draws. The output is each month's share of draws in state 0: the smoothed probability of the low-return state given all 217 months.

**Markov-switching regression.** statsmodels' `MarkovRegression` with a regime-specific constant and variance, fit by maximum likelihood with 20 random starts (cell 47) to the same series and all 217 months. Cell 48 keeps the smoothed probabilities for description and cell 55 the filtered ones, P(low | data up to *t*), for the backtests; cell 54 explains that smoothed probabilities "use future information".

**Template rule.** With *p*ₜ the filtered probability of the low-return regime, the weights are *p*ₜ times a risk-off portfolio (TLT 45 percent, GLD 25, VNQ 15, the rest 5 each) plus 1 − *p*ₜ times a risk-on one (SPY 30, EFA 20, DBC and VNQ 15, TLT and GLD 10), cells 56 and 57.

**Refined rule.** From the 37th month on (February 2011), for each month *t* (cells 66 and 67): weight all earlier months by their filtered low-regime probabilities, and separately by the high-regime ones; compute the weighted mean and covariance of the six returns; solve (Σ + 0.0002 I) *w* = μ and drop negative weights, falling back to inverse volatility if all are negative; project onto weights between 3 and 45 percent; blend 65/35 with equal weight and project again. The two regime portfolios are then mixed by *p*ₜ.

**Backtest.** Weights chosen at the end of month *t* earn the returns of month *t* + 1 (cells 58 and 68). A portfolio's monthly return is the weighted sum of the six log returns, and wealth is the running product of one plus that return. The Sharpe ratio is the annualized return over the annualized volatility, with no risk-free rate (cell 59). Turnover is reported but not charged.

### What is Bayesian here

The Gibbs sampler is a genuine Bayesian hidden Markov model, written from scratch: a sticky Dirichlet prior on the transitions, a conjugate Normal-Inverse-Gamma prior on each state's mean and variance, forward filtering and backward sampling for the state path, and conjugate updates for the rest. The code is correct and, with the notebook's seed, exactly reproducible. It joins the course's HMM and MCMC material, and it reports posterior spread: the low-return state's mean is −1.71 percent a month with a posterior standard deviation of 1.58 percent, against +0.81 percent with 0.20 for the calm state.

What stops short is everything downstream. The sampler returns smoothed probabilities, and no trade reads them: both allocation rules follow the Markov-switching regression, which the notebook calls "the regime model we should probably emphasize" and the write-up calls "frequentist, not Bayesian". No decision integrates over parameter uncertainty; the regime probability enters the portfolio only as a mixing weight. The model sees one series, not the six returns and the macro variables, and runs one chain with no convergence diagnostics. Ordering the states by their means is a weak way to tell them apart when one mean is this uncertain: in 8 of the 200 draws the state labelled "low-return" is the calmer of the two, so those draws count calm months as low-return months, and no month's low-return probability falls below 0.035. The two states differ far more clearly in volatility (5.83 against 2.70 percent), which would be the better ordering.

The plan was more Bayesian than the build. The data and methodology document promised a two- or three-state HMM over the returns and the macro variables with a Normal-Inverse-Wishart prior, in PyMC and NumPyro; a Bayesian change-point model as a robustness check; and "a Bayesian mean-variance allocation layer that integrates over regime uncertainty", judged on calibration as well. The write-up's notes say PyMC was missing from "the local environment". The hand-written sampler is the part of the plan that was built.

## Results

**Regime models.** The staff scripts' reruns give the numbers the notebook's markdown reports.

| | Gibbs HMM, low-return | Gibbs HMM, calm | Markov switching, low-return | Markov switching, high-return |
|---|---:|---:|---:|---:|
| Mean monthly return | −1.71% | 0.81% | −1.84% | 0.79% |
| Monthly volatility | 5.83% | 2.70% | 6.68% | 2.58% |
| Probability of staying | 0.825 | 0.952 | 0.806 | 0.974 |
| Months with probability above 0.5 | 25 of 217 | | 24 of 217 (smoothed) | |

The Markov-switching regression's expected stays are 5.2 and 38.7 months.

**Macro conditions by regime.** In the regression's low-return regime the average expanding z-scores (from `markov_switching_results.csv`) are 1.49 for the VIX, 1.10 for the credit spread, 0.73 for unemployment, 0.79 for its change, −1.21 for production growth and −1.14 for the activity index; in the other regime all seven lie between −0.35 and 0.22.

**Backtests as committed.** The write-up quotes the refined comparison, February 2011 to April 2026, 181 months.

| | Annual return | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Template rule | 5.93% | 10.56% | 0.562 | −22.6% |
| Refined rule | 7.51% | 10.37% | 0.724 | −23.7% |
| Equal weight | 6.08% | 10.08% | 0.603 | −19.6% |
| 60/40 | 8.78% | 10.03% | 0.875 | −27.5% |

The staff script recomputes these from `refined_regime_backtest_results.csv`, and its own implementation of the two rules reproduces the committed weights, which turn over about 5 percent of the portfolio a month. The template backtest itself (cells 58 and 59) runs from February 2008, and its own numbers differ: 5.40 percent and a Sharpe ratio of 0.438 for the template rule, 5.01 percent and 0.417 for equal weight, 7.29 percent and 0.675 for 60/40, with drawdowns of −32.7, −40.6 and −28.5 percent.

**Course staff's corrections.** From `staff/check_walk_forward.py`, March 2011 to April 2026, 180 months, every rule scored from the same first month.

| Sharpe ratio | As the notebook scores it | Simple returns | Less 10 bp per unit turnover | Over the bill rate |
|---|---:|---:|---:|---:|
| Template, in-sample fit (committed) | 0.563 | 0.691 | 0.684 | 0.542 |
| Template, real-time refit | 0.555 | 0.682 | 0.675 | 0.534 |
| Refined, in-sample fit (committed) | 0.727 | 0.858 | 0.851 | 0.706 |
| Refined, real-time refit | 0.739 | 0.871 | 0.864 | 0.718 |
| Same optimizer, no regimes | 0.752 | 0.885 | 0.881 | 0.729 |
| Equal weight | 0.580 | 0.718 | 0.714 | 0.562 |
| 60/40 | 0.859 | 0.969 | 0.966 | 0.809 |

With simple returns the annual returns are 7.26 percent (template), 8.88 (refined), 8.85 (no regimes), 7.18 (equal weight) and 9.74 (60/40). The bill rate averaged 1.50 percent a year over the window.

**The Bayesian HMM in the trading loop.** From `staff/check_bayesian_hmm_realtime.py`, same months, simple returns, no costs. "Real time" refits the sampler once a year on the data available then and filters forward with the posterior-mean parameters until the next refit.

| Sharpe ratio | Template rule | Refined rule |
|---|---:|---:|
| Smoothed Gibbs probabilities (look-ahead) | 0.830 | 0.876 |
| Real-time filtered Gibbs probabilities | 0.695 | 0.866 |
| Committed Markov-switching filtered probabilities | 0.691 | 0.858 |
| Equal weight, 60/40 | 0.718, 0.969 | |

The real-time Gibbs probability correlates 0.964 with the regression's filtered one and 0.804 with the Gibbs sampler's own smoothed one.

## What to learn from this project

Course staff added this section. The pipeline is clean and the write-up admits that 60/40 wins; the points below are about what the results can support.

- **Trade on filtered probabilities, and know which ones your sampler returns.** A smoothed probability P(*z*ₜ | all the data) uses months after *t*; a filtered one, P(*z*ₜ | data up to *t*), does not. The notebook gets this right for the regression and explains it well in cell 54. Its Gibbs sampler, though, returns only smoothed probabilities, since the share of draws in each state averages over paths sampled from the whole series. Trading the template rule on them gives a Sharpe ratio of 0.830, above equal weight's 0.718; on the sampler's real-time filtered probabilities it gives 0.695, below it. The look-ahead alone would turn a loss against the benchmark into a win.
- **Beware look-ahead in the parameters, not only in the states.** Filtered probabilities from a model fit to all 217 months still know the future through its parameters. Refitting every month on the data to date is the test, and here it changes little: the real-time probability correlates 0.957 with the committed one, differs by 0.024 on average and by 0.43 at most (October 2011), and the refined rule's Sharpe ratio moves from 0.727 to 0.739. Only a refit can show that. It also shows why priors matter: for February 2012, a single random-start search like cell 47's collapses one regime onto one month with a variance near zero, a degenerate maximum of the likelihood that the Inverse-Gamma prior in the authors' own sampler rules out. The staff script keeps the best proper fit of five searches.
- **Detecting regimes is not earning returns: compare against the same rule without them.** The decisive control is the refined optimizer with the regimes switched off, which `docs/modeling_blueprint.md` lists as a baseline ("rolling historical mean-variance without regimes") but the notebook never runs. It scores 0.752 against the refined rule's 0.727 as the notebook scores them, and 0.885 against 0.858 with simple returns, with a third of the turnover (0.016 against 0.047 a month). The refined rule beats the template because of its optimizer, caps and blend with equal weight, not because of the regimes, and the template rule, driven by the regimes alone, trails equal weight on every scoring of the common sample. The regimes match known episodes (2008, 2011, March 2020, 2022), but recognizing a stressed month once it has started does not earn money, and the write-up's "those states can improve dynamic allocation decisions" is not supported.
- **Put the title's inputs into the model.** The title says "Macro-Driven" and the write-up asks whether regimes "inferred from macro and return data" help, but no model reads a macro variable. The macro z-scores appear only as averages within each fitted regime (cells 31, 41 and 49), where the low-return regime has a VIX z-score of 1.49 and a production-growth z-score of −1.21. That describes the regimes; it does not drive them. So the expanding-window rescaling, the notebook's careful fix for look-ahead, protects variables that never reach a trade, while the traded model's parameters go unchecked.
- **Compound the returns you computed.** `Data Collection.ipynb` makes log returns, and every backtest and growth chart weights them as if they were simple returns and compounds one plus the log return. That understates every gain: SPY's cumulative return over the sample is 534.8 percent by the notebook's method (cell 18) and 711.9 percent in fact, and DBC's 14.5 percent is really 66.7. In the backtest, simple returns add 1.1 to 1.3 percentage points a year to every rule and leave the ranking unchanged. Convert with exp(*r*) − 1 before weighting assets.
- **Score every rule from the same first month, and charge for trading.** Because the weights are lagged with `shift(1)`, each strategy's first month has no weights, and pandas sums the missing products to zero: the strategies book 0 percent in February 2011 while equal weight books +3.84 percent and 60/40 +2.70 percent (+1.39 and −1.75 percent in February 2008). The staff table drops that month for every rule. Costs barely matter here: at about 5 percent turnover a month, 10 basis points per unit of turnover costs the refined rule 0.007 of Sharpe ratio, and subtracting the bill rate lowers every Sharpe ratio by about 0.15 and keeps the order.
- **Quote the sample you ran.** The write-up gives its template numbers under "First allocation layer", but they come from the refined comparison's shorter sample. The template backtest's own table, from 2008, is never shown; on it the template rule beats equal weight, 0.438 against 0.417, through a crisis its full-sample regression already knew.
- **Tune on one period and report another, or say that you did not tune.** The refined rule's settings (a 65/35 blend, a 45 percent cap, a 3 percent floor and a ridge of 0.0002) do not look chosen on the test period: among 180 combinations of the four, its Sharpe ratio of 0.727 ranks 76th, in a range of 0.635 to 0.856. But nothing separates a period for choosing settings from a period for judging them, and the best combination beats the authors' by 0.13.
- **Ship the plan or revise it in writing.** The proposal and methodology document promised a multivariate HMM over returns and macro variables in PyMC and NumPyro, a change-point model, an hmmlearn baseline, a Bayesian mean-variance layer, a calibration metric and data from 2000. Built were a univariate HMM, a mixture and a frequentist regression, and allocation rules that use none of the posterior. The quarter's Assignment 2, "Detecting Market Regimes with Bayesian Methods", had students run a Bayesian change-point model on SPY and an HMM filter on META and compare a regime-timed strategy with buy-and-hold; this project covers the same ground with more assets, macro data and its own sampler, and should say how it goes beyond the homework.
- **Keep the repository runnable and the data shareable.** With links to a laptop folder, a missing notebook name, no saved outputs and no requirements file, a reader sees no number without rebuilding the environment; the data notebook carried an API key, and the raw file republished Yahoo prices and VIX and Moody's levels.

What the project does well: an end-to-end pipeline from public data to a backtest, whose committed tables the code reproduces; a correct Gibbs sampler for a Bayesian HMM, written from scratch, with a sticky prior and forward filtering and backward sampling; a clear statement of why backtests need filtered rather than smoothed probabilities, and code that follows it for the traded model; a publication lag on the macro data, expanding-window z-scores, a one-month signal lag and estimates that exclude the month being decided; turnover reported next to returns; settings that were not cherry-picked; and a conclusion that admits a 60/40 portfolio beat everything. With the Bayesian model in the trading loop and the controls above, it shows how to test a trading idea honestly.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Spring 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff copied the authors' repository and made these edits. The authors' code is unchanged; the only change to the authors' files is the removed key below.

- Imported the authors' files as they were at commit dcc1b18 of [bosen999/Bayesian-Machine-Learning](https://github.com/bosen999/Bayesian-Machine-Learning), as one commit without the earlier history, because the first commit carries both of the next two items.
- Left out `macro_asset_raw.csv`: it held Yahoo Finance adjusted ETF prices and the VIX and Moody's Baa−Aaa spread levels, which their terms do not let us republish. `staff/rebuild_raw_data.py` rebuilds it.
- In cell 6 of `Data Collection.ipynb`, replaced the FRED API key written in as the default of `os.environ.get("FRED_API_KEY", ...)` with an empty string, so the notebook reads the key from the environment only. Nothing else in the notebook changed.
- Removed `.DS_Store`, and added a `.gitignore` that keeps a rebuilt raw file out of the repository.
- Added `requirements.txt` with the versions staff ran everything with, and `data/README.md`, on the sources, their terms and how to rebuild the raw file.
- Added `staff/rebuild_raw_data.py`, `staff/check_walk_forward.py` and `staff/check_bayesian_hmm_realtime.py`, with their shared helper `staff/common.py`, described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. The data's terms are in [data/README.md](data/README.md).
