#!/usr/bin/env python3
"""Course-staff check: the regime backtest, refit in real time and with controls.

Reads only the CSV files the authors committed, and reuses the authors' own allocation
functions and constants from final_project_bayesian.ipynb (see common.py). Its sections:

1. the authors' two-state Markov-switching regression (cell 47), refit on the full sample
   and compared with the committed probabilities;
2. the committed regime tables, and the cumulative returns as cell 9 computes them;
3. the committed backtests, recomputed from the committed CSVs, with the first month as
   booked and the months the model table leaves out;
4. real-time refits: the same regression refit every month on data up to that month only,
   keeping the best proper fit of five random-start searches, so that each trade uses a
   filtered probability a trader could have had;
5-7. the template and refined rules on the in-sample and the real-time probabilities, and
   a control, the same refined optimizer (cells 66-67) with the regimes switched off,
   all scored from the same first month three ways: as the notebook does (log returns
   weighted and compounded as if they were simple returns), with simple returns, and
   with simple returns less 10 basis points per unit of turnover;
8. the refined layer's settings varied, to see where the authors' choice ranks;
9. with --rf, Sharpe ratios on returns in excess of the 3-month Treasury bill (FRED
   series TB3MS, public domain, downloaded when the flag is given).

Needs NumPy, pandas, SciPy and statsmodels; under a minute.
Usage:  python staff/check_walk_forward.py [--rf]
"""
import argparse
import inspect
import itertools
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RETURN_COLS, notebook_definitions, performance, read_csv, table

NB = notebook_definitions(
    "project_weights", "weighted_mean_cov", "long_only_mean_variance",
    "risk_on_weights", "risk_off_weights", "lookback_months", "optimizer_blend",
)
RISK_ON = NB["risk_on_weights"][RETURN_COLS].to_numpy()
RISK_OFF = NB["risk_off_weights"][RETURN_COLS].to_numpy()
START = NB["lookback_months"]          # 36: the first decision is the 37th month
BLEND = NB["optimizer_blend"]          # 0.65
EQUAL = np.repeat(1 / len(RETURN_COLS), len(RETURN_COLS))
REFIT_SEEDS = (42, 43, 44, 45, 46)    # each real-time refit keeps the best of five searches
MIN_SIGMA2 = 1e-5                     # a regime volatility under 0.3% a month is a collapsed fit


# Cell 47 seeds NumPy's global generator, which statsmodels used for its random search
# up to 0.14. From 0.15 the search draws from fresh entropy unless fit() gets rng, so the
# authors' fit can move in the fourth decimal from run to run. Pass rng where it exists.
FIT_TAKES_RNG = "rng" in inspect.signature(sm.tsa.MarkovRegression.fit).parameters


def fit_ms(y, seeds=(42,)):
    """The authors' fit from cell 47, once per seed; keeps the highest likelihood.

    Returns (result, column order with the low-return regime first). Short windows can
    make the search fail; a failed seed is skipped, and a fit without search is the
    last resort. Maximum likelihood can also collapse a regime onto a single month, with
    a variance near zero and an unbounded likelihood; such a fit is kept only if no
    other seed gives a proper one.
    """
    best, best_degenerate = None, False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed, reps in [(s, 20) for s in seeds] + [(seeds[0], 0)]:
            if best is not None and reps == 0:
                break
            np.random.seed(seed)
            try:
                res = sm.tsa.MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True).fit(
                    em_iter=10, search_reps=reps, search_iter=10, disp=False,
                    **({"rng": seed} if FIT_TAKES_RNG else {}))
            except np.linalg.LinAlgError:
                continue
            degenerate = min(res.params[f"sigma2[{i}]"] for i in range(2)) < MIN_SIGMA2
            if best is None or (res.llf > best.llf and not degenerate) or best_degenerate:
                best, best_degenerate = res, degenerate
    order = np.argsort([best.params[f"const[{i}]"] for i in range(2)])
    return best, order


def template_weights(p_low):
    """Cell 57: blend the hand-set risk-off and risk-on portfolios by the low-regime probability."""
    return p_low * RISK_OFF + (1 - p_low) * RISK_ON


def refined_weights(history, p_low_history, p_low_now, blend=BLEND, cap=0.45, floor=0.03, ridge=2e-4):
    """The body of the loop in cell 67, with the settings as arguments (defaults are the authors')."""
    pw, wmc, lomv = NB["project_weights"], NB["weighted_mean_cov"], NB["long_only_mean_variance"]
    ports = []
    for w in (p_low_history, 1 - p_low_history):
        mean, cov = wmc(history, w)
        ports.append(pw(blend * pw(lomv(mean, cov, ridge), cap, floor) + (1 - blend) * EQUAL, cap, floor))
    return pw(p_low_now * ports[0] + (1 - p_low_now) * ports[1], cap, floor)


def as_notebook(weights, log_returns):
    """Cells 58 and 68: last month's weights times this month's log returns, summed."""
    return (weights.shift(1) * log_returns.loc[weights.index]).sum(axis=1).iloc[1:]


def simple_backtest(weights, simple_returns, cost_bps=0.0):
    """Hold each month's weights through the next month in simple returns; charge cost on turnover.

    Turnover is measured against the weights after they drifted with the month's returns.
    The first holding month carries no cost, for every strategy alike.
    """
    w = weights.to_numpy()
    r = simple_returns.loc[weights.index].to_numpy()
    out = []
    for t in range(1, len(w)):
        gross = float(w[t - 1] @ r[t])
        if t >= 2:
            drifted = w[t - 2] * (1 + r[t - 1])
            drifted /= drifted.sum()
            gross -= cost_bps / 1e4 * np.abs(w[t - 1] - drifted).sum()
        out.append(gross)
    return pd.Series(out, index=weights.index[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rf", action="store_true", help="download TB3MS from FRED for excess-return Sharpe ratios")
    args = ap.parse_args()

    data = read_csv("macro_asset_dataset_expanding_z.csv")
    log_r = data[RETURN_COLS]
    simple_r = np.exp(log_r) - 1
    y = log_r.mean(axis=1)                       # cells 37 and 47: equal-weight log return
    committed = read_csv("regime_backtest_results.csv")
    committed_ms = read_csv("markov_switching_results.csv")
    refined_csv = read_csv("refined_regime_backtest_results.csv")
    T = len(data)

    print("== 1. The authors' Markov-switching fit, refit on the full sample (cell 47)")
    res, order = fit_ms(y)
    means = np.array([res.params[f"const[{i}]"] for i in range(2)])[order]
    vols = np.sqrt(np.array([res.params[f"sigma2[{i}]"] for i in range(2)]))[order]
    P = res.regime_transition[:, :, 0].T[np.ix_(order, order)]
    print(pd.DataFrame({"mean_%": 100 * means, "vol_%": 100 * vols,
                        "stay_prob": np.diag(P), "duration_months": 1 / (1 - np.diag(P))},
                       index=["low-return", "high-return"]).round(3).to_string())
    filt = np.asarray(res.filtered_marginal_probabilities)[:, order[0]]
    smooth = np.asarray(res.smoothed_marginal_probabilities)[:, order[0]]
    print(f"max |filtered - committed| = {np.abs(filt - committed['ms_regime_0_filtered_prob']).max():.1e}, "
          f"max |smoothed - committed| = {np.abs(smooth - committed_ms['ms_regime_0_prob']).max():.1e}")
    print(f"months with smoothed P(low) > 0.5: {(smooth > 0.5).sum()} of {T}; "
          f"with filtered P(low) > 0.5: {(filt > 0.5).sum()}")

    print("\n== 2. The committed regime tables")
    gmm = read_csv("baseline_regime_results.csv")
    low = gmm["baseline_regime"] == 0
    print(f"mixture (cell 30): low-return cluster {low.sum()} of {len(gmm)} months ({100 * low.mean():.1f}%), "
          f"mean of the six assets' average returns there {100 * gmm.loc[low, RETURN_COLS].mean().mean():+.2f}% a month, "
          f"elsewhere {100 * gmm.loc[~low, RETURN_COLS].mean().mean():+.2f}%")
    hmm = read_csv("bayesian_hmm_results.csv")
    print(f"Gibbs HMM (cell 40): months with P(low) > 0.5: {(hmm.prob_regime_0 > 0.5).sum()}, "
          f"average P(low) {hmm.prob_regime_0.mean():.3f}")
    zcols = [c for c in committed_ms.columns if c.endswith("_z_exp")]
    print(f"Markov switching (cell 49): low-return regime {(committed_ms.ms_regime == 0).sum()} months; "
          "mean expanding z-scores there:")
    print(committed_ms.groupby("ms_regime")[zcols].mean().round(3).rename(index={0: "low", 1: "high"}).to_string())
    full_table = read_csv("macro_asset_dataset.csv")
    print(f"cumulative return {full_table.index[0]:%Y-%m} to {full_table.index[-1]:%Y-%m}, as cell 9 computes it "
          "((1 + log return) compounded) and correctly (exp of the summed log returns):")
    print(pd.DataFrame({"cell_9_%": 100 * ((1 + full_table[RETURN_COLS]).prod() - 1),
                        "correct_%": 100 * (np.exp(full_table[RETURN_COLS].sum()) - 1)}).round(1).to_string())

    print("\n== 3. The committed backtests, recomputed from the committed CSVs")
    first = committed[["regime_dynamic", "equal_weight", "sixty_forty"]]
    print(f"regime_backtest_results.csv, {first.index[0]:%Y-%m} to {first.index[-1]:%Y-%m}, {len(first)} months:")
    print(table(first))
    cols = ["template_regime_dynamic", "refined_regime_dynamic", "equal_weight", "sixty_forty"]
    ref = refined_csv[cols]
    print(f"\nrefined_regime_backtest_results.csv, {ref.index[0]:%Y-%m} to {ref.index[-1]:%Y-%m}, {len(ref)} months:")
    print(table(ref))
    print(f"\nfirst month of each file, as booked: {first.index[0]:%Y-%m} "
          + ", ".join(f"{c} {100 * first[c].iloc[0]:+.2f}%" for c in first.columns)
          + f"; {ref.index[0]:%Y-%m} " + ", ".join(f"{c} {100 * ref[c].iloc[0]:+.2f}%" for c in ref.columns))
    full = pd.date_range(data.index[0], data.index[-1], freq="ME")
    print("month-ends missing from the model table:", [f"{d:%Y-%m}" for d in full.difference(data.index)])

    print("\n== 4. Real-time refits: the regression refit every month on data up to that month")
    p_insample = committed["ms_regime_0_filtered_prob"].to_numpy()
    p_realtime = np.full(T, np.nan)
    W = {k: [] for k in ["template, in-sample fit", "refined, in-sample fit",
                         "template, real-time refit", "refined, real-time refit",
                         "same optimizer, no regimes"]}
    X = log_r.to_numpy()
    collapsed = []
    for t in range(START, T):
        np.random.seed(42)
        r_t, o_t = fit_ms(y.iloc[: t + 1], seeds=(42,))
        if min(r_t.params[f"sigma2[{i}]"] for i in range(2)) < MIN_SIGMA2:
            collapsed.append(f"{data.index[t]:%Y-%m}")
        r_t, o_t = fit_ms(y.iloc[: t + 1], seeds=REFIT_SEEDS)
        pf = np.asarray(r_t.filtered_marginal_probabilities)[:, o_t[0]]
        p_realtime[t] = pf[t]
        W["template, in-sample fit"].append(template_weights(p_insample[t]))
        W["refined, in-sample fit"].append(refined_weights(X[:t], p_insample[:t], p_insample[t]))
        W["template, real-time refit"].append(template_weights(pf[t]))
        W["refined, real-time refit"].append(refined_weights(X[:t], pf[:t], pf[t]))
        W["same optimizer, no regimes"].append(refined_weights(X[:t], np.zeros(t), 0.0))
    idx = data.index[START:]
    W = {k: pd.DataFrame(v, index=idx, columns=RETURN_COLS) for k, v in W.items()}
    W["equal weight"] = pd.DataFrame([EQUAL] * len(idx), index=idx, columns=RETURN_COLS)
    W["60/40"] = pd.DataFrame([[0.6, 0, 0.4, 0, 0, 0]] * len(idx), index=idx, columns=RETURN_COLS)
    for name, col in [("template, in-sample fit", "w_"), ("refined, in-sample fit", "refined_w_")]:
        src = committed if col == "w_" else refined_csv
        diff = np.abs(W[name].to_numpy() - src.loc[idx, [col + c for c in RETURN_COLS]].to_numpy()).max()
        print(f"{name}: max |weight - committed weight| = {diff:.1e}")
    print(f"refits where the authors' single search (seed 42) collapses a regime onto one month: "
          f"{len(collapsed)} of {T - START} {collapsed}")
    a, b = p_insample[START:], p_realtime[START:]
    print(f"decisions {idx[0]:%Y-%m} to {idx[-1]:%Y-%m}; P(low) in-sample vs real time: "
          f"correlation {np.corrcoef(a, b)[0, 1]:.3f}, mean |difference| {np.abs(a - b).mean():.3f}, "
          f"largest {np.abs(a - b).max():.2f} in {idx[np.abs(a - b).argmax()]:%Y-%m}; "
          f"months above 0.5: {(a > 0.5).sum()} and {(b > 0.5).sum()}")

    nb = pd.DataFrame({k: as_notebook(w, log_r) for k, w in W.items()})
    sim = pd.DataFrame({k: simple_backtest(w, simple_r) for k, w in W.items()})
    net = pd.DataFrame({k: simple_backtest(w, simple_r, 10) for k, w in W.items()})
    span = f"{nb.index[0]:%Y-%m} to {nb.index[-1]:%Y-%m}, {len(nb)} months, every strategy from the same month"
    print(f"\n== 5. Scored as the notebook does (log returns as simple returns), {span}")
    print(table(nb))
    print(f"\n== 6. Scored with simple returns, {span}")
    print(table(sim))
    print(f"\n== 7. Simple returns less 10 bp per unit of turnover, {span}")
    print(table(net))
    turnover = {k: W[k].diff().abs().sum(axis=1).iloc[1:].mean() for k in W}
    print("mean monthly turnover (sum of |weight changes|):",
          ", ".join(f"{k} {v:.3f}" for k, v in turnover.items()))

    print("\n== 8. The refined layer's settings: authors' scoring, committed probabilities")
    rows = []
    for blend, cap, floor, ridge in itertools.product([0.35, 0.5, 0.65, 0.8, 1.0], [0.3, 0.45, 0.6, 1.0],
                                                      [0.0, 0.03, 0.05], [1e-4, 2e-4, 5e-4]):
        w = pd.DataFrame([refined_weights(X[:t], p_insample[:t], p_insample[t], blend, cap, floor, ridge)
                          for t in range(START, T)], index=idx, columns=RETURN_COLS)
        rows.append((blend, cap, floor, ridge, performance(as_notebook(w, log_r))["sharpe"]))
    grid = pd.DataFrame(rows, columns=["blend", "cap", "floor", "ridge", "sharpe"])
    grid = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    mine = grid.index[(grid.blend == BLEND) & (grid.cap == 0.45) & (grid.floor == 0.03) & (grid.ridge == 2e-4)][0]
    print(f"{len(grid)} settings; Sharpe from {grid.sharpe.min():.3f} to {grid.sharpe.max():.3f}, "
          f"median {grid.sharpe.median():.3f}; the authors' setting ranks {mine + 1} "
          f"with {grid.sharpe[mine]:.3f}; best: {grid.iloc[0, :4].to_dict()}")

    if args.rf:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
        tb = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")["TB3MS"]
        tb.index = tb.index + pd.offsets.MonthEnd(0)
        rf = (tb.shift(1) / 100 / 12).reindex(sim.index)    # last month's annual yield, per month
        excess = pd.Series({k: performance(sim[k] - rf)["sharpe"] for k in sim}, name="excess_sharpe")
        print(f"\n== 9. Sharpe on returns in excess of the 3-month bill (TB3MS), simple returns, {span}")
        print(f"bill return over the window: {100 * ((1 + rf).prod() ** (12 / len(rf)) - 1):.2f}% a year")
        print(excess.round(3).to_string())

if __name__ == "__main__":
    main()
