#!/usr/bin/env python3
"""Course-staff check: the authors' Bayesian HMM, put into the trading loop.

The notebook's Gibbs sampler (cell 38) returns smoothed state probabilities, which use the
whole sample, and the backtests never read them: the trades follow the Markov-switching
regression. This script reuses the authors' sampler and allocation functions unchanged
(see common.py) and

1. reruns the full-sample fit of cell 39 with its seed and starting states, compares the
   result with the committed bayesian_hmm_results.csv, and checks how well ordering the
   states by their means (cell 38) identifies them;
2. refits the sampler once a year, from February 2011, on data up to the refit month
   only (shorter chains, a median split for the starting states), and runs a forward
   filter with the posterior-mean parameters through the following year, so that each
   month's probability uses only data up to that month;
3. trades the authors' template and refined rules on three probabilities: the smoothed
   full-sample one (look-ahead, shown only for contrast), the real-time filtered one,
   and the committed Markov-switching filtered one, scored with simple returns.

Needs NumPy, pandas and SciPy; about two minutes, almost all of it in the sampler.
Usage:  python staff/check_bayesian_hmm_realtime.py [--iter 700 --burn 200]
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RETURN_COLS, notebook_definitions, read_csv, table
from check_walk_forward import EQUAL, START, refined_weights, simple_backtest, template_weights

NB = notebook_definitions("sample_states_ffbs", "sample_emission_params", "relabel_by_mean", "run_bayesian_hmm")
run_bayesian_hmm = NB["run_bayesian_hmm"]


def forward_filter(y, pi, A, mu, sigma2):
    """P(state 0 | y_1..t) for every t, with fixed parameters (state 0 = lower mean)."""
    log_e = np.column_stack([norm.logpdf(y, mu[k], np.sqrt(sigma2[k])) for k in range(2)])
    la = np.empty_like(log_e)
    la[0] = np.log(pi) + log_e[0]
    for t in range(1, len(y)):
        la[t] = log_e[t] + logsumexp(la[t - 1][:, None] + np.log(A), axis=0)
    return np.exp(la[:, 0] - logsumexp(la, axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iter", type=int, default=700, help="Gibbs iterations per yearly refit (authors: 1600)")
    ap.add_argument("--burn", type=int, default=200, help="burn-in per yearly refit (authors: 600)")
    args = ap.parse_args()

    data = read_csv("macro_asset_dataset_expanding_z.csv")
    log_r = data[RETURN_COLS]
    y = log_r.mean(axis=1).to_numpy()
    T = len(y)
    committed = read_csv("bayesian_hmm_results.csv")
    baseline = read_csv("baseline_regime_results.csv")

    print("== 1. The authors' full-sample Gibbs run (cells 38-40), rerun")
    full = run_bayesian_hmm(y, baseline.loc[data.index, "baseline_regime"].to_numpy(),
                            n_iter=1600, burn=600, thin=5, seed=42)
    smoothed = full["state_probs"][:, 0]
    sd = np.sqrt(full["sigma2_draws"])
    print(pd.DataFrame({"mean_%": 100 * full["mu_draws"].mean(0), "mean_sd_%": 100 * full["mu_draws"].std(0),
                        "vol_%": 100 * sd.mean(0), "vol_sd_%": 100 * sd.std(0),
                        "stay_prob": np.diag(full["transition_mean"])},
                       index=["low-return", "high-return"]).round(3).to_string())
    print(f"{full['n_saved']} saved draws; max |P(low) - committed| = "
          f"{np.abs(smoothed - committed['prob_regime_0']).max():.1e}; "
          f"months with P(low) > 0.5: {(smoothed > 0.5).sum()}; smallest P(low) {smoothed.min():.3f}")
    mu_low = 100 * full["mu_draws"][:, 0]
    calmer = (full["sigma2_draws"][:, 0] < full["sigma2_draws"][:, 1]).sum()
    print(f"low-return mean: 90% interval {np.percentile(mu_low, 5):+.2f}% to {np.percentile(mu_low, 95):+.2f}%, "
          f"P(mean > 0) = {(mu_low > 0).mean():.2f}; draws in which the state labelled low-return "
          f"is the calmer one: {calmer} of {full['n_saved']}")

    print(f"\n== 2. Yearly real-time refits ({args.iter} iterations, {args.burn} burn-in, thin 5)")
    refits = list(range(START, T, 12))
    segments = []
    for i, t in enumerate(refits):
        yt = y[: t + 1]
        fit = run_bayesian_hmm(yt, (yt < np.median(yt)).astype(int),
                               n_iter=args.iter, burn=args.burn, thin=5, seed=42)
        end = refits[i + 1] if i + 1 < len(refits) else T
        pf = forward_filter(y[:end], fit["pi_mean"], fit["transition_mean"],
                            fit["mu_draws"].mean(0), fit["sigma2_draws"].mean(0))
        segments.append((t, end, pf))
        print(f"  refit {data.index[t]:%Y-%m}: low-return mean {100 * fit['mu_draws'].mean(0)[0]:+.2f}%, "
              f"vol {100 * np.sqrt(fit['sigma2_draws']).mean(0)[0]:.2f}%, "
              f"stay probabilities {np.diag(fit['transition_mean']).round(3).tolist()}")
    realtime = np.full(T, np.nan)
    for t, end, pf in segments:
        realtime[t:end] = pf[t:end]
    ms = read_csv("regime_backtest_results.csv")["ms_regime_0_filtered_prob"].to_numpy()
    s = slice(START, T)
    print(f"correlation of real-time P(low) with the smoothed one {np.corrcoef(realtime[s], smoothed[s])[0, 1]:.3f}, "
          f"with the Markov-switching filtered one {np.corrcoef(realtime[s], ms[s])[0, 1]:.3f}; "
          f"months above 0.5 from {data.index[START]:%Y-%m}: smoothed {(smoothed[s] > 0.5).sum()}, "
          f"real time {(realtime[s] > 0.5).sum()}")

    print("\n== 3. Trading on each probability, simple returns, no costs")
    X = log_r.to_numpy()
    idx = data.index[START:]
    W = {}
    for name, hist_and_now in {
        "smoothed Gibbs (look-ahead)": lambda t: (smoothed[:t], smoothed[t]),
        "real-time Gibbs": None,
        "Markov-switching, committed": lambda t: (ms[:t], ms[t]),
    }.items():
        tw, rw = [], []
        for t in range(START, T):
            if hist_and_now is None:       # history filtered with the parameters in use at t
                seg = next(pf for a, b, pf in segments if a <= t < b)
                hist, now = seg[:t], seg[t]
            else:
                hist, now = hist_and_now(t)
            tw.append(template_weights(now))
            rw.append(refined_weights(X[:t], hist, now))
        W[f"template, {name}"] = pd.DataFrame(tw, index=idx, columns=RETURN_COLS)
        W[f"refined, {name}"] = pd.DataFrame(rw, index=idx, columns=RETURN_COLS)
    W["equal weight"] = pd.DataFrame([EQUAL] * len(idx), index=idx, columns=RETURN_COLS)
    W["60/40"] = pd.DataFrame([[0.6, 0, 0.4, 0, 0, 0]] * len(idx), index=idx, columns=RETURN_COLS)
    sim = pd.DataFrame({k: simple_backtest(w, np.exp(log_r) - 1) for k, w in W.items()})
    print(f"{sim.index[0]:%Y-%m} to {sim.index[-1]:%Y-%m}, {len(sim)} months")
    print(table(sim))


if __name__ == "__main__":
    main()
