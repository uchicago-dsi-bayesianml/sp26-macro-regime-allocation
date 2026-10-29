"""Shared helpers for the course-staff checks in this folder.

The checks reuse the authors' own functions rather than retyping them. They read the
code cells of final_project_bayesian.ipynb, which this copy leaves unchanged, and
execute only the named top-level definitions (functions and constant assignments),
never whole cells. The notebook's SHA-256 is checked first, so an edited notebook
stops the checks instead of silently changing what they test.

Everything else here reads the CSV files that the authors committed.
"""
import ast
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "final_project_bayesian.ipynb"
NOTEBOOK_SHA256 = "8340ee8ebf00ee8f4b118d57ba931110de6119c953e03a94bdb25afe13e5f15c"

RETURN_COLS = ["SPY_ret", "EFA_ret", "TLT_ret", "GLD_ret", "DBC_ret", "VNQ_ret"]


def notebook_definitions(*names):
    """Execute the named top-level definitions from the authors' notebook.

    Returns a namespace (dict) holding them, with the imports the notebook uses.
    """
    raw = NOTEBOOK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise SystemExit(f"{NOTEBOOK.name} differs from the authors' version; refusing to reuse it.")
    from scipy.special import logsumexp
    from scipy.stats import invgamma, norm

    ns = {"np": np, "pd": pd, "logsumexp": logsumexp, "invgamma": invgamma, "norm": norm}
    wanted, found = set(names), set()
    for cell in json.loads(raw)["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                name = node.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            else:
                continue
            if name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), NOTEBOOK.name, "exec"), ns)
                found.add(name)
    missing = wanted - found
    if missing:
        raise SystemExit(f"not found in the notebook: {sorted(missing)}")
    return ns


def read_csv(name):
    return pd.read_csv(ROOT / name, index_col=0, parse_dates=True)


def performance(returns):
    """The authors' performance_metrics (notebook cell 59), in percent, without the cumulative return.

    Its Sharpe ratio is annualized return over annualized volatility, with no risk-free rate.
    """
    global _PERF
    if _PERF is None:
        _PERF = notebook_definitions("performance_metrics")["performance_metrics"]
    m = _PERF(returns)
    return pd.Series({
        "ann_return_%": 100 * m["annualized_return"],
        "ann_vol_%": 100 * m["annualized_volatility"],
        "sharpe": m["sharpe_ratio"],
        "max_drawdown_%": 100 * m["max_drawdown"],
    })


_PERF = None


def table(frame, digits=3):
    """One row of performance numbers per column of monthly returns."""
    return frame.apply(performance).T.round(digits).to_string()
