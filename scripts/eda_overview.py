from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "macro_asset_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

RETURN_COLS = ["SPY_ret", "EFA_ret", "TLT_ret", "GLD_ret", "DBC_ret", "VNQ_ret"]
MACRO_COLS = [
    "T10Y2Y_z",
    "VIX_z",
    "CREDIT_z",
    "UNRATE_z",
    "UNRATE_CHG_z",
    "INDPRO_YOY_z",
    "CFNAI_z",
]


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def save_summary_tables(df: pd.DataFrame) -> None:
    summary = df.describe().T[["mean", "std", "min", "max"]]
    summary.to_csv(TABLES_DIR / "summary_stats.csv")

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing.to_csv(TABLES_DIR / "missing_counts.csv")

    corr = df.corr()
    corr.to_csv(TABLES_DIR / "full_correlation_matrix.csv")


def plot_cumulative_returns(df: pd.DataFrame) -> None:
    cumulative = (1.0 + df[RETURN_COLS]).cumprod()
    ax = cumulative.plot(figsize=(12, 6), linewidth=2)
    ax.set_title("Cumulative Growth of $1 Across Asset ETFs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cumulative_returns.png", dpi=200)
    plt.close()


def plot_return_correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[RETURN_COLS].corr(), annot=True, cmap="RdBu_r", center=0)
    plt.title("Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "return_correlations.png", dpi=200)
    plt.close()


def plot_macro_correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[MACRO_COLS].corr(), annot=True, cmap="RdBu_r", center=0)
    plt.title("Macro Signal Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "macro_correlations.png", dpi=200)
    plt.close()


def plot_macro_series(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(MACRO_COLS), 1, figsize=(12, 16), sharex=True)
    for ax, col in zip(axes, MACRO_COLS):
        ax.plot(df.index, df[col], linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "macro_time_series.png", dpi=200)
    plt.close()


def plot_rolling_volatility(df: pd.DataFrame, window: int = 12) -> None:
    rolling_vol = df[RETURN_COLS].rolling(window).std() * (12 ** 0.5)
    ax = rolling_vol.plot(figsize=(12, 6), linewidth=1.8)
    ax.set_title(f"{window}-Month Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rolling_volatility.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = load_data()

    save_summary_tables(df)
    plot_cumulative_returns(df)
    plot_return_correlation(df)
    plot_macro_correlation(df)
    plot_macro_series(df)
    plot_rolling_volatility(df)

    print("EDA outputs written to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
