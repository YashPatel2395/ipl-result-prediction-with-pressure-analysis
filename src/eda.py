"""
eda.py
======
Exploratory Data Analysis for the IPL chase-pressure dataset.

All plots are saved to outputs/plots/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")

PALETTE  = {"win": "#2ECC71", "loss": "#E74C3C"}
WIN_COLS = {1: "#2ECC71", 0: "#E74C3C"}

plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       11,
})


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [EDA] Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Win / Loss distribution
# ---------------------------------------------------------------------------
def plot_win_loss(df: pd.DataFrame) -> None:
    match_level = df.groupby("match_id")["won"].first().value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Chase Lost (0)", "Chase Won (1)"],
                  [match_level.get(0, 0), match_level.get(1, 0)],
                  color=["#E74C3C", "#2ECC71"], width=0.5, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", fontweight="bold")
    ax.set_title("Chase Outcome Distribution (match level)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Matches")
    _save(fig, "01_win_loss_distribution.png")


# ---------------------------------------------------------------------------
# 2. Pressure distribution
# ---------------------------------------------------------------------------
def plot_pressure_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Overall histogram (fast; KDE on 125k rows is slow)
    bins_p = np.linspace(0, 10, 60)
    for outcome, grp in df.groupby("won"):
        label = "Won" if outcome == 1 else "Lost"
        color = WIN_COLS[outcome]
        axes[0].hist(grp["pressure"].clip(0, 10).values, bins=bins_p,
                     color=color, alpha=0.5, label=label, density=True, histtype="stepfilled")
    axes[0].set_xlabel("Pressure")
    axes[0].set_title("Pressure Distribution by Outcome")
    axes[0].legend()

    # Boxplot by outcome
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["won"].map({1: "Won", 0: "Lost"})
    sns.boxplot(data=df_plot, x="Outcome", y="pressure",
                palette={"Won": "#2ECC71", "Lost": "#E74C3C"}, ax=axes[1], width=0.4)
    axes[1].set_title("Pressure Boxplot by Outcome")
    axes[1].set_ylabel("Pressure")

    fig.suptitle("Pressure Metric — Distribution Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_pressure_distribution.png")


# ---------------------------------------------------------------------------
# 3. Pressure across overs
# ---------------------------------------------------------------------------
def plot_pressure_over_time(df: pd.DataFrame) -> None:
    over_stats = (
        df.groupby(["over", "won"])["pressure"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    for outcome, grp in over_stats.groupby("won"):
        label = "Chase Won" if outcome == 1 else "Chase Lost"
        color = WIN_COLS[outcome]
        ax.plot(grp["over"], grp["pressure"], marker="o", ms=5,
                label=label, color=color, linewidth=2)

    # Phase shading
    ax.axvspan(0,  5.5, alpha=0.06, color="blue",   label="Powerplay")
    ax.axvspan(5.5, 14.5, alpha=0.06, color="orange", label="Middle")
    ax.axvspan(14.5, 20, alpha=0.06, color="red",   label="Death")

    ax.set_xlabel("Over")
    ax.set_ylabel("Mean Pressure")
    ax.set_title("Average Pressure per Over — Won vs Lost Chases", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    fig.tight_layout()
    _save(fig, "03_pressure_over_time.png")


# ---------------------------------------------------------------------------
# 4. Wickets lost vs pressure
# ---------------------------------------------------------------------------
def plot_wickets_vs_pressure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["won"].map({1: "Won", 0: "Lost"})
    sns.violinplot(data=df_plot, x="wickets_lost", y="pressure",
                   hue="Outcome", split=True,
                   palette={"Won": "#2ECC71", "Lost": "#E74C3C"},
                   inner="quart", ax=ax)
    ax.set_title("Pressure Distribution by Wickets Lost", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wickets Lost")
    ax.set_ylabel("Pressure")
    fig.tight_layout()
    _save(fig, "04_wickets_vs_pressure.png")


# ---------------------------------------------------------------------------
# 5. Required run rate trends
# ---------------------------------------------------------------------------
def plot_rrr_trends(df: pd.DataFrame) -> None:
    rrr_stats = (
        df[df["required_run_rate"] < 50]            # filter impossible extremes
        .groupby(["over", "won"])["required_run_rate"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    for outcome, grp in rrr_stats.groupby("won"):
        label = "Won" if outcome == 1 else "Lost"
        ax.plot(grp["over"], grp["required_run_rate"], marker="s", ms=4,
                label=label, color=WIN_COLS[outcome], linewidth=2)

    ax.axhline(8.0, ls="--", color="grey", alpha=0.6, label="RRR = 8 (threshold)")
    ax.set_xlabel("Over")
    ax.set_ylabel("Mean Required Run Rate")
    ax.set_title("Required Run Rate Progression — Won vs Lost Chases", fontsize=13, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    fig.tight_layout()
    _save(fig, "05_rrr_trends.png")


# ---------------------------------------------------------------------------
# 6. Phase-level pressure comparison
# ---------------------------------------------------------------------------
def plot_phase_pressure(df: pd.DataFrame) -> None:
    phase_order = ["Powerplay", "Middle", "Death"]
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["won"].map({1: "Won", 0: "Lost"})

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_plot, x="phase", y="pressure", hue="Outcome",
                order=phase_order,
                palette={"Won": "#2ECC71", "Lost": "#E74C3C"},
                ci=95, ax=ax)
    ax.set_title("Mean Pressure by Match Phase", fontsize=13, fontweight="bold")
    ax.set_xlabel("Match Phase")
    ax.set_ylabel("Mean Pressure (95% CI)")
    fig.tight_layout()
    _save(fig, "06_phase_pressure.png")


# ---------------------------------------------------------------------------
# 7. Pressure percentile vs win rate
# ---------------------------------------------------------------------------
def plot_pressure_vs_win_rate(df: pd.DataFrame) -> None:
    """Bin pressure percentile into deciles and compute win rate per bin."""
    df_plot = df.copy()
    df_plot["pressure_bin"] = pd.cut(df_plot["pressure_pct"], bins=10,
                                     labels=[f"{i*10}–{(i+1)*10}%" for i in range(10)])
    bin_stats = df_plot.groupby("pressure_bin")["won"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(bin_stats["pressure_bin"].astype(str), bin_stats["won"],
                  color=sns.color_palette("RdYlGn_r", len(bin_stats)), edgecolor="white")
    for bar, val in zip(bars, bin_stats["won"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("Pressure Percentile Bin")
    ax.set_ylabel("Chase Win Rate")
    ax.set_title("Chase Win Rate by Pressure Percentile", fontsize=13, fontweight="bold")
    ax.axhline(0.5, ls="--", color="grey", alpha=0.6)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save(fig, "07_pressure_vs_win_rate.png")


def run_eda(df: pd.DataFrame) -> None:
    print("\n[EDA] Generating exploratory plots...")
    plot_win_loss(df)
    plot_pressure_distribution(df)
    plot_pressure_over_time(df)
    plot_wickets_vs_pressure(df)
    plot_rrr_trends(df)
    plot_phase_pressure(df)
    plot_pressure_vs_win_rate(df)
    print("[EDA] All plots saved.")
