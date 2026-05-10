"""
momentum_analysis.py
====================
Visualise momentum dynamics, pressure regimes, and collapse signatures.

Plots generated
---------------
  24  — momentum heatmap (momentum vs win probability)
  25  — pressure regime win rates (bar chart)
  26  — pressure trend trajectories (median + IQR ribbon by outcome)
  27  — collapse timeline (window around wicket burst events)
  28  — panic-zone map (pressure × momentum → win rate heatmap)
  29  — dynamic vs static pressure comparison (KDE by outcome)
  30  — wicket-pressure interaction (pressure trajectory around each wicket)
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
from scipy.ndimage import gaussian_filter

from src.dynamic_pressure import REGIME_LABELS

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [momentum_analysis] Saved → {path}")


# ---------------------------------------------------------------------------
# 24. Momentum vs win probability (scatter with 2D density)
# ---------------------------------------------------------------------------
def plot_momentum_vs_winprob(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: momentum distribution by outcome (histogram, not KDE for speed)
    bins_m = np.linspace(-3, 3, 50)
    for outcome, color, label in [(1, "#2ECC71", "Won"), (0, "#E74C3C", "Lost")]:
        data = df[df["won"] == outcome]["dp_momentum"].clip(-3, 3).values
        axes[0].hist(data, bins=bins_m, color=color, alpha=0.5, label=label,
                     density=True, histtype="stepfilled")
    axes[0].axvline(0, ls="--", color="grey", alpha=0.5, label="Zero momentum")
    axes[0].set_xlabel("Pressure Momentum (EWM_fast − EWM_slow)")
    axes[0].set_title("Pressure Momentum Distribution by Outcome", fontweight="bold")
    axes[0].legend()

    # Right: binned momentum → win rate
    df_plot = df.copy()
    df_plot["mom_bin"] = pd.cut(
        df_plot["dp_momentum"].clip(-3, 3), bins=12,
        labels=[f"{v:.1f}" for v in np.linspace(-3, 3, 12)]
    )
    win_rate = df_plot.groupby("mom_bin")["won"].mean()
    colors   = ["#2ECC71" if v < 0 else "#E74C3C" for v in np.linspace(-3, 3, 12)]
    axes[1].bar(range(len(win_rate)), win_rate.values, color=colors, alpha=0.8)
    axes[1].axhline(0.5, ls="--", color="grey", alpha=0.5)
    axes[1].set_xticks(range(len(win_rate)))
    axes[1].set_xticklabels(win_rate.index, rotation=45, fontsize=8)
    axes[1].set_ylabel("Chase Win Rate")
    axes[1].set_title("Win Rate by Pressure Momentum Bin", fontweight="bold")
    axes[1].set_ylim(0, 1)

    fig.suptitle("Pressure Momentum Analysis\n"
                 "(negative momentum = pressure easing → higher win rate)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "24_momentum_vs_winprob.png")


# ---------------------------------------------------------------------------
# 25. Pressure regime win rates
# ---------------------------------------------------------------------------
def plot_regime_win_rates(df: pd.DataFrame) -> None:
    regime_stats = (
        df.groupby("dp_regime")["won"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "n"})
        .reset_index()
    )
    regime_stats["label"] = regime_stats["dp_regime"].map(REGIME_LABELS)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {
        "Stable": "#2ECC71", "Building": "#F1C40F",
        "Escalating": "#E67E22", "Panic": "#E74C3C", "Recovery": "#3498DB",
    }
    colors = [palette.get(l, "#95A5A6") for l in regime_stats["label"]]

    bars = ax.bar(regime_stats["label"], regime_stats["win_rate"],
                  color=colors, width=0.5, edgecolor="white")
    for bar, row in zip(bars, regime_stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{row.win_rate:.2f}\n(n={row.n:,})", ha="center", fontsize=9)

    ax.axhline(0.5, ls="--", color="grey", alpha=0.6, label="50% baseline")
    ax.set_ylabel("Chase Win Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Chase Win Rate by Pressure Regime\n"
                 "(Stable → Building → Escalating → Panic / Recovery)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "25_regime_win_rates.png")


# ---------------------------------------------------------------------------
# 26. Pressure trend trajectories (median + IQR ribbon)
# ---------------------------------------------------------------------------
def plot_pressure_trend_trajectories(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, metric, title in [
        (axes[0], "pressure",     "Static Pressure"),
        (axes[1], "dp_ewm_fast",  "Dynamic Pressure (EWM fast)"),
    ]:
        for outcome, color, label in [(1, "#2ECC71", "Won"), (0, "#E74C3C", "Lost")]:
            sub = df[df["won"] == outcome].groupby("over")[metric]
            med = sub.median()
            q25 = sub.quantile(0.25)
            q75 = sub.quantile(0.75)

            ax.plot(med.index, med.values, color=color, linewidth=2.5, label=label)
            ax.fill_between(med.index, q25.values, q75.values,
                            alpha=0.15, color=color)

        ax.set_xlabel("Over")
        ax.set_ylabel(metric)
        ax.set_title(f"{title} Trajectory — Median ± IQR", fontweight="bold")
        ax.legend()
        ax.axhline(1.0, ls="--", color="grey", alpha=0.4)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    fig.suptitle("Pressure Trajectories: Static vs Dynamic (EWM)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "26_pressure_trend_trajectories.png")


# ---------------------------------------------------------------------------
# 27. Collapse timeline (pressure window around 2+ wickets in 12 balls)
# ---------------------------------------------------------------------------
def plot_collapse_timeline(df: pd.DataFrame) -> None:
    """
    For each 'collapse event' (collapse_indicator first goes 1 in a match),
    extract a ±15 ball window of pressure and average across events.
    """
    window = 15
    won_windows, lost_windows = [], []

    for match_id, grp in df.groupby("match_id"):
        grp = grp.sort_values("balls_bowled").reset_index(drop=True)
        collapse_idx = grp.index[
            (grp["collapse_indicator"] == 1) &
            (grp["collapse_indicator"].shift(1).fillna(0) == 0)
        ].tolist()

        if not collapse_idx:
            continue

        first_collapse = collapse_idx[0]
        press = grp["pressure"].values
        won   = grp["won"].iloc[0]

        lo = max(0, first_collapse - window)
        hi = min(len(press), first_collapse + window + 1)
        snippet = press[lo:hi]

        # Pad to full window
        padded = np.full(2 * window + 1, np.nan)
        offset = first_collapse - lo
        start  = window - offset
        padded[start: start + len(snippet)] = snippet

        (won_windows if won == 1 else lost_windows).append(padded)

    if not won_windows and not lost_windows:
        print("  [momentum_analysis] No collapse events found — skipping plot 27.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(-window, window + 1)

    for windows, color, label in [
        (won_windows,  "#2ECC71", f"Chase Won  ({len(won_windows)} events)"),
        (lost_windows, "#E74C3C", f"Chase Lost ({len(lost_windows)} events)"),
    ]:
        if not windows:
            continue
        mat = np.array(windows)
        med = np.nanmedian(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)

        ax.plot(x, med, color=color, linewidth=2.5, label=label)
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)

    ax.axvline(0, ls="--", color="black", alpha=0.7, label="Collapse onset (ball 0)")
    ax.axhline(1.0, ls=":", color="grey", alpha=0.4)
    ax.set_xlabel("Balls relative to collapse onset")
    ax.set_ylabel("Pressure (median ± IQR)")
    ax.set_title("Collapse Timeline — Pressure Window Around Wicket Burst Events\n"
                 "(collapse = 2+ wickets in 12 balls)", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "27_collapse_timeline.png")


# ---------------------------------------------------------------------------
# 28. Panic-zone map (pressure × momentum → win rate)
# ---------------------------------------------------------------------------
def plot_panic_zone_map(df: pd.DataFrame) -> None:
    """
    2D heatmap of (pressure_bin × momentum_bin) → chase win rate.
    Reveals 'panic zones' (top-right) and 'safe zones' (bottom-left).
    """
    df_plot = df.copy()
    df_plot["p_bin"]   = pd.cut(df_plot["pressure"].clip(0, 10),
                                 bins=10, labels=False)
    df_plot["mom_bin"] = pd.cut(df_plot["dp_momentum"].clip(-2, 2),
                                 bins=8, labels=False)

    pivot = df_plot.pivot_table(
        values="won", index="mom_bin", columns="p_bin",
        aggfunc="mean"
    )
    # Fill NaN with row means before smoothing to avoid nan propagation
    mat    = pivot.values.astype(float)
    row_means = np.nanmean(mat, axis=1, keepdims=True)
    mat    = np.where(np.isnan(mat), row_means, mat)
    smooth = gaussian_filter(mat, sigma=0.8)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(smooth, cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower", aspect="auto", interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Chase Win Rate", fontsize=11)

    # Axis labels
    p_labels  = [f"{v:.1f}" for v in np.linspace(0, 10, 10)]
    m_labels  = [f"{v:.1f}" for v in np.linspace(-2, 2, 8)]
    ax.set_xticks(range(len(p_labels)))
    ax.set_xticklabels(p_labels, fontsize=9)
    ax.set_yticks(range(len(m_labels)))
    ax.set_yticklabels(m_labels, fontsize=9)

    ax.set_xlabel("Static Pressure (bins, 0=low → 10=max)")
    ax.set_ylabel("Pressure Momentum (negative=easing, positive=rising)")
    ax.set_title("PANIC ZONE MAP — Chase Win Rate by (Pressure × Momentum)\n"
                 "Green = safe zone  |  Red = panic zone  |  Yellow = neutral",
                 fontsize=12, fontweight="bold")

    # Annotate regions
    ax.text(1.5, 6.5, "SAFE\nZONE", ha="center", fontsize=11,
            color="darkgreen", fontweight="bold")
    ax.text(8.0, 6.5, "ESCALATING\nPANIC", ha="center", fontsize=9,
            color="darkred", fontweight="bold")
    ax.text(8.0, 0.5, "PANIC\n(STATIC)", ha="center", fontsize=9,
            color="firebrick", fontweight="bold")
    ax.text(2.0, 0.5, "RECOVERY\nZONE", ha="center", fontsize=9,
            color="navy", fontweight="bold")

    fig.tight_layout()
    _save(fig, "28_panic_zone_map.png")


# ---------------------------------------------------------------------------
# 29. Dynamic vs static pressure comparison (KDE by outcome)
# ---------------------------------------------------------------------------
def plot_dynamic_vs_static(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, 10, 50)

    for ax, col, title in [
        (axes[0], "pressure",    "Static Pressure"),
        (axes[1], "dp_ewm_fast", "Dynamic Pressure (EWM fast)"),
    ]:
        for outcome, color, label in [(1, "#2ECC71", "Won"), (0, "#E74C3C", "Lost")]:
            data = df[df["won"] == outcome][col].clip(0, 10).values
            ax.hist(data, bins=bins, color=color, alpha=0.5, label=label,
                    density=True, histtype="stepfilled")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Pressure Value")
        ax.legend()
        ax.set_xlim(0, 10)

    fig.suptitle("Static vs Dynamic Pressure — Distribution by Chase Outcome\n"
                 "(better separation = more discriminative)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "29_dynamic_vs_static_pressure.png")


# ---------------------------------------------------------------------------
# 30. Wicket-pressure interaction
# ---------------------------------------------------------------------------
def plot_wicket_pressure_interaction(df: pd.DataFrame) -> None:
    """
    For each wicket number (1st–10th), show the distribution of pressure
    at the moment that wicket falls, split by final outcome.
    """
    if "is_wicket" not in df.columns:
        print("  [momentum_analysis] is_wicket not present — skipping plot 30.")
        return

    wkt_df = df[df["is_wicket"] == 1].copy()
    wkt_df["wicket_number"] = wkt_df["wickets_lost"].clip(1, 10)
    wkt_df["Outcome"] = wkt_df["won"].map({1: "Won", 0: "Lost"})

    # Use mean ± std bar chart instead of boxplot (much faster)
    fig, ax = plt.subplots(figsize=(13, 5))
    for outcome, color in [("Won", "#2ECC71"), ("Lost", "#E74C3C")]:
        sub = wkt_df[wkt_df["Outcome"] == outcome].groupby("wicket_number")["pressure"]
        means = sub.mean()
        stds  = sub.std().fillna(0)
        x = np.array(means.index) + (-0.15 if outcome == "Won" else 0.15)
        ax.bar(x, means.values, width=0.3, color=color, alpha=0.8, label=outcome)
        ax.errorbar(x, means.values, yerr=stds.values, fmt="none",
                    color="black", capsize=4, linewidth=1)

    ax.set_xlabel("Wicket Number")
    ax.set_ylabel("Mean Pressure at Wicket Fall")
    ax.set_title("Pressure at Each Wicket Fall — Chase Won vs Lost\n"
                 "(mean ± std; rising pressure at later wickets = mounting difficulty)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.legend(title="Outcome")
    fig.tight_layout()
    _save(fig, "30_wicket_pressure_interaction.png")


def run_momentum_analysis(df: pd.DataFrame) -> None:
    import sys
    print("\n[momentum_analysis] Generating momentum & dynamic pressure plots...")
    sys.stdout.flush()

    for fn, name in [
        (plot_momentum_vs_winprob,         "24_momentum_vs_winprob"),
        (plot_regime_win_rates,             "25_regime_win_rates"),
        (plot_pressure_trend_trajectories,  "26_pressure_trend_trajectories"),
        (plot_collapse_timeline,            "27_collapse_timeline"),
        (plot_panic_zone_map,               "28_panic_zone_map"),
        (plot_dynamic_vs_static,            "29_dynamic_vs_static"),
        (plot_wicket_pressure_interaction,  "30_wicket_pressure"),
    ]:
        print(f"  [momentum_analysis] Generating {name}...", end=" ", flush=True)
        try:
            fn(df)
            print("done", flush=True)
        except Exception as e:
            print(f"SKIPPED ({e})", flush=True)

    print("[momentum_analysis] All plots saved.", flush=True)
