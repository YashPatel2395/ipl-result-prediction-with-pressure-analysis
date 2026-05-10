"""
analysis.py
===========
Core research layer — answers the question:
"Is pressure a meaningful predictor of IPL chase outcomes?"

Generates:
  - Feature importance plots (RF + XGBoost)
  - Correlation heatmap
  - Pressure progression curves
  - Collapse analysis (wicket-fall pressure spikes)
  - Win probability as a function of pressure
  - SHAP summary (if shap is installed)
"""

import os
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

from src.modeling import NUMERIC_FEATURES, CATEGORICAL_FEATURES

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [analysis] Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "pressure", "rate_ratio", "wicket_factor", "time_factor",
        "current_run_rate", "required_run_rate", "wickets_lost",
        "wickets_in_hand", "balls_remaining", "runs_remaining",
        "over", "won",
    ]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=0.5, mask=~mask | np.eye(len(cols), dtype=bool))
    # Unmask diagonal
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=0.5, cbar=False)
    ax.set_title("Feature Correlation Heatmap (incl. target: won)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 2. Feature importance — Random Forest (native)
# ---------------------------------------------------------------------------
def plot_rf_feature_importance(rf_pipeline: Pipeline, df: pd.DataFrame) -> None:
    pre      = rf_pipeline.named_steps["pre"]
    clf      = rf_pipeline.named_steps["clf"]

    # Reconstruct feature names after one-hot encoding
    ohe_cols = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_cols = list(NUMERIC_FEATURES) + list(ohe_cols)

    importances = clf.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": all_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#E74C3C" if "pressure" in f else "#3498DB" for f in imp_df["feature"]]
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Random Forest — Top 15 Feature Importances\n"
                 "(red = pressure-related features)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "12_rf_feature_importance.png")

    return imp_df


# ---------------------------------------------------------------------------
# 3. Feature importance — XGBoost
# ---------------------------------------------------------------------------
def plot_xgb_feature_importance(xgb_pipeline: Pipeline, df: pd.DataFrame) -> None:
    pre      = xgb_pipeline.named_steps["pre"]
    clf      = xgb_pipeline.named_steps["clf"]

    ohe_cols = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_cols = list(NUMERIC_FEATURES) + list(ohe_cols)

    importances = clf.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": all_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#E74C3C" if "pressure" in f else "#E67E22" for f in imp_df["feature"]]
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost — Top 15 Feature Importances\n"
                 "(red = pressure-related features)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "13_xgb_feature_importance.png")

    return imp_df


# ---------------------------------------------------------------------------
# 4. Pressure progression during innings (sample matches)
# ---------------------------------------------------------------------------
def plot_pressure_progression(df: pd.DataFrame, n_matches: int = 6) -> None:
    """
    Show how pressure evolves ball-by-ball in a sample of matches,
    coloured by final outcome.
    """
    won_matches  = df[df["won"] == 1]["match_id"].unique()[:n_matches // 2]
    lost_matches = df[df["won"] == 0]["match_id"].unique()[:n_matches // 2]
    sample_ids   = list(won_matches) + list(lost_matches)

    fig, axes = plt.subplots(2, n_matches // 2, figsize=(5 * (n_matches // 2), 8),
                             sharey=False)

    for idx, match_id in enumerate(sample_ids):
        row_i, col_i = divmod(idx, n_matches // 2)
        ax = axes[row_i][col_i]

        mdf     = df[df["match_id"] == match_id].sort_values("balls_bowled")
        outcome = "WON" if mdf["won"].iloc[0] == 1 else "LOST"
        color   = "#2ECC71" if outcome == "WON" else "#E74C3C"

        ax.plot(mdf["balls_bowled"], mdf["pressure"], color=color, linewidth=1.5)
        ax.fill_between(mdf["balls_bowled"], mdf["pressure"], alpha=0.15, color=color)
        ax.axhline(1.0, ls="--", color="grey", alpha=0.5, linewidth=1)

        # Mark wicket-fall events
        wicket_balls = mdf[mdf["is_wicket"] == 1] if "is_wicket" in mdf.columns else pd.DataFrame()
        if not wicket_balls.empty:
            ax.scatter(wicket_balls["balls_bowled"], wicket_balls["pressure"],
                       color="black", s=25, zorder=5, marker="v")

        ax.set_title(f"Match {match_id} — {outcome}", fontsize=10, color=color, fontweight="bold")
        ax.set_xlabel("Ball")
        ax.set_ylabel("Pressure")
        ax.set_ylim(bottom=0)

    fig.suptitle("Pressure Progression — Sample Matches\n"
                 "(▼ = wicket fall, dashed line = balanced pressure)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "14_pressure_progression.png")


# ---------------------------------------------------------------------------
# 5. Win probability vs pressure (model-based)
# ---------------------------------------------------------------------------
def plot_win_prob_vs_pressure(
    model: Pipeline,
    X_test: pd.DataFrame,
    df_test_meta: pd.DataFrame,
) -> None:
    """
    Bin pressure into quantiles and plot model-estimated win probability.
    """
    proba = model.predict_proba(X_test)[:, 1]
    analysis_df = pd.DataFrame({
        "pressure": df_test_meta["pressure"].values,
        "win_prob": proba,
        "actual":   df_test_meta["won"].values,
    })

    analysis_df["pressure_bin"] = pd.qcut(analysis_df["pressure"], q=10, duplicates="drop")
    bin_stats = analysis_df.groupby("pressure_bin").agg(
        mean_prob   = ("win_prob", "mean"),
        actual_rate = ("actual",   "mean"),
        n           = ("actual",   "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(bin_stats))
    ax.bar(x, bin_stats["mean_prob"],   alpha=0.6, color="#3498DB", label="Model win prob")
    ax.plot(x, bin_stats["actual_rate"], "ro--", ms=7, linewidth=2, label="Actual win rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bin_stats["pressure_bin"]], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Pressure Bin (quantile)")
    ax.set_ylabel("Win Probability / Rate")
    ax.set_title("Model Win Probability vs Actual Win Rate Across Pressure Bins",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.5, ls="--", color="grey", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _save(fig, "15_win_prob_vs_pressure.png")


# ---------------------------------------------------------------------------
# 6. Collapse analysis — pressure spike at wicket fall
# ---------------------------------------------------------------------------
def plot_collapse_analysis(df: pd.DataFrame) -> None:
    """
    Show average pressure *before* and *after* wicket falls,
    grouped by innings position and outcome.
    """
    if "is_wicket" not in df.columns:
        print("  [analysis] is_wicket not available — skipping collapse plot.")
        return

    # For each ball, compute pressure delta at wicket events
    wicket_rows = df[df["is_wicket"] == 1].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: pressure at wicket fall by over
    for outcome, grp in wicket_rows.groupby("won"):
        label = "Won" if outcome == 1 else "Lost"
        color = "#2ECC71" if outcome == 1 else "#E74C3C"
        axes[0].scatter(grp["over"], grp["pressure"], alpha=0.3, color=color, s=10, label=label)
        # Smoothed mean
        over_mean = grp.groupby("over")["pressure"].mean()
        axes[0].plot(over_mean.index, over_mean.values, color=color, linewidth=2)

    axes[0].set_xlabel("Over")
    axes[0].set_ylabel("Pressure at Wicket Fall")
    axes[0].set_title("Pressure at Wicket Falls — Won vs Lost", fontweight="bold")
    axes[0].legend()

    # Right: distribution of wicket-fall pressure by outcome
    wicket_rows["Outcome"] = wicket_rows["won"].map({1: "Won", 0: "Lost"})
    sns.kdeplot(data=wicket_rows, x="pressure", hue="Outcome",
                palette={"Won": "#2ECC71", "Lost": "#E74C3C"},
                fill=True, alpha=0.3, ax=axes[1])
    axes[1].set_xlabel("Pressure")
    axes[1].set_title("Pressure Distribution at Wicket Falls", fontweight="bold")
    axes[1].axvline(wicket_rows[wicket_rows["won"] == 0]["pressure"].median(),
                    color="#E74C3C", ls="--", alpha=0.7, label="Median (Lost)")
    axes[1].axvline(wicket_rows[wicket_rows["won"] == 1]["pressure"].median(),
                    color="#2ECC71", ls="--", alpha=0.7, label="Median (Won)")

    fig.suptitle("Wicket-Fall (Collapse) Pressure Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "16_collapse_analysis.png")


# ---------------------------------------------------------------------------
# 7. SHAP summary (optional)
# ---------------------------------------------------------------------------
def plot_shap_summary(xgb_pipeline: Pipeline, X_test: pd.DataFrame) -> None:
    try:
        import shap
    except ImportError:
        print("  [analysis] shap not installed — skipping SHAP plot.")
        return

    pre      = xgb_pipeline.named_steps["pre"]
    clf      = xgb_pipeline.named_steps["clf"]
    X_trans  = pre.transform(X_test)

    ohe_cols = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    feat_names = list(NUMERIC_FEATURES) + list(ohe_cols)

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_trans, feature_names=feat_names,
                      show=False, max_display=15)
    plt.title("SHAP Feature Impact — XGBoost", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(plt.gcf(), "17_shap_summary.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_analysis(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_meta: pd.DataFrame,
) -> None:
    print("\n[analysis] Running pressure & interpretability analysis...")

    plot_correlation_heatmap(df)

    if "random_forest" in models:
        plot_rf_feature_importance(models["random_forest"], df)

    if "xgboost" in models:
        plot_xgb_feature_importance(models["xgboost"], df)

    # Add is_wicket back for progression if available
    plot_pressure_progression(df)
    plot_collapse_analysis(df)

    best_model = models.get("xgboost") or list(models.values())[0]
    plot_win_prob_vs_pressure(best_model, X_test, df_test_meta)

    if "xgboost" in models:
        plot_shap_summary(models["xgboost"], X_test)

    print("[analysis] All analysis plots saved.")
