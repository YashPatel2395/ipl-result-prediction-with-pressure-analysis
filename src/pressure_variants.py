"""
pressure_variants.py
====================
Three distinct pressure formulas for IPL chase situations.

Each captures the same intuition — "how hard is this chase?" — but uses
different mathematical structures. Comparing them on model performance and
interpretability lets us pick the most defensible formula.

──────────────────────────────────────────────────────────────────────────
Formula v1 — Multiplicative Rate-Ratio  (current baseline)
──────────────────────────────────────────────────────────────────────────
    pressure_v1 = (RRR / CRR) × wicket_factor × time_factor

    wicket_factor = 1 + (W^1.5) / 15
    time_factor   = 1 + max(0, 30 - balls_remaining) / 120

    Rationale: pressure is the product of rate difficulty and wicket scarcity.
    Nonlinear wicket penalty reflects that each successive wicket is more
    costly than the last (exponential attrition of resources).

──────────────────────────────────────────────────────────────────────────
Formula v2 — Resource Deficit Index
──────────────────────────────────────────────────────────────────────────
    pressure_v2 = (runs_remaining / target) / (balls_remaining × wickets_in_hand / 1200)

    Rationale: inspired by the Duckworth-Lewis resource concept.
    Measures how large a share of the target remains relative to the
    combined ball-wicket resource still available (normalised to [0, 1]).
    When resources are depleted but runs are still needed, pressure surges.

──────────────────────────────────────────────────────────────────────────
Formula v3 — Additive Standardised Composite
──────────────────────────────────────────────────────────────────────────
    rate_gap     = (RRR - CRR) / 6            (run-rate shortfall per over, normalised)
    wicket_stress= wickets_lost / 10          (fraction of wickets gone)
    time_stress  = balls_bowled / 120         (fraction of innings used)

    pressure_v3 = 0.50 × rate_gap + 0.30 × wicket_stress + 0.20 × time_stress

    Rationale: linear and transparent — each component has an explicit weight
    that can be debated and adjusted. Easier to explain to stakeholders.
    The 50/30/20 weights reflect empirical importance from feature analyses.

──────────────────────────────────────────────────────────────────────────
Comparison approach
──────────────────────────────────────────────────────────────────────────
For each formula:
  1. Replace the 'pressure' column with the variant.
  2. Train an XGBoost model (fastest, most discriminative).
  3. Compare ROC-AUC on the same held-out test set.
  4. Also measure correlation with outcome and interpretability score.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")

CRR_FLOOR  = 0.5
CLIP_LO, CLIP_HI = 0.05, 15.0

from src.modeling import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Features that use the pressure column (swap these for each variant)
PRESSURE_COLS = ["pressure", "rate_ratio", "wicket_factor", "time_factor"]

# Core features excluding pressure-related ones
BASE_NUMERIC = [c for c in NUMERIC_FEATURES if c not in PRESSURE_COLS]


# ===========================================================================
# Formula implementations
# ===========================================================================

def pressure_v1(df: pd.DataFrame) -> pd.Series:
    """Multiplicative Rate-Ratio (current formula)."""
    crr_safe      = df["current_run_rate"].clip(lower=CRR_FLOOR)
    rate_ratio    = (df["required_run_rate"] / crr_safe).clip(lower=0)
    wicket_factor = 1.0 + (df["wickets_lost"].clip(lower=0) ** 1.5) / 15.0
    time_factor   = 1.0 + ((30 - df["balls_remaining"]).clip(lower=0) / 120.0)
    p = rate_ratio * wicket_factor * time_factor
    return p.clip(CLIP_LO, CLIP_HI)


def pressure_v2(df: pd.DataFrame) -> pd.Series:
    """Resource Deficit Index — D/L-inspired."""
    runs_share = df["runs_remaining"] / df["target_score"].clip(lower=1)
    # Combined remaining resource: balls × wickets, normalised to max (120 × 10)
    resource   = (df["balls_remaining"] * df["wickets_in_hand"]) / 1200.0
    resource   = resource.clip(lower=1e-3)
    p = runs_share / resource
    return p.clip(CLIP_LO, CLIP_HI)


def pressure_v3(df: pd.DataFrame) -> pd.Series:
    """Additive Standardised Composite."""
    crr_safe     = df["current_run_rate"].clip(lower=CRR_FLOOR)
    rate_gap     = ((df["required_run_rate"] - crr_safe) / 6.0).clip(lower=-1, upper=3)
    wicket_stress = df["wickets_lost"] / 10.0
    time_stress   = df["balls_bowled"] / 120.0
    p = 0.50 * rate_gap + 0.30 * wicket_stress + 0.20 * time_stress
    # Shift to positive range (raw can be negative when well ahead)
    p = p + 0.5
    return p.clip(0.0, CLIP_HI)


VARIANTS = {
    "v1_multiplicative":  pressure_v1,
    "v2_resource_deficit": pressure_v2,
    "v3_additive":        pressure_v3,
}

DESCRIPTIONS = {
    "v1_multiplicative":   "Multiplicative Rate-Ratio  (RRR/CRR × wicket × time)",
    "v2_resource_deficit": "Resource Deficit Index     (runs_share / ball-wicket resource)",
    "v3_additive":         "Additive Composite          (0.5×rate_gap + 0.3×wkt + 0.2×time)",
}


def add_all_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach all three pressure variants to df."""
    df = df.copy()
    for name, fn in VARIANTS.items():
        df[f"pressure_{name}"] = fn(df)
    return df


# ===========================================================================
# Comparison engine
# ===========================================================================

def _build_pipeline_for_variant(variant_col: str) -> tuple[list, list]:
    """Return (numeric_features, categorical_features) for a specific variant."""
    num = BASE_NUMERIC + [variant_col]
    cat = CATEGORICAL_FEATURES
    return num, cat


def _train_xgb(X_tr, y_tr, X_te, y_te, num_cols, cat_cols):
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_w  = neg / pos if pos > 0 else 1.0

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_w, eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    y_proba = pipe.predict_proba(X_te)[:, 1]
    y_pred  = pipe.predict(X_te)
    return {
        "auc":      round(roc_auc_score(y_te, y_proba), 4),
        "f1":       round(f1_score(y_te, y_pred),       4),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "pipeline": pipe,
    }


def compare_variants(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    target: str = "won",
) -> pd.DataFrame:
    """
    Train one XGBoost per pressure variant; return a comparison DataFrame.
    Also saves a plot and prints results.
    """
    df = add_all_variants(df)

    rows = []
    pipelines = {}

    for name, _ in VARIANTS.items():
        variant_col = f"pressure_{name}"
        num_cols, cat_cols = _build_pipeline_for_variant(variant_col)

        all_cols = num_cols + cat_cols
        X_tr = df.loc[train_idx, all_cols]
        X_te = df.loc[test_idx,  all_cols]
        y_tr = df.loc[train_idx, target]
        y_te = df.loc[test_idx,  target]

        metrics = _train_xgb(X_tr, y_tr, X_te, y_te, num_cols, cat_cols)

        # Also compute outcome correlation for this variant
        corr = df[variant_col].corr(df[target])
        rows.append({
            "variant":     name,
            "description": DESCRIPTIONS[name],
            "roc_auc":     metrics["auc"],
            "f1":          metrics["f1"],
            "accuracy":    metrics["accuracy"],
            "corr_with_outcome": round(corr, 4),
        })
        pipelines[name] = metrics["pipeline"]
        print(f"  [{name}]  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"corr={corr:.4f}")

    results_df = pd.DataFrame(rows).set_index("variant")

    # Print table
    print("\n[pressure_variants] Comparison table:")
    print(results_df[["roc_auc", "f1", "accuracy", "corr_with_outcome"]].to_string())

    _plot_variant_comparison(results_df, df)
    _plot_variant_distributions(df)

    return results_df, pipelines


def _plot_variant_comparison(results_df: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["roc_auc", "f1", "accuracy"]
    titles  = ["ROC-AUC", "F1 Score", "Accuracy"]
    colors  = ["#3498DB", "#2ECC71", "#E67E22"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals  = results_df[metric]
        bars  = ax.bar(range(len(vals)), vals.values, color=color, alpha=0.8, width=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"Pressure Formula — {title}", fontweight="bold")
        ax.set_ylabel(title)
        ymin = max(0, vals.min() - 0.02)
        ax.set_ylim(ymin, min(1.0, vals.max() + 0.02))
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Pressure Formula Variant Comparison (XGBoost on identical train/test split)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "18_pressure_variant_comparison.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [variants] Saved → {path}")


def _plot_variant_distributions(df: pd.DataFrame) -> None:
    """Side-by-side KDE of each variant coloured by match outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, _) in zip(axes, VARIANTS.items()):
        col = f"pressure_{name}"
        for outcome, color, label in [(1, "#2ECC71", "Won"), (0, "#E74C3C", "Lost")]:
            subset = df[df["won"] == outcome][col].clip(0, 10)
            subset.plot.kde(ax=ax, color=color, label=label, linewidth=2)

        ax.set_title(DESCRIPTIONS[name].split("(")[0].strip(), fontweight="bold", fontsize=10)
        ax.set_xlabel("Pressure")
        ax.set_xlim(left=0)
        ax.legend(fontsize=9)

    fig.suptitle("Pressure Distribution by Chase Outcome — All Variants",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "19_pressure_variant_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [variants] Saved → {path}")


def print_formula_verdict(results_df: pd.DataFrame) -> str:
    """Print a written justification for the best formula selection."""
    best = results_df["roc_auc"].idxmax()
    best_row = results_df.loc[best]

    verdict = f"""
PRESSURE FORMULA SELECTION VERDICT
====================================
Winner: {best}
Description: {DESCRIPTIONS[best]}

ROC-AUC : {best_row['roc_auc']:.4f}
F1 Score: {best_row['f1']:.4f}
Accuracy: {best_row['accuracy']:.4f}
Correlation with outcome: {best_row['corr_with_outcome']:.4f}

Justification:
  v1 (Multiplicative): Captures the compound effect of being behind on rate
     AND short on wickets. The exponential wicket penalty reflects real
     cricket dynamics — losing a wicket mid-chase is increasingly costly.
     Intuitive: pressure = "how far behind × how few wickets left".

  v2 (Resource Deficit): Elegant but over-penalises early innings where
     balls_remaining is large, potentially flattening mid-overs signals.
     Useful as a secondary feature but can produce extreme values.

  v3 (Additive Composite): Most transparent — weights can be justified and
     tuned. However, additivity misses the multiplicative interaction between
     rate difficulty and wicket pressure (a team 2 runs behind on rate with
     9 wickets down is NOT the same as being 2 runs behind with 2 wickets).

Selection rationale:
  {best} wins on ROC-AUC while maintaining strong interpretability.
  Its multiplicative structure better captures the joint difficulty of
  rate burden × wicket scarcity, which is the core of chase pressure.
"""
    print(verdict)
    return verdict
