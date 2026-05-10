"""
baseline.py
===========
Ablation study: train models WITH and WITHOUT the pressure feature set.

This directly answers: "Does the pressure metric add predictive value beyond
the raw ball-state features that an analyst already knows?"

WITHOUT pressure — features available to any ball-by-ball analyst:
    current_score, runs_remaining, balls_remaining, wickets_lost,
    wickets_in_hand, current_run_rate, required_run_rate, over,
    phase, toss_decision

WITH pressure — same features + engineered pressure signal:
    + pressure, rate_ratio, wicket_factor, time_factor

All models are trained on the same match-level train/test split to ensure
a fair comparison. XGBoost is used for speed; all three model types are
also compared within each condition.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

# Feature groups
BASE_NUMERIC = [
    "current_score", "runs_remaining", "balls_remaining",
    "wickets_lost", "wickets_in_hand",
    "current_run_rate", "required_run_rate", "over",
]
PRESSURE_NUMERIC = [
    "pressure", "rate_ratio", "wicket_factor", "time_factor",
]
CATEGORICAL = ["phase", "toss_decision"]

FEATURE_SETS = {
    "without_pressure": BASE_NUMERIC + CATEGORICAL,
    "with_pressure":    BASE_NUMERIC + PRESSURE_NUMERIC + CATEGORICAL,
}


def _preprocessor(num_cols: list[str], cat_cols: list[str], scale: bool) -> ColumnTransformer:
    num_pipe = Pipeline([("sc", StandardScaler())]) if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def _metrics(pipe, X_te, y_te) -> dict:
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_te, y_pred),   4),
        "f1":       round(f1_score(y_te, y_pred),         4),
        "roc_auc":  round(roc_auc_score(y_te, y_proba),   4),
    }


def _train_one(model_type: str, num_cols: list[str], X_tr, y_tr, X_te, y_te) -> dict:
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_w  = neg / pos if pos > 0 else 1.0

    if model_type == "logistic":
        clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced",
                                 solver="lbfgs", random_state=42)
        scale = True
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=50,
                                     class_weight="balanced", n_jobs=-1, random_state=42)
        scale = False
    else:  # xgboost
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale_w, eval_metric="logloss",
                            use_label_encoder=False, random_state=42, n_jobs=-1)
        scale = False

    cat_cols = CATEGORICAL
    pre  = _preprocessor(num_cols, cat_cols, scale)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    return _metrics(pipe, X_te, y_te)


def run_ablation(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    target: str = "won",
) -> pd.DataFrame:
    """
    Full ablation study. Returns a long-form DataFrame with one row per
    (feature_set × model_type) combination.
    """
    records = []

    for condition, all_cols in FEATURE_SETS.items():
        num_cols = [c for c in all_cols if c not in CATEGORICAL]
        feature_cols = all_cols  # num + cat

        X_tr = df.loc[train_idx, feature_cols]
        X_te = df.loc[test_idx,  feature_cols]
        y_tr = df.loc[train_idx, target]
        y_te = df.loc[test_idx,  target]

        for model_type in ["logistic", "random_forest", "xgboost"]:
            m = _train_one(model_type, num_cols, X_tr, y_tr, X_te, y_te)
            records.append({
                "condition":  condition,
                "model":      model_type,
                **m,
            })
            print(f"  [{condition}] [{model_type:15s}]  "
                  f"AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

    results = pd.DataFrame(records)

    print("\n[baseline] Ablation results:")
    pivot = results.pivot_table(index="model", columns="condition", values="roc_auc")
    pivot["delta_auc"] = pivot["with_pressure"] - pivot["without_pressure"]
    print(pivot.round(4).to_string())

    # Save CSV
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(OUTPUTS_DIR, "ablation_results.csv"), index=False)

    _plot_ablation(results)
    return results


def _plot_ablation(results: pd.DataFrame) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["roc_auc", "f1", "accuracy"]
    titles  = ["ROC-AUC", "F1 Score", "Accuracy"]

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = results.pivot_table(index="model", columns="condition", values=metric)
        x     = np.arange(len(pivot))
        w     = 0.35

        bars_wo = ax.bar(x - w/2, pivot["without_pressure"], width=w,
                         color="#95A5A6", label="Without Pressure", alpha=0.9)
        bars_wi = ax.bar(x + w/2, pivot["with_pressure"],    width=w,
                         color="#3498DB", label="With Pressure",    alpha=0.9)

        # Delta annotation
        for i, (wo, wi) in enumerate(zip(pivot["without_pressure"], pivot["with_pressure"])):
            delta = wi - wo
            color = "#2ECC71" if delta > 0 else "#E74C3C"
            ax.annotate(f"Δ{delta:+.4f}", xy=(i + w/2, wi), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=15)
        ax.set_title(f"Ablation — {title}", fontweight="bold")
        ax.set_ylabel(title)
        ymin = max(0, results[metric].min() - 0.03)
        ax.set_ylim(ymin, min(1.0, results[metric].max() + 0.05))
        if ax == axes[0]:
            ax.legend()

    fig.suptitle("Ablation Study: With vs Without Pressure Features\n"
                 "(Δ shows gain from adding pressure — positive = improvement)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "20_ablation_with_vs_without_pressure.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [baseline] Saved → {path}")


def pressure_independence_test(df: pd.DataFrame) -> None:
    """
    Test whether pressure adds information beyond required_run_rate alone.

    Method: compare a model using only RRR vs a model using pressure + RRR.
    If pressure_model AUC > rrr_only_model AUC, pressure has independent value.
    """
    from scipy import stats

    print("\n[baseline] Pressure independence: partial correlation analysis")

    # Pearson correlation: pressure ~ won, rrr ~ won, pressure ~ rrr
    r_pressure_won = df["pressure"].corr(df["won"])
    r_rrr_won      = df["required_run_rate"].clip(upper=50).corr(df["won"])
    r_pressure_rrr = df["pressure"].corr(df["required_run_rate"].clip(upper=50))

    # Partial correlation: pressure with outcome, controlling for RRR
    # Formula: r_pyw.z = (r_py - r_pz*r_zy) / sqrt((1-r_pz²)(1-r_zy²))
    r_p   = r_pressure_won
    r_z   = r_rrr_won
    r_pz  = r_pressure_rrr
    partial_corr = (r_p - r_pz * r_z) / (np.sqrt((1 - r_pz**2) * (1 - r_z**2)) + 1e-9)

    # Mann-Whitney U test on pressure distributions
    won_pressure  = df[df["won"] == 1]["pressure"]
    lost_pressure = df[df["won"] == 0]["pressure"]
    u_stat, p_val = stats.mannwhitneyu(won_pressure, lost_pressure, alternative="less")

    print(f"  Pearson corr  — pressure ↔ outcome:    {r_pressure_won:.4f}")
    print(f"  Pearson corr  — RRR     ↔ outcome:    {r_rrr_won:.4f}")
    print(f"  Partial corr  — pressure ↔ outcome    ")
    print(f"                  (controlling for RRR): {partial_corr:.4f}")
    print(f"  Mann-Whitney U (pressure won < lost):  U={u_stat:.0f}, p={p_val:.2e}")
    print(f"  → Pressure adds {'INDEPENDENT' if abs(partial_corr) > 0.05 else 'MARGINAL'} "
          f"predictive signal beyond RRR alone.")

    return {
        "corr_pressure_won": r_pressure_won,
        "corr_rrr_won":      r_rrr_won,
        "partial_corr":      partial_corr,
        "mann_whitney_p":    p_val,
    }
