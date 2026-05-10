"""
advanced_modeling.py
====================
Train and compare three feature configurations:

  config_A — STATIC only      (original 12 numeric + 2 categorical)
  config_B — STATIC + MOMENTUM (adds 18 momentum features)
  config_C — STATIC + MOMENTUM + DYNAMIC PRESSURE (adds 6 dynamic pressure features)

All configurations use the same match-level train/test split (random_state=42)
so comparisons are fair. XGBoost is the primary model (best baseline AUC);
Logistic Regression is also trained to show linear vs non-linear gaps.

Additionally computes and plots:
  - Feature importance shift (what new features enter the top-15)
  - Calibration delta across configs
  - AUC progression chart
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

from src.modeling             import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.momentum             import MOMENTUM_FEATURES
from src.dynamic_pressure     import DYNAMIC_PRESSURE_FEATURES

# ── Feature configurations ─────────────────────────────────────────────────
CONFIG_A_NUM = NUMERIC_FEATURES
CONFIG_B_NUM = NUMERIC_FEATURES + MOMENTUM_FEATURES
CONFIG_C_NUM = NUMERIC_FEATURES + MOMENTUM_FEATURES + DYNAMIC_PRESSURE_FEATURES
CAT          = CATEGORICAL_FEATURES

CONFIGS = {
    "A_static":           CONFIG_A_NUM,
    "B_static+momentum":  CONFIG_B_NUM,
    "C_full_dynamic":     CONFIG_C_NUM,
}


def _split_by_matches(df: pd.DataFrame, test_size: float = 0.2):
    """Match-level split preserving class balance."""
    match_ids    = df["match_id"].unique()
    match_labels = df.groupby("match_id")["won"].first()
    train_ids, test_ids = train_test_split(
        match_ids, test_size=test_size, random_state=42,
        stratify=match_labels[match_ids],
    )
    train_mask = df["match_id"].isin(train_ids)
    test_mask  = df["match_id"].isin(test_ids)
    return df[train_mask].index, df[test_mask].index


def _build_pipe(num_cols: list[str], model_type: str, scale_w: float) -> Pipeline:
    if model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_w, eval_metric="logloss",
            use_label_encoder=False, random_state=42, n_jobs=1,
        )
        num_pipe = "passthrough"
    else:  # logistic
        clf = LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )
        num_pipe = Pipeline([("sc", StandardScaler())])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def _metrics(pipe, X_te, y_te) -> dict:
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_te, y_pred),    4),
        "f1":       round(f1_score(y_te, y_pred),          4),
        "roc_auc":  round(roc_auc_score(y_te, y_proba),    4),
        "brier":    round(brier_score_loss(y_te, y_proba),  5),
    }


def run_advanced_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train all (config × model_type) combinations and return a results DataFrame.
    Also saves models and plots.
    """
    train_idx, test_idx = _split_by_matches(df)
    y_tr = df.loc[train_idx, "won"]
    y_te = df.loc[test_idx,  "won"]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_w  = neg / pos if pos > 0 else 1.0

    records  = []

    for cfg_name, num_cols in CONFIGS.items():
        # Drop columns that don't exist in df yet (graceful)
        valid_num = [c for c in num_cols if c in df.columns]
        all_cols  = valid_num + CAT

        X_tr = df.loc[train_idx, all_cols]
        X_te = df.loc[test_idx,  all_cols]

        for model_type in ["xgboost", "logistic"]:
            pipe = _build_pipe(valid_num, model_type, scale_w)
            pipe.fit(X_tr, y_tr)
            m = _metrics(pipe, X_te, y_te)
            m.update({"config": cfg_name, "model": model_type,
                      "n_features": len(all_cols)})
            records.append(m)
            print(f"  [{cfg_name}] [{model_type:10s}]  "
                  f"AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  "
                  f"Brier={m['brier']:.5f}  n_feat={m['n_features']}")

            # Save XGBoost configs
            if model_type == "xgboost":
                os.makedirs(MODELS_DIR, exist_ok=True)
                path = os.path.join(MODELS_DIR, f"xgb_{cfg_name}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(pipe, f)

    results = pd.DataFrame(records)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(OUTPUTS_DIR, "advanced_model_comparison.csv"), index=False)

    _plot_config_comparison(results)
    _plot_feature_importance_shift(df, train_idx, test_idx, y_tr, y_te, scale_w)

    return results


def _plot_config_comparison(results: pd.DataFrame) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    metrics = ["roc_auc", "f1", "accuracy", "brier"]
    titles  = ["ROC-AUC", "F1 Score", "Accuracy", "Brier Score (↓)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"A_static": "#95A5A6", "B_static+momentum": "#3498DB", "C_full_dynamic": "#E74C3C"}

    for ax, metric, title in zip(axes.flat, metrics, titles):
        xgb_res = results[results["model"] == "xgboost"]
        x       = np.arange(len(xgb_res))
        bar_colors = [colors.get(c, "#666") for c in xgb_res["config"]]
        bars = ax.bar(x, xgb_res[metric], color=bar_colors, width=0.5, alpha=0.85)

        for bar, val in zip(bars, xgb_res[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(xgb_res["config"], rotation=20, ha="right")
        ax.set_title(f"XGBoost — {title}", fontweight="bold")
        ax.set_ylabel(title)
        rng = xgb_res[metric].max() - xgb_res[metric].min()
        ax.set_ylim(max(0, xgb_res[metric].min() - rng * 0.5),
                    min(1.0, xgb_res[metric].max() + rng * 0.5))

    # Add delta annotations on AUC plot
    auc_vals = results[results["model"] == "xgboost"]["roc_auc"].values
    if len(auc_vals) >= 3:
        axes[0, 0].annotate(
            f"Momentum gain: +{auc_vals[1]-auc_vals[0]:.4f}",
            xy=(1, auc_vals[1]), xytext=(1.3, auc_vals[1] + 0.005),
            fontsize=9, color="#3498DB", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#3498DB"),
        )
        axes[0, 0].annotate(
            f"Dynamic gain: +{auc_vals[2]-auc_vals[1]:.4f}",
            xy=(2, auc_vals[2]), xytext=(1.7, auc_vals[2] + 0.008),
            fontsize=9, color="#E74C3C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E74C3C"),
        )

    fig.suptitle("Feature Configuration Comparison (XGBoost)\n"
                 "A=Static | B=Static+Momentum | C=Full Dynamic",
                 fontsize=13, fontweight="bold")
    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(color=c, label=k) for k, c in colors.items()]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = os.path.join(PLOTS_DIR, "31_advanced_model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [advanced_modeling] Saved → {path}")


def _plot_feature_importance_shift(df, train_idx, test_idx, y_tr, y_te, scale_w):
    """
    Show how feature importances shift when momentum features are added.
    Highlights which new features enter the top-15.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (cfg_name, num_cols) in zip(axes, list(CONFIGS.items())[:2]):
        valid_num = [c for c in num_cols if c in df.columns]
        all_cols  = valid_num + CAT
        X_tr = df.loc[train_idx, all_cols]

        pipe = _build_pipe(valid_num, "xgboost", scale_w)
        pipe.fit(X_tr, y_tr)

        pre      = pipe.named_steps["pre"]
        clf      = pipe.named_steps["clf"]
        ohe_cols = pre.named_transformers_["cat"].get_feature_names_out(CAT)
        all_feat  = list(valid_num) + list(ohe_cols)

        imp_df = (
            pd.DataFrame({"feature": all_feat, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(15)
        )

        # Colour: red = momentum/dynamic, blue = original
        original_set = set(NUMERIC_FEATURES)
        colors = [
            "#E74C3C" if f not in original_set else "#3498DB"
            for f in imp_df["feature"]
        ]
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                color=colors[::-1], alpha=0.85)
        ax.set_title(f"Config {cfg_name.split('_')[0]}: Top 15 Feature Importances\n"
                     f"(red = momentum/dynamic | blue = original)",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Importance (Gain)")

    fig.suptitle("Feature Importance Shift: Static → Momentum-Enhanced",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "32_feature_importance_shift.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [advanced_modeling] Saved → {path}")
