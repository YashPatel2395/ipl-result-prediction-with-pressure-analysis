"""
evaluation.py
=============
Evaluate trained models and generate comparison reports + plots.
"""

import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [eval] Saved → {path}")


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """Return a metrics dictionary for one model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred),   4),
        "precision": round(precision_score(y_test, y_pred),  4),
        "recall":    round(recall_score(y_test, y_pred),     4),
        "f1":        round(f1_score(y_test, y_pred),         4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),   4),
    }
    return metrics


def evaluate_all(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame."""
    results = [evaluate_model(m, X_test, y_test, name) for name, m in models.items()]
    df_res  = pd.DataFrame(results).set_index("model")

    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)
    print(df_res.to_string())
    print("="*60 + "\n")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df_res.to_csv(os.path.join(OUTPUTS_DIR, "model_evaluation.csv"))
    return df_res


def plot_confusion_matrices(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    n     = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Lost", "Won"], yticklabels=["Lost", "Won"],
                    cbar=False)
        ax.set_title(name.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_confusion_matrices.png")


def plot_roc_curves(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#3498DB", "#2ECC71", "#E67E22"]

    for (name, model), color in zip(models.items(), colors):
        y_proba      = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test, y_proba)
        auc          = roc_auc_score(y_test, y_proba)
        label        = f"{name.replace('_', ' ').title()} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, label=label, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "09_roc_curves.png")


def plot_metrics_comparison(df_res: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_melt = df_res[metrics].reset_index().melt(id_vars="model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="model",
                palette=["#3498DB", "#2ECC71", "#E67E22"], ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend(title="Model")
    ax.set_ylabel("Score")
    fig.tight_layout()
    _save(fig, "10_metrics_comparison.png")
