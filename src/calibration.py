"""
calibration.py
==============
Evaluate whether model-estimated win probabilities are TRUSTWORTHY.

A model with 70% predicted probability should be correct 70% of the time.
This property is called *calibration* and matters when win probability is
used operationally (e.g., live broadcast overlays, betting odds, team tactics).

Methods:
  - Reliability diagram (calibration curve)
  - Expected Calibration Error (ECE)
  - Brier Score (strictly proper scoring rule)
  - Max Calibration Error (MCE)

Optional: isotonic regression post-hoc calibration.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

N_BINS = 10


def expected_calibration_error(y_true, y_proba, n_bins=N_BINS) -> float:
    """ECE: probability-weighted average of calibration gap across bins."""
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_proba[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def max_calibration_error(y_true, y_proba, n_bins=N_BINS) -> float:
    """MCE: worst-case calibration gap across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_proba[mask].mean()
        mce  = max(mce, abs(acc - conf))
    return mce


def calibration_metrics(model_name: str, y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    brier = brier_score_loss(y_true, y_proba)
    ece   = expected_calibration_error(y_true, y_proba)
    mce   = max_calibration_error(y_true, y_proba)
    return {
        "model":   model_name,
        "brier":   round(brier, 5),
        "ece":     round(ece,   5),
        "mce":     round(mce,   5),
    }


def plot_reliability_diagrams(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Plot calibration curves (reliability diagrams) for all models
    and return a calibration metrics summary.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    n      = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    colors  = ["#3498DB", "#2ECC71", "#E67E22"]
    all_metrics = []
    y_true  = y_test.values

    for ax, (name, model), color in zip(axes, models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]

        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=N_BINS)

        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2.5,
                markersize=7, label=name.replace("_", " ").title())

        # Confidence intervals via binomial stderr
        n_vals = []
        bins   = np.linspace(0, 1, N_BINS + 1)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_proba >= lo) & (y_proba < hi)
            n_vals.append(mask.sum())

        se = np.array([
            np.sqrt(fp * (1 - fp) / max(nv, 1))
            for fp, nv in zip(frac_pos, n_vals)
        ])
        ax.fill_between(mean_pred, frac_pos - 1.96 * se, frac_pos + 1.96 * se,
                        alpha=0.2, color=color, label="95% CI")

        m = calibration_metrics(name, y_true, y_proba)
        all_metrics.append(m)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Actual Win Rate)")
        ax.set_title(
            f"{name.replace('_', ' ').title()}\n"
            f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}  MCE={m['mce']:.4f}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle("Reliability Diagrams — Win Probability Calibration\n"
                 "(Points on the diagonal = perfectly calibrated model)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "21_reliability_diagrams.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [calibration] Saved → {path}")

    df_cal = pd.DataFrame(all_metrics).set_index("model")
    print("\n[calibration] Calibration metrics:")
    print(df_cal.to_string())
    print("  (lower Brier/ECE/MCE = better calibration)")

    # Save
    df_cal.to_csv(os.path.join(OUTPUTS_DIR, "calibration_metrics.csv"))
    return df_cal


def plot_win_prob_histogram(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Histogram of predicted win probabilities, split by actual outcome.
    Well-separated distributions = good discrimination.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    best_name = list(models.keys())[-1]  # XGBoost (last)
    model     = models[best_name]
    y_proba   = model.predict_proba(X_test)[:, 1]
    y_true    = y_test.values

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.6, color="#2ECC71",
            label="Chase Won  (actual)", density=True)
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.6, color="#E74C3C",
            label="Chase Lost (actual)", density=True)
    ax.axvline(0.5, ls="--", color="grey", alpha=0.7, label="Decision threshold 0.5")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted Win Probability Distribution — {best_name.replace('_',' ').title()}",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "22_win_prob_histogram.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [calibration] Saved → {path}")


def run_calibration(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    print("\n[calibration] Evaluating win probability calibration...")
    df_cal = plot_reliability_diagrams(models, X_test, y_test)
    plot_win_prob_histogram(models, X_test, y_test)
    return df_cal
