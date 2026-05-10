"""
interpretability_advanced.py
=============================
Advanced interpretability for the momentum-aware chase model.

Identifies and visualises:

  PANIC ZONES        — high pressure + rising momentum + late overs
                       (when the chase feels and IS out of control)

  STABLE CHASE ZONES — low pressure, negative/flat momentum, wickets intact
                       (when the batting side is in comfortable control)

  DEATH-OVER COLLAPSE SIGNATURES
                     — characteristic ball sequence that precedes a lost chase
                       in the final 4 overs: high consecutive dots, rapid wicket
                       loss, pressure spike above threshold

  MOMENTUM IMPORTANCE ANALYSIS
                     — which dynamic features matter most, and when in the
                       innings (early / mid / death overs) they peak in importance

All plots saved to outputs/plots/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

from src.modeling         import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.momentum         import MOMENTUM_FEATURES
from src.dynamic_pressure import DYNAMIC_PRESSURE_FEATURES


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [interpretability] Saved → {path}")


# ---------------------------------------------------------------------------
# Zone definitions (rule-based for clear communication)
# ---------------------------------------------------------------------------

ZONE_RULES = {
    "Panic": {
        "pressure":    (4.0, np.inf),
        "dp_momentum": (0.5, np.inf),
        "color": "#E74C3C",
    },
    "Stable": {
        "pressure":    (0.0, 1.5),
        "dp_momentum": (-np.inf, 0.2),
        "color": "#2ECC71",
    },
    "Escalating": {
        "pressure":    (2.0, 4.0),
        "dp_momentum": (0.1, np.inf),
        "color": "#E67E22",
    },
    "Recovery": {
        "pressure":    (2.0, np.inf),
        "dp_momentum": (-np.inf, -0.3),
        "color": "#3498DB",
    },
    "Neutral": {
        "pressure":    (0.0, np.inf),
        "dp_momentum": (-np.inf, np.inf),
        "color": "#BDC3C7",
    },
}


def assign_zone(df: pd.DataFrame) -> pd.Series:
    zones = pd.Series("Neutral", index=df.index)
    for zone, rules in ZONE_RULES.items():
        if zone == "Neutral":
            continue
        p_lo, p_hi = rules["pressure"]
        m_lo, m_hi = rules["dp_momentum"]
        mask = (
            (df["pressure"]    >= p_lo) & (df["pressure"]    <  p_hi) &
            (df["dp_momentum"] >= m_lo) & (df["dp_momentum"] <  m_hi)
        )
        zones[mask] = zone
    return zones


# ---------------------------------------------------------------------------
# 34. Zone win rates and statistics
# ---------------------------------------------------------------------------
def plot_zone_analysis(df: pd.DataFrame) -> None:
    df_z = df.copy()
    df_z["zone"] = assign_zone(df_z)

    zone_stats = (
        df_z.groupby("zone")["won"]
        .agg(win_rate="mean", count="count")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: win rate by zone
    colors = [ZONE_RULES.get(z, {"color": "#BDC3C7"})["color"]
              for z in zone_stats["zone"]]
    bars = axes[0].bar(zone_stats["zone"], zone_stats["win_rate"],
                       color=colors, width=0.5, alpha=0.9)
    for bar, row in zip(bars, zone_stats.itertuples()):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{row.win_rate:.2f}\n(n={row.count:,})", ha="center", fontsize=9)
    axes[0].axhline(0.5, ls="--", color="grey", alpha=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Chase Win Rate by Pressure Zone", fontweight="bold")
    axes[0].set_ylabel("Win Rate")

    # Right: zone composition over match progression
    df_z["over_group"] = pd.cut(df_z["over"], bins=[0, 5, 10, 15, 20],
                                 labels=["PP (0-5)", "Mid-early (6-10)",
                                         "Mid-late (11-15)", "Death (16-19)"])
    zone_over = (
        df_z.groupby(["over_group", "zone"], observed=True)
        .size()
        .unstack(fill_value=0)
        .apply(lambda r: r / r.sum(), axis=1)
    )
    zone_over.plot(kind="bar", stacked=True, ax=axes[1],
                   color=[ZONE_RULES.get(c, {"color": "#BDC3C7"})["color"]
                          for c in zone_over.columns],
                   alpha=0.85, width=0.6)
    axes[1].set_title("Zone Distribution by Match Phase", fontweight="bold")
    axes[1].set_ylabel("Proportion of Balls")
    axes[1].set_xlabel("Match Phase")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Zone", loc="upper right", fontsize=8)

    fig.suptitle("Pressure Zone Analysis — Stable / Escalating / Panic / Recovery",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "34_zone_analysis.png")

    # Print zone summary
    print("\n[interpretability] Pressure Zone Win Rates:")
    print(zone_stats.to_string(index=False))


# ---------------------------------------------------------------------------
# 35. Death-over collapse signatures
# ---------------------------------------------------------------------------
def plot_collapse_signature(df: pd.DataFrame) -> None:
    """
    Characterise the 'collapse signature': the typical ball-sequence
    in the final 5 overs of a losing chase.

    Compare: won vs lost chases in overs 15–19.
    Features: dot rate, wicket rate, pressure, scoring_acceleration.
    """
    death = df[df["over"] >= 15].copy()
    death["Outcome"] = death["won"].map({1: "Won", 0: "Lost"})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features  = [
        ("dot_balls_last_12",   "Dot Balls (last 12)"),
        ("wickets_last_12",     "Wickets (last 12)"),
        ("pressure",            "Pressure"),
        ("scoring_acceleration","Scoring Acceleration"),
    ]

    for ax, (feat, label) in zip(axes.flat, features):
        if feat not in death.columns:
            continue
        for outcome, color in [("Won", "#2ECC71"), ("Lost", "#E74C3C")]:
            sub = death[death["Outcome"] == outcome].groupby("over")[feat].mean()
            ax.plot(sub.index, sub.values, color=color, linewidth=2.5,
                    marker="o", ms=5, label=outcome)
        ax.set_title(f"Death Overs — {label}", fontweight="bold")
        ax.set_xlabel("Over")
        ax.set_ylabel(label)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    fig.suptitle("Death-Over Collapse Signature (Overs 15–19)\n"
                 "Red = losing chase | Green = winning chase",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "35_collapse_signature.png")


# ---------------------------------------------------------------------------
# 36. Momentum feature importance by innings phase
# ---------------------------------------------------------------------------
def plot_phase_importance(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> None:
    """
    Train separate XGBoost models on each phase (PP / Middle / Death)
    and compare feature importances. Reveals when dynamic features matter most.
    """
    all_num = [c for c in (NUMERIC_FEATURES + MOMENTUM_FEATURES + DYNAMIC_PRESSURE_FEATURES)
               if c in df.columns]
    cat     = CATEGORICAL_FEATURES

    phases = {
        "Powerplay (0-5)": (0, 5),
        "Middle (6-14)":   (6, 14),
        "Death (15-19)":   (15, 19),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (phase_name, (ov_lo, ov_hi)) in zip(axes, phases.items()):
        phase_mask  = (df["over"] >= ov_lo) & (df["over"] <= ov_hi)
        df_phase    = df[phase_mask]

        tr_phase    = df_phase.index.intersection(train_idx)
        if len(tr_phase) < 500:
            ax.set_title(f"{phase_name}\n(insufficient data)", fontsize=9)
            continue

        X_tr = df_phase.loc[tr_phase, all_num + cat]
        y_tr = df_phase.loc[tr_phase, "won"]

        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=neg/max(pos,1), eval_metric="logloss",
            use_label_encoder=False, random_state=42, n_jobs=1,
        )
        pre = ColumnTransformer([
            ("num", "passthrough", all_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ])
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        ohe_cols = pre.named_transformers_["cat"].get_feature_names_out(cat)
        all_feat  = all_num + list(ohe_cols)
        imp_df = (
            pd.DataFrame({"feature": all_feat, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(12)
        )

        momentum_set = set(MOMENTUM_FEATURES + DYNAMIC_PRESSURE_FEATURES)
        colors = ["#E74C3C" if f in momentum_set else "#3498DB"
                  for f in imp_df["feature"]]

        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                color=colors[::-1], alpha=0.85)
        ax.set_title(f"{phase_name}\n(red=momentum | blue=static)",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Importance (Gain)")

    fig.suptitle("Feature Importance by Match Phase — When Does Momentum Matter Most?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "36_phase_importance.png")


# ---------------------------------------------------------------------------
# 37. Early collapse detection — can we detect trouble coming?
# ---------------------------------------------------------------------------
def plot_early_detection(df: pd.DataFrame) -> None:
    """
    For matches that ended in a loss, analyse:
    - At what ball did pressure first exceed 3.0?
    - At what ball did dp_momentum first exceed 1.0?
    - How early can we predict the outcome?

    Compare accuracy of prediction at each over mark.
    """
    from sklearn.metrics import accuracy_score

    records = []
    # For each over milestone, train a simple threshold model
    for over_cutoff in range(5, 20, 2):
        # Take the state at exactly over=over_cutoff (or nearest)
        at_over = (
            df[df["over"] == over_cutoff]
            .groupby("match_id")
            .first()
            .reset_index()
        )
        if at_over.empty or len(at_over) < 100:
            continue

        # Predict using pressure threshold > 2.0 → loss
        thresh = 2.0
        preds  = (at_over["pressure"] > thresh).astype(int)
        # Predict loss (0) when pressure > threshold, win (1) otherwise
        preds_flip = 1 - preds
        acc = accuracy_score(at_over["won"], preds_flip)

        records.append({
            "over":     over_cutoff,
            "accuracy": acc,
            "n_matches": len(at_over),
        })

    if not records:
        return

    res = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res["over"], res["accuracy"], "o-", color="#9B59B6",
            linewidth=2.5, ms=8)
    ax.axhline(max(df["won"].mean(), 1 - df["won"].mean()),
               ls="--", color="grey", alpha=0.5, label="Majority class baseline")
    ax.set_xlabel("Over at which prediction is made")
    ax.set_ylabel("Accuracy (threshold: pressure > 2.0 → predict loss)")
    ax.set_title("Early Collapse Detection — Predictive Accuracy by Over\n"
                 "(Can pressure alone detect a losing chase early?)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    fig.tight_layout()
    _save(fig, "37_early_detection.png")

    print("\n[interpretability] Early collapse detection accuracy by over:")
    print(res.to_string(index=False))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_advanced_interpretability(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> None:
    print("\n[interpretability] Running advanced interpretability analysis...")
    plot_zone_analysis(df)
    plot_collapse_signature(df)
    plot_phase_importance(df, train_idx, test_idx)
    plot_early_detection(df)
    print("[interpretability] All interpretability plots saved.")
