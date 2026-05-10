"""
model_utils.py
==============
Loads the trained IPL chase prediction model and provides
feature engineering for single-ball match states.

The feature engineering here must exactly match what was done
during training in src/feature_engineering.py and src/pressure.py.
"""

import os
import pickle

import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "xgboost.pkl")

# ── Feature lists — must match training exactly (src/modeling.py) ────────────
NUMERIC_FEATURES = [
    "current_score",
    "runs_remaining",
    "balls_remaining",
    "wickets_lost",
    "wickets_in_hand",
    "current_run_rate",
    "required_run_rate",
    "pressure",
    "rate_ratio",
    "wicket_factor",
    "time_factor",
    "over",
]

CATEGORICAL_FEATURES = ["phase", "toss_decision"]

# ── Cached model (loaded once on first request) ──────────────────────────────
_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


# ── Feature engineering ──────────────────────────────────────────────────────

def compute_features(
    target_score: int,
    current_score: int,
    balls_completed: int,
    wickets_lost: int,
    toss_decision: str = "field",
) -> dict:
    """
    Reproduce exactly the feature engineering from training:
      - src/feature_engineering.py  (match-state columns)
      - src/pressure.py             (pressure metric components)

    Parameters
    ----------
    target_score    : runs the chasing team needs to exceed
    current_score   : runs scored so far in the chase
    balls_completed : total legal deliveries bowled (0–120)
    wickets_lost    : wickets lost so far (0–9)
    toss_decision   : "bat" or "field" (the toss-winner's choice)

    Returns
    -------
    dict with all 14 features used by the XGBoost pipeline
    """
    # ── Derived quantities ────────────────────────────────────────────────
    runs_remaining  = max(target_score - current_score, 0)
    balls_remaining = max(120 - balls_completed, 0)
    wickets_in_hand = max(10 - wickets_lost, 0)

    # over index (0-indexed, matches training data column "over")
    # In training: balls_bowled = over*6 + ball_in_over (ball_in_over is 1-6)
    # So over = (balls_bowled - 1) // 6 for balls_bowled >= 1
    over = max((balls_completed - 1) // 6, 0) if balls_completed > 0 else 0

    # ── Run rates ────────────────────────────────────────────────────────
    if balls_completed > 0:
        crr_raw = (current_score / balls_completed) * 6.0
    else:
        crr_raw = 0.0

    if balls_remaining > 0:
        rrr_raw = (runs_remaining / balls_remaining) * 6.0
    elif runs_remaining == 0:
        rrr_raw = 0.0
    else:
        rrr_raw = 999.0   # theoretically impossible but safe

    # ── Pressure components (exact formula from src/pressure.py) ─────────
    crr_safe      = max(crr_raw, 0.5)                            # floor prevents ÷0 early on
    rate_ratio    = float(np.clip(rrr_raw / crr_safe, 0.0, 30.0))
    wicket_factor = 1.0 + (wickets_lost ** 1.5) / 15.0
    time_factor   = 1.0 + max(0.0, (30 - balls_remaining) / 120.0)

    pressure_raw  = rate_ratio * wicket_factor * time_factor
    pressure      = float(np.clip(pressure_raw, 0.05, 15.0))

    # ── Phase ─────────────────────────────────────────────────────────────
    if over < 6:
        phase = "Powerplay"
    elif over < 15:
        phase = "Middle"
    else:
        phase = "Death"

    return {
        # Numeric features (must be in NUMERIC_FEATURES order)
        "current_score":     float(current_score),
        "runs_remaining":    float(runs_remaining),
        "balls_remaining":   float(balls_remaining),
        "wickets_lost":      float(wickets_lost),
        "wickets_in_hand":   float(wickets_in_hand),
        "current_run_rate":  round(crr_raw, 4),
        "required_run_rate": round(min(rrr_raw, 99.9), 4),   # cap for display
        "pressure":          round(pressure, 4),
        "rate_ratio":        round(rate_ratio, 4),
        "wicket_factor":     round(wicket_factor, 4),
        "time_factor":       round(time_factor, 4),
        "over":              float(over),
        # Categorical features
        "phase":             phase,
        "toss_decision":     toss_decision,
    }


# ── Pressure zone classification ─────────────────────────────────────────────

def get_pressure_zone(pressure: float) -> dict:
    """Map pressure score to a human-readable zone."""
    if pressure < 1.0:
        return {
            "name":        "Comfortable",
            "color":       "#22C55E",
            "bg":          "rgba(34,197,94,0.15)",
            "description": "Chase well in control",
        }
    elif pressure < 1.5:
        return {
            "name":        "Building",
            "color":       "#F59E0B",
            "bg":          "rgba(245,158,11,0.15)",
            "description": "Moderate pressure building",
        }
    elif pressure < 2.5:
        return {
            "name":        "Escalating",
            "color":       "#F97316",
            "bg":          "rgba(249,115,22,0.15)",
            "description": "High-pressure situation",
        }
    else:
        return {
            "name":        "Panic",
            "color":       "#EF4444",
            "bg":          "rgba(239,68,68,0.15)",
            "description": "Critical — chase in serious trouble",
        }


# ── Narrative explanation ────────────────────────────────────────────────────

def generate_explanation(features: dict, win_prob: float) -> str:
    crr   = features["current_run_rate"]
    rrr   = features["required_run_rate"]
    wih   = int(features["wickets_in_hand"])
    br    = int(features["balls_remaining"])
    p     = features["pressure"]
    phase = features["phase"]
    overs_left = br // 6
    balls_left = br % 6

    sentences = []

    # Win probability framing
    if win_prob >= 0.80:
        sentences.append(f"Dominant chase — {win_prob:.0%} win probability.")
    elif win_prob >= 0.65:
        sentences.append(f"Batting side in a strong position ({win_prob:.0%} win probability).")
    elif win_prob >= 0.52:
        sentences.append(f"Slight edge to the batting side ({win_prob:.0%}).")
    elif win_prob >= 0.48:
        sentences.append(f"Match evenly poised ({win_prob:.0%}).")
    elif win_prob >= 0.35:
        sentences.append(f"Pressure mounting — batting side at {win_prob:.0%}.")
    else:
        sentences.append(f"Difficult chase. Win probability just {win_prob:.0%}.")

    # Run rate comparison
    if crr == 0.0 and br == 120:
        sentences.append(f"Needs to maintain {rrr:.1f} RPO throughout.")
    elif rrr <= 0:
        sentences.append("Target already achieved.")
    elif crr > rrr + 0.5:
        sentences.append(
            f"Well ahead of the required rate — scoring {crr:.1f} vs needing {rrr:.1f} RPO."
        )
    elif abs(crr - rrr) <= 0.5:
        sentences.append(
            f"Running neck-and-neck with the required rate ({crr:.1f} vs {rrr:.1f} RPO)."
        )
    else:
        gap = rrr - crr
        sentences.append(
            f"Behind the required rate by {gap:.1f} RPO (scoring {crr:.1f}, need {rrr:.1f})."
        )

    # Wickets context
    if wih >= 8:
        sentences.append(f"Plenty of resources left — {wih} wickets in hand.")
    elif wih >= 6:
        sentences.append(f"{wih} wickets remaining keeps pressure manageable.")
    elif wih >= 4:
        sentences.append(f"Only {wih} wickets left — any further loss increases risk sharply.")
    elif wih >= 2:
        sentences.append(f"Down to the lower order with just {wih} wickets remaining.")
    else:
        sentences.append("Last wicket pair — one wicket ends the innings.")

    # Phase and overs context
    if phase == "Death" and overs_left > 0:
        sentences.append(
            f"Death overs: {overs_left} over(s) and {balls_left} ball(s) left — boundaries critical."
        )
    elif phase == "Powerplay":
        sentences.append("Fielding restrictions still in play.")
    elif phase == "Middle" and overs_left >= 5:
        sentences.append(f"{overs_left} overs left in the middle phase — building partnerships is key.")

    return " ".join(sentences)


# ── Main prediction function ─────────────────────────────────────────────────

def predict(
    target_score: int,
    current_score: int,
    balls_completed: int,
    wickets_lost: int,
    toss_decision: str = "field",
) -> dict:
    """
    End-to-end: feature engineering → model inference → formatted response.

    Returns a dict ready to be JSON-serialised by Flask.
    """
    model    = load_model()
    features = compute_features(
        target_score, current_score, balls_completed, wickets_lost, toss_decision
    )

    # Build DataFrame in exact column order the Pipeline expects
    X = pd.DataFrame([features])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba    = model.predict_proba(X)
    win_prob = float(proba[0, 1])

    zone        = get_pressure_zone(features["pressure"])
    explanation = generate_explanation(features, win_prob)

    return {
        # Core prediction
        "win_probability":       round(win_prob, 4),
        "win_probability_pct":   f"{win_prob * 100:.1f}%",
        # Pressure
        "pressure":              round(features["pressure"], 2),
        "pressure_zone":         zone["name"],
        "pressure_zone_color":   zone["color"],
        "pressure_zone_bg":      zone["bg"],
        "pressure_zone_desc":    zone["description"],
        # Pressure components (for breakdown display)
        "rate_ratio":            round(features["rate_ratio"], 2),
        "wicket_factor":         round(features["wicket_factor"], 2),
        "time_factor":           round(features["time_factor"], 2),
        # Run rates
        "current_run_rate":      round(features["current_run_rate"], 2),
        "required_run_rate":     round(min(features["required_run_rate"], 99.9), 2),
        # Match state snapshot
        "runs_remaining":        int(features["runs_remaining"]),
        "balls_remaining":       int(features["balls_remaining"]),
        "overs_remaining":       f"{int(features['balls_remaining']) // 6}.{int(features['balls_remaining']) % 6}",
        "wickets_in_hand":       int(features["wickets_in_hand"]),
        "phase":                 features["phase"],
        "over":                  int(features["over"]),
        # Explanation
        "explanation":           explanation,
    }
