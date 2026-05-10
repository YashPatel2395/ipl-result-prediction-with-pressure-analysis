"""
feature_engineering.py
=======================
Build the ball-by-ball second-innings match-state dataset.

Each output row represents a LIVE CHASE STATE — i.e., what an analyst would
observe after a delivery is bowled, before the next ball is delivered.

No future-ball information is used. Target leakage is avoided by construction:
only information available at match-time is included.
"""

import os
import numpy as np
import pandas as pd

from src.pressure import add_pressure_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINEERED_PATH = os.path.join(PROJECT_ROOT, "data", "engineered", "chase_states.csv")

# Maximum legal deliveries in a T20 innings
MAX_BALLS = 120


def _phase(over: int) -> str:
    if over < 6:
        return "Powerplay"
    elif over < 15:
        return "Middle"
    return "Death"


def _derive_target(match_id: int, deliveries: pd.DataFrame, matches: pd.DataFrame) -> int:
    """
    Return the D/L-adjusted (or standard) target for the second innings.
    Priority:
      1. target_runs from matches.csv  (handles D/L, super-overs, etc.)
      2. First-innings total + 1       (fallback for missing metadata)
    """
    row = matches[matches["id"] == match_id]
    if not row.empty and pd.notna(row["target_runs"].values[0]) and row["target_runs"].values[0] > 0:
        return int(row["target_runs"].values[0])

    # Fallback: sum first innings deliveries
    first_inn = deliveries[(deliveries["match_id"] == match_id) & (deliveries["inning"] == 1)]
    if first_inn.empty:
        return 0
    return int(first_inn["total_runs"].sum()) + 1


def build_match_state_dataset(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core feature engineering step.

    For every ball in every second innings, compute:
    - cumulative score, wickets, run rates
    - runs / balls remaining
    - match phase
    - outcome label (won = 1/0)

    Returns a DataFrame with one row per delivery.
    """
    second_innings = deliveries[deliveries["inning"] == 2].copy()
    second_innings = second_innings.sort_values(["match_id", "over", "ball"]).reset_index(drop=True)

    # balls_bowled from start of innings (1-indexed ball → 0-indexed position)
    # 'over' is 0-indexed (0–19), 'ball' is 1-indexed within over (1–6)
    second_innings["balls_bowled"] = second_innings["over"] * 6 + second_innings["ball"]

    records = []

    for match_id, group in second_innings.groupby("match_id", sort=False):
        group = group.sort_values(["over", "ball"]).reset_index(drop=True)

        match_info = matches[matches["id"] == match_id]
        if match_info.empty:
            continue

        match_info = match_info.iloc[0]
        target = _derive_target(match_id, deliveries, matches)
        if target <= 0:
            continue

        winner          = match_info["winner"]
        venue           = match_info.get("venue", "Unknown")
        toss_decision   = match_info.get("toss_decision", "Unknown")
        toss_winner     = match_info.get("toss_winner", "Unknown")
        season          = match_info.get("season", "Unknown")
        match_date      = match_info.get("date", "Unknown")

        batting_team  = group["batting_team"].iloc[0]
        bowling_team  = group["bowling_team"].iloc[0]

        # Did the chasing team win?
        chase_won = 1 if winner == batting_team else 0

        # Cumulative stats (inclusive — i.e., after this ball)
        group["cum_runs"]    = group["total_runs"].cumsum()
        group["cum_wickets"] = group["is_wicket"].cumsum()

        for _, ball_row in group.iterrows():
            balls_bowled    = int(ball_row["balls_bowled"])
            current_score   = int(ball_row["cum_runs"])
            wickets_lost    = int(ball_row["cum_wickets"])
            wickets_in_hand = 10 - wickets_lost
            runs_remaining  = max(target - current_score, 0)
            balls_remaining = max(MAX_BALLS - balls_bowled, 0)

            # CRR: runs scored per over so far
            crr = (current_score / balls_bowled) * 6 if balls_bowled > 0 else 0.0

            # RRR: runs needed per over from this point
            rrr = (runs_remaining / balls_remaining) * 6 if balls_remaining > 0 else (
                0.0 if runs_remaining == 0 else 999.0
            )

            ov  = int(ball_row["over"])
            bl  = int(ball_row["ball"])

            records.append({
                # Identifiers
                "match_id":       match_id,
                "season":         season,
                "date":           match_date,
                "venue":          venue,
                "batting_team":   batting_team,
                "bowling_team":   bowling_team,
                "toss_winner":    toss_winner,
                "toss_decision":  toss_decision,

                # Ball position
                "over":           ov,
                "ball_in_over":   bl,
                "balls_bowled":   balls_bowled,

                # Match state features
                "target_score":     target,
                "current_score":    current_score,
                "runs_remaining":   runs_remaining,
                "balls_remaining":  balls_remaining,
                "wickets_lost":     wickets_lost,
                "wickets_in_hand":  wickets_in_hand,
                "current_run_rate": round(crr, 4),
                "required_run_rate": round(rrr, 4),

                # Context
                "phase": _phase(ov),

                # Target (no leakage — outcome is pre-known at match time)
                "won": chase_won,
            })

    df = pd.DataFrame(records)
    print(f"[feature_engineering] Ball-state rows created: {len(df):,}")
    print(f"  Matches covered:  {df['match_id'].nunique()}")
    print(f"  Win / Loss split: {df['won'].value_counts().to_dict()}")
    return df


def save_engineered(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(ENGINEERED_PATH), exist_ok=True)
    df.to_csv(ENGINEERED_PATH, index=False)
    print(f"[feature_engineering] Saved → {ENGINEERED_PATH}")


def load_engineered() -> pd.DataFrame:
    return pd.read_csv(ENGINEERED_PATH)


def engineer_pipeline(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Full feature engineering + pressure computation pipeline."""
    df = build_match_state_dataset(deliveries, matches)
    df = add_pressure_features(df)
    if save:
        save_engineered(df)
    return df
