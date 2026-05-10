"""
momentum.py
===========
Dynamic momentum feature engineering for IPL chase ball-state dataset.

Every feature here is computed using ONLY past balls (shift-before-rolling),
maintaining strict temporal integrity. No current-ball information leaks
into its own rolling window.

Feature groups
--------------
RECENT SCORING
  runs_last_{6,12,18}      — runs scored in last N deliveries
  boundaries_last_12        — boundary count (batsman_runs ≥ 4) in last 12 balls
  dot_balls_last_12         — dot ball count in last 12 balls

WICKET MOMENTUM
  wickets_last_{12,18}      — wickets lost in last N deliveries
  consecutive_dot_balls     — current streak of consecutive dot balls
  collapse_indicator        — binary: 2+ wickets in last 12 balls

RATE SHIFT
  crr_change                — change in CRR over last 6 balls
  rrr_change                — change in RRR over last 6 balls
  scoring_acceleration      — runs_last_6 minus runs_6_to_12 (are we speeding up?)

PRESSURE TREND
  pressure_delta_last_over  — pressure now minus pressure 6 balls ago
  rolling_pressure_mean_6   — rolling mean of past 6 balls' pressure
  rolling_pressure_mean_12  — rolling mean of past 12 balls' pressure
  rolling_pressure_std_12   — rolling std of past 12 balls' pressure (volatility)
  pressure_ewm              — exponentially weighted pressure (span=6)
  pressure_acceleration     — second derivative of pressure (change-in-change)
"""

import numpy as np
import pandas as pd


def _consecutive_dot_streak(dot_series: pd.Series) -> pd.Series:
    """Count the streak of consecutive 1s (dots) ending BEFORE each position."""
    shifted = dot_series.shift(1).fillna(0)
    streaks, count = [], 0
    for val in shifted:
        count = count + 1 if val == 1 else 0
        streaks.append(count)
    return pd.Series(streaks, index=dot_series.index, dtype=float)


def _delivery_level_features(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-delivery rolling features within one match's second innings.
    Input: sub-frame for one match_id, sorted by balls_bowled.
    """
    g = grp.sort_values("balls_bowled").copy()

    # ── helper shifted series (past-only) ──────────────────────────────
    runs_s  = g["total_runs"].shift(1).fillna(0)
    bnd_s   = g["is_boundary"].shift(1).fillna(0)
    dot_s   = g["is_dot"].shift(1).fillna(0)
    wkt_s   = g["is_wicket"].shift(1).fillna(0)

    # ── recent scoring ──────────────────────────────────────────────────
    g["runs_last_6"]  = runs_s.rolling(6,  min_periods=0).sum()
    g["runs_last_12"] = runs_s.rolling(12, min_periods=0).sum()
    g["runs_last_18"] = runs_s.rolling(18, min_periods=0).sum()
    g["boundaries_last_12"] = bnd_s.rolling(12, min_periods=0).sum()
    g["dot_balls_last_12"]  = dot_s.rolling(12, min_periods=0).sum()

    # ── wicket momentum ─────────────────────────────────────────────────
    g["wickets_last_12"] = wkt_s.rolling(12, min_periods=0).sum()
    g["wickets_last_18"] = wkt_s.rolling(18, min_periods=0).sum()
    g["consecutive_dot_balls"] = _consecutive_dot_streak(g["is_dot"])
    g["collapse_indicator"]    = (g["wickets_last_12"] >= 2).astype(float)

    # ── scoring acceleration ────────────────────────────────────────────
    runs_6   = runs_s.rolling(6,  min_periods=0).sum()
    runs_12  = runs_s.rolling(12, min_periods=0).sum()
    g["scoring_acceleration"] = runs_6 - (runs_12 - runs_6)   # last6 − prev6

    return g


def _pressure_level_features(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pressure-trend features within one match.
    Input: sub-frame that already has the 'pressure', 'current_run_rate',
    'required_run_rate', and 'balls_bowled' columns.
    """
    g = grp.sort_values("balls_bowled").copy()

    press = g["pressure"]

    # shifted past pressure for rolling (don't include current ball)
    press_s = press.shift(1).fillna(press.iloc[0] if len(press) > 0 else 0)

    g["rolling_pressure_mean_6"]  = press_s.rolling(6,  min_periods=1).mean()
    g["rolling_pressure_mean_12"] = press_s.rolling(12, min_periods=1).mean()
    g["rolling_pressure_std_12"]  = press_s.rolling(12, min_periods=1).std().fillna(0)

    # EWM pressure: exponentially weighted over last ~6 balls
    g["pressure_ewm"] = press_s.ewm(span=6, adjust=False).mean()

    # Pressure delta: how much has pressure changed over last over (6 balls)
    g["pressure_delta_last_over"] = press - press.shift(6).fillna(press)

    # Pressure acceleration: rate-of-change of rate-of-change
    delta1 = press - press.shift(6).fillna(press)
    delta2 = press.shift(6).fillna(press) - press.shift(12).fillna(press)
    g["pressure_acceleration"] = delta1 - delta2

    # Rate shift features
    crr = g["current_run_rate"]
    rrr = g["required_run_rate"]
    g["crr_change"] = crr - crr.shift(6).fillna(crr)
    g["rrr_change"] = rrr - rrr.shift(6).fillna(rrr)

    return g


def add_momentum_features(
    df: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main entry point.

    1. Computes delivery-level rolling features from raw deliveries
       (boundaries, dots, wicket streaks).
    2. Merges them onto df using (match_id, over, ball_in_over).
    3. Computes pressure-trend features directly on df.

    Returns an augmented copy of df.
    """
    # ── Phase 1: delivery-level momentum from raw deliveries ────────────
    inn2 = (
        deliveries[deliveries["inning"] == 2]
        .copy()
        .sort_values(["match_id", "over", "ball"])
    )
    inn2["balls_bowled"] = inn2["over"] * 6 + inn2["ball"]
    inn2["is_boundary"]  = (inn2["batsman_runs"] >= 4).astype(float)
    inn2["is_dot"]       = (inn2["total_runs"] == 0).astype(float)

    # Explicit loop avoids pandas groupby-apply index/column edge cases
    parts = []
    for _, grp in inn2.groupby("match_id", sort=False):
        parts.append(_delivery_level_features(grp))
    delivery_features = pd.concat(parts, ignore_index=True)

    DELIVERY_COLS = [
        "match_id", "over", "ball",
        "runs_last_6", "runs_last_12", "runs_last_18",
        "boundaries_last_12", "dot_balls_last_12",
        "wickets_last_12", "wickets_last_18",
        "consecutive_dot_balls", "collapse_indicator",
        "scoring_acceleration",
    ]
    delivery_features = (
        delivery_features[DELIVERY_COLS]
        .rename(columns={"ball": "ball_in_over"})
    )

    df_out = df.merge(delivery_features, on=["match_id", "over", "ball_in_over"], how="left")

    # Ensure balls_bowled column exists
    if "balls_bowled" not in df_out.columns:
        df_out["balls_bowled"] = df_out["over"] * 6 + df_out["ball_in_over"]

    # ── Phase 2: pressure-trend features ────────────────────────────────
    parts2 = []
    for _, grp in df_out.groupby("match_id", sort=False):
        parts2.append(_pressure_level_features(grp))
    df_out = pd.concat(parts2, ignore_index=True)

    # Fill early-innings NaN (rolling windows before enough data)
    momentum_cols = [
        "runs_last_6", "runs_last_12", "runs_last_18",
        "boundaries_last_12", "dot_balls_last_12",
        "wickets_last_12", "wickets_last_18",
        "consecutive_dot_balls", "collapse_indicator",
        "scoring_acceleration",
        "rolling_pressure_mean_6", "rolling_pressure_mean_12",
        "rolling_pressure_std_12", "pressure_ewm",
        "pressure_delta_last_over", "pressure_acceleration",
        "crr_change", "rrr_change",
    ]
    df_out[momentum_cols] = df_out[momentum_cols].fillna(0)

    print(f"[momentum] Added {len(momentum_cols)} momentum features → "
          f"dataset shape: {df_out.shape}")
    return df_out.reset_index(drop=True)


# Canonical list for downstream modules
MOMENTUM_FEATURES = [
    "runs_last_6", "runs_last_12", "runs_last_18",
    "boundaries_last_12", "dot_balls_last_12",
    "wickets_last_12", "wickets_last_18",
    "consecutive_dot_balls", "collapse_indicator",
    "scoring_acceleration",
    "pressure_delta_last_over",
    "rolling_pressure_mean_6", "rolling_pressure_mean_12",
    "rolling_pressure_std_12",
    "pressure_ewm",
    "pressure_acceleration",
    "crr_change", "rrr_change",
]
