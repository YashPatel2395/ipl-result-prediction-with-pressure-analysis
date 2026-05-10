"""
dynamic_pressure.py
===================
Redesign pressure as a TEMPORAL concept.

Static pressure captures the difficulty of the current state.
Dynamic pressure captures how that difficulty is evolving.

This module produces four temporal pressure signals:

  dp_ewm_fast     — Exponentially Weighted Moving Average (span=6)
                    React quickly to boundaries and wickets.

  dp_ewm_slow     — Exponentially Weighted Moving Average (span=24)
                    Reflects the innings trend; slow to react.

  dp_momentum     — dp_ewm_fast − dp_ewm_slow
                    Positive  → pressure accelerating (chase deteriorating)
                    Negative  → pressure easing (batting side recovering)
                    Analogous to MACD in technical analysis.

  dp_regime       — Categorical zone derived from static + momentum:
                    0 = STABLE    (low pressure, flat/falling)
                    1 = BUILDING  (moderate pressure, rising slowly)
                    2 = ESCALATING(high pressure, rising fast)
                    3 = PANIC     (very high pressure, momentum positive)
                    4 = RECOVERY  (pressure falling from high level)

  dp_trend_slope  — OLS slope of pressure over the last 12 balls.
                    Positive = pressure rising (per ball), Negative = easing.

All features are computed on PAST pressure only (shifted before EWM/rolling)
to prevent any information from future balls entering the current state.
"""

import numpy as np
import pandas as pd


def _ols_slope(arr: np.ndarray) -> float:
    """Return slope of OLS regression of arr against its index."""
    n = len(arr)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    return float(np.dot(x, arr - arr.mean()) / (np.dot(x, x) + 1e-9))


def _assign_regime(pressure: float, momentum: float) -> int:
    """
    Classify a (pressure, momentum) pair into one of 5 chase regimes.

    Thresholds derived from empirical quantiles of the IPL dataset:
      pressure median ≈ 1.5,  momentum near-zero typical
    """
    if pressure < 1.2:
        return 0   # STABLE — well inside target
    if pressure < 2.0 and momentum <= 0.1:
        return 1   # BUILDING — moderate, not worsening fast
    if momentum > 0.3 and pressure >= 2.0:
        return 3   # PANIC — high pressure AND rising fast
    if momentum < -0.2 and pressure >= 1.8:
        return 4   # RECOVERY — pressure dropping from high level
    return 2       # ESCALATING — default mid-danger zone


REGIME_LABELS = {
    0: "Stable",
    1: "Building",
    2: "Escalating",
    3: "Panic",
    4: "Recovery",
}


def _dynamic_per_match(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("balls_bowled").copy()

    # Use shifted pressure so current ball doesn't feed into its own EWM
    p_shifted = grp["pressure"].shift(1).bfill()

    grp["dp_ewm_fast"]    = p_shifted.ewm(span=6,  adjust=False).mean()
    grp["dp_ewm_slow"]    = p_shifted.ewm(span=24, adjust=False).mean()
    grp["dp_momentum"]    = grp["dp_ewm_fast"] - grp["dp_ewm_slow"]

    # OLS slope over rolling 12-ball window of past pressure
    grp["dp_trend_slope"] = (
        p_shifted.rolling(12, min_periods=3)
        .apply(_ols_slope, raw=True)
        .fillna(0)
    )

    # Regime (vectorised via apply row-wise for clarity)
    grp["dp_regime"] = [
        _assign_regime(p, m)
        for p, m in zip(grp["pressure"].values, grp["dp_momentum"].values)
    ]

    # Relative pressure: deviation from innings running mean so far
    grp["dp_vs_innings_mean"] = (
        grp["pressure"] - p_shifted.expanding().mean()
    )

    return grp


def add_dynamic_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all dynamic pressure features and attach to df.
    Requires df to have 'pressure' and 'balls_bowled' columns.
    """
    if "balls_bowled" not in df.columns:
        df = df.copy()
        df["balls_bowled"] = df["over"] * 6 + df["ball_in_over"]

    parts = []
    for _, grp in df.groupby("match_id", sort=False):
        parts.append(_dynamic_per_match(grp))
    df_out = pd.concat(parts, ignore_index=True)

    dynamic_cols = [
        "dp_ewm_fast", "dp_ewm_slow", "dp_momentum",
        "dp_trend_slope", "dp_regime", "dp_vs_innings_mean",
    ]
    df_out[dynamic_cols] = df_out[dynamic_cols].fillna(0)

    print(f"[dynamic_pressure] Added {len(dynamic_cols)} dynamic pressure features.")
    return df_out.reset_index(drop=True)


DYNAMIC_PRESSURE_FEATURES = [
    "dp_ewm_fast", "dp_ewm_slow", "dp_momentum",
    "dp_trend_slope", "dp_regime", "dp_vs_innings_mean",
]
