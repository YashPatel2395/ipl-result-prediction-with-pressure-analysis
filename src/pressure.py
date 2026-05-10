"""
pressure.py
===========
Pressure metric design for IPL chase situations.

Formula
-------
    pressure = rate_ratio * wicket_factor * time_factor

Components:
  rate_ratio    = RRR / max(CRR, floor)
                  How far behind the required scoring rate the batting side is.
                  > 1  → falling behind  |  < 1 → ahead of rate

  wicket_factor = 1 + (wickets_lost ^ 1.5) / 15
                  Exponential penalty for lost wickets.
                  Rationale: each wicket is worth progressively more in a chase
                  because options shrink non-linearly.

  time_factor   = 1 + max(0, (30 - balls_remaining) / 120)
                  Mild urgency bump in the final 5 overs. Kept subtle so that
                  rate_ratio remains the dominant signal.

Pressure ≈ 1.0  →  balanced chase (on rate, wickets intact)
Pressure >> 1.0 →  high difficulty
Pressure < 1.0  →  comfortable chase (ahead of rate, wickets in hand)

Clipped to [0.05, 15] before normalisation so outliers don't distort the range.
"""

import numpy as np
import pandas as pd

CRR_FLOOR        = 0.5    # minimum CRR used in division (avoids inf at ball-1)
PRESSURE_CLIP_LO = 0.05
PRESSURE_CLIP_HI = 15.0


def wicket_factor(wickets_lost: pd.Series) -> pd.Series:
    """Exponential wicket penalty: 1.0 at 0 wkts → ~3.6 at 9 wkts."""
    return 1.0 + (wickets_lost.clip(lower=0) ** 1.5) / 15.0


def time_factor(balls_remaining: pd.Series) -> pd.Series:
    """Mild urgency bump in the final 30 balls."""
    return 1.0 + ((30 - balls_remaining).clip(lower=0) / 120.0)


def compute_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Compute the composite pressure metric for each ball-state row.

    Input columns required:
        current_run_rate, required_run_rate, wickets_lost, balls_remaining
    """
    crr_safe    = df["current_run_rate"].clip(lower=CRR_FLOOR)
    rate_ratio  = (df["required_run_rate"] / crr_safe).clip(lower=0)
    wf          = wicket_factor(df["wickets_lost"])
    tf          = time_factor(df["balls_remaining"])

    pressure = rate_ratio * wf * tf
    pressure = pressure.clip(lower=PRESSURE_CLIP_LO, upper=PRESSURE_CLIP_HI)

    return pressure


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append pressure and its sub-components as new columns.
    """
    df = df.copy()
    crr_safe = df["current_run_rate"].clip(lower=CRR_FLOOR)

    df["rate_ratio"]    = (df["required_run_rate"] / crr_safe).clip(lower=0)
    df["wicket_factor"] = wicket_factor(df["wickets_lost"])
    df["time_factor"]   = time_factor(df["balls_remaining"])
    df["pressure"]      = compute_pressure(df)

    # Normalised pressure: z-score across the full dataset
    mu, sigma            = df["pressure"].mean(), df["pressure"].std()
    df["pressure_z"]     = (df["pressure"] - mu) / sigma

    # Percentile rank [0, 1]
    df["pressure_pct"]   = df["pressure"].rank(pct=True)

    return df
