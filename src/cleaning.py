"""
cleaning.py
===========
Standardise team names, remove bad matches, and tidy both datasets.
"""

import pandas as pd

# Historical IPL franchise aliases → canonical name
TEAM_ALIASES: dict[str, str] = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
    "Deccan Chargers":             "Sunrisers Hyderabad",
    "Pune Warriors":               "Pune Warriors India",
    "Kochi Tuskers Kerala":        "Kochi Tuskers Kerala",
}


def standardize_team_names(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply franchise alias mapping across specified columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ALIASES)
    return df


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop abandoned / no-result matches
    - Standardise team name columns
    - Fill missing target_overs with 20 (standard T20)
    """
    before = len(matches)

    # Drop rows where result is NA or "no result"
    matches = matches[matches["result"].notna()]
    matches = matches[~matches["result"].str.lower().str.contains("no result|abandoned", na=False)]
    matches = matches[matches["winner"].notna()]

    print(f"[cleaning] Matches removed (abandoned / no result): {before - len(matches)}")

    team_cols = ["team1", "team2", "toss_winner", "winner"]
    matches = standardize_team_names(matches, team_cols)

    matches["target_overs"] = matches["target_overs"].fillna(20.0)
    matches["target_runs"]  = matches["target_runs"].fillna(0)

    matches = matches.reset_index(drop=True)
    return matches


def clean_deliveries(deliveries: pd.DataFrame, valid_match_ids: set) -> pd.DataFrame:
    """
    - Keep only deliveries from valid matches
    - Standardise team name columns
    - Ensure numeric types on key columns
    """
    before = len(deliveries)

    deliveries = deliveries[deliveries["match_id"].isin(valid_match_ids)].copy()
    print(f"[cleaning] Deliveries removed (invalid match_id): {before - len(deliveries)}")

    team_cols = ["batting_team", "bowling_team"]
    deliveries = standardize_team_names(deliveries, team_cols)

    for col in ["over", "ball", "batsman_runs", "extra_runs", "total_runs", "is_wicket"]:
        deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce")

    deliveries = deliveries.dropna(subset=["match_id", "inning", "over", "ball", "total_runs"])
    deliveries = deliveries.reset_index(drop=True)
    return deliveries
