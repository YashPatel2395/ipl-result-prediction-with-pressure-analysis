"""
data_loader.py
==============
Load raw IPL datasets and print an inspection summary.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matches, deliveries) as DataFrames."""
    matches_path    = os.path.join(PROJECT_ROOT, "matches.csv")
    deliveries_path = os.path.join(PROJECT_ROOT, "deliveries.csv")

    for p in (matches_path, deliveries_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected dataset not found: {p}")

    matches    = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


def inspect_datasets(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Print shape, dtypes, and missing-value summary for both datasets."""
    for name, df in [("MATCHES", matches), ("DELIVERIES", deliveries)]:
        print(f"\n{'='*60}")
        print(f"  {name}  —  {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"{'='*60}")
        print(f"\nColumns:\n  {list(df.columns)}")

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("\nMissing values: none")
        else:
            print("\nMissing values:")
            for col, cnt in missing.items():
                pct = cnt / len(df) * 100
                print(f"  {col:<30} {cnt:>6,}  ({pct:.1f}%)")

    print(f"\nSeason range: {matches['season'].unique()}")
    print(f"Unique teams (matches):    {sorted(set(matches['team1']) | set(matches['team2']))}")
    print(f"Innings in deliveries:     {sorted(deliveries['inning'].unique())}")
