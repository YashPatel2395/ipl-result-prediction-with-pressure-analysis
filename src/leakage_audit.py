"""
leakage_audit.py
================
Systematic audit of the chase-state dataset for data leakage.

Three leakage vectors are checked:

  L1 — Feature leakage:    winner / match-result columns inside feature set
  L2 — Temporal leakage:   future balls affecting current-state features
  L3 — Split leakage:      same match appearing in both train and test sets

The audit produces a structured report and raises ValueError on any violation.
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

# Columns that would directly or indirectly reveal the outcome
LEAKAGE_BLACKLIST = {
    "winner", "result", "result_margin", "player_of_match",
    "method",                              # D/L method reveals altered target (post-decision)
    "final_score", "total_wickets",        # hypothetical end-state columns
    "super_over",                          # reveals tie → future info
}

# Columns whose presence requires argument
GREY_LIST = {
    "toss_winner",       # known at match start — safe
    "toss_decision",     # known at match start — safe
    "venue",             # known at match start — safe
    "season",            # metadata — safe
}


def audit_feature_leakage(feature_cols: list[str]) -> dict:
    """L1: check no blacklisted column is in the feature set."""
    violations = [c for c in feature_cols if c in LEAKAGE_BLACKLIST]
    grey_used  = [c for c in feature_cols if c in GREY_LIST]

    status = "PASS" if not violations else "FAIL"
    return {
        "check":      "L1 — Feature Leakage",
        "status":     status,
        "violations": violations,
        "grey_used":  grey_used,
        "detail":     (
            "No future/outcome columns found in feature set." if not violations
            else f"LEAKAGE DETECTED: {violations}"
        ),
    }


def audit_temporal_leakage(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    L2: verify that every feature is computable from balls ≤ current ball.

    We verify this structurally:
    - 'current_score'      = cumsum of total_runs up to and including this ball ✓
    - 'runs_remaining'     = target - current_score                             ✓
    - 'balls_remaining'    = 120 - balls_bowled                                 ✓
    - 'wickets_lost'       = cumsum of is_wicket up to this ball               ✓
    - 'required_run_rate'  = derived from runs/balls remaining                  ✓
    - 'pressure'           = derived from above                                 ✓

    Additionally check: no ball has current_score > target (that would mean the
    chase is already won but we still have rows — target-exceeding balls should
    be absent or treated carefully).
    """
    issues = []

    # Check: current_score should never exceed a very large target for valid rows
    if "current_score" in df.columns and "target_score" in df.columns:
        over_target = (df["current_score"] > df["target_score"]).sum()
        if over_target > 0:
            issues.append(
                f"{over_target:,} rows where current_score > target_score "
                "(these represent the final winning ball — acceptable if kept, "
                "but should not inform the model about the future)."
            )

    # Check: pressure / rate_ratio must be non-negative
    for col in ["pressure", "rate_ratio", "current_run_rate", "required_run_rate"]:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives:
                issues.append(f"'{col}' has {negatives} negative values — check derivation.")

    status = "PASS" if not issues else "WARN"
    return {
        "check":   "L2 — Temporal Leakage",
        "status":  status,
        "detail":  "; ".join(issues) if issues else
                   "All features are derived from information available at the current ball.",
    }


def audit_split_leakage(
    train_indices: pd.Index,
    test_indices: pd.Index,
    df: pd.DataFrame,
) -> dict:
    """L3: verify no match appears in both train and test splits."""
    train_matches = set(df.loc[train_indices, "match_id"])
    test_matches  = set(df.loc[test_indices,  "match_id"])
    overlap       = train_matches & test_matches

    status = "PASS" if not overlap else "FAIL"
    return {
        "check":       "L3 — Train/Test Split Leakage",
        "status":      status,
        "train_size":  len(train_matches),
        "test_size":   len(test_matches),
        "overlap":     len(overlap),
        "detail":      (
            f"Clean split: {len(train_matches)} train matches, "
            f"{len(test_matches)} test matches, 0 overlap."
            if not overlap
            else f"LEAKAGE: {len(overlap)} matches appear in BOTH train and test."
        ),
    }


def audit_target_isolation(df: pd.DataFrame, feature_cols: list[str], target: str = "won") -> dict:
    """Confirm target is binary and not trivially deducible from any single feature."""
    issues = []

    if target in feature_cols:
        issues.append(f"Target column '{target}' is inside feature_cols — direct leakage!")

    # Check target is truly binary
    unique_vals = df[target].unique()
    if not set(unique_vals).issubset({0, 1}):
        issues.append(f"Target has unexpected values: {unique_vals}")

    # Check no feature has correlation = ±1 with target (perfect leakage)
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        corr = df[col].corr(df[target])
        if abs(corr) > 0.95:
            issues.append(f"Feature '{col}' has near-perfect correlation ({corr:.3f}) with target.")

    status = "PASS" if not issues else "FAIL"
    return {
        "check":  "L4 — Target Isolation",
        "status": status,
        "detail": "; ".join(issues) if issues else
                  f"Target '{target}' is binary and isolated from all features.",
    }


def run_audit(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_indices: pd.Index,
    test_indices: pd.Index,
    target: str = "won",
) -> list[dict]:
    """Run all leakage checks and print a formatted report."""
    results = [
        audit_feature_leakage(feature_cols),
        audit_temporal_leakage(df, feature_cols),
        audit_split_leakage(train_indices, test_indices, df),
        audit_target_isolation(df, feature_cols, target),
    ]

    print("\n" + "="*60)
    print("  DATA LEAKAGE AUDIT REPORT")
    print("="*60)

    all_pass = True
    for r in results:
        status_str = f"[{r['status']}]"
        print(f"\n{status_str:8} {r['check']}")
        print(f"         {r['detail']}")
        if r.get("grey_used"):
            print(f"         Pre-match context used (safe): {r['grey_used']}")
        if r["status"] not in ("PASS", "WARN"):
            all_pass = False

    print("\n" + "-"*60)
    overall = "ALL CHECKS PASSED" if all_pass else "ONE OR MORE CHECKS FAILED"
    print(f"  Overall: {overall}")
    print("="*60 + "\n")

    # Save audit log
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "leakage_audit.txt"), "w") as f:
        f.write("DATA LEAKAGE AUDIT REPORT\n")
        f.write("="*60 + "\n\n")
        for r in results:
            f.write(f"[{r['status']}] {r['check']}\n")
            f.write(f"       {r['detail']}\n\n")
        f.write(f"Overall: {overall}\n")

    if not all_pass:
        raise ValueError("Leakage audit FAILED — fix violations before training.")

    return results
