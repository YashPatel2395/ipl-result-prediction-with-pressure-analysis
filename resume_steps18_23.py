"""
resume_steps18_23.py
====================
Resumes the IPL pipeline from Step 18 onwards.
Loads cleaned + engineered data from disk; skips Steps 1-17.

Run with:
    PYTHONUNBUFFERED=1 python resume_steps18_23.py 2>&1 | tee /tmp/resume_18_23.log
"""

import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")


def banner(text: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  {text}", flush=True)
    print(f"{'='*60}", flush=True)


t0 = time.time()

# ---------------------------------------------------------------------------
# Load saved data
# ---------------------------------------------------------------------------
banner("LOADING SAVED DATA")

cleaned_dir    = os.path.join(PROJECT_ROOT, "data", "cleaned")
engineered_dir = os.path.join(PROJECT_ROOT, "data", "engineered")

print("[resume] Reading cleaned deliveries...", flush=True)
deliveries = pd.read_csv(os.path.join(cleaned_dir, "deliveries_clean.csv"))

print("[resume] Reading cleaned matches...", flush=True)
matches = pd.read_csv(os.path.join(cleaned_dir, "matches_clean.csv"))

print("[resume] Reading engineered chase states...", flush=True)
df = pd.read_csv(os.path.join(engineered_dir, "chase_states.csv"))

print(f"[resume] Loaded df: {df.shape}", flush=True)

# Merge is_wicket for collapse / story analysis
second_inn_wkts = (
    deliveries[deliveries["inning"] == 2]
    [["match_id", "over", "ball", "is_wicket"]]
    .rename(columns={"ball": "ball_in_over"})
)
df = df.merge(second_inn_wkts, on=["match_id", "over", "ball_in_over"], how="left")
df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)

if "balls_bowled" not in df.columns:
    df["balls_bowled"] = df["over"] * 6 + df["ball_in_over"]

print(f"[resume] Final df shape after merges: {df.shape}", flush=True)

# ---------------------------------------------------------------------------
# Imports for Steps 18–23
# ---------------------------------------------------------------------------
from src.momentum             import add_momentum_features
from src.dynamic_pressure     import add_dynamic_pressure
from src.momentum_analysis    import run_momentum_analysis
from src.advanced_modeling    import run_advanced_comparison
from src.sequence_model       import run_sequence_models
from src.interpretability_advanced import run_advanced_interpretability
from src.modeling             import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.baseline             import pressure_independence_test

# ===========================================================================
# STEP 18 — MOMENTUM FEATURE ENGINEERING
# ===========================================================================
banner("STEP 18 — MOMENTUM FEATURE ENGINEERING")
print("[main] Computing ball-by-ball momentum features from raw deliveries...", flush=True)

df_momentum = add_momentum_features(df, deliveries)


# ===========================================================================
# STEP 19 — DYNAMIC PRESSURE
# ===========================================================================
banner("STEP 19 — DYNAMIC PRESSURE COMPUTATION")
print("[main] Adding temporal EWM and momentum pressure signals...", flush=True)

df_momentum = add_dynamic_pressure(df_momentum)

print("\n[main] Dynamic pressure summary:", flush=True)
dp_cols = ["dp_ewm_fast", "dp_ewm_slow", "dp_momentum", "dp_trend_slope"]
print(df_momentum[dp_cols].describe().round(3).to_string(), flush=True)


# ===========================================================================
# STEP 20 — MOMENTUM ANALYSIS & VISUALISATION
# ===========================================================================
banner("STEP 20 — MOMENTUM & DYNAMIC PRESSURE ANALYSIS")
run_momentum_analysis(df_momentum)


# ===========================================================================
# STEP 21 — ADVANCED MODEL COMPARISON
# ===========================================================================
banner("STEP 21 — ADVANCED MODEL COMPARISON (Static → Momentum → Full Dynamic)")
print("[main] Training 3 feature configs × 2 model types...", flush=True)

advanced_results = run_advanced_comparison(df_momentum)

print("\n[main] Advanced model comparison summary:", flush=True)
pivot_adv = advanced_results.pivot_table(
    index="config", columns="model", values="roc_auc"
)
print(pivot_adv.round(4).to_string(), flush=True)

best_advanced = advanced_results.loc[advanced_results["roc_auc"].idxmax()]
print(f"\n[main] Best configuration: {best_advanced['config']} | "
      f"{best_advanced['model']} | AUC={best_advanced['roc_auc']:.4f}", flush=True)

# Capture summary stats before clearing
_momentum_gain = (
    advanced_results[advanced_results["config"] == "B_static+momentum"]["roc_auc"].max()
    - advanced_results[advanced_results["config"] == "A_static"]["roc_auc"].max()
)
_dynamic_gain = (
    advanced_results[advanced_results["config"] == "C_full_dynamic"]["roc_auc"].max()
    - advanced_results[advanced_results["config"] == "B_static+momentum"]["roc_auc"].max()
)
_best_cfg = best_advanced["config"]
_best_mdl = best_advanced["model"]
_best_auc = best_advanced["roc_auc"]

# Free Step 21 models before LSTM training to reclaim RAM
import gc
del advanced_results
gc.collect()
print("[main] Memory freed after Step 21.", flush=True)

# ===========================================================================
# STEP 22 — SEQUENCE MODELING (LSTM / GRU)
# ===========================================================================
banner("STEP 22 — SEQUENCE MODELING (LSTM / GRU)")
print("[main] Training sequence models on last-20-ball context windows...", flush=True)

try:
    seq_results = run_sequence_models(df_momentum)
    print("\n[main] Sequence model results:", flush=True)
    print(seq_results.to_string(index=False), flush=True)
except Exception as e:
    print(f"[main] Sequence modeling encountered an error: {e}", flush=True)
    seq_results = pd.DataFrame()


# ===========================================================================
# STEP 23 — ADVANCED INTERPRETABILITY
# ===========================================================================
banner("STEP 23 — ADVANCED INTERPRETABILITY (Panic Zones / Collapse Signatures)")

from sklearn.model_selection import train_test_split as sk_split
m_ids  = df_momentum["match_id"].unique()
m_lbls = df_momentum.groupby("match_id")["won"].first()
tr_ids, te_ids = sk_split(m_ids, test_size=0.2, random_state=42,
                           stratify=m_lbls[m_ids])
tr_idx_mom = df_momentum[df_momentum["match_id"].isin(tr_ids)].index
te_idx_mom = df_momentum[df_momentum["match_id"].isin(te_ids)].index

run_advanced_interpretability(df_momentum, tr_idx_mom, te_idx_mom)


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
banner("STEPS 18–23 COMPLETE — SUMMARY")

elapsed = time.time() - t0

print(f"\n  Steps 18–23 completed in {elapsed:.1f}s", flush=True)
print(f"\n{'─'*60}", flush=True)
print(f"  Dataset: {df_momentum['match_id'].nunique():,} matches | {len(df_momentum):,} ball-states", flush=True)
print(f"  Total features: {df_momentum.shape[1]} columns", flush=True)
print(f"\n  ── Momentum & dynamic results ──", flush=True)
print(f"  Best advanced config:    {_best_cfg} | {_best_mdl}", flush=True)
print(f"  Best advanced AUC:       {_best_auc:.4f}", flush=True)
print(f"  Momentum feature gain:  +{_momentum_gain:.4f} AUC over static", flush=True)
print(f"  Dynamic pressure gain:  +{_dynamic_gain:.4f} AUC over momentum", flush=True)
print(f"\n  ── Pressure & momentum statistics ──", flush=True)
print(f"  Momentum (dp_momentum) corr: "
      f"{df_momentum['dp_momentum'].corr(df_momentum['won']):.4f}", flush=True)

if not seq_results.empty:
    best_seq = seq_results.loc[seq_results["roc_auc"].idxmax()]
    print(f"\n  ── Sequence model results ──", flush=True)
    print(f"  Best sequence model:     {best_seq['model']}  AUC={best_seq['roc_auc']:.4f}", flush=True)

n_plots = len(os.listdir(PLOTS_DIR)) if os.path.exists(PLOTS_DIR) else 0
print(f"\n  All plots → outputs/plots/  ({n_plots} charts)", flush=True)
print(f"{'='*60}\n", flush=True)
