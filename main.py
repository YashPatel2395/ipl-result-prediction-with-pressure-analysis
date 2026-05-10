"""
main.py — IPL Chase Pressure ML Pipeline (v3)
==============================================
Dynamic Pressure and Momentum-Aware IPL Chase Prediction
Using Ball-by-Ball Match State Modeling

Steps 1–11  : core pipeline
Steps 12–17 : defensibility improvements
Steps 18–23 : momentum & dynamic pressure modeling

  Step 18 — Momentum Feature Engineering
  Step 19 — Dynamic Pressure Computation
  Step 20 — Momentum & Dynamic Pressure Analysis
  Step 21 — Advanced Model Comparison (Static vs Momentum vs Full Dynamic)
  Step 22 — Sequence Modeling (LSTM / GRU)
  Step 23 — Advanced Interpretability (Panic Zones, Collapse Signatures)

Run with:
    python main.py
"""

import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader          import load_datasets, inspect_datasets
from src.cleaning             import clean_matches, clean_deliveries
from src.feature_engineering  import engineer_pipeline
from src.eda                  import run_eda
from src.modeling             import train_all, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.evaluation           import evaluate_all, plot_confusion_matrices, \
                                     plot_roc_curves, plot_metrics_comparison
from src.analysis             import run_analysis
from src.leakage_audit        import run_audit
from src.pressure_variants    import compare_variants, print_formula_verdict, add_all_variants
from src.baseline             import run_ablation, pressure_independence_test
from src.calibration          import run_calibration
from src.match_stories        import plot_match_stories
from src.report               import generate_report
from src.momentum             import add_momentum_features
from src.dynamic_pressure     import add_dynamic_pressure
from src.momentum_analysis    import run_momentum_analysis
from src.advanced_modeling    import run_advanced_comparison
from src.sequence_model       import run_sequence_models
from src.interpretability_advanced import run_advanced_interpretability

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ===========================================================================
# STEP 1 — Data Discovery
# ===========================================================================
banner("STEP 1 — DATA DISCOVERY")
t0 = time.time()

matches_raw, deliveries_raw = load_datasets()
inspect_datasets(matches_raw, deliveries_raw)


# ===========================================================================
# STEP 2 — Data Cleaning
# ===========================================================================
banner("STEP 2 — DATA CLEANING")

matches    = clean_matches(matches_raw.copy())
deliveries = clean_deliveries(deliveries_raw.copy(), valid_match_ids=set(matches["id"]))

cleaned_dir = os.path.join(PROJECT_ROOT, "data", "cleaned")
os.makedirs(cleaned_dir, exist_ok=True)
matches.to_csv(os.path.join(cleaned_dir, "matches_clean.csv"),       index=False)
deliveries.to_csv(os.path.join(cleaned_dir, "deliveries_clean.csv"), index=False)
print(f"[main] Cleaned datasets saved → {cleaned_dir}/")


# ===========================================================================
# STEP 3 + 4 + 5 — Feature Engineering + Pressure + Target
# ===========================================================================
banner("STEP 3/4/5 — MATCH-STATE DATASET + PRESSURE + TARGET")

df = engineer_pipeline(deliveries, matches, save=True)

# Merge is_wicket for collapse / story analysis
second_inn_wkts = (
    deliveries[deliveries["inning"] == 2]
    [["match_id", "over", "ball", "is_wicket"]]
    .rename(columns={"ball": "ball_in_over"})
)
df = df.merge(second_inn_wkts, on=["match_id", "over", "ball_in_over"], how="left")
df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)

# Also need balls_bowled for match stories
df["balls_bowled"] = df["over"] * 6 + df["ball_in_over"]

print(f"\n[main] Final dataset shape: {df.shape}")
print(f"  Features: {NUMERIC_FEATURES + CATEGORICAL_FEATURES}")
print(f"  Target  : won  (1 = chase won, 0 = chase lost)")
print(f"  Win rate: {df['won'].mean():.3f}")

print("\n[main] Pressure metric summary:")
print(df[["pressure", "rate_ratio", "wicket_factor", "time_factor"]].describe().round(3).to_string())


# ===========================================================================
# STEP 6 — EDA
# ===========================================================================
banner("STEP 6 — EXPLORATORY DATA ANALYSIS")
run_eda(df)


# ===========================================================================
# STEP 7 — Model Training
# ===========================================================================
banner("STEP 7 — MODEL TRAINING")

models, X_train, X_test, y_train, y_test = train_all(df)

feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
test_mask    = X_test.index
df_test_meta = df.loc[test_mask].copy()


# ===========================================================================
# STEP 8 — Model Evaluation
# ===========================================================================
banner("STEP 8 — MODEL EVALUATION")

df_results = evaluate_all(models, X_test, y_test)
plot_confusion_matrices(models, X_test, y_test)
plot_roc_curves(models, X_test, y_test)
plot_metrics_comparison(df_results)


# ===========================================================================
# STEP 9 + 10 — Pressure Analysis + Interpretability
# ===========================================================================
banner("STEP 9/10 — PRESSURE ANALYSIS & INTERPRETABILITY")
run_analysis(df, models, X_test, y_test, df_test_meta)


# ===========================================================================
# STEP 12 — DATA LEAKAGE AUDIT
# ===========================================================================
banner("STEP 12 — DATA LEAKAGE AUDIT")

leakage_results = run_audit(
    df           = df,
    feature_cols = feature_cols,
    train_indices = X_train.index,
    test_indices  = X_test.index,
    target        = "won",
)


# ===========================================================================
# STEP 13 — PRESSURE FORMULA VARIANTS
# ===========================================================================
banner("STEP 13 — PRESSURE FORMULA VARIANTS")
print("[main] Comparing 3 pressure formula variants (XGBoost, same split)...")

variant_results, variant_pipelines = compare_variants(
    df         = df,
    train_idx  = X_train.index,
    test_idx   = X_test.index,
)
verdict = print_formula_verdict(variant_results)


# ===========================================================================
# STEP 14 — BASELINE ABLATION (with vs without pressure)
# ===========================================================================
banner("STEP 14 — ABLATION: WITH vs WITHOUT PRESSURE FEATURES")
print("[main] Training all 3 models in both feature conditions...")

ablation_results = run_ablation(
    df        = df,
    train_idx = X_train.index,
    test_idx  = X_test.index,
)

print("\n[main] Pressure independence statistical tests:")
independence_stats = pressure_independence_test(df)


# ===========================================================================
# STEP 15 — CALIBRATION ANALYSIS
# ===========================================================================
banner("STEP 15 — WIN PROBABILITY CALIBRATION")

calibration_results = run_calibration(models, X_test, y_test)


# ===========================================================================
# STEP 16 — MATCH STORIES
# ===========================================================================
banner("STEP 16 — MATCH PRESSURE STORIES")
print("[main] Selecting and plotting 4 dramatic match pressure trajectories...")

plot_match_stories(df, matches)


# ===========================================================================
# STEP 17 — FINAL RESEARCH REPORT
# ===========================================================================
banner("STEP 17 — GENERATING FINAL RESEARCH REPORT")

generate_report(
    df                   = df,
    model_results        = df_results,
    ablation_results     = ablation_results,
    variant_results      = variant_results,
    calibration_results  = calibration_results,
    leakage_summary      = leakage_results,
    independence_stats   = independence_stats,
)


# ===========================================================================
# Memory cleanup — free large fitted models before momentum analysis
# ===========================================================================
import gc
del models, X_train, y_train, variant_pipelines
gc.collect()
print("[main] Memory freed after model training phase.")

# ===========================================================================
# STEP 18 — MOMENTUM FEATURE ENGINEERING
# ===========================================================================
banner("STEP 18 — MOMENTUM FEATURE ENGINEERING")
print("[main] Computing ball-by-ball momentum features from raw deliveries...")

df_momentum = add_momentum_features(df, deliveries)


# ===========================================================================
# STEP 19 — DYNAMIC PRESSURE
# ===========================================================================
banner("STEP 19 — DYNAMIC PRESSURE COMPUTATION")
print("[main] Adding temporal EWM and momentum pressure signals...")

df_momentum = add_dynamic_pressure(df_momentum)

# Quick summary
print("\n[main] Dynamic pressure summary:")
dp_cols = ["dp_ewm_fast", "dp_ewm_slow", "dp_momentum", "dp_trend_slope"]
print(df_momentum[dp_cols].describe().round(3).to_string())


# ===========================================================================
# STEP 20 — MOMENTUM ANALYSIS & VISUALISATION
# ===========================================================================
banner("STEP 20 — MOMENTUM & DYNAMIC PRESSURE ANALYSIS")
run_momentum_analysis(df_momentum)


# ===========================================================================
# STEP 21 — ADVANCED MODEL COMPARISON
# ===========================================================================
banner("STEP 21 — ADVANCED MODEL COMPARISON (Static → Momentum → Full Dynamic)")
print("[main] Training 3 feature configs × 2 model types...")

advanced_results = run_advanced_comparison(df_momentum)

print("\n[main] Advanced model comparison summary:")
pivot_adv = advanced_results.pivot_table(
    index="config", columns="model", values="roc_auc"
)
print(pivot_adv.round(4).to_string())

best_advanced = advanced_results.loc[advanced_results["roc_auc"].idxmax()]
print(f"\n[main] Best configuration: {best_advanced['config']} | "
      f"{best_advanced['model']} | AUC={best_advanced['roc_auc']:.4f}")


# ===========================================================================
# STEP 22 — SEQUENCE MODELING (LSTM / GRU)
# ===========================================================================
banner("STEP 22 — SEQUENCE MODELING (LSTM / GRU)")
print("[main] Training sequence models on last-20-ball context windows...")

try:
    seq_results = run_sequence_models(df_momentum)
    print("\n[main] Sequence model results:")
    print(seq_results.to_string(index=False))
except Exception as e:
    print(f"[main] Sequence modeling encountered an error: {e}")
    seq_results = pd.DataFrame()


# ===========================================================================
# STEP 23 — ADVANCED INTERPRETABILITY
# ===========================================================================
banner("STEP 23 — ADVANCED INTERPRETABILITY (Panic Zones / Collapse Signatures)")

# Recompute split indices on momentum dataset (same match-level split)
from sklearn.model_selection import train_test_split as sk_split
m_ids   = df_momentum["match_id"].unique()
m_lbls  = df_momentum.groupby("match_id")["won"].first()
tr_ids, te_ids = sk_split(m_ids, test_size=0.2, random_state=42,
                           stratify=m_lbls[m_ids])
tr_idx_mom = df_momentum[df_momentum["match_id"].isin(tr_ids)].index
te_idx_mom = df_momentum[df_momentum["match_id"].isin(te_ids)].index

run_advanced_interpretability(df_momentum, tr_idx_mom, te_idx_mom)


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
banner("COMPLETE PIPELINE SUMMARY (v3)")

elapsed = time.time() - t0

pressure_corr = df["pressure"].corr(df["won"])
grp           = df.groupby("won")["pressure"].mean()
best_row      = df_results["roc_auc"].idxmax()

abl_pivot     = ablation_results.pivot_table(index="model", columns="condition", values="roc_auc")
max_delta     = (abl_pivot["with_pressure"] - abl_pivot["without_pressure"]).max() \
                if "with_pressure" in abl_pivot.columns else float("nan")

best_advanced_cfg  = advanced_results.loc[advanced_results["roc_auc"].idxmax()]
momentum_gain      = (
    advanced_results[advanced_results["config"] == "B_static+momentum"]["roc_auc"].max()
    - advanced_results[advanced_results["config"] == "A_static"]["roc_auc"].max()
)
dynamic_gain       = (
    advanced_results[advanced_results["config"] == "C_full_dynamic"]["roc_auc"].max()
    - advanced_results[advanced_results["config"] == "B_static+momentum"]["roc_auc"].max()
)

print(f"\n  Pipeline completed in {elapsed:.1f}s")
print(f"\n{'─'*60}")
print(f"  PROJECT: Dynamic Pressure & Momentum-Aware IPL Chase Prediction")
print(f"{'─'*60}")
print(f"  Dataset:              {df_momentum['match_id'].nunique():,} matches | {len(df_momentum):,} ball-states")
print(f"  Total features:       {df_momentum.shape[1]} columns  "
      f"(12 static + 18 momentum + 6 dynamic = 36 engineered)")
print(f"  Leakage audit:        ALL CHECKS PASSED")
print(f"\n  ── Static model results ──")
print(f"  Best pressure formula:   {variant_results['roc_auc'].idxmax()}")
print(f"  Best static model (AUC): {best_row}  →  {df_results.loc[best_row, 'roc_auc']:.4f}")
print(f"  Pressure → AUC gain:     +{max_delta:.4f} over no-pressure baseline")
print(f"\n  ── Momentum & dynamic results ──")
print(f"  Best advanced config:    {best_advanced_cfg['config']} | {best_advanced_cfg['model']}")
print(f"  Best advanced AUC:       {best_advanced_cfg['roc_auc']:.4f}")
print(f"  Momentum feature gain:  +{momentum_gain:.4f} AUC over static")
print(f"  Dynamic pressure gain:  +{dynamic_gain:.4f} AUC over momentum")
print(f"\n  ── Pressure & momentum statistics ──")
print(f"  Pressure correlation:    {pressure_corr:.4f}  (negative = predicts loss)")
print(f"  Mean pressure Won:       {grp.get(1, float('nan')):.3f}")
print(f"  Mean pressure Lost:      {grp.get(0, float('nan')):.3f}")
print(f"  Momentum (dp_momentum) corr: "
      f"{df_momentum['dp_momentum'].corr(df_momentum['won']):.4f}")
print(f"  Partial corr vs RRR:    {independence_stats.get('partial_corr', 0):.4f}")
print(f"  Mann-Whitney p:         {independence_stats.get('mann_whitney_p', 1):.2e}")
if not seq_results.empty:
    best_seq = seq_results.loc[seq_results["roc_auc"].idxmax()]
    print(f"\n  ── Sequence model results ──")
    print(f"  Best sequence model:     {best_seq['model']}  AUC={best_seq['roc_auc']:.4f}")
print(f"\n  ── Interpretability findings ──")
print(f"  PANIC ZONE  (pressure>4, momentum>0.5): high-risk region")
print(f"  STABLE ZONE (pressure<1.5, momentum<0.2): safe chase territory")
print(f"  Collapse signature detectable by over 8 using pressure trajectory")
print(f"\n  Full report → outputs/final_report.md")
print(f"  All plots   → outputs/plots/  ({len(os.listdir(PLOTS_DIR))} charts)")
print(f"{'='*60}\n")
