"""
modeling.py
===========
Train and persist three models on the chase-state dataset:
  1. Logistic Regression (baseline linear)
  2. Random Forest       (ensemble, handles non-linearity)
  3. XGBoost             (gradient-boosted, industry benchmark)

Design choices:
  - Match-level train/test split to prevent data leakage across balls of
    the same match appearing in both sets.
  - Categorical features (phase, toss_decision) one-hot encoded.
  - Numerical features standardised for Logistic Regression only
    (tree models are scale-invariant).
  - Class imbalance handled via class_weight / scale_pos_weight.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# ------------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------------
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

CATEGORICAL_FEATURES = [
    "phase",
    "toss_decision",
]

TARGET = "won"


def prepare_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split at the MATCH level so no match's balls straddle train/test.
    Returns X_train, X_test, y_train, y_test DataFrames/Series.
    """
    match_ids    = df["match_id"].unique()
    match_labels = df.groupby("match_id")["won"].first()

    train_ids, test_ids = train_test_split(
        match_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=match_labels[match_ids],
    )

    train_df = df[df["match_id"].isin(train_ids)]
    test_df  = df[df["match_id"].isin(test_ids)]

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    print(f"[modeling] Train: {len(train_df):,} rows ({len(train_ids)} matches) | "
          f"Test: {len(test_df):,} rows ({len(test_ids)} matches)")
    return X_train, X_test, y_train, y_test


def _preprocessor(scale: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features."""
    num_pipeline = Pipeline([("scaler", StandardScaler())]) if scale else "passthrough"

    return ColumnTransformer([
        ("num",  num_pipeline,                NUMERIC_FEATURES),
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CATEGORICAL_FEATURES),
    ])


def train_logistic_regression(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("pre", _preprocessor(scale=True)),
        ("clf", LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    print("[modeling] Logistic Regression trained.")
    return pipe


def train_random_forest(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("pre", _preprocessor(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=50,       # prevents memorising single balls
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    print("[modeling] Random Forest trained.")
    return pipe


def train_xgboost(X_train, y_train) -> Pipeline:
    neg, pos    = (y_train == 0).sum(), (y_train == 1).sum()
    scale_w     = neg / pos if pos > 0 else 1.0

    pipe = Pipeline([
        ("pre", _preprocessor(scale=False)),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_w,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    print("[modeling] XGBoost trained.")
    return pipe


def save_model(model: Pipeline, name: str) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[modeling] Saved → {path}")


def load_model(name: str) -> Pipeline:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def train_all(df: pd.DataFrame):
    """Full training pipeline. Returns (models_dict, X_train, X_test, y_train, y_test)."""
    X_train, X_test, y_train, y_test = prepare_splits(df)

    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest":       train_random_forest(X_train, y_train),
        "xgboost":             train_xgboost(X_train, y_train),
    }

    for name, model in models.items():
        save_model(model, name)

    return models, X_train, X_test, y_train, y_test
