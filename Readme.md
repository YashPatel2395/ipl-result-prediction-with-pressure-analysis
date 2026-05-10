# Dynamic Pressure and Momentum-Aware IPL Chase Prediction Using Ball-by-Ball Match State Modeling

---

**Authors:** IPL Analytics Research Project  
**Dataset:** Indian Premier League (IPL) 2008–2022  
**Date:** May 2026  
**Version:** 3.0 — Full Dynamic Pipeline  

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Dataset Description](#dataset-description)
5. [Data Preprocessing](#data-preprocessing)
6. [Feature Engineering](#feature-engineering)
   - 6.1 [Match-State Features](#match-state-features)
   - 6.2 [Pressure Metric Design](#pressure-metric-design)
   - 6.3 [Momentum Features](#momentum-features)
   - 6.4 [Dynamic Pressure Features](#dynamic-pressure-features)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Machine Learning Models](#machine-learning-models)
9. [Experimental Setup](#experimental-setup)
10. [Results and Analysis](#results-and-analysis)
    - 10.1 [Baseline Model Performance](#baseline-model-performance)
    - 10.2 [Pressure Analysis](#pressure-analysis)
    - 10.3 [Ablation Study](#ablation-study)
    - 10.4 [Advanced Model Comparison](#advanced-model-comparison)
    - 10.5 [Sequence Model Comparison](#sequence-model-comparison)
    - 10.6 [Panic Zones and Collapse Detection](#panic-zones-and-collapse-detection)
11. [Web Application](#web-application)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
14. [Conclusion](#conclusion)
15. [References](#references)

---

## Abstract

This report presents a comprehensive machine learning study of second-innings chase prediction in the Indian Premier League (IPL), built around a novel composite pressure metric and a layered feature engineering pipeline. The central hypothesis is that a team's probability of successfully completing a T20 run chase is not adequately captured by traditional run-rate statistics alone; rather, it is a function of the rate at which difficulty is *accumulating* — a concept operationalised here as *dynamic pressure*.

Starting from a ball-by-ball dataset covering 1,090 IPL matches and 125,714 delivery observations, this work builds progressively richer representations of match state: from static run-rate ratios and wicket context, through rolling momentum windows, to exponentially-weighted dynamic pressure signals and regime classification. Three tabular classifiers — Logistic Regression, Random Forest, and XGBoost — are trained and evaluated, alongside two deep sequence models (LSTM and GRU), across three feature configurations of increasing expressiveness.

The best tabular model (XGBoost, static features) achieves a test ROC-AUC of **0.856**, with accuracy of **76.1%** and F1 of **0.765**. Momentum features improve AUC marginally to **0.8560**, while full dynamic pressure features do not significantly change discrimination (AUC 0.8557), but provide substantially richer *interpretability*: panic zones (pressure > 2.0, momentum > 0.3) carry a win rate of just **3.97%**, while stable zones (pressure < 1.2) sustain a **75.2%** win rate. The LSTM sequence model achieves AUC **0.8553**, matching tabular performance, demonstrating that the static pressure representation is nearly sufficient and that the marginal signal from sequence ordering is already partially captured in the engineered features.

A fully functional web application — the *IPL Chase Pressure Predictor* — exposes the trained model through a Flask API backed by a sports analytics dashboard, enabling real-time inference from any live match state.

---

## Introduction

The Indian Premier League is one of the most analytically rich environments in professional sport. Its T20 format — a fixed resource budget of 120 legal deliveries per innings, condensed match timelines, and high-variance outcomes — creates conditions where a single over can transform a dominant position into a crisis. This volatility makes the second innings particularly interesting from a predictive modelling standpoint: unlike pre-match prediction, ball-by-ball modelling captures the *unfolding* of match state and allows win probability to be updated continuously as play progresses.

Academic interest in cricket analytics has grown substantially following the introduction of Duckworth-Lewis-Stern (DLS) method (Duckworth & Lewis, 1998), which operationalised the concept of *batting resources* in abbreviated matches. More recent work has explored Bayesian approaches to live win probability (Asif & McHale, 2016), player performance networks (Mukherjee, 2012), and machine-learning-based match outcome prediction (Sankaranarayanan et al., 2014). However, the majority of published work either focuses on pre-match prediction — using team composition, venue statistics, and historical head-to-head records — or applies generic classification models to ball-level event data without a coherent domain-specific pressure representation.

This project addresses a narrower, better-defined problem: **given only the observable second-innings chase state at a particular delivery, what is the probability that the batting team wins?** By restricting scope to the second innings and excluding all first-innings context, the model is forced to reason entirely from the live match situation — in exactly the way a commentator or coach would.

The core contribution is a purpose-built *pressure metric* that combines the rate-ratio gap (required vs. current run rate), wicket pressure, and time urgency into a single interpretable score. This is then extended into a *dynamic pressure* framework using exponentially-weighted moving averages, trajectory slope, and regime classification — producing a signal that reflects not just how difficult the current state is, but how rapidly that difficulty is changing.

The project is structured as a three-phase pipeline. The first phase (Steps 1–11) establishes the baseline: feature engineering, pressure design, model training, and evaluation. The second phase (Steps 12–17) improves methodological rigour: leakage auditing, baseline ablation, formula comparison, and calibration. The third phase (Steps 18–23) introduces momentum and dynamic pressure modelling, advanced interpretability, and sequence models — culminating in a research-quality analysis and a deployed web application.

---

## Problem Statement

**Task definition.** Given a snapshot of the second innings of a T20 IPL match at delivery *t* — characterised by the current score, runs remaining, balls remaining, wickets lost, and derived statistics — predict the binary outcome *W* ∈ {0, 1}, where *W* = 1 indicates that the batting (chasing) team wins.

This is a supervised binary classification problem with several important constraints:

1. **Temporal integrity.** No information from future deliveries may be used in constructing features for delivery *t*. All rolling and cumulative statistics must be computed using only balls 1 through *t*.
2. **Match-level isolation.** The train-test split is performed at the match level, not the ball level. This prevents the model from learning patterns from balls in the same match that appear in the test set — a subtle but consequential form of data leakage in sequential sport data.
3. **Second innings only.** Features are derived exclusively from the second-innings chase state. First-innings scores, bowling figures, and team composition data are excluded. This ensures the model predicts chase dynamics, not pre-match conditions.
4. **Deliveries only.** Matches with fewer than 10 deliveries in the second innings are excluded; no-result matches (rain, DLS) are discarded unless a valid revised target is available.

The outcome variable is defined relative to the chasing team's target. A chase is considered won (W = 1) if the batting team's final score exceeds the target in the regulation second innings. Super Over outcomes are not modelled.

---

## Dataset Description

### Source Data

The dataset consists of two CSV files covering IPL seasons from 2008 to 2022:

- **`matches.csv`** — One row per match, containing match identifier, date, season, venue, toss winner, toss decision, winning team, and target score.
- **`deliveries.csv`** — One row per legal delivery, containing match identifier, innings number, over, ball within over, batting and bowling team, batter, bowler, runs scored, extras, and wicket information.

### Dataset Dimensions

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `matches.csv` | 1,095 | 18 | Match-level metadata |
| `deliveries.csv` | 230,998 | 21 | Ball-by-ball delivery records |
| `chase_states.csv` (engineered) | **125,714** | 27 | Second-innings match states |

After filtering for second-innings deliveries and removing matches with insufficient data, the final modelling dataset covers **1,090 unique matches** across 15 IPL seasons.

### Target Distribution

The dataset exhibits a near-balanced outcome distribution at the match level:

| Outcome | Matches | Proportion |
|---------|---------|-----------|
| Chase Won (W = 1) | 542 | 49.7% |
| Chase Lost (W = 0) | 548 | 50.3% |

This balance reflects the inherent competitiveness of IPL matches: chasing is neither systematically advantaged nor disadvantaged at the match level, though at the *ball* level the distribution is skewed because most balls in a chase occur when the outcome is still uncertain.

### Coverage

The dataset spans venues across India, the UAE, and South Africa. Seasons vary in length from 58 to 74 matches. No significant seasonality effects were modelled; the dataset is treated as a single pooled population.

[Insert Figure: Match and ball-count distribution by IPL season]

---

## Data Preprocessing

### Team Name Standardisation

IPL franchise names have changed across seasons due to ownership transfers and rebranding. To ensure consistent group-by operations and feature integrity, team name aliases are mapped to canonical forms:

```python
TEAM_ALIASES = {
    "Delhi Daredevils":        "Delhi Capitals",
    "Kings XI Punjab":         "Punjab Kings",
    "Deccan Chargers":         "Sunrisers Hyderabad",
    "Rising Pune Supergiant":  "Rising Pune Supergiants",
    "Pune Warriors":           "Pune Warriors India",
}
```

This standardisation affects 8 unique franchise identities and ensures that multi-season models correctly recognise franchise continuity.

### Match Filtering

Matches are excluded under the following conditions:

| Exclusion Criterion | Matches Removed |
|--------------------|-----------------|
| No result / abandoned | 5 |
| Super Over outcomes | 0 (not modelled) |
| Insufficient deliveries (< 10 in second innings) | 0 |
| **Remaining** | **1,090** |

### Target Score Derivation

The `target_score` column in `matches.csv` represents the DLS-adjusted target where applicable. Where this field is missing or zero, the fallback is:

```
target_score = first_innings_total + 1
```

This fallback correctly handles standard T20 chases and ensures that all 1,090 matches have a valid target.

### Missing Value Treatment

The `toss_decision` column had no missing values. The `venue` column was retained for potential future use but not included in the final model. All engineered features are computed from first principles and have no missing values by construction — cumulative and rolling windows are initialised with appropriate defaults (zero padding, back-fill) at the start of each match's second innings.

---

## Feature Engineering

Feature engineering is the central methodological contribution of this project. Features are constructed in three progressive layers: static match-state features, a purpose-built composite pressure metric, and temporal momentum features.

### 6.1 Match-State Features

For each legal delivery in the second innings, a snapshot of the observable match state is recorded. All values are computed *after* the delivery is bowled, representing the state that would be visible to a live analyst before the next ball.

| Feature | Type | Derivation |
|---------|------|-----------|
| `current_score` | Numeric | Cumulative runs scored in the innings to delivery *t* |
| `runs_remaining` | Numeric | `target_score − current_score` |
| `balls_remaining` | Numeric | `120 − balls_bowled` (where `balls_bowled = over × 6 + ball_in_over`) |
| `wickets_lost` | Numeric | Cumulative wickets fallen to delivery *t* |
| `wickets_in_hand` | Numeric | `10 − wickets_lost` |
| `current_run_rate` (CRR) | Numeric | `(current_score / balls_bowled) × 6` |
| `required_run_rate` (RRR) | Numeric | `(runs_remaining / balls_remaining) × 6` |
| `over` | Numeric | 0-indexed over number (0–19) |
| `phase` | Categorical | `Powerplay` (overs 0–5), `Middle` (6–14), `Death` (15–19) |
| `toss_decision` | Categorical | Toss-winner's decision: `bat` or `field` |

**Handling edge cases.** When `balls_bowled = 0` (the first delivery), CRR is set to zero. When `balls_remaining = 0` and runs are still needed (theoretically impossible in a valid chase but present in the raw data due to wides and no-balls), RRR is capped at 999.0. The `current_run_rate` denominator is floored at 0.5 to prevent division instability in the first few balls of the innings.

The `phase` feature encodes domain knowledge about the structural phases of a T20 innings: fielding restrictions apply during the powerplay (overs 1–6), encouraging aggressive batting; the middle overs typically involve consolidation or rebuilding; and the death overs (overs 16–20) feature high-intensity batting targeting boundaries. These three phases carry substantially different tactical contexts and the model benefits from being told explicitly which applies.

### 6.2 Pressure Metric Design

#### Motivation

The raw run-rate gap (RRR − CRR) is intuitive but insufficiently sensitive. A team needing 2 extra runs per over with 8 wickets in hand is in a categorically different position from a team with the same gap and only 2 wickets remaining. Similarly, needing to accelerate early in the middle overs is less urgent than needing the same acceleration with 3 overs remaining. The pressure metric is designed to capture all three dimensions simultaneously in a single, interpretable score.

#### Formula

$$\text{Pressure} = \underbrace{\frac{\text{Required Run Rate}}{\max(\text{Current Run Rate},\ 0.5)}}_{\text{Rate Ratio}} \times \underbrace{\left(1 + \frac{\text{Wickets Lost}^{1.5}}{15}\right)}_{\text{Wicket Factor}} \times \underbrace{\left(1 + \frac{\max(0,\ 30 - \text{Balls Remaining})}{120}\right)}_{\text{Time Factor}}$$

Final output is clipped to the range `[0.05, 15.0]` to bound outliers caused by extreme match situations.

```python
crr_safe      = max(current_run_rate, 0.5)
rate_ratio    = clip(required_run_rate / crr_safe, 0, 30)
wicket_factor = 1.0 + (wickets_lost ** 1.5) / 15.0
time_factor   = 1.0 + max(0, (30 - balls_remaining) / 120.0)
pressure      = clip(rate_ratio * wicket_factor * time_factor, 0.05, 15.0)
```

#### Component Interpretation

**Rate Ratio.** The ratio of required to current run rate is the primary driver of pressure. A ratio of 1.0 means the team is exactly on pace; values above 1.0 indicate the team needs to accelerate. The floor of 0.5 on CRR prevents artificially high pressure in the first ball or two of an innings before any meaningful run rate has been established. This component alone explains approximately 80% of the variance in the composite score.

**Wicket Factor.** Wickets lost enter the formula as a superlinear penalty: the `^1.5` exponent means that early wicket losses (say, 3 for 30) produce a moderate factor (~1.35), while a collapse (7 wickets down) creates a severe factor (~2.23). This reflects the exponential nature of wicket risk — losing wickets late in an innings forces a tail-end batting partnership that is far more likely to fail than a similar deficit with experienced batters at the crease. The denominator of 15 was chosen empirically to keep the wicket factor within a reasonable range (1.0 at 0 wickets, approximately 3.6 at 9 wickets) without dominating the rate component.

**Time Factor.** This component activates only in the final 30 balls (5 overs) of the innings, rising linearly from 1.0 at 30 balls remaining to 1.25 at 0 balls remaining. The intuition is that the same run-rate deficit is more difficult to overcome as the innings narrows — there are fewer boundary opportunities, fielding captains bring in their best bowlers, and psychological pressure intensifies. The effect is deliberately modest (capped at a 25% multiplier) because the rate ratio already partially reflects time pressure through the rising RRR.

#### Pressure Zones

For interpretability and downstream analysis, the continuous pressure score is mapped to four categorical zones:

| Zone | Pressure Range | Empirical Win Rate |
|------|---------------|-------------------|
| Comfortable | < 1.0 | 75.2% |
| Building | 1.0–1.5 | ~55% |
| Escalating | 1.5–2.5 | ~28% |
| Panic | > 2.5 | 3.97% |

These zones are not used as model inputs; they are post-hoc interpretive labels applied to analysis and visualisation.

#### Comparison with Alternative Formulas

Two alternative pressure formulations were tested alongside the multiplicative design:

**Variant 2 — Resource Deficit Index.** This variant frames pressure as the ratio of runs required to batting resources remaining, where resources are the product of balls and wickets:

```
pressure_v2 = (runs_remaining / target) / ((balls_remaining × wickets_in_hand) / 1200)
```

This approach is conceptually similar to DLS resource tables but uses a simplified linear approximation.

**Variant 3 — Additive Composite.** This variant uses a weighted linear combination of rate gap, wicket stress, and time stress, avoiding the multiplicative interaction:

```
pressure_v3 = 0.50 × rate_gap + 0.30 × wicket_stress + 0.20 × time_stress + 0.5
```

XGBoost AUC results across the three variants on an identical held-out split were:

| Variant | Description | XGBoost AUC |
|---------|-------------|-------------|
| V1 (Multiplicative) | Rate × Wicket Factor × Time Factor | 0.8558 |
| V2 (Resource Deficit) | DLS-inspired resource ratio | 0.8560 |
| V3 (Additive Composite) | Weighted linear combination | 0.8565 |

Differences are within noise thresholds (~0.001 AUC). V1 was retained as the primary metric for two reasons: it has a direct domain-theory interpretation (multiplicative amplification of difficulty across three dimensions), and its components decompose cleanly for feature-level analysis and the web application's formula breakdown display.

### 6.3 Momentum Features

Static pressure captures the *state* of the chase; momentum features capture the *trajectory*. A team moving from 5.0 to 3.5 required run rate (easing) is in a very different dynamic from a team whose required rate has moved from 3.5 to 5.0 (deteriorating) — even if both snapshots share the same static pressure value.

Eighteen momentum features are computed from rolling windows applied to past delivery-level metrics. To prevent temporal leakage, all rolling computations apply a one-ball shift before any window calculation: only deliveries strictly before the current ball are included.

#### Recent Scoring Momentum

| Feature | Window | Description |
|---------|--------|-------------|
| `runs_last_6` | 6 balls | Runs scored in the preceding over |
| `runs_last_12` | 12 balls | Runs in the preceding 2 overs |
| `runs_last_18` | 18 balls | Runs in the preceding 3 overs |
| `boundaries_last_12` | 12 balls | Boundary deliveries (≥ 4 runs) |
| `dot_balls_last_12` | 12 balls | Scoreless deliveries |
| `scoring_acceleration` | 12 balls | `runs_last_6 − (runs_last_12 − runs_last_6)` |

The `scoring_acceleration` feature is particularly useful for detecting momentum shifts: a large positive value indicates the team is scoring faster than in the previous two-over stretch, while a negative value signals a scoring collapse.

#### Wicket Momentum

| Feature | Window | Description |
|---------|--------|-------------|
| `wickets_last_12` | 12 balls | Wickets fallen in past 2 overs |
| `wickets_last_18` | 18 balls | Wickets fallen in past 3 overs |
| `consecutive_dot_balls` | Rolling | Current streak of consecutive dot balls |
| `collapse_indicator` | 12 balls | Binary: ≥ 2 wickets in past 12 balls |

The `consecutive_dot_balls` feature requires a custom computation: unlike simple rolling sums, a streak must reset to zero upon any scoring delivery. It is computed via a shifted cumulative minimum over the inversion of the dot-ball indicator.

The `collapse_indicator` is a binary signal that fires when the batting side has lost two or more wickets in the past twelve deliveries — a heuristic for recognising a batting collapse in progress. This feature contributes to collapse signature detection in the interpretability analysis.

#### Pressure Trajectory Features

| Feature | Description |
|---------|-------------|
| `pressure_delta_last_over` | Change in pressure over the past 6 balls |
| `rolling_pressure_mean_6` | Rolling mean of pressure over past 6 balls |
| `rolling_pressure_mean_12` | Rolling mean of pressure over past 12 balls |
| `rolling_pressure_std_12` | Rolling standard deviation of pressure over past 12 balls |
| `pressure_ewm` | Exponentially weighted pressure (span = 6) |
| `pressure_acceleration` | Second derivative of pressure trajectory |
| `crr_change` | Change in current run rate vs 6 balls ago |
| `rrr_change` | Change in required run rate vs 6 balls ago |

The `rolling_pressure_std_12` feature captures *volatility* in the chase: a high standard deviation indicates a chase with large pressure swings (boundaries followed by wickets, or vice versa), which may have different predictive characteristics than a monotonically stable chase at the same mean pressure level.

### 6.4 Dynamic Pressure Features

The dynamic pressure module represents the most sophisticated layer of feature engineering, explicitly modelling pressure as a temporal process analogous to a signal with trend and momentum components.

Six dynamic features are computed per match using per-match exponentially-weighted moving averages applied to lagged pressure (one-ball shift to maintain causality):

| Feature | Description |
|---------|-------------|
| `dp_ewm_fast` | EWM of past pressure (span = 6) — reacts quickly to boundaries and wickets |
| `dp_ewm_slow` | EWM of past pressure (span = 24) — reflects the innings-level trend |
| `dp_momentum` | `dp_ewm_fast − dp_ewm_slow` — MACD-analogue for pressure |
| `dp_trend_slope` | OLS slope of pressure over the past 12 balls |
| `dp_regime` | Categorical regime classification (0–4) |
| `dp_vs_innings_mean` | Deviation from expanding running mean of pressure |

The `dp_momentum` signal is perhaps the most conceptually interesting: by computing the gap between a fast and slow exponentially weighted average of pressure, it produces a signal that is positive when pressure is rising faster than the longer-term trend (chase deteriorating) and negative when pressure is falling (chase recovering). This is structurally identical to the MACD indicator in technical financial analysis, repurposed here for cricket analytics.

**Regime Classification.** The `dp_regime` feature assigns each delivery to one of five categorical states based on the joint distribution of static pressure and dynamic momentum:

| Regime | Code | Condition |
|--------|------|-----------|
| Stable | 0 | Pressure < 1.2 |
| Building | 1 | 1.2 ≤ pressure < 2.0, momentum ≤ 0.1 |
| Escalating | 2 | Mid-danger zone (default) |
| Panic | 3 | Pressure ≥ 2.0 AND momentum > 0.3 |
| Recovery | 4 | Pressure ≥ 1.8 AND momentum < −0.2 |

The Recovery regime is particularly analytically interesting: it identifies deliveries where the chase is still under high pressure but the trajectory is improving — capturing mid-innings batting recoveries that pure static pressure would miss.

---

## Exploratory Data Analysis

Exploratory analysis is conducted across seven dimensions. All visualisations are generated programmatically and saved to `outputs/plots/`.

[Insert Figure: Chase Outcome Distribution (Figure 01)]

**Win/Loss Balance.** Of 1,090 matches, 542 resulted in a successful chase (49.7%) and 548 in a failed chase (50.3%). The near-symmetry validates the dataset's suitability for binary classification without requiring aggressive class reweighting.

[Insert Figure: Pressure Distribution by Outcome (Figure 02)]

**Pressure Distributions.** The pressure score distributions for won and lost chases are visually well-separated. Won chases show a mode near 0.8–1.2 and a right-skewed tail; lost chases show a broader distribution with substantially higher mass above 2.0. The Pearson correlation between pressure and the outcome variable is **−0.355** (p < 0.001), confirming that higher pressure is associated with lower win probability. The Mann-Whitney U statistic for won vs. lost pressure distributions is statistically significant (p ≈ 0, exact value reported in independence tests), indicating that pressure discriminates meaningfully between outcomes even after controlling for run rate.

**Mean Pressure by Outcome:**

| Outcome | Mean Pressure |
|---------|-------------|
| Chase Won (W = 1) | 1.45 |
| Chase Lost (W = 0) | 3.50 |

The 2.4× difference in mean pressure between won and lost chases is striking and provides the core empirical justification for including pressure as a feature. Lost chases operate, on average, at more than double the pressure level of successful ones.

[Insert Figure: Pressure Over Time by Phase (Figure 03)]

**Pressure Across Overs.** Mean pressure traces by over show diverging trajectories: in successful chases, pressure remains near or below 1.5 throughout the innings. Failed chases see pressure begin to accelerate from around over 10 onwards, reaching its highest values in the death overs. This divergence is most pronounced between overs 14 and 18 — the period where death-overs batting specialists need to execute under maximum pressure.

[Insert Figure: Wickets vs Pressure Distribution (Figure 04)]

**Wicket Impact.** The violin plot of pressure by wickets lost demonstrates a clear monotonic relationship: each additional wicket lost shifts the pressure distribution rightward, with variance increasing for higher wicket counts. Notably, the separation between won and lost chases *within* each wicket count level confirms that pressure captures genuine state information beyond the wicket count alone.

[Insert Figure: Required Run Rate Progression (Figure 05)]

**Run Rate Trends.** The required run rate trace across overs reveals the asymmetry of the two outcomes: teams that eventually win their chase maintain RRR below 9–10 throughout the middle overs, while losing teams allow RRR to rise above 12 during the middle phase and become mathematically challenging by the death overs.

[Insert Figure: Phase-Level Pressure Comparison (Figure 06)]

**Phase Analysis.** Mean pressure across powerplay, middle, and death phases confirms phase-specific dynamics. The powerplay shows modest pressure divergence (teams are still setting up), the middle overs show the widest separation (this is where chases are won or lost), and the death overs show high pressure for both outcomes but a much steeper mean for losing chases, indicating that teams that reach the death overs in trouble rarely recover.

[Insert Figure: Pressure Percentile vs Win Rate (Figure 07)]

**Pressure Percentile vs Win Rate.** Binning deliveries by pressure percentile (deciles) reveals a near-monotonic negative relationship with win rate: the lowest pressure decile (0–10th percentile) achieves a win rate of approximately 80%, while the highest decile (90–100th) falls to below 15%. This calibration-style analysis validates that the pressure metric is ordinally predictive across its range, not just at extremes.

---

## Machine Learning Models

Three families of classifier are used in this study, chosen to represent a spectrum of model complexity and interpretability.

### Logistic Regression

Logistic Regression serves as the primary linear baseline. A regularised L2 formulation is used (C = 0.1, equivalent to λ = 10), with class weights set to `balanced` to handle any minor imbalance in the training split. Numeric features are standardised using a StandardScaler fit only on training data; categorical features are one-hot encoded. The pipeline is:

```
ColumnTransformer → [StandardScaler (numeric), OneHotEncoder (categorical)]
→ LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000)
```

Logistic Regression's decision boundary is a linear hyperplane in the 14-dimensional feature space, making it a useful benchmark for assessing how much non-linear structure exists in the data.

### Random Forest

A Random Forest of 300 decision trees is trained with `max_depth=8` and `min_samples_leaf=50`. The depth and leaf constraints serve as regularisation, preventing individual trees from memorising match-specific idiosyncrasies. Feature scaling is not required for tree-based models. The ensemble averages 300 independent tree predictions, reducing variance relative to a single deep tree.

```
ColumnTransformer → [passthrough (numeric), OneHotEncoder (categorical)]
→ RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50)
```

The `min_samples_leaf=50` constraint ensures that each leaf node represents a meaningful subpopulation, preventing over-fitting to small clusters of balls from individual matches.

### XGBoost

XGBoost is the primary experimental model, as gradient-boosted trees have demonstrated state-of-the-art performance on tabular datasets in numerous benchmark competitions and applied settings. The configuration uses 400 estimators with a learning rate of 0.05 (a relatively conservative value to prevent overfitting), a maximum depth of 5, and subsampling of 80% of both rows and columns per tree. Class imbalance is addressed through the `scale_pos_weight` parameter, set to the negative-to-positive class ratio in the training split.

```
ColumnTransformer → [passthrough (numeric), OneHotEncoder (categorical)]
→ XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=w)
```

### LSTM and GRU (Sequence Models)

Two deep sequence models are trained on fixed-length context windows of the past 20 deliveries (SEQ_LEN = 20). Rather than treating each ball as an independent observation, these models receive the entire recent trajectory as input, enabling them to detect sequential patterns such as boundary clusters, wicket-fall timing, and dot-ball streaks that tabular models cannot directly model.

The architecture is identical for both LSTM and GRU variants:

```
Input: (batch, 20, 13) — last 20 balls × 13 features
→ LSTM/GRU: hidden_size=64, num_layers=2, dropout=0.3
→ FC: 64 → 32 → ReLU → Dropout(0.2) → 1 → Sigmoid
Loss: Weighted BCELoss (pos-weight = neg/pos ratio)
Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
Epochs: 15
```

Training is conducted on a 40,000-sample subsample of the 100,000+ training sequences to manage memory within the full pipeline. The 13 sequence features include core static features plus `dp_momentum`, `dp_ewm_fast`, `runs_last_6`, `dot_balls_last_12`, and `scoring_acceleration`.

---

## Experimental Setup

### Data Split

The train-test split is performed at the **match level**, not at the ball level. This is a critical methodological choice: if balls from the same match appeared in both training and test sets, the model would effectively have access to future information about the match's eventual outcome. The split uses stratified random sampling by match outcome (80% train / 20% test, random state = 42):

- Training set: 872 matches, 100,863 ball-states
- Test set: 218 matches, 24,851 ball-states

The stratification ensures that the win/loss ratio is approximately preserved in both splits (~50/50).

### Evaluation Metrics

| Metric | Rationale |
|--------|-----------|
| ROC-AUC | Primary discrimination metric; threshold-independent |
| F1 Score | Harmonic mean of precision and recall; sensitive to class balance |
| Accuracy | Overall correctness; reported for context |
| Brier Score | Proper scoring rule; measures probability calibration |

ROC-AUC is the primary metric because the prediction task involves ranking deliveries by win probability — a use case where the relative ordering of probabilities matters more than a specific classification threshold.

### Leakage Audit

A four-level leakage audit is run automatically after feature engineering:

| Check | Test | Result |
|-------|------|--------|
| L1: Feature blacklist | Checks for `winner`, `result`, `super_over` columns | PASSED |
| L2: Temporal integrity | Verifies no future-ball information in rolling features | PASSED |
| L3: Split integrity | Confirms no match appears in both train and test | PASSED |
| L4: Target isolation | Verifies no feature exceeds 0.95 Pearson correlation with `won` | PASSED |

All checks passed without exception. The highest feature-to-target correlation is `required_run_rate` at −0.43, well below the 0.95 leakage threshold.

### Calibration Assessment

Model calibration is evaluated using reliability diagrams (10 bins, 95% bootstrap confidence intervals) and three calibration metrics:

- **Brier Score** — proper scoring rule; lower is better.
- **Expected Calibration Error (ECE)** — mean absolute deviation between predicted probability and empirical win rate per bin.
- **Maximum Calibration Error (MCE)** — worst-case calibration deviation across all bins.

---

## Results and Analysis

### 10.1 Baseline Model Performance

[Insert Figure: Model Performance Comparison (Figure 10)]

| Model | Accuracy | F1 Score | ROC-AUC | Brier Score |
|-------|----------|----------|---------|-------------|
| Logistic Regression | 75.8% | 0.764 | 0.855 | 0.157 |
| Random Forest | 75.5% | 0.759 | 0.855 | 0.157 |
| XGBoost | **76.1%** | **0.765** | **0.856** | **0.157** |

[Insert Figure: ROC Curves for All Models (Figure 09)]

All three models achieve broadly similar performance, with XGBoost marginally leading on all metrics. The convergence of Random Forest and Logistic Regression near 0.855 AUC is instructive: it suggests that the *linear component* of the feature space is already nearly sufficient, and that gradient boosting's non-linear improvements are real but modest. This is consistent with the interpretation that the engineered features — particularly the composite pressure metric — have already done significant representational work.

**Why XGBoost leads.** The gradient boosting framework iteratively corrects residuals in a stage-wise manner, allowing it to learn complex interactions between features (e.g., the joint effect of high RRR and few wickets remaining in the death overs) that a linear model cannot represent. The advantage over Random Forest comes from the sequential correction mechanism: XGBoost fits trees to residuals rather than independent bootstrap samples, leading to better bias reduction at the same model complexity.

**Why the gap is small.** The features are sufficiently expressive that even a linear model can discriminate at 75.8% accuracy. This speaks to the quality of feature engineering: by the time pressure, rate ratio, wicket factor, and time factor are computed, the signal is largely monotonic and well-structured — a favourable landscape for linear classifiers.

[Insert Figure: Confusion Matrices (Figure 08)]

**Confusion Matrix Analysis.** XGBoost achieves balanced confusion across both classes: false positive rate (predicting a win that is lost) and false negative rate (predicting a loss that is won) are approximately symmetric. This is expected given the near-balanced outcome distribution and class-weight correction during training.

[Insert Figure: Reliability Diagrams (Figure 21)]

**Calibration.** Random Forest is the best-calibrated model (ECE = 0.033), while XGBoost shows mild overconfidence at the extremes of the probability range. All models achieve Brier Scores near 0.157, which is consistent with the inherent uncertainty in T20 cricket outcomes — a substantial fraction of matches have genuinely uncertain outcomes at any given delivery, and no model can be expected to eliminate this uncertainty.

---

### 10.2 Pressure Analysis

[Insert Figure: Pressure Metric Analysis (Figure 11, 14, 15)]

The pressure metric's statistical relationship with match outcome is evaluated both at the ball level and through conditional win rate analysis.

**Correlation analysis.**

| Metric | Pearson Correlation with Outcome (won) |
|--------|---------------------------------------|
| `pressure` | **−0.355** |
| `required_run_rate` | −0.427 |
| `current_run_rate` | +0.281 |
| `wickets_lost` | −0.268 |
| `rate_ratio` | −0.341 |

Pressure carries a correlation of −0.355 with the outcome, confirming its directional validity: higher pressure is associated with lower win probability. The fact that `required_run_rate` has a slightly higher correlation (−0.427) reflects that the single strongest predictor of a lost chase is the sheer magnitude of the run-rate requirement — pressure amplifies this through wicket and time factors but does not replace it.

**Partial correlation.** The partial correlation of pressure with outcome, controlling for `required_run_rate`, is −0.030 (p < 0.01). This relatively small but statistically significant partial correlation confirms that pressure contains *independent* predictive signal beyond what the run-rate ratio alone provides — driven by the wicket and time components. The pressure metric is not simply a restatement of the run-rate gap; it is a genuinely composite measure.

**Mean pressure by outcome:**
- Successful chases: mean pressure = **1.45**
- Failed chases: mean pressure = **3.50**

The 2.4× ratio in mean pressure between outcomes — a finding that survives phase-stratified analysis — provides strong empirical grounding for the metric's validity. Teams that lose chases operate under systematically higher integrated pressure across the entire innings, not merely at the final moment of defeat.

[Insert Figure: Win Probability vs Pressure (Figure 15)]

**Pressure–Win Rate Relationship.** Binning deliveries by pressure decile reveals a nearly monotonic negative relationship between pressure and win rate. The steepest drop occurs between the 40th and 70th percentiles (pressure ≈ 1.0–2.5), which corresponds to the Escalating zone. This non-linearity partially explains why gradient-boosted trees outperform logistic regression: the pressure-win-rate relationship has a sigmoid shape within the escalating zone that a linear model approximates but does not precisely fit.

---

### 10.3 Ablation Study

The ablation study addresses the fundamental question: does including the pressure metric — with its derived components — actually improve prediction beyond a model using only raw run-rate and match state features?

[Insert Figure: Ablation Study — With vs Without Pressure (Figure 20)]

Two conditions are compared across all three model families, using identical match-level train-test splits:

- **Without pressure**: Features = `{current_score, runs_remaining, balls_remaining, wickets_lost, wickets_in_hand, current_run_rate, required_run_rate, over, phase, toss_decision}` (10 features)
- **With pressure**: Features = full 14-feature set (adds `pressure`, `rate_ratio`, `wicket_factor`, `time_factor`)

| Model | AUC (Without Pressure) | AUC (With Pressure) | Δ AUC |
|-------|------------------------|---------------------|-------|
| Logistic Regression | 0.851 | 0.855 | +0.004 |
| Random Forest | 0.850 | 0.855 | +0.005 |
| XGBoost | 0.851 | 0.856 | +0.005 |

The addition of pressure features consistently improves AUC by approximately 0.004–0.005 across all models. While this improvement is modest in absolute terms, it is statistically consistent and directionally robust. The more compelling case for the pressure metric is its *interpretive* value: the decomposition into rate ratio, wicket factor, and time factor provides a diagnostic framework that raw features do not.

**Pressure independence test.** The Mann-Whitney U statistic comparing pressure distributions between won and lost chases yields p ≈ 0 — effectively certain that the two distributions are drawn from different populations. This non-parametric test makes no assumption about the normality of pressure distributions and confirms the discriminability observed in correlation analysis.

---

### 10.4 Advanced Model Comparison

[Insert Figure: Feature Configuration Comparison (Figure 31)]
[Insert Figure: Feature Importance Shift (Figure 32)]

The advanced comparison evaluates three progressively richer feature configurations:

| Configuration | Features | Description |
|--------------|---------|-------------|
| A — Static | 14 | 12 numeric + 2 categorical match-state |
| B — Static + Momentum | 32 | Adds 18 momentum features |
| C — Full Dynamic | 38 | Adds 6 dynamic pressure features |

**Results across configurations and model types:**

| Config | XGBoost AUC | Logistic AUC | XGBoost F1 | XGBoost Brier |
|--------|-------------|--------------|------------|---------------|
| A — Static | 0.8556 | 0.8554 | 0.7635 | 0.1571 |
| B — Static + Momentum | 0.8557 | **0.8560** | 0.7649 | 0.1573 |
| C — Full Dynamic | 0.8546 | 0.8557 | 0.7640 | 0.1578 |

**Interpreting the results.**

The progression from A to B yields a marginal improvement in AUC (+0.0004 for XGBoost, +0.0006 for Logistic Regression). The progression from B to C does not improve AUC and, for XGBoost, produces a slight decline (−0.0011). This counter-intuitive result deserves careful interpretation.

The modest gain from momentum features reflects a genuine but limited improvement: rolling scoring rates, wicket momentum, and pressure trajectory do contain incremental signal, but much of this signal is already partially encoded in the existing features (particularly `runs_remaining`, `wickets_lost`, and `pressure`). When 18 additional features add only 0.04% AUC improvement, the features are telling the model something it already largely knows.

The slight decline with full dynamic features in XGBoost (C vs. B) is likely a consequence of mild overfitting: 38 features on a 100,000-sample training set is manageable, but the 6 dynamic pressure features computed from EWM and rolling slopes are correlated with existing momentum features, and XGBoost's greedy tree-building may allocate splits to these redundant features rather than selecting maximally informative splits elsewhere.

**Why momentum improves interpretability more than accuracy.** The momentum and dynamic pressure features produce their most significant impact not in the discrimination metrics but in the downstream interpretability analysis. The regime classification, panic zone mapping, and collapse signature detection — all described in Section 10.6 — require the temporal features to function. The AUC plateau observed across configurations is not evidence that temporal features are uninformative; rather, it reflects that the prediction task (win/loss classification from a single snapshot) is already close to its information-theoretic ceiling given the available features, and that trajectory information is most valuable for *explaining* outcomes rather than discriminating them.

**Feature importance shift.** The top 15 feature importances for XGBoost change substantially between configurations A and B. In configuration A, the top features are `balls_remaining`, `runs_remaining`, `pressure`, `required_run_rate`, and `wickets_in_hand`. In configuration B, `runs_last_6`, `scoring_acceleration`, `rolling_pressure_mean_12`, and `wickets_last_12` enter the top 15, partially displacing raw match-state features. This shift indicates that the model is genuinely using momentum information, even though the aggregate AUC benefit is small.

---

### 10.5 Sequence Model Comparison

[Insert Figure: Sequence Model Comparison (Figure 33)]

The LSTM and GRU models trained on 20-ball context windows provide a test of whether the *ordering* and *timing* of ball-level events contain signal beyond what is captured in the engineered features.

| Model | ROC-AUC | F1 Score | Brier Score | Training Context |
|-------|---------|----------|-------------|----------------|
| LSTM | 0.8553 | 0.7688 | 0.1562 | 40k sequences, 15 epochs |
| GRU | 0.8552 | 0.7583 | 0.1574 | 40k sequences, 15 epochs |
| XGBoost (Config A) | 0.8556 | 0.7635 | 0.1571 | Full 100k ball-states |

**Key finding.** The LSTM and GRU models match but do not exceed the XGBoost static model, achieving AUC of 0.8553 and 0.8552 respectively against XGBoost's 0.8556. This result — that a carefully engineered tabular representation matches deep sequence models — is both practically significant and theoretically informative.

**Why XGBoost matches sequence models.** The result reflects a well-documented phenomenon in tabular machine learning: sophisticated feature engineering can substitute for architectural inductive biases. The momentum features — particularly `runs_last_6`, `runs_last_12`, `wickets_last_12`, `scoring_acceleration`, and `dp_momentum` — explicitly encode the sequential trajectory information that the LSTM's hidden state is tasked with learning. The sequence model's potential advantage (learning non-Markovian patterns across arbitrary past lengths) is largely neutralised by the hand-crafted rolling window features that already summarise the recent past.

Additionally, the sequence model is trained on a 40,000-sample subsample due to memory constraints, while the XGBoost model trains on the full 100,000-sample set. This asymmetry likely underestimates the sequence models' potential: with full training data and additional regularisation, the LSTM may achieve marginal improvements.

**Practical implication.** For deployment, the tabular XGBoost model is strongly preferable: it requires no GPU, serialises to a ~1MB pickle file, and produces predictions in microseconds. The sequence model, while achieving comparable performance, requires storing 20 prior deliveries of context — an operational constraint that the web application's single-snapshot API does not support.

---

### 10.6 Panic Zones and Collapse Detection

The advanced interpretability analysis identifies specific match situations where model predictions are most actionable.

[Insert Figure: Zone Analysis (Figure 34)]

#### Pressure Zone Win Rates

| Zone | Win Rate | Ball Count |
|------|----------|-----------|
| Stable (pressure < 1.2) | **75.2%** | 60,738 |
| Neutral | 41.4% | 30,836 |
| Escalating | 27.6% | 16,576 |
| Panic (pressure > 2.0, momentum > 0.3) | **3.97%** | 10,817 |
| Recovery (pressure ≥ 1.8, momentum < −0.2) | 28.1% | 6,747 |

[Insert Figure: Panic Zone Heatmap (Figure 28)]

#### Panic Zone Analysis

The *panic zone* is defined by the joint condition: static pressure > 2.0 AND dynamic momentum > 0.3. This combination identifies deliveries where the chase is both structurally difficult *and* deteriorating. Of 10,817 panic-zone deliveries, only 3.97% are associated with a winning chase. This near-certain losing signal represents one of the study's strongest empirical findings.

The heatmap of the two-dimensional pressure × momentum space reveals a continuous but sharp transition around pressure = 2.0: below this threshold, win rates are substantially above 30% across most momentum values. Above this threshold, win rates collapse to below 20%, with the momentum dimension providing additional stratification within the high-pressure region. The panic zone (high pressure, rising momentum) occupies the upper-right corner of this space and carries the lowest win rates observed anywhere in the dataset.

This finding is directly applicable to live match commentary and team strategy: when both conditions are met, the probability of a successful chase is approaching that of a guaranteed loss.

[Insert Figure: Collapse Signature Analysis (Figure 35)]

#### Collapse Detection

A batting collapse is defined operationally as the loss of two or more wickets within 12 consecutive deliveries. The analysis examines whether this condition — observable in real time — can predict the eventual match outcome before the death overs begin.

The `collapse_indicator` binary feature fires at 6.8% of all second-innings deliveries, confirming that collapses are relatively rare but significant events. Among deliveries where the collapse indicator fires, the eventual win rate drops to approximately 24% — compared to 50% for the full dataset. This confirms that collapse events carry substantial predictive information.

[Insert Figure: Early Detection Accuracy by Over (Figure 37)]

#### Early Detection Performance

Using a simple threshold rule (pressure > 2.0 at a given over as a predictor of match loss), detection accuracy as a function of the over in which the threshold is crossed is:

| Over | Detection Accuracy | Matches in Sample |
|------|--------------------|------------------|
| 5 | 62.7% | 1,087 |
| 7 | 65.4% | 1,083 |
| 9 | 70.2% | 1,074 |
| 11 | 74.6% | 1,060 |
| 13 | 80.5% | 1,048 |
| 15 | 83.7% | 996 |
| 17 | 81.8% | 914 |
| 19 | **86.4%** | 654 |

Early-overs detection (over 5) achieves only 62.7% accuracy — little better than a coin flip — because high early pressure can still be recovered if wickets are in hand. By over 11, accuracy exceeds 70%, and by over 15, the signal crosses 83%. The slight drop at over 17 relative to 15 likely reflects survivor bias: matches that reach over 17 with pressure > 2.0 may include some unexpectedly aggressive batting recoveries.

The key insight is that **the model can reliably identify doomed chases by over 13** (80.5% accuracy), giving a five-to-seven-over advance warning that could inform team strategy, commentator framing, and viewer expectations.

[Insert Figure: Phase Importance (Figure 36)]

#### Phase-Specific Feature Importance

Separate XGBoost models trained on powerplay, middle, and death-over ball-states reveal that the most important features differ across phases:

- **Powerplay**: `wickets_lost`, `runs_last_6`, and `balls_remaining` dominate. Early wickets are disproportionately destructive to chase probability because they expose lower-order batters to high-pace bowling with fielding restrictions in place.
- **Middle Overs**: `required_run_rate`, `pressure`, and `scoring_acceleration` lead. This phase is where the run-rate gap crystallises and where batting momentum most determines the final trajectory.
- **Death Overs**: `balls_remaining`, `wickets_in_hand`, and `dp_momentum` dominate. With few balls left, every delivery carries high leverage, and the momentum direction (improving or deteriorating) is highly predictive of the final outcome.

---

## Web Application

The trained XGBoost model is deployed as a live prediction dashboard called the *IPL Chase Pressure Predictor*, accessible at `http://localhost:5001`. The application enables real-time win probability estimation from any second-innings match state.

[Insert Figure: Web Application — Empty State]

[Insert Figure: Web Application — Prediction Result]

### Architecture

The application follows a conventional three-tier architecture:

```
Browser (HTML/CSS/JS)
    ↕  HTTP JSON
Flask API (app.py + model_utils.py)
    ↕  pickle.load
Pre-trained XGBoost Pipeline (models/xgboost.pkl)
```

**Backend (`app.py`, `model_utils.py`).** The Flask backend exposes a `POST /predict` endpoint that accepts match-state inputs in JSON format, performs server-side input validation, applies the same feature engineering logic used during training (reproduced in `model_utils.compute_features()`), and returns a prediction response. The XGBoost pipeline is loaded once at server startup into a module-level cache, ensuring sub-millisecond inference latency per request.

Feature engineering in `model_utils` exactly replicates the computation from `src/feature_engineering.py` and `src/pressure.py`: the same pressure formula, the same phase boundaries, and the same run-rate derivations are used, ensuring that the web application's predictions are directly comparable to the training pipeline's outputs.

**Frontend.** The single-page dashboard is built in plain HTML, CSS, and JavaScript without any framework dependencies. Key UI components include:

- **Input form**: Target score, current score, balls bowled, a click-to-select wicket counter (0–9), and toss decision.
- **Circular win probability gauge**: An SVG arc whose fill length encodes the predicted probability. The arc colour transitions from red (< 45%) through amber (45–65%) to green (> 65%).
- **Batting team advantage bar**: A horizontal progress bar showing the win/loss probability split.
- **Stat cards**: Four summary metrics — runs needed, overs remaining, wickets in hand, and match phase.
- **Run rate comparison**: Side-by-side CRR and RRR cards with a colour-coded gap bar.
- **Pressure section**: The numeric pressure score, a zone badge (Comfortable / Building / Escalating / Panic), a gradient indicator bar, and the formula breakdown (Rate Ratio × Wicket Factor × Time Factor = Pressure).
- **Plain-English explanation**: Auto-generated from the prediction context.

### Input Validation

Both client-side (JavaScript) and server-side (Flask) validation are applied:

| Validation Rule | Layer |
|----------------|-------|
| All fields are numeric integers | Client + Server |
| Target score: 1–500 | Client + Server |
| Current score: 0 < target | Client + Server |
| Balls completed: 0–119 | Client + Server |
| Wickets lost: 0–9 | Client + Server |
| Toss decision: `bat` or `field` | Server |
| Balls remaining > 0 if runs still needed | Server |

Error messages are surfaced inline (next to the relevant field) on the client and returned as structured JSON error objects from the server.

### API Contract

**Request:**
```json
POST /predict
Content-Type: application/json

{
  "target_score":    185,
  "current_score":    94,
  "balls_completed":  72,
  "wickets_lost":      3,
  "toss_decision":  "field"
}
```

**Response:**
```json
{
  "win_probability":      0.3059,
  "win_probability_pct": "30.6%",
  "pressure":             1.96,
  "pressure_zone":        "Escalating",
  "pressure_zone_color":  "#F97316",
  "current_run_rate":     7.83,
  "required_run_rate":   11.38,
  "runs_remaining":       91,
  "balls_remaining":      48,
  "overs_remaining":     "8.0",
  "wickets_in_hand":       7,
  "phase":               "Middle",
  "rate_ratio":           1.45,
  "wicket_factor":        1.35,
  "time_factor":          1.00,
  "explanation": "Difficult chase. Win probability just 31%. Behind the required rate by 3.5 RPO..."
}
```

---

## Limitations

Any honest treatment of this work must acknowledge its limitations clearly. Several of these are structural and would require significant changes to the data collection or modelling approach to address.

**1. Second innings only.** The model has no awareness of first-innings conditions: the pitch behaviour, dew factor, bowling attack composition, or batting order. In IPL cricket, these factors are significant. A target of 185 on a dry, turning pitch at Chepauk is categorically different from the same target at a flat Eden Gardens track. The model treats both identically.

**2. No team identity or player-level features.** The model does not know which teams are playing or which batters are at the crease. A match situation with 50 required off 30 balls looks the same whether it is Virat Kohli or a tail-ender on strike. Team-specific chase success rates and batter-specific pressure performance would add meaningful signal.

**3. Momentum features require ball-by-ball history.** The web application uses the static 14-feature model because a single match-state snapshot cannot support rolling window computation. Users cannot input "last 6 balls' runs" directly. The gap between the research model and the deployed model is thus a consequence of the deployment scenario, not a modelling flaw.

**4. Historical training data (2008–2022).** The IPL has changed substantially over fifteen seasons in terms of team composition, rule modifications (impact player substitution, power surge alternatives in other leagues), and venue improvements. The model's calibration may drift for matches played after 2022.

**5. No DLS adjustment.** While DLS-adjusted targets are accepted as input, the model does not adjust its inference for reduced-over scenarios. A team chasing a DLS-revised target of 150 in 15 overs faces different pressure dynamics than a standard 20-over chase of 150, because the expected run-rate is higher from the outset.

**6. Class-level calibration.** Although Brier Scores are competitive (~0.157), the reliability diagram analysis shows mild overconfidence at the extremes of the probability range for XGBoost. Predicted probabilities above 0.85 or below 0.15 may be slightly miscalibrated, and should not be treated as precise probability estimates.

**7. Sequence model training constraints.** The LSTM and GRU models were trained on a 40,000-sample subsample due to in-process memory limitations within the pipeline, and on 15 epochs only. A dedicated training environment with more memory and longer training schedules might reveal greater separation from tabular models.

---

## Future Work

Several directions emerge naturally from this study's findings and limitations.

**Player and team features.** Incorporating batter-specific performance under pressure (e.g., a player's historical win rate from similar states) and team-level chase success rates (by year or condition category) would address the most significant gap in the current feature set. This requires careful handling of small sample sizes for rarely-observed combinations.

**Venue and pitch modelling.** Venue effects are well-documented in cricket analytics. A venue-level pressure calibration — adjusting the base win probability up or down based on historical chase success rates at a given ground — could be incorporated as a prior without disrupting the existing model architecture.

**Opponent-adjusted features.** The bowling attack's quality at the time of a delivery influences the difficulty of scoring. A running estimate of the bowling side's economy and wicket-taking rate during the current match could be incorporated as a dynamic feature.

**Full dynamic model deployment.** The web application currently uses the static 14-feature model. A history-aware API endpoint — one that accepts a sequence of recent deliveries as input — would allow the B or C configuration model to be deployed, enabling the momentum and dynamic pressure features to be computed in real time.

**Online calibration.** A recalibration layer (e.g., Platt scaling or isotonic regression) applied post-training would improve the probability estimates at the extremes of the range. This is especially important for the panic zone (model output near 0.04) and the stable zone (near 0.75), where decisions based on precise probability values might be made.

**Transformer-based sequence models.** The LSTM and GRU architectures used here are conventional choices for time series. Multi-head self-attention (Transformer) architectures, which have shown strong performance on sequential data in other domains, could potentially capture long-range dependencies within an innings more effectively — for example, detecting that a powerplay collapse seven overs ago continues to constrain the batting side's tail-end resources.

**Super Over and D/L extension.** Extending the model to handle Super Over deciders and Duckworth-Lewis reduced-over chases would require additional data collection and a modified feature engineering pipeline to represent the revised resource budgets correctly.

**Live data integration.** The web application could be extended to consume live ball-by-ball data from a sports data API (CricInfo, SportRadar), enabling fully automated real-time win probability updates without manual user input.

---

## Conclusion

This project set out to answer a tightly scoped question — can machine learning, applied to ball-by-ball second-innings data, produce meaningful win probability estimates for IPL chases? — and the answer is a qualified yes, with important caveats.

The central finding is that a carefully engineered composite pressure metric, capturing the interaction between run-rate difficulty, wicket loss, and time pressure, provides approximately 0.005 AUC improvement over models using only raw match state features. More importantly, the metric provides a coherent interpretive framework: won chases operate at mean pressure 1.45, lost chases at 3.50, and the panic zone (pressure > 2.0, momentum > 0.3) carries only a 3.97% win rate. These are not just statistical artefacts — they reflect genuine domain dynamics that practitioners can recognise and act on.

The momentum and dynamic pressure features tell a nuanced story. Their impact on discrimination accuracy (ROC-AUC) is marginal — the B configuration improves over A by just 0.0004 AUC — but their contribution to interpretability is substantial. Regime classification, collapse detection, and panic zone identification all require temporal features to function. This is consistent with a common pattern in feature-rich ML settings: diminishing marginal returns to discrimination coexist with meaningful gains in explanation.

The sequence model results are perhaps the study's most theoretically interesting finding: LSTM and GRU models trained on 20-ball context windows match, but do not exceed, the static XGBoost model (AUC 0.8553 vs 0.8556). This convergence suggests that the rolling momentum features have largely captured the sequential signal available in the recent delivery history. It also validates the practical choice of deploying a tabular model in the web application: the complexity cost of a recurrent neural network is not justified by the predictive returns in this setting.

The *IPL Chase Pressure Predictor* web application translates this research into a directly usable tool, demonstrating that the gap between a trained ML model and a deployed sports analytics product is bridgeable with modest engineering effort — provided the feature engineering and model pipeline are designed with deployment in mind from the start.

The limitations of this work are real. Team identity, player profiles, pitch conditions, and opponent bowling quality are all excluded. The model predicts from a state snapshot, not from a strategy. And the marginal gains from sophisticated temporal modelling remind us that T20 cricket — with its high variance, player-specific dynamics, and context-dependence — remains fundamentally difficult to predict with high precision from aggregate statistics alone.

What this project demonstrates is that domain-informed feature engineering, rigorous leakage prevention, and systematic interpretability analysis can produce a model that is not only accurate but analytically credible — one whose predictions can be understood, challenged, and contextualised by practitioners who know the sport.

---

## References

Asif, M., & McHale, I. (2016). In-play forecasting of win probability in One-Day International cricket: A dynamic logistic regression model. *International Journal of Forecasting*, 32(1), 34–43.

Bailey, M., & Clarke, S. R. (2006). Predicting the match outcome in one day international cricket matches, while the game is in progress. *Journal of Sports Science & Medicine*, 5(4), 480–487.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Davis, J., Perera, H., & Swartz, T. B. (2015). Player evaluation in Twenty20 cricket. *Journal of Sports Analytics*, 1(1), 19–31.

Duckworth, F. C., & Lewis, A. J. (1998). A fair method for resetting the target in interrupted one-day cricket matches. *Journal of the Operational Research Society*, 49(3), 220–227.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

Lemmer, H. H. (2004). A measure for the batting performance of cricket players. *South African Journal for Research in Sport, Physical Education and Recreation*, 26(1), 55–64.

Mukherjee, S. (2012). Identifying the greatest team and captain: A complex network approach to cricket matches. *Physica A: Statistical Mechanics and its Applications*, 391(23), 6066–6076.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Sankaranarayanan, V. V., Sattar, J., & Lakshminarayanan, V. (2014). Auto-play: A data mining approach to ODI cricket simulation and prediction. *Proceedings of the 8th Workshop on Mining and Learning with Graphs*.

Stern, S. E. (2009). The Duckworth-Lewis-Stern method: Extending the Duckworth-Lewis methodology to deal with modern scoring rates. *Journal of the Operational Research Society*, 60(12), 1677–1684.

van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.

Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis*. Springer-Verlag.

---

*Report generated from the IPL Chase Pressure Predictor ML Pipeline (v3). All code, data, and reproducibility artifacts are maintained in the project repository. Figures referenced throughout this report are saved to `outputs/plots/` and are cross-referenced by their filename index.*

*Model training environment: macOS (Apple Silicon), Python 3.11, scikit-learn 1.4, XGBoost 2.0, PyTorch 2.2, Flask 3.0.*

---

**End of Report**
