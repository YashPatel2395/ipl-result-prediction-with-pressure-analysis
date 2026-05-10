# IPL Chase Pressure Predictor — Web Application

A sports analytics dashboard that turns the trained IPL pressure-aware chase model into
a live interactive predictor. Enter the current second-innings match state and instantly
get win probability, pressure score, required run rate, and a plain-English explanation.

---

## Quick Start

```bash
# 1. Install dependencies (from the project root)
pip install flask

# 2. Run the app
cd web_app
python app.py

# Or with hot-reload during development:
FLASK_ENV=development python app.py
```

Open **http://localhost:5000** in your browser.

---

## What the Model Predicts

The model predicts the probability that the **chasing team wins** at any point during the
second innings of a T20 IPL match. It is **not** a pre-match predictor.

### Inputs

| Field | Description | Range |
|-------|-------------|-------|
| Target Score | Runs the chasing team must exceed | 1–500 |
| Current Score | Runs scored so far in the chase | 0–499 |
| Balls Bowled | Total legal deliveries completed (e.g., 3.2 overs = 20 balls) | 0–119 |
| Wickets Lost | How many wickets have fallen | 0–9 |
| Toss Decision | What the toss-winning captain chose | bat / field |

### Outputs

| Field | Description |
|-------|-------------|
| `win_probability` | Chasing team win probability (0–1) |
| `pressure` | Composite pressure score (0.05–15) |
| `pressure_zone` | Comfortable / Building / Escalating / Panic |
| `current_run_rate` | Runs per over currently being scored |
| `required_run_rate` | Runs per over needed to win from here |
| `runs_remaining` | Runs still needed |
| `balls_remaining` | Deliveries left |
| `overs_remaining` | Overs.balls remaining |
| `wickets_in_hand` | Wickets still in hand |
| `phase` | Match phase (Powerplay / Middle / Death) |
| `explanation` | Plain-English narrative summary |

---

## Pressure Formula

```
pressure = rate_ratio × wicket_factor × time_factor

rate_ratio    = RRR / max(CRR, 0.5)
wicket_factor = 1.0 + (wickets_lost ^ 1.5) / 15.0
time_factor   = 1.0 + max(0, (30 - balls_remaining) / 120.0)
pressure      = clip(pressure, 0.05, 15.0)
```

### Pressure Zones

| Zone | Pressure | Win Rate (empirical) |
|------|----------|---------------------|
| Comfortable | < 1.0 | ~75% |
| Building | 1.0–1.5 | ~55% |
| Escalating | 1.5–2.5 | ~28% |
| Panic | > 2.5 | ~4% |

---

## API

### `POST /predict`

```json
// Request
{
  "target_score":    185,
  "current_score":   94,
  "balls_completed": 72,
  "wickets_lost":    3,
  "toss_decision":   "field"
}

// Response
{
  "win_probability": 0.6211,
  "pressure":        1.18,
  "pressure_zone":   "Building",
  "current_run_rate": 7.83,
  "required_run_rate": 7.67,
  "runs_remaining":  91,
  "balls_remaining": 48,
  "overs_remaining": "8.0",
  "wickets_in_hand": 7,
  "phase":           "Middle",
  "explanation":     "Slight edge to the batting side ..."
}
```

### `GET /health`

Returns `{"status": "ok", "model": "xgboost.pkl"}`.

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (gradient-boosted trees) |
| Features | 14 (12 numeric + 2 categorical) |
| Training data | 1,090 IPL matches · 125,714 ball-states |
| Test ROC-AUC | 0.856 |
| Test F1 | 0.765 |
| Split | Match-level stratified (80/20) |

### Features Used

Numeric: `current_score`, `runs_remaining`, `balls_remaining`, `wickets_lost`,
`wickets_in_hand`, `current_run_rate`, `required_run_rate`, `pressure`,
`rate_ratio`, `wicket_factor`, `time_factor`, `over`

Categorical (one-hot encoded): `phase`, `toss_decision`

---

## Limitations

1. **Second innings only** — the model sees only the chasing team's perspective. First-innings
   data (pitch, dew, batting order) is not used and cannot predict pre-match outcomes.

2. **Momentum features not available** — the full dynamic model (Steps 21–23 of the research
   pipeline) requires ball-by-ball history. The web app uses the static 14-feature model which
   achieves 0.856 AUC from a single match-state snapshot.

3. **Historical IPL data** — trained on IPL seasons 2008–2022. Venue or team dynamics may
   have shifted since then.

4. **Toss decision** — captures toss-winner's strategy, not necessarily the chasing team's.
   Defaults to "field" (most common IPL choice when winning the toss).

5. **No D/L support** — Duckworth-Lewis-Stern adjusted targets are accepted as `target_score`
   but revised over counts must be entered manually.

---

## Project Structure

```
web_app/
├── app.py           Flask backend — routes + validation
├── model_utils.py   Feature engineering + model loading
├── templates/
│   └── index.html   Single-page dashboard UI
├── static/
│   ├── style.css    Dark sports-analytics CSS
│   └── script.js    Form handling + animated results
└── README.md        This file

models/              (project root — pre-trained pickles)
├── xgboost.pkl      ← used by the web app
└── ...
```

---

## Example Input / Output

**Scenario**: CSK chasing 190, currently 87/3 after 12 overs.

```
Target:          190
Current score:    87
Balls completed:  72   (12.0 overs)
Wickets lost:      3
Toss decision:   field
```

**Prediction**:

```
Win probability:  54.2%
Pressure score:   1.62  (Escalating)
CRR:              7.25
RRR:              8.67
Explanation:      "Evenly poised match (54%). Behind required rate by 1.4 RPO. 
                   7 wickets in hand keeps pressure manageable.
                   5 overs left in the middle phase — building partnerships is key."
```
