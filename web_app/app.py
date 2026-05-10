"""
app.py
======
Flask backend for IPL Chase Pressure Predictor.

Run:
    cd web_app
    FLASK_ENV=development python app.py

Or from project root:
    python web_app/app.py
"""

import os
import sys

# Ensure the project root is on the path so model_utils can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify

import model_utils

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False


# ── Warm-up: load model at startup so the first request is instant ────────────
try:
    model_utils.load_model()
    print(f"[app] Model loaded: {model_utils.MODEL_PATH}", flush=True)
except Exception as e:
    print(f"[app] WARNING — could not pre-load model: {e}", flush=True)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with the current chase state and return a prediction.

    Expected body:
        {
            "target_score":    int  (1–500),
            "current_score":   int  (0–499),
            "balls_completed": int  (0–119),
            "wickets_lost":    int  (0–9),
            "toss_decision":   str  ("bat" | "field")
        }

    Returns JSON with win probability, pressure, run rates, and explanation.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # ── Parse ────────────────────────────────────────────────────────────
    try:
        target_score    = int(data["target_score"])
        current_score   = int(data["current_score"])
        balls_completed = int(data["balls_completed"])
        wickets_lost    = int(data["wickets_lost"])
        toss_decision   = str(data.get("toss_decision", "field")).strip().lower()
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({
            "error": f"Invalid or missing field: {exc}. "
                     "Required: target_score, current_score, balls_completed, wickets_lost."
        }), 400

    # ── Validate ─────────────────────────────────────────────────────────
    errors = {}

    if not (1 <= target_score <= 500):
        errors["target_score"] = "Target score must be between 1 and 500."

    if not (0 <= current_score <= 499):
        errors["current_score"] = "Current score must be between 0 and 499."

    if current_score >= target_score:
        errors["current_score"] = (
            "Current score must be less than the target — the chase is still in progress."
        )

    if not (0 <= balls_completed <= 119):
        errors["balls_completed"] = (
            "Balls completed must be between 0 (start) and 119 (last ball)."
        )

    if not (0 <= wickets_lost <= 9):
        errors["wickets_lost"] = "Wickets lost must be between 0 and 9."

    if toss_decision not in ("bat", "field"):
        errors["toss_decision"] = "Toss decision must be 'bat' or 'field'."

    # Cross-field sanity checks
    runs_needed  = target_score - current_score
    balls_left   = 120 - balls_completed
    if balls_left == 0 and runs_needed > 0:
        errors["balls_completed"] = "No balls remaining but target not reached — impossible state."

    if errors:
        return jsonify({"error": "Validation failed.", "details": errors}), 422

    # ── Predict ──────────────────────────────────────────────────────────
    try:
        result = model_utils.predict(
            target_score    = target_score,
            current_score   = current_score,
            balls_completed = balls_completed,
            wickets_lost    = wickets_lost,
            toss_decision   = toss_decision,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": os.path.basename(model_utils.MODEL_PATH)})


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    print(f"[app] Starting IPL Chase Pressure Predictor on http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=debug)
