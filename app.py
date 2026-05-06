"""
=============================================================
  CAAP IoMT IDS — Flask AI Server v4
  POST /predict  →  label, confidence, TR_score, cluster,
                    CAS, action, dimensions (TR/CC/TS/AE/TC)
  GET  /health   →  status, classes, feature list
  GET  /features →  full feature + medical field schema

  Input uses EXACT CIC IoMT 2024 network feature column names.
  Medical metadata (cc_score, shift, time_sensitivity) is
  passed alongside network features for CAS scoring.

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
  Run    : python app.py
  Port   : 5001
=============================================================
"""

import os, sys
import joblib
import numpy as np
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cas_engine import score_alert

# Optional CORS — install flask-cors if using with a separate frontend
try:
    from flask_cors import CORS
    _cors_available = True
except ImportError:
    _cors_available = False

app = Flask(__name__)
if _cors_available:
    CORS(app)

MODEL_DIR = "models"

# ── LOAD ALL MODELS ON STARTUP ─────────────────────────────────────────────────
print("\nLoading CAAP models...")
rf           = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
iso          = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
km           = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

CLUSTER_LABELS = {0: "active", 1: "routine", 2: "idle"}

print(f"✓ Models loaded | {len(le.classes_)} classes | {len(FEATURE_COLS)} features")
print(f"  Classes      : {list(le.classes_)}")
print(f"  Feature cols : {FEATURE_COLS}")
print(f"  Listening on : http://0.0.0.0:5001\n")


# ── POST /predict ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Send all 45 CIC IoMT 2024 network feature columns as JSON keys,
    plus optional medical metadata for CAS scoring.

    Example request body:
    {
      "Header_Length": -0.094,
      "Protocol Type": 1.42,
      "Duration": -0.088,
      "Rate": -0.38,
      "Srate": -0.38,
      "Drate": 0,
      "fin_flag_number": -0.15,
      "syn_flag_number": -0.47,
      ... (all 45 network features) ...
      "IAT": 4.73,
      "Number": -4.73,
      "Magnitue": 7.65,
      "Radius": 12.28,
      "Covariance": 13.52,
      "Variance": 3.47,
      "Weight": -4.74,

      // Optional — used for CAS scoring:
      "cc_score": 10,
      "shift": "night",
      "time_sensitivity": 5
    }
    """
    data = request.get_json(force=True)

    # Build feature vector using the trained model's feature order
    vec        = np.array([[float(data.get(col, 0.0)) for col in FEATURE_COLS]])
    vec_scaled = scaler.transform(vec)

    # ── ML inference ──
    pred_idx   = rf.predict(vec_scaled)[0]
    proba      = rf.predict_proba(vec_scaled)[0]
    label      = le.inverse_transform([pred_idx])[0]
    confidence = round(float(proba.max()), 4)
    iso_score  = round(float(iso.decision_function(vec_scaled)[0]), 4)
    is_anomaly = bool(iso.predict(vec_scaled)[0] == -1)
    cluster    = CLUSTER_LABELS.get(int(km.predict(vec_scaled)[0]), "unknown")

    # ── CAS scoring ──
    cas_r = score_alert(
        predicted_label = label,
        confidence      = confidence,
        iso_score       = iso_score,
        is_anomaly      = is_anomaly,
        cc_score_raw    = data.get("cc_score",        5),
        shift           = data.get("shift",           "day"),
        ts_col_val      = data.get("time_sensitivity", 3),
    )

    return jsonify({
        "label"      : label,
        "confidence" : confidence,
        "TR_score"   : iso_score,
        "cluster"    : cluster,
        "is_anomaly" : is_anomaly,
        "CAS"        : cas_r["CAS"],
        "action"     : cas_r["action"],
        "dimensions" : {
            "TR": cas_r["TR"],
            "CC": cas_r["CC"],
            "TS": cas_r["TS"],
            "AE": cas_r["AE"],
            "TC": cas_r["TC"],
        },
    })


# ── GET /health ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"      : "ok",
        "classes"     : list(le.classes_),
        "n_features"  : len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "cas_weights" : {
            "TR": 0.25, "CC": 0.30, "TS": 0.25, "AE": 0.10, "TC": 0.10
        },
        "action_thresholds": {
            "Immediate":   "CAS >= 8.0",
            "Investigate": "CAS >= 5.0",
            "Monitor":     "CAS < 5.0",
        },
    })


# ── GET /features ──────────────────────────────────────────────────────────────
@app.route("/features", methods=["GET"])
def features():
    """Returns the full input schema expected by /predict."""
    return jsonify({
        "network_features": FEATURE_COLS,
        "medical_fields": [
            {
                "name"   : "cc_score",
                "type"   : "int",
                "range"  : "1–10",
                "desc"   : "Device clinical criticality score from dataset "
                           "(ICU Ventilator=10, Admin PC=1)",
            },
            {
                "name"   : "shift",
                "type"   : "string",
                "values" : ["day", "evening", "night"],
                "desc"   : "Current hospital shift (affects TC dimension)",
            },
            {
                "name"   : "time_sensitivity",
                "type"   : "int",
                "range"  : "1–5",
                "desc"   : "Time sensitivity score from dataset",
            },
        ],
        "response_schema": {
            "label"      : "string — predicted attack class",
            "confidence" : "float — RF predict_proba max (0–1)",
            "TR_score"   : "float — Isolation Forest decision score",
            "cluster"    : "string — active / routine / idle",
            "is_anomaly" : "bool — IF anomaly flag",
            "CAS"        : "float — Clinical Alert Score (0–10)",
            "action"     : "string — Immediate / Investigate / Monitor",
            "dimensions" : {
                "TR": "float 1–5 — Threat Risk",
                "CC": "float 1–5 — Clinical Criticality",
                "TS": "float 1–5 — Time Sensitivity",
                "AE": "float 0–5 — Active Exploitation",
                "TC": "float 1–5 — Temporal Context",
            },
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
