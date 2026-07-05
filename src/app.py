"""
CAAP AI Server — Phase 6
=========================
Flask server exposing POST /predict.

Loads the three trained models (Random Forest, Isolation Forest, K-Means) plus
the fitted StandardScaler, runs a single flow through all three, computes the
5-dimension Clinical Alert Score (CAS), maps it to an action, and returns a
SHAP-based explanation.

Run:
    python src/app.py

Test:
    curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d @sample_row.json
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")  # matches your actual "models/" folder

# Behavioural subset used specifically by Isolation Forest (Phase 4, 9.1)
# NOTE: these must be a subset of whatever feature_cols.pkl actually contains
# — check against the startup printout and adjust names if yours differ.
IF_FEATURES = ["flow_bytes_s", "flow_packets_s", "fwd_pkt_len_mean", "active_mean", "idle_mean"]

# Flow-context subset used by K-Means (Phase 4, 9.3)
KMEANS_FEATURES = ["flow_bytes_s", "flow_packets_s", "flow_iat_mean", "fwd_pkt_len_mean"]

# K-Means cluster label mapping.
# NOTE: empirical testing (Phase 5) showed k=2 gives better cluster separation
# than the originally planned k=3 — "routine" was dropped. If you re-trained
# with k=3, add "routine" back in here and in kmeans.pkl.
CLUSTER_LABELS = {0: "idle", 1: "active"}  # confirm actual index<->label mapping from training

# Clinical Criticality (CC) lookup — rule-based, no ML (Phase 2, Day 4)
CC_LOOKUP = {
    "ICU Ventilator": 5,
    "Infusion Pump": 4,
    "Radiology": 3,
    "Nurse WS": 2,
    "Admin PC": 1,
}
DEFAULT_CC = 1  # unknown device types treated as lowest criticality

# CAS formula weights (Phase 3)
CAS_WEIGHTS = {"TR": 0.25, "CC": 0.30, "TS": 0.25, "AE": 0.10, "TC": 0.10}

# Action thresholds (Phase 6, Day 3)
ACTION_THRESHOLDS = [(8, "Immediate"), (5, "Investigate")]
DEFAULT_ACTION = "Monitor"

# --------------------------------------------------------------------------
# App + model loading
# --------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # tighten origins for production (Node.js on :5000)

print("[CAAP] Loading models...")
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
FEATURE_COLUMNS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
IF_THRESHOLD = joblib.load(os.path.join(MODEL_DIR, "if_threshold.pkl"))
print(f"[CAAP] Models loaded. {len(FEATURE_COLUMNS)} features, IF threshold={IF_THRESHOLD}")
print(f"[CAAP] Feature columns: {FEATURE_COLUMNS}")

# Diagnostic: how many features does each model actually expect?
_if_n = getattr(iso_forest, "n_features_in_", None)
_km_n = getattr(kmeans, "n_features_in_", None)
_rf_n = getattr(rf_model, "n_features_in_", None)
print(f"[CAAP] n_features_in_  -> RF: {_rf_n}  IsolationForest: {_if_n}  KMeans: {_km_n}  (total available: {len(FEATURE_COLUMNS)})")

# If IF/KMeans expect the FULL feature set, use all columns instead of a hand-picked
# subset — this removes the guesswork entirely when train.py fit them on everything.
if _if_n == len(FEATURE_COLUMNS):
    IF_FEATURES = FEATURE_COLUMNS
    print("[CAAP] IsolationForest was trained on the FULL feature set — using all columns.")
if _km_n == len(FEATURE_COLUMNS):
    KMEANS_FEATURES = FEATURE_COLUMNS
    print("[CAAP] KMeans was trained on the FULL feature set — using all columns.")

# Sanity check: IF_FEATURES / KMEANS_FEATURES must exist in the real feature list
_missing_if = [f for f in IF_FEATURES if f not in FEATURE_COLUMNS]
_missing_km = [f for f in KMEANS_FEATURES if f not in FEATURE_COLUMNS]
if _missing_if:
    print(f"[CAAP][WARNING] IF_FEATURES not found in feature_cols.pkl: {_missing_if}")
    print(f"[CAAP][WARNING] -> IsolationForest expects {_if_n} features but IF_FEATURES has {len(IF_FEATURES)} names that don't match. Edit IF_FEATURES in app.py to match real column names above.")
if _missing_km:
    print(f"[CAAP][WARNING] KMEANS_FEATURES not found in feature_cols.pkl: {_missing_km}")
    print(f"[CAAP][WARNING] -> KMeans expects {_km_n} features but KMEANS_FEATURES has {len(KMEANS_FEATURES)} names that don't match. Edit KMEANS_FEATURES in app.py to match real column names above.")

# SHAP explainer built once at startup (tree explainer is fast for RF)
shap_explainer = shap.TreeExplainer(rf_model)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def to_feature_frame(payload: dict) -> pd.DataFrame:
    """Pull the 44 model features out of the incoming JSON, in training order."""
    row = {col: payload.get(col, 0.0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def rf_to_tr_score(confidence: float) -> float:
    """Map Random Forest confidence (0-1) -> Threat Risk score (1-5)."""
    return round(1 + confidence * 4, 2)


def if_to_ts_score(anomaly_score: float, hour_of_day: int) -> float:
    """
    Map Isolation Forest decision_function() output -> Time Sensitivity (1-5).
    score < IF_THRESHOLD (loaded from if_threshold.pkl) -> anomalous -> TS = 5
    otherwise -> routine traffic -> TS driven by time-of-day
                 (off-hours = higher sensitivity, since fewer staff on duty)
    """
    if anomaly_score < IF_THRESHOLD:
        return 5.0
    if hour_of_day < 6 or hour_of_day >= 22:  # night shift, low staffing
        return 3.5
    return 1.5


def lookup_cc(device_type: str) -> float:
    return CC_LOOKUP.get(device_type, DEFAULT_CC)


def lookup_ae(cve_known_exploited: bool) -> float:
    """Active Exploitation — rule based on CVE/CVSS lookup (stubbed input flag)."""
    return 5.0 if cve_known_exploited else 1.0


def lookup_tc(hour_of_day: int) -> float:
    """Temporal Context — shift-based rule (night shift = higher weight)."""
    return 4.0 if (hour_of_day < 6 or hour_of_day >= 22) else 2.0


def compute_cas(tr, cc, ts, ae, tc) -> float:
    cas = (
        CAS_WEIGHTS["TR"] * tr
        + CAS_WEIGHTS["CC"] * cc
        + CAS_WEIGHTS["TS"] * ts
        + CAS_WEIGHTS["AE"] * ae
        + CAS_WEIGHTS["TC"] * tc
    )
    return round(cas, 2)


def cas_to_action(cas: float) -> str:
    for threshold, action in ACTION_THRESHOLDS:
        if cas >= threshold:
            return action
    return DEFAULT_ACTION


def top_shap_features(scaled_row: np.ndarray, n: int = 3) -> str:
    """
    Return a short text explanation naming the top-N contributing features.

    Handles both SHAP output conventions:
      - older shap: list of (n_samples, n_features) arrays, one per class
      - newer shap: single array shaped (n_samples, n_features, n_classes)
    """
    shap_values = shap_explainer.shap_values(scaled_row)
    predicted_class_idx = int(np.argmax(rf_model.predict_proba(scaled_row)))

    if isinstance(shap_values, list):
        # Older convention: list of per-class arrays, each (n_samples, n_features)
        class_shap = shap_values[predicted_class_idx][0]
    else:
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 3:
            # Newer convention: (n_samples, n_features, n_classes)
            class_shap = shap_arr[0, :, predicted_class_idx]
        else:
            # Binary classification or already-flat: (n_samples, n_features)
            class_shap = shap_arr[0]

    top_idx = np.argsort(np.abs(class_shap))[::-1][:n]
    parts = [f"{FEATURE_COLUMNS[i]} ({class_shap[i]:+.2f})" for i in top_idx]
    return "Top contributing features: " + ", ".join(parts)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Missing or invalid JSON body"}), 400

    try:
        # --- 1. Build feature frame + scale -------------------------------
        X = to_feature_frame(payload)
        X_scaled = scaler.transform(X)

        # --- 2. Random Forest — attack classification (TR) ----------------
        proba = rf_model.predict_proba(X_scaled)[0]
        pred_idx = int(np.argmax(proba))
        encoded_label = rf_model.classes_[pred_idx]
        label = label_encoder.inverse_transform([encoded_label])[0]
        confidence = float(proba[pred_idx])
        tr_score = rf_to_tr_score(confidence)

        # --- 3. Isolation Forest — anomaly detection (TS) -----------------
        X_if = pd.DataFrame([{c: payload.get(c, 0.0) for c in IF_FEATURES}], columns=IF_FEATURES)
        X_if_scaled = scaler.transform(X)[:, [FEATURE_COLUMNS.index(c) for c in IF_FEATURES]]
        anomaly_score = float(iso_forest.decision_function(X_if_scaled)[0])
        hour_of_day = int(payload.get("hour_of_day", 12))
        ts_score = if_to_ts_score(anomaly_score, hour_of_day)

        # --- 4. K-Means — traffic context ----------------------------------
        X_km = scaler.transform(X)[:, [FEATURE_COLUMNS.index(c) for c in KMEANS_FEATURES]]
        cluster_idx = int(kmeans.predict(X_km)[0])
        cluster_label = CLUSTER_LABELS.get(cluster_idx, "unknown")

        # --- 5. Rule-based dimensions (CC, AE, TC) -------------------------
        cc_score = lookup_cc(payload.get("device_type", ""))
        ae_score = lookup_ae(bool(payload.get("cve_known_exploited", False)))
        tc_score = lookup_tc(hour_of_day)

        # --- 6. CAS + action -------------------------------------------------
        cas = compute_cas(tr_score, cc_score, ts_score, ae_score, tc_score)
        action = cas_to_action(cas)

        # --- 7. SHAP explanation --------------------------------------------
        explanation = top_shap_features(X_scaled)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4),
            "TR_score": tr_score,
            "TS_score": ts_score,
            "cluster": cluster_label,
            "CC_score": cc_score,
            "AE_score": ae_score,
            "TC_score": tc_score,
            "CAS": cas,
            "action": action,
            "explanation": explanation,
        })

    except Exception as exc:  # keep the API resilient; log for debugging
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Node.js backend runs on :5000 — Flask AI layer on :5001 (Phase 6, Day 5)
    app.run(host="0.0.0.0", port=5001, debug=True)
