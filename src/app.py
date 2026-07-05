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
import json
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
# Clinical Criticality (CC) lookup — rule-based, no ML (Phase 2, Day 4)
# Rescaled 1-10 (doubled from the plan's 1-5 table) so CAS ceiling reaches 10,
# matching the documented thresholds (Immediate >= 8, Investigate >= 5).
CC_LOOKUP = {
    "ICU Ventilator": 10,
    "Infusion Pump": 8,
    "Radiology": 6,
    "Nurse WS": 4,
    "Admin PC": 2,
}
DEFAULT_CC = 2  # unknown device types treated as lowest criticality

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
    """Map Random Forest confidence (0-1) -> Threat Risk score (1-10)."""
    return round(1 + confidence * 9, 2)


def if_to_ts_score(anomaly_score: float, hour_of_day: int) -> float:
    """
    Map Isolation Forest decision_function() output -> Time Sensitivity (1-10).
    score < IF_THRESHOLD (loaded from if_threshold.pkl) -> anomalous -> TS = 10
    otherwise -> routine traffic -> TS driven by time-of-day
                 (off-hours = higher sensitivity, since fewer staff on duty)
    """
    if anomaly_score < IF_THRESHOLD:
        return 10.0
    if hour_of_day < 6 or hour_of_day >= 22:  # night shift, low staffing
        return 6.0
    return 2.0


def lookup_cc(device_type: str) -> float:
    return CC_LOOKUP.get(device_type, DEFAULT_CC)


def lookup_ae(cve_known_exploited: bool) -> float:
    """Active Exploitation — rule based on CVE/CVSS lookup (stubbed input flag)."""
    return 10.0 if cve_known_exploited else 2.0


def lookup_tc(hour_of_day: int) -> float:
    """Temporal Context — shift-based rule (night shift = higher weight)."""
    return 8.0 if (hour_of_day < 6 or hour_of_day >= 22) else 4.0


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
# Simple built-in test dashboard (GET /)
# --------------------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CAAP AI Server — Test Console</title>
<style>
  :root {
    --bg: #0f1420; --panel: #171d2b; --border: #2a3244; --text: #e6e9ef;
    --muted: #8b93a7; --accent: #4f8cff; --green: #2ecc71; --orange: #f39c12; --red: #e74c3c;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 32px; line-height: 1.5;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1100px; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px;
  }
  .card h2 { font-size: 15px; margin: 0 0 12px; display: flex; align-items: center; gap: 8px; }
  button {
    background: var(--accent); color: white; border: none; border-radius: 6px;
    padding: 9px 16px; font-size: 13px; cursor: pointer; font-weight: 600;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  textarea {
    width: 100%; height: 340px; background: #0c111c; color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: 10px;
    font-family: "SF Mono", Consolas, monospace; font-size: 12px; resize: vertical;
  }
  pre {
    background: #0c111c; border: 1px solid var(--border); border-radius: 6px;
    padding: 12px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
    min-height: 60px; margin-top: 10px;
  }
  .status-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); display: inline-block; }
  .status-dot.ok { background: var(--green); }
  .status-dot.fail { background: var(--red); }
  .row { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700;
  }
  .badge.Immediate { background: var(--red); color: white; }
  .badge.Investigate { background: var(--orange); color: white; }
  .badge.Monitor { background: var(--green); color: white; }
  .muted { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>

<h1>CAAP AI Server — Test Console</h1>
<div class="subtitle">Quick manual checks for /health and /predict — no curl/Postman needed.</div>

<div class="grid">

  <div class="card">
    <h2><span class="status-dot" id="health-dot"></span> Health Check</h2>
    <p class="muted">Confirms the server is running and models are loaded.</p>
    <button onclick="checkHealth()">Test /health</button>
    <pre id="health-result">Not checked yet.</pre>
  </div>

  <div class="card">
    <h2><span class="status-dot" id="predict-dot"></span> Predict</h2>
    <p class="muted">Edit the JSON below (or leave the sample as-is), then send it to /predict.</p>
    <textarea id="predict-input">SAMPLE_JSON_PLACEHOLDER</textarea>
    <div class="row" style="margin-top:10px;">
      <button onclick="runPredict()">Send to /predict</button>
      <span id="predict-summary"></span>
    </div>
    <pre id="predict-result">No request sent yet.</pre>
  </div>

</div>

<script>
async function checkHealth() {
  const dot = document.getElementById('health-dot');
  const out = document.getElementById('health-result');
  out.textContent = 'Checking...';
  try {
    const res = await fetch('/health');
    const data = await res.json();
    dot.className = 'status-dot ' + (res.ok ? 'ok' : 'fail');
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    dot.className = 'status-dot fail';
    out.textContent = 'Request failed: ' + err;
  }
}

async function runPredict() {
  const dot = document.getElementById('predict-dot');
  const out = document.getElementById('predict-result');
  const summary = document.getElementById('predict-summary');
  const input = document.getElementById('predict-input').value;
  summary.innerHTML = '';
  out.textContent = 'Sending...';

  let payload;
  try {
    payload = JSON.parse(input);
  } catch (err) {
    dot.className = 'status-dot fail';
    out.textContent = 'Invalid JSON in the box above: ' + err;
    return;
  }

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    dot.className = 'status-dot ' + (res.ok ? 'ok' : 'fail');
    out.textContent = JSON.stringify(data, null, 2);

    if (data.action) {
      summary.innerHTML = `<span class="badge ${data.action}">${data.action}</span>` +
        ` &nbsp; label: <b>${data.label}</b> &nbsp; CAS: <b>${data.CAS}</b> &nbsp; confidence: ${data.confidence}`;
    }
  } catch (err) {
    dot.className = 'status-dot fail';
    out.textContent = 'Request failed: ' + err;
  }
}

// Run a health check automatically on page load
checkHealth();
</script>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Simple built-in test dashboard — buttons for /health and /predict, no curl needed."""
    sample_json = json.dumps(
        {**{col: 0.0 for col in FEATURE_COLUMNS}, **{
            "device_type": "ICU Ventilator",
            "department": "ICU",
            "hour_of_day": 3,
            "cve_known_exploited": True,
        }},
        indent=2,
    )
    return INDEX_HTML.replace("SAMPLE_JSON_PLACEHOLDER", sample_json)


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
