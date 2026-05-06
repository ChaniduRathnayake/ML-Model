"""
=============================================================
  CAAP IoMT IDS — TESTING / INFERENCE SCRIPT v4
  Loads saved models and evaluates on real test dataset.
  Also simulates the Flask POST /predict response.

  ► Requires models/ to be populated by train.py first.
  ► Requires data/test/iomt_test.csv

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
  Usage  : python test.py
=============================================================
"""

import os, sys, json, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cas_engine import score_alert

warnings.filterwarnings("ignore")

TEST_CSV   = "data/test/iomt_test.csv"
MODEL_DIR  = "models"
MEDICAL_COLS = [
    "device_type", "cc_score", "department",
    "patient_dependency", "time_sensitivity", "shift",
]
CLUSTER_LABELS = {0: "active", 1: "routine", 2: "idle"}


# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  LOADING SAVED MODELS")
print("=" * 62)

rf           = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
iso          = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
km           = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

print(f"  ✓ All models loaded")
print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")


# ── LOAD TEST DATA ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  LOADING TEST DATA")
print("=" * 62)

def smart_read(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_csv(path, encoding="latin1")

test_df   = smart_read(TEST_CSV).dropna()
test_meta = test_df[[c for c in MEDICAL_COLS if c in test_df.columns]].copy()

for col in FEATURE_COLS:
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
test_df   = test_df.dropna(subset=FEATURE_COLS)
test_meta = test_meta.loc[test_df.index].reset_index(drop=True)
test_df   = test_df.reset_index(drop=True)

X_test     = scaler.transform(test_df[FEATURE_COLS].values)
y_test_raw = test_df["label"].values
y_test     = le.transform(y_test_raw)
print(f"  ✓ {len(X_test):,} test rows loaded")


# ── EVALUATE ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FULL TEST EVALUATION")
print("=" * 62)

y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)
confidence = y_proba_rf.max(axis=1)
y_pred_iso = iso.predict(X_test)
iso_scores = iso.decision_function(X_test)
y_pred_km  = km.predict(X_test)

acc = accuracy_score(y_test, y_pred_rf)
print(f"\n  ✅ Random Forest Accuracy : {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

n_anom = (y_pred_iso == -1).sum()
print(f"  Isolation Forest : {n_anom:,} anomalies / {len(y_pred_iso):,} "
      f"({n_anom / len(y_pred_iso) * 100:.1f}%)")

u, c = np.unique(y_pred_km, return_counts=True)
for ui, ci in zip(u, c):
    print(f"  K-Means Cluster {ui} ({CLUSTER_LABELS.get(ui, '?'):8s}): {ci:,}")


# ── PER-CLASS BREAKDOWN ────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  PER ATTACK TYPE ACCURACY")
print("=" * 62)
print(f"  {'Attack Type':<28} {'Samples':>8}  {'Accuracy':>9}")
print("  " + "─" * 55)
for attack in le.classes_:
    mask = y_test_raw == attack
    if not mask.any():
        continue
    ai  = accuracy_score(y_test[mask], y_pred_rf[mask])
    bar = "█" * int(ai * 20)
    status = "✅" if ai >= 0.90 else "⚠️ " if ai >= 0.75 else "❌"
    print(f"  {status} {attack:<26} {mask.sum():>8,}  {ai * 100:>8.2f}%  {bar}")


# ── SIMULATE Flask /predict ────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SIMULATING Flask POST /predict (single sample)")
print("=" * 62)


def predict_single(raw_features: dict, medical: dict = None) -> dict:
    """Mirrors the Flask /predict endpoint exactly."""
    vec        = np.array([[float(raw_features.get(col, 0.0)) for col in FEATURE_COLS]])
    vec_scaled = scaler.transform(vec)

    pred_idx   = rf.predict(vec_scaled)[0]
    proba      = rf.predict_proba(vec_scaled)[0]
    label      = le.inverse_transform([pred_idx])[0]
    conf       = round(float(proba.max()), 4)
    iso_score  = round(float(iso.decision_function(vec_scaled)[0]), 4)
    is_anomaly = bool(iso.predict(vec_scaled)[0] == -1)
    cluster_id = int(km.predict(vec_scaled)[0])

    med = medical or {}
    cas_r = score_alert(
        label, conf, iso_score, is_anomaly,
        cc_score_raw = med.get("cc_score",         5),
        shift        = med.get("shift",             "day"),
        ts_col_val   = med.get("time_sensitivity",  3),
    )

    return {
        "label"      : label,
        "confidence" : conf,
        "TR_score"   : iso_score,
        "cluster"    : CLUSTER_LABELS.get(cluster_id, "unknown"),
        "is_anomaly" : is_anomaly,
        "CAS"        : cas_r["CAS"],
        "action"     : cas_r["action"],
        "dimensions" : {k: cas_r[k] for k in ["TR", "CC", "TS", "AE", "TC"]},
    }


sample_feat   = test_df[FEATURE_COLS].iloc[0].to_dict()
sample_med    = test_meta.iloc[0].to_dict() if not test_meta.empty else {}
result        = predict_single(sample_feat, sample_med)

print(f"\n  Actual label  : '{y_test_raw[0]}'")
print(f"  Device type   : '{sample_med.get('device_type', 'N/A')}'")
print(f"  Shift         : '{sample_med.get('shift', 'N/A')}'")
print(f"  cc_score      : {sample_med.get('cc_score', 'N/A')}")
print(f"\n  Flask /predict response:")
print("  " + json.dumps(result, indent=4).replace("\n", "\n  "))


# ── SAVE PREDICTIONS CSV ──────────────────────────────────────────────────────
os.makedirs("reports", exist_ok=True)
out = test_df[["label"]].copy()
out["predicted"]  = le.inverse_transform(y_pred_rf)
out["confidence"] = np.round(confidence, 4)
out["iso_score"]  = np.round(iso_scores, 4)
out["is_anomaly"] = y_pred_iso == -1
out["cluster"]    = [CLUSTER_LABELS.get(ci, str(ci)) for ci in y_pred_km]
out["correct"]    = out["label"] == out["predicted"]
for col in ["device_type", "cc_score", "shift", "time_sensitivity"]:
    if col in test_meta.columns:
        out[col] = test_meta[col].values
out.to_csv("reports/test_predictions.csv", index=False)
print(f"\n  ✓ Saved {len(out):,} rows → reports/test_predictions.csv")

print("\n" + "=" * 62)
print(f"  ✅ TESTING COMPLETE!   Accuracy: {acc * 100:.2f}%")
print("=" * 62 + "\n")
