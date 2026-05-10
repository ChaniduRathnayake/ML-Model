"""
=============================================================
  CAAP IoMT IDS — TESTING / INFERENCE SCRIPT v5
  Loads saved models and evaluates on all test pcap CSVs.
  Also simulates the Flask POST /predict response.

  ► Requires models/ to be populated by train.py first.
  ► Requires data/test/*.pcap.csv  (individual attack files)

  Labels are derived from filenames — same mapping as train.py.
  CAS scoring uses real Dst Port + Protocol from each packet row.

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
  Usage  : python test.py
=============================================================
"""

import os, sys, re, json, glob, warnings, datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cas_engine import score_alert

warnings.filterwarnings("ignore")

TEST_DIR   = "data/test"
MODEL_DIR  = "models"

CLUSTER_LABELS = {0: "active", 1: "idle"}

# IP protocol number → string (for CAS engine lookup)
PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp", 2: "igmp"}

# ── LABEL MAP (mirrors train.py exactly) ─────────────────────────────────────
LABEL_MAP = [
    (r"^arp_spoofing",               "ARP_Spoofing"),
    (r"^benign",                     "Benign"),
    (r"^mqtt.*flood",                "MQTT_Publish_Flood"),
    (r"^mqtt.*malformed",            "MQTT_Brute_Force"),
    (r"^mqtt",                       "MQTT_Publish_Flood"),
    (r"^recon",                      "Recon"),
    (r"^tcp_ip.*(dos|ddos).*(syn|tcp|icmp|udp)", "DoS_TCP"),
]

def filename_to_label(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0].lower()
    stem = re.sub(r"[_\-](train|test)(\.pcap)?$", "", stem)
    stem = re.sub(r"\.pcap$", "", stem)
    for pattern, label in LABEL_MAP:
        if re.search(pattern, stem, re.IGNORECASE):
            return label
    return stem.replace("-", "_").replace(" ", "_").title()


def normalise_protocol(val) -> str:
    """Convert IP protocol number or string to a CAS engine key."""
    try:
        return PROTO_MAP.get(int(float(val)), "tcp")
    except (TypeError, ValueError):
        s = str(val).strip().lower()
        return s if s in ("tcp", "udp", "icmp", "igmp", "mqtt") else "tcp"


def current_shift() -> str:
    h = datetime.datetime.now().hour
    if 7 <= h < 15:  return "day"
    if 15 <= h < 23: return "evening"
    return "night"


# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  LOADING SAVED MODELS")
print("=" * 64)

rf           = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
iso          = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
km           = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

print(f"  ✓ All models loaded")
print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")


# ── LOAD & MERGE TEST DATA ────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  LOADING & MERGING TEST CSV FILES")
print("=" * 64)

csv_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError(
        f"\n  ✗ No CSV files found in: {os.path.abspath(TEST_DIR)}\n"
        f"  → Place pcap CSV files there and re-run.\n"
    )

frames = []
for f in csv_files:
    try:
        df = pd.read_csv(f, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    lbl = filename_to_label(f)
    df["label"] = lbl
    frames.append(df)
    print(f"    {os.path.basename(f):<55}  {len(df):>8,} rows  →  {lbl}")

test_df = pd.concat(frames, ignore_index=True).dropna(how="all")
print(f"\n  ✓ Combined test set: {len(test_df):,} rows")

# ── EXTRACT REAL PORT / PROTOCOL BEFORE ANY FILTERING ─────────────────────────
# Find column names case-insensitively
_col_map = {c.lower(): c for c in test_df.columns}
_dst_col  = _col_map.get("dst port") or _col_map.get("dst_port")
_prot_col = _col_map.get("protocol type") or _col_map.get("protocol") or _col_map.get("protocol_type")

_meta_dst   = test_df[_dst_col].copy()  if _dst_col  else pd.Series(0,     index=test_df.index)
_meta_proto = test_df[_prot_col].copy() if _prot_col else pd.Series("tcp",  index=test_df.index)

if _dst_col:
    print(f"  ✓ Dst Port column   : '{_dst_col}'")
else:
    print(f"  ⚠  'Dst Port' column not found — CAS will use default device profile")
if _prot_col:
    print(f"  ✓ Protocol column   : '{_prot_col}'")

# Keep only labels known to the model
known_labels = set(le.classes_)
unknown = set(test_df["label"].unique()) - known_labels
if unknown:
    print(f"  ⚠  Unknown labels dropped: {unknown}")
    test_df = test_df[test_df["label"].isin(known_labels)]

# Coerce features to numeric; track which rows survive the NaN drop
for col in FEATURE_COLS:
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

surviving_idx = test_df.dropna(subset=FEATURE_COLS).index
test_df       = test_df.loc[surviving_idx].reset_index(drop=True)

# Align metadata to surviving rows
dst_ports   = _meta_dst.loc[surviving_idx].reset_index(drop=True)
protocols_s = _meta_proto.loc[surviving_idx].reset_index(drop=True)

X_test     = scaler.transform(test_df[FEATURE_COLS].values.astype(np.float64))
y_test_raw = test_df["label"].values
y_test     = le.transform(y_test_raw)

print(f"  ✓ {len(X_test):,} valid test rows after NaN drop")


# ── FULL EVALUATION ───────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  FULL TEST EVALUATION")
print("=" * 64)

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
print("\n" + "=" * 64)
print("  PER ATTACK TYPE ACCURACY")
print("=" * 64)
print(f"  {'Attack Type':<32} {'Samples':>8}  {'Accuracy':>9}")
print("  " + "─" * 58)
for attack in le.classes_:
    mask = y_test_raw == attack
    if not mask.any():
        continue
    ai  = accuracy_score(y_test[mask], y_pred_rf[mask])
    bar = "█" * int(ai * 20)
    status = "✅" if ai >= 0.90 else "⚠️ " if ai >= 0.75 else "❌"
    print(f"  {status} {attack:<30} {mask.sum():>8,}  {ai * 100:>8.2f}%  {bar}")


# ── CAS SCORING ───────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  CAS SCORING  (real Dst Port + Protocol from packets)")
print("=" * 64)

shift = current_shift()
print(f"  Current shift : {shift}")

cas_col, action_col, device_col, fda_col = [], [], [], []
for i in range(len(y_pred_rf)):
    dst  = int(pd.to_numeric(dst_ports.iloc[i], errors="coerce") or 0)
    prot = normalise_protocol(protocols_s.iloc[i])
    r = score_alert(
        le.inverse_transform([y_pred_rf[i]])[0],
        float(confidence[i]),
        float(iso_scores[i]),
        bool(y_pred_iso[i] == -1),
        dst_port = dst,
        protocol = prot,
        shift    = shift,
    )
    cas_col.append(r["CAS"])
    action_col.append(r["action"])
    device_col.append(r["device_name"])
    fda_col.append(r["fda_class"])

# Show first 10 with CAS
print(f"\n  {'True':<24} {'Predicted':<24} {'Conf':>6}  {'Device':<30}  {'FDA':>5}  {'CAS':>6}  Action")
print("  " + "─" * 108)
for i in range(min(10, len(y_pred_rf))):
    true_lbl = y_test_raw[i]
    pred_lbl = le.inverse_transform([y_pred_rf[i]])[0]
    match    = "✓" if pred_lbl == true_lbl else "✗"
    dev      = device_col[i][:28]
    print(f"  {match} {true_lbl:<23} {pred_lbl:<24} {confidence[i]:>5.3f}"
          f"  {dev:<30}  {fda_col[i]:>5}  {cas_col[i]:>6.2f}  {action_col[i]}")

# CAS action summary
from collections import Counter
action_counts = Counter(action_col)
print(f"\n  CAS Action Summary:")
for act in ["Immediate", "Investigate", "Monitor"]:
    n = action_counts.get(act, 0)
    bar = "█" * int(n / max(action_counts.values()) * 30)
    print(f"    {act:<12}  {n:>8,}  {bar}")


# ── SIMULATE Flask /predict ────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  SIMULATING Flask POST /predict (single sample)")
print("=" * 64)


def predict_single(raw_features: dict, dst_port: int = 0,
                   protocol: str = "tcp", shift: str = "day") -> dict:
    """Mirrors the Flask /predict endpoint with real CAS scoring."""
    vec        = np.array([[float(raw_features.get(col, 0.0)) for col in FEATURE_COLS]])
    vec_scaled = scaler.transform(vec)

    pred_idx   = rf.predict(vec_scaled)[0]
    proba      = rf.predict_proba(vec_scaled)[0]
    label      = le.inverse_transform([pred_idx])[0]
    conf       = round(float(proba.max()), 4)
    iso_score  = round(float(iso.decision_function(vec_scaled)[0]), 4)
    is_anomaly = bool(iso.predict(vec_scaled)[0] == -1)
    cluster_id = int(km.predict(vec_scaled)[0])

    cas_r = score_alert(label, conf, iso_score, is_anomaly,
                        dst_port=dst_port, protocol=protocol, shift=shift)

    return {
        "label"       : label,
        "confidence"  : conf,
        "iso_score"   : iso_score,
        "cluster"     : CLUSTER_LABELS.get(cluster_id, "unknown"),
        "is_anomaly"  : is_anomaly,
        "CAS"         : cas_r["CAS"],
        "action"      : cas_r["action"],
        "device_name" : cas_r["device_name"],
        "fda_class"   : cas_r["fda_class"],
        "dimensions"  : {k: cas_r[k] for k in ["TR", "CC", "TS", "AE", "TC"]},
    }


sample_feat = test_df[FEATURE_COLS].iloc[0].to_dict()
sample_dst  = int(pd.to_numeric(dst_ports.iloc[0], errors="coerce") or 0)
sample_prot = normalise_protocol(protocols_s.iloc[0])
result      = predict_single(sample_feat, sample_dst, sample_prot, shift)

print(f"\n  Actual label  : '{y_test_raw[0]}'")
print(f"  Dst Port      : {sample_dst}   Protocol : {sample_prot}   Shift : {shift}")
print(f"\n  Flask /predict response:")
print("  " + json.dumps(result, indent=4).replace("\n", "\n  "))


# ── SAVE PREDICTIONS CSV ──────────────────────────────────────────────────────
os.makedirs("reports", exist_ok=True)
out = test_df[["label"]].copy()
out["predicted"]   = le.inverse_transform(y_pred_rf)
out["confidence"]  = np.round(confidence, 4)
out["iso_score"]   = np.round(iso_scores, 4)
out["is_anomaly"]  = y_pred_iso == -1
out["cluster"]     = [CLUSTER_LABELS.get(ci, str(ci)) for ci in y_pred_km]
out["correct"]     = out["label"] == out["predicted"]
out["dst_port"]    = dst_ports.values
out["protocol"]    = [normalise_protocol(p) for p in protocols_s]
out["device_name"] = device_col
out["fda_class"]   = fda_col
out["CAS"]         = cas_col
out["action"]      = action_col

out.to_csv("reports/test_predictions.csv", index=False)
print(f"\n  ✓ Saved {len(out):,} rows → reports/test_predictions.csv")
print(f"    Columns: {list(out.columns)}")

print("\n" + "=" * 64)
print(f"  ✅ TESTING COMPLETE!   Accuracy: {acc * 100:.2f}%")
print("=" * 64 + "\n")
