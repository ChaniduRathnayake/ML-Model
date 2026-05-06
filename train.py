"""
=============================================================
  CAAP IoMT IDS — TRAINING SCRIPT v4
  3-Model Pipeline: Random Forest + Isolation Forest + K-Means
  + CAS Scoring Engine Integration

  ► DROP YOUR FILES HERE BEFORE RUNNING:
      data/train/iomt_train.csv
      data/test/iomt_test.csv

  Real CIC IoMT 2024 column schema (45 network features):
    Header_Length, Protocol Type, Duration, Rate, Srate, Drate,
    fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
    ack_flag_number, ece_flag_number, cwr_flag_number,
    ack_count, syn_count, fin_count, rst_count,
    HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
    TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC,
    Tot sum, Min, Max, AVG, Std, Tot size,
    IAT, Number, Magnitue, Radius, Covariance, Variance, Weight

  Medical metadata columns (used for CAS only, NOT ML input):
    device_type, cc_score, department,
    patient_dependency, time_sensitivity, shift

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
  Usage  : python train.py
  Output : models/*.pkl   reports/*.png   reports/classification_report.txt
=============================================================
"""

import os, sys, warnings, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, silhouette_score,
)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cas_engine import score_alert

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_CSV  = "data/train/iomt_train.csv"
TEST_CSV   = "data/test/iomt_test.csv"
MODEL_DIR  = "models"
REPORT_DIR = "reports"
N_CLUSTERS = 3

# Medical metadata — used for CAS scoring, excluded from ML feature matrix
MEDICAL_COLS = [
    "device_type", "cc_score", "department",
    "patient_dependency", "time_sensitivity", "shift",
]

CLUSTER_LABELS = {0: "active", 1: "routine", 2: "idle"}

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

t_start = time.time()


# ── STEP 1 : LOAD DATA ────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 1 — LOADING DATA")
print("=" * 62)

# Support both CSV and Excel (in case files were saved as .xlsx)
def smart_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ✗ File not found: {path}\n"
            f"  → Place your dataset at: {os.path.abspath(path)}\n"
        )
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_csv(path, encoding="latin1")

train_df = smart_read(TRAIN_CSV).dropna()
test_df  = smart_read(TEST_CSV).dropna()

print(f"  ✓ Train : {len(train_df):>8,} rows  |  {train_df.shape[1]} columns")
print(f"  ✓ Test  : {len(test_df):>8,} rows  |  {test_df.shape[1]} columns")
print(f"  Train labels : {sorted(train_df['label'].unique())}")
print(f"  Test  labels : {sorted(test_df['label'].unique())}")

# Check for label mismatch
missing_in_train = set(test_df["label"].unique()) - set(train_df["label"].unique())
if missing_in_train:
    print(f"\n  ⚠  Labels in test but not train: {missing_in_train}")
    print(f"  → These rows will be dropped from test.")
    test_df = test_df[test_df["label"].isin(train_df["label"].unique())]
else:
    print(f"  ✓ All test labels present in training data.")


# ── STEP 2 : PREPROCESSING ────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 2 — PREPROCESSING")
print("=" * 62)

# Save medical metadata for CAS (separate from ML training)
train_meta = train_df[[c for c in MEDICAL_COLS if c in train_df.columns]].copy()
test_meta  = test_df[[c  for c in MEDICAL_COLS if c in test_df.columns]].copy()

# Build feature column list: all numeric columns except label & medical metadata
exclude = {"label"} | set(MEDICAL_COLS)
FEATURE_COLS = []
dropped = []
for col in train_df.columns:
    if col in exclude:
        continue
    try:
        train_df[col] = pd.to_numeric(train_df[col])
        test_df[col]  = pd.to_numeric(test_df[col])
        FEATURE_COLS.append(col)
    except Exception:
        dropped.append(col)

if dropped:
    print(f"  ⚠  Dropped non-numeric columns: {dropped}")

print(f"  Network feature columns : {len(FEATURE_COLS)}")
print(f"  Medical metadata cols   : {list(train_meta.columns)}")
print(f"  Feature list: {FEATURE_COLS}")

# Labels
y_train_raw = train_df["label"].values
y_test_raw  = test_df["label"].values

le = LabelEncoder()
le.fit(np.concatenate([y_train_raw, y_test_raw]))
y_train = le.transform(y_train_raw)
y_test  = le.transform(y_test_raw)
print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")

# Feature matrices
X_train_raw = train_df[FEATURE_COLS].values
X_test_raw  = test_df[FEATURE_COLS].values

# StandardScaler — fit ONLY on training data
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
print(f"  StandardScaler fitted on {len(X_train):,} training rows ✓")

# Class-balanced sample weights for Random Forest
sample_weights = compute_sample_weight("balanced", y_train)

print(f"\n  Class distribution (train):")
for cls, cnt in sorted(zip(*np.unique(y_train_raw, return_counts=True)),
                        key=lambda x: -x[1]):
    pct = cnt / len(y_train_raw) * 100
    bar = "█" * int(pct / 2)
    print(f"    {cls:<25} {cnt:>8,}  ({pct:5.1f}%)  {bar}")


# ── STEP 3 : TRAIN MODELS ─────────────────────────────────────────────────────

# ── 3a : Random Forest ────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 3a — Random Forest  [TR dimension — main classifier]")
print("=" * 62)

rf = RandomForestClassifier(
    n_estimators   = 300,      # 300 trees — strong ensemble
    max_depth      = None,     # Grow fully — IoMT features support this
    min_samples_leaf = 1,
    min_samples_split = 2,
    max_features   = "sqrt",   # Standard for classification
    class_weight   = "balanced",
    bootstrap      = True,
    oob_score      = True,     # Free validation estimate
    random_state   = 42,
    n_jobs         = -1,
    verbose        = 1,
)
rf.fit(X_train, y_train, sample_weight=sample_weights)
print(f"\n  ✓ Random Forest trained")
print(f"  OOB Score (free train estimate): {rf.oob_score_ * 100:.2f}%")

# ── 3b : Isolation Forest ─────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 3b — Isolation Forest  [TS dimension — anomaly detector]")
print("=" * 62)

iso = IsolationForest(
    n_estimators  = 200,
    contamination = 0.10,   # ~10% of traffic expected anomalous
    max_samples   = "auto",
    max_features  = 1.0,
    random_state  = 42,
    n_jobs        = -1,
    verbose       = 1,
)
iso.fit(X_train)
print(f"  ✓ Isolation Forest trained")

# ── 3c : K-Means ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print(f"  STEP 3c — K-Means k={N_CLUSTERS}  [traffic behaviour clustering]")
print("=" * 62)

km = KMeans(
    n_clusters = N_CLUSTERS,
    init       = "k-means++",
    n_init     = 20,
    max_iter   = 500,
    random_state = 42,
    verbose    = 1,
)
km.fit(X_train)
print(f"  ✓ K-Means trained")


# ── STEP 4 : SAVE ARTIFACTS ───────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 4 — SAVING ARTIFACTS  →  models/")
print("=" * 62)

joblib.dump(rf,           os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(iso,          os.path.join(MODEL_DIR, "isolation_forest.pkl"))
joblib.dump(km,           os.path.join(MODEL_DIR, "kmeans.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le,           os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

for fname in ["random_forest.pkl", "isolation_forest.pkl", "kmeans.pkl",
              "scaler.pkl", "label_encoder.pkl", "feature_cols.pkl"]:
    size = os.path.getsize(os.path.join(MODEL_DIR, fname)) / 1024
    print(f"  ✓ {fname:<30}  ({size:>7.1f} KB)")


# ── STEP 5 : EVALUATE ─────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 5 — EVALUATING ON TEST SET")
print("=" * 62)

y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)
confidence = y_proba_rf.max(axis=1)
y_pred_iso = iso.predict(X_test)
iso_scores = iso.decision_function(X_test)
y_pred_km  = km.predict(X_test)

acc = accuracy_score(y_test, y_pred_rf)
print(f"\n  ✅  Random Forest Accuracy  : {acc * 100:.2f}%")

try:
    auc = roc_auc_score(y_test, y_proba_rf, multi_class="ovr", average="macro")
    print(f"  ✅  AUC-ROC (macro OvR)     : {auc:.4f}")
except Exception:
    auc = None

n_anom = (y_pred_iso == -1).sum()
print(f"  ✅  IF Anomalies detected   : {n_anom:,} / {len(y_pred_iso):,} "
      f"({n_anom / len(y_pred_iso) * 100:.1f}%)")

try:
    sil_idx = np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)
    sil = silhouette_score(X_test[sil_idx], y_pred_km[sil_idx])
    print(f"  ✅  K-Means Silhouette     : {sil:.4f}")
except Exception:
    sil = None

print(f"\n{classification_report(y_test, y_pred_rf, target_names=le.classes_)}")

# Save classification report
with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
    f.write("CAAP IoMT IDS — Random Forest Classification Report\n")
    f.write(f"Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security\n")
    f.write(f"Overall Test Accuracy : {acc * 100:.2f}%\n")
    if auc:
        f.write(f"AUC-ROC (macro OvR)   : {auc:.4f}\n")
    f.write("\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=le.classes_))

print(f"  K-Means traffic clusters:")
u, c = np.unique(y_pred_km, return_counts=True)
for ui, ci in zip(u, c):
    print(f"    {CLUSTER_LABELS.get(ui, '?'):8s}: {ci:,}")


# ── STEP 6 : PER-CLASS ACCURACY ───────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 6 — PER ATTACK TYPE ACCURACY")
print("=" * 62)
print(f"  {'Attack Type':<28} {'Samples':>8}  {'Accuracy':>9}  {'Status'}")
print("  " + "─" * 62)

class_accs = {}
for attack in le.classes_:
    mask = y_test_raw == attack
    if not mask.any():
        continue
    ai = accuracy_score(y_test[mask], y_pred_rf[mask])
    class_accs[attack] = ai * 100
    bar    = "█" * int(ai * 20)
    status = "✅" if ai >= 0.90 else "⚠️ " if ai >= 0.75 else "❌"
    print(f"  {status} {attack:<26} {mask.sum():>8,}  {ai * 100:>8.2f}%  {bar}")


# ── STEP 7 : CAS SCORING DEMO ─────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 7 — CAS SCORING (first 10 test predictions)")
print("=" * 62)
print(f"\n  {'True':<22} {'Predicted':<22} {'Conf':>6}  {'Device':<20}  {'CAS':>6}  Action")
print("  " + "─" * 92)

for i in range(min(10, len(X_test))):
    pred_lbl = le.inverse_transform([y_pred_rf[i]])[0]
    true_lbl = y_test_raw[i]
    cc_raw   = test_meta["cc_score"].iloc[i]   if "cc_score"         in test_meta.columns else 5
    shift    = test_meta["shift"].iloc[i]       if "shift"            in test_meta.columns else "day"
    ts_val   = test_meta["time_sensitivity"].iloc[i] if "time_sensitivity" in test_meta.columns else 3
    dev      = test_meta["device_type"].iloc[i] if "device_type"      in test_meta.columns else "Unknown"
    cas_r = score_alert(pred_lbl, float(confidence[i]), float(iso_scores[i]),
                        bool(y_pred_iso[i] == -1), cc_raw, str(shift), ts_val)
    match = "✓" if pred_lbl == true_lbl else "✗"
    print(f"  {match} {true_lbl:<21} {pred_lbl:<22} {confidence[i]:>5.3f}"
          f"  {str(dev):<20}  {cas_r['CAS']:>6.2f}  {cas_r['action']}")


# ── STEP 8 : SAVE FULL PREDICTIONS WITH CAS ───────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 8 — SAVING PREDICTIONS CSV")
print("=" * 62)

out = test_df[["label"]].copy().reset_index(drop=True)
out["predicted"]  = le.inverse_transform(y_pred_rf)
out["confidence"] = np.round(confidence, 4)
out["iso_score"]  = np.round(iso_scores, 4)
out["is_anomaly"] = y_pred_iso == -1
out["cluster"]    = [CLUSTER_LABELS.get(ci, str(ci)) for ci in y_pred_km]
out["correct"]    = out["label"] == out["predicted"]

# Add medical metadata columns if present
for col in ["device_type", "cc_score", "shift", "department",
            "patient_dependency", "time_sensitivity"]:
    if col in test_meta.columns:
        out[col] = test_meta[col].values

# Compute CAS for every test row
cas_col, action_col = [], []
for i in range(len(out)):
    cc_raw = test_meta["cc_score"].iloc[i]         if "cc_score"         in test_meta.columns else 5
    shift  = test_meta["shift"].iloc[i]             if "shift"            in test_meta.columns else "day"
    ts_val = test_meta["time_sensitivity"].iloc[i]  if "time_sensitivity" in test_meta.columns else 3
    r = score_alert(out["predicted"].iloc[i], float(confidence[i]),
                    float(iso_scores[i]), bool(y_pred_iso[i] == -1),
                    cc_raw, str(shift), ts_val)
    cas_col.append(r["CAS"])
    action_col.append(r["action"])

out["CAS"]    = cas_col
out["action"] = action_col
out.to_csv(os.path.join(REPORT_DIR, "predictions.csv"), index=False)
print(f"  ✓ {len(out):,} rows saved → reports/predictions.csv")


# ── STEP 9 : CHARTS ───────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 9 — GENERATING CHARTS  →  reports/")
print("=" * 62)

plt.style.use("seaborn-v0_8-whitegrid")

# 9a — Confusion Matrix
fig, ax = plt.subplots(figsize=(12, 9))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title(f"Confusion Matrix — RF Accuracy: {acc * 100:.2f}%",
             fontsize=14, pad=12)
ax.set_ylabel("Actual Label"); ax.set_xlabel("Predicted Label")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(); print("  ✓ confusion_matrix.png")

# 9b — Per-Class Accuracy
colors = ["#2ecc71" if v >= 90 else "#e67e22" if v >= 75 else "#e74c3c"
          for v in class_accs.values()]
fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(class_accs.keys(), class_accs.values(),
              color=colors, edgecolor="white", linewidth=0.8)
ax.axhline(90, color="#27ae60", linestyle="--", alpha=0.7, label="90% target")
ax.axhline(75, color="#e67e22", linestyle="--", alpha=0.5, label="75% baseline")
for bar, val in zip(bars, class_accs.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Per-Class Accuracy on Test Set", fontsize=14)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 115)
plt.xticks(rotation=35, ha="right"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "per_class_accuracy.png"), dpi=150)
plt.close(); print("  ✓ per_class_accuracy.png")

# 9c — Training Class Distribution
fig, ax = plt.subplots(figsize=(13, 4))
counts = pd.Series(y_train_raw).value_counts()
bars2 = ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
ax.set_title("Training Set — Samples per Class", fontsize=14)
ax.set_ylabel("Count"); ax.set_xlabel("Attack / Traffic Type")
for bar, val in zip(bars2, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
            f"{val:,}", ha="center", fontsize=9)
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "class_distribution.png"), dpi=150)
plt.close(); print("  ✓ class_distribution.png")

# 9d — Feature Importances (Top 20)
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
top20 = importances.nlargest(20).sort_values()
fig, ax = plt.subplots(figsize=(11, 7))
colors_fi = ["#e74c3c" if v >= top20.quantile(0.75) else "#3498db"
             for v in top20.values]
top20.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.set_title("Top 20 Feature Importances — Random Forest", fontsize=13)
ax.set_xlabel("Importance Score"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "feature_importances.png"), dpi=150)
plt.close(); print("  ✓ feature_importances.png")

# 9e — Isolation Forest Score Distribution
fig, ax = plt.subplots(figsize=(11, 4))
is_attack = y_test_raw != "Benign"
ax.hist(iso_scores[~is_attack], bins=80, alpha=0.65, color="#3498db",
        label="Benign", density=True)
ax.hist(iso_scores[is_attack],  bins=80, alpha=0.65, color="#e74c3c",
        label="Attack", density=True)
ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="IF boundary (0)")
ax.set_title("Isolation Forest — Anomaly Score Distribution", fontsize=13)
ax.set_xlabel("IF Decision Score (more negative = more anomalous)")
ax.set_ylabel("Density"); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "if_anomaly_dist.png"), dpi=150)
plt.close(); print("  ✓ if_anomaly_dist.png")

# 9f — K-Means Cluster Sizes
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([CLUSTER_LABELS.get(ui, str(ui)) for ui in u], c,
       color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="white")
ax.set_title("K-Means — Traffic Behaviour Groups", fontsize=13)
ax.set_ylabel("Sample Count"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "kmeans_clusters.png"), dpi=150)
plt.close(); print("  ✓ kmeans_clusters.png")

# 9g — CAS Distribution by Action Tier
fig, ax = plt.subplots(figsize=(11, 4))
tier_colors = {"Immediate": "#e74c3c", "Investigate": "#e67e22", "Monitor": "#2ecc71"}
for tier, col in tier_colors.items():
    vals = [cas for cas, act in zip(cas_col, action_col) if act == tier]
    if vals:
        ax.hist(vals, bins=40, alpha=0.7, color=col, label=f"{tier} ({len(vals):,})",
                density=True)
ax.axvline(8.0, color="red",    linestyle="--", alpha=0.6, label="CAS 8.0 threshold")
ax.axvline(5.0, color="orange", linestyle="--", alpha=0.6, label="CAS 5.0 threshold")
ax.set_title("CAS Score Distribution by Action Tier", fontsize=13)
ax.set_xlabel("CAS Score (0–10)"); ax.set_ylabel("Density")
ax.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "cas_distribution.png"), dpi=150)
plt.close(); print("  ✓ cas_distribution.png")

# 9h — Model Comparison Summary
fig, ax = plt.subplots(figsize=(9, 5))
summary = {
    "RF Accuracy (%)":      acc * 100,
    "RF OOB Score (%)":     rf.oob_score_ * 100,
    "IF Anomaly Rate (%)":  n_anom / len(y_pred_iso) * 100,
    "AUC-ROC (×100)":       (auc * 100) if auc else 0,
    "K-Means Silhouette\n(×100)": (sil * 100) if sil else 0,
}
bar_colors = ["#2ecc71", "#27ae60", "#e67e22", "#9b59b6", "#3498db"]
y_pos = range(len(summary))
hbars = ax.barh(list(summary.keys()), list(summary.values()),
                color=bar_colors, edgecolor="white")
for bar, val in zip(hbars, summary.values()):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontweight="bold")
ax.set_xlim(0, 115)
ax.set_title("CAAP Multi-Model Summary", fontsize=13)
ax.set_xlabel("Score"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "model_summary.png"), dpi=150)
plt.close(); print("  ✓ model_summary.png")


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
print("\n" + "=" * 62)
print(f"  ✅  ALL DONE!")
print(f"  Overall Accuracy : {acc * 100:.2f}%")
if auc:
    print(f"  AUC-ROC          : {auc:.4f}")
print(f"  OOB Score        : {rf.oob_score_ * 100:.2f}%")
print(f"  Training time    : {elapsed:.1f}s")
print("=" * 62)
print(f"  Models  → {MODEL_DIR}/")
print(f"  Reports → {REPORT_DIR}/")
print("=" * 62 + "\n")
