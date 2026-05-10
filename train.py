"""
=============================================================
  CAAP IoMT IDS — TRAINING SCRIPT v5
  3-Model Pipeline: Random Forest + Isolation Forest + K-Means

  ► HOW TO USE:
      Drop ALL individual pcap CSV files into:
          data/train/   (e.g. ARP_Spoofing_train.pcap.csv)
          data/test/    (e.g. ARP_Spoofing_test.pcap.csv)

      Labels are derived automatically from filenames.
      No single merged CSV required — this script merges them.

  ► Label mapping (filename prefix → label group):
      ARP_Spoofing*         → ARP_Spoofing
      Benign*               → Benign
      MQTT-DDoS-*           → MQTT_Publish_Flood
      MQTT-DoS-*            → MQTT_Publish_Flood
      MQTT-Malformed*       → MQTT_Brute_Force
      Recon-*               → Recon
      TCP_IP-DDoS-SYN*      → DoS_TCP
      TCP_IP-DoS-SYN*       → DoS_TCP
      TCP_IP-DDoS-TCP*      → DoS_TCP
      TCP_IP-DoS-TCP*       → DoS_TCP
      TCP_IP-DDoS-ICMP*     → DoS_TCP
      TCP_IP-DoS-ICMP*      → DoS_TCP
      TCP_IP-DDoS-UDP*      → DoS_TCP
      TCP_IP-DoS-UDP*       → DoS_TCP

  Real CIC IoMT 2024 network feature columns (used as ML input):
    Header_Length, Protocol Type, Duration, Rate, Srate, Drate,
    fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
    ack_flag_number, ece_flag_number, cwr_flag_number,
    ack_count, syn_count, fin_count, rst_count,
    HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
    TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC,
    Tot sum, Min, Max, AVG, Std, Tot size,
    IAT, Number, Magnitue, Radius, Covariance, Variance, Weight

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
  Usage  : python train.py
  Output : models/*.pkl   reports/*.png   reports/classification_report.txt
=============================================================
"""

import os, re, warnings, time, glob
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

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = "data/train"
TEST_DIR   = "data/test"
MODEL_DIR  = "models"
REPORT_DIR = "reports"
N_CLUSTERS = 2

CLUSTER_LABELS = {0: "active", 1: "idle"}

# ── LABEL MAP: filename prefix → canonical label ──────────────────────────────
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
    """Derive label from a pcap CSV filename."""
    stem = os.path.splitext(os.path.basename(fname))[0].lower()
    stem = re.sub(r"[_\-](train|test)(\.pcap)?$", "", stem)
    stem = re.sub(r"\.pcap$", "", stem)
    for pattern, label in LABEL_MAP:
        if re.search(pattern, stem, re.IGNORECASE):
            return label
    return stem.replace("-", "_").replace(" ", "_").title()


# ── HELPERS ───────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
t_start = time.time()


def load_and_merge(directory: str) -> pd.DataFrame:
    """
    Reads every *.csv in `directory`, derives the label from the filename,
    appends a 'label' column, concatenates everything, drops NaNs,
    and returns the combined DataFrame.
    """
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"\n  ✗ No CSV files found in: {os.path.abspath(directory)}\n"
            f"  → Place your pcap CSV files there and re-run.\n"
        )

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)

        lbl = filename_to_label(f)
        df["label"] = lbl
        frames.append(df)
        print(f"    {os.path.basename(f):<55}  {len(df):>8,} rows  →  {lbl}")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = combined.columns.str.strip()
    combined.dropna(how="all", inplace=True)

    return combined.reset_index(drop=True)


# ── STEP 1 : LOAD DATA ────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 1 — LOADING & MERGING CSV FILES")
print("=" * 68)

print(f"\n  [TRAIN — {TRAIN_DIR}]")
train_df = load_and_merge(TRAIN_DIR)

print(f"\n  [TEST  — {TEST_DIR}]")
test_df  = load_and_merge(TEST_DIR)

print(f"\n  ✓ Train combined : {len(train_df):>8,} rows  |  {train_df.shape[1]} columns")
print(f"  ✓ Test  combined : {len(test_df):>8,} rows  |  {test_df.shape[1]} columns")
print(f"\n  Train labels : {sorted(train_df['label'].unique())}")
print(f"  Test  labels : {sorted(test_df['label'].unique())}")

# Drop test labels not seen during training
missing_in_train = set(test_df["label"].unique()) - set(train_df["label"].unique())
if missing_in_train:
    print(f"\n  ⚠  Labels in test but not train: {missing_in_train}")
    test_df = test_df[test_df["label"].isin(train_df["label"].unique())].reset_index(drop=True)
    print(f"  → Dropped. Test now {len(test_df):,} rows.")
else:
    print(f"  ✓ All test labels present in training data.")


# ── STEP 2 : PREPROCESSING ────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 2 — PREPROCESSING")
print("=" * 68)

exclude    = {"label"}
FEATURE_COLS, dropped = [], []

for col in train_df.columns:
    if col in exclude:
        continue
    try:
        train_df[col] = pd.to_numeric(train_df[col], errors="raise")
        test_df[col]  = pd.to_numeric(test_df[col],  errors="raise")
        FEATURE_COLS.append(col)
    except Exception:
        dropped.append(col)

if dropped:
    print(f"  ⚠  Dropped non-numeric / text columns ({len(dropped)}): {dropped[:10]}{'...' if len(dropped)>10 else ''}")

train_df = train_df[FEATURE_COLS + ["label"]].dropna().reset_index(drop=True)
test_df  = test_df[FEATURE_COLS  + ["label"]].dropna().reset_index(drop=True)

print(f"  Network feature columns : {len(FEATURE_COLS)}")
print(f"  Feature list: {FEATURE_COLS}")
print(f"\n  After NaN drop — Train: {len(train_df):,}   Test: {len(test_df):,}")

# Label encoding
y_train_raw = train_df["label"].values
y_test_raw  = test_df["label"].values

le = LabelEncoder()
le.fit(np.concatenate([y_train_raw, y_test_raw]))
y_train = le.transform(y_train_raw)
y_test  = le.transform(y_test_raw)
print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")

# Feature matrices
X_train_raw = train_df[FEATURE_COLS].values.astype(np.float64)
X_test_raw  = test_df[FEATURE_COLS].values.astype(np.float64)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
print(f"  StandardScaler fitted on {len(X_train):,} training rows ✓")

sample_weights = compute_sample_weight("balanced", y_train)

print(f"\n  Class distribution (train):")
for cls, cnt in sorted(zip(*np.unique(y_train_raw, return_counts=True)), key=lambda x: -x[1]):
    pct = cnt / len(y_train_raw) * 100
    bar = "█" * int(pct / 2)
    print(f"    {cls:<28} {cnt:>10,}  ({pct:5.1f}%)  {bar}")


# ── STEP 3 : TRAIN MODELS ─────────────────────────────────────────────────────

# ── 3a : Random Forest ────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 3a — Random Forest  [main classifier]")
print("=" * 68)

rf = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = None,
    min_samples_leaf = 1,
    min_samples_split= 2,
    max_features     = "sqrt",
    class_weight     = "balanced",
    bootstrap        = True,
    oob_score        = True,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = 1,
)
rf.fit(X_train, y_train, sample_weight=sample_weights)
print(f"\n  ✓ Random Forest trained | OOB Score: {rf.oob_score_ * 100:.2f}%")

# ── 3b : Isolation Forest ─────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 3b — Isolation Forest  [anomaly detector]")
print("=" * 68)

iso = IsolationForest(
    n_estimators  = 200,
    contamination = 0.10,
    max_samples   = "auto",
    max_features  = 1.0,
    random_state  = 42,
    n_jobs        = -1,
    verbose       = 1,
)
iso.fit(X_train)
print(f"  ✓ Isolation Forest trained")

# ── 3c : K-Means ─────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"  STEP 3c — K-Means k={N_CLUSTERS}  [traffic behaviour clustering]")
print("=" * 68)

km = KMeans(
    n_clusters   = N_CLUSTERS,
    init         = "k-means++",
    n_init       = 20,
    max_iter     = 500,
    random_state = 42,
    verbose      = 1,
)
km.fit(X_train)
print(f"  ✓ K-Means trained")


# ── STEP 4 : SAVE ARTIFACTS ───────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 4 — SAVING ARTIFACTS  →  models/")
print("=" * 68)

joblib.dump(rf,           os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(iso,          os.path.join(MODEL_DIR, "isolation_forest.pkl"))
joblib.dump(km,           os.path.join(MODEL_DIR, "kmeans.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le,           os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

for fname in ["random_forest.pkl", "isolation_forest.pkl", "kmeans.pkl",
              "scaler.pkl", "label_encoder.pkl", "feature_cols.pkl"]:
    size = os.path.getsize(os.path.join(MODEL_DIR, fname)) / 1024
    print(f"  ✓ {fname:<32}  ({size:>8.1f} KB)")


# ── STEP 5 : EVALUATE ─────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 5 — EVALUATING ON TEST SET")
print("=" * 68)

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

with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
    f.write("CAAP IoMT IDS — Random Forest Classification Report\n")
    f.write("Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security\n")
    f.write(f"Overall Test Accuracy : {acc * 100:.2f}%\n")
    if auc:
        f.write(f"AUC-ROC (macro OvR)   : {auc:.4f}\n")
    f.write("\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=le.classes_))

print(f"\n  K-Means traffic clusters:")
u, c = np.unique(y_pred_km, return_counts=True)
for ui, ci in zip(u, c):
    print(f"    {CLUSTER_LABELS.get(ui, '?'):8s}: {ci:,}")


# ── STEP 6 : PER-CLASS ACCURACY ───────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 6 — PER ATTACK TYPE ACCURACY")
print("=" * 68)
print(f"  {'Attack Type':<32} {'Samples':>8}  {'Accuracy':>9}  {'Status'}")
print("  " + "─" * 65)

class_accs = {}
for attack in le.classes_:
    mask = y_test_raw == attack
    if not mask.any():
        continue
    ai = accuracy_score(y_test[mask], y_pred_rf[mask])
    class_accs[attack] = ai * 100
    bar    = "█" * int(ai * 20)
    status = "✅" if ai >= 0.90 else "⚠️ " if ai >= 0.75 else "❌"
    print(f"  {status} {attack:<30} {mask.sum():>8,}  {ai * 100:>8.2f}%  {bar}")


# ── STEP 7 : SAVE FULL PREDICTIONS ────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 7 — SAVING PREDICTIONS CSV")
print("=" * 68)

out = test_df[["label"]].copy().reset_index(drop=True)
out["predicted"]  = le.inverse_transform(y_pred_rf)
out["confidence"] = np.round(confidence, 4)
out["iso_score"]  = np.round(iso_scores, 4)
out["is_anomaly"] = y_pred_iso == -1
out["cluster"]    = [CLUSTER_LABELS.get(ci, str(ci)) for ci in y_pred_km]
out["correct"]    = out["label"] == out["predicted"]

out.to_csv(os.path.join(REPORT_DIR, "predictions.csv"), index=False)
print(f"  ✓ {len(out):,} rows saved → reports/predictions.csv")


# ── STEP 8 : CHARTS ───────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  STEP 8 — GENERATING CHARTS  →  reports/")
print("=" * 68)

plt.style.use("seaborn-v0_8-whitegrid")

# 8a — Confusion Matrix
fig, ax = plt.subplots(figsize=(12, 9))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title(f"Confusion Matrix — RF Accuracy: {acc * 100:.2f}%", fontsize=14, pad=12)
ax.set_ylabel("Actual Label"); ax.set_xlabel("Predicted Label")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(); print("  ✓ confusion_matrix.png")

# 8b — Per-Class Accuracy
colors = ["#2ecc71" if v >= 90 else "#e67e22" if v >= 75 else "#e74c3c"
          for v in class_accs.values()]
fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(class_accs.keys(), class_accs.values(),
              color=colors, edgecolor="white", linewidth=0.8)
ax.axhline(90, color="#27ae60", linestyle="--", alpha=0.7, label="90% target")
ax.axhline(75, color="#e67e22", linestyle="--", alpha=0.5, label="75% baseline")
for bar, val in zip(bars, class_accs.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_title("Per-Class Accuracy on Test Set", fontsize=14)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 115)
plt.xticks(rotation=35, ha="right"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "per_class_accuracy.png"), dpi=150)
plt.close(); print("  ✓ per_class_accuracy.png")

# 8c — Training Class Distribution
fig, ax = plt.subplots(figsize=(14, 4))
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

# 8d — Feature Importances (Top 20)
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
top20 = importances.nlargest(20).sort_values()
fig, ax = plt.subplots(figsize=(11, 7))
colors_fi = ["#e74c3c" if v >= top20.quantile(0.75) else "#3498db" for v in top20.values]
top20.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.set_title("Top 20 Feature Importances — Random Forest", fontsize=13)
ax.set_xlabel("Importance Score"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "feature_importances.png"), dpi=150)
plt.close(); print("  ✓ feature_importances.png")

# 8e — Isolation Forest Score Distribution
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

# 8f — K-Means Cluster Sizes
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([CLUSTER_LABELS.get(ui, str(ui)) for ui in u], c,
       color=["#e74c3c", "#2ecc71"], edgecolor="white")
ax.set_title("K-Means — Traffic Behaviour Groups", fontsize=13)
ax.set_ylabel("Sample Count"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "kmeans_clusters.png"), dpi=150)
plt.close(); print("  ✓ kmeans_clusters.png")

# 8g — Model Comparison Summary
fig, ax = plt.subplots(figsize=(9, 5))
summary = {
    "RF Accuracy (%)":            acc * 100,
    "RF OOB Score (%)":           rf.oob_score_ * 100,
    "IF Anomaly Rate (%)":        n_anom / len(y_pred_iso) * 100,
    "AUC-ROC (×100)":             (auc * 100) if auc else 0,
    "K-Means Silhouette\n(×100)": (sil * 100) if sil else 0,
}
bar_colors = ["#2ecc71", "#27ae60", "#e67e22", "#9b59b6", "#3498db"]
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
print("\n" + "=" * 68)
print(f"  ✅  ALL DONE!")
print(f"  Classes trained on  : {list(le.classes_)}")
print(f"  Feature columns     : {len(FEATURE_COLS)}")
print(f"  Overall Accuracy    : {acc * 100:.2f}%")
if auc:
    print(f"  AUC-ROC             : {auc:.4f}")
print(f"  OOB Score           : {rf.oob_score_ * 100:.2f}%")
print(f"  Training time       : {elapsed:.1f}s")
print("=" * 68)
print(f"  Models  → {MODEL_DIR}/")
print(f"  Reports → {REPORT_DIR}/")
print("=" * 68 + "\n")
