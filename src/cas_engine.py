"""
=============================================================
  CAAP — Clinical Alert Score (CAS) Engine
  Formula: CAS = 0.25·TR + 0.30·CC + 0.25·TS + 0.10·AE + 0.10·TC

  Uses real CIC IoMT 2024 medical metadata columns:
    cc_score         → CC dimension (1–10 in dataset → normalised 1–5)
    shift            → TC dimension (day / evening / night)
    time_sensitivity → TS base value (1–5 in dataset)

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
=============================================================
"""

# ── DIMENSION WEIGHTS ─────────────────────────────────────────────────────────
WEIGHTS = {
    "TR": 0.25,   # Threat Risk      — RF predict_proba confidence
    "CC": 0.30,   # Clinical Criticality — cc_score column (device type lookup)
    "TS": 0.25,   # Time Sensitivity — Isolation Forest + time_sensitivity col
    "AE": 0.10,   # Active Exploitation — known CVE / attack severity
    "TC": 0.10,   # Temporal Context — shift column (day/evening/night)
}

# ── ACTIVE EXPLOITATION TABLE — by attack label ───────────────────────────────
AE_TABLE = {
    "DoS_TCP":             5,   # Kills medical devices — highest severity
    "ARP_Spoofing":        4,   # MITM in IoMT network
    "MQTT_Publish_Flood":  4,   # Saturates IoMT broker
    "MQTT_Brute_Force":    4,   # Credential compromise on IoMT devices
    "Replay":              3,   # Known IoMT protocol weakness
    "Recon":               2,   # Precursor / information gathering
    "Benign":              0,
}

# ── SHIFT → TC SCORE ─────────────────────────────────────────────────────────
TC_TABLE = {
    "night":   5,   # Minimal staff — slowest response
    "evening": 3,   # Reduced staff
    "day":     1,   # Full staff — fastest response
}


def get_cc_score(cc_score_raw) -> float:
    """
    Normalise cc_score (1–10 in real dataset) → CAS CC dimension (1–5).
    Device mapping in real dataset:
      Admin PC=1, Lab Analyser=3, Nurse Workstation=3, X-Ray=4,
      MQTT Gateway=5, Pulse Oximeter=6, MRI Scanner=6,
      Cardiac Monitor=8, Infusion Pump=8,
      Defibrillator=9, ICU Ventilator=10
    """
    try:
        raw = float(cc_score_raw)
    except (TypeError, ValueError):
        raw = 5.0
    return round(max(1.0, min(5.0, raw / 2.0)), 2)


def get_tr_score(confidence: float) -> float:
    """Map RF predict_proba max confidence → TR dimension (1–5)."""
    if confidence >= 0.95: return 5.0
    if confidence >= 0.85: return 4.0
    if confidence >= 0.70: return 3.0
    if confidence >= 0.50: return 2.0
    return 1.0


def get_ts_score(iso_score: float, is_anomaly: bool, ts_col_val=3) -> float:
    """
    TS = Isolation Forest anomaly signal + dataset time_sensitivity column.
    Anomalous traffic always elevates TS; ts_col_val sets the floor.
    """
    ts_base = min(5, max(1, int(ts_col_val)))
    if is_anomaly:
        if iso_score < -0.25:
            return min(5.0, max(5.0, float(ts_base)))
        else:
            return min(5.0, max(4.0, float(ts_base)))
    else:
        return min(3.0, float(ts_base))


def get_ae_score(predicted_label: str) -> float:
    """Known attack CVE severity → AE dimension (0–5)."""
    return float(AE_TABLE.get(predicted_label, 1))


def get_tc_score(shift: str = "day") -> float:
    """Hospital shift → TC dimension (1–5)."""
    return float(TC_TABLE.get(str(shift).strip().lower(), 2))


def compute_cas(tr, cc, ts, ae, tc) -> float:
    """
    Raw CAS on 0–5 scale → multiplied by 2 → final 0–10 scale.
    Matches CVSS convention for familiarity with SOC teams.
    """
    raw = (WEIGHTS["TR"] * tr + WEIGHTS["CC"] * cc +
           WEIGHTS["TS"] * ts + WEIGHTS["AE"] * ae + WEIGHTS["TC"] * tc)
    return round(raw * 2, 2)


def get_action(cas: float, label: str) -> str:
    """Map CAS score → SOC action tier."""
    if label == "Benign":
        return "Monitor"
    if cas >= 8.0:
        return "Immediate"
    if cas >= 5.0:
        return "Investigate"
    return "Monitor"


def score_alert(
    predicted_label: str,
    confidence: float,
    iso_score: float,
    is_anomaly: bool,
    cc_score_raw=5,
    shift: str = "day",
    ts_col_val=3,
) -> dict:
    """
    Full CAS pipeline. Call this from train.py, test.py, and app.py.

    Args:
        predicted_label : RF predicted attack class
        confidence      : RF predict_proba max value
        iso_score       : Isolation Forest decision_function score
        is_anomaly      : True if IF predicted anomaly (-1)
        cc_score_raw    : cc_score column value from dataset (1–10)
        shift           : shift column value (day/evening/night)
        ts_col_val      : time_sensitivity column value (1–5)

    Returns dict with TR, CC, TS, AE, TC, CAS, action
    """
    tr  = get_tr_score(confidence)
    cc  = get_cc_score(cc_score_raw)
    ts  = get_ts_score(iso_score, is_anomaly, ts_col_val)
    ae  = get_ae_score(predicted_label)
    tc  = get_tc_score(shift)
    cas = compute_cas(tr, cc, ts, ae, tc)
    return {
        "TR": tr, "CC": cc, "TS": ts, "AE": ae, "TC": tc,
        "CAS": cas,
        "action": get_action(cas, predicted_label),
    }


# ── SELF-TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("  CAS Engine Self-Test | CAAP SLIIT | IT22061270")
    print("=" * 72)

    scenarios = [
        # label,               conf,   iso,   anom,  cc, shift,     ts
        ("DoS_TCP",            0.98,  -0.45, True,  10, "night",    5),
        ("ARP_Spoofing",       0.92,  -0.28, True,   8, "evening",  4),
        ("MQTT_Brute_Force",   0.81,  -0.12, True,   5, "night",    4),
        ("MQTT_Publish_Flood", 0.87,  -0.20, True,   8, "day",      4),
        ("Recon",              0.65,   0.05, False,  3, "evening",  3),
        ("Replay",             0.73,  -0.08, True,   6, "night",    4),
        ("Benign",             0.99,   0.22, False,  4, "day",      1),
    ]

    print(f"\n  {'Label':<22} {'CC':>4} {'Shift':>8}  "
          f"{'TR':>4} {'CC':>4} {'TS':>4} {'AE':>4} {'TC':>4}  {'CAS':>6}  Action")
    print("  " + "─" * 76)
    for lbl, conf, iso, anom, cc, sh, ts in scenarios:
        r = score_alert(lbl, conf, iso, anom, cc, sh, ts)
        print(f"  {lbl:<22} {cc:>4} {sh:>8}  "
              f"{r['TR']:>4.1f} {r['CC']:>4.1f} {r['TS']:>4.1f} "
              f"{r['AE']:>4.1f} {r['TC']:>4.1f}  {r['CAS']:>6.2f}  {r['action']}")
    print()
