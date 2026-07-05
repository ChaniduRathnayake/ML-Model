"""
test_real_rows.py — Phase 6 validation helper
================================================
Your CICIoMT2024 test data is split into one CSV PER attack subtype
(e.g. data/test/ARP_Spoofing_test.pcap.csv, data/test/TCP_IP-DoS-SYN_test.pcap.csv),
with no label column inside — the label lives in the filename.

This script scans a directory of such files, samples a few real rows from
each, sends each to your running Flask /predict endpoint, and prints the
filename-derived label next to what the model actually predicted.

NOTE: your label_encoder.pkl / RF model may have been trained on a
CONSOLIDATED label set (e.g. 6 classes per your work plan: ARP_Spoofing,
Benign, DoS_TCP, MQTT_Brute_Force, MQTT_Publish_Flood, Recon) while these
filenames use a more granular ~21-subtype taxonomy (e.g. "TCP_IP-DoS-SYN",
"MQTT-DDoS-Connect_Flood"). The script prints both labels raw — you decide
whether they should map onto each other and whether that mapping was
applied consistently during training.

Usage:
    python test_real_rows.py --dir data/test --rows_per_file 1
    python test_real_rows.py --dir data/test --rows_per_file 2 --url http://127.0.0.1:5001/predict
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

# Extra fields /predict needs that aren't in the CSVs themselves.
# Edit these if you want to test different device/time scenarios.
EXTRA_FIELDS = {
    "device_type": "ICU Ventilator",
    "department": "ICU",
    "hour_of_day": 14,          # daytime, normal shift
    "cve_known_exploited": False,
}


def label_from_filename(path: Path) -> str:
    """Strip '_test.pcap.csv' (or similar) suffix to recover the attack label."""
    name = path.stem  # drops final .csv
    name = re.sub(r"\.pcap$", "", name)  # drops .pcap if present
    name = re.sub(r"_test$", "", name)   # drops trailing _test
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/test", help="Directory containing per-class test CSVs")
    parser.add_argument("--url", default="http://127.0.0.1:5001/predict", help="Flask /predict URL")
    parser.add_argument("--rows_per_file", type=int, default=1, help="How many rows to sample per file")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for test files")
    args = parser.parse_args()

    test_dir = Path(args.dir)
    files = sorted(test_dir.glob(args.pattern))
    if not files:
        print(f"[ERROR] No files matching '{args.pattern}' found in {test_dir}")
        sys.exit(1)

    print(f"[test] Found {len(files)} test files in {test_dir}")
    match_count = 0
    total_count = 0

    for f in files:
        true_label = label_from_filename(f)
        print(f"\n{'='*70}\nFile: {f.name}  ->  filename-derived label: {true_label}\n{'='*70}")

        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"[ERROR] Could not read {f.name}: {exc}")
            continue

        if df.empty:
            print("[skip] empty file")
            continue

        sample = df.sample(n=min(args.rows_per_file, len(df)), random_state=42)

        for _, row in sample.iterrows():
            payload = row.to_dict()
            payload.update(EXTRA_FIELDS)

            try:
                resp = requests.post(args.url, json=payload, timeout=15)
                resp.raise_for_status()
                result = resp.json()
                print(json.dumps(result, indent=2))

                predicted = result.get("label")
                total_count += 1
                is_match = str(predicted) == str(true_label)
                match_count += int(is_match)
                tag = "✓ MATCH" if is_match else "✗ DIFFERENT (check if this is a granular-vs-consolidated label issue, not necessarily wrong)"
                print(f"\n{tag}  filename_label={true_label}  predicted_label={predicted}  "
                      f"confidence={result.get('confidence')}  CAS={result.get('CAS')}  action={result.get('action')}")
            except requests.exceptions.RequestException as exc:
                print(f"[ERROR] Request failed: {exc}")

    print(f"\n\n{'='*70}\nSUMMARY: {match_count}/{total_count} exact label matches\n{'='*70}")
    print("Remember: a 'DIFFERENT' result may just mean your RF was trained on")
    print("consolidated classes while these filenames use granular subtypes —")
    print("check label_encoder.classes_ to see the exact label set RF predicts from.")


if __name__ == "__main__":
    main()
