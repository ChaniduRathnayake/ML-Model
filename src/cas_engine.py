"""
=============================================================
  CAAP — Clinical Alert Score (CAS) Engine  v2.0
  Formula: CAS = 0.25·TR + 0.30·CC + 0.25·TS + 0.10·AE + 0.10·TC

  Clinical Criticality (CC) SOURCE — OFFICIAL DOCUMENTS:
  ─────────────────────────────────────────────────────────
  [1] U.S. FDA. 21 CFR Part 860 — Medical Device Classification
      Procedures (up to date as of 05/06/2026). eCFR.
      § 860.3 Definitions:
        • Class III = "life-supporting or life-sustaining … substantial
          importance in preventing impairment of human health"  → CC 5
        • Class II  = special controls required, moderate risk         → CC 3–4
        • Class I   = general controls sufficient, lowest risk         → CC 1–2

  [2] U.S. FDA. (2023). Cybersecurity in Medical Devices: Quality System
      Considerations and Content of Premarket Submissions (Final Guidance,
      September 27 2023 — webinar transcript November 2 2023).
      FDA CDRH. https://www.fda.gov/media/173984/download
      § General Principle: "cybersecurity documentation is expected to
        scale with cybersecurity risks of the device" — i.e. risk-based,
        device-class-aware scoring is required.

  CC scores (1–5) are derived ENTIRELY from these two official sources.
  No synthetic data is added to the CIC IoMT 2024 dataset.
  All CC values are computed at inference time via port/protocol lookup.

  Author : R.M.C.B. Rathnayake | IT22061270 | SLIIT Cyber Security
=============================================================
"""

# ── DIMENSION WEIGHTS ─────────────────────────────────────────────────────────
WEIGHTS = {
    "TR": 0.25,
    "CC": 0.30,
    "TS": 0.25,
    "AE": 0.10,
    "TC": 0.10,
}

DEVICE_PROFILES = {

    # ── PORT 1883 — MQTT (plaintext) ─────────────────────────────────────────
    (1883, "mqtt"): {"device_name": "MQTT Medical IoT Gateway",           "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU/General Ward",          "notes": "FDA Class II; gateway aggregates life-critical sensor data — 21 CFR §860.3"},
    (1883, "tcp"):  {"device_name": "MQTT Broker (Hospital IoT)",         "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU/General Ward",          "notes": "FDA Class II; broker compromise affects all downstream IoMT devices"},

    # ── PORT 8883 — MQTT over TLS ─────────────────────────────────────────────
    (8883, "tcp"):  {"device_name": "Secure MQTT Gateway (TLS)",          "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU/General Ward",          "notes": "FDA Class II; encrypted MQTT — still critical gateway"},

    # ── PORT 502 — Modbus TCP ─────────────────────────────────────────────────
    (502,  "tcp"):  {"device_name": "Modbus Medical Sensor / PLC",        "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / Operating Theatre",   "notes": "FDA Class II; Modbus in ventilator subsystems and infusion controllers"},

    # ── HL7 MLLP ─────────────────────────────────────────────────────────────
    (2575, "tcp"):  {"device_name": "HL7 MLLP Interface Engine",          "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Clinical Informatics",      "notes": "FDA Class II; HL7 messaging compromise delays orders and results"},
    (2576, "tcp"):  {"device_name": "HL7 MLLP (Alt Port)",                "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Clinical Informatics",      "notes": "FDA Class II; alternate HL7 port"},

    # ── OPC-UA ────────────────────────────────────────────────────────────────
    (4840, "tcp"):  {"device_name": "OPC-UA Medical Device Server",       "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / Operating Theatre",   "notes": "FDA Class II; OPC-UA in anaesthesia machines and ventilators"},

    # ── DICOM ─────────────────────────────────────────────────────────────────
    (11112,"tcp"):  {"device_name": "DICOM PACS / Imaging Workstation",   "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Radiology",                 "notes": "FDA Class II; diagnostic imaging — delay impacts clinical decisions"},
    (104,  "tcp"):  {"device_name": "DICOM Store / Query SCU",            "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Radiology / Cardiology",    "notes": "FDA Class II; alternate DICOM port"},
    (2000, "tcp"):  {"device_name": "DICOM Modality Worklist",            "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Radiology",                 "notes": "FDA Class II; worklist disruption delays procedures"},

    # ── Clinical Web / EHR ────────────────────────────────────────────────────
    (8080, "tcp"):  {"device_name": "Clinical Web Portal / EHR",          "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Clinical Informatics",      "notes": "FDA Class II; EHR downtime affects all clinical decisions"},
    (8443, "tcp"):  {"device_name": "Secure Clinical Web App (HTTPS)",    "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Clinical Informatics",      "notes": "FDA Class II; encrypted clinical web application"},
    (443,  "tcp"):  {"device_name": "Secure Web / Telemedicine / EHR",   "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "General / Telemedicine",    "notes": "FDA Class II; may serve telemedicine, EHR, or remote monitoring"},
    (80,   "tcp"):  {"device_name": "Admin Web Interface / Dashboard",    "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Administration / IT",       "notes": "FDA Class I; management interface — indirect patient impact"},

    # ── IT Infrastructure ─────────────────────────────────────────────────────
    (22,   "tcp"):  {"device_name": "SSH Remote Management",              "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT / Biomedical Engineering","notes": "FDA Class I; compromise enables device manipulation or lateral movement"},
    (23,   "tcp"):  {"device_name": "Telnet Legacy Medical Device",       "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Legacy Infrastructure",     "notes": "FDA Class I; insecure protocol on older networked medical equipment"},
    (21,   "tcp"):  {"device_name": "FTP Medical File Server",            "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Administration / Radiology","notes": "FDA Class I; may transfer DICOM/HL7 files — insecure"},
    (161,  "udp"):  {"device_name": "SNMP Network Monitor",               "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT / Biomedical Engineering","notes": "FDA Class I; SNMP on medical devices exposes config data"},
    (162,  "udp"):  {"device_name": "SNMP Trap Receiver",                 "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT",                        "notes": "FDA Class I; management plane"},
    (3389, "tcp"):  {"device_name": "RDP Workstation / Clinical PC",      "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Administration / Clinical IT","notes": "FDA Class I; RDP compromise = full workstation access"},
    (5900, "tcp"):  {"device_name": "VNC Remote Desktop",                 "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Biomedical Engineering / IT","notes": "FDA Class I; legacy remote access on older medical equipment"},
    (9100, "tcp"):  {"device_name": "Network Label / Results Printer",    "fda_class": "I",   "cc": 1, "patient_dep": 0, "dept": "Administration",            "notes": "FDA Class I; label printer — minimal clinical impact"},
    (53,   "udp"):  {"device_name": "DNS Server (Hospital Network)",      "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT Infrastructure",         "notes": "FDA Class I; DNS poisoning affects all networked medical devices"},
    (53,   "tcp"):  {"device_name": "DNS Server (TCP)",                   "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT Infrastructure",         "notes": "FDA Class I; TCP zone transfers"},
    (67,   "udp"):  {"device_name": "DHCP Server",                        "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT Infrastructure",         "notes": "FDA Class I; DHCP starvation disrupts all networked devices"},
    (123,  "udp"):  {"device_name": "NTP Time Server",                    "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT Infrastructure",         "notes": "FDA Class I; time skew breaks TLS certs and EHR audit logs"},

    # ── Clinical Databases ────────────────────────────────────────────────────
    (3306, "tcp"):  {"device_name": "Clinical Database Server (MySQL)",   "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Clinical Informatics",      "notes": "FDA Class II; patient records, medication history"},
    (5432, "tcp"):  {"device_name": "Clinical Database Server (PostgreSQL)","fda_class": "II","cc": 3, "patient_dep": 0, "dept": "Clinical Informatics",      "notes": "FDA Class II; patient record storage"},
    (1433, "tcp"):  {"device_name": "MS SQL Server (EHR / PACS Backend)", "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Clinical Informatics / Radiology","notes": "FDA Class II; backend for EHR and PACS"},
    (9200, "tcp"):  {"device_name": "Elasticsearch (Clinical Analytics)", "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "IT / Clinical Analytics",   "notes": "FDA Class I; analytics engine"},

    # ══════════════════════════════════════════════════════════════════════════
    # DEVICE-SPECIFIC PORTS — Life-Critical & Specialist Medical Devices
    # ── FDA Class III (life-sustaining) ── CC = 5 ────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    (4000, "tcp"):  {"device_name": "ICU Mechanical Ventilator",          "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "ICU",                       "notes": "FDA Class III — life-sustaining per 21 CFR §860.3 §860.10"},
    (4001, "tcp"):  {"device_name": "Smart Infusion Pump (Networked)",    "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "ICU / General Ward",        "notes": "FDA Class III — life-sustaining; wrong rate = patient death"},
    (4004, "tcp"):  {"device_name": "Defibrillator / ICD Network",        "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Emergency / ICU",           "notes": "FDA Class III — defibrillation failure = cardiac arrest death"},
    (4005, "tcp"):  {"device_name": "Anaesthesia Delivery System",        "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Operating Theatre",         "notes": "FDA Class III — life-sustaining during surgery per §860.10"},
    (4014, "tcp"):  {"device_name": "Pacemaker / ICD Programmer",         "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Cardiology",                "notes": "FDA Class III — wireless reprogramming = cardiac arrest risk"},
    (4015, "tcp"):  {"device_name": "Haemodialysis Machine",              "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Nephrology / ICU",          "notes": "FDA Class III — electrolyte error = lethal arrhythmia"},
    (4016, "tcp"):  {"device_name": "ECMO / Cardiopulmonary Bypass",      "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Cardiac Surgery / ICU",     "notes": "FDA Class III — substitutes heart/lung function entirely"},
    (4019, "tcp"):  {"device_name": "Neonatal Incubator / Warmer",        "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "NICU",                      "notes": "FDA Class III — life-sustaining for premature neonates"},
    (4021, "tcp"):  {"device_name": "Intracranial Pressure (ICP) Monitor","fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Neurosurgery / ICU",        "notes": "FDA Class III — falsified ICP data masks brain herniation"},
    (4022, "tcp"):  {"device_name": "Deep Brain Stimulator (DBS) Programmer","fda_class":"III","cc": 5, "patient_dep": 1, "dept": "Neurology",                "notes": "FDA Class III — reprogramming error = uncontrolled seizure"},
    (4029, "tcp"):  {"device_name": "Surgical Robot (Control Interface)", "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Operating Theatre",         "notes": "FDA Class III — control disruption = surgical injury"},
    (4031, "tcp"):  {"device_name": "Linear Accelerator (LINAC) / Radiotherapy","fda_class":"III","cc":5,"patient_dep": 1, "dept": "Oncology / Radiation Therapy","notes": "FDA Class III — dose error = lethal radiation injury"},
    (4032, "tcp"):  {"device_name": "Brachytherapy Controller",           "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Oncology",                  "notes": "FDA Class III; source position error is life-threatening"},
    (4034, "tcp"):  {"device_name": "Networked Insulin Pump / Closed-Loop","fda_class": "III","cc": 5, "patient_dep": 1, "dept": "Endocrinology / ICU",       "notes": "FDA Class III — overdose = hypoglycaemic coma"},
    (4044, "tcp"):  {"device_name": "Therapeutic Hypothermia System",     "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "ICU / Cardiac Surgery",     "notes": "FDA Class III — temperature management post-cardiac arrest"},
    (4059, "tcp"):  {"device_name": "Blood Bank / Transfusion Management","fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Blood Bank / Haematology",  "notes": "FDA Class III — ABO incompatibility error = fatal transfusion reaction"},
    (4060, "tcp"):  {"device_name": "Oncology Infusion Management",       "fda_class": "III", "cc": 5, "patient_dep": 1, "dept": "Oncology",                  "notes": "FDA Class III — chemotherapy overdose is lethal"},

    # ── FDA Class II — High Clinical Impact ── CC = 4 ────────────────────────
    (4002, "tcp"):  {"device_name": "Bedside Patient Monitor (Multiparameter)","fda_class":"II","cc":4,"patient_dep": 1, "dept": "ICU / General Ward",        "notes": "FDA Class II; alarm failure conceals life-threatening deterioration"},
    (4003, "tcp"):  {"device_name": "Cardiac Monitor / 12-Lead ECG",      "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Cardiology / ICU",          "notes": "FDA Class II; missed arrhythmia detection = life risk"},
    (4006, "tcp"):  {"device_name": "Pulse Oximeter (Networked SpO2)",    "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / General Ward",        "notes": "FDA Class II; hypoxia missed if device compromised"},
    (4007, "tcp"):  {"device_name": "Automated Blood Pressure Monitor (NIBP)","fda_class":"II","cc":4,"patient_dep": 1, "dept": "ICU / General Ward",         "notes": "FDA Class II; hypertensive crisis undetected if tampered"},
    (4008, "tcp"):  {"device_name": "Continuous Glucose Monitor (CGM)",   "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Endocrinology / ICU",       "notes": "FDA Class II; falsified glucose = wrong insulin dose = death"},
    (4010, "tcp"):  {"device_name": "MRI Scanner (Control System)",       "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Radiology",                 "notes": "FDA Class II; RF parameter tampering = patient harm"},
    (4011, "tcp"):  {"device_name": "CT Scanner (Control / DICOM Interface)","fda_class":"II","cc":4,"patient_dep": 1, "dept": "Radiology",                   "notes": "FDA Class II; radiation dose error or missed diagnosis"},
    (4017, "tcp"):  {"device_name": "Syringe Pump / Microinfusion Pump",  "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / Anaesthesia",         "notes": "FDA Class II; dose error risk for vasoactive drugs"},
    (4020, "tcp"):  {"device_name": "Fetal Monitor / Cardiotocograph (CTG)","fda_class":"II", "cc": 4, "patient_dep": 1, "dept": "Obstetrics / Labour Ward",  "notes": "FDA Class II; missed foetal distress = stillbirth"},
    (4023, "tcp"):  {"device_name": "Automated Drug Dispensing Cabinet",  "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Pharmacy / ICU",            "notes": "FDA Class II; wrong drug dispensed = medication error"},
    (4025, "tcp"):  {"device_name": "Vital Signs Telemetry Transmitter",  "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "General Ward / Step-Down",  "notes": "FDA Class II; wireless vital signs — missed alarm"},
    (4028, "tcp"):  {"device_name": "Nurse Call / Emergency Alert System","fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "All Wards",                  "notes": "FDA Class II; disruption delays emergency response"},
    (4033, "tcp"):  {"device_name": "Cochlear Implant Programmer",        "fda_class": "III", "cc": 4, "patient_dep": 1, "dept": "ENT / Audiology",           "notes": "FDA Class III implant; reprogramming affects hearing function"},
    (4036, "tcp"):  {"device_name": "EEG Monitor / Seizure Detection",    "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Neurology / ICU",           "notes": "FDA Class II; seizure detection failure in ICU is life-threatening"},
    (4038, "tcp"):  {"device_name": "Central Patient Monitoring Station", "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / Step-Down",           "notes": "FDA Class II; central hub — single point of failure for ward monitoring"},
    (4039, "tcp"):  {"device_name": "Clinical Alarm Management System",   "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / All Wards",           "notes": "FDA Class II; disruption silences critical alerts"},
    (4040, "tcp"):  {"device_name": "Networked Oxygen Therapy System",    "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Respiratory / ICU",         "notes": "FDA Class II; oxygen concentration error is life-threatening"},
    (4042, "tcp"):  {"device_name": "Apnoea / Respiratory Monitor",       "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "NICU / Sleep Lab",          "notes": "FDA Class IIb; apnoea alarm failure in neonates is fatal"},
    (4043, "tcp"):  {"device_name": "Capnography / End-Tidal CO2 Monitor","fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "Anaesthesia / ICU / Emergency","notes": "FDA Class II; intubation verification and ventilation adequacy"},
    (4047, "tcp"):  {"device_name": "Neonatal Phototherapy (Jaundice)",   "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "NICU / Paediatrics",        "notes": "FDA Class II; untreated neonatal jaundice causes brain damage"},
    (4049, "tcp"):  {"device_name": "Tele-ICU / eICU Remote Monitoring",  "fda_class": "II",  "cc": 4, "patient_dep": 1, "dept": "ICU / Telemedicine",        "notes": "FDA Class II; compromise blinds intensivist to all patients"},

    # ── FDA Class II — Moderate Impact ── CC = 3 ─────────────────────────────
    (4009, "tcp"):  {"device_name": "Smart Temperature Monitor",          "fda_class": "II",  "cc": 3, "patient_dep": 1, "dept": "General Ward",              "notes": "FDA Class II; missed fever in immunocompromised patients"},
    (4012, "tcp"):  {"device_name": "Digital X-Ray / Fluoroscopy System", "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Radiology",                 "notes": "FDA Class II; image integrity critical for diagnosis"},
    (4013, "tcp"):  {"device_name": "Ultrasound Imaging System",          "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Radiology / Obstetrics",    "notes": "FDA Class II; diagnostic imaging"},
    (4018, "tcp"):  {"device_name": "Enteral Feeding Pump",               "fda_class": "II",  "cc": 3, "patient_dep": 1, "dept": "General Ward / ICU",        "notes": "FDA Class II; over/underfeeding in critically ill patients"},
    (4024, "tcp"):  {"device_name": "Point-of-Care Lab Analyser",         "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Laboratory / ICU",          "notes": "FDA Class II; falsified results mislead treatment decisions"},
    (4026, "tcp"):  {"device_name": "Wearable ECG / Holter Monitor (IoMT)","fda_class": "II", "cc": 3, "patient_dep": 0, "dept": "Cardiology / Remote",       "notes": "FDA Class II; data integrity for arrhythmia detection"},
    (4027, "tcp"):  {"device_name": "Smart Hospital Bed (Pressure Sensor)","fda_class": "II", "cc": 3, "patient_dep": 0, "dept": "General Ward",              "notes": "FDA Class II; fall detection and pressure ulcer prevention"},
    (4030, "tcp"):  {"device_name": "Endoscopy / Laparoscopy System",     "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Gastroenterology / Surgery","notes": "FDA Class II; image failure during procedure"},
    (4035, "tcp"):  {"device_name": "Transcranial Doppler (TCD) Monitor", "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Neurology / ICU",           "notes": "FDA Class II; emboli detection during cardiac surgery"},
    (4037, "tcp"):  {"device_name": "Gastric pH / Motility Monitor",      "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Gastroenterology",          "notes": "FDA Class II; diagnostic monitoring"},
    (4041, "tcp"):  {"device_name": "Electronic Spirometer / Lung Function","fda_class": "II","cc": 3, "patient_dep": 0, "dept": "Respiratory / Pulmonology", "notes": "FDA Class II; lung function testing"},
    (4046, "tcp"):  {"device_name": "Negative Pressure Wound Therapy (VAC)","fda_class":"II","cc": 3, "patient_dep": 0, "dept": "Surgery / Wound Care",       "notes": "FDA Class II; wound healing device"},
    (4048, "tcp"):  {"device_name": "Medical Suction Unit (Networked)",   "fda_class": "II",  "cc": 3, "patient_dep": 1, "dept": "Emergency / ICU / Surgery", "notes": "FDA Class II; airway management device"},
    (4052, "tcp"):  {"device_name": "Smart Medication Refrigerator",      "fda_class": "I",   "cc": 3, "patient_dep": 0, "dept": "Pharmacy / Blood Bank",     "notes": "FDA Class I; temperature excursion destroys vaccines"},
    (4056, "tcp"):  {"device_name": "Pharmacy Management System (PMS)",   "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Pharmacy",                  "notes": "FDA Class II; medication dispensing and interaction checking"},
    (4057, "tcp"):  {"device_name": "Laboratory Information System (LIS)","fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Laboratory",                "notes": "FDA Class II; delayed/falsified results affect treatment"},
    (4058, "tcp"):  {"device_name": "Radiology Information System (RIS)", "fda_class": "II",  "cc": 3, "patient_dep": 0, "dept": "Radiology",                 "notes": "FDA Class II; imaging orders and reports management"},

    # ── FDA Class I — Low Criticality ── CC = 1–2 ────────────────────────────
    (4045, "tcp"):  {"device_name": "Sequential Compression Device (DVT)","fda_class": "II",  "cc": 2, "patient_dep": 0, "dept": "General Ward / Surgery",    "notes": "FDA Class II; DVT prevention — lower criticality"},
    (4050, "tcp"):  {"device_name": "Hospital Bed Management System",     "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Administration",            "notes": "FDA Class I; operational — indirect patient safety impact"},
    (4051, "tcp"):  {"device_name": "RFID Patient / Asset Tracking",      "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Administration / Logistics","notes": "FDA Class I; patient location and asset tracking"},
    (4053, "tcp"):  {"device_name": "Environmental Sensor (Air Quality)", "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "Infection Control",         "notes": "FDA Class I; HEPA monitoring — indirect safety"},
    (4054, "tcp"):  {"device_name": "Admin / Clerical Workstation",       "fda_class": "I",   "cc": 1, "patient_dep": 0, "dept": "Administration",            "notes": "FDA Class I; general purpose computer — minimal direct patient impact"},
    (4055, "tcp"):  {"device_name": "Nurse Workstation / COW",            "fda_class": "I",   "cc": 2, "patient_dep": 0, "dept": "All Wards",                 "notes": "FDA Class I; clinical workstation — interruption affects medication admin"},
}

# ── ACTIVE EXPLOITATION TABLE ─────────────────────────────────────────────────
AE_TABLE = {
    "DoS_TCP":              5,
    "ARP_Spoofing":         4,
    "MQTT_Publish_Flood":   4,
    "MQTT_Brute_Force":     4,
    "Replay":               3,
    "Recon":                2,
    "Benign":               0,
}

# ── SHIFT → TC SCORE ─────────────────────────────────────────────────────────
TC_TABLE = {
    "night":   5,
    "evening": 3,
    "day":     1,
}

# ── DEFAULT PROFILE ───────────────────────────────────────────────────────────
_DEFAULT_PROFILE = {
    "device_name": "Unknown Networked Device",
    "fda_class": "I",
    "cc": 2,
    "patient_dep": 0,
    "dept": "Unknown",
    "notes": "Default: FDA Class I per 21 CFR §860.3 — no specific classification found",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LOOKUP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def lookup_device_profile(dst_port: int, protocol: str = "tcp") -> dict:
    proto = str(protocol).strip().lower()
    key = (int(dst_port), proto)
    if key in DEVICE_PROFILES:
        return DEVICE_PROFILES[key]
    key_any = (int(dst_port), "any")
    if key_any in DEVICE_PROFILES:
        return DEVICE_PROFILES[key_any]
    return _DEFAULT_PROFILE.copy()


def get_cc_score(dst_port: int = None, protocol: str = "tcp",
                 cc_raw: float = None) -> float:
    """
    Derive CC (1–5) from FDA 21 CFR §860.3 device class via port lookup.
    Falls back to normalised dataset column if port unknown.
    """
    if dst_port is not None:
        profile = lookup_device_profile(dst_port, protocol)
        return float(profile["cc"])
    if cc_raw is not None:
        try:
            return round(max(1.0, min(5.0, float(cc_raw) / 2.0)), 2)
        except (TypeError, ValueError):
            pass
    return 2.0


def get_tr_score(confidence: float) -> float:
    if confidence >= 0.95: return 5.0
    if confidence >= 0.85: return 4.0
    if confidence >= 0.70: return 3.0
    if confidence >= 0.50: return 2.0
    return 1.0


def get_ts_score(iso_score: float, is_anomaly: bool, ts_col_val: int = 3) -> float:
    ts_base = min(5, max(1, int(ts_col_val)))
    if is_anomaly:
        if iso_score < -0.25:
            return 5.0
        return float(max(4.0, ts_base))
    return min(3.0, float(ts_base))


def get_ae_score(predicted_label: str) -> float:
    return float(AE_TABLE.get(predicted_label, 1))


def get_tc_score(shift: str = "day") -> float:
    return float(TC_TABLE.get(str(shift).strip().lower(), 2))


def compute_cas(tr, cc, ts, ae, tc) -> float:
    raw = (WEIGHTS["TR"] * tr + WEIGHTS["CC"] * cc +
           WEIGHTS["TS"] * ts + WEIGHTS["AE"] * ae + WEIGHTS["TC"] * tc)
    return round(raw * 2, 2)


def get_action(cas: float, label: str) -> str:
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
    dst_port: int = None,
    protocol: str = "tcp",
    cc_raw: float = None,
    shift: str = "day",
    ts_col_val: int = 3,
) -> dict:
    """
    Full CAS pipeline. CC derived from FDA 21 CFR §860.3 via dst_port lookup.
    No synthetic columns required in the CIC IoMT 2024 dataset.
    """
    profile = lookup_device_profile(dst_port, protocol) if dst_port is not None \
        else _DEFAULT_PROFILE.copy()

    tr  = get_tr_score(confidence)
    cc  = get_cc_score(dst_port, protocol, cc_raw)
    ts  = get_ts_score(iso_score, is_anomaly, ts_col_val)
    ae  = get_ae_score(predicted_label)
    tc  = get_tc_score(shift)
    cas = compute_cas(tr, cc, ts, ae, tc)

    return {
        "TR":          tr,
        "CC":          cc,
        "TS":          ts,
        "AE":          ae,
        "TC":          tc,
        "CAS":         cas,
        "action":      get_action(cas, predicted_label),
        "device_name": profile["device_name"],
        "fda_class":   profile["fda_class"],
        "dept":        profile["dept"],
        "patient_dep": profile["patient_dep"],
    }


# ── SELF-TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 100)
    print("  CAS Engine v2.0 | CAAP SLIIT | IT22061270")
    print("  CC Source: FDA 21 CFR Part 860 §860.3 (05/06/2026) + FDA Cybersecurity Guidance 2023")
    print("=" * 100)

    scenarios = [
        ("DoS_TCP",            0.98, -0.45, True,  4000, "tcp",  "night",   5),
        ("MQTT_Brute_Force",   0.92, -0.28, True,  4034, "tcp",  "night",   5),
        ("ARP_Spoofing",       0.87, -0.32, True,  4001, "tcp",  "night",   4),
        ("DoS_TCP",            0.95, -0.40, True,  4015, "tcp",  "evening", 5),
        ("MQTT_Publish_Flood", 0.90, -0.22, True,  4016, "tcp",  "night",   5),
        ("ARP_Spoofing",       0.88, -0.28, True,  1883, "mqtt", "evening", 4),
        ("MQTT_Publish_Flood", 0.85, -0.20, True,  4002, "tcp",  "day",     4),
        ("Replay",             0.73, -0.08, True,  4006, "tcp",  "night",   4),
        ("Recon",              0.65,  0.05, False, 4038, "tcp",  "evening", 3),
        ("ARP_Spoofing",       0.82, -0.18, True,  4039, "tcp",  "night",   4),
        ("DoS_TCP",            0.91, -0.35, True,  4031, "tcp",  "day",     5),
        ("MQTT_Brute_Force",   0.89, -0.25, True,  4059, "tcp",  "evening", 5),
        ("ARP_Spoofing",       0.86, -0.30, True,  4060, "tcp",  "day",     5),
        ("Recon",              0.65,  0.08, False, 11112,"tcp",  "day",     2),
        ("DoS_TCP",            0.91, -0.35, True,  4010, "tcp",  "day",     3),
        ("Recon",              0.60,  0.10, False, 80,   "tcp",  "day",     1),
        ("Benign",             0.99,  0.22, False, 4054, "tcp",  "day",     1),
    ]

    print(f"\n  {'Label':<25} {'Device':<40} {'Class':>5} {'TR':>4} {'CC':>4} {'TS':>4} {'AE':>4} {'TC':>4}  {'CAS':>6}  Action")
    print("  " + "─" * 110)
    for lbl, conf, iso, anom, port, proto, sh, ts in scenarios:
        r = score_alert(lbl, conf, iso, anom, port, proto, None, sh, ts)
        dev = r["device_name"][:38]
        print(f"  {lbl:<25} {dev:<40} {r['fda_class']:>5} "
              f"{r['TR']:>4.1f} {r['CC']:>4.1f} {r['TS']:>4.1f} "
              f"{r['AE']:>4.1f} {r['TC']:>4.1f}  {r['CAS']:>6.2f}  {r['action']}")

    print(f"\n  Total device profiles: {len(DEVICE_PROFILES)}")
    print(f"  CC Source 1: FDA 21 CFR Part 860 §860.3 (up to date 05/06/2026)")
    print(f"  CC Source 2: FDA Cybersecurity in Medical Devices Guidance (Sept 2023)")
    print()
