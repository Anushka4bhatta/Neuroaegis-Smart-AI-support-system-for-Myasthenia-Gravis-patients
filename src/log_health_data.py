import os
import pandas as pd
from datetime import datetime

# ------------------------------------------------------
# Base paths (common "data" folder for all logs)
# ------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --------- Module A: Fatigue (numeric) ---------
PATIENT_CSV_PATH = os.path.join(DATA_DIR, "patient_log.csv")
PATIENT_HTML_PATH = os.path.join(DATA_DIR, "patient_log.html")

# --------- Module B: Voice / Dysarthria ---------
VOICE_CSV_PATH = os.path.join(DATA_DIR, "voice_log.csv")
VOICE_HTML_PATH = os.path.join(DATA_DIR, "voice_log.html")

# --------- Module C: Eye / Ptosis ---------------
EYE_CSV_PATH = os.path.join(DATA_DIR, "eye_log.csv")
EYE_HTML_PATH = os.path.join(DATA_DIR, "eye_log.html")

# --------- MASTER: Central combined log ----------
CENTRAL_CSV_PATH = os.path.join(DATA_DIR, "central_log.csv")
CENTRAL_HTML_PATH = os.path.join(DATA_DIR, "central_log.html")


# ======================================================
# 1) MODULE A — Fatigue (sleep, stress, muscle strength)
#    → starts a NEW central row (session)
# ======================================================
def log_patient_data(sleep_hours, stress_level, muscle_strength, prediction):
    """
    Logs numeric fatigue-related data into patient_log.csv
    AND starts a new row in central_log.csv (one assessment session).
    """

    # -------- write to own fatigue log --------
    entry = {
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sleep Hours": sleep_hours,
        "Stress Level": stress_level,
        "Muscle Strength": muscle_strength,
        "Prediction": prediction,   # e.g. "FATIGUED" / "STABLE"
    }

    df = pd.DataFrame([entry])

    if not os.path.exists(PATIENT_CSV_PATH):
        df.to_csv(PATIENT_CSV_PATH, index=False)
    else:
        df.to_csv(PATIENT_CSV_PATH, mode="a", index=False, header=False)

    df_all = pd.read_csv(PATIENT_CSV_PATH)
    df_all.to_html(PATIENT_HTML_PATH, index=False)

    print("✅ Fatigue data logged!")
    print(f"   → CSV:  {PATIENT_CSV_PATH}")
    print(f"   → HTML: {PATIENT_HTML_PATH}")

    # -------- also create a NEW row in central log --------
    central_entry = {
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sleep Hours": sleep_hours,
        "Stress Level": stress_level,
        "Muscle Strength": muscle_strength,
        "Fatigue Prediction": prediction,
        "Voice Status": "",
        "Average EAR": "",
        "Eye Status": "",
    }

    cdf = pd.DataFrame([central_entry])

    if not os.path.exists(CENTRAL_CSV_PATH):
        cdf.to_csv(CENTRAL_CSV_PATH, index=False)
    else:
        cdf.to_csv(CENTRAL_CSV_PATH, mode="a", index=False, header=False)

    cdf_all = pd.read_csv(CENTRAL_CSV_PATH)
    cdf_all.to_html(CENTRAL_HTML_PATH, index=False)

    print("📘 Central log: NEW SESSION row created.")
    print(f"   → CSV:  {CENTRAL_CSV_PATH}")
    print(f"   → HTML: {CENTRAL_HTML_PATH}")


# ======================================================
# 2) MODULE B — Voice Fatigue / Dysarthria
#    → updates LAST row's Voice Status in central_log.csv
# ======================================================
def log_voice_data(status, jitter_first, jitter_last,
                   shimmer_first, shimmer_last, energy_ratio):
    """
    Logs acoustic voice fatigue features into voice_log.csv
    AND updates the last row in central_log.csv with Voice Status.
    """

    # -------- write to own voice log --------
    entry = {
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Jitter First": jitter_first,
        "Jitter Last": jitter_last,
        "Δ Jitter": jitter_last - jitter_first,
        "Shimmer First": shimmer_first,
        "Shimmer Last": shimmer_last,
        "Δ Shimmer": shimmer_last - shimmer_first,
        "Energy Ratio (last/first)": energy_ratio,
        "Voice Status": status,
    }

    df = pd.DataFrame([entry])

    if not os.path.exists(VOICE_CSV_PATH):
        df.to_csv(VOICE_CSV_PATH, index=False)
    else:
        df.to_csv(VOICE_CSV_PATH, mode="a", index=False, header=False)

    df_all = pd.read_csv(VOICE_CSV_PATH)
    df_all.to_html(VOICE_HTML_PATH, index=False)

    print("🎤 Voice data logged!")
    print(f"   → CSV:  {VOICE_CSV_PATH}")
    print(f"   → HTML: {VOICE_HTML_PATH}")

    # -------- update Voice Status in LAST central row --------
    if os.path.exists(CENTRAL_CSV_PATH):
        cdf = pd.read_csv(CENTRAL_CSV_PATH)
        if len(cdf) > 0:
            last_idx = cdf.index[-1]
            cdf.loc[last_idx, "Voice Status"] = status
            cdf.to_csv(CENTRAL_CSV_PATH, index=False)
            cdf.to_html(CENTRAL_HTML_PATH, index=False)
            print("📘 Central log: Voice Status updated in last session row.")
        else:
            print("⚠ Central log empty, could not update voice status.")
    else:
        print("⚠ Central log missing, could not update voice status.")


# ======================================================
# 3) MODULE C — Eye / Ptosis (EAR-based)
#    → updates LAST row's Eye fields in central_log.csv
# ======================================================
def log_eye_data(avg_ear, status):
    """
    Logs eyelid openness (EAR-like average) and classification
    into eye_log.csv AND updates last row in central_log.csv.
    """

    # -------- write to own eye log --------
    entry = {
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Average EAR": avg_ear,
        "Eye Status": status,
    }

    df = pd.DataFrame([entry])

    if not os.path.exists(EYE_CSV_PATH):
        df.to_csv(EYE_CSV_PATH, index=False)
    else:
        df.to_csv(EYE_CSV_PATH, mode="a", index=False, header=False)

    df_all = pd.read_csv(EYE_CSV_PATH)
    df_all.to_html(EYE_HTML_PATH, index=False)

    print("👁 Eye data logged!")
    print(f"   → CSV:  {EYE_CSV_PATH}")
    print(f"   → HTML: {EYE_HTML_PATH}")

    # -------- update Eye fields in LAST central row --------
    if os.path.exists(CENTRAL_CSV_PATH):
        cdf = pd.read_csv(CENTRAL_CSV_PATH)
        if len(cdf) > 0:
            last_idx = cdf.index[-1]
            cdf.loc[last_idx, "Average EAR"] = avg_ear
            cdf.loc[last_idx, "Eye Status"] = status
            cdf.to_csv(CENTRAL_CSV_PATH, index=False)
            cdf.to_html(CENTRAL_HTML_PATH, index=False)
            print("📘 Central log: Eye fields updated in last session row.")
        else:
            print("⚠ Central log empty, could not update eye status.")
    else:
        print("⚠ Central log missing, could not update eye status.")


# ======================================================
# Optional quick test when run directly
# ======================================================
if __name__ == "__main__":
    # This will just log a test fatigue entry (and new central row)
    log_patient_data(6, 8, 4.5, "FATIGUED")
