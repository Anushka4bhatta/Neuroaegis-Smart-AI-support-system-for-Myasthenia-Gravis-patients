# src/server.py
import os
import sys
import subprocess
import csv
import tempfile
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional robust imports for image/audio processing
import cv2
import numpy as np

# Mediapipe for ptosis processing
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# Import your log helper (must be in src/)
# server.py is in src/, so relative import by name works if src is in sys.path
# If not, we can append BASE_DIR to sys.path (done below).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../neuroaegis/src
ROOT_DIR = os.path.dirname(BASE_DIR)                    # .../neuroaegis
PYTHON_EXE = sys.executable                             # current python env

sys.path.insert(0, BASE_DIR)  # allow importing modules from src/

try:
    from log_health_data import log_eye_data
except Exception:
    # If import fails, define dummy logger to prevent crash.
    def log_eye_data(avg_ear, status):
        print("log_eye_data unavailable; got", avg_ear, status)

# If your voice analyzer is in predict_voice_dysarthria.py and exposes analyze_dysarthria()
try:
    import predict_voice_dysarthria as pv_module
    ANALYZE_VOICE_AVAILABLE = hasattr(pv_module, "analyze_dysarthria")
except Exception:
    pv_module = None
    ANALYZE_VOICE_AVAILABLE = False

# ---------------------------
# Data paths (centralized)
# ---------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PATIENT_LOG = os.path.join(DATA_DIR, "patient_log.csv")
CENTRAL_LOG = os.path.join(DATA_DIR, "central_log.csv")
VOICE_LOG = os.path.join(DATA_DIR, "voice_log.csv")
EYE_LOG = os.path.join(DATA_DIR, "eye_log.csv")

app = Flask(__name__)
CORS(app)   # allow cross origin calls (e.g., Live Server)

# ------------------------------------------------
# Helper: run a python script silently (nonblocking)
# ------------------------------------------------
def run_script(path_inside_src):
    script_path = os.path.join(BASE_DIR, path_inside_src)
    if not os.path.exists(script_path):
        return subprocess.CompletedProcess(args=[PYTHON_EXE, script_path], returncode=1, stdout="", stderr=f"Script not found: {script_path}")
    result = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True)
    return result

# ------------------------------------------------
# Helper: read last row from a CSV as dict
# ------------------------------------------------
def read_last_row(csv_path):
    if not os.path.exists(csv_path):
        return None
    last_row = None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    return last_row

# ------------------------------------------------
# Helper: Mediapipe frame -> EAR-like measure
# ------------------------------------------------
if MP_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    # landmark indices used earlier in your code
    LEFT_UPPER = 159
    LEFT_LOWER = 145
    RIGHT_UPPER = 386
    RIGHT_LOWER = 374

    def compute_eye_opening(landmarks, upper_idx, lower_idx):
        upper = landmarks[upper_idx].y
        lower = landmarks[lower_idx].y
        return abs(lower - upper)

    def analyze_frame_for_ear(image_bgr):
        """
        image_bgr: numpy array BGR
        returns: avg_ear(float) or None
        """
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            left_open = compute_eye_opening(landmarks, LEFT_UPPER, LEFT_LOWER)
            right_open = compute_eye_opening(landmarks, RIGHT_UPPER, RIGHT_LOWER)
            avg_open = float((left_open + right_open) / 2.0)
            return avg_open
else:
    def analyze_frame_for_ear(image_bgr):
        raise RuntimeError("mediapipe not available on server environment")


# ================== API: FATIGUE ==================
@app.post("/api/fatigue")
def api_fatigue():
    """
    Preferred flow:
     - If client POSTs JSON { sleep_hours, stress_level, muscle_strength } we could
       call the same Python function or script that logs and writes central_log.csv.
     - For now, keep backward-compatible: run predict_fatigue.py (script) to create logs,
       then read last central_log row and return to frontend.
    """
    # If client sends JSON values, we can save them into patient_log by calling your script
    # or by adding a dedicated function. We use existing script for simplicity.
    # Optional: accept JSON body (future-proof)
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    # if you want to pass the values to your script, change predict_fatigue.py to accept CLI args.
    result = run_script("predict_fatigue.py")

    if result.returncode != 0:
        # debugging output available: result.stderr / result.stdout
        return jsonify({"ok": False, "message": "Failed to run fatigue module", "stderr": result.stderr}), 500

    log_path = CENTRAL_LOG if os.path.exists(CENTRAL_LOG) else PATIENT_LOG
    row = read_last_row(log_path)
    if not row:
        return jsonify({"ok": False, "message": "No fatigue log data found."}), 500

    sleep = row.get("Sleep Hours") or row.get("Sleep") or ""
    stress = row.get("Stress Level") or row.get("Stress") or ""
    muscle = row.get("Muscle Strength") or row.get("Muscle") or ""
    prediction = row.get("Fatigue Prediction") or row.get("Prediction") or ""

    summary = f"Sleep: {sleep}, Stress: {stress}, Muscle: {muscle}, Status: {prediction}"
    return jsonify({
        "ok": True,
        "type": "fatigue",
        "time": row.get("Date & Time"),
        "sleep": sleep,
        "stress": stress,
        "muscle": muscle,
        "status": prediction,
        "summary": summary,
    })


# ================== API: VOICE (file upload) ==================
@app.post("/api/voice")
def api_voice_upload():
    """
    Expects a multipart/form-data POST with field 'audio' containing the audio file (wav/webm).
    Saves to a temp file and calls analyze_dysarthria(file_path) from your predict_voice module.
    Returns JSON summary & voice-log last row.
    """
    if "audio" not in request.files:
        return jsonify({"ok": False, "message": "Missing 'audio' file in form-data."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"ok": False, "message": "Empty audio file name."}), 400

    # Save to a temp WAV path
    tmp_dir = os.path.join(DATA_DIR, "temp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "temp_voice.wav")
    audio_file.save(tmp_path)

    # Call analyze function if available
    if ANALYZE_VOICE_AVAILABLE:
        try:
            status = pv_module.analyze_dysarthria(tmp_path)
        except Exception as e:
            return jsonify({"ok": False, "message": f"Voice analyze failed: {e}"}), 500
    else:
        # fallback: run script (will record from mic — not ideal when file provided)
        result = run_script("predict_voice_dysarthria.py")
        if result.returncode != 0:
            return jsonify({"ok": False, "message": "Voice script failed", "stderr": result.stderr}), 500

    # Read last row from voice log
    row = read_last_row(VOICE_LOG)
    if not row:
        return jsonify({"ok": False, "message": "No voice log data found."}), 500

    summary = (
        f"Jitter: {row.get('Jitter First')} → {row.get('Jitter Last')}, "
        f"Shimmer: {row.get('Shimmer First')} → {row.get('Shimmer Last')}, "
        f"Energy Ratio: {row.get('Energy Ratio (last/first)')}, "
        f"Status: {row.get('Voice Status')}"
    )

    return jsonify({
        "ok": True,
        "type": "voice",
        "time": row.get("Date & Time"),
        "jitter_first": row.get("Jitter First"),
        "jitter_last": row.get("Jitter Last"),
        "shimmer_first": row.get("Shimmer First"),
        "shimmer_last": row.get("Shimmer Last"),
        "energy_ratio": row.get("Energy Ratio (last/first)"),
        "status": row.get("Voice Status"),
        "summary": summary,
    })


# ================== API: PTOSIS (frame upload) ==================
@app.post("/api/ptosis")
def api_ptosis_frame():
    """
    Expects multipart/form-data with field 'frame' (image blob, jpeg/png).
    Uses Mediapipe to compute an EAR-like value and logs via log_eye_data().
    Frontend should capture a single frame (or multiple frames and send the last/average).
    """
    if not MP_AVAILABLE:
        return jsonify({"ok": False, "message": "Mediapipe not available on this environment."}), 500

    if "frame" not in request.files:
        return jsonify({"ok": False, "message": "Missing 'frame' file in form-data."}), 400

    frame_file = request.files["frame"]
    data = frame_file.read()
    if not data:
        return jsonify({"ok": False, "message": "Empty frame data."}), 400

    # Load image from bytes
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "message": "Failed to decode image."}), 400

    # Analyze single frame
    try:
        avg_ear = analyze_frame_for_ear(img)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Frame analyze error: {e}"}), 500

    if avg_ear is None:
        return jsonify({"ok": False, "message": "No face found in frame."}), 400

    # Decide status by thresholds (same logic as predict_ptosis.py)
    try:
        if avg_ear < 0.018:
            status = "EYE_DROOPING"
        elif avg_ear < 0.022:
            status = "EYE_BORDERLINE"
        else:
            status = "EYE_STABLE"
    except Exception:
        status = "EYE_UNKNOWN"

    # Log using your helper (will update central CSV)
    try:
        log_eye_data(avg_ear, status)
    except Exception as e:
        # still return result even if logging fails
        return jsonify({"ok": True, "avg_ear": avg_ear, "status": status, "warning": f"Logging failed: {e}"}), 200

    summary = f"EAR: {avg_ear:.4f}, Status: {status}"
    return jsonify({"ok": True, "type": "ptosis", "avg_ear": avg_ear, "status": status, "summary": summary})


# ================== API: CENTRAL LOG LATEST ==================
@app.get("/api/central-log-latest")
def api_central_log_latest():
    row = read_last_row(CENTRAL_LOG)
    if not row:
        return jsonify({"ok": False, "message": "No central log found."}), 404
    return jsonify({"ok": True, "row": row})


# ================== RUN FLASK ==================
if __name__ == "__main__":
    print("Starting NeuroAegis API server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
