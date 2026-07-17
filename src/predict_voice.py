print("✅ Script started...")

import os
import numpy as np
import librosa
import joblib
import sounddevice as sd
import soundfile as sf

print("✅ Libraries loaded successfully.")


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "voice_fatigue_model.joblib")
TEMP_AUDIO_PATH = os.path.join(BASE_DIR, "..", "data", "voice_samples", "temp_test.wav")

# ----------------------------
# Load Model
# ----------------------------
print("🔍 Loading trained voice fatigue model...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=500))
    energy = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfccs_mean, mfccs_std, pitch, energy])

# ----------------------------
# Record a new voice
# ----------------------------
def record_voice(duration=5, fs=44100):
    print(f"🎤 Recording for {duration} seconds... Speak normally or tiredly!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(TEMP_AUDIO_PATH, audio, fs)
    print(f"✅ Recording saved at: {TEMP_AUDIO_PATH}")

# ----------------------------
# Predict fatigue
# ----------------------------
def predict_fatigue():
    record_voice()  # record live input
    features = extract_features(TEMP_AUDIO_PATH).reshape(1, -1)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    label = "FATIGUED 😴" if pred == 1 else "NORMAL 😊"
    confidence = round(max(prob) * 100, 2)

    print("\n🧠 Voice Analysis Result:")
    print(f"   ➤ Prediction: {label}")
    print(f"   ➤ Confidence: {confidence}%")

if __name__ == "__main__":
    predict_fatigue()

