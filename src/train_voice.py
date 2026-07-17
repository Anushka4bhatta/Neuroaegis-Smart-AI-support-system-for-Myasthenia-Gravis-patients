import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "voice_samples")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "voice_fatigue_model.joblib")

# ----------------------------
# Helper function to extract features
# ----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)

    # Extract pitch and energy
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=500))
    energy = np.mean(librosa.feature.rms(y=y))

    # Combine all features into one array
    return np.hstack([mfccs_mean, mfccs_std, pitch, energy])


# ----------------------------
# Load data
# ----------------------------
X, y = [], []

for label, folder in enumerate(["normal", "fatigued"]):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(folder_path, file))
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples.")

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# Save model
# ----------------------------
os.makedirs(os.path.join(BASE_DIR, "..", "models"), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
