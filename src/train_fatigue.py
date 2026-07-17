import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Path to save trained model
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "fatigue_detector.joblib")

# --------------------------   
#  Synthetic Dataset (for MG)
# --------------------------
# Features chosen based on MG symptoms:
#  - sleep_hours: less sleep -> more fatigue
#  - stress_level: higher stress worsens symptoms
#  - muscle_strength: key indicator for MG patients

N = 400
sleep_hours = np.clip(np.random.normal(7, 1.5, N), 2, 10)
stress_level = np.random.randint(1, 10, N)
muscle_strength = np.clip(np.random.normal(7, 2, N), 2, 10)

# Label rule for fatigue (1 = fatigued, 0 = stable)
y = ((sleep_hours < 6) & (stress_level >= 6)) | (muscle_strength < 5)
y = y.astype(int)     

# Add some noise to simulate real-world variation
noise = np.random.rand(N) < 0.1
y = np.where(noise, 1 - y, y)

X = pd.DataFrame({
    "sleep_hours": sleep_hours,
    "stress_level": stress_level,
    "muscle_strength": muscle_strength
})

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --------------------------
# Train Model
# --------------------------
clf = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
)
clf.fit(X_train, y_train)

# --------------------------
# Evaluation
# --------------------------
y_pred = clf.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# Save Model
# --------------------------
joblib.dump(clf, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
