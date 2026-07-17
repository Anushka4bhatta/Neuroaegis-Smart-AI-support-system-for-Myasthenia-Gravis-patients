import os
import joblib
import numpy as np
import random
import datetime
from log_health_data import log_patient_data

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fatigue_detector.joblib")
clf = joblib.load(MODEL_PATH)

# -------------------------------
# 🧠 Smart Automatic Data Collection
# -------------------------------
def auto_collect_data():
    # Simulate realistic conditions
    current_hour = datetime.datetime.now().hour

    # Sleep hours estimated based on time of day
    if current_hour < 8:
        sleep_hours = random.uniform(4.0, 6.0)  # maybe didn’t sleep enough
    else:
        sleep_hours = random.uniform(6.5, 8.5)  # had enough rest

    # Randomly simulate stress (AI can later take input from voice tone)
    stress_level = random.randint(3, 9)

    # Simulate muscle strength (fluctuates with stress & sleep)
    muscle_strength = round(random.uniform(4.0, 9.0), 1)

    return round(sleep_hours, 1), stress_level, muscle_strength


# -------------------------------
# 🧠 Fatigue Prediction
# -------------------------------
def predict_fatigue_auto():
    sleep_hours, stress_level, muscle_strength = auto_collect_data()

    X = np.array([[sleep_hours, stress_level, muscle_strength]])
    y = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0, 1]

    status = "FATIGUED 😴" if y == 1 else "STABLE 💪"

    print("\n🧠 Automatic Fatigue Analysis")
    print(f"   ➤ Sleep Hours: {sleep_hours}")
    print(f"   ➤ Stress Level: {stress_level}")
    print(f"   ➤ Muscle Strength: {muscle_strength}")
    print(f"   ➤ Status: {status} | Confidence: {round(prob * 100, 2)}%")

    # Log the automatically generated health data
    log_patient_data(sleep_hours, stress_level, muscle_strength, status)


# -------------------------------
# Run the module
# -------------------------------
if __name__ == "__main__":
    predict_fatigue_auto()
