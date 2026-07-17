import cv2
import mediapipe as mp
import numpy as np
from log_health_data import log_eye_data

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# We’ll use these landmarks for upper & lower eyelid (left & right)
# (indices from Mediapipe FaceMesh)
LEFT_UPPER = 159
LEFT_LOWER = 145
RIGHT_UPPER = 386
RIGHT_LOWER = 374

def compute_eye_opening(landmarks, upper_idx, lower_idx):
    """
    landmarks: list of 468 Mediapipe face landmarks
    upper_idx, lower_idx: integer indices of eyelid points
    Returns vertical distance (normalized, since Mediapipe y is 0–1)
    """
    upper = landmarks[upper_idx].y
    lower = landmarks[lower_idx].y
    return abs(lower - upper)  # bigger = more open, smaller = more drooped

def run_ptosis_detection():
    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("👁 Ptosis monitoring started…")
    print("   ➤ Look at the camera for ~15–20 seconds")
    print("   ➤ Press 'q' to stop test")

    eye_open_values = []  # store average eye opening per frame

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status_text = "No face detected"

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Compute simple "eye opening" for each eye
                left_open = compute_eye_opening(face_landmarks, LEFT_UPPER, LEFT_LOWER)
                right_open = compute_eye_opening(face_landmarks, RIGHT_UPPER, RIGHT_LOWER)

                avg_open = (left_open + right_open) / 2.0
                eye_open_values.append(avg_open)

                # Simple live threshold (you can tune 0.020)
                if avg_open < 0.020:
                    status_text = "Drooping detected ⚠️"
                    color = (0, 0, 255)
                else:
                    status_text = "Eyes OK 🙂"
                    color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"EAR approx: {avg_open:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            cv2.imshow("Ptosis / Eyelid Drooping Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(eye_open_values) == 0:
        print("❌ No valid eye data recorded.")
        return

    avg_ear = float(np.mean(eye_open_values))
    print("\n🔍 EYE SUMMARY")
    print(f"Average eye opening (EAR-like): {avg_ear:.4f}")

    # Decide final status based on average EAR-like value
    # You can tweak thresholds after testing on yourself
    if avg_ear < 0.018:
        status = "EYE_DROOPING"
        print("FINAL STATUS: EYE DROOPING / FATIGUED 😴")
    elif avg_ear < 0.022:
        status = "EYE_BORDERLINE"
        print("FINAL STATUS: BORDERLINE 🟠")
    else:
        status = "EYE_STABLE"
        print("FINAL STATUS: STABLE 😊")

    # Log to CSV/HTML
    log_eye_data(avg_ear, status)
    print("✅ Eye data logged into eye_log.csv / eye_log.html")

if __name__ == "__main__":
    run_ptosis_detection()
