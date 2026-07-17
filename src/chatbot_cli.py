import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CENTRAL_CSV_PATH = os.path.join(DATA_DIR, "central_log.csv")


def load_latest_session():
    """Load the last row from central_log.csv (latest assessment)."""
    if not os.path.exists(CENTRAL_CSV_PATH):
        print("⚠ No central_log.csv found. Run tests first (fatigue/voice/eye).")
        return None

    df = pd.read_csv(CENTRAL_CSV_PATH)
    if df.empty:
        print("⚠ central_log.csv is empty.")
        return None

    return df.iloc[-1]  # last row


def compute_risk_level(row):
    """Compute a simple MG risk label based on fatigue, voice, and eye status."""
    score = 0

    fatigue = str(row.get("Fatigue Prediction", "")).upper()
    voice = str(row.get("Voice Status", "")).upper()
    eye = str(row.get("Eye Status", "")).upper()

    if "FATIGUED" in fatigue:
        score += 2
    elif "STABLE" not in fatigue and fatigue != "":
        score += 1

    if "FATIGUED" in voice:
        score += 2
    elif "BORDERLINE" in voice:
        score += 1

    if "DROOP" in eye:
        score += 2
    elif "BORDERLINE" in eye:
        score += 1

    if score >= 4:
        return "HIGH RISK", score
    elif score >= 2:
        return "MODERATE RISK", score
    else:
        return "LOW RISK", score


def answer_question(user_msg, row):
    """MG-aware chatbot that can talk more naturally."""
    user = user_msg.lower()

    sleep = row.get("Sleep Hours", None)
    stress = row.get("Stress Level", None)
    muscle = row.get("Muscle Strength", None)
    fatigue = str(row.get("Fatigue Prediction", "Unknown"))
    voice = str(row.get("Voice Status", "Unknown"))
    ear = row.get("Average EAR", None)
    eye_status = str(row.get("Eye Status", "Unknown"))

    risk_label, risk_score = compute_risk_level(row)

    # ---------- Small-talk & emotional support ----------
    if any(g in user for g in ["hi", "hello", "hey"]):
        return (
            "Hi! I’m your NeuroAegis MG support assistant 🌸\n"
            "I can talk to you about your fatigue, voice, eye drooping, risk level, "
            "or just help you understand what’s going on."
        )

    if "how are you" in user:
        return "I’m here and fully focused on you 💙 Tell me how you’re feeling."

    if any(w in user for w in ["scared", "afraid", "anxious", "worried"]):
        return (
            "I’m really sorry that you’re feeling this way. MG can be scary and exhausting sometimes.\n"
            "You’re not alone in this. I can help you understand your latest health status, "
            "but for emotional support, talking to a friend, family member, or counselor can really help too. 💙"
        )

    if any(w in user for w in ["sad", "depressed", "alone", "lonely"]):
        return (
            "I hear you. Living with a chronic condition can feel very lonely sometimes.\n"
            "Your feelings are valid. Please remember you deserve support, rest, and kindness.\n"
            "If these feelings are very strong or lasting many days, it’s important to talk to a mental health professional too."
        )

    # ---------- 1) Overall summary ----------
    if "summary" in user or ("how" in user and "overall" in user) or "report" in user:
        return (
            f"Here is your latest MG health summary:\n"
            f"• Sleep: {sleep} hours\n"
            f"• Stress level: {stress}/10\n"
            f"• Muscle strength score: {muscle}\n"
            f"• Fatigue status: {fatigue}\n"
            f"• Voice status: {voice}\n"
            f"• Eye status: {eye_status}\n"
            f"• Overall risk level: {risk_label} (score {risk_score})\n\n"
            "If you feel worse than usual (breathing difficulty, trouble swallowing, or double vision), "
            "please contact your neurologist or nearest hospital immediately."
        )

    # ---------- 2) Fatigue / weakness ----------
    if any(w in user for w in ["tired", "fatigue", "weak", "energy", "exhausted"]):
        return (
            f"Your latest fatigue status is: {fatigue}.\n"
            f"You reported sleeping about {sleep} hours with stress level {stress}/10.\n"
            "In Myasthenia Gravis, less sleep and high stress can strongly worsen muscle weakness.\n"
            "Try to rest, avoid over-exertion, keep cool, and space out activities. "
            "If weakness suddenly increases or affects breathing/swallowing, please seek urgent medical help."
        )

    # ---------- 3) Voice / speech ----------
    if any(w in user for w in ["voice", "talk", "speaking", "speech", "slurred"]):
        return (
            f"Your last voice assessment result was: {voice}.\n"
            "This is based on how stable your pitch and loudness were over time.\n"
            "If your voice sounds nasal, slurred, or very weak—especially at the end of talking—it may indicate bulbar muscle fatigue.\n"
            "If this is getting worse or affects eating / swallowing, please inform your doctor."
        )

    # ---------- 4) Eye / ptosis / vision ----------
    if any(w in user for w in ["eye", "vision", "droop", "ptosis", "lid"]):
        return (
            f"Your last eye status was: {eye_status}.\n"
            f"The system measured your eyelid opening (EAR ≈ {ear}).\n"
            "Drooping eyelids (ptosis) and double vision are common in ocular Myasthenia Gravis.\n"
            "If your drooping becomes severe or you see double, avoid driving and seek medical advice."
        )

    # ---------- 5) Risk / seriousness ----------
    if any(w in user for w in ["risk", "danger", "serious", "critical", "emergency"]):
        return (
            f"Your current combined MG risk level (based on fatigue, voice and eye) is: {risk_label} (score {risk_score}).\n"
            "This tool only gives a rough indication.\n"
            "If you are experiencing breathing difficulty, swallowing trouble, or rapidly worsening weakness, "
            "that is an EMERGENCY situation. Please go to the nearest hospital or contact your neurologist immediately."
        )

    # ---------- 6) Doctor / hospital / help ----------
    if any(w in user for w in ["doctor", "hospital", "help", "neuro", "treatment"]):
        return (
            "This assistant can help you track MG-related fatigue, voice, and eye changes.\n"
            "But I’m not a replacement for a doctor.\n"
            "Based on your latest readings, your overall risk is "
            f"{risk_label}. If you feel unwell or your symptoms are progressing, "
            "please contact your neurologist or visit a hospital with Neurology / MG care.\n"
            "For regular care, keep a symptom diary, take medicines on time, and avoid over-exertion."
        )

    # ---------- Default: general supportive reply ----------
       # ---------- Default: gentle short reply ----------
    return (
        "I’m here for MG-related support 💙\n"
        "You can ask me about your fatigue, voice, eyes, risk level, or say "
        "\"give me a summary\"."
    )



def main():
    print("🧠 NeuroAegis – MG Support Chatbot")
    print("(Console version)\n")
    print("Loading your latest health log...\n")

    row = load_latest_session()
    if row is None:
        return

    print("✅ Latest session loaded from central_log.csv.")
    print("You can talk to me about your fatigue, voice, eyes, risk, or feelings.")
    print("Type 'exit' or 'bye' to quit.\n")

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in ("exit", "bye", "quit"):
            print("Bot: Take care 💙 Remember to rest and follow your doctor's advice.")
            break

        reply = answer_question(user_msg, row)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
