print("🔥 DEBUG: Script is running at least")

import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from log_health_data import log_voice_data


# --------------------------------------------------------
# RECORD VOICE (20 sec)
# --------------------------------------------------------
def record_voice(duration: int = 20, fs: int = 44100) -> str:
    print("🎤 Recording for 20 seconds...")
    print("   ➤ Speak continuously (count 1–50 OR say 'Aaaaaah')")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "voice_samples",
        "dysarthria_test.wav",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf.write(save_path, audio, fs)

    print(f"✅ Recording saved at: {save_path}")
    return save_path


# --------------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------------
def compute_jitter(pitch_values: np.ndarray) -> float:
    # Remove unvoiced / NaN
    pitch_values = pitch_values[np.isfinite(pitch_values)]
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) < 3:
        return 0.0

    diffs = np.abs(np.diff(pitch_values))
    return float(np.mean(diffs / pitch_values[:-1]))


def compute_shimmer(energy: np.ndarray) -> float:
    if len(energy) < 3:
        return 0.0
    diffs = np.abs(np.diff(energy))
    return float(np.mean(diffs / energy[:-1]))


# --------------------------------------------------------
# ANALYSIS FUNCTION
# --------------------------------------------------------
def analyze_dysarthria(file_path: str) -> str:
    print("\n🔥 Running dysarthria voice module")

    y, sr = librosa.load(file_path, sr=22050)

    duration = librosa.get_duration(y=y, sr=sr)
    print(f"\n⏱ Total duration recorded: {duration:.1f} seconds")

    # Split into early and late halves
    mid = len(y) // 2
    first = y[:mid]
    last = y[mid:]

    # Pitch estimation using librosa.yin with keyword args
    pitches_first = librosa.yin(first, fmin=50, fmax=500, sr=sr)
    pitches_last = librosa.yin(last,  fmin=50, fmax=500, sr=sr)

    # Energy (RMS)
    ener_first = librosa.feature.rms(y=first)[0]
    ener_last = librosa.feature.rms(y=last)[0]

    # Jitter & shimmer
    jitter_first = compute_jitter(pitches_first)
    jitter_last = compute_jitter(pitches_last)

    shimmer_first = compute_shimmer(ener_first)
    shimmer_last = compute_shimmer(ener_last)

    # Energy ratio: last vs first
    energy_ratio = (float(np.mean(ener_last)) + 1e-6) / (float(np.mean(ener_first)) + 1e-6)

    # PRINT FEATURE TABLE
    print("\n📊 Acoustic Feature Summary")
    print("────────────────────────────")
    print(f"Jitter (first):          {jitter_first:.4f}")
    print(f"Jitter (last):           {jitter_last:.4f}")
    print(f"Δ Jitter (last-first):   {(jitter_last - jitter_first):.4f}")
    print()
    print(f"Shimmer (first):         {shimmer_first:.4f}")
    print(f"Shimmer (last):          {shimmer_last:.4f}")
    print(f"Δ Shimmer (last-first):  {(shimmer_last - shimmer_first):.4f}")
    print()
    print(f"Energy ratio last/first: {energy_ratio:.3f}")

    # -----------------------------------------------------------------
    # DECISION SECTION
    # -----------------------------------------------------------------
    fatigued_flags = []

    # relative worsening
    if jitter_first > 0 and jitter_last > jitter_first * 1.25:
        fatigued_flags.append("increasing jitter (pitch instability)")
    if shimmer_first > 0 and shimmer_last > shimmer_first * 1.25:
        fatigued_flags.append("increasing shimmer (loudness instability)")
    # energy drop
    if energy_ratio < 0.70:
        fatigued_flags.append("drop in energy (breathiness / fatigue)")

    print("\n🧠 Interpretation")
    print("────────────────────────────")

    # FINAL STATUS
    if len(fatigued_flags) >= 2:
        status = "VOICE_FATIGUED"
        print("VOICE STATUS: Possible FATIGUE / DYSARTHRIA 😴")
        for f in fatigued_flags:
            print(f"• {f}")
    elif len(fatigued_flags) == 1:
        status = "VOICE_BORDERLINE"
        print("VOICE STATUS: BORDERLINE — Needs monitoring 🟠")
        print(f"• {fatigued_flags[0]}")
    else:
        status = "VOICE_STABLE"
        print("VOICE STATUS: Likely STABLE 😊")
        print("• No major instability detected.")

    # Log to patient log
    print("\n📝 Saving voice analysis to patient log…")
    log_voice_data(status,
    jitter_first,
    jitter_last,
    shimmer_first,
    shimmer_last,
    energy_ratio)
    print("✅ Logged voice status into voicelog.csv")

    return status


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    audio_path = record_voice()
    analyze_dysarthria(audio_path)
