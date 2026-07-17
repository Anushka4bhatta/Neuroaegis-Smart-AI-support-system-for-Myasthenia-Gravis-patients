import sounddevice as sd
import soundfile as sf
import os

# Path where voice recordings will be saved
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "voice_samples")
os.makedirs(SAVE_DIR, exist_ok=True)

def record_voice(filename="voice_sample.wav", duration=5, fs=44100):
    print(f"🎤 Recording for {duration} seconds... Speak normally.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    path = os.path.join(SAVE_DIR, filename)
    sf.write(path, audio, fs)
    print(f"✅ Saved recording at: {path}")

if __name__ == "__main__":
    record_voice()
